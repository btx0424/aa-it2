import torch
import torch.nn as nn
from active_adaptation.learning.modules import MLP
from typing import Type, Literal
from jaxtyping import Float


from torch.nn.attention import SDPBackend, sdpa_kernel


def _make_cnn_norm(
    channels: int,
    norm: Literal["none", "group"],
    groups: int,
) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm != "group":
        raise ValueError(f"Unsupported cnn_norm={norm!r}, expected 'none' or 'group'.")

    group_count = min(groups, channels)
    while channels % group_count != 0 and group_count > 1:
        group_count -= 1
    return nn.GroupNorm(group_count, channels)


class EncoderOne(nn.Module):
    """Fuse query-feature, proprio, and extero into a single feature vector.

    Each modality is embedded to a shared ``token_dim``. Three tokens attend to each
    other with multi-head self-attention, then a per-token FFN. The flattened tokens
    are projected to ``hidden_dim`` for actor/critic heads.
    """

    def __init__(
        self,
        proprio_shape: torch.Size,
        extero_shape: torch.Size, # usually a height map of shape [1, H, W]
        token_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
        cnn_norm: Literal["none", "group"] = "none",
        cnn_norm_groups: int = 8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        extero_channels = extero_shape[0] if len(extero_shape) == 3 else 1

        self.proprio_mlp = MLP([proprio_shape[-1], 256, token_dim], activation=activation, first_non_muon=True)
        self.extero_mlp = nn.Sequential(
            nn.Conv2d(extero_channels, 32, kernel_size=3, stride=2, padding=1),
            _make_cnn_norm(32, cnn_norm, cnn_norm_groups),
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _make_cnn_norm(64, cnn_norm, cnn_norm_groups),
            activation(),
            nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
            _make_cnn_norm(token_dim, cnn_norm, cnn_norm_groups),
            activation(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.modality_embedding = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        self.modality_embedding._non_muon = True
        
        self.fusion = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(3 * token_dim),
            nn.Linear(3 * token_dim, hidden_dim),
            activation(),
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        queries_inp: Float[torch.Tensor, "... D"],
        proprio_inp: Float[torch.Tensor, ".. D"],
        extero_inp: Float[torch.Tensor, ".. C H W"],
    ):
        batch_shape = proprio_inp.shape[:-1]

        query_flat = queries_inp.reshape(-1, queries_inp.shape[-1])
        proprio_flat = proprio_inp.reshape(-1, proprio_inp.shape[-1])
        extero_flat = extero_inp.reshape(-1, *extero_inp.shape[-3:])

        query_feature = query_flat
        proprio_feature = self.proprio_mlp(proprio_flat)
        extero_feature = self.extero_mlp(extero_flat)

        tokens = torch.stack(
            [query_feature, proprio_feature, extero_feature], dim=1
        )
        tokens = tokens + self.modality_embedding.unsqueeze(0)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            attn_out, _ = self.fusion(tokens, tokens, tokens, need_weights=False)
        tokens = self.fusion_norm(tokens + attn_out)
        tokens = tokens + self.ffn(tokens)

        fused = self.out_proj(tokens.flatten(start_dim=1))
        return fused.reshape(*batch_shape, self.hidden_dim)


class EncoderTwo(nn.Module):
    """Multi-modal encoder like :class:`EncoderOne`, with an extra query-conditioned step.

    Pipeline:

    1. **Embed** proprio and extero into tokens and consume a precomputed query
       feature token (order: query → proprio → extero) plus learned modality embeddings.
    2. **Self-attention** over all three tokens (mixing modalities), residual +
       layer norm, then the same per-token FFN + residual as ``EncoderOne``.
    3. **Cross-attention**: the query token is the **query**; proprio and extero
       tokens are **keys/values**. The query representation is updated with a
       residual and layer norm (``query_refined``).
    4. **Aggregate**: concatenate ``query_refined`` with the (post-FFN) proprio and
       extero tokens and linearly project to ``hidden_dim``.

    The cross-attention step lets the query slot explicitly pull task-relevant
    structure from proprioception and exteroception after they have already mixed
    via self-attention.
    """

    def __init__(
        self,
        proprio_shape: torch.Size,
        extero_shape: torch.Size,
        token_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
        cnn_norm: Literal["none", "group"] = "none",
        cnn_norm_groups: int = 8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        extero_channels = extero_shape[0] if len(extero_shape) == 3 else 1

        self.proprio_mlp = MLP([proprio_shape[-1], 256, token_dim], activation=activation, first_non_muon=True)
        self.extero_mlp = nn.Sequential(
            nn.Conv2d(extero_channels, 32, kernel_size=3, stride=2, padding=1),
            _make_cnn_norm(32, cnn_norm, cnn_norm_groups),
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            _make_cnn_norm(64, cnn_norm, cnn_norm_groups),
            activation(),
            nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
            _make_cnn_norm(token_dim, cnn_norm, cnn_norm_groups),
            activation(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, token_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(token_dim)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            activation(),
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        queries_inp: Float[torch.Tensor, "... M token_dim"],
        proprio_inp: Float[torch.Tensor, "... D"],
        extero_inp: Float[torch.Tensor, "... C H W"],
        attn_mask_self: torch.Tensor | None = None,
        attn_mask_cross: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "... M hidden_dim"]:
        batch_shape = queries_inp.shape[:-2]
        N = batch_shape.numel()
        M = queries_inp.shape[-2]

        queries = queries_inp.reshape(N, M, self.token_dim)
        proprio_flat = proprio_inp.reshape(N, proprio_inp.shape[-1])
        extero_flat = extero_inp.reshape(N, *extero_inp.shape[-3:])

        proprio_feature = self.proprio_mlp(proprio_flat).reshape(N, 1, self.token_dim)
        extero_feature = self.extero_mlp(extero_flat).reshape(N, 1, self.token_dim)

        tokens = torch.cat(
            [queries, proprio_feature, extero_feature], dim=1
        ) # [N, M + 2, self.token_dim]

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            sa_out, _ = self.self_attn(
                tokens,
                tokens,
                tokens,
                attn_mask=attn_mask_self,
                need_weights=False,
            )
        tokens = self.self_attn_norm(tokens + sa_out)
        tokens = tokens + self.ffn(tokens)

        query_token = tokens[:, :-2, :] # [N, M, self.token_dim]
        proprio_extero = tokens[:, -2:, :] # [N, 2, self.token_dim]
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            cross_out, _ = self.cross_attn(
                query_token,
                proprio_extero,
                proprio_extero,
                attn_mask=attn_mask_cross,
                need_weights=False,
            )
        query_refined = self.cross_attn_norm(query_token + cross_out) # [N, M, self.token_dim]

        fused = self.out_proj(query_refined) # [N, M, self.hidden_dim]
        return fused.reshape(*batch_shape, M, self.hidden_dim)

    def forward_policy(
        self,
        policy_query: Float[torch.Tensor, "... 1 token_dim"],
        proprio_inp: Float[torch.Tensor, "... D"],
        extero_inp: Float[torch.Tensor, "... C H W"],
    ) -> Float[torch.Tensor, "... hidden_dim"]:
        return self(policy_query, proprio_inp, extero_inp)
    
    def forward_policy_future(
        self,
        queries: Float[torch.Tensor, "... 3 token_dim"],
        proprio_inp: Float[torch.Tensor, "... D"],
        extero_inp: Float[torch.Tensor, "... C H W"],
    ) -> Float[torch.Tensor, "... 3 hidden_dim"]:
        M = queries.shape[-2]
        assert M == 3, f"EncoderTwo.compute_policy_future_feature expects M==3, got M={M}."

        # Token layout after concat: [q0, q1, q2, proprio, extero] -> indices [0..4]
        # KEEP THIS COMMENT: in this case, we construct the attn_mask so that:
        # 1. each query does not attend to other queries
        # 2. the prior query[..., 1, :] attends to only the proprio token (index 3)
        # 3. the posterior query[..., 2, :] attends to both proprio and extero tokens (indices 3 and 4)
        attn_mask_self = torch.zeros(5, 5, dtype=torch.bool, device=queries.device)
        # block query-to-query off-diagonal
        attn_mask_self[:3, :3] = ~torch.eye(3, dtype=torch.bool, device=queries.device)

        # prior query row (index 1): only proprio (col 3) allowed
        attn_mask_self[1, :] = True
        attn_mask_self[1, 3] = False

        # posterior query row (index 2): proprio (col 3) and extero (col 4) allowed
        attn_mask_self[2, :] = True
        attn_mask_self[2, 3] = False
        attn_mask_self[2, 4] = False

        attn_mask_cross = torch.tensor([
            [False, False], # policy query sees both proprio and extero
            [False, True], # prior query sees only proprio
            [False, False], # posterior query sees both proprio and extero
        ], dtype=torch.bool, device=queries.device)

        return self(
            queries,
            proprio_inp,
            extero_inp,
            attn_mask_self=attn_mask_self,
            attn_mask_cross=attn_mask_cross,
        )
