import torch
import torch.nn as nn
from active_adaptation.learning.modules import MLP
from typing import Type
from jaxtyping import Float


from torch.nn.attention import SDPBackend, sdpa_kernel


class EncoderOne(nn.Module):
    """Fuse command-feature, proprio, and extero into a single feature vector.

    Each modality is embedded to a shared ``token_dim``. Three tokens attend to each
    other with multi-head self-attention, then a per-token FFN. The flattened tokens
    are projected to ``hidden_dim`` for actor/critic heads.
    """

    def __init__(
        self,
        proprio_shape: torch.Size,
        extero_shape: torch.Size, # usually a height map of shape [1, H, W]
        token_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        extero_channels = extero_shape[0] if len(extero_shape) == 3 else 1

        self.proprio_mlp = MLP([proprio_shape[-1], 256, token_dim], activation=activation, first_non_muon=True)
        self.extero_mlp = nn.Sequential(
            nn.Conv2d(extero_channels, 32, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
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

    def forward(
        self,
        cmd_feature_inp: Float[torch.Tensor, "... D"],
        proprio_inp: Float[torch.Tensor, ".. D"],
        extero_inp: Float[torch.Tensor, ".. C H W"],
    ):
        batch_shape = proprio_inp.shape[:-1]

        cmd_flat = cmd_feature_inp.reshape(-1, cmd_feature_inp.shape[-1])
        proprio_flat = proprio_inp.reshape(-1, proprio_inp.shape[-1])
        extero_flat = extero_inp.reshape(-1, *extero_inp.shape[-3:])

        cmd_feature = cmd_flat
        proprio_feature = self.proprio_mlp(proprio_flat)
        extero_feature = self.extero_mlp(extero_flat)

        tokens = torch.stack(
            [cmd_feature, proprio_feature, extero_feature], dim=1
        )
        tokens = tokens + self.modality_embedding.unsqueeze(0)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            attn_out, _ = self.fusion(tokens, tokens, tokens, need_weights=False)
        tokens = self.fusion_norm(tokens + attn_out)
        tokens = tokens + self.ffn(tokens)

        fused = self.out_proj(tokens.flatten(start_dim=1))
        return fused.reshape(*batch_shape, self.hidden_dim)


class EncoderTwo(nn.Module):
    """Multi-modal encoder like :class:`EncoderOne`, with an extra command-conditioned step.

    Pipeline:

    1. **Embed** proprio and extero into tokens and consume a precomputed command
       feature token (order: cmd → proprio → extero) plus learned modality embeddings.
    2. **Self-attention** over all three tokens (mixing modalities), residual +
       layer norm, then the same per-token FFN + residual as ``EncoderOne``.
    3. **Cross-attention**: the command token is the **query**; proprio and extero
       tokens are **keys/values**. The command representation is updated with a
       residual and layer norm (``cmd_refined``).
    4. **Aggregate**: concatenate ``cmd_refined`` with the (post-FFN) proprio and
       extero tokens and linearly project to ``hidden_dim``.

    The cross-attention step lets the command slot explicitly pull task-relevant
    structure from proprioception and exteroception after they have already mixed
    via self-attention.
    """

    def __init__(
        self,
        proprio_shape: torch.Size,
        extero_shape: torch.Size,
        token_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        extero_channels = extero_shape[0] if len(extero_shape) == 3 else 1

        self.proprio_mlp = MLP([proprio_shape[-1], 256, token_dim], activation=activation, first_non_muon=True)
        self.extero_mlp = nn.Sequential(
            nn.Conv2d(extero_channels, 32, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.modality_embedding = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        self.modality_embedding._non_muon = True

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
            nn.LayerNorm(3 * token_dim),
            nn.Linear(3 * token_dim, hidden_dim),
            activation(),
        )

    def forward(
        self,
        cmd_feature_inp: Float[torch.Tensor, "... D"],
        proprio_inp: Float[torch.Tensor, "... D"],
        extero_inp: Float[torch.Tensor, "... C H W"],
    ):
        batch_shape = proprio_inp.shape[:-1]

        cmd_flat = cmd_feature_inp.reshape(-1, cmd_feature_inp.shape[-1])
        proprio_flat = proprio_inp.reshape(-1, proprio_inp.shape[-1])
        extero_flat = extero_inp.reshape(-1, *extero_inp.shape[-3:])

        cmd_feature = cmd_flat
        proprio_feature = self.proprio_mlp(proprio_flat)
        extero_feature = self.extero_mlp(extero_flat)

        tokens = torch.stack(
            [cmd_feature, proprio_feature, extero_feature], dim=1
        )
        tokens = tokens + self.modality_embedding.unsqueeze(0)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            sa_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.self_attn_norm(tokens + sa_out)
        tokens = tokens + self.ffn(tokens)

        cmd_token = tokens[:, 0:1, :]
        proprio_extero = tokens[:, 1:, :]
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            cross_out, _ = self.cross_attn(
                cmd_token, proprio_extero, proprio_extero, need_weights=False
            )
        cmd_refined = self.cross_attn_norm(cmd_token + cross_out).squeeze(1)

        fused = self.out_proj(
            torch.cat([cmd_refined, tokens[:, 1], tokens[:, 2]], dim=-1)
        )
        return fused.reshape(*batch_shape, self.hidden_dim)
