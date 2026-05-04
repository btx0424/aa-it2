import torch
import torch.nn as nn
from active_adaptation.learning.modules import MLP
from typing import Type, Literal
from jaxtyping import Float


from torch.nn.attention import SDPBackend, sdpa_kernel

ExteroEncoderKind = Literal["cnn", "defm_cnn"]
DefmCnnVariant = Literal["defm_resnet18", "defm_regnet_y_400mf"]


def build_policy_future_attn_masks(
    num_command_slots: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (self_attn, cross_attn) boolean masks for :class:`EncoderTwo` future training.

    Future prediction uses **exactly two** learned query embeddings (``fut_0``, ``fut_1``).
    Layout: ``[cmd_0, …, cmd_{K-1}, fut_0, fut_1, proprio, extero]`` with ``K = num_command_slots``.

    **Self-attention** (shape ``(K+4, K+4)``): query–query block blocks cross-talk between
    query slots; command slots reach proprio/extero by default; ``fut_0`` may attend only to
    proprio; ``fut_1`` to proprio and extero.

    **Cross-attention** (shape ``(K+2, 2)``): command slots attend to both modalities;
    ``fut_0`` only to proprio; ``fut_1`` to both.

    ``True`` means *masked* (cannot attend), per :class:`torch.nn.MultiheadAttention`.
    """
    F = 2
    if num_command_slots < 1:
        raise ValueError(f"num_command_slots must be >= 1, got {num_command_slots}")

    K = num_command_slots
    M = K + F
    L = M + 2
    prop_i, ext_i = M, M + 1

    mask_self = torch.zeros(L, L, dtype=torch.bool, device=device)
    mask_self[:M, :M] = ~torch.eye(M, dtype=torch.bool, device=device)

    for j in range(F):
        idx = K + j
        mask_self[idx, :] = True
        mask_self[idx, prop_i] = False
        if j >= 1:
            mask_self[idx, ext_i] = False

    mask_cross = torch.zeros(M, 2, dtype=torch.bool, device=device)
    for i in range(K):
        mask_cross[i, :] = False
    for j in range(F):
        idx = K + j
        mask_cross[idx, 0] = False
        mask_cross[idx, 1] = j == 0

    return mask_self, mask_cross


def build_policy_attn_masks(
    num_command_slots: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Self/cross masks for policy forward **without** future query slots.

    Layout: ``[cmd_0, …, cmd_{K-1}, proprio, extero]`` (length ``K + 2``).

    **Self-attention**: command tokens do not attend to *other* command tokens (same
    pattern as :func:`build_policy_future_attn_masks` for the command block); each
    command may still attend to itself, proprio, and extero. Proprio and extero
    attend without restriction.

    **Cross-attention** (shape ``(K, 2)``): every command attends to both modalities
    (no masking).

    ``True`` means *masked* (cannot attend), per :class:`torch.nn.MultiheadAttention`.
    """
    if num_command_slots < 1:
        raise ValueError(f"num_command_slots must be >= 1, got {num_command_slots}")

    K = num_command_slots
    L = K + 2

    mask_self = torch.zeros(L, L, dtype=torch.bool, device=device)
    mask_self[:K, :K] = ~torch.eye(K, dtype=torch.bool, device=device)

    mask_cross = torch.zeros(K, 2, dtype=torch.bool, device=device)
    return mask_self, mask_cross


def simple_extero_encoder(
    extero_channels: int,
    token_dim: int,
    activation: Type[nn.Module],
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(extero_channels, 32, kernel_size=3, stride=2, padding=1),
        activation(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        activation(),
        nn.Conv2d(64, token_dim, kernel_size=3, stride=2, padding=1),
        activation(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1),
    )


class ExteroDefmCnn(nn.Module):
    """DeFM convolutional backbone (ResNet/RegNet + BiFPN) → linear projection to ``token_dim``.

    Expects ``extero_inp`` shaped ``(..., C, H, W)`` with **channel 0** as a metric depth map
    (meters). Uses :func:`defm.utils.preprocess_depth_batch` (vectorized torch, same logic as
    ``preprocess_depth_image``). Extra channels are ignored (only the first depth channel is used).
    """

    VARIANTS: tuple[str, ...] = ("defm_resnet18", "defm_regnet_y_400mf")

    def __init__(
        self,
        variant: DefmCnnVariant,
        token_dim: int,
        pretrained: bool = True,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(
                f"defm_cnn variant {variant!r} must be one of {self.VARIANTS}"
            )
        
        from defm.model_factory import create_defm_model
        self.variant = variant
        self.backbone = create_defm_model(variant, pretrained=pretrained)
        self.backbone.requires_grad_(False)

        for m in self.backbone.modules():
            m._defm_no_reinit = True  # skip PPOPolicy.init_ orthogonal reset on pretrained backbone
        
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            out = self.backbone(torch.zeros(1, 3, 224, 224, device=device))
        in_dim = out["global_backbone"].shape[-1]
        self.proj = nn.Linear(in_dim, token_dim)
        self.proj.weight._non_muon = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from defm.utils import preprocess_depth_batch

        if x.ndim != 4:
            raise ValueError(f"ExteroDefmCnn expects (N, C, H, W), got {tuple(x.shape)}")
        _, _, h, w = x.shape
        # Batched DeFM depth → (N, 3, H', W') on `x.device` (no per-sample numpy loop).
        x3 = preprocess_depth_batch(
            x,
            target_size=(h, w),
            patch_size=32,
            device=x.device,
        )
        feat = self.backbone(x3)["global_backbone"]
        return self.proj(feat)


def build_extero_encoder(
    *,
    extero_channels: int,
    extero_encoder: ExteroEncoderKind,
    defm_variant: DefmCnnVariant,
    defm_pretrained: bool,
    token_dim: int,
    activation: Type[nn.Module],
) -> nn.Module:
    if extero_encoder == "cnn":
        return simple_extero_encoder(
            extero_channels,
            token_dim,
            activation,
        )
    if extero_encoder == "defm_cnn":
        return ExteroDefmCnn(
            variant=defm_variant,
            token_dim=token_dim,
            pretrained=defm_pretrained,
        )
    raise ValueError(
        f"extero_encoder must be 'cnn' or 'defm_cnn', got {extero_encoder!r}"
    )


class EncoderTwo(nn.Module):
    """Fuse multiple command query tokens with proprio and extero.

    Pipeline:

    1. **Embed** proprio and extero into tokens; **LN** precomputed query tokens
       (order: queries → proprio → extero).
    2. **Self-attention** over all ``M + 2`` tokens (residual + layer norm + FFN).
    3. **Cross-attention**: query rows attend to proprio and extero tokens (optional
       per-slot mask).
    4. **Return** refined query tokens ``(..., M, token_dim)``.
    """

    def __init__(
        self,
        proprio_shape: torch.Size,
        extero_shape: torch.Size,
        token_dim: int = 256,
        num_heads: int = 4,
        activation: Type[nn.Module] = nn.SiLU,
        extero_encoder: ExteroEncoderKind = "cnn",
        defm_variant: DefmCnnVariant = "defm_resnet18",
        defm_pretrained: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        extero_channels = extero_shape[0] if len(extero_shape) == 3 else 1

        self.proprio_mlp = MLP([proprio_shape[-1], 256, token_dim], activation=activation, first_non_muon=True)
        self.extero_cnn = build_extero_encoder(
            extero_channels=extero_channels,
            extero_encoder=extero_encoder,
            defm_variant=defm_variant,
            defm_pretrained=defm_pretrained,
            token_dim=token_dim,
            activation=activation,
        )
        self.query_ln = nn.LayerNorm(token_dim)
        self.proprio_ln = nn.LayerNorm(token_dim)
        self.extero_ln = nn.LayerNorm(token_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            activation(),
            nn.Linear(token_dim, token_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(token_dim)

        self.output_dim = token_dim

        def init_(module: nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0.)
            nn.init.orthogonal_(module.out_proj.weight, 0.02)
            if module.out_proj.bias is not None:
                nn.init.constant_(module.out_proj.bias, 0.)

        init_(self.self_attn)
        init_(self.cross_attn)

    def forward(
        self,
        queries_inp: Float[torch.Tensor, "... M token_dim"],
        proprio_inp: Float[torch.Tensor, "... D"],
        extero_inp: Float[torch.Tensor, "... C H W"],
        attn_mask_self: torch.Tensor | None = None,
        attn_mask_cross: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "... M token_dim"]:
        batch_shape = queries_inp.shape[:-2]
        N = batch_shape.numel()
        M = queries_inp.shape[-2]

        queries = queries_inp.reshape(N, M, self.token_dim)
        proprio_flat = proprio_inp.reshape(N, proprio_inp.shape[-1])
        extero_flat = extero_inp.reshape(N, *extero_inp.shape[-3:])

        queries = self.query_ln(queries)
        proprio_feature = self.proprio_ln(self.proprio_mlp(proprio_flat)).reshape(N, 1, self.token_dim)
        extero_feature = self.extero_ln(self.extero_cnn(extero_flat)).reshape(N, 1, self.token_dim)

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
        return query_refined.reshape(*batch_shape, M, self.token_dim)
