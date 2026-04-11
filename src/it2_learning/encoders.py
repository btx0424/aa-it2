import torch
import torch.nn as nn
from active_adaptation.learning.modules import MLP
from typing import Type

class EncoderOne(nn.Module):
    def __init__(
        self,
        cmd_shape: torch.Size,
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

        self.cmd_mlp = MLP([cmd_shape[-1], 128, token_dim], activation=activation)
        self.proprio_mlp = MLP([proprio_shape[-1], 256, token_dim], activation=activation)
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
        cmd_inp: torch.Tensor,
        proprio_inp: torch.Tensor,
        extero_inp: torch.Tensor,
    ):
        batch_shape = proprio_inp.shape[:-1]

        cmd_flat = cmd_inp.reshape(-1, cmd_inp.shape[-1])
        proprio_flat = proprio_inp.reshape(-1, proprio_inp.shape[-1])
        extero_flat = extero_inp.reshape(-1, *extero_inp.shape[-3:])

        cmd_feature = self.cmd_mlp(cmd_flat)
        proprio_feature = self.proprio_mlp(proprio_flat)
        extero_feature = self.extero_mlp(extero_flat)

        tokens = torch.stack(
            [cmd_feature, proprio_feature, extero_feature], dim=1
        )
        tokens = tokens + self.modality_embedding.unsqueeze(0)

        attn_out, _ = self.fusion(tokens, tokens, tokens, need_weights=False)
        tokens = self.fusion_norm(tokens + attn_out)
        tokens = tokens + self.ffn(tokens)

        fused = self.out_proj(tokens.flatten(start_dim=1))
        return fused.reshape(*batch_shape, self.hidden_dim)