from __future__ import annotations

import math
from collections import OrderedDict
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.nn.parallel import DistributedDataParallel as DDP

from it2_learning.networks import ConditionalUNet1D


def maybe_unwrap(module: nn.Module):
    if isinstance(module, DDP):
        module = module.module
    return module


class CFMVectorField(nn.Module):
    """``v_θ(x_t, t; c)`` with ``x_t`` of shape ``(N, T, D)`` (batch of trajectories).

    Forwards to :class:`ConditionalUNet1D` as ``(N, T, D)``. Requires ``T > 4`` so the UNet
    downsampling path has valid length (see :meth:`CFMFuturePredictor.__init__`).
    """

    def __init__(
        self,
        context_dim: int,
        *,
        seq_len: int,
        channel_dim: int,
        unet_base_channels: int = 64,
        unet_channel_mults: tuple[int, ...] = (1, 2, 4),
        unet_dropout: float = 0.0,
        trajectory_time_embedding: int = 0,
    ) -> None:
        super().__init__()
        if seq_len <= 4:
            raise ValueError(
                f"CFM UNet requires trajectory seq_len > 4, got seq_len={seq_len}. "
                "Use a relabeler with a longer horizon (e.g. FutureTrajectory) or a different "
                "future_predictor (e.g. VAE) for short horizons."
            )
        self.context_dim = context_dim
        self.seq_len = seq_len
        self.channel_dim = channel_dim

        self.backbone = ConditionalUNet1D(
            input_dim=channel_dim,
            output_dim=channel_dim,
            base_channels=unet_base_channels,
            channel_mults=unet_channel_mults,
            cond_dim=context_dim,
            dropout=unet_dropout,
            trajectory_time_embedding=trajectory_time_embedding,
            trajectory_length=seq_len,
        )

    def forward(
        self,
        xt: Float[torch.Tensor, "N T D"],
        context: Float[torch.Tensor, "N context_dim"],
        t: Float[torch.Tensor, "N 1 1"],
    ) -> Float[torch.Tensor, "N T D"]:
        n = xt.shape[0]
        t_b = t.reshape(n)
        return self.backbone(xt, cond=context, t=t_b)


class CFMFuturePredictor(nn.Module):
    """Flow matching with **FM time**: :math:`t=0` is noise, :math:`t=1` is data. The field
    ``v_θ`` matches the constant velocity :math:`x_1 - x_0` on the straight path in
    :meth:`compute_loss`. Inputs ``x_t`` use shape ``[..., T, D]`` (trajectory length ``T``,
    per-step channel dim ``D``); leading dimensions are flattened for the vector field.
    """

    def __init__(
        self,
        context_dim: int,
        *,
        trajectory_shape: torch.Size | tuple[int, ...],
        unet_base_channels: int = 128,
        unet_channel_mults: tuple[int, ...] = (1, 2, 4),
        unet_dropout: float = 0.0,
        trajectory_time_embedding: int = 0,
    ):
        super().__init__()
        if len(trajectory_shape) != 2:
            raise ValueError(
                f"trajectory_shape must be (seq_len, channel_dim), got {trajectory_shape!r}"
            )
        self.context_dim = context_dim
        self.trajectory_shape = torch.Size(trajectory_shape)
        self.seq_len = int(self.trajectory_shape[0])
        self.channel_dim = int(self.trajectory_shape[1])

        self.query_embedding = nn.Embedding(2, context_dim)
        self.query_embedding.weight._non_muon = True

        self.v = CFMVectorField(
            context_dim,
            seq_len=self.seq_len,
            channel_dim=self.channel_dim,
            unet_base_channels=unet_base_channels,
            unet_channel_mults=unet_channel_mults,
            unet_dropout=unet_dropout,
            trajectory_time_embedding=trajectory_time_embedding,
        )

    def wrap_DDP(self, device_ids: list[int]):
        self.query_embedding = DDP(self.query_embedding, device_ids=device_ids)
        self.v = DDP(self.v, device_ids=device_ids)

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["query_embedding"] = maybe_unwrap(self.query_embedding).state_dict()
        state_dict["v"] = maybe_unwrap(self.v).state_dict()
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True):
        maybe_unwrap(self.query_embedding).load_state_dict(
            state_dict["query_embedding"], strict=strict
        )
        maybe_unwrap(self.v).load_state_dict(state_dict["v"], strict=strict)

    def forward(
        self,
        xt: Float[torch.Tensor, "... T D"],
        context: Float[torch.Tensor, "... Dc"],
        t: Float[torch.Tensor, "... 1 1"],
    ) -> Float[torch.Tensor, "... T D"]:
        """
        ``x_t`` has shape ``[..., T, D]``; ``context`` and ``t`` broadcast over the same
        leading dimensions (``t`` ends with a singleton time-in-(0,1) axis).
        """
        if xt.shape[-2:] != (self.seq_len, self.channel_dim):
            raise ValueError(
                f"xt must end with (T, D)=({self.seq_len}, {self.channel_dim}), got {tuple(xt.shape)}"
            )
        batch_shape = xt.shape[:-2]
        n = math.prod(batch_shape) if batch_shape else 1
        t_flat = t.reshape(n, 1, 1)
        if context.shape[:-1] != xt.shape[:-2]:
            raise ValueError(
                f"context leading shape {tuple(context.shape[:-1])} must match xt[:-2] {tuple(xt.shape[:-2])}"
            )
        output = self.v(
            xt.reshape(n, self.seq_len, self.channel_dim),
            context=context.reshape(n, self.context_dim),
            t=t_flat,
        )
        return output.reshape(*batch_shape, self.seq_len, self.channel_dim)

    @torch.inference_mode()
    def sample_prior(
        self,
        context: Float[torch.Tensor, "N context_dim"],
        num_samples: int = 1,
        steps: int = 40,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Euler integration from :math:`t=0` (noise) to :math:`t=1` (data), same straight
        path and target field as :meth:`compute_loss`: :math:`x_t = (1 - t) x_0 + t x_1`,
        :math:`v \\approx x_1 - x_0`.

        Returns ``(pred, None, zero_like_entropy)`` for a third slot compatible with
        :class:`VAEFuturePredictor` call sites. ``pred`` is ``(N, T, D)`` if
        ``num_samples == 1`` else ``(N, num_samples, T, D)`` (suitable for :meth:`FutureRelabel.process_pred`).
        """
        device = context.device
        dtype = context.dtype if context.is_floating_point() else torch.float32
        n = context.shape[0]
        t_sz, d_ch = self.seq_len, self.channel_dim
        ctx2 = context[:, None, :].expand(-1, num_samples, -1)
        b = n * num_samples
        x = torch.randn(b, t_sz, d_ch, device=device, dtype=dtype)
        ctxf = ctx2.reshape(b, self.context_dim)
        dt = 1.0 / float(steps)
        for k in range(steps):
            t_mid = (k + 0.5) * dt
            tb = x.new_full((b, 1, 1), t_mid)
            v = self.v(x, ctxf, tb)
            x = x + dt * v
        if num_samples == 1:
            out = x.view(n, t_sz, d_ch)
        else:
            out = x.view(n, num_samples, t_sz, d_ch)
        return out, None, out.new_zeros(n)

    def compute_loss(
        self,
        proprio_context: torch.Tensor,
        extero_context: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        One straight line per row, FM time: :math:`t=0` at noise, :math:`t=1` at data,
        :math:`x_t = (1 - t) x_0 + t x_1` with :math:`x_0 \sim \mathcal N(0, I)`,
        :math:`t \sim U(0,1)`, and target :math:`v = x_1 - x_0` for each context head
        (sum of per-dimension squared errors; yaw channel scaled by ``1/\\pi`` like
        :class:`VAEFuturePredictor`).

        If ``valid_mask`` is set, the loss and logged MSEs are **masked means** (sum
        over valid / count of valid), same layout as :meth:`VAEFuturePredictor.compute_loss`.
        """
        device = target.device
        x1 = target
        x0 = torch.randn_like(x1)
        t = torch.rand(*x1.shape[:-2], 1, 1, device=device, dtype=target.dtype)
        xt = (1.0 - t) * x0 + t * x1

        v_hat_proprio = self(xt, context=proprio_context, t=t)
        v_hat_extero = self(xt, context=extero_context, t=t)
        v_target = x1 - x0

        lik_p = (v_hat_proprio - v_target).square().mean(dim=(-1, -2))
        lik_e = (v_hat_extero - v_target).square().mean(dim=(-1, -2))
        per = lik_p + lik_e

        if valid_mask is not None:
            maskf = valid_mask.to(dtype=per.dtype)
            if maskf.shape[-1] == 1 and maskf.dim() == per.dim() + 1:
                maskf = maskf.squeeze(-1)
            maskf = maskf.reshape_as(per)
            denom = maskf.sum().clamp_min(1.0)
            loss = (per * maskf).sum() / denom
            loss_proprio_m = (lik_p * maskf).sum() / denom
            loss_extero_m = (lik_e * maskf).sum() / denom
        else:
            loss = per.mean()
            loss_proprio_m = lik_p.mean()
            loss_extero_m = lik_e.mean()

        return loss, {
            "future/loss_proprio": loss_proprio_m,
            "future/loss_extero": loss_extero_m,
        }
