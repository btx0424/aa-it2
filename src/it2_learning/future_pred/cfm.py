from __future__ import annotations

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch.nn.parallel import DistributedDataParallel as DDP


def maybe_unwrap(module: nn.Module):
    if isinstance(module, DDP):
        module = module.module
    return module


def _time_fourier_features(
    t: torch.Tensor, num_frequencies: int
) -> torch.Tensor:
    """
    Sinusoids of ``t * 2^i * π`` (``i = 0 … num_frequencies-1``) in sin/cos pairs.

    ``t`` is last-dim-1; output last dim is ``2 * num_frequencies`` (suitable for
    a learned map into ``time_dim``).
    """
    freqs = 2.0 ** torch.arange(
        num_frequencies, device=t.device, dtype=t.dtype
    ) * (math.pi)
    # t: (..., 1) * (num_freq,) -> (..., num_frequ)
    args = t * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class VelocityPredictionModel(nn.Module):
    """MLP vector field ``v_θ(xt, t; context)`` for conditional flow matching.

    Inputs are concatenated: ``[xt, context, ϕ(t)]`` where ``ϕ`` is a SiLU-activated
    linear map of multi-frequency sines and cosines of ``t`` (typical in diffusion-style
    time conditioning). ``t`` is processed as ``(N, 1)`` after a flat batch dimension ``N``.
    """

    def __init__(
        self,
        context_dim: int,
        pred_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_frequencies: int = 8,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.pred_dim = pred_dim
        self.time_dim = time_dim
        self.num_frequencies = num_frequencies

        fourier_dim = 2 * num_frequencies
        self.time_proj = nn.Linear(fourier_dim, time_dim)
        in_dim = pred_dim + context_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pred_dim),
        )

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        t2 = t.reshape(-1, 1)
        f = _time_fourier_features(t2, self.num_frequencies)
        return F.silu(self.time_proj(f))

    def forward(
        self,
        xt: Float[torch.Tensor, "N pred_dim"],
        context: Float[torch.Tensor, "N context_dim"],
        t: Float[torch.Tensor, "N 1"],
    ) -> Float[torch.Tensor, "N pred_dim"]:
        """
        ``(N, pred_dim)``, ``(N, context_dim)``, and ``(N, 1)``; one flow time in ``[0, 1]`` per row.
        """
        t_emb = self._embed_time(t)
        h = torch.cat([xt, context, t_emb], dim=-1)
        v = self.mlp(h)
        return v


class CFMFuturePredictor(nn.Module):
    """Flow matching with **FM time**: :math:`t=0` is noise, :math:`t=1` is data. The field
    ``v_θ`` matches the constant velocity :math:`x_1 - x_0` on the straight path in
    :meth:`compute_loss`. The wrapper flattens leading batch dims to one row per
    :math:`(x_t, t, \text{context})`.
    """
    def __init__(
        self,
        context_dim: int,
        pred_dim: int,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.pred_dim = pred_dim

        self.query_embedding = nn.Embedding(2, context_dim)
        self.query_embedding.weight._non_muon = True

        self.v = VelocityPredictionModel(context_dim, pred_dim)
    
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
        xt: Float[torch.Tensor, "... pred_dim"],
        context: Float[torch.Tensor, "... context_dim"],
        t: Float[torch.Tensor, "... 1"],
    ) -> Float[torch.Tensor, "... pred_dim"]:
        """
        Arbitrary leading batch shape, then ``pred_dim`` / ``context_dim``; ``t`` has the
        same lead shape and a final ``1`` (``… 1``). Internally flattens to apply ``.v``.
        """
        batch_shape = xt.shape[:-1]
        N = math.prod(batch_shape) if batch_shape else 1
        output = self.v(
            xt.reshape(N, self.pred_dim),
            context=context.reshape(N, self.context_dim),
            t=t.reshape(N, 1),
        )
        return output.reshape(*batch_shape, self.pred_dim)
    
    @torch.inference_mode()
    def sample_prior(
        self,
        context: Float[torch.Tensor, "N context_dim"],
        num_samples: int = 1,
        steps: int = 10,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Euler integration from :math:`t=0` (noise) to :math:`t=1` (data), same straight
        path and target field as :meth:`compute_loss`: :math:`x_t = (1 - t) x_0 + t x_1`,
        :math:`v \\approx x_1 - x_0`.

        Returns ``(pred, None, zero_like_entropy)`` for a third slot compatible with
        :class:`VAEFuturePredictor` call sites. ``pred`` is ``(N, pred_dim)`` if
        ``num_samples == 1`` else ``(N, num_samples, pred_dim)``.
        """
        device = context.device
        dtype = context.dtype if context.is_floating_point() else torch.float32
        N = context.shape[0]
        ctx2 = context[:, None, :].expand(-1, num_samples, -1)
        B = N * num_samples
        x = torch.randn(B, self.pred_dim, device=device, dtype=dtype)
        ctxf = ctx2.reshape(B, self.context_dim)
        dt = 1.0 / float(steps)
        for k in range(steps):
            t_mid = (k + 0.5) * dt
            tb = x.new_full((B, 1), t_mid)
            v = self.v(x, ctxf, tb)
            x = x + dt * v
        if num_samples == 1:
            out = x.view(N, self.pred_dim)
        else:
            out = x.view(N, num_samples, self.pred_dim)
        return out, None, out.new_zeros(N)

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
        lead = target.shape[:-1]
        x1 = target
        x0 = torch.randn_like(x1)
        t = torch.rand(*lead, 1, device=device, dtype=target.dtype)
        xt = (1.0 - t) * x0 + t * x1

        v_hat_proprio = self(xt, context=proprio_context, t=t)
        v_hat_extero = self(xt, context=extero_context, t=t)
        v_target = x1 - x0

        lik_p = (v_hat_proprio - v_target).square().mean(dim=-1)
        lik_e = (v_hat_extero - v_target).square().mean(dim=-1)
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

