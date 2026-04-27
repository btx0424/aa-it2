from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from it2_learning.networks import Downsample1d, Upsample1d


# ``F.softplus(0) = log(2)``; we normalize so that ``ρ = 0`` gives ``var = 1`` (``std = 1``) per dim.
_LOG2 = math.log(2.0)


def gaussian_moments_from_head(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, rho = out.chunk(2, dim=-1)
    var = F.softplus(rho, beta=_LOG2)
    logvar = var.clamp_min(1e-8).log()
    return mu, logvar


def kl_gaussian(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence between two Gaussian distributions.

    Sum over the last (event) dimension.
    """
    return 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1
    ).sum(dim=-1)


def entropy_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Differential entropy of a diagonal Gaussian, in nats, **summed** over the last axis.

    For ``d`` independent dimensions, ``H = 0.5 * sum_i (1 + log(2π) + log σ²_i)``. With
    small per-dim variances (large negative ``logvar``), the **total** is often a large
    **negative** number—this is normal, not a sign bug. For interpretability, divide the
    logged training metrics by ``latent_dim`` to get a per-dimension scale.
    """
    # H = 0.5 * sum_i (1 + log(2*pi) + logvar_i) for diagonal N(mu, diag(exp(logvar))).
    two_pi = mu.new_tensor(2.0 * torch.pi)
    return 0.5 * (1.0 + logvar + two_pi.log()).sum(dim=-1)


def maybe_unwrap(module: nn.Module):
    if isinstance(module, DDP):
        module = module.module
    return module


class ConvTargetEncoder(nn.Module):
    """Encode ``(B, T, C)`` targets into a fixed feature vector."""

    def __init__(
        self,
        channel_dim: int,
        *,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2),
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        if not channel_mults:
            raise ValueError("channel_mults must be non-empty")
        layers: list[nn.Module] = []
        in_channels = channel_dim
        widths = [base_channels * mult for mult in channel_mults]
        for idx, width in enumerate(widths):
            layers.extend(
                [
                    nn.Conv1d(in_channels, width, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.GroupNorm(min(8, width), width),
                    nn.Conv1d(width, width, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.GroupNorm(min(8, width), width),
                ]
            )
            if idx < len(widths) - 1:
                layers.append(Downsample1d(width))
            in_channels = width
        layers.extend(
            [
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(widths[-1], output_dim),
                nn.SiLU(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return self.net(target.transpose(1, 2))


class ConvTargetDecoder(nn.Module):
    """Decode a context/latent feature vector into ``(B, T, C)`` trajectories."""

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        channel_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.low_res_len = max(4, math.ceil(seq_len / 4))
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * self.low_res_len),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8, hidden_dim),
            Upsample1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8, hidden_dim),
            Upsample1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv1d(hidden_dim, channel_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).reshape(x.shape[0], self.low_res_len, self.hidden_dim)
        y = self.conv(h.transpose(1, 2))
        if y.shape[-1] != self.seq_len:
            y = F.interpolate(y, size=self.seq_len, mode="nearest")
        y = y.transpose(1, 2)
        return y


class VAEFuturePredictor(nn.Module):
    """Conditional VAE-style predictor for multi-modal future targets."""

    def __init__(
        self,
        context_dim: int,
        target_shape: torch.Size,
        latent_dim: int,
        kl_coef: float,
        encoder_base_channels: int = 32,
        encoder_channel_mults: tuple[int, ...] = (1, 2),
        encoder_output_dim: int = 256,
    ):
        super().__init__()
        self.context_dim = context_dim
        if len(target_shape) != 2:
            raise ValueError(f"target_shape must be (seq_len, channel_dim), got {target_shape!r}")
        self.target_shape = torch.Size(target_shape)
        self.seq_len = int(self.target_shape[0])
        self.channel_dim = int(self.target_shape[1])
        if self.seq_len != 1 and self.seq_len <= 4:
            raise NotImplementedError(
                "VAEFuturePredictor supports target_shape[0] == 1 or trajectories with seq_len > 4"
            )
        self.use_cnn = self.seq_len > 1
        self.encoder_output_dim = encoder_output_dim
        self.latent_dim = latent_dim
        self.kl_coef = kl_coef

        # self.query_embedding = nn.Parameter(torch.randn(2, context_dim) * 0.02)
        # self.query_embedding._non_muon = True
        self.query_embedding = nn.Embedding(2, context_dim)
        self.query_embedding.weight._non_muon = True

        self.prior = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )
        if self.use_cnn:
            self.target_encoder = ConvTargetEncoder(
                self.channel_dim,
                base_channels=encoder_base_channels,
                channel_mults=encoder_channel_mults,
                output_dim=encoder_output_dim,
            )
            posterior_in_dim = context_dim + encoder_output_dim
            decoder_out = ConvTargetDecoder(
                context_dim + latent_dim,
                self.seq_len,
                self.channel_dim,
            )
        else:
            self.target_encoder = nn.Flatten(start_dim=1)
            posterior_in_dim = context_dim + self.channel_dim
            decoder_out = nn.Sequential(
                nn.Linear(context_dim + latent_dim, 256),
                nn.SiLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, self.channel_dim),
            )
        self.posterior = nn.Sequential(
            nn.Linear(posterior_in_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )
        self.decoder = decoder_out
        self.reset_parameters()

    def reset_parameters(self):
        """PPO-style orthogonal inits. ``rho`` (second head half) stays zero-biased so initial per-dim ``std ≈ 1`` (see :func:`gaussian_moments_from_head`)."""

        def orth_linear(m: nn.Linear, gain: float) -> None:
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def reset_layernorms(seq: nn.Sequential) -> None:
            for m in seq.modules():
                if isinstance(m, nn.LayerNorm):
                    m.reset_parameters()
                elif isinstance(m, nn.GroupNorm):
                    m.reset_parameters()

        def init_gaussian_mlp(seq: nn.Sequential) -> None:
            linears = [m for m in seq if isinstance(m, nn.Linear)]
            for lin in linears[:-1]:
                orth_linear(lin, 0.1)
            last = linears[-1]
            orth_linear(last, 0.02)

        with torch.no_grad():
            self.query_embedding.weight.normal_(0.0, 0.02)
        init_gaussian_mlp(self.prior)
        init_gaussian_mlp(self.posterior)
        reset_layernorms(self.prior)
        reset_layernorms(self.posterior)

        for m in self.target_encoder.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.orthogonal_(m.weight, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                orth_linear(m, 0.02)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.orthogonal_(m.weight, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        reset_layernorms(self.target_encoder)
        reset_layernorms(self.decoder)

    def wrap_DDP(self, device_ids: list[int]):
        self.query_embedding = DDP(self.query_embedding, device_ids=device_ids)
        self.prior = DDP(self.prior, device_ids=device_ids)
        if any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder = DDP(self.target_encoder, device_ids=device_ids)
        self.posterior = DDP(self.posterior, device_ids=device_ids)
        self.decoder = DDP(self.decoder, device_ids=device_ids)

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["query_embedding"] = maybe_unwrap(self.query_embedding).state_dict()
        state_dict["prior"] = maybe_unwrap(self.prior).state_dict()
        state_dict["target_encoder"] = maybe_unwrap(self.target_encoder).state_dict()
        state_dict["posterior"] = maybe_unwrap(self.posterior).state_dict()
        state_dict["decoder"] = maybe_unwrap(self.decoder).state_dict()
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True):
        maybe_unwrap(self.query_embedding).load_state_dict(
            state_dict["query_embedding"], strict=strict
        )
        maybe_unwrap(self.prior).load_state_dict(state_dict["prior"], strict=strict)
        maybe_unwrap(self.target_encoder).load_state_dict(
            state_dict.get("target_encoder", {}), strict=strict
        )
        maybe_unwrap(self.posterior).load_state_dict(
            state_dict["posterior"], strict=strict
        )
        maybe_unwrap(self.decoder).load_state_dict(state_dict["decoder"], strict=strict)

    def _decode(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        lead = context.shape[:-1]
        decoder_in = torch.cat([context, z], dim=-1).reshape(
            -1, self.context_dim + self.latent_dim
        )
        pred = self.decoder(decoder_in)
        return pred.reshape(*lead, *self.target_shape)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: sample ``z`` from ``p(z|context)`` and decode. Returns ``(pred, z)``."""
        mu, logvar = gaussian_moments_from_head(self.prior(context))
        entropy = entropy_gaussian(mu.detach(), logvar.detach())
        z = mu + torch.randn_like(mu) * torch.exp(logvar * 0.5)
        pred = self._decode(context, z)
        return pred, z, entropy

    def sample_prior(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``z ~ p(z|context)`` and decode.

        ``context`` is proprio- or extero-conditioned, matching ``forward``.

        If ``num_samples == 1``, shapes match ``forward``: ``pred`` is
        ``(..., *target_shape)``, ``z`` is ``(..., latent_dim)``. Otherwise a sample axis is
        inserted: ``(..., num_samples, *target_shape)`` and ``(..., num_samples, latent_dim)``.
        """
        mu, logvar = gaussian_moments_from_head(self.prior(context))
        entropy = entropy_gaussian(mu.detach(), logvar.detach())
        std = torch.exp(0.5 * logvar)
        if num_samples == 1:
            z = mu + torch.randn_like(mu) * std
            pred = self._decode(context, z)
            return pred, z, entropy
        eps = torch.randn(
            *mu.shape[:-1],
            num_samples,
            mu.shape[-1],
            device=mu.device,
            dtype=mu.dtype,
        )
        z = mu.unsqueeze(-2) + eps * std.unsqueeze(-2)
        ctx = context.unsqueeze(-2).expand(
            *context.shape[:-1], num_samples, context.shape[-1]
        )
        pred = self._decode(ctx, z)
        return pred, z, entropy

    def compute_loss(
        self,
        proprio_context: torch.Tensor, # (N, C)
        extero_context: torch.Tensor, # (N, C)
        target: torch.Tensor, # (N, *target_shape)
        prior_kl_coef: float = 0.5,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """ELBO-style loss: recon with ``z ~ q``, ``KL(q ‖ p(z|o_ext))``, and ``prior_kl_coef * KL(p_ext ‖ p_pro)``.

        - ``kl_posterior`` is ``KL(q(z|o_ext,y) ‖ p(z|o_ext))`` with ``o``.
        - ``kl_prior`` is ``KL(p(z|o_ext) ‖ p(z|o_pro))`` with extero moments detached so gradients align ``p(z|o_pro)`` toward ``p(z|o_ext)``.

        Return dict keys match PPO logging names, e.g. ``"future/latent_ig"`` is the minibatch mean of
        ``kl_prior`` (prior-shift / IG-flavored term). KL and pred entries are masked sums divided by
        valid count (same as mean per valid element). Entropy entries are masked means when
        ``valid_mask`` is set.
        """
        if target.shape[-2:] != self.target_shape:
            raise ValueError(
                f"target must end with target_shape={tuple(self.target_shape)}, got {tuple(target.shape)}"
            )
        if proprio_context.shape[:-1] != target.shape[:-2]:
            raise ValueError(
                f"proprio_context leading shape {tuple(proprio_context.shape[:-1])} "
                f"must match target leading shape {tuple(target.shape[:-2])}"
            )
        if extero_context.shape[:-1] != target.shape[:-2]:
            raise ValueError(
                f"extero_context leading shape {tuple(extero_context.shape[:-1])} "
                f"must match target leading shape {tuple(target.shape[:-2])}"
            )

        lead = target.shape[:-2]
        flat_n = math.prod(lead) if lead else 1
        proprio_context = proprio_context.reshape(flat_n, self.context_dim)
        extero_context = extero_context.reshape(flat_n, self.context_dim)
        target = target.reshape(flat_n, *self.target_shape)

        proprio_mu, proprio_logvar = gaussian_moments_from_head(
            self.prior(proprio_context)
        )
        extero_mu, extero_logvar = gaussian_moments_from_head(self.prior(extero_context))

        target_feature = self.target_encoder(target)
        posterior_in = torch.cat([extero_context, target_feature], dim=-1)
        posterior_mu, posterior_logvar = gaussian_moments_from_head(
            self.posterior(posterior_in)
        )

        entropy_prior_proprio = entropy_gaussian(proprio_mu.detach(), proprio_logvar.detach())
        entropy_prior_extero = entropy_gaussian(extero_mu.detach(), extero_logvar.detach())
        entropy_posterior = entropy_gaussian(posterior_mu.detach(), posterior_logvar.detach())

        kl_posterior = kl_gaussian(
            posterior_mu,
            posterior_logvar,
            extero_mu,
            extero_logvar,
        )
        kl_prior = kl_gaussian(
            extero_mu.detach(),
            extero_logvar.detach(),
            proprio_mu,
            proprio_logvar,
        )

        z = posterior_mu + torch.randn_like(posterior_mu) * torch.exp(
            posterior_logvar * 0.5
        )
        pred = self._decode(extero_context, z)
        # yaw is ``target[..., 3:4]`` (relative pos, yaw, linvel); scale its squared error by ``1/pi``.
        assert pred.shape == target.shape
        sq_err = (pred - target) ** 2
        sq_err[..., 3:4] = sq_err[..., 3:4] / torch.pi
        likelihood = sq_err.sum(dim=(-1, -2))

        per_term = likelihood + self.kl_coef * kl_posterior + prior_kl_coef * kl_prior

        if valid_mask is not None:
            maskf = valid_mask.to(dtype=likelihood.dtype)
            if maskf.shape[-1] == 1 and maskf.dim() == likelihood.dim() + 1:
                maskf = maskf.squeeze(-1)
            maskf = maskf.reshape_as(likelihood)
            denom = maskf.sum().clamp_min(1.0)
            loss = (per_term * maskf).sum() / denom
            recon_weight = maskf.sum()
            sqerr = (likelihood * maskf).sum()
            kl_sum = (kl_posterior * maskf).sum()
            kl_prior_sum = (kl_prior * maskf).sum()
            latent_ig_mean = kl_prior_sum / denom
            entropy_prior_proprio_mean = (entropy_prior_proprio * maskf).sum() / denom
            entropy_prior_extero_mean = (entropy_prior_extero * maskf).sum() / denom
            entropy_posterior_mean = (entropy_posterior * maskf).sum() / denom
        else:
            loss = per_term.mean()
            n = float(likelihood.numel())
            recon_weight = likelihood.new_tensor(n)
            sqerr = likelihood.sum()
            kl_sum = kl_posterior.sum()
            kl_prior_sum = kl_prior.sum()
            latent_ig_mean = kl_prior.mean()
            entropy_prior_proprio_mean = entropy_prior_proprio.mean()
            entropy_prior_extero_mean = entropy_prior_extero.mean()
            entropy_posterior_mean = entropy_posterior.mean()

        rw = recon_weight.clamp_min(1.0)
        ld = float(self.latent_dim)
        info = {
            "future/pred_loss": sqerr / rw,
            "future/KL(q(z|o_ext,y)||p(z|o_ext))": kl_sum / rw,
            "future/KL(p(z|o_ext)||p(z|o_pro))": kl_prior_sum / rw,
            "future/latent_ig": latent_ig_mean.detach(),
            "future/H[p(z|o_pro)]": entropy_prior_proprio_mean.detach(),
            "future/H[p(z|o_ext)]": entropy_prior_extero_mean.detach(),
            "future/H[q(z|o_ext,y)]": entropy_posterior_mean.detach(),
            # Same as above, but total nats / latent_dim (typical range O(1) when not collapsed)
            "future/nats_per_latent/p(z|o_pro)": (entropy_prior_proprio_mean / ld).detach(),
            "future/nats_per_latent/p(z|o_ext)": (entropy_prior_extero_mean / ld).detach(),
            "future/nats_per_latent/q(z|o_ext,y)": (entropy_posterior_mean / ld).detach(),
        }
        return loss, info
