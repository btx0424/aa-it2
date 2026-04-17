from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def kl_gaussian(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence between two Gaussian distributions.

    Sum over the last dimension.
    """
    return 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1
    ).sum(dim=-1)


def maybe_unwrap(module: nn.Module):
    if isinstance(module, DDP):
        module = module.module
    return module


class FuturePredictor(nn.Module):
    """Conditional VAE-style predictor for multi-modal future targets."""

    def __init__(
        self,
        context_dim: int,
        pred_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.pred_dim = pred_dim
        self.latent_dim = latent_dim

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
        self.posterior = nn.Sequential(
            nn.Linear(context_dim + pred_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, pred_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.query_embedding.weight.data.normal_(0.0, 0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.02)
                nn.init.constant_(module.bias, 0.0)

    def wrap_DDP(self, device_ids: list[int]):
        self.query_embedding = DDP(self.query_embedding, device_ids=device_ids)
        self.prior = DDP(self.prior, device_ids=device_ids)
        self.posterior = DDP(self.posterior, device_ids=device_ids)
        self.decoder = DDP(self.decoder, device_ids=device_ids)

    def state_dict(self):
        state_dict = OrderedDict()
        state_dict["query_embedding"] = maybe_unwrap(self.query_embedding).state_dict()
        state_dict["prior"] = maybe_unwrap(self.prior).state_dict()
        state_dict["posterior"] = maybe_unwrap(self.posterior).state_dict()
        state_dict["decoder"] = maybe_unwrap(self.decoder).state_dict()
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True):
        maybe_unwrap(self.query_embedding).load_state_dict(
            state_dict["query_embedding"], strict=strict
        )
        maybe_unwrap(self.prior).load_state_dict(state_dict["prior"], strict=strict)
        maybe_unwrap(self.posterior).load_state_dict(
            state_dict["posterior"], strict=strict
        )
        maybe_unwrap(self.decoder).load_state_dict(state_dict["decoder"], strict=strict)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference: sample ``z`` from ``p(z|context)`` and decode. Returns ``(pred, z)``."""
        mu, logvar = self.prior(context).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(logvar * 0.5)
        pred = self.decoder(torch.cat([context, z], dim=-1))
        return pred, z

    def sample_prior(
        self,
        context: torch.Tensor,
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``z ~ p(z|context)`` and decode.

        ``context`` is proprio- or extero-conditioned, matching ``forward``.

        If ``num_samples == 1``, shapes match ``forward``: ``pred`` is ``(..., pred_dim)``,
        ``z`` is ``(..., latent_dim)``. Otherwise a sample axis is inserted before the last
        dimension: ``(..., num_samples, pred_dim)`` and ``(..., num_samples, latent_dim)``.
        """
        mu, logvar = self.prior(context).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        if num_samples == 1:
            z = mu + torch.randn_like(mu) * std
            pred = self.decoder(torch.cat([context, z], dim=-1))
            return pred, z
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
        pred = self.decoder(torch.cat([ctx, z], dim=-1))
        return pred, z

    def compute_loss(
        self,
        proprio_context: torch.Tensor,
        extero_context: torch.Tensor,
        target: torch.Tensor,
        kl_coef: float,
        prior_kl_coef: float = 0.5,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """ELBO-style loss: recon with ``z ~ q``, ``KL(q ‖ p(z|o_ext))``, and ``prior_kl_coef * KL(p_ext ‖ p_pro)``.

        - ``kl_posterior`` is ``KL(q(z|o_ext,y) ‖ p(z|o_ext))`` with ``o``.
        - ``kl_prior`` is ``KL(p(z|o_ext) ‖ p(z|o_pro))`` with extero moments detached so gradients align ``p(z|o_pro)`` toward ``p(z|o_ext)``.

        ``meta["latent_ig"]`` is the minibatch mean of ``kl_prior`` (prior-shift / IG-flavored term).
        """
        proprio_mu, proprio_logvar = self.prior(proprio_context).chunk(2, dim=-1)
        extero_mu, extero_logvar = self.prior(extero_context).chunk(2, dim=-1)

        posterior_in = torch.cat([extero_context, target], dim=-1)
        posterior_mu, posterior_logvar = self.posterior(posterior_in).chunk(2, dim=-1)

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
        pred = self.decoder(torch.cat([extero_context, z], dim=-1))
        likelihood = F.mse_loss(pred, target, reduction="none").sum(dim=-1)

        per_term = likelihood + kl_coef * kl_posterior + prior_kl_coef * kl_prior

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
        else:
            loss = per_term.mean()
            n = float(likelihood.numel())
            recon_weight = likelihood.new_tensor(n)
            sqerr = likelihood.sum()
            kl_sum = kl_posterior.sum()
            kl_prior_sum = kl_prior.sum()
            latent_ig_mean = kl_prior.mean()

        w = kl_posterior.new_tensor(1.0)
        meta = {
            "recon_weight": recon_weight,
            "sqerr": sqerr,
            "kl_loss": kl_sum,
            "kl_prior_loss": kl_prior_sum,
            "kl_weight": w,
            "latent_ig": latent_ig_mean.detach(),
        }
        return loss, meta
