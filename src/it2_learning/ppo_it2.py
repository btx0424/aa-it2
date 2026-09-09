# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torch.utils._pytree as pytree
import functools
from contextlib import nullcontext

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as TDMod,
    TensorDictSequential as TDSeq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, asdict
from typing import Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import (
    IndependentNormal,
    VecNorm,
    MLP,
)
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import check_parameters
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.ppo.common import (
    make_batch,
    Actor,
    Critic,
    GAE,
    ResidualFC,
    OBS_KEY, ACTION_KEY, REWARD_KEY, TERM_KEY, DONE_KEY,
)
CMD_KEY = "primary_command"

from it2_learning.encoders import EncoderTwo, build_policy_attn_masks

import active_adaptation as aa
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_it2.PPOConfig"
    name: str = "ppo_it2"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    token_dim: int = 512

    muon: bool = False
    compile: bool = False
    use_ddp: bool = True
    use_amp: bool = False

    value_loss_coef: float = 0.5
    # EMA momentum for the return std used to normalize the value loss.
    # Updated once per rollout from full-batch returns; all minibatch steps within
    # that rollout use the same frozen value (analogous to VecNorm freeze).
    ret_std_ema_momentum: float = 0.99
    # LR multiplier for the encoder and cmd_encoder relative to actor/critic.
    # The encoder is a deeper shared network and benefits from a more conservative update.
    encoder_lr_scale: float = 0.5

    in_keys: Tuple[str, ...] = (
        CMD_KEY, "command_mode",
        OBS_KEY, "extero",
    )
    # Extero: "cnn" = small built-in conv stack; "defm_cnn" = DeFM ResNet/RegNet + BiFPN backbone.
    extero_encoder: str = "cnn" # or "defm_cnn"
    defm_variant: str = "defm_resnet18"  # or "defm_regnet_y_400mf"
    defm_pretrained: bool = True

    def get_class(self):
        return PPOPolicy


cs = ConfigStore.instance()
cs.store("ppo_it2", node=PPOConfig(), group="algo")

class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: Composite, 
        action_spec: Composite, 
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, PPOConfig) else PPOConfig(**cfg)
        self.device = torch.device(device)

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 2.0
        self.token_dim = self.cfg.token_dim
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        
        fake_input = observation_spec.zero()
        cmd_shape = fake_input[CMD_KEY].shape[-1:] # (D,)
        proprio_shape = fake_input[OBS_KEY].shape[-1:] # (D,)
        extero_shape = fake_input["extero"].shape[-3:] # (C, H, W)
        self.action_dim = env.action_manager.action_dim

        self.mlp_norm = VecNorm(proprio_shape, proprio_shape, 1.0)
        self.cnn_norm = VecNorm(extero_shape, [extero_shape[0], 1, 1], 1.0)
        
        self.vecnorm = TDSeq(
            TDMod(self.mlp_norm, [OBS_KEY], ["_obs_normed"]),
            TDMod(self.cnn_norm, ["extero"], ["_extero_normed"]),
        ).to(self.device)
        
        self.cmd_transform = env.observation_groups[CMD_KEY].symmetry_transform().to(self.device)

        self.cmd_vecnorm = VecNorm(cmd_shape, decay=1.0).to(self.device)

        self.obs_transform = env.observation_groups[OBS_KEY].symmetry_transform().to(self.device)
        self.extero_transform = env.observation_groups["extero"].symmetry_transform().to(self.device)
        self.act_transform = env.action_manager.symmetry_transform().to(self.device)

        _actor = nn.Sequential(ResidualFC(self.token_dim, self.token_dim), Actor(self.action_dim))
        _critic = nn.Sequential(ResidualFC(self.token_dim, self.token_dim), Critic(1))
        
        self.cmd_encoder = MLP([cmd_shape[-1], self.token_dim], first_non_muon=True).to(self.device)

        self.fusion_encoder: nn.Module = EncoderTwo(
            proprio_shape,
            extero_shape,
            token_dim=self.token_dim,
            extero_encoder=self.cfg.extero_encoder,
            defm_variant=self.cfg.defm_variant,
            defm_pretrained=self.cfg.defm_pretrained,
        ).to(self.device)

        actor_module = TDMod(_actor, ["_shared_feature"], ["loc", "scale"])
        critic_module = TDMod(_critic, ["_shared_feature"], ["state_value"])

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = critic_module.to(self.device)

        with torch.no_grad():
            self.run_policy(fake_input, actor=True, critic=True)
        
        def init_(module):
            if getattr(module, "_defm_no_reinit", False):
                return                
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.02)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.)
        
        self.cmd_encoder.apply(init_)
        self.fusion_encoder.apply(init_)
        self.actor.apply(init_)
        self.critic.apply(init_)

        if aa.is_distributed():
            self._configure_distributed()
        self._configure_optimizers()

        self.update = self._update
        # Running EMA of return std for scale-invariant value loss normalization.
        # Initialized conservatively at 1.0 so early training uses a larger (not smaller)
        # value gradient — the EMA ramps up to the true scale within the first few rollouts.
        self._ret_std_ema: float = 1.0

    @classmethod
    def from_env(cls, cfg: PPOConfig, env, device: str):
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
            env=env,
        )

    def run_policy(
        self,
        tensordict: TensorDict,
        *,
        actor: bool = False,
        critic: bool = False,
    ) -> TensorDict:
        self.vecnorm(tensordict)
        cmd_normed = self.cmd_vecnorm(tensordict[CMD_KEY])
        cmd_query = self.cmd_encoder(cmd_normed).reshape(*tensordict.shape, 1, self.token_dim)

        attn_self, attn_cross = build_policy_attn_masks(1, device=self.device)
        feature = self.fusion_encoder.forward(
            cmd_query,
            tensordict["_obs_normed"],
            tensordict["_extero_normed"],
            attn_mask_self=attn_self,
            attn_mask_cross=attn_cross,
        )
        tensordict["_shared_feature"] = feature.squeeze(-2)
        if actor:
            tensordict = self.actor(tensordict)
        if critic:
            tensordict = self.critic(tensordict)
        return tensordict
    
    def _configure_distributed(self):
        if self.cfg.use_ddp:
            aa.bind_local_rank_device()
            local_cuda = aa.get_local_cuda_index()
            self.fusion_encoder = DDP(
                self.fusion_encoder,
                device_ids=[local_cuda],
                find_unused_parameters=True,
            )
            self.cmd_encoder = DDP(self.cmd_encoder, device_ids=[local_cuda])
            self.actor = DDP(self.actor, device_ids=[local_cuda])
            self.critic = DDP(self.critic, device_ids=[local_cuda])
        else:
            for param in self.fusion_encoder.parameters():
                distr.broadcast(param, src=0)
            for param in self.cmd_encoder.parameters():
                distr.broadcast(param, src=0)
            for param in self.actor.parameters():
                distr.broadcast(param, src=0)
            for param in self.critic.parameters():
                distr.broadcast(param, src=0)
        self.world_size = aa.get_world_size()
    
    def _configure_optimizers(self):
        encoder_lr = self.cfg.lr * self.cfg.encoder_lr_scale
        if self.cfg.muon:
            # MuonAdamWWrapper does not support per-group LR; encoder_lr_scale has no effect here.
            self.opt = MuonAdamWWrapper(
                [self.fusion_encoder, self.cmd_encoder, self.actor, self.critic],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    # param groups 0-1: encoder (lower LR — deeper shared network)
                    {"params": self.fusion_encoder.parameters(), "lr": encoder_lr},
                    {"params": self.cmd_encoder.parameters(), "lr": encoder_lr},
                    # param groups 2-3: actor / critic heads
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
        if self.cfg.use_amp and self.device.type != "cuda":
            warnings.warn(
                "PPOConfig.use_amp=True requires a CUDA device; mixed precision disabled.",
                UserWarning,
                stacklevel=2,
            )
        self._amp_enabled = bool(self.cfg.use_amp) and self.device.type == "cuda"
        self._amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

    def on_stage_start(self, stage: str, env=None):
        pass

    def get_rollout_policy(self, mode: str="train", critic: bool=False):
        policy = functools.partial(self.run_policy, actor=True, critic=critic)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        tensordict = tensordict.exclude("stats", ("next", "stats"))
        return self.train_policy(tensordict)

    def train_policy(self, tensordict: TensorDict):
        if hasattr(self, "prev_cfg") and self.prev_cfg.muon != self.cfg.muon:
            raise RuntimeError(
                "Muon optimizer setting must be consistent across runs/checkpoints: "
                f"checkpoint muon={self.prev_cfg.muon}, current muon={self.cfg.muon}."
            )
        
        infos = []
        with ScopedTimer("compute_advantage"):
            self._compute_advantage(tensordict, "adv", "ret")

        # Update EMA of return std from the full rollout before the PPO epoch loop,
        # so every minibatch update for this rollout uses the same normalization factor.
        # In distributed training, synchronize the local ret_std across ranks first so
        # all ranks update their EMA with the same global value — matching the scale of
        # gradients that DDP will average.
        ret_std_t = tensordict["ret"].std()
        if aa.is_distributed():
            distr.all_reduce(ret_std_t, op=distr.ReduceOp.SUM)
            ret_std_t = ret_std_t / aa.get_world_size()
        m = self.cfg.ret_std_ema_momentum
        self._ret_std_ema = m * self._ret_std_ema + (1.0 - m) * ret_std_t.item()

        action = tensordict[ACTION_KEY]
        adv_unnormalized = tensordict["adv"].clone()
        log_probs_before = tensordict["action_log_prob"]
        
        adv = tensordict["adv"]
        role = tensordict[CMD_KEY][:, :, -2].bool()
        adv[role], std0 = normalize(adv[role], subtract_mean=True) # chaser
        adv[~role], std1 = normalize(adv[~role], subtract_mean=True) # evader
        tensordict["adv"] = adv

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                minibatch = self._augment_symmetry(minibatch)
                with ScopedTimer("update_minibatch"):
                    infos.append(self.update(minibatch))
        
        with torch.no_grad():
            tensordict_ = tensordict.copy()
            self.run_policy(tensordict_, actor=True, critic=False)
            dist = IndependentNormal(tensordict_["loc"], tensordict_["scale"])
            log_probs_after = dist.log_prob(action)
            log_ratio = (log_probs_after - log_probs_before).reshape_as(adv_unnormalized)
            policy_gain = log_ratio * adv_unnormalized
            weighted_ratio = log_ratio.exp() * adv_unnormalized
                
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["curriculum/std_chaser"] = std0.mean().item()
        infos["curriculum/std_evader"] = std1.mean().item()
        infos["actor/lr"] = self.opt.param_groups[-1]["lr"]  # actor/critic groups are last
        infos["encoder/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/policy_gain"] = policy_gain.mean().item()
        infos["actor/weighted_ratio"] = weighted_ratio.mean().item()

        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_var"] = tensordict["ret"].var().item()
        infos["critic/ret_std_ema"] = self._ret_std_ema
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        if aa.is_distributed():
            self.cmd_vecnorm.synchronize(mode="broadcast")
            self.mlp_norm.synchronize(mode="broadcast")
            self.cnn_norm.synchronize(mode="broadcast")
            infos["encoder/diff"] = check_parameters(self.fusion_encoder)
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        adv_key: str="adv",
        ret_key: str="ret",
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                self.run_policy(tensordict_flat, actor=False, critic=True)
                self.run_policy(tensordict_flat["next"], actor=False, critic=True)

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)# .clamp_min(0.)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict
    
    def _augment_symmetry(self, tensordict: TensorDict) -> TensorDict:
        symmetry = tensordict.empty()
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry[CMD_KEY] = self.cmd_transform(tensordict[CMD_KEY])
        symmetry["command_mode"] = tensordict["command_mode"]
        symmetry["extero"] = self.extero_transform(tensordict["extero"])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)
        return tensordict

    def _update(self, tensordict: TensorDict):
        bsize = tensordict.shape[0]

        amp_ctx = (
            torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=self._amp_enabled)
            if self._amp_enabled
            else nullcontext()
        )
        with amp_ctx:
            action_data = tensordict[ACTION_KEY]
            log_probs_data = tensordict["action_log_prob"]
            value_targets = tensordict["ret"]
            adv = tensordict["adv"]

            self.run_policy(tensordict, actor=True, critic=True)

            valid = (~tensordict["is_init"])
            valid_cnt = valid.sum()

            dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
            log_probs = dist.log_prob(action_data)
            entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

            adv = tensordict["adv"]
            log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
            ratio = torch.exp(log_ratio)
            surr1 = adv * ratio
            surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
            policy_loss = - (torch.min(surr1, surr2).reshape_as(valid) * valid).sum() / valid_cnt
            entropy_loss = - self.entropy_coef * entropy

            values = tensordict["state_value"]
            value_loss_raw = self.critic_loss_fn(values, value_targets)
            value_loss_raw = (value_loss_raw.reshape_as(valid) * valid).sum() / valid_cnt
            ret_std_ema = max(self._ret_std_ema, 1.0)
            value_loss = self.cfg.value_loss_coef * value_loss_raw / ret_std_ema ** 2

            loss = policy_loss + entropy_loss + value_loss

        self.opt.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.opt)
        else:
            loss.backward()

        if aa.is_distributed() and not self.cfg.use_ddp:
            allreduce_grads(self.cmd_encoder.parameters())
            allreduce_grads(self.fusion_encoder.parameters())
            allreduce_grads(self.actor.parameters())
            allreduce_grads(self.critic.parameters())

        encoder_grad_norm = nn.utils.clip_grad_norm_(self.fusion_encoder.parameters(), self.max_grad_norm)
        cmd_encoder_grad_norm = nn.utils.clip_grad_norm_(self.cmd_encoder.parameters(), self.max_grad_norm)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        if self._amp_enabled:
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - value_loss_raw / value_targets[valid].var()
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize//2:], self.act_transform(dist.mean[:bsize//2]))
        return {
            "encoder/grad_norm": encoder_grad_norm,
            "encoder/cmd_grad_norm": cmd_encoder_grad_norm,
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/approx_kl": approx_kl,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "critic/value_loss": value_loss_raw.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        state_dict["cfg"] = asdict(self.cfg)
        state_dict["_ret_std_ema"] = self._ret_std_ema
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        if "cfg" in state_dict:
            self.prev_cfg = PPOConfig(**state_dict["cfg"])
        if "_ret_std_ema" in state_dict:
            self._ret_std_ema = state_dict["_ret_std_ema"]
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    std = x.std()
    if subtract_mean:
        return (x - x.mean()) / std.clamp(1e-7), std
    else:
        return x  / std.clamp(1e-7), std


def allreduce_grads(params):
    """Synchronize gradients across ranks for manual (non-DDP) training."""
    for param in params:
        if param.grad is None:
            continue
        distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
        param.grad /= aa.get_world_size()
