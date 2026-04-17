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
from typing import Union, Tuple, Literal
from collections import OrderedDict

from active_adaptation.learning.modules import (
    IndependentNormal,
    VecNorm,
    MLP,
)
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    euler_from_quat,
    wrap_to_pi,
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
    CMD_KEY, OBS_KEY, ACTION_KEY, REWARD_KEY, TERM_KEY, DONE_KEY,
)
from it2_learning.encoders import EncoderOne, EncoderTwo
from it2_learning.networks import FuturePredictor

import active_adaptation as aa
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_it2.PPOPolicy"
    name: str = "ppo_it2"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    encoder_type: str = "one"
    muon: bool = False
    compile: bool = False
    use_ddp: bool = True
    use_amp: bool = False

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, "extero", "root_state_w")
    cnn_norm: str = "none"
    cnn_norm_groups: int = 8
    future_pred_dim: int = 7
    future_pred_coef: float = 0.0
    future_pred_minibatches: int = 4
    future_latent_dim: int = 16
    future_kl_coef: float = 0.02
    future_prior_kl_coef: float = 0.5
    stages: Tuple[str, ...] = ("policy", "future")


cs = ConfigStore.instance()
cs.store("ppo_it2", node=PPOConfig(stages=("policy",)), group="algo")
cs.store("ppo_it2_future", node=PPOConfig(stages=("future",), future_pred_coef=1.0), group="algo")


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
        self.cfg = PPOConfig(**cfg)
        self.device = torch.device(device)

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        
        fake_input = observation_spec.zero()
        cmd_shape = fake_input[CMD_KEY].shape[-1:] # (D,)
        proprio_shape = fake_input[OBS_KEY].shape[-1:] # (D,)
        extero_shape = fake_input["extero"].shape[-3:] # (C, H, W)
        self.action_dim = env.action_manager.action_dim

        self.cmd_norm = VecNorm(cmd_shape, cmd_shape, 1.0)
        self.mlp_norm = VecNorm(proprio_shape, proprio_shape, 1.0)
        self.cnn_norm = VecNorm(extero_shape, [extero_shape[0], 1, 1], 1.0)
        
        self.vecnorm = TDSeq(
            TDMod(self.cmd_norm, [CMD_KEY], ["_cmd_normed"]),
            TDMod(self.mlp_norm, [OBS_KEY], ["_obs_normed"]),
            TDMod(self.cnn_norm, ["extero"], ["_extero_normed"]),
        ).to(self.device)
        
        self.cmd_transform = env.observation_funcs[CMD_KEY].symmetry_transform().to(self.device)
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
        self.extero_transform = env.observation_funcs["extero"].symmetry_transform().to(self.device)
        self.act_transform = env.input_managers[ACTION_KEY].symmetry_transform().to(self.device)

        _actor = nn.Sequential(ResidualFC(256, 256), Actor(self.action_dim))
        _critic = nn.Sequential(ResidualFC(256, 256), Critic(1))

        EncoderClass = {
            "one": EncoderOne,
            "two": EncoderTwo,
        }[self.cfg.encoder_type]
        
        self.cmd_feature_encoder = TDMod(
            MLP(
                [cmd_shape[-1], 128, 256],
                first_non_muon=True,
            ),
            ["_cmd_normed"],
            ["_cmd_feature"],
        ).to(self.device)

        self.fusion_encoder: nn.Module = EncoderClass(
            proprio_shape,
            extero_shape,
            token_dim=256,
            cnn_norm=self.cfg.cnn_norm,
            cnn_norm_groups=self.cfg.cnn_norm_groups,
            hidden_dim=256,
        ).to(self.device)

        self.future_predictor = FuturePredictor(
            context_dim=self.fusion_encoder.output_dim,
            pred_dim=self.cfg.future_pred_dim,
            latent_dim=self.cfg.future_latent_dim,
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

        self.run_policy(fake_input, True, True)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.02)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.)
        
        self.cmd_feature_encoder.apply(init_)
        self.fusion_encoder.apply(init_)
        self.actor.apply(init_)
        self.critic.apply(init_)

        if aa.is_distributed():
            self._configure_distributed()
        self._configure_optimizers()

        self.update = self._update
        # if self.cfg.compile and not aa.is_distributed():
        #     # TODO: compile for multi-gpu training?
        #     self.update = torch.compile(self.update, fullgraph=True)
        #     # self.update = CudaGraphModule(self.update)
        self.prev_tensordict = None
        self._future_enabled = False

    def run_policy(
        self,
        tensordict: TensorDict,
        actor: bool=False,
        critic: bool=False,
        future_prediction: bool=False,
    ) -> TensorDict:
        N = tensordict.shape[0]
        tensordict = self.vecnorm(tensordict)
        tensordict = self.cmd_feature_encoder(tensordict)
        cmd = tensordict["_cmd_feature"].reshape(N, 1, -1)
        if future_prediction:
            future_query = self.future_predictor.query_embedding(torch.arange(2, device=self.device))
            future_query = future_query.expand(N, 2, -1)
            query = torch.cat([cmd, future_query], dim=-2) # [N, 3, token_dim]
            feature = EncoderTwo.forward_policy_future(
                self.fusion_encoder,
                query,
                tensordict["_obs_normed"],
                tensordict["_extero_normed"],
            )
            tensordict["_shared_feature"] = feature[:, 0]
            tensordict["_proprio_context"] = feature[:, 1]
            tensordict["_extero_context"] = feature[:, 2]
        else:
            query = cmd # [N, 1, token_dim]
            feature = EncoderTwo.forward_policy(
                self.fusion_encoder,
                query,
                tensordict["_obs_normed"],
                tensordict["_extero_normed"],
            )
            tensordict["_shared_feature"] = feature[:, 0]
        if actor:
            tensordict = self.actor(tensordict)
        if critic:
            tensordict = self.critic(tensordict)
        return tensordict
    
    def _configure_distributed(self):
        if self.cfg.use_ddp:
            local_rank = aa.get_local_rank()
            self.fusion_encoder = DDP(self.fusion_encoder, device_ids=[local_rank])
            self.cmd_feature_encoder = DDP(self.cmd_feature_encoder, device_ids=[local_rank])
            self.future_predictor.wrap_DDP(device_ids=[local_rank])
            self.actor = DDP(self.actor, device_ids=[local_rank])
            self.critic = DDP(self.critic, device_ids=[local_rank])
        else:
            for param in self.fusion_encoder.parameters():
                distr.broadcast(param, src=0)
            for param in self.cmd_feature_encoder.parameters():
                distr.broadcast(param, src=0)
            for param in self.future_predictor.parameters():
                distr.broadcast(param, src=0)
            for param in self.actor.parameters():
                distr.broadcast(param, src=0)
            for param in self.critic.parameters():
                distr.broadcast(param, src=0)
        self.world_size = aa.get_world_size()
    
    def _configure_optimizers(self):
        if self.cfg.muon:
            self.opt = MuonAdamWWrapper(
                [self.fusion_encoder, self.cmd_feature_encoder, self.actor, self.critic],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.fusion_encoder.parameters()},
                    {"params": self.cmd_feature_encoder.parameters()},
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
        # Future prediction always uses its own AdamW optimizer.
        self.opt_future = torch.optim.AdamW(
            [
                {"params": self.future_predictor.parameters()},
            ],
            lr=self.cfg.lr,
            weight_decay=0.01,
        )

        if self.cfg.use_amp and self.device.type != "cuda":
            warnings.warn(
                "PPOConfig.use_amp=True requires a CUDA device; mixed precision disabled.",
                UserWarning,
                stacklevel=2,
            )
        self._amp_enabled = bool(self.cfg.use_amp) and self.device.type == "cuda"
        if self._amp_enabled:
            self._amp_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self._amp_dtype = torch.float32
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

    def on_stage_start(self, stage: str):
        self._future_enabled = (stage == "future")
        # Reset temporal cache at stage boundaries to avoid cross-stage leakage.
        self.prev_tensordict = None

    def get_rollout_policy(self, mode: str="train", critic: bool=False):
        if mode == "eval" and self.cfg.stages[0] == "future":
            def policy(tensordict: TensorDict):
                self.run_policy(tensordict, actor=True, critic=False, future_prediction=True)
                pred_0, _ = self.future_predictor.sample_prior(
                    tensordict["_proprio_context"],
                    num_samples=3,
                )
                pred_1, _ = self.future_predictor.sample_prior(
                    tensordict["_extero_context"],
                    num_samples=3,
                )
                tensordict["proprio_pred"] = pred_0[..., :3]
                tensordict["extero_pred"] = pred_1[..., :3]
                return tensordict
        else:
            policy = functools.partial(self.run_policy, actor=True, critic=critic)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"

        infos = {}
        tensordict = tensordict.exclude("stats", ("next", "stats"))
        if not self._future_enabled:
            infos.update(self.train_policy(tensordict))

        if (
            self._future_enabled
            and self.prev_tensordict is not None
            and self.cfg.future_pred_coef > 0.0
        ):
            infos.update(self.train_future_prediction(tensordict))
        self.prev_tensordict = tensordict.clone()

        return dict(sorted(infos.items()))
    
    def train_future_prediction(self, tensordict: TensorDict):
        self.fusion_encoder.requires_grad_(False) # freeze the fusion encoder
        eid = tensordict["episode_id"]
        prev_eid = self.prev_tensordict["episode_id"] # (N, T)
        root_state_w = tensordict.get("root_state_w")
        prev_root_state_w = self.prev_tensordict.get("root_state_w")
        same_episode = (eid == prev_eid).reshape_as(tensordict["is_init"]) # (N, T, 1)
        valid_mask = same_episode & (~tensordict["is_init"]) & (~self.prev_tensordict["is_init"])

        prev_pos = prev_root_state_w[..., :3]
        prev_quat = prev_root_state_w[..., 3:7]
        curr_pos = root_state_w[..., :3]
        curr_quat = root_state_w[..., 3:7]
        curr_linvel = root_state_w[..., 7:10]

        # convert to relative
        rel_pos = quat_rotate_inverse(prev_quat, curr_pos - prev_pos)
        rel_quat = quat_mul(quat_conjugate(prev_quat), curr_quat)
        rel_yaw = wrap_to_pi(euler_from_quat(rel_quat)[..., 2]).unsqueeze(-1)
        rel_linvel = quat_rotate_inverse(prev_quat, curr_linvel)
        target = torch.cat([rel_pos, rel_yaw, rel_linvel], dim=-1)
        future_td = self.prev_tensordict.select(CMD_KEY, OBS_KEY, "extero").copy()
        future_td["_future_target"] = target
        future_td["_future_valid"] = valid_mask

        total_weight = target.new_zeros(())
        total_sqerr = target.new_zeros(())
        total_kl_weighted = target.new_zeros(())
        total_kl_prior_weighted = target.new_zeros(())
        total_latent_ig = target.new_zeros(())
        for minibatch in make_batch(future_td, self.cfg.future_pred_minibatches):
            self.run_policy(minibatch, future_prediction=True)
            loss, meta = self.future_predictor.compute_loss(
                minibatch["_proprio_context"],
                minibatch["_extero_context"],
                minibatch["_future_target"],
                self.cfg.future_kl_coef,
                prior_kl_coef=self.cfg.future_prior_kl_coef,
                valid_mask=minibatch.get("_future_valid"),
            )

            self.opt_future.zero_grad(set_to_none=True)
            loss.backward()
            if aa.is_distributed() and not self.cfg.use_ddp:
                allreduce_grads(self.future_predictor.parameters())
            self.opt_future.step()
            total_weight += meta["recon_weight"]
            total_sqerr += meta["sqerr"]
            total_kl_weighted += meta["kl_loss"] * meta["kl_weight"]
            total_kl_prior_weighted += meta["kl_prior_loss"] * meta["kl_weight"]
            total_latent_ig += meta["latent_ig"] * meta["recon_weight"]

        self.fusion_encoder.requires_grad_(True)
        pred_loss = total_sqerr / total_weight.clamp_min(1.0)
        kl_loss = total_kl_weighted / valid_mask.float().sum().clamp_min(1.0)
        kl_prior_loss = total_kl_prior_weighted / valid_mask.float().sum().clamp_min(1.0)
        latent_ig = total_latent_ig / total_weight.clamp_min(1.0)
        param_diff = check_parameters(self.future_predictor)
        return {
            "future/param_diff": param_diff,
            "future/pred_loss": pred_loss.detach().item(),
            "future/KL(q(z|o_ext,y)||p(z|o_ext))": kl_loss.detach().item(),
            "future/KL(p(z|o_ext)||p(z|o_pro))": kl_prior_loss.detach().item(),
            "future/latent_ig": latent_ig.detach().item(),
            "future/valid_ratio": valid_mask.float().mean().detach().item(),
        }

    def train_policy(self, tensordict: TensorDict):
        if hasattr(self, "prev_cfg") and self.prev_cfg.muon != self.cfg.muon:
            raise RuntimeError(
                "Muon optimizer setting must be consistent across runs/checkpoints: "
                f"checkpoint muon={self.prev_cfg.muon}, current muon={self.cfg.muon}."
            )
        
        infos = []
        with ScopedTimer("compute_advantage"):
            self._compute_advantage(tensordict, "adv", "ret")
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
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/policy_gain"] = policy_gain.mean().item()
        infos["actor/weighted_ratio"] = weighted_ratio.mean().item()

        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_var"] = tensordict["ret"].var().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        if aa.is_distributed():
            self.cmd_norm.synchronize(mode="broadcast")
            self.mlp_norm.synchronize(mode="broadcast")
            self.cnn_norm.synchronize(mode="broadcast")
            infos["encoder/diff"] = check_parameters(self.fusion_encoder)
        return infos

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
                self.run_policy(tensordict_flat, False, True)
                self.run_policy(tensordict_flat["next"], False, True)

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
        symmetry[CMD_KEY] = self.cmd_transform(tensordict[CMD_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
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

            self.run_policy(tensordict, True, True)

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
            value_loss = self.critic_loss_fn(values, value_targets)
            value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

            loss = policy_loss + entropy_loss + value_loss

        self.opt.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.opt)
        else:
            loss.backward()

        if aa.is_distributed() and not self.cfg.use_ddp:
            allreduce_grads(self.cmd_feature_encoder.parameters())
            allreduce_grads(self.fusion_encoder.parameters())
            allreduce_grads(self.actor.parameters())
            allreduce_grads(self.critic.parameters())

        encoder_grad_norm = nn.utils.clip_grad_norm_(self.fusion_encoder.parameters(), self.max_grad_norm)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        if self._amp_enabled:
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - value_loss / value_targets[valid].var()
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize//2:], self.act_transform(dist.mean[:bsize//2]))
        return {
            "encoder/grad_norm": encoder_grad_norm,
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/approx_kl": approx_kl,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "critic/value_loss": value_loss.detach(),
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

