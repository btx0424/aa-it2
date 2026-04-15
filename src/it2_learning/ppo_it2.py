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
from dataclasses import dataclass
from typing import Union, Tuple, Literal
from collections import OrderedDict

from active_adaptation.learning.modules import (
    IndependentNormal,
    VecNorm,
    MLP,
)
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import check_parameters
from active_adaptation.learning.ppo.common import (
    make_batch,
    Actor,
    Critic,
    GAE,
    ResidualFC,
    CMD_KEY, OBS_KEY, ACTION_KEY, REWARD_KEY, TERM_KEY, DONE_KEY,
)
from it2_learning.encoders import EncoderOne, EncoderTwo

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
    cmd_feature_dim: int = 128

cs = ConfigStore.instance()
cs.store("ppo_it2", node=PPOConfig, group="algo")


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
        self.device = device

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

        def make_fusion_encoder(out_key: str):
            return TDMod(
                EncoderClass(
                    proprio_shape,
                    extero_shape,
                    token_dim=self.cfg.cmd_feature_dim,
                ),
                ["_cmd_feature", "_obs_normed", "_extero_normed"],
                [out_key],
            )
        
        self.cmd_feature_encoder = TDMod(
            MLP(
                [cmd_shape[-1], 128, self.cfg.cmd_feature_dim],
                first_non_muon=True,
            ),
            ["_cmd_normed"],
            ["_cmd_feature"],
        ).to(self.device)

        self.encoder = make_fusion_encoder("_shared_feature").to(self.device)
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

        self.vecnorm(fake_input)
        self.cmd_feature_encoder(fake_input)
        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)
        
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
        self.encoder.apply(init_)
        self.actor.apply(init_)
        self.critic.apply(init_)

        if aa.is_distributed():
            if self.cfg.use_ddp:
                local_rank = aa.get_local_rank()
                self.encoder = DDP(self.encoder, device_ids=[local_rank])
                self.cmd_feature_encoder = DDP(self.cmd_feature_encoder, device_ids=[local_rank])
                self.actor = DDP(self.actor, device_ids=[local_rank])
                self.critic = DDP(self.critic, device_ids=[local_rank])
            else:
                for param in self.encoder.parameters():
                    distr.broadcast(param, src=0)
                for param in self.cmd_feature_encoder.parameters():
                    distr.broadcast(param, src=0)
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
            self.world_size = aa.get_world_size()
        self._configure_optimizers()

        _dev = torch.device(device) if not isinstance(device, torch.device) else device
        if self.cfg.use_amp and _dev.type != "cuda":
            warnings.warn(
                "PPOConfig.use_amp=True requires a CUDA device; mixed precision disabled.",
                UserWarning,
                stacklevel=2,
            )
        self._amp_enabled = bool(self.cfg.use_amp) and _dev.type == "cuda"
        if self._amp_enabled:
            self._amp_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self._amp_dtype = torch.float32
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

        self.update = self._update
        # if self.cfg.compile and not aa.is_distributed():
        #     # TODO: compile for multi-gpu training?
        #     self.update = torch.compile(self.update, fullgraph=True)
        #     # self.update = CudaGraphModule(self.update)
        self.root_state_w = None
    
    def _configure_optimizers(self):
        if self.cfg.muon:
            self.opt = MuonAdamWWrapper(
                [self.encoder, self.cmd_feature_encoder, self.actor, self.critic],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.encoder.parameters()},
                    {"params": self.cmd_feature_encoder.parameters()},
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01
            )

    def on_stage_start(self, stage: str):
        pass

    def get_rollout_policy(self, mode: str="train", critic: bool=False):
        modules = [self.vecnorm, self.cmd_feature_encoder, self.encoder]
        modules.append(self.actor)
        if critic:
            modules.append(self.critic)
        policy = TDSeq(*modules)
        if self.cfg.compile:
            policy = torch.compile(policy)
        return policy

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"

        tensordict = tensordict.exclude("stats", ("next", "stats"))
        policy_info = self.train_policy(tensordict)

        root_state_w = tensordict.get("root_state_w", None)
        if root_state_w is not None and self.root_state_w is None:
            # TODO: learn to predict the relative state in the future
            # for i in range(self.cfg.train_every):
            #     s_curr = self.root_state_w[:, i]
            #     s_next = root_state_w[:, i]
            #     # transform s_next to the current frame
            pass

        self.root_state_w = root_state_w.clone()
        
        return dict(sorted(policy_info.items()))

    def train_policy(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret")
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
                infos.append(self.update(minibatch))
        
        with torch.no_grad():
            tensordict_ = tensordict.copy()
            self.encoder(tensordict_)
            tensordict_ = self.actor(tensordict_)
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
            infos["encoder/diff"] = check_parameters(self.encoder)
        return infos

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TDMod, 
        adv_key: str="adv",
        ret_key: str="ret",
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                self.vecnorm(tensordict_flat)
                self.cmd_feature_encoder(tensordict_flat)
                self.encoder(tensordict_flat)
                critic(tensordict_flat)
                self.vecnorm(tensordict_flat["next"])
                self.cmd_feature_encoder(tensordict_flat["next"])
                self.encoder(tensordict_flat["next"])
                critic(tensordict_flat["next"])

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
            self.vecnorm(tensordict)
            self.cmd_feature_encoder(tensordict)
            self.encoder(tensordict)

            valid = (~tensordict["is_init"])
            valid_cnt = valid.sum()

            action_data = tensordict[ACTION_KEY]
            log_probs_data = tensordict["action_log_prob"]
            self.actor(tensordict)
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

            b_returns = tensordict["ret"]
            values = self.critic(tensordict)["state_value"]
            value_loss = self.critic_loss_fn(b_returns, values)
            value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

            loss = policy_loss + entropy_loss + value_loss

        self.opt.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.opt)
        else:
            loss.backward()

        if aa.is_distributed() and not self.cfg.use_ddp:
            for param in self.cmd_feature_encoder.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= aa.get_world_size()
            for param in self.encoder.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= aa.get_world_size()
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= aa.get_world_size()
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= aa.get_world_size()

        encoder_grad_norm = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        if self._amp_enabled:
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var()
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
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    std = x.std()
    if subtract_mean:
        return (x - x.mean()) / std.clamp(1e-7), std
    else:
        return x  / std.clamp(1e-7), std
