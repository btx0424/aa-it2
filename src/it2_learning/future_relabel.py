"""
Future prediction targets from consecutive `root_state_w` samples.

**Convention (built-in modes):** The first three components are always body-frame
`rel_pos` (relative to `prev_tensordict`), so eval code can slice `..., :3` for
position overlays without depending on the mode name.

`root_state_w` layout: pos (3), quat (4), lin vel (3), ang vel (3), ...

**Relabel keys**

- ``_future_target``: predicted quantity at each output time.
- ``_future_valid``: boolean mask, shape ``(N, t_out)``, same time layout as
  ``_future_target`` (with ``t_out = T - H`` for :class:`FutureState`). Per output
  time ``t``, it marks valid training when start and H-step target share an episode; see
  each ``relabel`` for the predicate. Loss code can weight or mask on this tensor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    euler_from_quat,
    wrap_to_pi,
)
from active_adaptation.learning.modules import VecNorm

# Mode name -> target dimension (last axis). Keep in sync with `compute_future_target`.
FUTURE_LABEL_MODES: dict[str, tuple[int, tuple[str, ...]]] = {
    "state7": (7, ("rel_pos", "rel_yaw", "rel_linvel")),
    "state10": (10, ("rel_pos", "rel_yaw", "rel_linvel", "rel_angvel")),
}


class FutureRelabel(nn.Module, ABC):
    """Build ``_future_target`` and ``_future_valid`` from ``root_state_w`` (and friends)."""

    mode: str
    horizon: int
    target_shape: torch.Size
    keys: tuple[str, ...]
    vecnorm: VecNorm

    @abstractmethod
    def relabel(self, tensordict: TensorDict, normalize: bool = True) -> TensorDict:
        """Return a sliced tensordict with ``_future_target`` and ``_future_valid``."""
        ...

    def denormalize(self, input_vector: torch.Tensor):
        return self.vecnorm.denormalize(input_vector)


class FutureState(FutureRelabel):

    def __init__(self, mode: str, horizon: int) -> None:
        super().__init__()
        self.mode = mode
        self.horizon = horizon
        if self.mode not in FUTURE_LABEL_MODES:
            raise ValueError(
                f"Unknown future label mode {self.mode!r}. "
                f"Expected one of {list(FUTURE_LABEL_MODES)}"
            )
        state_dim, self.keys = FUTURE_LABEL_MODES[self.mode]
        self.target_shape = torch.Size((1, state_dim))
        self.vecnorm = VecNorm(self.target_shape, decay=1.0)

    def relabel(self, tensordict: TensorDict, normalize: bool = True) -> TensorDict:
        """Slice to ``t_out = T - H + 1`` and add ``_future_target`` / ``_future_valid``.

        The future label is the root state at time ``t + H`` (after **H** forward steps
        along the rollout axis) relative to the reference at ``t``. This matches a usual
        ``train_every=H`` notion of "H steps into the future" for overlays. Uses
        ``t_out = T - H`` so ``t + H`` never exceeds the last time index (requires ``T > H``).

        ``_future_valid`` is a boolean tensor with shape ``(N, t_out)``. Index ``t`` is
        True when ``episode_id[:, t] == episode_id[:, t + H]`` so the loss can ignore
        pairs that straddle a reset.
        """
        N, T = tensordict.shape[:2]
        H = self.horizon
        t_out = T - H
        if t_out < 1:
            raise ValueError(
                f"T - H = {t_out} < 1: need more time steps than the horizon (T={T}, H={H})"
            )
        result = tensordict[:, :t_out]
        future_targets: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        for t in range(t_out):
            root_state_ref = tensordict["root_state_w"][:, t]
            root_state_future = tensordict["root_state_w"][:, t + H]
            rel_kinematics = relative_kinematics(
                root_state=root_state_future.unsqueeze(1),
                root_state_ref=root_state_ref,
            )
            future_target = torch.cat([rel_kinematics[k] for k in self.keys], dim=-1)
            future_targets.append(future_target)
            valid = tensordict["episode_id"][:, t] == tensordict["episode_id"][:, t + H]
            valid_masks.append(valid)
        future_targets = torch.stack(future_targets, dim=1).reshape(N, t_out, *self.target_shape)
        self.vecnorm._update(future_targets)
        if normalize:
            future_targets = self.vecnorm._normalize(future_targets)
        result["_future_target"] = future_targets
        result["_future_valid"] = torch.stack(valid_masks, dim=1)
        return result


class FutureTrajectory(FutureRelabel):

    def __init__(self, mode: str, horizon: int) -> None:
        super().__init__()
        self.mode = mode
        self.horizon = horizon
        if self.mode not in FUTURE_LABEL_MODES:
            raise ValueError(
                f"Unknown future label mode {self.mode!r}. "
                f"Expected one of {list(FUTURE_LABEL_MODES)}"
            )
        state_dim, self.keys = FUTURE_LABEL_MODES[self.mode]
        self.target_shape = torch.Size((self.horizon, state_dim))
        self.vecnorm = VecNorm(self.target_shape, decay=1.0)

    def relabel(self, tensordict: TensorDict, normalize: bool = True) -> TensorDict:
        """Slice to ``t_out = T - H + 1`` and add ``_future_target`` / ``_future_valid``.

        The target stacks relative kinematics for each time in ``[t, t + H)``, then flattens
        to shape ``(H, state_dim)`` (``target_shape`` for this relabeler).

        ``_future_valid`` is a boolean ``(N, t_out)`` tensor. At output time ``t`` it is
        True when ``episode_id`` at the window start and end match,
        ``episode_id[:, t] == episode_id[:, t + H - 1]``, i.e. the whole horizon window
        lies in a single episode (no reset between the first and last step of the chunk).
        """
        N, T = tensordict.shape[:2]
        H = self.horizon
        t_out = T - H + 1
        if t_out < 1:
            raise ValueError(f"T - H + 1 = {t_out} < 1")
        result = tensordict[:, :t_out]
        future_targets: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        for t in range(t_out):
            root_state_ref = tensordict["root_state_w"][:, t]
            root_state_future = tensordict["root_state_w"][:, t : t + H]
            rel_kinematics = relative_kinematics(
                root_state_future,
                root_state_ref,
            )
            future_target = torch.cat([rel_kinematics[k] for k in self.keys], dim=-1)
            future_targets.append(future_target)
            # whether t and t+H-1 belong to the same episode
            valid = (
                tensordict["episode_id"][:, t] == tensordict["episode_id"][:, t + H - 1]
            )
            valid_masks.append(valid)
        # (N, t_out, H, self.dim) -> flatten trajectory per time for a fixed H * self.dim head
        future_targets = torch.stack(future_targets, dim=1)
        self.vecnorm._update(future_targets)
        if normalize:
            future_targets = self.vecnorm._normalize(future_targets)
        result["_future_target"] = future_targets # (N, t_out, H, state_dim)
        result["_future_valid"] = torch.stack(valid_masks, dim=1)
        return result


def relative_kinematics(
    root_state: Float[torch.Tensor, "N T 13"],
    root_state_ref: Float[torch.Tensor, "N 13"],
) -> dict[str, torch.Tensor]:
    """Express current root kinematics in the previous root body frame (single step)."""
    pos = root_state[..., :3]
    quat = root_state[..., 3:7]
    linvel = root_state[..., 7:10]
    angvel = root_state[..., 10:13]

    pos_ref = root_state_ref[:, None, :3]
    quat_ref = root_state_ref[:, None, 3:7]
    # linvel_ref = root_state_ref[:, 7:10] # unused
    # angvel_ref = root_state_ref[:, 10:13] # unused

    rel_quat = quat_mul(quat_conjugate(quat_ref), quat)  # (N, T, 4)
    rel_yaw = wrap_to_pi(euler_from_quat(rel_quat)[..., 2:3])  # (N, T, 1)
    rel_pos = quat_rotate_inverse(quat_ref, pos - pos_ref)  # (N, T, 3)
    rel_linvel = quat_rotate_inverse(quat_ref, linvel)  # (N, T, 3)
    rel_angvel = quat_rotate_inverse(quat_ref, angvel)  # (N, T, 3)

    return {
        "rel_quat": rel_quat,
        "rel_pos": rel_pos,
        "rel_yaw": rel_yaw,
        "rel_linvel": rel_linvel,
        "rel_angvel": rel_angvel,
    }
