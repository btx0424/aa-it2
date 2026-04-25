"""
Future prediction targets from consecutive `root_state_w` samples.

**Convention (built-in modes):** The first three components are always body-frame
`rel_pos` (relative to `prev_tensordict`), so eval code can slice `..., :3` for
position overlays without depending on the mode name.

`root_state_w` layout: pos (3), quat (4), lin vel (3), ang vel (3), ...

**Relabel keys**

- ``_future_target``: predicted quantity at each output time.
- ``_future_valid``: boolean mask, shape ``(N, t_out)``, same time layout as
  ``_future_target``. Per output time ``t``, it marks whether the future label is
  considered valid for training (no episode break inside the compared span); see
  :meth:`FutureState.relabel` and :meth:`FutureTrajectory.relabel` for the exact
  predicate. Loss code can weight or mask on this tensor.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from tensordict import TensorDict
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_mul,
    quat_conjugate,
    euler_from_quat,
    wrap_to_pi,
)

# Mode name -> target dimension (last axis). Keep in sync with `compute_future_target`.
FUTURE_LABEL_MODES: dict[str, tuple[int, tuple[str, ...]]] = {
    "state7": (7, ("rel_pos", "rel_yaw", "rel_linvel")),
    "state10": (10, ("rel_pos", "rel_yaw", "rel_linvel", "rel_angvel")),
}


class FutureState:
    def __init__(self, mode: str, horizon: int):
        self.mode = mode
        self.horizon = horizon
        if self.mode not in FUTURE_LABEL_MODES:
            raise ValueError(
                f"Unknown FutureState mode {self.mode!r}. "
                f"Expected one of {list(FUTURE_LABEL_MODES)}"
            )
        self.dim, self.keys = FUTURE_LABEL_MODES[self.mode]

    def relabel(self, tensordict: TensorDict) -> TensorDict:
        """Slice to ``t_out = T - H + 1`` and add ``_future_target`` / ``_future_valid``.

        The future label is the root state at time ``t + H`` relative to the reference
        at ``t``.

        ``_future_valid`` is a boolean tensor with shape ``(N, t_out)``. Index ``t`` is
        True when ``episode_id[:, t] == episode_id[:, t + H - 1]``, so the loss can
        ignore pairs that straddle a reset.
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
            root_state_future = tensordict["root_state_w"][:, t + H-1]
            rel_kinematics = relative_kinematics(
                root_state_future.unsqueeze(1),
                root_state_ref,
            )
            future_target = torch.cat([rel_kinematics[k] for k in self.keys], dim=-1)
            future_targets.append(future_target)
            # whether t and t+H-1 belong to the same episode
            valid = tensordict["episode_id"][:, t] == tensordict["episode_id"][:, t + H-1]
            valid_masks.append(valid)
        result["_future_target"] = torch.stack(future_targets, dim=1).reshape(N, t_out, self.dim)
        result["_future_valid"] = torch.stack(valid_masks, dim=1)
        return result


class FutureTrajectory:
    def __init__(self, mode: str, horizon: int):
        self.mode = mode
        self.horizon = horizon
        if self.mode not in FUTURE_LABEL_MODES:
            raise ValueError(
                f"Unknown FutureTrajectory mode {self.mode!r}. "
                f"Expected one of {list(FUTURE_LABEL_MODES)}"
            )
        self.dim, self.keys = FUTURE_LABEL_MODES[self.mode]

    def relabel(self, tensordict: TensorDict) -> TensorDict:
        """Slice to ``t_out = T - H + 1`` and add ``_future_target`` / ``_future_valid``.

        The target stacks relative kinematics for each time in ``[t, t + H)``.

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
            valid = tensordict["episode_id"][:, t] == tensordict["episode_id"][:, t + H-1]
            valid_masks.append(valid)
        result["_future_target"] = torch.stack(future_targets, dim=1).reshape(N, t_out, t_out * self.dim)
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

    pos_ref = root_state_ref[:, :3]
    quat_ref = root_state_ref[:, 3:7]
    # linvel_ref = root_state_ref[:, 7:10] # unused
    # angvel_ref = root_state_ref[:, 10:13] # unused

    rel_quat = quat_mul(quat_conjugate(quat_ref).unsqueeze(1), quat)  # (N, T, 4)
    rel_pos = quat_rotate_inverse(rel_quat, pos - pos_ref.unsqueeze(1))  # (N, T, 3)
    rel_yaw = wrap_to_pi(euler_from_quat(rel_quat)[..., 2:3])  # (N, T, 1)
    rel_linvel = quat_rotate_inverse(rel_quat, linvel)  # (N, T, 3)
    rel_angvel = quat_rotate_inverse(rel_quat, angvel)  # (N, T, 3)

    return {
        "rel_quat": rel_quat,
        "rel_pos": rel_pos,
        "rel_yaw": rel_yaw,
        "rel_linvel": rel_linvel,
        "rel_angvel": rel_angvel,
    }
