import torch
import warp as wp

from active_adaptation.envs.mdp import Command, Reward, Termination
from active_adaptation.utils.math import (
    quat_rotate_inverse,
    quat_rotate,
    normalize,
    quat_mul,
    sample_quat_yaw,
)
from active_adaptation.utils.symmetry import SymmetryTransform


class Game(Command):
    """
    A two-agent chaser-evader game for robots to learn locomotion skills.
    """
    def __init__(self, env, catch_radius: float = 0.8) -> None:
        super().__init__(env)
        self.catch_radius = catch_radius

        from active_adaptation.envs.terrain import BetterTerrainImporter, BetterTerrainGenerator
        from active_adaptation.envs.backends.isaac import IsaacSceneAdapter
        from it2.utils import find_flat_patches

        self.scene: IsaacSceneAdapter = self.env.scene
        self.terrain_importer: BetterTerrainImporter = self.scene.terrain
        self.terrain_generator: BetterTerrainGenerator = self.terrain_importer.terrain_generator
        if self.terrain_importer.cfg.terrain_type == "generator":
            self.origins = self.terrain_importer.terrain_origins.reshape(-1, 3)
        else:
            self.origins = self.scene.env_origins

        sub_terrain_size = self.terrain_generator.cfg.size
        half_x = self.terrain_generator.num_rows * sub_terrain_size[0] / 2
        half_y = self.terrain_generator.num_cols * sub_terrain_size[1] / 2
        self.flat_patches, self.all_ray_hits = find_flat_patches(
            wp_mesh=self.env.ground_mesh,
            num_patches=self.terrain_generator.num_rows * self.terrain_generator.num_cols * 4,
            patch_radius=0.5,
            origin=torch.tensor([0.0, 0.0, 0.0], device=self.device),
            x_range=(-half_x, half_x),
            y_range=(-half_y, half_y),
            z_range=(-0.1, 0.1),
            max_height_diff=0.1,
        )
        
        with torch.device(self.device):
            self.role = torch.arange(self.num_envs) % 2
            self.target_caught_time = torch.zeros(self.num_envs, 1)
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.last_distance = torch.zeros(self.num_envs, 1)
            self.distance_change = torch.zeros(self.num_envs, 1)

        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.marker = self.scene.create_arrow_marker(
                prim_path="/Visuals/Command/arrow",
                color=(1.0, 0.0, 0.0),
                scale=(1.0, 0.1, 0.1),
            )
            self.sphere_marker = self.scene.create_sphere_marker(
                prim_path="/Visuals/Command/sphere",
                color=(0.0, 1.0, 0.0),
                radius=0.1,
            )
            self.sphere_marker_1 = self.scene.create_sphere_marker(
                prim_path="/Visuals/Command/sphere_1",
                color=(1.0, 1.0, 0.0),
                radius=0.1,
            )
        self.update()

    @property
    def command(self):
        arange = torch.arange(self.num_envs, device=self.device)
        quat = self.asset.data.root_link_quat_w
        return torch.cat(
            [
                quat_rotate_inverse(quat, self.target_diff),
                quat_rotate_inverse(quat, self.target_lin_vel_w),
                (arange % 2 == 0).reshape(self.num_envs, 1),
                (arange % 2 == 1).reshape(self.num_envs, 1),
            ],
            dim=-1,
        )

    @property
    def command_mode(self):
        return self.role.reshape(self.num_envs, 1)

    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(8), signs=torch.tensor([1, -1, 1, 1, -1, 1, 1, 1])
        )

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        num_envs = len(env_ids)
        chase = env_ids % 2 == 0
        init_root_state = self.init_root_state[env_ids]
        
        idx = torch.randint(
            0, len(self.origins), (num_envs,), device=self.device
        )
        origins = self.origins[idx]

        init_pos = origins[chase]
        angle = torch.rand(len(init_pos), device=self.device) * 2 * torch.pi
        radius = (torch.rand(len(init_pos), device=self.device) + 1.0).unsqueeze(1)
        offset = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1) * radius
        init_root_state[chase, :3] += init_pos + offset
        init_root_state[~chase, :3] += init_pos - offset
        quat = sample_quat_yaw(num_envs, device=self.device)
        init_root_state[:, 3:7] = quat
        return init_root_state

    def reset(self, env_ids: torch.Tensor):
        self.target_caught_time[env_ids] = 0.0
        return super().reset(env_ids)

    def update(self):
        self.target_pos_w = torch.stack(
            [
                self.asset.data.root_pos_w[1::2],
                self.asset.data.root_pos_w[::2],
            ],
            1,
        ).reshape(self.num_envs, 3)
        self.target_lin_vel_w = torch.cat(
            [
                self.asset.data.root_link_lin_vel_w[1::2],
                self.asset.data.root_link_lin_vel_w[::2],
            ],
            1,
        ).reshape(self.num_envs, 3)
        self.target_diff = self.target_pos_w - self.asset.data.root_pos_w
        
        distance = self.target_diff[:, :2].norm(dim=-1, keepdim=True)
        self.distance_change = distance - self.last_distance
        self.last_distance = distance.clone()
        self.distance = distance

        self.target_caught = self.distance < 0.8
        self.target_caught_time = torch.where(
            self.target_caught,
            self.target_caught_time + self.env.step_dt,
            torch.zeros_like(self.target_caught_time),
        )

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_pos_w[::2],
            self.target_diff[::2],
            # self.asset.data.root_pos_w[1::2] - self.asset.data.root_pos_w[::2],
            color=(1, 0, 0, 1),
        )
        self.marker.visualize(
            self.asset.data.root_pos_w[::2]
            + torch.tensor([0.0, 0.0, 0.2], device=self.device),
            self.asset.data.root_link_quat_w[::2],
            scales=torch.tensor([[4.0, 1.0, 0.1]]).expand(self.num_envs // 2, 3),
        )
        self.sphere_marker.visualize(
            self.flat_patches.reshape(-1, 3),
        )
        self.sphere_marker_1.visualize(
            self.all_ray_hits.reshape(-1, 3),
        )


class chase_distance_change(Reward[Game]):
    namespace = "game"

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_chaser = self.command_manager.role[:, None] == 0
        rew = -self.command_manager.distance_change
        return rew.reshape(self.num_envs, 1), is_chaser.reshape(self.num_envs, 1)


class chase_velocity(Reward[Game]):
    namespace = "game"

    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_chaser = self.command_manager.role[:, None] == 0
        direction = normalize(self.command_manager.target_diff[:, :2])
        velocity = self.asset.data.root_link_lin_vel_w[:, :2]
        rew = torch.sum(direction * velocity, dim=1, keepdim=True)
        rew = torch.where(rew > 0, rew.log1p(), rew)
        return rew.reshape(self.num_envs, 1), is_chaser.reshape(self.num_envs, 1)


class evade_velocity(Reward[Game]):
    namespace = "game"

    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_evader = self.command_manager.role[:, None] == 1
        direction = normalize(self.command_manager.target_diff[:, :2])
        velocity = self.asset.data.root_link_lin_vel_w[:, :2]
        # reward moving away from the chaser (negative projection of velocity on diff)
        rew = -torch.sum(direction * velocity, dim=1, keepdim=True)
        rew = torch.where(rew > 0, rew.log1p(), rew)
        return rew.reshape(self.num_envs, 1), is_evader.reshape(self.num_envs, 1)


class evade_distance_change(Reward[Game]):
    namespace = "game"

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_evader = self.command_manager.role[:, None] == 1
        rew = self.command_manager.distance_change
        return rew.reshape(self.num_envs, 1), is_evader.reshape(self.num_envs, 1)


class evade_distance(Reward[Game]):
    namespace = "game"

    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        is_active = torch.arange(self.num_envs, device=self.device) % 2 == 1
        rew = 1 - torch.exp(-self.command_manager.distance * 0.5).reshape(
            self.num_envs, 1
        )
        return rew.reshape(self.num_envs, 1), is_active.reshape(self.num_envs, 1)


class target_in_sight(Reward[Game]):
    namespace = "game"

    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        forward_vec = quat_rotate(
            self.asset.data.root_link_quat_w,
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
        )
        diff = normalize(self.command_manager.target_diff)
        rew = torch.sum(forward_vec[:, :2] * diff[:, :2], dim=1, keepdim=True)
        rew = torch.where(self.command_manager.role[:, None] == 0, rew, -rew)
        return rew.reshape(self.num_envs, 1)


class caught_reward(Reward[Game]):
    namespace = "game"

    def _compute(self) -> torch.Tensor:
        caught = self.command_manager.target_caught.float()
        return torch.where(self.command_manager.role[:, None] == 0, caught, -caught)


class stall_penalty(Reward[Game]):
    namespace = "game"

    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        # only penalize chaser (role == 0)
        is_chaser = self.command_manager.role[:, None] == 0

        dist = self.command_manager.distance  # [num_envs, 1]
        rel_vel_vec = (
            self.command_manager.target_lin_vel_w[:, :2]
            - self.asset.data.root_link_lin_vel_w[:, :2]
        )
        rel_speed = rel_vel_vec.norm(dim=-1, keepdim=True)

        near = dist < self.command_manager.catch_radius * 2.0
        slow = rel_speed < 0.1
        stall = (near & slow & is_chaser).float()

        rew = -stall
        return rew.reshape(self.num_envs, 1), is_chaser.reshape(self.num_envs, 1)


class both_terminate(Termination[Game]):
    """
    Terminate odd envs if even envs terminate, and vice versa.
    """

    namespace = "game"

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        termination = termination.reshape(-1, 2)
        termination = termination | termination.flip(1)
        return termination.reshape(self.num_envs, 1)


class caught_termination(Termination[Game]):
    namespace = "game"

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        return self.command_manager.target_caught_time > 0.1
