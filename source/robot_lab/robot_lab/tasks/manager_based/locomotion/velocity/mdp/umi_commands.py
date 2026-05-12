# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


UMI_DOG_JOINT_NAMES: list[str] = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

UMI_ARM_JOINT_NAMES: list[str] = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

UMI_UNIFIED_JOINT_NAMES: list[str] = UMI_DOG_JOINT_NAMES + UMI_ARM_JOINT_NAMES

UMI_DEFAULT_JOINT_OFFSETS: dict[str, float] = {
    "FR_hip_joint": 0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "FL_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RR_hip_joint": 0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "RL_hip_joint": -0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "arm_joint1": 0.0,
    "arm_joint2": 0.3,
    "arm_joint3": 0.5,
    "arm_joint4": 0.0,
    "arm_joint5": 0.0,
    "arm_joint6": 0.0,
}

# Keep delta semantics and existing Go2-X5-lab stability biases.
UMI_UNIFIED_ACTION_SCALE: dict[str, float] = {
    ".*_hip_joint": 0.125,
    ".*_thigh_joint": 0.25,
    ".*_calf_joint": 0.25,
    "arm_joint1": 0.10,
    "arm_joint2": 0.10,
    "arm_joint3": 0.10,
    "arm_joint4": 0.10,
    "arm_joint5": 0.10,
    "arm_joint6": 0.10,
}

UMI_COMMAND_DIM: int = 7
UMI_ACTION_DIM: int = 18


class UniformUmiLocomotionCommand(mdp.UniformVelocityCommand):
    """UMI locomotion6d command with 7-D task targets.

    The first three dimensions follow Isaac Lab's base velocity command convention:
    `[vx, vy, yaw_rate]` in the base frame. The remaining task targets align with the
    UMI locomotion6d task: `[target_height, gravity_x, gravity_y, gravity_z]`.
    """

    cfg: "UniformUmiLocomotionCommandCfg"  # type: ignore

    def __init__(self, cfg: "UniformUmiLocomotionCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.target_z_height = torch.zeros(self.num_envs, device=self.device)
        self.target_local_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_local_gravity[:, 2] = -1.0

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.vel_command_b, self.target_z_height.unsqueeze(-1), self.target_local_gravity],
            dim=-1,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        lin_speed = torch.linalg.norm(self.vel_command_b[env_ids_tensor, :2], dim=-1)
        active_mask = lin_speed > self.cfg.min_target_lin_vel
        self.vel_command_b[env_ids_tensor, :2] *= active_mask.unsqueeze(-1)

        self.target_z_height[env_ids_tensor] = torch.empty(len(env_ids_tensor), device=self.device).uniform_(
            *self.cfg.ranges.z_height
        )

        roll = torch.empty(len(env_ids_tensor), device=self.device).uniform_(*self.cfg.ranges.roll)
        pitch = torch.empty(len(env_ids_tensor), device=self.device).uniform_(*self.cfg.ranges.pitch)

        gravity = torch.zeros(len(env_ids_tensor), 3, device=self.device)
        gravity[:, 0] = torch.tan(roll)
        gravity[:, 1] = torch.tan(pitch)
        gravity[:, 2] = -1.0
        gravity /= torch.linalg.norm(gravity, dim=-1, keepdim=True)
        self.target_local_gravity[env_ids_tensor] = gravity


@configclass
class UniformUmiLocomotionCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the 7-D UMI locomotion6d command."""

    class_type: type = UniformUmiLocomotionCommand
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    min_target_lin_vel: float = 0.1
    lin_vel_obs_scale: float = 2.0
    ang_vel_obs_scale: float = 0.25

    @configclass
    class Ranges(mdp.UniformVelocityCommandCfg.Ranges):
        z_height: tuple[float, float] = (0.1, 0.4)
        roll: tuple[float, float] = (-0.3, 0.3)
        pitch: tuple[float, float] = (-0.3, 0.3)

    ranges: Ranges = Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.5, 1.5),
        heading=None,
        z_height=(0.1, 0.4),
        roll=(-0.3, 0.3),
        pitch=(-0.3, 0.3),
    )


@configclass
class UmiLocomotion6dCommandsCfg:
    """Command group for UMI locomotion6d environments."""

    locomotion6d = UniformUmiLocomotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        rel_standing_envs=0.0,
        debug_vis=False,
    )
    base_velocity = None
    arm_joint_pos = None


def build_umi_joint_position_action_kwargs(asset_name: str = "robot") -> dict[str, object]:
    """Helper kwargs for `JointPositionActionCfg` using unified 18-DoF delta control."""

    return {
        "asset_name": asset_name,
        "joint_names": copy.deepcopy(UMI_UNIFIED_JOINT_NAMES),
        "scale": copy.deepcopy(UMI_UNIFIED_ACTION_SCALE),
        "use_default_offset": True,
        "clip": None,
        "preserve_order": True,
    }


def build_umi_phase1_randomization_specs(asset_name: str = "robot") -> dict[str, dict[str, object]]:
    """Return env-cfg-side event specs for the phase-1 UMI migration.

    These objects are intended to be wrapped by `EventTermCfg` inside the env cfg. The
    phase-1 critic uses only a subset of the randomized setup (`kp`, `kd`, base mass, base
    COM) to stay aligned with the 110-D privileged observation target.
    """

    return {
        "material_friction": {
            "func": mdp.randomize_rigid_body_material,
            "mode": "startup",
            "params": {
                "asset_cfg": SceneEntityCfg(asset_name, body_names=".*"),
                "static_friction_range": (0.2, 2.0),
                "dynamic_friction_range": (0.2, 2.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        },
        "base_mass": {
            "func": mdp.randomize_rigid_body_mass,
            "mode": "startup",
            "params": {
                "asset_cfg": SceneEntityCfg(asset_name, body_names=["base"], preserve_order=True),
                "mass_distribution_params": (-0.5, 0.5),
                "operation": "add",
                "recompute_inertia": True,
            },
        },
        "base_com": {
            "func": mdp.randomize_rigid_body_com,
            "mode": "startup",
            "params": {
                "asset_cfg": SceneEntityCfg(asset_name, body_names=["base"], preserve_order=True),
                "com_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.1, 0.1),
                },
            },
        },
        "actuator_gains": {
            "func": mdp.randomize_actuator_gains,
            "mode": "reset",
            "params": {
                "asset_cfg": SceneEntityCfg(asset_name, joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True),
                "stiffness_distribution_params": (0.5, 1.5),
                "damping_distribution_params": (0.5, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        },
    }
