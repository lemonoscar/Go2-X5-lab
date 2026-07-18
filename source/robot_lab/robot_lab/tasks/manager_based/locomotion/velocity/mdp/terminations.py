# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def bad_orientation_after_steps(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    minimum_episode_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate excessive tilt after a short reset-recovery window."""
    asset: RigidObject = env.scene[asset_cfg.name]
    projected_up = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    bad_orientation = torch.acos(projected_up).abs() > limit_angle
    settled = env.episode_length_buf >= minimum_episode_steps
    return bad_orientation & settled


def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


def root_lin_vel_z_above_maximum(
    env: ManagerBasedRLEnv, maximum_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the absolute root vertical velocity is too large."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_lin_vel_b[:, 2]) > maximum_speed


def root_ang_vel_xy_above_maximum(
    env: ManagerBasedRLEnv, maximum_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the root roll/pitch angular speed is too large."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1) > maximum_speed


def pct_stair_path_completed(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_force_threshold: float,
    maximum_root_ang_vel_xy: float,
    maximum_root_lin_vel_z: float,
    minimum_upright_projection: float = 0.94,
    maximum_cross_track_error: float = 0.45,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """End a PCT episode as a success timeout after stable path completion."""
    command_term = env.command_manager.get_term(command_name)
    completed = command_term.path_completed
    upright = -env.scene["robot"].data.projected_gravity_b[:, 2] >= minimum_upright_projection
    inside_corridor = torch.abs(command_term.cross_track_error_m) <= maximum_cross_track_error
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_safe = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2)) < contact_force_threshold
    dynamically_stable = torch.linalg.norm(env.scene["robot"].data.root_ang_vel_b[:, :2], dim=1)
    dynamically_stable = dynamically_stable <= maximum_root_ang_vel_xy
    dynamically_stable &= torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 2]) <= maximum_root_lin_vel_z
    return completed & upright & inside_corridor & contact_safe & dynamically_stable


def pct_stair_path_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    maximum_cross_track_error: float = 0.45,
) -> torch.Tensor:
    """Terminate robots that leave the PCT stair corridor."""
    cross_track_error = env.command_manager.get_term(command_name).cross_track_error_m
    return torch.abs(cross_track_error) > maximum_cross_track_error


def pct_stair_nonfoot_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimum_episode_steps: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate meaningful non-foot contacts after the reset settling window."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    maximum_force = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))
    settled = env.episode_length_buf >= minimum_episode_steps
    return torch.logical_and(maximum_force > threshold, settled)
