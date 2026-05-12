# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


UMI_LOCOMOTION6D_Z_COMMAND_INDEX = 3
UMI_LOCOMOTION6D_GRAVITY_COMMAND_INDICES = (4, 5, 6)

# Raw UMI defaults come from:
# - tasks/local_2d_vel.yaml
# - tasks/locomotion6d.yaml
# - combo_go2ARX5_locomotion6d.yaml
# - default_constraints.yaml + constraints/*.yaml
UMI_STAGE1_REWARD_DEFAULT_WEIGHTS = {
    "umi_track_lin_vel_xy_exp": 1.0,
    "umi_track_yaw_exp": 0.5,
    "umi_track_z_height_exp": 0.5,
    "umi_track_gravity_exp": 0.5,
    "umi_action_rate_l2": -0.02,
    "umi_joint_acc_l2": -2.5e-7,
    "umi_joint_power": -1.0e-6,
    "umi_joint_torques_l2": 0.0,
    "umi_joint_pos_limits": -10.0,
    "umi_undesired_contacts": -1.0,
    "umi_feet_drag_penalty": -0.01,
    "umi_feet_air_time": 1.0,
}


def _resolve_body_indices(body_ids) -> list[int]:
    if isinstance(body_ids, slice):
        return []
    if body_ids is None:
        return []
    if isinstance(body_ids, torch.Tensor):
        return body_ids.tolist()
    if isinstance(body_ids, Sequence):
        return list(body_ids)
    return []


def _reference_body_height(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    body_ids = _resolve_body_indices(asset_cfg.body_ids)
    if body_ids:
        return torch.mean(asset.data.body_pos_w[:, body_ids, 2], dim=1)
    return asset.data.root_pos_w[:, 2]


def _reference_body_quat(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    body_ids = _resolve_body_indices(asset_cfg.body_ids)
    if body_ids:
        return asset.data.body_quat_w[:, body_ids[0], :]
    return asset.data.root_quat_w


def umi_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    power: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """UMI locomotion6d xy linear velocity tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.linalg.norm(
        env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2],
        dim=1,
    )
    return torch.exp(-(torch.abs(lin_vel_error) ** power) / tracking_sigma)


def umi_track_yaw_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    power: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """UMI locomotion6d yaw tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_error = torch.abs(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-(yaw_error**power) / tracking_sigma)


def umi_track_z_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    z_height_sigma: float,
    z_command_index: int = UMI_LOCOMOTION6D_Z_COMMAND_INDEX,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """UMI locomotion6d body height tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_height = env.command_manager.get_command(command_name)[:, z_command_index]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        measured_terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        current_height = _reference_body_height(asset, asset_cfg) - measured_terrain_height
    else:
        current_height = _reference_body_height(asset, asset_cfg)
    height_error = torch.square(target_height - current_height)
    return torch.exp(-height_error / z_height_sigma)


def umi_track_gravity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    gravity_sigma: float,
    gravity_command_indices: tuple[int, int, int] = UMI_LOCOMOTION6D_GRAVITY_COMMAND_INDICES,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """UMI locomotion6d gravity direction tracking reward."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_local_gravity = command[:, list(gravity_command_indices)]
    target_local_gravity = torch.nn.functional.normalize(target_local_gravity, dim=1)

    world_gravity = torch.zeros((env.num_envs, 3), device=asset.device)
    world_gravity[:, 2] = -1.0
    current_local_gravity = math_utils.quat_apply_inverse(_reference_body_quat(asset, asset_cfg), world_gravity)

    gravity_error = torch.sum(torch.square(target_local_gravity - current_local_gravity), dim=1)
    return torch.exp(-gravity_error / gravity_sigma)


def umi_action_rate_l2(env: ManagerBasedRLEnv, power: float = 2.0) -> torch.Tensor:
    """UMI action-rate penalty over the unified 18-DoF action vector."""
    delta_action = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.abs(delta_action) ** power, dim=1)


def umi_joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """UMI joint acceleration penalty."""
    return core_mdp.joint_acc_l2(env, asset_cfg=asset_cfg)


def umi_joint_torques_l2(
    env: ManagerBasedRLEnv,
    power: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Optional torque penalty matching UMI torque-limit penalty semantics."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) ** power, dim=1)


def umi_joint_power(
    env: ManagerBasedRLEnv,
    power: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """UMI energy penalty based on mechanical power usage."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_energy = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_energy) ** power, dim=1)


def umi_joint_pos_limits(
    env: ManagerBasedRLEnv,
    penalty_scale: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """UMI joint-limit penalty that starts before the hard joint boundary."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]

    mid = 0.5 * (joint_limits[:, :, 0] + joint_limits[:, :, 1])
    half_range = 0.5 * (joint_limits[:, :, 1] - joint_limits[:, :, 0])
    penalizing_lower = mid - penalty_scale * half_range
    penalizing_upper = mid + penalty_scale * half_range

    out_of_limits = -(joint_pos - penalizing_lower).clip(max=0.0)
    out_of_limits += (joint_pos - penalizing_upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def umi_undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """UMI collision/contact penalty."""
    return core_mdp.undesired_contacts(env, threshold=threshold, sensor_cfg=sensor_cfg)


def umi_feet_drag_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    penalty_feet_drag_height: float,
    feet_drag_sigma: float,
) -> torch.Tensor:
    """UMI feet-drag penalty."""
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height_diff = torch.clamp(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - penalty_feet_drag_height,
        max=0.0,
    )
    feet_planar_speed = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=2)
    penalty_height_scale = -(torch.exp(feet_height_diff / feet_drag_sigma) - 1.0)
    return torch.sum(penalty_height_scale * feet_planar_speed, dim=1)


def umi_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """UMI feet-air-time reward."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
