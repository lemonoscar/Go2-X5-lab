# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _vx_tracking_tolerance(
    command_x: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> torch.Tensor:
    """Return the user-facing vx tolerance with an absolute low-speed floor."""
    return torch.maximum(
        torch.full_like(command_x, absolute_tolerance),
        torch.abs(command_x) * relative_tolerance,
    )


def track_vx_tolerance_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    absolute_tolerance: float = 0.1,
    relative_tolerance: float = 0.1,
    outside_tolerance_std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward vx tracking with a flat optimum inside the accepted error band."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    tolerance = _vx_tracking_tolerance(command_x, absolute_tolerance, relative_tolerance)
    excess_error = torch.clamp(torch.abs(command_x - asset.data.root_lin_vel_b[:, 0]) - tolerance, min=0.0)
    reward = torch.exp(-torch.square(excess_error / max(outside_tolerance_std, 1.0e-6)))
    return reward * _reward_upright_scale(env)


def vx_tracking_excess_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    absolute_tolerance: float = 0.1,
    relative_tolerance: float = 0.1,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only the vx error that exceeds the accepted error band."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    tolerance = _vx_tracking_tolerance(command_x, absolute_tolerance, relative_tolerance)
    excess_error = torch.clamp(torch.abs(command_x - asset.data.root_lin_vel_b[:, 0]) - tolerance, min=0.0)
    return torch.clamp(excess_error / torch.clamp(tolerance, min=1.0e-6), max=max_penalty)


def uncommanded_velocity_excess_l1(
    env: ManagerBasedRLEnv,
    lateral_tolerance: float = 0.1,
    yaw_tolerance: float = 0.1,
    max_penalty: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral and yaw drift only outside their accepted bands."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lateral_excess = torch.clamp(
        torch.abs(asset.data.root_lin_vel_b[:, 1]) - lateral_tolerance,
        min=0.0,
    ) / max(lateral_tolerance, 1.0e-6)
    yaw_excess = torch.clamp(
        torch.abs(asset.data.root_ang_vel_b[:, 2]) - yaw_tolerance,
        min=0.0,
    ) / max(yaw_tolerance, 1.0e-6)
    return torch.clamp(lateral_excess + yaw_excess, max=max_penalty)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _command_direction_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed planar speed along the commanded direction and command magnitude."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_magnitude = torch.linalg.norm(command_xy, dim=1)
    command_direction = command_xy / torch.clamp(command_magnitude, min=1.0e-6).unsqueeze(1)
    signed_speed = torch.sum(asset.data.root_lin_vel_b[:, :2] * command_direction, dim=1)
    return signed_speed, command_magnitude


def command_direction_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed planar speed along a non-zero velocity command."""
    signed_speed, command_magnitude = _command_direction_speed(env, command_name, asset_cfg)
    active_command = (command_magnitude > command_threshold).to(signed_speed.dtype)
    return signed_speed * active_command * _reward_upright_scale(env)


def commanded_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    min_progress_speed: float,
    max_penalty: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low or reverse progress while a planar velocity command is active."""
    signed_speed, command_magnitude = _command_direction_speed(env, command_name, asset_cfg)
    normalizer = max(float(min_progress_speed), 1.0e-6)
    deficit = torch.clamp((min_progress_speed - signed_speed) / normalizer, min=0.0, max=max_penalty)
    active_command = (command_magnitude > command_threshold).to(signed_speed.dtype)
    return deficit * active_command * _reward_upright_scale(env)


class PctStairPathProgressReward(ManagerTermBase):
    """Reward signed centerline progress only while height, contact, and posture remain valid."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.nonfoot_body_ids = cfg.params["sensor_cfg"].body_ids
        self.previous_progress = torch.zeros(self.num_envs, device=self.device)
        self.just_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_progress[env_ids] = 0.0
        self.just_reset[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        height_tracking_std: float,
        contact_force_threshold: float,
        maximum_progress_speed: float,
        maximum_regress_speed: float,
        regression_scale: float,
        sensor_cfg: SceneEntityCfg,
        minimum_upright_projection: float = 0.0,
    ) -> torch.Tensor:
        del sensor_cfg
        command_term = env.command_manager.get_term(command_name)
        current_progress = command_term.path_progress_m
        progress_speed = (current_progress - self.previous_progress) / max(env.step_dt, 1.0e-6)
        self.previous_progress.copy_(current_progress)
        progress_speed = torch.where(self.just_reset, torch.zeros_like(progress_speed), progress_speed)
        self.just_reset.fill_(False)
        progress_speed = torch.clamp(
            progress_speed,
            min=-maximum_regress_speed,
            max=maximum_progress_speed,
        )

        height_error = command_term.height_gain_m - command_term.expected_height_gain_m
        height_gate = torch.exp(-torch.square(height_error) / max(height_tracking_std**2, 1.0e-6))
        nonfoot_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.nonfoot_body_ids, :]
        maximum_nonfoot_force = torch.linalg.norm(nonfoot_forces, dim=-1).amax(dim=(1, 2))
        contact_gate = (maximum_nonfoot_force < contact_force_threshold).to(torch.float32)
        upright_gate = _reward_upright_scale(env)
        if minimum_upright_projection > 0.0:
            upright_projection = -env.scene["robot"].data.projected_gravity_b[:, 2]
            upright_gate *= (upright_projection >= minimum_upright_projection).to(torch.float32)
        safe_progress_gate = height_gate * contact_gate * upright_gate

        return torch.where(
            progress_speed >= 0.0,
            progress_speed * safe_progress_gate,
            regression_scale * progress_speed,
        )


class PctRearFootSupportReward(ManagerTermBase):
    """Reward forward motion supported by rear feet that have caught up on the stairs."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset_foot_ids = cfg.params["asset_cfg"].body_ids
        self.sensor_foot_ids = cfg.params["sensor_cfg"].body_ids

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        maximum_progress_lag: float,
        progress_lag_std: float,
        maximum_height_lag: float,
        height_lag_std: float,
        activation_progress: float,
        activation_width: float,
        contact_force_threshold: float,
        command_threshold: float,
        target_speed: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del sensor_cfg, asset_cfg
        command_term = env.command_manager.get_term(command_name)

        path_vector = command_term.path_points_xy[-1] - command_term.path_points_xy[0]
        path_tangent = path_vector / torch.clamp(torch.linalg.norm(path_vector), min=1.0e-6)
        rear_foot_xy = (
            self.robot.data.body_pos_w[:, self.asset_foot_ids, :2]
            - env.scene.env_origins[:, None, :2]
        )
        rear_foot_progress = torch.sum(
            (rear_foot_xy - command_term.path_points_xy[0]) * path_tangent,
            dim=2,
        )
        progress_lag = command_term.path_progress_m[:, None] - rear_foot_progress
        progress_excess = torch.clamp(progress_lag - maximum_progress_lag, min=0.0)
        progress_quality = torch.exp(
            -torch.square(progress_excess) / max(progress_lag_std**2, 1.0e-6)
        )

        rear_foot_height_gain = (
            self.robot.data.body_pos_w[:, self.asset_foot_ids, 2]
            - env.scene.env_origins[:, None, 2]
        )
        minimum_support_height = command_term.expected_height_gain_m[:, None] - maximum_height_lag
        height_deficit = torch.clamp(minimum_support_height - rear_foot_height_gain, min=0.0)
        height_quality = torch.exp(
            -torch.square(height_deficit) / max(height_lag_std**2, 1.0e-6)
        )

        rear_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_foot_ids, :]
        rear_contact = (
            torch.linalg.norm(rear_forces, dim=-1).amax(dim=1) > contact_force_threshold
        ).to(torch.float32)
        support_quality = torch.mean(progress_quality * height_quality * rear_contact, dim=1)

        signed_speed, command_magnitude = _command_direction_speed(
            env, command_name, SceneEntityCfg("robot")
        )
        speed_gate = torch.clamp(signed_speed / max(target_speed, 1.0e-6), min=0.0, max=1.0)
        command_gate = (command_magnitude > command_threshold).to(torch.float32)
        activation_gate = torch.clamp(
            (command_term.path_progress_m - activation_progress) / max(activation_width, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        return support_quality * speed_gate * command_gate * activation_gate * _reward_upright_scale(env)


def pct_stair_height_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_tracking_std: float,
    command_threshold: float,
    target_speed: float,
) -> torch.Tensor:
    """Reward forward motion when root height follows the active PCT path segment."""
    command_term = env.command_manager.get_term(command_name)
    height_error = command_term.height_gain_m - command_term.expected_height_gain_m
    height_alignment = torch.exp(-torch.square(height_error) / max(height_tracking_std**2, 1.0e-6))
    signed_speed, command_magnitude = _command_direction_speed(
        env, command_name, SceneEntityCfg("robot")
    )
    speed_gate = torch.clamp(signed_speed / max(target_speed, 1.0e-6), min=0.0, max=1.0)
    active_command = (command_magnitude > command_threshold).to(torch.float32)
    return height_alignment * speed_gate * active_command * _reward_upright_scale(env)


def pct_stair_base_clearance_deficit(
    env: ManagerBasedRLEnv,
    command_name: str,
    clearance_margin: float,
    maximum_deficit: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the root falling below a small clearance margin above the PCT path."""
    command_term = env.command_manager.get_term(command_name)
    desired_height_gain = command_term.expected_height_gain_m + clearance_margin
    deficit = torch.clamp(
        (desired_height_gain - command_term.height_gain_m) / max(maximum_deficit, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    command_magnitude = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    active_command = (command_magnitude > command_threshold).to(torch.float32)
    return deficit * active_command * _reward_upright_scale(env)


def pct_stair_cross_track_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    cross_track_std: float,
    command_threshold: float,
    target_speed: float,
) -> torch.Tensor:
    """Reward forward motion close to the PCT centerline."""
    command_term = env.command_manager.get_term(command_name)
    centerline_alignment = torch.exp(
        -torch.square(command_term.cross_track_error_m) / max(cross_track_std**2, 1.0e-6)
    )
    signed_speed, command_magnitude = _command_direction_speed(
        env, command_name, SceneEntityCfg("robot")
    )
    speed_gate = torch.clamp(signed_speed / max(target_speed, 1.0e-6), min=0.0, max=1.0)
    active_command = (command_magnitude > command_threshold).to(torch.float32)
    return centerline_alignment * speed_gate * active_command * _reward_upright_scale(env)


def pct_stair_completion_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_force_threshold: float,
    maximum_root_ang_vel_xy: float,
    maximum_root_lin_vel_z: float,
    minimum_upright_projection: float,
    maximum_cross_track_error: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward completion only when the same step also satisfies the safety gates."""
    command_term = env.command_manager.get_term(command_name)
    upright = -env.scene["robot"].data.projected_gravity_b[:, 2] >= minimum_upright_projection
    inside_corridor = torch.abs(command_term.cross_track_error_m) <= maximum_cross_track_error
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_safe = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2)) < contact_force_threshold
    robot = env.scene["robot"]
    dynamically_stable = torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1) <= maximum_root_ang_vel_xy
    dynamically_stable &= torch.abs(robot.data.root_lin_vel_b[:, 2]) <= maximum_root_lin_vel_z
    safe_completion = command_term.path_completed & upright & inside_corridor & contact_safe & dynamically_stable
    return safe_completion.to(torch.float32)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def _reward_upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _motion_gate(
    env: ManagerBasedRLEnv,
    asset: Articulation | RigidObject,
    command_name: str,
    command_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    return torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold)


def _crawl_phase(
    env: ManagerBasedRLEnv,
    foot_count: int,
    cycle_time: float,
    swing_start_fraction: float,
    swing_end_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt / max(cycle_time, 1e-6), 1.0)
    phase_scaled = phase * foot_count
    swing_index = torch.clamp(phase_scaled.to(torch.long), max=foot_count - 1)
    phase_progress = torch.remainder(phase_scaled, 1.0)
    swing_active = (phase_progress >= swing_start_fraction) & (phase_progress <= swing_end_fraction)
    return swing_index, swing_active, phase_progress


def _ordered_body_ids(entity, foot_names: list[str]) -> list[int]:
    body_ids = []
    for foot_name in foot_names:
        ids = entity.find_bodies(foot_name)[0]
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one body for foot name {foot_name!r}, got {ids}.")
        body_ids.append(ids[0])
    return body_ids


class CrawlGaitReward(ManagerTermBase):
    """Reward a static crawl gait: one swing foot, three stance feet."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.foot_names = list(cfg.params["foot_names"])
        self.foot_ids = _ordered_body_ids(self.contact_sensor, self.foot_names)
        self.foot_count = len(self.foot_ids)
        if self.foot_count != 4:
            raise ValueError("CrawlGaitReward expects exactly four feet.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_names: list[str],
        cycle_time: float,
        swing_start_fraction: float,
        swing_end_fraction: float,
        command_threshold: float,
        velocity_threshold: float,
        contact_force_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del foot_names, sensor_cfg, asset_cfg
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_ids, :]
        contact = contact_forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold
        swing_index, swing_active, _ = _crawl_phase(
            env, self.foot_count, cycle_time, swing_start_fraction, swing_end_fraction
        )

        foot_range = torch.arange(self.foot_count, device=env.device).unsqueeze(0)
        expected_air = (foot_range == swing_index.unsqueeze(1)) & swing_active.unsqueeze(1)
        expected_contact = ~expected_air

        contact_match = torch.where(expected_contact, contact, ~contact).to(torch.float32).mean(dim=1)
        stance_count = torch.sum(contact & expected_contact, dim=1)
        air_count = torch.sum(~contact, dim=1)
        stance_ok = stance_count >= 3
        swing_ok = torch.where(swing_active, air_count == 1, air_count <= 1)

        reward = contact_match * stance_ok.to(torch.float32) * swing_ok.to(torch.float32)
        reward *= _motion_gate(env, self.asset, command_name, command_threshold, velocity_threshold)
        reward *= _reward_upright_scale(env)
        return reward


class CrawlSupportPolygonReward(ManagerTermBase):
    """Reward the root projection staying inside the three-foot support triangle."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.foot_names = list(cfg.params["foot_names"])
        self.sensor_foot_ids = _ordered_body_ids(self.contact_sensor, self.foot_names)
        self.asset_foot_ids = _ordered_body_ids(self.asset, self.foot_names)
        self.foot_count = len(self.asset_foot_ids)
        if self.foot_count != 4:
            raise ValueError("CrawlSupportPolygonReward expects exactly four feet.")

    @staticmethod
    def _triangle_margin(point: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
        a = triangle[:, 0, :]
        b = triangle[:, 1, :]
        c = triangle[:, 2, :]
        v0 = c - a
        v1 = b - a
        v2 = point - a
        dot00 = torch.sum(v0 * v0, dim=1)
        dot01 = torch.sum(v0 * v1, dim=1)
        dot02 = torch.sum(v0 * v2, dim=1)
        dot11 = torch.sum(v1 * v1, dim=1)
        dot12 = torch.sum(v1 * v2, dim=1)
        denom = torch.clamp(dot00 * dot11 - dot01 * dot01, min=1e-6)
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        return torch.minimum(torch.minimum(u, v), 1.0 - u - v)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_names: list[str],
        cycle_time: float,
        swing_start_fraction: float,
        swing_end_fraction: float,
        command_threshold: float,
        velocity_threshold: float,
        contact_force_threshold: float,
        margin_scale: float,
        outside_penalty_scale: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del foot_names, sensor_cfg, asset_cfg
        foot_xy = self.asset.data.body_pos_w[:, self.asset_foot_ids, :2]
        point = self.asset.data.root_pos_w[:, :2]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_foot_ids, :]
        contact = contact_forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold
        swing_index, swing_active, _ = _crawl_phase(
            env, self.foot_count, cycle_time, swing_start_fraction, swing_end_fraction
        )

        per_swing_reward = []
        per_swing_contact = []
        for swing_foot in range(self.foot_count):
            stance_ids = [idx for idx in range(self.foot_count) if idx != swing_foot]
            margin = self._triangle_margin(point, foot_xy[:, stance_ids, :])
            inside_score = torch.clamp(margin / max(margin_scale, 1e-6), min=0.0, max=1.0)
            outside_score = -outside_penalty_scale * torch.clamp(-margin / max(margin_scale, 1e-6), min=0.0, max=1.0)
            per_swing_reward.append(torch.where(margin >= 0.0, inside_score, outside_score))
            per_swing_contact.append(contact[:, stance_ids].to(torch.float32).mean(dim=1))

        reward_by_swing = torch.stack(per_swing_reward, dim=1)
        contact_by_swing = torch.stack(per_swing_contact, dim=1)
        gather_ids = swing_index.unsqueeze(1)
        reward = torch.gather(reward_by_swing, 1, gather_ids).squeeze(1)
        stance_contact = torch.gather(contact_by_swing, 1, gather_ids).squeeze(1)

        reward *= stance_contact
        reward *= swing_active.to(torch.float32)
        reward *= _motion_gate(env, self.asset, command_name, command_threshold, velocity_threshold)
        reward *= _reward_upright_scale(env)
        return reward


class CrawlSwingClearanceReward(ManagerTermBase):
    """Reward the scheduled swing foot clearing the ground instead of dragging."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.foot_names = list(cfg.params["foot_names"])
        self.sensor_foot_ids = _ordered_body_ids(self.contact_sensor, self.foot_names)
        self.asset_foot_ids = _ordered_body_ids(self.asset, self.foot_names)
        self.foot_count = len(self.asset_foot_ids)
        if self.foot_count != 4:
            raise ValueError("CrawlSwingClearanceReward expects exactly four feet.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_names: list[str],
        cycle_time: float,
        swing_start_fraction: float,
        swing_end_fraction: float,
        target_clearance: float,
        clearance_std: float,
        command_threshold: float,
        velocity_threshold: float,
        contact_force_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del foot_names, sensor_cfg, asset_cfg
        swing_index, swing_active, _ = _crawl_phase(
            env, self.foot_count, cycle_time, swing_start_fraction, swing_end_fraction
        )
        foot_z = self.asset.data.body_pos_w[:, self.asset_foot_ids, 2]
        swing_z = torch.gather(foot_z, 1, swing_index.unsqueeze(1)).squeeze(1)
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_foot_ids, :]
        contact = contact_forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold
        swing_contact = torch.gather(contact, 1, swing_index.unsqueeze(1)).squeeze(1)

        clearance_error = torch.square(swing_z - target_clearance)
        reward = torch.exp(-clearance_error / max(clearance_std**2, 1e-6))
        reward *= (~swing_contact).to(torch.float32)
        reward *= swing_active.to(torch.float32)
        reward *= _motion_gate(env, self.asset, command_name, command_threshold, velocity_threshold)
        reward *= _reward_upright_scale(env)
        return reward


class CrawlStrideLengthReward(ManagerTermBase):
    """Reward forward touchdown displacement for the scheduled crawl swing foot."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.foot_names = list(cfg.params["foot_names"])
        self.sensor_foot_ids = _ordered_body_ids(self.contact_sensor, self.foot_names)
        self.asset_foot_ids = _ordered_body_ids(self.asset, self.foot_names)
        self.foot_count = len(self.asset_foot_ids)
        if self.foot_count != 4:
            raise ValueError("CrawlStrideLengthReward expects exactly four feet.")
        self._liftoff_pos_w = torch.zeros((env.num_envs, self.foot_count, 2), device=env.device)
        self._was_contact = torch.zeros((env.num_envs, self.foot_count), dtype=torch.bool, device=env.device)
        self._initialized = False

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._liftoff_pos_w[env_ids] = 0.0
        self._was_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_names: list[str],
        cycle_time: float,
        swing_start_fraction: float,
        swing_end_fraction: float,
        target_stride: float,
        stride_std: float,
        short_stride_penalty_scale: float,
        command_threshold: float,
        velocity_threshold: float,
        contact_force_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del foot_names, sensor_cfg, asset_cfg
        foot_xy = self.asset.data.body_pos_w[:, self.asset_foot_ids, :2]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_foot_ids, :]
        contact = contact_forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold

        if not self._initialized:
            self._liftoff_pos_w.copy_(foot_xy)
            self._was_contact.copy_(contact)
            self._initialized = True

        episode_start = env.episode_length_buf <= 1
        if torch.any(episode_start):
            self._liftoff_pos_w[episode_start] = foot_xy[episode_start]
            self._was_contact[episode_start] = contact[episode_start]

        first_air = (~contact) & self._was_contact
        if torch.any(first_air):
            self._liftoff_pos_w[first_air] = foot_xy[first_air]

        first_contact = contact & ~self._was_contact
        swing_index, _, _ = _crawl_phase(
            env, self.foot_count, cycle_time, swing_start_fraction, swing_end_fraction
        )
        foot_range = torch.arange(self.foot_count, device=env.device).unsqueeze(0)
        expected_touchdown = first_contact & (foot_range == swing_index.unsqueeze(1))

        stride_w = foot_xy - self._liftoff_pos_w
        yaw = math_utils.euler_xyz_from_quat(self.asset.data.root_quat_w)[2]
        stride_x_b = torch.cos(yaw).unsqueeze(1) * stride_w[:, :, 0] + torch.sin(yaw).unsqueeze(1) * stride_w[:, :, 1]
        stride_x = torch.sum(torch.clamp(stride_x_b, min=0.0) * expected_touchdown.to(torch.float32), dim=1)

        stride_score = torch.exp(-torch.square(stride_x - target_stride) / max(stride_std**2, 1e-6))
        short_penalty = short_stride_penalty_scale * torch.clamp((target_stride - stride_x) / max(target_stride, 1e-6), 0.0, 1.0)
        touched = torch.any(expected_touchdown, dim=1)
        reward = torch.where(touched, stride_score - short_penalty, torch.zeros_like(stride_score))

        self._was_contact.copy_(contact)
        reward *= _motion_gate(env, self.asset, command_name, command_threshold, velocity_threshold)
        reward *= _reward_upright_scale(env)
        return reward


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_drag_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    penalty_feet_drag_height: float,
    feet_drag_sigma: float,
) -> torch.Tensor:
    """Penalize low-clearance feet that keep moving in the plane.

    This mirrors the UMI-on-Legs feet-drag penalty and replaces the need to aggressively
    hard-code a high body-frame swing-foot target.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height_diff = torch.clamp(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - penalty_feet_drag_height,
        max=0.0,
    )
    feet_planar_speed = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=2)
    penalty_height_scale = -(torch.exp(feet_height_diff / feet_drag_sigma) - 1.0)
    reward = torch.sum(penalty_height_scale * feet_planar_speed, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ========== Arm-related reward functions ==========


def arm_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize arm joint velocities using L2 squared kernel.

    This reward helps to suppress rapid arm movements that could destabilize the robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return reward


def arm_joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize arm joint accelerations using L2 squared kernel.

    This reward helps to ensure smooth arm movements by penalizing sudden changes in velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute acceleration as change in velocity over time
    # Note: This requires storing previous velocity, using joint_acc if available
    reward = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return reward


def arm_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize arm joint torques using L2 squared kernel.

    This reward helps to reduce energy consumption and prevent excessive forces on the arm joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward


def arm_action_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize arm action rate using L2 squared kernel.

    This reward ensures smooth control commands by penalizing rapid changes in arm actions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get the arm joint indices
    arm_joint_ids = asset_cfg.joint_ids
    # Compute the action rate for arm joints only
    action_diff = env.action_manager.action[:, arm_joint_ids] - env.action_manager.prev_action[:, arm_joint_ids]
    reward = torch.sum(torch.square(action_diff), dim=1)
    return reward


def arm_joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize arm joint positions approaching their limits.

    This reward helps to prevent the arm from reaching extreme positions that could
    cause instability or damage.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get current joint positions for arm
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # Get joint limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    # Compute distance to limits
    out_of_limits = -(joint_pos - joint_pos_limits[:, :, 0]).clip(max=0.0)  # lower limit violation
    out_of_limits += (joint_pos - joint_pos_limits[:, :, 1]).clip(min=0.0)  # upper limit violation
    reward = torch.sum(out_of_limits, dim=1)
    return reward


def arm_joint_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize arm joint positions deviating from their default positions.

    This reward encourages the arm to stay near its default (folded) position,
    helping to maintain stability during locomotion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute deviation from default position
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(deviation), dim=1)
    return reward


def arm_joint_pos_tracking_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize deviation between arm joint positions and commanded targets."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos = env.command_manager.get_command(command_name)
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pos
    reward = torch.sum(torch.square(error), dim=1)
    return reward


def arm_motion_tilt_penalty(
    env: ManagerBasedRLEnv,
    base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_clip: float | None = 1.0,
    vel_clip: float | None = None,
) -> torch.Tensor:
    """Penalize base tilt that co-occurs with arm motion.

    This term couples the arm swing speed with the base roll/pitch error so that fast arm motions
    are discouraged when they inject large disturbances into the body.
    """
    base_asset: RigidObject = env.scene[base_asset_cfg.name]
    arm_asset: Articulation = env.scene[arm_asset_cfg.name]
    base_tilt = torch.linalg.norm(base_asset.data.projected_gravity_b[:, :2], dim=1)
    if tilt_clip is not None:
        base_tilt = torch.clamp(base_tilt, max=tilt_clip)
    arm_speed = torch.linalg.norm(arm_asset.data.joint_vel[:, arm_asset_cfg.joint_ids], dim=1)
    if vel_clip is not None:
        arm_speed = torch.clamp(arm_speed, max=vel_clip)
    reward = base_tilt * arm_speed
    return reward


def arm_pose_conditioned_base_stability(
    env: ManagerBasedRLEnv,
    arm_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_clip: float | None = None,
    speed_clip: float | None = None,
    pose_weight: float = 1.0,
    speed_weight: float = 0.35,
    tilt_weight: float = 1.0,
    ang_vel_weight: float = 0.35,
    lin_vel_weight: float = 0.20,
) -> torch.Tensor:
    """Penalize base instability more strongly when the arm is far from its default pose.

    This term models the intuition that a heavy arm at a large offset creates a harder
    stabilization problem than a folded arm near the default posture.
    """
    arm_asset: Articulation = env.scene[arm_asset_cfg.name]
    base_asset: RigidObject = env.scene[base_asset_cfg.name]

    arm_offset = (
        arm_asset.data.joint_pos[:, arm_asset_cfg.joint_ids] - arm_asset.data.default_joint_pos[:, arm_asset_cfg.joint_ids]
    )
    arm_pose_mag = torch.linalg.norm(arm_offset, dim=1)
    arm_speed = torch.linalg.norm(arm_asset.data.joint_vel[:, arm_asset_cfg.joint_ids], dim=1)
    if pose_clip is not None:
        arm_pose_mag = torch.clamp(arm_pose_mag, max=pose_clip)
    if speed_clip is not None:
        arm_speed = torch.clamp(arm_speed, max=speed_clip)

    arm_difficulty = pose_weight * arm_pose_mag + speed_weight * arm_speed
    base_tilt = torch.linalg.norm(base_asset.data.projected_gravity_b[:, :2], dim=1)
    ang_vel_xy = torch.linalg.norm(base_asset.data.root_ang_vel_b[:, :2], dim=1)
    lin_vel_z = torch.abs(base_asset.data.root_lin_vel_b[:, 2])
    base_instability = tilt_weight * base_tilt + ang_vel_weight * ang_vel_xy + lin_vel_weight * lin_vel_z
    return arm_difficulty * base_instability


def arm_stable_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    arm_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tracking_std: float = 0.1,
    tilt_std: float = 0.2,
    vel_z_std: float = 0.25,
    command_scale: float = 0.5,
) -> torch.Tensor:
    """Reward accurate arm tracking that happens while the base remains stable.

    The reward is factored into:
    - Exponential tracking bonus on the joint position error.
    - Exponential bonuses on base roll/pitch (projected gravity) and vertical velocity.
    - A gate on commanded arm motion magnitude to avoid rewarding idle arms.
    """
    arm_asset: Articulation = env.scene[arm_asset_cfg.name]
    base_asset: RigidObject = env.scene[base_asset_cfg.name]
    safe_tracking_std = max(tracking_std, 1e-6)
    safe_tilt_std = max(tilt_std, 1e-6)
    safe_vel_std = max(vel_z_std, 1e-6)
    safe_command_scale = max(command_scale, 1e-3)

    target_pos = env.command_manager.get_command(command_name)
    track_error = torch.sum(torch.square(arm_asset.data.joint_pos[:, arm_asset_cfg.joint_ids] - target_pos), dim=1)
    tracking_term = torch.exp(-track_error / (safe_tracking_std**2))

    base_tilt = torch.linalg.norm(base_asset.data.projected_gravity_b[:, :2], dim=1)
    tilt_term = torch.exp(-torch.square(base_tilt) / (safe_tilt_std**2))
    vertical_term = torch.exp(-torch.square(base_asset.data.root_lin_vel_b[:, 2]) / (safe_vel_std**2))

    command_mag = torch.linalg.norm(target_pos, dim=1)
    command_gate = torch.tanh(command_mag / safe_command_scale)

    reward = tracking_term * tilt_term * vertical_term * command_gate
    return reward


class ZeroCmdXYPositionDriftUnderArmMotion(ManagerTermBase):
    """Penalize planar drift relative to an anchor during zero-base-command arm motion."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._xy_anchor = torch.zeros((env.num_envs, 2), device=env.device)
        self._prev_arm_command = torch.zeros((env.num_envs, 0), device=env.device)
        self._prev_zero_cmd_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = False

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._xy_anchor[env_ids] = 0.0
        self._prev_zero_cmd_mask[env_ids] = False
        if self._prev_arm_command.shape[1] > 0:
            self._prev_arm_command[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        arm_command_name: str,
        base_asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.08,
        arm_command_change_threshold: float = 0.05,
        arm_pose_weight: float = 0.6,
        arm_speed_weight: float = 0.4,
    ) -> torch.Tensor:
        base_asset: RigidObject = env.scene[base_asset_cfg.name]
        base_cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
        zero_cmd_mask = base_cmd < command_threshold

        arm_command = env.command_manager.get_command(arm_command_name)
        if not self._initialized or self._prev_arm_command.shape[1] != arm_command.shape[1]:
            self._prev_arm_command = torch.zeros_like(arm_command)
            self._initialized = True

        arm_delta = torch.linalg.norm(arm_command - self._prev_arm_command, dim=1)
        entered_zero_cmd = zero_cmd_mask & ~self._prev_zero_cmd_mask
        episode_start = env.episode_length_buf <= 1
        refresh_anchor = episode_start | entered_zero_cmd | (zero_cmd_mask & (arm_delta > arm_command_change_threshold))
        if torch.any(refresh_anchor):
            self._xy_anchor[refresh_anchor] = base_asset.data.root_pos_w[refresh_anchor, :2]

        arm_asset_cfg = self.cfg.params["arm_asset_cfg"]
        arm_asset: Articulation = env.scene[arm_asset_cfg.name]
        arm_offset = (
            arm_asset.data.joint_pos[:, arm_asset_cfg.joint_ids] - arm_asset.data.default_joint_pos[:, arm_asset_cfg.joint_ids]
        )
        arm_pose_mag = torch.linalg.norm(arm_offset, dim=1)
        arm_speed = torch.linalg.norm(arm_asset.data.joint_vel[:, arm_asset_cfg.joint_ids], dim=1)
        arm_activity = arm_pose_weight * arm_pose_mag + arm_speed_weight * arm_speed

        xy_drift = torch.linalg.norm(base_asset.data.root_pos_w[:, :2] - self._xy_anchor, dim=1)

        self._prev_arm_command.copy_(arm_command)
        self._prev_zero_cmd_mask.copy_(zero_cmd_mask)
        return xy_drift * arm_activity * zero_cmd_mask


class ZeroCmdYawDriftUnderArmMotion(ManagerTermBase):
    """Penalize yaw drift relative to an anchor during zero-base-command arm motion."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._yaw_anchor = torch.zeros(env.num_envs, device=env.device)
        self._prev_arm_command = torch.zeros((env.num_envs, 0), device=env.device)
        self._prev_zero_cmd_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = False

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._yaw_anchor[env_ids] = 0.0
        self._prev_zero_cmd_mask[env_ids] = False
        if self._prev_arm_command.shape[1] > 0:
            self._prev_arm_command[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        arm_command_name: str,
        base_asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.08,
        arm_command_change_threshold: float = 0.05,
        arm_pose_weight: float = 0.6,
        arm_speed_weight: float = 0.4,
    ) -> torch.Tensor:
        base_asset: RigidObject = env.scene[base_asset_cfg.name]
        base_cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
        zero_cmd_mask = base_cmd < command_threshold

        arm_command = env.command_manager.get_command(arm_command_name)
        if not self._initialized or self._prev_arm_command.shape[1] != arm_command.shape[1]:
            self._prev_arm_command = torch.zeros_like(arm_command)
            self._initialized = True

        arm_delta = torch.linalg.norm(arm_command - self._prev_arm_command, dim=1)
        entered_zero_cmd = zero_cmd_mask & ~self._prev_zero_cmd_mask
        episode_start = env.episode_length_buf <= 1
        refresh_anchor = episode_start | entered_zero_cmd | (zero_cmd_mask & (arm_delta > arm_command_change_threshold))

        current_yaw = math_utils.euler_xyz_from_quat(base_asset.data.root_quat_w)[2]
        if torch.any(refresh_anchor):
            self._yaw_anchor[refresh_anchor] = current_yaw[refresh_anchor]

        arm_asset_cfg = self.cfg.params["arm_asset_cfg"]
        arm_asset: Articulation = env.scene[arm_asset_cfg.name]
        arm_offset = (
            arm_asset.data.joint_pos[:, arm_asset_cfg.joint_ids] - arm_asset.data.default_joint_pos[:, arm_asset_cfg.joint_ids]
        )
        arm_pose_mag = torch.linalg.norm(arm_offset, dim=1)
        arm_speed = torch.linalg.norm(arm_asset.data.joint_vel[:, arm_asset_cfg.joint_ids], dim=1)
        arm_activity = arm_pose_weight * arm_pose_mag + arm_speed_weight * arm_speed

        yaw_drift = torch.atan2(torch.sin(current_yaw - self._yaw_anchor), torch.cos(current_yaw - self._yaw_anchor)).abs()

        self._prev_arm_command.copy_(arm_command)
        self._prev_zero_cmd_mask.copy_(zero_cmd_mask)
        return yaw_drift * arm_activity * zero_cmd_mask


def zero_cmd_drift_under_arm_motion(
    env: ManagerBasedRLEnv,
    command_name: str,
    arm_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.08,
    pose_weight: float = 0.6,
    speed_weight: float = 0.4,
    xy_vel_weight: float = 1.0,
    yaw_weight: float = 0.35,
) -> torch.Tensor:
    """Penalize base drift under arm motion when the commanded base velocity is near zero."""
    arm_asset: Articulation = env.scene[arm_asset_cfg.name]
    base_asset: RigidObject = env.scene[base_asset_cfg.name]

    base_cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    zero_cmd_mask = base_cmd < command_threshold

    arm_offset = (
        arm_asset.data.joint_pos[:, arm_asset_cfg.joint_ids] - arm_asset.data.default_joint_pos[:, arm_asset_cfg.joint_ids]
    )
    arm_pose_mag = torch.linalg.norm(arm_offset, dim=1)
    arm_speed = torch.linalg.norm(arm_asset.data.joint_vel[:, arm_asset_cfg.joint_ids], dim=1)
    arm_activity = pose_weight * arm_pose_mag + speed_weight * arm_speed

    drift = xy_vel_weight * torch.linalg.norm(base_asset.data.root_lin_vel_b[:, :2], dim=1)
    drift += yaw_weight * torch.abs(base_asset.data.root_ang_vel_b[:, 2])
    return drift * arm_activity * zero_cmd_mask


def arm_action_in_unstable_base(
    env: ManagerBasedRLEnv,
    arm_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = 0.2,
    lin_vel_z_threshold: float = 0.5,
    ang_vel_threshold: float = 1.5,
) -> torch.Tensor:
    """Penalize large arm actions when the base is already unstable."""
    arm_asset: Articulation = env.scene[arm_asset_cfg.name]
    base_asset: RigidObject = env.scene[base_asset_cfg.name]

    base_tilt = torch.linalg.norm(base_asset.data.projected_gravity_b[:, :2], dim=1)
    lin_vel_z = torch.abs(base_asset.data.root_lin_vel_b[:, 2])
    ang_vel_xy = torch.linalg.norm(base_asset.data.root_ang_vel_b[:, :2], dim=1)

    instability = torch.relu(base_tilt - tilt_threshold)
    instability += 0.5 * torch.relu(lin_vel_z - lin_vel_z_threshold)
    instability += 0.5 * torch.relu(ang_vel_xy - ang_vel_threshold)

    arm_action_norm = torch.linalg.norm(env.action_manager.action[:, arm_asset_cfg.joint_ids], dim=1)
    reward = instability * arm_action_norm
    return reward
