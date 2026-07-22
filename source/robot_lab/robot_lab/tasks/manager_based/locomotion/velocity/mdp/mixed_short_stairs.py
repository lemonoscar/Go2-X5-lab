# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Mode-aware commands and reward masks for the mixed short-stair task."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .commands import ArmJointPositionCommand, ArmJointPositionCommandCfg
from .curriculums import terrain_levels_vel_hard
from .rewards import (
    ZeroCmdXYPositionDriftUnderArmMotion,
    ZeroCmdYawDriftUnderArmMotion,
    command_direction_progress,
    commanded_stall_penalty,
    planar_velocity_tracking_excess_l1,
)
from .mixed_short_stairs_utils import (
    MODE_ASCENT,
    MODE_COMBINED,
    MODE_DESCENT,
    MODE_PURE_VX,
    MODE_PURE_VY,
    MODE_STAND,
    MODE_TURN,
    sample_flat_mixed_short_stairs_commands,
    validate_mixed_short_stairs_command_spec,
)
from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


class MixedShortStairsVelocityCommand(UniformVelocityCommand):
    """Sample strict flat commands and conservative terrain-conditioned stair commands."""

    cfg: "MixedShortStairsVelocityCommandCfg"

    def __init__(self, cfg: "MixedShortStairsVelocityCommandCfg", env: ManagerBasedEnv):
        probabilities = (
            cfg.standing_probability,
            cfg.pure_vx_probability,
            cfg.pure_vy_probability,
            cfg.combined_probability,
            cfg.turn_probability,
        )
        validate_mixed_short_stairs_command_spec(
            cfg.vx_values,
            cfg.vy_values,
            cfg.ascent_speed_values,
            cfg.descent_speed_values,
            probabilities,
            cfg.minimum_translation_speed,
            cfg.arm_motion_probability_within_standing,
            cfg.turn_angle_rad,
        )

        super().__init__(cfg, env)

        self.vx_values = torch.tensor(cfg.vx_values, dtype=torch.float32, device=self.device)
        self.vy_values = torch.tensor(cfg.vy_values, dtype=torch.float32, device=self.device)
        self.ascent_speed_values = torch.tensor(
            cfg.ascent_speed_values, dtype=torch.float32, device=self.device
        )
        self.descent_speed_values = torch.tensor(
            cfg.descent_speed_values, dtype=torch.float32, device=self.device
        )
        self.category_cdf = torch.tensor(
            probabilities, dtype=torch.float32, device=self.device
        ).cumsum(dim=0)

        self.flat_env_mask = is_env_assigned_to_terrain(env, cfg.flat_terrain_name)
        self.ascent_env_mask = is_env_assigned_to_terrain(env, cfg.ascent_terrain_name)
        self.descent_env_mask = is_env_assigned_to_terrain(env, cfg.descent_terrain_name)
        self.stair_env_mask = self.ascent_env_mask | self.descent_env_mask
        assigned_count = self.flat_env_mask.to(torch.int8)
        assigned_count += self.ascent_env_mask.to(torch.int8)
        assigned_count += self.descent_env_mask.to(torch.int8)
        if torch.any(assigned_count != 1):
            raise ValueError("Every mixed short-stair environment must belong to exactly one terrain type.")

        self.mode_buffer = torch.full(
            (self.num_envs,), MODE_STAND, dtype=torch.long, device=self.device
        )
        self.arm_motion_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for metric_name in (
            "mode_flat_stand",
            "mode_flat_arm",
            "mode_flat_vx",
            "mode_flat_vy",
            "mode_flat_combined",
            "mode_flat_turn",
            "mode_ascent",
            "mode_descent",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _sample_values(values: torch.Tensor, count: int) -> torch.Tensor:
        indices = torch.randint(0, len(values), (count,), device=values.device)
        return values[indices]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.vel_command_b[ids] = 0.0
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False
        self.arm_motion_mask[ids] = False

        local_ascent = self.ascent_env_mask[ids]
        if torch.any(local_ascent):
            ascent_ids = ids[local_ascent]
            self.vel_command_b[ascent_ids, 0] = self._sample_values(
                self.ascent_speed_values, len(ascent_ids)
            )
            self.mode_buffer[ascent_ids] = MODE_ASCENT

        local_descent = self.descent_env_mask[ids]
        if torch.any(local_descent):
            descent_ids = ids[local_descent]
            self.vel_command_b[descent_ids, 0] = self._sample_values(
                self.descent_speed_values, len(descent_ids)
            )
            self.mode_buffer[descent_ids] = MODE_DESCENT

        flat_ids = ids[self.flat_env_mask[ids]]
        if len(flat_ids) == 0:
            return

        commands, modes, arm_motion, turn_delta = sample_flat_mixed_short_stairs_commands(
            len(flat_ids),
            self.vx_values,
            self.vy_values,
            self.category_cdf,
            self.cfg.arm_motion_probability_within_standing,
            self.cfg.turn_angle_rad,
        )
        self.vel_command_b[flat_ids] = commands
        self.mode_buffer[flat_ids] = modes
        self.arm_motion_mask[flat_ids] = arm_motion
        stand_ids = flat_ids[modes == MODE_STAND]
        turn_ids = flat_ids[modes == MODE_TURN]
        self.is_standing_env[stand_ids] = True
        if len(turn_ids) > 0:
            self.heading_target[turn_ids] = math_utils.wrap_to_pi(
                self.robot.data.heading_w[turn_ids] + turn_delta[modes == MODE_TURN]
            )
            self.is_heading_env[turn_ids] = True
            self.mode_buffer[turn_ids] = MODE_TURN

    def _update_metrics(self):
        super()._update_metrics()
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        scale = 1.0 / max(max_command_steps, 1.0)
        masks = {
            "mode_flat_stand": (self.mode_buffer == MODE_STAND) & ~self.arm_motion_mask,
            "mode_flat_arm": self.arm_motion_mask,
            "mode_flat_vx": self.mode_buffer == MODE_PURE_VX,
            "mode_flat_vy": self.mode_buffer == MODE_PURE_VY,
            "mode_flat_combined": self.mode_buffer == MODE_COMBINED,
            "mode_flat_turn": self.mode_buffer == MODE_TURN,
            "mode_ascent": self.mode_buffer == MODE_ASCENT,
            "mode_descent": self.mode_buffer == MODE_DESCENT,
        }
        for metric_name, mask in masks.items():
            self.metrics[metric_name] += mask.to(torch.float32) * scale


@configclass
class MixedShortStairsVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`MixedShortStairsVelocityCommand`."""

    class_type: type = MixedShortStairsVelocityCommand

    vx_values: tuple[float, ...] = (-0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40)
    vy_values: tuple[float, ...] = (-0.20, -0.15, -0.10, 0.10, 0.15, 0.20)
    ascent_speed_values: tuple[float, ...] = (0.18, 0.22, 0.25)
    descent_speed_values: tuple[float, ...] = (0.12, 0.16, 0.20)
    minimum_translation_speed: float = 0.10

    standing_probability: float = 0.25
    pure_vx_probability: float = 0.25
    pure_vy_probability: float = 0.20
    combined_probability: float = 0.10
    turn_probability: float = 0.20
    arm_motion_probability_within_standing: float = 0.40
    turn_angle_rad: float = math.pi / 4.0

    flat_terrain_name: str = "flat"
    ascent_terrain_name: str = "short_stairs_up"
    descent_terrain_name: str = "short_stairs_down"


class ConditionalStandingArmJointPositionCommand(ArmJointPositionCommand):
    """Move the arm only in the base command term's designated standing subset."""

    cfg: "ConditionalStandingArmJointPositionCommandCfg"

    def __init__(self, cfg: "ConditionalStandingArmJointPositionCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._previous_motion_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """Return the default pose immediately outside the standing-arm subset.

        The command manager currently evaluates the arm term before the base
        term.  Applying the mask here makes a stand-to-locomotion transition
        lock the arm in the same control step even if the arm buffer itself is
        refreshed on the next command-manager update.
        """
        motion_mask = self._motion_mask()
        defaults = self.asset.data.default_joint_pos[:, self.joint_ids]
        return torch.where(motion_mask[:, None], self.command_buffer, defaults)

    def _motion_mask(self) -> torch.Tensor:
        base_term = self._env.command_manager.get_term(self.cfg.base_command_name)
        if not hasattr(base_term, "arm_motion_mask"):
            raise TypeError(
                f"Command '{self.cfg.base_command_name}' does not expose arm_motion_mask."
            )
        return base_term.arm_motion_mask

    def _write_default_targets(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.command_buffer[env_ids] = self.asset.data.default_joint_pos[env_ids][
            :, self.joint_ids
        ]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._write_default_targets(ids)
        motion_ids = ids[self._motion_mask()[ids]]
        if len(motion_ids) > 0:
            super()._resample_command(motion_ids)
        self._previous_motion_mask[ids] = self._motion_mask()[ids]

    def _update_command(self):
        motion_mask = self._motion_mask()
        entered_motion = motion_mask & ~self._previous_motion_mask
        if torch.any(entered_motion):
            super()._resample_command(torch.where(entered_motion)[0])
        self._write_default_targets(torch.where(~motion_mask)[0])
        self._previous_motion_mask.copy_(motion_mask)


@configclass
class ConditionalStandingArmJointPositionCommandCfg(ArmJointPositionCommandCfg):
    """Configuration for :class:`ConditionalStandingArmJointPositionCommand`."""

    class_type: type = ConditionalStandingArmJointPositionCommand
    base_command_name: str = "base_velocity"


def mixed_short_stairs_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    move_up_distance_ratio: float = 0.30,
    move_down_command_ratio: float = 0.20,
    move_down_min_distance: float = 0.60,
) -> torch.Tensor:
    """Keep the first reset on two steps, then use progress-based promotion.

    Isaac Lab evaluates reset curricula before the robots have been placed at
    their terrain origins.  The generic distance curriculum consequently sees
    a large, artificial displacement on that first call and promotes almost
    every environment immediately.  Skipping only that call preserves the
    intended two-step start without changing later promotion/demotion logic.
    """
    terrain = env.scene.terrain
    if env.common_step_counter == 0:
        if terrain.terrain_levels is None:
            return torch.tensor(0.0, device=env.device)
        return torch.mean(terrain.terrain_levels.float())

    return terrain_levels_vel_hard(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        move_up_distance_ratio=move_up_distance_ratio,
        move_down_command_ratio=move_down_command_ratio,
        move_down_min_distance=move_down_min_distance,
    )


class MixedShortStairsZeroCmdXYPositionDrift(ZeroCmdXYPositionDriftUnderArmMotion):
    """Expose the arm entity parameter required internally by the inherited term."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        arm_command_name: str,
        arm_asset_cfg: SceneEntityCfg,
        base_asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.08,
        arm_command_change_threshold: float = 0.05,
        arm_pose_weight: float = 0.6,
        arm_speed_weight: float = 0.4,
    ) -> torch.Tensor:
        del arm_asset_cfg
        return super().__call__(
            env,
            command_name=command_name,
            arm_command_name=arm_command_name,
            base_asset_cfg=base_asset_cfg,
            command_threshold=command_threshold,
            arm_command_change_threshold=arm_command_change_threshold,
            arm_pose_weight=arm_pose_weight,
            arm_speed_weight=arm_speed_weight,
        )


class MixedShortStairsZeroCmdYawDrift(ZeroCmdYawDriftUnderArmMotion):
    """Expose the arm entity parameter required internally by the inherited term."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        arm_command_name: str,
        arm_asset_cfg: SceneEntityCfg,
        base_asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.08,
        arm_command_change_threshold: float = 0.05,
        arm_pose_weight: float = 0.6,
        arm_speed_weight: float = 0.4,
    ) -> torch.Tensor:
        del arm_asset_cfg
        return super().__call__(
            env,
            command_name=command_name,
            arm_command_name=arm_command_name,
            base_asset_cfg=base_asset_cfg,
            command_threshold=command_threshold,
            arm_command_change_threshold=arm_command_change_threshold,
            arm_pose_weight=arm_pose_weight,
            arm_speed_weight=arm_speed_weight,
        )


def flat_planar_velocity_tracking_excess_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    vx_absolute_tolerance: float,
    vy_absolute_tolerance: float,
    relative_tolerance: float,
    max_penalty: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apply the strict planar excess penalty only to flat terrain cells."""
    command_term = env.command_manager.get_term(command_name)
    penalty = planar_velocity_tracking_excess_l1(
        env,
        command_name=command_name,
        vx_absolute_tolerance=vx_absolute_tolerance,
        vy_absolute_tolerance=vy_absolute_tolerance,
        relative_tolerance=relative_tolerance,
        max_penalty=max_penalty,
        asset_cfg=asset_cfg,
    )
    return penalty * command_term.flat_env_mask.to(penalty.dtype)


def stair_command_direction_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward signed progress only on the ascent/descent terrain columns."""
    command_term = env.command_manager.get_term(command_name)
    reward = command_direction_progress(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        asset_cfg=asset_cfg,
    )
    return reward * command_term.stair_env_mask.to(reward.dtype)


def stair_commanded_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    min_progress_speed: float,
    max_penalty: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize stalls only on the ascent/descent terrain columns."""
    command_term = env.command_manager.get_term(command_name)
    penalty = commanded_stall_penalty(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        min_progress_speed=min_progress_speed,
        max_penalty=max_penalty,
        asset_cfg=asset_cfg,
    )
    return penalty * command_term.stair_env_mask.to(penalty.dtype)


def terrain_invariant_feet_drag_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_foot_height_body: float,
    feet_drag_sigma: float,
) -> torch.Tensor:
    """Penalize fast, low swing feet without assuming a world-frame floor height.

    The shared drag term uses an absolute world-Z threshold, which stops being
    meaningful on the elevated approach of a descent route.  This variant uses
    body-frame foot height for the clearance gate and world-frame planar foot
    speed so a correctly planted stance foot is not mistaken for a dragging
    swing foot.
    """
    if feet_drag_sigma <= 0.0:
        raise ValueError("feet_drag_sigma must be positive.")

    asset = env.scene[asset_cfg.name]
    foot_pos_relative_w = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        - asset.data.root_pos_w[:, None, :]
    )
    foot_pos_body = math_utils.quat_apply_inverse(
        asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 4),
        foot_pos_relative_w.reshape(-1, 3),
    ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
    below_clearance = torch.clamp(
        foot_pos_body[..., 2] - minimum_foot_height_body,
        max=0.0,
    )
    low_height_scale = 1.0 - torch.exp(below_clearance / feet_drag_sigma)
    foot_planar_speed_sq = torch.sum(
        torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=2
    )
    penalty = torch.sum(low_height_scale * foot_planar_speed_sq, dim=1)
    upright_scale = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * upright_scale
