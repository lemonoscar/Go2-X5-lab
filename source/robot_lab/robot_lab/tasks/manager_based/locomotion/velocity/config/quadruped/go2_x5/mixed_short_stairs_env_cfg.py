# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""DogOnly task mixing strict flat tracking with fixed two/three-step stairs."""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .mixed_short_stairs_terrain import MixedShortStairsTerrainCfg
from .train_route_env_cfg import (
    DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT,
    Go2X5DogOnlyStairsEnvCfg,
)


MIXED_SHORT_STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(6.0, 6.0),
    border_width=20.0,
    num_rows=2,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.60),
        "short_stairs_up": MixedShortStairsTerrainCfg(
            proportion=0.20,
            descent=False,
        ),
        "short_stairs_down": MixedShortStairsTerrainCfg(
            proportion=0.20,
            descent=True,
        ),
    },
)


@configclass
class Go2X5DogOnlyMixedShortStairsEnvCfg(Go2X5DogOnlyStairsEnvCfg):
    """One 260-observation/12-action policy for flat motion and short stairs."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.episode_length_s = 20.0
        self.scene.terrain.terrain_generator = MIXED_SHORT_STAIRS_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 0

        self.commands.base_velocity = mdp.MixedShortStairsVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 7.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=True,
            heading_control_stiffness=1.2,
            debug_vis=False,
            ranges=mdp.MixedShortStairsVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.40, 0.40),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.60, 0.60),
                heading=(-math.pi, math.pi),
            ),
        )
        self.commands.arm_joint_pos = mdp.ConditionalStandingArmJointPositionCommandCfg(
            asset_name="robot",
            joint_names=self.arm_joint_names,
            base_command_name="base_velocity",
            resampling_time_range=(2.5, 4.0),
            position_range=[(-0.15, 0.15)] * len(self.arm_joint_names),
            use_default_offset=True,
            clip_to_joint_limits=True,
            preserve_order=True,
        )

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.mixed_short_stairs_terrain_levels,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "move_up_distance_ratio": 0.30,
                "move_down_command_ratio": 0.20,
                "move_down_min_distance": 0.60,
            },
        )

        # The inherited stair task has unmasked progress terms. Replace them so
        # flat reverse/lateral/standing samples cannot acquire a forward bias.
        self.rewards.command_direction_progress = None
        self.rewards.commanded_stall_penalty = None
        self.rewards.flat_planar_tracking_excess = RewTerm(
            func=mdp.flat_planar_velocity_tracking_excess_l1,
            weight=-1.50,
            params={
                "command_name": "base_velocity",
                "vx_absolute_tolerance": 0.04,
                "vy_absolute_tolerance": 0.03,
                "relative_tolerance": 0.10,
                "max_penalty": 6.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stair_forward_progress = RewTerm(
            func=mdp.stair_command_direction_progress,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.08,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stair_stall = RewTerm(
            func=mdp.stair_commanded_stall_penalty,
            weight=-0.80,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.08,
                "min_progress_speed": 0.07,
                "max_penalty": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        base_body_cfg = SceneEntityCfg("robot", body_names=[self.base_link_name])
        arm_joint_cfg = SceneEntityCfg(
            "robot", joint_names=self.arm_joint_names, preserve_order=True
        )
        self.rewards.arm_joint_pos_tracking_l2 = RewTerm(
            func=mdp.arm_joint_pos_tracking_l2,
            weight=-2.0,
            params={"command_name": "arm_joint_pos", "asset_cfg": arm_joint_cfg},
        )
        self.rewards.arm_motion_tilt_penalty = RewTerm(
            func=mdp.arm_motion_tilt_penalty,
            weight=-0.60,
            params={
                "base_asset_cfg": base_body_cfg,
                "arm_asset_cfg": arm_joint_cfg,
                "tilt_clip": 1.0,
                "vel_clip": 4.0,
            },
        )
        self.rewards.arm_pose_conditioned_base_stability = RewTerm(
            func=mdp.arm_pose_conditioned_base_stability,
            weight=-0.60,
            params={
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "pose_clip": 2.0,
                "speed_clip": 4.0,
                "pose_weight": 1.0,
                "speed_weight": 0.30,
                "tilt_weight": 1.0,
                "ang_vel_weight": 0.35,
                "lin_vel_weight": 0.20,
            },
        )
        self.rewards.zero_cmd_drift_under_arm_motion = RewTerm(
            func=mdp.zero_cmd_drift_under_arm_motion,
            weight=-1.50,
            params={
                "command_name": "base_velocity",
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "command_threshold": 0.08,
                "pose_weight": 0.6,
                "speed_weight": 0.4,
                "xy_vel_weight": 1.0,
                "yaw_weight": 0.35,
            },
        )
        self.rewards.zero_cmd_xy_position_drift_under_arm_motion = RewTerm(
            func=mdp.MixedShortStairsZeroCmdXYPositionDrift,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "arm_command_name": "arm_joint_pos",
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "command_threshold": 0.08,
                "arm_command_change_threshold": 0.04,
                "arm_pose_weight": 0.6,
                "arm_speed_weight": 0.4,
            },
        )
        self.rewards.zero_cmd_yaw_drift_under_arm_motion = RewTerm(
            func=mdp.MixedShortStairsZeroCmdYawDrift,
            weight=-0.60,
            params={
                "command_name": "base_velocity",
                "arm_command_name": "arm_joint_pos",
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "command_threshold": 0.08,
                "arm_command_change_threshold": 0.04,
                "arm_pose_weight": 0.6,
                "arm_speed_weight": 0.4,
            },
        )

        mixed_weights = {
            "lin_vel_z_l2": -0.60,
            "ang_vel_xy_l2": -0.12,
            "flat_orientation_l2": -0.55,
            "base_height_l2": -0.18,
            "body_lin_acc_l2": -0.018,
            "joint_torques_l2": -1.5e-5,
            "joint_acc_l2": -1.0e-7,
            "joint_pos_limits": -2.0,
            "joint_power": -1.0e-5,
            "stand_still": -3.5,
            "joint_pos_penalty": -0.50,
            "action_rate_l2": -0.018,
            "undesired_contacts": -1.5,
            "contact_forces": -1.5e-4,
            "track_lin_vel_xy_exp": 5.0,
            "track_ang_vel_z_exp": 3.0,
            "feet_air_time": 0.45,
            "feet_air_time_variance": -0.12,
            "feet_contact_without_cmd": 0.40,
            "feet_slide": -0.18,
            "feet_drag": -0.05,
            "feet_height_body": -0.65,
            "feet_gait": 0.12,
            "upward": 0.80,
        }
        for attr_name, weight in mixed_weights.items():
            reward_term = getattr(self.rewards, attr_name, None)
            if reward_term is not None:
                reward_term.weight = weight

        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.18
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.15
        self.rewards.base_height_l2.params["target_height"] = (
            DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT
        )
        self.rewards.stand_still.params["command_threshold"] = 0.08
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.08
        self.rewards.joint_pos_penalty.params["velocity_threshold"] = 0.16
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 3.0
        self.rewards.feet_air_time.params["threshold"] = 0.46
        self.rewards.feet_height_body.params["target_height"] = -0.15
        self.rewards.feet_height_body.params["tanh_mult"] = 1.40
        self.rewards.feet_drag.func = mdp.terrain_invariant_feet_drag_penalty
        self.rewards.feet_drag.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[self.foot_link_name]
            ),
            "minimum_foot_height_body": -0.18,
            "feet_drag_sigma": 0.04,
        }
        self.rewards.feet_gait.params["command_threshold"] = 0.08
        self.rewards.feet_gait.params["velocity_threshold"] = 0.16
        self.rewards.feet_gait.params["max_err"] = 0.20

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.03),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.04, 0.04),
            },
            "velocity_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.03, 0.03),
                "roll": (-0.04, 0.04),
                "pitch": (-0.04, 0.04),
                "yaw": (-0.04, 0.04),
            },
        }
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (
            0.60,
            1.10,
        )
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (
            0.50,
            1.00,
        )
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.05)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (
            0.98,
            1.02,
        )
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (
            0.98,
            1.02,
        )
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.005, 0.005),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (
            0.95,
            1.05,
        )
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (
            0.95,
            1.05,
        )
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.sim2sim_action_delay_range = (1, 1)
        self.sim2sim_action_hold_prob = 0.02
        self.sim2sim_action_noise_std = 0.0015
        self.sim2sim_obs_delay_steps = 0

        self.terminations.bad_orientation.params["limit_angle"] = math.radians(35.0)
        self.terminations.root_ang_vel_xy_above_maximum.params["maximum_speed"] = 4.0

        self.disable_zero_weight_rewards()
