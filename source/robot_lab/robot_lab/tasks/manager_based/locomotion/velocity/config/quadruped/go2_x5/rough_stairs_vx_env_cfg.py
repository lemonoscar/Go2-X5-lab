# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unified DogOnly task for bidirectional stair traversal and precise forward speed tracking."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .train_route_env_cfg import (
    DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT,
    DOG_ONLY_ROUGH_CHECKPOINT_JOINT_POS,
    Go2X5DogOnlyRoughEnvCfg,
)


ROUGH_STAIRS_VX_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        # Inverted stairs spawn at the bottom; positive body-frame vx climbs outward.
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.157),
            step_width=0.2856,
            platform_width=1.50,
            border_width=0.50,
            holes=False,
        ),
        # Pyramid stairs spawn at the top; positive body-frame vx descends outward.
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.157),
            step_width=0.2856,
            platform_width=1.50,
            border_width=0.50,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.10),
            noise_step=0.01,
            border_width=0.25,
            downsampled_scale=0.20,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.03, 0.14),
            platform_width=2.0,
            holes=False,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class Go2X5DogOnlyRoughStairsVxEnvCfg(Go2X5DogOnlyRoughEnvCfg):
    """One 260-observation/12-action policy for flat vx and up/down stairs."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.terrain.terrain_generator = ROUGH_STAIRS_VX_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 1

        # The selected model_26250 continuation checkpoint was trained around
        # this pose, not the newer independent DogOnly crouch experiment.
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(DOG_ONLY_ROUGH_CHECKPOINT_JOINT_POS)
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            pos=(0.0, 0.0, DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT),
            joint_pos=joint_pos,
        )

        self.commands.base_velocity = mdp.StratifiedVxVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 6.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            speed_values=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            stair_speed_values=(0.15, 0.20, 0.25, 0.30),
            stair_terrain_names=("stairs_up", "stairs_down"),
            initial_active_speed_count=5,
            promotion_interval_iterations=250,
            full_range_rehearsal_probability=0.35,
            ranges=mdp.StratifiedVxVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.7),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
        )

        self.curriculum.command_levels_lin_vel = CurrTerm(
            func=mdp.stratified_vx_command_curriculum,
            params={
                "command_name": "base_velocity",
                "reward_term_name": "track_vx_tolerance",
                "steps_per_iteration": 32,
                "full_range_iteration": 2000,
                "performance_threshold": 0.75,
            },
        )
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.rough_stairs_vx_terrain_levels,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "steps_per_iteration": 32,
                "full_difficulty_iteration": 2000,
                "move_up_distance_ratio": 0.25,
                "move_down_command_ratio": 0.25,
                "move_down_min_distance": 0.70,
            },
        )

        # Keep the inherited broad reward as a weak shaping signal.  The new
        # terms define the actual acceptance band and ignore errors <=0.1 m/s.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.20
        self.rewards.track_vx_tolerance = RewTerm(
            func=mdp.track_vx_tolerance_exp,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "absolute_tolerance": 0.1,
                "relative_tolerance": 0.1,
                "outside_tolerance_std": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.vx_tracking_excess = RewTerm(
            func=mdp.vx_tracking_excess_l1,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "absolute_tolerance": 0.1,
                "relative_tolerance": 0.1,
                "max_penalty": 4.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.uncommanded_velocity_excess = RewTerm(
            func=mdp.uncommanded_velocity_excess_l1,
            weight=-0.5,
            params={
                "lateral_tolerance": 0.1,
                "yaw_tolerance": 0.1,
                "max_penalty": 4.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        unified_weights = {
            "lin_vel_z_l2": -0.70,
            "ang_vel_xy_l2": -0.10,
            "flat_orientation_l2": -0.50,
            "base_height_l2": -0.15,
            "stand_still": -2.4,
            "undesired_contacts": -1.3,
            "feet_air_time": 0.30,
            "feet_air_time_variance": -0.14,
            "feet_contact_without_cmd": 0.30,
            "feet_slide": -0.16,
            "feet_drag": -0.05,
            "feet_height_body": -0.50,
            "feet_gait": 0.12,
            "upward": 0.80,
        }
        for attr_name, weight in unified_weights.items():
            reward_term = getattr(self.rewards, attr_name, None)
            if reward_term is not None:
                reward_term.weight = weight

        self.rewards.base_height_l2.params["target_height"] = DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT
        self.rewards.stand_still.params["command_threshold"] = 0.05
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.05
        self.rewards.feet_gait.params["command_threshold"] = 0.05
        self.rewards.feet_gait.params["velocity_threshold"] = 0.16
        self.rewards.feet_gait.params["max_err"] = 0.24
        self.rewards.feet_air_time.params["threshold"] = 0.54
        self.rewards.feet_height_body.params["target_height"] = -0.13
        self.rewards.feet_height_body.params["tanh_mult"] = 1.30
        self.rewards.feet_drag.params["penalty_feet_drag_height"] = 0.16
        self.rewards.feet_drag.params["feet_drag_sigma"] = 0.035

        self.events.randomize_reset_base.params["pose_range"]["z"] = (0.0, 0.10)
        self.events.randomize_reset_base.params["pose_range"]["roll"] = (-0.10, 0.10)
        self.events.randomize_reset_base.params["pose_range"]["pitch"] = (-0.10, 0.10)
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.50, 1.20)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.40, 1.05)

        self.sim2sim_action_hold_prob = 0.035
        self.sim2sim_action_noise_std = 0.0025

        self.disable_zero_weight_rewards()
