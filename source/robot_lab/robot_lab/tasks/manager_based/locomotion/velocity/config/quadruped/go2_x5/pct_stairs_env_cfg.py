# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""PCT first-flight straight-stair continuation task for the DogOnly policy."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .pct_stairs_terrain import (
    PCT_REGULAR_STAIR_APPROACH_M,
    PCT_REGULAR_STAIR_COUNT,
    PCT_REGULAR_STAIR_FLIGHT_RUN_M,
    PCT_REGULAR_STAIR_FLIGHT_RISE_M,
    PCT_REGULAR_STAIR_PATH_HEIGHT_FRACTIONS,
    PCT_REGULAR_STAIR_PATH_LENGTH_M,
    PCT_REGULAR_STAIR_PATH_POINTS_XY,
    PCT_REGULAR_STAIR_PLATFORM_GATE_M,
    PCT_REGULAR_STAIR_RISER_M,
    PCT_REGULAR_STAIR_TOP_EXIT_M,
    PCT_REGULAR_STAIR_WIDTH_M,
    PCT_REGULAR_UP_DOWN_BOTTOM_EXIT_M,
    PCT_REGULAR_UP_DOWN_COMPLETION_GATE_M,
    PCT_REGULAR_UP_DOWN_DESCENT_START_M,
    PCT_REGULAR_UP_DOWN_PATH_HEIGHT_FRACTIONS,
    PCT_REGULAR_UP_DOWN_PATH_LENGTH_M,
    PCT_REGULAR_UP_DOWN_PATH_POINTS_XY,
    PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M,
    PCT_STRAIGHT_STAIR_PATH_HEIGHT_FRACTIONS,
    PCT_STRAIGHT_STAIR_PATH_POINTS_XY,
    PctRegularUpDownStairsTerrainCfg,
    PctScannedStraightStairsTerrainCfg,
    PctStraightStairsTerrainCfg,
)
from .train_route_env_cfg import Go2X5DogOnlyStairsEnvCfg


PCT_STAIRS_V2_TERRAIN_CFG = TerrainGeneratorCfg(
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
        "pct_straight_nominal": PctStraightStairsTerrainCfg(
            proportion=0.10,
            route_width=1.0,
            riser_variation=0.0,
        ),
        "pct_straight_irregular": PctStraightStairsTerrainCfg(
            proportion=0.10,
            route_width=0.86,
            riser_variation=0.08,
        ),
        "pct_scanned_first_flight": PctScannedStraightStairsTerrainCfg(
            proportion=0.80,
        ),
    },
)


PCT_STAIRS_HARD_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_straight_nominal": PctStraightStairsTerrainCfg(
            proportion=0.05,
            route_width=1.0,
            step_height_range=(0.0785, 0.0785),
            riser_variation=0.0,
        ),
        "pct_straight_irregular": PctStraightStairsTerrainCfg(
            proportion=0.05,
            route_width=0.86,
            step_height_range=(0.0785, 0.0785),
            riser_variation=0.08,
        ),
        "pct_scanned_first_flight": PctScannedStraightStairsTerrainCfg(
            proportion=0.90,
            scan_target_rise_range=(1.57, 1.57),
        ),
    },
)


PCT_REGULAR_STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=32,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_regular_stairs": PctStraightStairsTerrainCfg(
            proportion=1.0,
            route_width=PCT_REGULAR_STAIR_WIDTH_M,
            approach_length=PCT_REGULAR_STAIR_APPROACH_M,
            flight_run=PCT_REGULAR_STAIR_FLIGHT_RUN_M,
            flight_steps=PCT_REGULAR_STAIR_COUNT,
            step_height_range=(PCT_REGULAR_STAIR_RISER_M, PCT_REGULAR_STAIR_RISER_M),
            top_platform_exit_length=PCT_REGULAR_STAIR_TOP_EXIT_M,
            riser_variation=0.0,
        ),
    },
)


PCT_REGULAR_ASCENT_MIN_RISER_M = 0.08
PCT_REGULAR_ASCENT_CURRICULUM_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=32,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_regular_stairs": PctStraightStairsTerrainCfg(
            proportion=1.0,
            route_width=PCT_REGULAR_STAIR_WIDTH_M,
            approach_length=PCT_REGULAR_STAIR_APPROACH_M,
            flight_run=PCT_REGULAR_STAIR_FLIGHT_RUN_M,
            flight_steps=PCT_REGULAR_STAIR_COUNT,
            step_height_range=(
                PCT_REGULAR_ASCENT_MIN_RISER_M,
                PCT_REGULAR_STAIR_RISER_M,
            ),
            top_platform_exit_length=PCT_REGULAR_STAIR_TOP_EXIT_M,
            riser_variation=0.0,
        ),
    },
)


PCT_REGULAR_ASCENT_SIM2REAL_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=32,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_regular_nominal": PctStraightStairsTerrainCfg(
            proportion=0.5,
            route_width=PCT_REGULAR_STAIR_WIDTH_M,
            approach_length=PCT_REGULAR_STAIR_APPROACH_M,
            flight_run=PCT_REGULAR_STAIR_FLIGHT_RUN_M,
            flight_steps=PCT_REGULAR_STAIR_COUNT,
            step_height_range=(
                PCT_REGULAR_ASCENT_MIN_RISER_M,
                PCT_REGULAR_STAIR_RISER_M,
            ),
            top_platform_exit_length=PCT_REGULAR_STAIR_TOP_EXIT_M,
            riser_variation=0.0,
        ),
        "pct_regular_irregular": PctStraightStairsTerrainCfg(
            proportion=0.5,
            route_width=PCT_REGULAR_STAIR_WIDTH_M,
            approach_length=PCT_REGULAR_STAIR_APPROACH_M,
            flight_run=PCT_REGULAR_STAIR_FLIGHT_RUN_M,
            flight_steps=PCT_REGULAR_STAIR_COUNT,
            step_height_range=(
                PCT_REGULAR_ASCENT_MIN_RISER_M,
                PCT_REGULAR_STAIR_RISER_M,
            ),
            top_platform_exit_length=PCT_REGULAR_STAIR_TOP_EXIT_M,
            riser_variation=0.05,
        ),
    },
)


PCT_REGULAR_UP_DOWN_STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=1,
    num_cols=32,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_regular_up_down_stairs": PctRegularUpDownStairsTerrainCfg(
            proportion=1.0,
            route_width=PCT_REGULAR_STAIR_WIDTH_M,
            approach_length=PCT_REGULAR_STAIR_APPROACH_M,
            flight_run=PCT_REGULAR_STAIR_FLIGHT_RUN_M,
            flight_steps=PCT_REGULAR_STAIR_COUNT,
            step_height=PCT_REGULAR_STAIR_RISER_M,
            middle_platform_length=PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M,
            bottom_exit_length=PCT_REGULAR_UP_DOWN_BOTTOM_EXIT_M,
        ),
    },
)


# Measured from the centerline of the unscaled canonical collision crop.  The
# first 0.05 m is the terrain generator's approach offset; the remaining
# points are scan progress 0.425, 0.60, 0.75, and 1.00 m respectively.
PCT_FIRST_STEPS_PATH_POINTS_XY = (
    (0.0, 0.0),
    (0.0, 0.475),
    (0.0, 0.650),
    (0.0, 0.800),
    (0.0, 1.050),
)
PCT_FIRST_STEPS_TARGET_RISE = 0.314
PCT_FIRST_STEPS_PATH_HEIGHT_FRACTIONS = (
    0.0,
    0.0,
    0.141 / PCT_FIRST_STEPS_TARGET_RISE,
    0.178 / PCT_FIRST_STEPS_TARGET_RISE,
    1.0,
)
PCT_FIRST_RISE_PATH_POINTS_XY = (
    (0.0, 0.0),
    (0.0, 0.475),
    (0.0, 0.650),
)
PCT_FIRST_RISE_PATH_HEIGHT_FRACTIONS = (0.0, 0.0, 1.0)
PCT_FIRST_RISE_TARGET = 0.141
PCT_FIRST_RISE_EXACT_TARGET = 0.150
PCT_SECOND_RISE_PATH_POINTS_XY = PCT_FIRST_STEPS_PATH_POINTS_XY[:4]
PCT_SECOND_RISE_TARGET = 0.178
PCT_SECOND_RISE_PATH_HEIGHT_FRACTIONS = (
    0.0,
    0.0,
    PCT_FIRST_RISE_EXACT_TARGET / PCT_SECOND_RISE_TARGET,
    1.0,
)

# Median of downward ray hits at centerline cross-track offsets
# {-0.12, 0.0, +0.12} m in the same unscaled collision PLY. Heights are
# relative to the entrance surface (-0.13 m in the source mesh). The real
# flight reaches nearly full height by progress 3.15 m; a single linear ramp
# over 3.85 m over-commands the first risers and under-commands the landing.
PCT_PROFILED_PROGRESS_HEIGHT_M = (
    (0.00, 0.000),
    (0.45, 0.000),
    (0.60, 0.141),
    (0.75, 0.178),
    (0.90, 0.305),
    (1.05, 0.334),
    (1.20, 0.468),
    (1.35, 0.484),
    (1.50, 0.615),
    (1.65, 0.673),
    (1.80, 0.775),
    (1.95, 0.861),
    (2.10, 0.931),
    (2.25, 1.033),
    (2.40, 1.087),
    (2.55, 1.209),
    (2.70, 1.254),
    (2.85, 1.410),
    (3.00, 1.430),
    (3.15, 1.569),
    (3.85, 1.570),
)
PCT_PROFILED_PATH_POINTS_XY = (
    (0.0, 0.0),
    *((0.0, 0.05 + progress) for progress, _ in PCT_PROFILED_PROGRESS_HEIGHT_M),
    (0.0, 4.60),
)
PCT_PROFILED_PATH_HEIGHT_FRACTIONS = (
    0.0,
    *(height / 1.57 for _, height in PCT_PROFILED_PROGRESS_HEIGHT_M),
    1.0,
)
PCT_PROFILED_MID_STAIR_SPAWN_ANCHORS = tuple(
    (progress, height)
    for progress, height in PCT_PROFILED_PROGRESS_HEIGHT_M
    if progress in (0.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85)
)
PCT_PROFILED_TOP_LANDING_SPAWN_ANCHORS = tuple(
    (progress, height)
    for progress, height in PCT_PROFILED_PROGRESS_HEIGHT_M
    if progress in (2.55, 2.70, 2.85, 3.00)
)
PCT_PLATFORM_PROGRESS_M = 3.902
PCT_PLATFORM_INTERIOR_PROGRESS_M = 4.15
PCT_PROFILED_PATH_LENGTH_M = 4.60


PCT_STAIRS_FIRST_STEPS_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_scanned_first_flight": PctScannedStraightStairsTerrainCfg(
            proportion=1.0,
            scan_target_rise_range=(1.57, 1.57),
        ),
    },
)


PCT_STAIRS_FIRST_RISE_EXACT_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.80,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "pct_scanned_first_flight": PctScannedStraightStairsTerrainCfg(
            proportion=1.0,
            start_position=(2.0, 1.30),
            scan_target_rise_range=(1.57, 1.57),
            scan_crop_mode="local_volume",
            scan_crop_cross_track_half_width=1.0,
            scan_crop_height_range=(-0.8, 2.0),
            scan_include_auxiliary_floor=False,
            scan_include_auxiliary_top_platform=False,
        ),
    },
)


@configclass
class Go2X5DogOnlyPctStairsEnvCfg(Go2X5DogOnlyStairsEnvCfg):
    """Learn one straight PCT stair flight without changing the 260-D/12-D contract."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 512
        self.episode_length_s = 32.0
        self.scene.terrain.terrain_generator = PCT_STAIRS_V2_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 0

        self.commands.base_velocity = mdp.PctStairVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=False,
            path_points_xy=PCT_STRAIGHT_STAIR_PATH_POINTS_XY,
            path_height_fractions=PCT_STRAIGHT_STAIR_PATH_HEIGHT_FRACTIONS,
            forward_velocity=0.25,
            max_lateral_velocity=0.12,
            max_angular_velocity=0.50,
            heading_kp=2.0,
            cross_track_kp=0.80,
            full_speed_heading_error=0.10,
            stop_forward_heading_error=0.30,
            waypoint_tolerance=0.22,
            goal_tolerance=0.25,
            segment_advance_ratio=0.92,
            completion_progress_ratio=0.94,
            completion_height_ratio=0.88,
            nominal_base_height=0.30,
            minimum_total_rise=0.35,
            maximum_total_rise=1.72,
            terrain_num_rows=10,
        )

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.pct_stair_completion_levels,
            params={
                "command_name": "base_velocity",
                "contact_force_threshold": 35.0,
                "base_head_arm_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["^(base|Head_.*|arm_.*)$"]
                ),
                "hip_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip"]),
                "thigh_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh"]),
                "required_consecutive_successes": 3,
                "move_down_progress_ratio": 0.55,
                "minimum_upright_projection": math.cos(math.radians(20.0)),
            },
        )

        v2_weights = {
            "lin_vel_z_l2": -0.40,
            "ang_vel_xy_l2": -0.16,
            "flat_orientation_l2": -0.55,
            "body_lin_acc_l2": -0.025,
            "action_rate_l2": -0.018,
            "undesired_contacts": -2.0,
            "contact_forces": -2.0e-4,
            "track_lin_vel_xy_exp": 4.5,
            "track_ang_vel_z_exp": 1.8,
            "feet_air_time": 0.55,
            "feet_air_time_variance": -0.14,
            "feet_slide": -0.22,
            "feet_height_body": -3.0,
            "feet_gait": 0.05,
            "upward": 0.80,
            "command_direction_progress": 0.50,
            "commanded_stall_penalty": -0.75,
        }
        for attr_name, weight in v2_weights.items():
            reward_term = getattr(self.rewards, attr_name, None)
            if reward_term is not None:
                reward_term.weight = weight

        unsafe_body_sensor_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["^(base|Head_.*|arm_.*|.*_hip|.*_thigh)$"],
        )
        self.rewards.pct_path_progress = RewTerm(
            func=mdp.PctStairPathProgressReward,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "height_tracking_std": 0.20,
                "contact_force_threshold": 35.0,
                "maximum_progress_speed": 0.45,
                "maximum_regress_speed": 0.25,
                "regression_scale": 1.5,
                "sensor_cfg": unsafe_body_sensor_cfg,
            },
        )
        self.rewards.pct_height_alignment = RewTerm(
            func=mdp.pct_stair_height_alignment,
            weight=1.25,
            params={
                "command_name": "base_velocity",
                "height_tracking_std": 0.20,
                "command_threshold": 0.08,
                "target_speed": 0.25,
            },
        )
        self.rewards.pct_base_clearance_deficit = RewTerm(
            func=mdp.pct_stair_base_clearance_deficit,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "clearance_margin": 0.04,
                "maximum_deficit": 0.12,
                "command_threshold": 0.08,
            },
        )
        self.rewards.pct_cross_track_alignment = RewTerm(
            func=mdp.pct_stair_cross_track_alignment,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "cross_track_std": 0.20,
                "command_threshold": 0.08,
                "target_speed": 0.25,
            },
        )
        self.rewards.pct_unsafe_contact = RewTerm(
            func=mdp.pct_stair_nonfoot_contact,
            weight=-1000.0,
            params={
                "sensor_cfg": unsafe_body_sensor_cfg,
                "threshold": 35.0,
                "minimum_episode_steps": 50,
            },
        )
        self.rewards.pct_path_completion = RewTerm(
            func=mdp.pct_stair_completion_bonus,
            weight=100.0,
            params={
                "command_name": "base_velocity",
                "contact_force_threshold": 35.0,
                "maximum_root_ang_vel_xy": 4.0,
                "maximum_root_lin_vel_z": 2.0,
                "minimum_upright_projection": math.cos(math.radians(20.0)),
                "maximum_cross_track_error": 0.45,
                "sensor_cfg": unsafe_body_sensor_cfg,
            },
        )

        self.terminations.pct_path_completed = DoneTerm(
            func=mdp.pct_stair_path_completed,
            params={
                "command_name": "base_velocity",
                "contact_force_threshold": 35.0,
                "maximum_root_ang_vel_xy": 4.0,
                "maximum_root_lin_vel_z": 2.0,
                "minimum_upright_projection": math.cos(math.radians(20.0)),
                "maximum_cross_track_error": 0.45,
                "sensor_cfg": unsafe_body_sensor_cfg,
            },
            time_out=True,
        )
        self.terminations.pct_path_deviation = DoneTerm(
            func=mdp.pct_stair_path_deviation,
            params={
                "command_name": "base_velocity",
                "maximum_cross_track_error": 0.45,
            },
        )
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.pct_stair_nonfoot_contact,
            params={
                "sensor_cfg": unsafe_body_sensor_cfg,
                "threshold": 35.0,
                "minimum_episode_steps": 50,
            },
        )
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(32.0)
        self.terminations.root_lin_vel_z_above_maximum.params["maximum_speed"] = 2.0
        self.terminations.root_ang_vel_xy_above_maximum.params["maximum_speed"] = 4.0

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.04),
            "roll": (-0.04, 0.04),
            "pitch": (-0.04, 0.04),
            "yaw": (math.pi / 2.0 - 0.12, math.pi / 2.0 + 0.12),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.03, 0.03),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.55, 1.15)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.45, 1.00)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.98, 1.02)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.98, 1.02)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.005, 0.005),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)

        self.sim2sim_action_hold_prob = 0.02
        self.sim2sim_action_noise_std = 0.0015

        self.disable_zero_weight_rewards()


@configclass
class Go2X5DogOnlyPctStairsHardEnvCfg(Go2X5DogOnlyPctStairsEnvCfg):
    """Train only on full-height first-flight PCT stairs without difficulty levels."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = PCT_STAIRS_HARD_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 0
        self.commands.base_velocity.minimum_total_rise = 1.57
        self.commands.base_velocity.maximum_total_rise = 1.57
        self.commands.base_velocity.terrain_num_rows = 1


@configclass
class Go2X5DogOnlyPctStairsFirstStepsEnvCfg(Go2X5DogOnlyPctStairsHardEnvCfg):
    """Bootstrap the first measured rise segment on the unchanged full-height scan."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.scene.terrain.terrain_generator = PCT_STAIRS_FIRST_STEPS_TERRAIN_CFG.copy()
        self.commands.base_velocity.path_points_xy = PCT_FIRST_STEPS_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = PCT_FIRST_STEPS_PATH_HEIGHT_FRACTIONS
        self.commands.base_velocity.minimum_total_rise = PCT_FIRST_STEPS_TARGET_RISE
        self.commands.base_velocity.maximum_total_rise = PCT_FIRST_STEPS_TARGET_RISE
        self.commands.base_velocity.waypoint_tolerance = 0.08
        self.commands.base_velocity.goal_tolerance = 0.12


@configclass
class Go2X5DogOnlyPctStairsFirstRiseEnvCfg(Go2X5DogOnlyPctStairsFirstStepsEnvCfg):
    """Bootstrap the first non-zero measured scan rise before extending the goal."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0
        self.commands.base_velocity.path_points_xy = PCT_FIRST_RISE_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = PCT_FIRST_RISE_PATH_HEIGHT_FRACTIONS
        self.commands.base_velocity.minimum_total_rise = PCT_FIRST_RISE_TARGET
        self.commands.base_velocity.maximum_total_rise = PCT_FIRST_RISE_TARGET


@configclass
class Go2X5DogOnlyPctStairsFirstRiseExactEnvCfg(Go2X5DogOnlyPctStairsFirstRiseEnvCfg):
    """Use the deployed local collision volume and require a stable first rise."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.terrain.terrain_generator = PCT_STAIRS_FIRST_RISE_EXACT_TERRAIN_CFG.copy()
        self.commands.base_velocity.minimum_total_rise = PCT_FIRST_RISE_EXACT_TARGET
        self.commands.base_velocity.maximum_total_rise = PCT_FIRST_RISE_EXACT_TARGET
        self.commands.base_velocity.completion_hold_steps = 25


@configclass
class Go2X5DogOnlyPctStairsFirstRiseExactScan1mEnvCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactEnvCfg
):
    """Use the exact collision volume with PCT's floor-local ray origin."""

    def __post_init__(self):
        super().__post_init__()

        # The exact local volume contains upper-floor geometry.  A 20 m ray
        # origin sees those surfaces before the stair under the robot, while
        # the deployed PCT profile deliberately scans from within one storey.
        self.scene.height_scanner.offset.pos = (0.0, 0.0, 1.0)
        self.scene.height_scanner_base.offset.pos = (0.0, 0.0, 1.0)


@configclass
class Go2X5DogOnlyPctStairsFirstRiseExactHighStepEnvCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactScan1mEnvCfg
):
    """Add swing-foot clearance above the measured 0.15 m first riser."""

    def __post_init__(self):
        super().__post_init__()

        # With the nominal 0.30 m base height, -0.13 m leaves only about
        # 0.02 m above the measured first rise.  Target 0.23 m world clearance
        # during swing, giving roughly 0.08 m margin without changing rewards.
        self.rewards.feet_height_body.params["target_height"] = -0.07


@configclass
class Go2X5DogOnlyPctStairsFirstStepsExactHighStepEnvCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactHighStepEnvCfg
):
    """Extend the learned exact first rise to the measured 1.05 m segment."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.commands.base_velocity.path_points_xy = PCT_FIRST_STEPS_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = PCT_FIRST_STEPS_PATH_HEIGHT_FRACTIONS
        self.commands.base_velocity.minimum_total_rise = PCT_FIRST_STEPS_TARGET_RISE
        self.commands.base_velocity.maximum_total_rise = PCT_FIRST_STEPS_TARGET_RISE


@configclass
class Go2X5DogOnlyPctStairsSecondRiseExactHighStepEnvCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactHighStepEnvCfg
):
    """Bridge the exact first rise to the next measured surface breakpoint."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.path_points_xy = PCT_SECOND_RISE_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = PCT_SECOND_RISE_PATH_HEIGHT_FRACTIONS
        self.commands.base_velocity.minimum_total_rise = PCT_SECOND_RISE_TARGET
        self.commands.base_velocity.maximum_total_rise = PCT_SECOND_RISE_TARGET


@configclass
class Go2X5DogOnlyPctStairsSecondRiseExactSlowEnvCfg(
    Go2X5DogOnlyPctStairsSecondRiseExactHighStepEnvCfg
):
    """Match the deployed PCT approach speed while learning the second rise."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.forward_velocity = 0.18


@configclass
class Go2X5DogOnlyPctStairsFullFlightExactSlowEnvCfg(
    Go2X5DogOnlyPctStairsSecondRiseExactSlowEnvCfg
):
    """Train the complete real PCT first flight at the deployed approach speed."""

    def __post_init__(self):
        super().__post_init__()

        # 4.60 m at 0.18 m/s needs about 26 s even without contact delays.
        # Keep enough time for deliberate stepping and the 0.5 s completion dwell.
        self.episode_length_s = 40.0
        self.commands.base_velocity.path_points_xy = PCT_STRAIGHT_STAIR_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = (
            PCT_STRAIGHT_STAIR_PATH_HEIGHT_FRACTIONS
        )
        self.commands.base_velocity.minimum_total_rise = 1.57
        self.commands.base_velocity.maximum_total_rise = 1.57

        # V5.7 showed that the inherited objective can make standing at the
        # first riser locally profitable. Rebalance only the complete-flight
        # task around safe physical progress; bootstrap tasks stay unchanged.
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.upward.weight = 0.2
        self.rewards.command_direction_progress.weight = 2.0
        self.rewards.commanded_stall_penalty.weight = -3.0
        self.rewards.pct_path_progress.weight = 8.0
        self.rewards.pct_height_alignment.weight = 2.0
        self.rewards.pct_cross_track_alignment.weight = 1.0
        self.rewards.pct_path_completion.weight = 250.0


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledSlowEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightExactSlowEnvCfg
):
    """Use the measured non-linear centerline heights of the real PCT flight."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.path_points_xy = PCT_PROFILED_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = (
            PCT_PROFILED_PATH_HEIGHT_FRACTIONS
        )


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledUprightEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledSlowEnvCfg
):
    """Reward path progress only inside the real PCT 20-degree posture gate."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.pct_path_progress.params["minimum_upright_projection"] = math.cos(
            math.radians(20.0)
        )


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledUprightEnvCfg
):
    """Make stable recovery preferable to ending an episode by falling."""

    def __post_init__(self):
        super().__post_init__()

        # V6.2 learned to reach the first riser and then terminate at 32 degrees
        # in essentially every episode.  A hard progress gate alone gives no
        # gradient for recovering posture, and termination avoids future stall
        # costs.  Keep the real safety gates, but make tilt continuously costly
        # and abnormal termination explicitly worse than remaining upright.
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-500.0)


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledDeploymentSpeedEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalEnvCfg
):
    """Train the measured full flight at the real no-Float PCT command speed."""

    def __post_init__(self):
        super().__post_init__()

        # StairLocomotionExecutor uses pct_carry_max_linear_velocity=0.25 m/s
        # in the real no-Float PCT gate.  V5.7--V6.5 trained at 0.18 m/s,
        # leaving the policy to absorb a 39% command increase only at runtime.
        self.commands.base_velocity.forward_velocity = 0.25


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledDeploymentSpeedEnvCfg
):
    """Teach the rear feet to catch up and support continued stair progress."""

    def __post_init__(self):
        super().__post_init__()

        rear_foot_names = ["RL_foot", "RR_foot"]
        self.rewards.pct_rear_foot_support = RewTerm(
            func=mdp.PctRearFootSupportReward,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "maximum_progress_lag": 0.38,
                "progress_lag_std": 0.12,
                "maximum_height_lag": 0.10,
                "height_lag_std": 0.06,
                "activation_progress": 0.55,
                "activation_width": 0.15,
                "contact_force_threshold": 5.0,
                "command_threshold": 0.08,
                "target_speed": 0.25,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=rear_foot_names, preserve_order=True
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=rear_foot_names, preserve_order=True
                ),
            },
        )


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledStableCompletionEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportEnvCfg
):
    """Consolidate full-flight progress with denser posture and contact costs."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = -8.0
        self.rewards.undesired_contacts.weight = -4.0
        self.rewards.pct_base_clearance_deficit.weight = -6.0
        self.rewards.pct_path_completion.weight = 500.0


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledTopLandingEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportEnvCfg
):
    """Cover the last measured risers while preserving full bottom-start episodes."""

    def __post_init__(self):
        super().__post_init__()

        # A safe real-PCT ascent naturally reaches about 29 degrees relative to
        # world level.  Keep the 20-degree completion gate and 32-degree failure
        # limit, but do not zero valid progress reward on the active flight.
        self.rewards.pct_path_progress.params["minimum_upright_projection"] = math.cos(
            math.radians(30.0)
        )
        self.rewards.pct_path_completion.weight = 500.0

        self.events.randomize_reset_base.func = mdp.reset_root_state_along_pct_stair_path
        self.events.randomize_reset_base.params = {
            "path_progress_height_anchors": PCT_PROFILED_TOP_LANDING_SPAWN_ANCHORS,
            "bottom_start_fraction": 0.40,
            "path_start_offset": 0.05,
            "mid_stair_pitch": -math.radians(18.0),
            "base_clearance": 0.08,
            "lateral_offset_range": (-0.04, 0.04),
            "forward_offset_range": (-0.02, 0.02),
            "height_offset_range": (0.00, 0.02),
            "roll_range": (-0.02, 0.02),
            "pitch_jitter_range": (-0.02, 0.02),
            "yaw_jitter_range": (math.pi / 2.0 - 0.06, math.pi / 2.0 + 0.06),
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportEnvCfg
):
    """Prioritize bottom-to-platform progress and stop at the real PCT landing gate."""

    def __post_init__(self):
        super().__post_init__()

        # The real first-flight gate is 3.902 m, while the training centerline
        # extends to 4.60 m.  Stop and dwell on the platform instead of asking
        # the policy to continue to the inherited 94% target (4.324 m).
        self.commands.base_velocity.completion_progress_ratio = (
            PCT_PLATFORM_PROGRESS_M / PCT_PROFILED_PATH_LENGTH_M
        )
        self.commands.base_velocity.completion_height_ratio = 0.985
        self.commands.base_velocity.completion_hold_steps = 25

        self.rewards.command_direction_progress.weight = 3.0
        self.rewards.pct_path_progress.weight = 12.0
        self.rewards.pct_path_progress.params["minimum_upright_projection"] = math.cos(
            math.radians(30.0)
        )
        self.rewards.pct_path_completion.weight = 1000.0

        # Treat 35--50 N as a reported minor-contact warning.  Meaningful body
        # impacts remain gated at 50 N, far below the rejected 700--1400 N runs.
        accepted_contact_force = 50.0
        self.rewards.pct_path_progress.params["contact_force_threshold"] = (
            accepted_contact_force
        )
        self.rewards.pct_unsafe_contact.params["threshold"] = accepted_contact_force
        self.rewards.pct_path_completion.params["contact_force_threshold"] = (
            accepted_contact_force
        )
        self.terminations.illegal_contact.params["threshold"] = accepted_contact_force
        self.terminations.pct_path_completed.params["contact_force_threshold"] = (
            accepted_contact_force
        )
        self.curriculum.terrain_levels.params["contact_force_threshold"] = (
            accepted_contact_force
        )


@configclass
class Go2X5DogOnlyPctRegularStairsEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressEnvCfg
):
    """Evaluate the PCT policy on a noise-free flight with measured outer dimensions."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = PCT_REGULAR_STAIRS_TERRAIN_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.path_points_xy = PCT_REGULAR_STAIR_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = (
            PCT_REGULAR_STAIR_PATH_HEIGHT_FRACTIONS
        )
        self.commands.base_velocity.minimum_total_rise = PCT_REGULAR_STAIR_FLIGHT_RISE_M
        self.commands.base_velocity.maximum_total_rise = PCT_REGULAR_STAIR_FLIGHT_RISE_M
        self.commands.base_velocity.terrain_num_rows = 1
        self.commands.base_velocity.completion_progress_ratio = (
            PCT_REGULAR_STAIR_PLATFORM_GATE_M / PCT_REGULAR_STAIR_PATH_LENGTH_M
        )


@configclass
class Go2X5DogOnlyPctRegularAscentCurriculumEnvCfg(
    Go2X5DogOnlyPctRegularStairsEnvCfg
):
    """Raise regular risers from 0.08 m toward the fixed 0.157 m ascent."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 512
        self.episode_length_s = 30.0
        self.scene.terrain.terrain_generator = (
            PCT_REGULAR_ASCENT_CURRICULUM_TERRAIN_CFG.copy()
        )
        self.scene.terrain.max_init_terrain_level = 0

        self.commands.base_velocity.minimum_total_rise = (
            PCT_REGULAR_ASCENT_MIN_RISER_M * PCT_REGULAR_STAIR_COUNT
        )
        self.commands.base_velocity.maximum_total_rise = PCT_REGULAR_STAIR_FLIGHT_RISE_M
        self.commands.base_velocity.terrain_num_rows = 10

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.pct_stair_completion_levels,
            params={
                "command_name": "base_velocity",
                "contact_force_threshold": 50.0,
                "base_head_arm_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["^(base|Head_.*|arm_.*)$"]
                ),
                "hip_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_hip"]
                ),
                "thigh_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_thigh"]
                ),
                "required_consecutive_successes": 3,
                "move_down_progress_ratio": 0.55,
                "minimum_upright_projection": math.cos(math.radians(20.0)),
            },
        )


@configclass
class Go2X5DogOnlyPctRegularAscentRepairEnvCfg(
    Go2X5DogOnlyPctRegularAscentCurriculumEnvCfg
):
    """Repair exact ascent without a dead zone before the formal tilt limit."""

    def __post_init__(self):
        super().__post_init__()

        # The source unified task allowed substantially larger tilt, while the
        # fixed-stair evaluator terminates at 32 degrees.  The inherited exact
        # task already supplies a continuous posture cost and a failure cost;
        # keep both and let safe path progress remain active up to that same
        # formal boundary instead of disappearing at the inherited 30 degrees.
        formal_tilt_limit = math.radians(32.0)
        self.rewards.pct_path_progress.params["minimum_upright_projection"] = math.cos(
            formal_tilt_limit
        )
        self.terminations.bad_orientation.params["limit_angle"] = formal_tilt_limit
        self.terminations.root_ang_vel_xy_above_maximum.params["maximum_speed"] = 4.0


@configclass
class Go2X5DogOnlyPctRegularAscentSim2RealEnvCfg(
    Go2X5DogOnlyPctRegularAscentRepairEnvCfg
):
    """Consolidate exact ascent under measured Sim2Real uncertainty."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = (
            PCT_REGULAR_ASCENT_SIM2REAL_TERRAIN_CFG.copy()
        )
        self.scene.terrain.max_init_terrain_level = 4

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (
            0.45,
            1.25,
        )
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (
            0.35,
            1.10,
        )
        self.events.randomize_rigid_body_material.params["restitution_range"] = (
            0.0,
            0.20,
        )
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (
            0.90,
            1.10,
        )
        self.events.randomize_rigid_body_mass_others.params[
            "mass_distribution_params"
        ] = (0.95, 1.05)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.015, 0.015),
            "y": (-0.015, 0.015),
            "z": (-0.015, 0.015),
        }
        self.events.randomize_actuator_gains.params[
            "stiffness_distribution_params"
        ] = (0.85, 1.15)
        self.events.randomize_actuator_gains.params[
            "damping_distribution_params"
        ] = (0.85, 1.15)
        self.events.randomize_push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 20.0),
            params={"velocity_range": {"x": (-0.12, 0.12), "y": (-0.12, 0.12)}},
        )

        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.12, n_max=0.12)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.04, n_max=0.04)
        self.observations.policy.projected_gravity.noise = Unoise(
            n_min=-0.03, n_max=0.03
        )
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.015, n_max=0.015)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-2.0, n_max=2.0)
        self.observations.policy.height_scan.noise = Unoise(n_min=-0.12, n_max=0.12)

        self.sim2sim_obs_delay_steps = 1
        delayed_sensor_terms = (
            ("base_lin_vel", mdp.delayed_base_lin_vel),
            ("base_ang_vel", mdp.delayed_base_ang_vel),
            ("projected_gravity", mdp.delayed_projected_gravity),
            ("joint_pos", mdp.delayed_joint_pos_rel),
            ("joint_vel", mdp.delayed_joint_vel_rel),
            ("height_scan", mdp.delayed_height_scan),
        )
        for term_name, func in delayed_sensor_terms:
            term = getattr(self.observations.policy, term_name)
            term.func = func
            if term.params is None:
                term.params = {}
            term.params["delay_steps"] = self.sim2sim_obs_delay_steps

        self.sim2sim_action_delay_range = (1, 2)
        self.sim2sim_action_hold_prob = 0.04
        self.sim2sim_action_noise_std = 0.003


@configclass
class Go2X5DogOnlyPctRegularUpDownStairsEnvCfg(
    Go2X5DogOnlyPctRegularStairsEnvCfg
):
    """Require a complete regular-box ascent, top crossing, and descent."""

    def __post_init__(self):
        super().__post_init__()

        # At 0.25 m/s the ideal traversal takes about 34 seconds.  Sixty
        # seconds leaves enough time for careful foot placement on both flights.
        self.episode_length_s = 60.0
        self.scene.terrain.terrain_generator = PCT_REGULAR_UP_DOWN_STAIRS_TERRAIN_CFG.copy()

        self.commands.base_velocity.path_points_xy = PCT_REGULAR_UP_DOWN_PATH_POINTS_XY
        self.commands.base_velocity.path_height_fractions = (
            PCT_REGULAR_UP_DOWN_PATH_HEIGHT_FRACTIONS
        )
        self.commands.base_velocity.completion_progress_ratio = (
            PCT_REGULAR_UP_DOWN_COMPLETION_GATE_M
            / PCT_REGULAR_UP_DOWN_PATH_LENGTH_M
        )
        self.commands.base_velocity.completion_height_ratio = 0.95
        self.commands.base_velocity.completion_return_height_tolerance = 0.12
        self.commands.base_velocity.completion_peak_height_ratio = 0.95
        self.commands.base_velocity.completion_hold_steps = 25

        # This term was introduced specifically to make rear feet catch up
        # during a one-way ascent.  The bidirectional route already uses path
        # progress and height alignment, so remove the asymmetric shaping term.
        self.rewards.pct_rear_foot_support = None


@configclass
class Go2X5DogOnlyPctRegularDescentStartEnvCfg(
    Go2X5DogOnlyPctRegularUpDownStairsEnvCfg
):
    """Train the exact descending flight from a stable top-platform reset."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 512
        self.episode_length_s = 35.0
        self.events.randomize_reset_base.func = mdp.reset_root_state_along_pct_stair_path
        self.events.randomize_reset_base.params = {
            "path_progress_height_anchors": (
                (
                    PCT_REGULAR_UP_DOWN_DESCENT_START_M - 0.55,
                    PCT_REGULAR_STAIR_FLIGHT_RISE_M,
                ),
            ),
            "bottom_start_fraction": 0.0,
            "path_start_offset": 0.0,
            "mid_stair_pitch": 0.0,
            "base_clearance": 0.02,
            "lateral_offset_range": (-0.04, 0.04),
            "forward_offset_range": (-0.04, 0.04),
            "height_offset_range": (0.0, 0.02),
            "roll_range": (-0.02, 0.02),
            "pitch_jitter_range": (-0.02, 0.02),
            "yaw_jitter_range": (
                math.pi / 2.0 - 0.06,
                math.pi / 2.0 + 0.06,
            ),
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledPlatformEntryEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressEnvCfg
):
    """Continue 0.25 m beyond the first-platform gate before dwelling."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.completion_progress_ratio = (
            PCT_PLATFORM_INTERIOR_PROGRESS_M / PCT_PROFILED_PATH_LENGTH_M
        )


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledCoverageEnvCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalEnvCfg
):
    """Expose one policy to bottom starts and the full measured stair flight."""

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_reset_base.func = mdp.reset_root_state_along_pct_stair_path
        self.events.randomize_reset_base.params = {
            "path_progress_height_anchors": PCT_PROFILED_MID_STAIR_SPAWN_ANCHORS,
            "bottom_start_fraction": 0.25,
            "path_start_offset": 0.05,
            "mid_stair_pitch": -math.radians(18.0),
            "base_clearance": 0.08,
            "lateral_offset_range": (-0.04, 0.04),
            "forward_offset_range": (-0.02, 0.02),
            "height_offset_range": (0.00, 0.02),
            "roll_range": (-0.02, 0.02),
            "pitch_jitter_range": (-0.02, 0.02),
            "yaw_jitter_range": (math.pi / 2.0 - 0.06, math.pi / 2.0 + 0.06),
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.bad_orientation.func = mdp.bad_orientation_after_steps
        self.terminations.bad_orientation.params = {
            "limit_angle": math.radians(32.0),
            "minimum_episode_steps": 200,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.illegal_contact.params["minimum_episode_steps"] = 200
