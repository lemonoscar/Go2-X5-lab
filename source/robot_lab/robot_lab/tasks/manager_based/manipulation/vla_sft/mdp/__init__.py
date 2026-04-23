# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP (Markov Decision Process) functions for VLA-SFT environments."""

from isaaclab.envs.mdp import *  # noqa: F401,F403

# Re-export MDP functions from ground_pick for compatibility
from ...ground_pick.mdp import *  # noqa: F401,F403

from .scene_randomization import *  # noqa: F401,F403
from .render_randomization import *  # noqa: F401,F403

# Explicit exports for vla_sft.configs
from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    AbsBinaryJointPositionActionCfg,
    action_rate_l2,
    joint_vel_l2,
    reset_root_state_uniform,
    reset_scene_to_default,
    root_height_below_minimum,
    time_out,
)

from ...ground_pick.mdp.observations import (
    object_position_in_robot_root_frame,
    object_height,
    ee_to_object_vector,
    gripper_opening,
)

from ...ground_pick.mdp.rewards import (
    object_ee_distance,
    gripper_closed_around_object,
    object_is_lifted,
    success_bonus,
    stable_base_bonus,
)

from ...ground_pick.mdp.terminations import ground_pick_success

__all__ = [
    # From isaaclab.envs.mdp
    "JointPositionActionCfg",
    "AbsBinaryJointPositionActionCfg",
    "action_rate_l2",
    "joint_vel_l2",
    "reset_root_state_uniform",
    "reset_scene_to_default",
    "root_height_below_minimum",
    "time_out",
    # From ground_pick.mdp.observations
    "object_position_in_robot_root_frame",
    "object_height",
    "ee_to_object_vector",
    "gripper_opening",
    # From ground_pick.mdp.rewards
    "object_ee_distance",
    "gripper_closed_around_object",
    "object_is_lifted",
    "success_bonus",
    "stable_base_bonus",
    # From ground_pick.mdp.terminations
    "ground_pick_success",
    # From scene_randomization
    "reset_object_pose_from_scene",
    "reset_object_color",
    "spawn_clutter_objects",
    "spawn_table_surface",
    "apply_floor_visual_material",
]
