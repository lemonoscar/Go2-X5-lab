# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def reward_weights_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    p1_weights: dict,
    p2_weights: dict,
    curriculum_iterations: int = 2000,
) -> None:
    """Gradually transition reward weights from Phase 1 (Foundation Flat) to Phase 2 (Robust Rough) values.

    This curriculum function linearly interpolates reward weights from P1 values to P2 values
    over a specified number of training iterations. This helps smooth the transition when
    fine-tuning a model trained on flat terrain to rough terrain.

    Args:
        env: The learning environment.
        env_ids: Environment IDs (unused, kept for curriculum interface compatibility).
        p1_weights: Dictionary of reward weights from Phase 1 (Foundation Flat).
        p2_weights: Dictionary of target reward weights for Phase 2 (Robust Rough).
        curriculum_iterations: Number of training iterations to complete the transition.
    """
    # Calculate current progress based on iteration count (if available)
    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)

    # Initialize tracking variables on first call
    if not hasattr(env, "_reward_curriculum_initialized"):
        env._reward_curriculum_initialized = True
        env._reward_curriculum_start_iter = current_iter
        env._reward_curriculum_p1_weights = p1_weights
        env._reward_curriculum_p2_weights = p2_weights
        env._reward_curriculum_total_iters = curriculum_iterations

    # Calculate progress (0.0 = P1 weights, 1.0 = P2 weights)
    progress = min((current_iter - env._reward_curriculum_start_iter) / env._reward_curriculum_total_iters, 1.0)

    # Update reward weights based on current progress
    for attr_name, p1_weight in p1_weights.items():
        if attr_name not in p2_weights:
            continue
        p2_weight = p2_weights[attr_name]

        # Linear interpolation
        current_weight = p1_weight + (p2_weight - p1_weight) * progress

        # Update the reward weight
        if hasattr(env.reward_manager, "_term_names"):
            term_names = env.reward_manager._term_names
            if attr_name in term_names:
                reward_term = env.reward_manager.get_term(attr_name)
                if hasattr(reward_term, "cfg"):
                    reward_term.cfg.weight = current_weight
                elif hasattr(reward_term, "weight"):
                    reward_term.weight = current_weight

    return progress
