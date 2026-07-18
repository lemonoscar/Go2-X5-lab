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

from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain

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


def stratified_vx_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    reward_term_name: str,
    steps_per_iteration: int = 32,
    full_range_iteration: int = 2000,
    performance_threshold: float = 0.75,
) -> dict[str, torch.Tensor]:
    """Unlock vx bins on good tracking and guarantee the full range by phase C2."""
    command_term = env.command_manager.get_term(command_name)
    iteration = int(env.common_step_counter // max(steps_per_iteration, 1))
    total_speed_count = len(command_term.speed_values)

    if iteration >= full_range_iteration:
        command_term.set_active_speed_count(total_speed_count)
    elif (
        command_term.active_speed_count < total_speed_count
        and iteration - command_term.last_speed_promotion_iteration
        >= command_term.cfg.promotion_interval_iterations
    ):
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        normalized_reward = torch.mean(episode_sums[ids]) / max(env.max_episode_length_s, 1.0e-6)
        target_reward = performance_threshold * max(float(reward_cfg.weight), 1.0e-6)
        if normalized_reward >= target_reward:
            command_term.set_active_speed_count(command_term.active_speed_count + 1)
            command_term.last_speed_promotion_iteration = iteration

    return {
        "active_speed_bins": torch.tensor(command_term.active_speed_count, device=env.device),
        "active_max_vx": torch.tensor(command_term.active_max_speed, device=env.device),
        "full_range_rehearsal_probability": torch.tensor(
            command_term.cfg.full_range_rehearsal_probability, device=env.device
        ),
        "training_iteration": torch.tensor(iteration, device=env.device),
    }


def rough_stairs_vx_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steps_per_iteration: int = 32,
    full_difficulty_iteration: int = 2000,
    move_up_distance_ratio: float = 0.25,
    move_down_command_ratio: float = 0.25,
    move_down_min_distance: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Terrain curriculum with a low-rise warm-up during the first 2000 updates."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        zero = torch.tensor(0.0, device=env.device)
        return {"terrain_level": zero, "terrain_level_cap": zero}

    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    iteration = int(env.common_step_counter // max(steps_per_iteration, 1))
    maximum_level = terrain.cfg.terrain_generator.num_rows - 1
    warmup_ratio = min(iteration / max(full_difficulty_iteration, 1), 1.0)
    level_cap = max(1, round(warmup_ratio * maximum_level))

    distance = torch.linalg.norm(
        asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1
    )
    command = env.command_manager.get_command("base_velocity")
    commanded_distance = torch.linalg.norm(command[ids, :2], dim=1) * env.max_episode_length_s
    move_down_distance = torch.maximum(
        commanded_distance * move_down_command_ratio,
        torch.full_like(commanded_distance, move_down_min_distance),
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] * move_up_distance_ratio
    move_up &= terrain.terrain_levels[ids] < level_cap
    move_down = (distance < move_down_distance) & ~move_up
    terrain.update_env_origins(ids, move_up, move_down)

    return {
        "terrain_level": torch.mean(terrain.terrain_levels.float()),
        "terrain_level_cap": torch.tensor(level_cap, device=env.device),
        "training_iteration": torch.tensor(iteration, device=env.device),
    }


def terrain_levels_vel_hard(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_distance_ratio: float = 0.35,
    move_down_command_ratio: float = 0.25,
    move_down_min_distance: float = 0.8,
) -> torch.Tensor:
    """Terrain curriculum biased toward harder rows for high-difficulty rough training.

    The stock velocity curriculum requires roughly half a terrain tile of progress before
    moving up. This variant promotes earlier on obstacle-heavy tiles and only demotes an
    environment when it barely leaves its origin relative to the commanded distance.
    """

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if terrain.terrain_origins is None:
        return torch.tensor(0.0, device=env.device)

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    tile_length = terrain.cfg.terrain_generator.size[0]
    commanded_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    move_down_distance = torch.maximum(
        commanded_distance * move_down_command_ratio,
        torch.full_like(commanded_distance, move_down_min_distance),
    )

    move_up = distance > tile_length * move_up_distance_ratio
    move_down = distance < move_down_distance
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def pct_stair_completion_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    contact_force_threshold: float,
    base_head_arm_sensor_cfg: SceneEntityCfg,
    hip_sensor_cfg: SceneEntityCfg,
    thigh_sensor_cfg: SceneEntityCfg,
    required_consecutive_successes: int = 1,
    move_down_progress_ratio: float = 0.55,
    minimum_upright_projection: float = 0.94,
) -> dict[str, torch.Tensor]:
    """Advance PCT terrain only after a stable, contact-free path completion."""
    if required_consecutive_successes < 1:
        raise ValueError("required_consecutive_successes must be at least one.")

    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        zero = torch.tensor(0.0, device=env.device)
        return {
            "terrain_level": zero,
            "completion_rate": zero,
            "path_progress_ratio": zero,
            "height_ratio": zero,
            "scan_completion_rate": zero,
            "scan_path_progress_ratio": zero,
            "scan_height_ratio": zero,
            "procedural_completion_rate": zero,
            "promotion_rate": zero,
            "success_streak_mean": zero,
            "base_head_arm_contact_rate": zero,
            "hip_contact_rate": zero,
            "thigh_contact_rate": zero,
        }

    command_term = env.command_manager.get_term(command_name)
    progress_ratio = command_term.path_progress_ratio[env_ids]
    target_height = torch.clamp(command_term.target_height_gain_m[env_ids], min=1.0e-6)
    height_ratio = command_term.height_gain_m[env_ids] / target_height
    upright = -env.scene["robot"].data.projected_gravity_b[env_ids, 2] >= minimum_upright_projection

    reset_time_outs = getattr(env, "reset_time_outs", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    reset_terminated = getattr(
        env, "reset_terminated", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    successful_completion = command_term.path_completed[env_ids].clone()
    successful_completion &= reset_time_outs[env_ids]
    successful_completion &= ~reset_terminated[env_ids]
    successful_completion &= upright

    success_streak = getattr(env, "_pct_stair_success_streak", None)
    if success_streak is None:
        success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._pct_stair_success_streak = success_streak
    updated_streak = torch.where(
        successful_completion,
        success_streak[env_ids] + 1,
        torch.zeros_like(success_streak[env_ids]),
    )
    move_up = successful_completion & (updated_streak >= required_consecutive_successes)
    success_streak[env_ids] = torch.where(move_up, torch.zeros_like(updated_streak), updated_streak)

    move_down = ~successful_completion
    move_down &= torch.logical_or(
        progress_ratio < move_down_progress_ratio,
        reset_terminated[env_ids],
    )
    terrain.update_env_origins(env_ids, move_up, move_down)

    scan_mask = is_env_assigned_to_terrain(env, "pct_scanned_first_flight")[env_ids]
    procedural_mask = ~scan_mask

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype)
        return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1.0)

    contact_sensor = env.scene.sensors[base_head_arm_sensor_cfg.name]

    def _contact_rate(sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        forces = contact_sensor.data.net_forces_w_history[env_ids][:, :, sensor_cfg.body_ids, :]
        maximum_force = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))
        return torch.mean((maximum_force > contact_force_threshold).to(torch.float32))

    return {
        "terrain_level": torch.mean(terrain.terrain_levels.float()),
        "completion_rate": torch.mean(successful_completion.to(torch.float32)),
        "path_progress_ratio": torch.mean(progress_ratio),
        "height_ratio": torch.mean(height_ratio),
        "scan_completion_rate": _masked_mean(successful_completion.float(), scan_mask),
        "scan_path_progress_ratio": _masked_mean(progress_ratio, scan_mask),
        "scan_height_ratio": _masked_mean(height_ratio, scan_mask),
        "procedural_completion_rate": _masked_mean(successful_completion.float(), procedural_mask),
        "promotion_rate": torch.mean(move_up.to(torch.float32)),
        "success_streak_mean": torch.mean(success_streak[env_ids].to(torch.float32)),
        "base_head_arm_contact_rate": _contact_rate(base_head_arm_sensor_cfg),
        "hip_contact_rate": _contact_rate(hip_sensor_cfg),
        "thigh_contact_rate": _contact_rate(thigh_sensor_cfg),
    }


def arm_joint_position_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    initial_position_range: Sequence[Sequence[float]],
    final_position_range: Sequence[Sequence[float]],
    curriculum_iterations: int = 2000,
) -> None:
    """Linearly expand arm joint command ranges over training.

    This is used for arm-unlock stages that resume from a checkpoint trained with the arm fixed at
    its default pose. The network interface stays unchanged while the arm command distribution is
    widened gradually.
    """

    del env_ids

    if len(initial_position_range) != len(final_position_range):
        raise ValueError("initial_position_range and final_position_range must have the same length.")

    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)
    state = getattr(env, "_arm_joint_range_curriculum_state", None)
    if state is None:
        state = {}
        env._arm_joint_range_curriculum_state = state

    if command_name not in state:
        state[command_name] = {
            "start_iter": current_iter,
            "initial_position_range": [tuple(bounds) for bounds in initial_position_range],
            "final_position_range": [tuple(bounds) for bounds in final_position_range],
            "total_iters": curriculum_iterations,
        }

    cfg = env.command_manager.get_term(command_name).cfg
    command_state = state[command_name]
    progress = min(
        (current_iter - command_state["start_iter"]) / max(command_state["total_iters"], 1),
        1.0,
    )

    current_position_range = []
    for init_bounds, final_bounds in zip(
        command_state["initial_position_range"], command_state["final_position_range"], strict=True
    ):
        lower = init_bounds[0] + (final_bounds[0] - init_bounds[0]) * progress
        upper = init_bounds[1] + (final_bounds[1] - init_bounds[1]) * progress
        current_position_range.append((float(lower), float(upper)))

    cfg.position_range = current_position_range
    return progress


def arm_joint_position_range_staged_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    position_ranges: Sequence[Sequence[Sequence[float]]],
    stage_iterations: Sequence[int],
) -> None:
    """Interpolate arm joint command ranges across multiple curriculum stages.

    Args:
        env: The learning environment.
        env_ids: Environment IDs (unused, kept for curriculum interface compatibility).
        command_name: Command term name.
        position_ranges: A sequence of per-joint position ranges. Each element defines one stage.
        stage_iterations: Cumulative iteration anchors for the corresponding stages. The first entry
            should typically be 0. For example, with 3 stages and ``[0, 48, 128]``, the curriculum
            interpolates stage 1 -> stage 2 over iterations [0, 48], then stage 2 -> stage 3 over
            iterations [48, 128].
    """

    del env_ids

    if len(position_ranges) != len(stage_iterations):
        raise ValueError("position_ranges and stage_iterations must have the same length.")
    if len(position_ranges) < 2:
        raise ValueError("At least two stages are required for staged arm position curriculum.")
    if any(stage_iterations[idx] > stage_iterations[idx + 1] for idx in range(len(stage_iterations) - 1)):
        raise ValueError("stage_iterations must be non-decreasing.")

    joint_count = len(position_ranges[0])
    if any(len(stage_range) != joint_count for stage_range in position_ranges):
        raise ValueError("All staged arm position ranges must have the same joint count.")

    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)
    cfg = env.command_manager.get_term(command_name).cfg

    if current_iter <= stage_iterations[0]:
        cfg.position_range = [tuple(bounds) for bounds in position_ranges[0]]
        return 0.0

    if current_iter >= stage_iterations[-1]:
        cfg.position_range = [tuple(bounds) for bounds in position_ranges[-1]]
        return float(len(position_ranges) - 1)

    stage_idx = 1
    while current_iter > stage_iterations[stage_idx]:
        stage_idx += 1

    prev_iter = stage_iterations[stage_idx - 1]
    next_iter = stage_iterations[stage_idx]
    denom = max(next_iter - prev_iter, 1)
    progress = (current_iter - prev_iter) / denom

    current_position_range = []
    for prev_bounds, next_bounds in zip(position_ranges[stage_idx - 1], position_ranges[stage_idx], strict=True):
        lower = prev_bounds[0] + (next_bounds[0] - prev_bounds[0]) * progress
        upper = prev_bounds[1] + (next_bounds[1] - prev_bounds[1]) * progress
        current_position_range.append((float(lower), float(upper)))

    cfg.position_range = current_position_range
    return float(stage_idx - 1) + progress


def base_velocity_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    initial_lin_vel_x: Sequence[float],
    final_lin_vel_x: Sequence[float],
    initial_lin_vel_y: Sequence[float],
    final_lin_vel_y: Sequence[float],
    initial_ang_vel_z: Sequence[float],
    final_ang_vel_z: Sequence[float],
    curriculum_iterations: int = 6800,
) -> None:
    """Linearly widen the base velocity command range over PPO iterations."""

    del env_ids

    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)
    state = getattr(env, "_base_velocity_range_curriculum_state", None)
    if state is None:
        state = {
            "start_iter": current_iter,
            "total_iters": curriculum_iterations,
            "initial_lin_vel_x": tuple(initial_lin_vel_x),
            "final_lin_vel_x": tuple(final_lin_vel_x),
            "initial_lin_vel_y": tuple(initial_lin_vel_y),
            "final_lin_vel_y": tuple(final_lin_vel_y),
            "initial_ang_vel_z": tuple(initial_ang_vel_z),
            "final_ang_vel_z": tuple(final_ang_vel_z),
        }
        env._base_velocity_range_curriculum_state = state

    progress = min((current_iter - state["start_iter"]) / max(state["total_iters"], 1), 1.0)

    def _interpolate(initial_range, final_range):
        lower = initial_range[0] + (final_range[0] - initial_range[0]) * progress
        upper = initial_range[1] + (final_range[1] - initial_range[1]) * progress
        return (float(lower), float(upper))

    ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    ranges.lin_vel_x = _interpolate(state["initial_lin_vel_x"], state["final_lin_vel_x"])
    ranges.lin_vel_y = _interpolate(state["initial_lin_vel_y"], state["final_lin_vel_y"])
    ranges.ang_vel_z = _interpolate(state["initial_ang_vel_z"], state["final_ang_vel_z"])
    return progress


def reward_parameter_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    param_name: str,
    initial_value: float,
    final_value: float,
    curriculum_iterations: int = 800,
) -> None:
    """Linearly update one scalar reward parameter over PPO iterations."""

    del env_ids

    current_iter = getattr(env, "common_step_counter", 0) // getattr(env, "max_episode_length", 1)
    state = getattr(env, "_reward_parameter_curriculum_state", None)
    if state is None:
        state = {}
        env._reward_parameter_curriculum_state = state

    key = (reward_term_name, param_name)
    if key not in state:
        state[key] = {
            "start_iter": current_iter,
            "initial_value": float(initial_value),
            "final_value": float(final_value),
            "total_iters": curriculum_iterations,
        }

    term_state = state[key]
    progress = min((current_iter - term_state["start_iter"]) / max(term_state["total_iters"], 1), 1.0)
    current_value = term_state["initial_value"] + (term_state["final_value"] - term_state["initial_value"]) * progress

    if hasattr(env.reward_manager, "_term_names") and reward_term_name in env.reward_manager._term_names:
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        reward_term_cfg.params[param_name] = float(current_value)
        env.reward_manager.set_term_cfg(reward_term_name, reward_term_cfg)
    return float(current_value)


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

        # Update the reward weight through the public manager config API.
        if hasattr(env.reward_manager, "_term_names"):
            term_names = env.reward_manager._term_names
            if attr_name in term_names:
                reward_term_cfg = env.reward_manager.get_term_cfg(attr_name)
                reward_term_cfg.weight = current_weight
                env.reward_manager.set_term_cfg(attr_name, reward_term_cfg)

    return progress
