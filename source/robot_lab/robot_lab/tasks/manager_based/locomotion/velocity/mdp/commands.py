# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_env_assigned_to_terrain, is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class StratifiedVxVelocityCommand(mdp.UniformVelocityCommand):
    """Sample exact forward-speed bins while keeping stair commands conservative.

    Flat and generic rough cells use the active prefix of ``speed_values``.  Stair
    cells always use the smaller stair-speed set so adding the 0.7 m/s flat-ground
    requirement does not turn stair traversal into a high-speed task.
    """

    cfg: "StratifiedVxVelocityCommandCfg"

    def __init__(self, cfg: "StratifiedVxVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not cfg.speed_values:
            raise ValueError("speed_values must contain at least one forward speed.")
        if not cfg.stair_speed_values:
            raise ValueError("stair_speed_values must contain at least one forward speed.")
        if tuple(sorted(cfg.speed_values)) != cfg.speed_values:
            raise ValueError("speed_values must be sorted in ascending order.")
        if cfg.speed_values[0] != 0.0:
            raise ValueError("speed_values must start at 0.0 so stopping is trained explicitly.")
        if not 0.0 <= cfg.full_range_rehearsal_probability <= 1.0:
            raise ValueError("full_range_rehearsal_probability must stay inside [0, 1].")

        self.speed_values = torch.tensor(cfg.speed_values, dtype=torch.float32, device=self.device)
        self.stair_speed_values = torch.tensor(
            cfg.stair_speed_values, dtype=torch.float32, device=self.device
        )
        self.active_speed_count = int(
            max(1, min(cfg.initial_active_speed_count, len(cfg.speed_values)))
        )
        # Hold the initial bins for one complete promotion window.  Starting at
        # ``-interval`` would allow the first good episode to unlock a bin at
        # update 1, defeating the intended C1 warm-up.
        self.last_speed_promotion_iteration = 0

        self.stair_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for terrain_name in cfg.stair_terrain_names:
            self.stair_env_mask |= is_env_assigned_to_terrain(env, terrain_name)

    @property
    def active_max_speed(self) -> float:
        return float(self.speed_values[self.active_speed_count - 1].item())

    def set_active_speed_count(self, count: int) -> None:
        """Set the number of active flat/rough speed bins."""
        self.active_speed_count = max(1, min(int(count), len(self.speed_values)))

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        flat_indices = torch.randint(
            low=0,
            high=self.active_speed_count,
            size=(len(env_ids_tensor),),
            device=self.device,
        )
        sampled_speeds = self.speed_values[flat_indices]

        local_stair_mask = self.stair_env_mask[env_ids_tensor]
        if (
            self.active_speed_count < len(self.speed_values)
            and self.cfg.full_range_rehearsal_probability > 0.0
        ):
            rehearsal_mask = (
                torch.rand(len(env_ids_tensor), device=self.device)
                < self.cfg.full_range_rehearsal_probability
            )
            rehearsal_mask &= ~local_stair_mask
            if torch.any(rehearsal_mask):
                full_range_indices = torch.randint(
                    low=0,
                    high=len(self.speed_values),
                    size=(int(rehearsal_mask.sum().item()),),
                    device=self.device,
                )
                sampled_speeds[rehearsal_mask] = self.speed_values[full_range_indices]

        if torch.any(local_stair_mask):
            stair_count = int(local_stair_mask.sum().item())
            stair_indices = torch.randint(
                low=0,
                high=len(self.stair_speed_values),
                size=(stair_count,),
                device=self.device,
            )
            sampled_speeds[local_stair_mask] = self.stair_speed_values[stair_indices]

        self.vel_command_b[env_ids_tensor] = 0.0
        self.vel_command_b[env_ids_tensor, 0] = sampled_speeds
        self.is_heading_env[env_ids_tensor] = False
        self.is_standing_env[env_ids_tensor] = sampled_speeds == 0.0


@configclass
class StratifiedVxVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for :class:`StratifiedVxVelocityCommand`."""

    class_type: type = StratifiedVxVelocityCommand
    speed_values: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
    )
    stair_speed_values: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
    stair_terrain_names: tuple[str, ...] = ("stairs_up", "stairs_down")
    initial_active_speed_count: int = 5
    promotion_interval_iterations: int = 250
    full_range_rehearsal_probability: float = 0.0


class PctStairVelocityCommand(CommandTerm):
    """Vectorized counterpart of the deployed PCT stair centerline tracker."""

    cfg: "PctStairVelocityCommandCfg"

    def __init__(self, cfg: "PctStairVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        if len(cfg.path_points_xy) < 2:
            raise ValueError("PctStairVelocityCommand requires at least two path points.")
        if len(cfg.path_points_xy) != len(cfg.path_height_fractions):
            raise ValueError("path_points_xy and path_height_fractions must have equal length.")
        if any(value < 0.0 or value > 1.0 for value in cfg.path_height_fractions):
            raise ValueError("path_height_fractions must stay inside [0, 1].")
        if cfg.stop_forward_heading_error <= cfg.full_speed_heading_error:
            raise ValueError("stop_forward_heading_error must exceed full_speed_heading_error.")
        if cfg.completion_hold_steps < 1:
            raise ValueError("completion_hold_steps must be at least one.")
        if (
            cfg.completion_return_height_tolerance is not None
            and cfg.completion_return_height_tolerance <= 0.0
        ):
            raise ValueError("completion_return_height_tolerance must be positive when enabled.")
        if not 0.0 <= cfg.completion_peak_height_ratio <= 1.0:
            raise ValueError("completion_peak_height_ratio must stay inside [0, 1].")

        self.path_points_xy = torch.tensor(cfg.path_points_xy, dtype=torch.float32, device=self.device)
        self.path_height_fractions = torch.tensor(
            cfg.path_height_fractions, dtype=torch.float32, device=self.device
        )
        segment_vectors = self.path_points_xy[1:] - self.path_points_xy[:-1]
        self.segment_lengths = torch.linalg.norm(segment_vectors, dim=1)
        if torch.any(self.segment_lengths <= 1.0e-6):
            raise ValueError("PctStairVelocityCommand path cannot contain duplicate XY points.")
        self.segment_tangents = segment_vectors / self.segment_lengths.unsqueeze(1)
        self.segment_headings = torch.atan2(segment_vectors[:, 1], segment_vectors[:, 0])
        self.cumulative_lengths = torch.cat(
            (torch.zeros(1, device=self.device), torch.cumsum(self.segment_lengths, dim=0))
        )
        self.path_length_m = float(self.cumulative_lengths[-1].item())

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.segment_index = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.path_progress_m = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_progress_m = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_progress_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cross_track_error_m = torch.zeros(self.num_envs, device=self.device)
        self.desired_heading_rad = torch.zeros(self.num_envs, device=self.device)
        self.height_gain_m = torch.zeros(self.num_envs, device=self.device)
        self.peak_height_gain_m = torch.zeros(self.num_envs, device=self.device)
        self.expected_height_gain_m = torch.zeros(self.num_envs, device=self.device)
        self.target_height_gain_m = torch.zeros(self.num_envs, device=self.device)
        self.path_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_candidate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["path_progress_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["path_progress_from_start_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["height_gain_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cross_track_error_m"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["path_completed"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "PctStairVelocityCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tPath points: {len(self.path_points_xy)}\n"
            f"\tPath length: {self.path_length_m:.3f} m\n"
            f"\tForward speed: {self.cfg.forward_velocity:.3f} m/s"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def path_progress_ratio(self) -> torch.Tensor:
        return self.path_progress_m / max(self.path_length_m, 1.0e-6)

    def _update_metrics(self):
        normalizer = max(float(self._env.max_episode_length), 1.0)
        self.metrics["error_vel_xy"] += torch.linalg.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=1
        ) / normalizer
        self.metrics["error_vel_yaw"] += torch.abs(
            self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        ) / normalizer
        self.metrics["path_progress_m"] = torch.maximum(
            self.metrics["path_progress_m"], self.path_progress_m
        )
        progress_from_start = torch.clamp(
            self.path_progress_m - self.episode_start_progress_m, min=0.0
        )
        self.metrics["path_progress_from_start_m"] = torch.maximum(
            self.metrics["path_progress_from_start_m"], progress_from_start
        )
        self.metrics["height_gain_m"] = torch.maximum(self.metrics["height_gain_m"], self.height_gain_m)
        self.metrics["cross_track_error_m"] += torch.abs(self.cross_track_error_m) / normalizer
        self.metrics["path_completed"] = torch.maximum(
            self.metrics["path_completed"], self.path_completed.to(torch.float32)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        self.segment_index[env_ids] = 1
        self.path_progress_m[env_ids] = 0.0
        self.episode_start_progress_m[env_ids] = 0.0
        self.episode_start_progress_initialized[env_ids] = False
        self.cross_track_error_m[env_ids] = 0.0
        self.height_gain_m[env_ids] = 0.0
        self.peak_height_gain_m[env_ids] = 0.0
        self.expected_height_gain_m[env_ids] = 0.0
        self.target_height_gain_m[env_ids] = self.cfg.minimum_total_rise
        self.path_completed[env_ids] = False
        self.completion_candidate[env_ids] = False
        self.completion_hold_count[env_ids] = 0
        self.vel_command_b[env_ids] = 0.0

    def _segment_projection(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_xy = self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
        segment_ids = self.segment_index - 1
        starts = self.path_points_xy[segment_ids]
        ends = self.path_points_xy[self.segment_index]
        tangents = self.segment_tangents[segment_ids]
        lengths = self.segment_lengths[segment_ids]
        relative = local_xy - starts
        projection = torch.sum(relative * tangents, dim=1) / lengths
        signed_cross_track = tangents[:, 0] * relative[:, 1] - tangents[:, 1] * relative[:, 0]
        endpoint_distance = torch.linalg.norm(local_xy - ends, dim=1)
        return projection, signed_cross_track, endpoint_distance

    def _update_path_state(self):
        last_segment_index = len(self.path_points_xy) - 1
        for _ in range(last_segment_index - 1):
            projection, _, endpoint_distance = self._segment_projection()
            can_advance = self.segment_index < last_segment_index
            can_advance &= torch.logical_or(
                projection >= self.cfg.segment_advance_ratio,
                endpoint_distance <= self.cfg.waypoint_tolerance,
            )
            self.segment_index += can_advance.to(torch.long)

        projection, signed_cross_track, _ = self._segment_projection()
        clamped_projection = torch.clamp(projection, 0.0, 1.0)
        segment_ids = self.segment_index - 1
        self.path_progress_m = (
            self.cumulative_lengths[segment_ids] + clamped_projection * self.segment_lengths[segment_ids]
        )
        new_episode_start = ~self.episode_start_progress_initialized
        self.episode_start_progress_m = torch.where(
            new_episode_start, self.path_progress_m, self.episode_start_progress_m
        )
        self.episode_start_progress_initialized |= new_episode_start
        self.cross_track_error_m = signed_cross_track
        self.desired_heading_rad = self.segment_headings[segment_ids]

        terrain_levels = getattr(self._env.scene.terrain, "terrain_levels", None)
        if terrain_levels is None:
            difficulty_lower = torch.zeros(self.num_envs, device=self.device)
        else:
            difficulty_lower = terrain_levels.to(torch.float32) / max(float(self.cfg.terrain_num_rows), 1.0)
        self.target_height_gain_m = self.cfg.minimum_total_rise + difficulty_lower * (
            self.cfg.maximum_total_rise - self.cfg.minimum_total_rise
        )
        start_height_fraction = self.path_height_fractions[segment_ids]
        end_height_fraction = self.path_height_fractions[self.segment_index]
        expected_fraction = start_height_fraction + clamped_projection * (
            end_height_fraction - start_height_fraction
        )
        self.expected_height_gain_m = expected_fraction * self.target_height_gain_m
        self.height_gain_m = (
            self.robot.data.root_pos_w[:, 2]
            - self._env.scene.env_origins[:, 2]
            - self.cfg.nominal_base_height
        )
        self.peak_height_gain_m = torch.maximum(self.peak_height_gain_m, self.height_gain_m)
        progress_gate = self.path_progress_ratio >= self.cfg.completion_progress_ratio
        if self.cfg.completion_return_height_tolerance is None:
            height_gate = (
                self.height_gain_m
                >= self.cfg.completion_height_ratio * self.target_height_gain_m
            )
        else:
            # Up-and-down routes finish near their final expected height, not
            # at the maximum route elevation.  Still require the robot to have
            # reached the top earlier so it cannot earn completion by bypassing
            # the staircase at ground level.
            returned_to_route_height = (
                torch.abs(self.height_gain_m - self.expected_height_gain_m)
                <= self.cfg.completion_return_height_tolerance
            )
            reached_route_peak = (
                self.peak_height_gain_m
                >= self.cfg.completion_peak_height_ratio * self.target_height_gain_m
            )
            height_gate = returned_to_route_height & reached_route_peak
        self.completion_candidate = progress_gate & height_gate
        self.completion_hold_count = torch.where(
            self.completion_candidate,
            self.completion_hold_count + 1,
            torch.zeros_like(self.completion_hold_count),
        )
        self.path_completed = self.completion_hold_count >= self.cfg.completion_hold_steps

    def _update_command(self):
        self._update_path_state()
        heading_error = math_utils.wrap_to_pi(self.desired_heading_rad - self.robot.data.heading_w)
        speed_scale = 1.0 - torch.clamp(
            (torch.abs(heading_error) - self.cfg.full_speed_heading_error)
            / (self.cfg.stop_forward_heading_error - self.cfg.full_speed_heading_error),
            min=0.0,
            max=1.0,
        )
        forward_world = self.cfg.forward_velocity * speed_scale

        final_segment = self.segment_index == len(self.path_points_xy) - 1
        _, _, endpoint_distance = self._segment_projection()
        final_scale = torch.clamp(
            endpoint_distance / max(2.0 * self.cfg.goal_tolerance, 1.0e-6),
            min=0.35,
            max=1.0,
        )
        forward_world = torch.where(final_segment, forward_world * final_scale, forward_world)

        tangent_x = torch.cos(self.desired_heading_rad)
        tangent_y = torch.sin(self.desired_heading_rad)
        lateral_world = torch.clamp(
            -self.cfg.cross_track_kp * self.cross_track_error_m,
            min=-self.cfg.max_lateral_velocity,
            max=self.cfg.max_lateral_velocity,
        )
        velocity_world_x = tangent_x * forward_world - tangent_y * lateral_world
        velocity_world_y = tangent_y * forward_world + tangent_x * lateral_world
        robot_yaw = self.robot.data.heading_w
        self.vel_command_b[:, 0] = torch.cos(robot_yaw) * velocity_world_x + torch.sin(robot_yaw) * velocity_world_y
        self.vel_command_b[:, 1] = -torch.sin(robot_yaw) * velocity_world_x + torch.cos(robot_yaw) * velocity_world_y
        self.vel_command_b[:, 2] = torch.clamp(
            self.cfg.heading_kp * heading_error,
            min=-self.cfg.max_angular_velocity,
            max=self.cfg.max_angular_velocity,
        )
        # Stop at the target during the dwell window.  Completion is granted
        # only if the robot keeps the required height instead of jumping
        # through the goal for one control step.
        self.vel_command_b[self.completion_candidate] = 0.0


@configclass
class PctStairVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`PctStairVelocityCommand`."""

    class_type: type = PctStairVelocityCommand
    asset_name: str = "robot"
    path_points_xy: tuple[tuple[float, float], ...] = MISSING
    path_height_fractions: tuple[float, ...] = MISSING
    forward_velocity: float = 0.25
    max_lateral_velocity: float = 0.12
    max_angular_velocity: float = 0.50
    heading_kp: float = 2.0
    cross_track_kp: float = 0.80
    full_speed_heading_error: float = 0.10
    stop_forward_heading_error: float = 0.30
    waypoint_tolerance: float = 0.22
    goal_tolerance: float = 0.25
    segment_advance_ratio: float = 0.92
    completion_progress_ratio: float = 0.94
    completion_height_ratio: float = 0.88
    # Leave this disabled for the original one-way ascent tasks.  Up-and-down
    # routes enable it to require a return to the final path height after first
    # reaching the configured fraction of the route's maximum elevation.
    completion_return_height_tolerance: float | None = None
    completion_peak_height_ratio: float = 0.0
    completion_hold_steps: int = 1
    nominal_base_height: float = 0.30
    minimum_total_rise: float = 1.18
    maximum_total_rise: float = 3.00
    terrain_num_rows: int = 10


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class ArmJointPositionCommand(CommandTerm):
    """Command generator that samples target joint positions for arm joints."""

    cfg: ArmJointPositionCommandCfg  # type: ignore

    def __init__(self, cfg: ArmJointPositionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]
        joint_names = cfg.joint_names
        if joint_names is None:
            raise ValueError("ArmJointPositionCommand requires joint_names to be specified.")
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        self.joint_ids, _ = self.asset.find_joints(joint_names, preserve_order=cfg.preserve_order)
        if len(self.joint_ids) == 0:
            raise ValueError(f"No joints matched joint_names={joint_names} in asset '{cfg.asset_name}'.")
        self.command_buffer = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        joint_defaults = self.asset.data.default_joint_pos[env_ids][:, self.joint_ids]
        pos_range = self.cfg.position_range
        if isinstance(pos_range, (list, tuple)) and len(pos_range) > 0 and isinstance(pos_range[0], (list, tuple)):
            if len(pos_range) != len(self.joint_ids):
                raise ValueError("position_range list must match arm joint count.")
            min_range = torch.tensor([r[0] for r in pos_range], device=self.device)
            max_range = torch.tensor([r[1] for r in pos_range], device=self.device)
            offsets = torch.empty((len(env_ids), len(self.joint_ids)), device=self.device)
            offsets.uniform_(0.0, 1.0)
            offsets = min_range + offsets * (max_range - min_range)
        else:
            offsets = torch.empty((len(env_ids), len(self.joint_ids)), device=self.device).uniform_(*pos_range)
        if self.cfg.use_default_offset:
            target_pos = joint_defaults + offsets
        else:
            target_pos = offsets
        if self.cfg.clip_to_joint_limits:
            limits = self.asset.data.soft_joint_pos_limits[env_ids][:, self.joint_ids, :]
            min_pos = limits[..., 0]
            max_pos = limits[..., 1]
            target_pos = torch.max(torch.min(target_pos, max_pos), min_pos)
        self.command_buffer[env_ids] = target_pos

    def _update_command(self):
        pass


@configclass
class ArmJointPositionCommandCfg(CommandTermCfg):
    """Configuration for arm joint position commands."""

    class_type: type = ArmJointPositionCommand
    asset_name: str = "robot"
    joint_names: list[str] | str | None = None
    preserve_order: bool = True
    position_range: tuple[float, float] | list[tuple[float, float]] = (-0.5, 0.5)
    use_default_offset: bool = True
    clip_to_joint_limits: bool = True
