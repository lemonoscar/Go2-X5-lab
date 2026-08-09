"""MDP terms for the Walk These Ways Go2-X5 continuation task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


WTW_FRAME_DIM = 70
WTW_HISTORY_LENGTH = 30
WTW_ACTION_DIM = 12
WTW_COMMAND_DIM = 15
WTW_ACTION_CLIP = 10.0
WTW_OBSERVATION_CLIP = 100.0

WTW_MODE_PURE_VX = 0
WTW_MODE_PURE_VY = 1
WTW_MODE_PURE_YAW = 2
WTW_MODE_MIXED = 3
WTW_MODE_STAND = 4

WTW_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
WTW_GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")
WTW_GRIPPER_DEFAULT_JOINT_POS = (0.044, 0.044)
WTW_DEFAULT_JOINT_POS = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)
WTW_COMMAND_SCALES = (
    2.0,
    2.0,
    0.25,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.15,
    0.3,
    0.3,
    1.0,
    1.0,
    1.0,
)


def validate_wtw_walking_command_spec(
    vx_values: Sequence[float],
    vy_values: Sequence[float],
    yaw_values: Sequence[float],
    category_probabilities: Sequence[float],
    min_mixed_planar_speed: float = 0.25,
    standing_probability: float = 0.0,
) -> None:
    """Validate the discrete trot support and its optional STAND branch."""

    for label, values in (("vx_values", vx_values), ("vy_values", vy_values), ("yaw_values", yaw_values)):
        if not values:
            raise ValueError(f"{label} must contain at least one value.")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{label} must contain only finite values.")
        if any(abs(float(value)) < 1.0e-9 for value in values):
            raise ValueError(f"{label} must contain only non-zero walking values.")

    if len(category_probabilities) != 4:
        raise ValueError("category_probabilities must contain pure-vx, pure-vy, pure-yaw, and mixed entries.")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in category_probabilities):
        raise ValueError("category_probabilities must be finite and non-negative.")
    if not math.isclose(sum(float(value) for value in category_probabilities), 1.0, abs_tol=1.0e-6):
        raise ValueError("category_probabilities must sum to 1.0.")
    if min_mixed_planar_speed <= 0.0:
        raise ValueError("min_mixed_planar_speed must be positive.")
    if not math.isfinite(standing_probability) or not 0.0 <= standing_probability < 1.0:
        raise ValueError("standing_probability must be finite and in [0, 1).")

    minimum_mixed_speed = math.hypot(
        min(abs(float(value)) for value in vx_values),
        min(abs(float(value)) for value in vy_values),
    )
    if minimum_mixed_speed + 1.0e-9 < min_mixed_planar_speed:
        raise ValueError(
            f"mixed planar speed must be >= {min_mixed_planar_speed}, got minimum {minimum_mixed_speed}."
        )


def sample_wtw_walking_commands(
    count: int,
    vx_values: torch.Tensor,
    vy_values: torch.Tensor,
    yaw_values: torch.Tensor,
    category_cdf: torch.Tensor,
    standing_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample exact WTW trot bins plus an optional all-zero STAND branch."""

    if count <= 0:
        raise ValueError("count must be positive.")
    if any(values.ndim != 1 or values.numel() == 0 for values in (vx_values, vy_values, yaw_values)):
        raise ValueError("vx_values, vy_values, and yaw_values must be non-empty 1-D tensors.")
    if category_cdf.shape != (4,):
        raise ValueError("category_cdf must have shape (4,).")
    if not (vx_values.device == vy_values.device == yaw_values.device == category_cdf.device):
        raise ValueError("all command tensors must use the same device.")
    if not math.isfinite(standing_probability) or not 0.0 <= standing_probability < 1.0:
        raise ValueError("standing_probability must be finite and in [0, 1).")

    device = vx_values.device
    commands = torch.zeros((count, 3), dtype=vx_values.dtype, device=device)
    modes = torch.searchsorted(category_cdf, torch.rand(count, device=device), right=False).clamp_max(
        WTW_MODE_MIXED
    )

    pure_vx = modes == WTW_MODE_PURE_VX
    pure_vy = modes == WTW_MODE_PURE_VY
    pure_yaw = modes == WTW_MODE_PURE_YAW
    mixed = modes == WTW_MODE_MIXED

    if torch.any(pure_vx):
        commands[pure_vx, 0] = vx_values[
            torch.randint(vx_values.numel(), (int(pure_vx.sum().item()),), device=device)
        ]
    if torch.any(pure_vy):
        commands[pure_vy, 1] = vy_values[
            torch.randint(vy_values.numel(), (int(pure_vy.sum().item()),), device=device)
        ]
    if torch.any(pure_yaw):
        commands[pure_yaw, 2] = yaw_values[
            torch.randint(yaw_values.numel(), (int(pure_yaw.sum().item()),), device=device)
        ]
    if torch.any(mixed):
        mixed_count = int(mixed.sum().item())
        commands[mixed, 0] = vx_values[torch.randint(vx_values.numel(), (mixed_count,), device=device)]
        commands[mixed, 1] = vy_values[torch.randint(vy_values.numel(), (mixed_count,), device=device)]
        commands[mixed, 2] = yaw_values[torch.randint(yaw_values.numel(), (mixed_count,), device=device)]

    if standing_probability > 0.0:
        standing = torch.rand(count, device=device) < standing_probability
        commands[standing] = 0.0
        modes[standing] = WTW_MODE_STAND

    return commands, modes


class WTWWalkingVelocityCommand(UniformVelocityCommand):
    """Sample STAND or an in-distribution nominal-trot velocity command."""

    cfg: "WTWWalkingVelocityCommandCfg"

    def __init__(self, cfg: "WTWWalkingVelocityCommandCfg", env):
        validate_wtw_walking_command_spec(
            cfg.vx_values,
            cfg.vy_values,
            cfg.yaw_values,
            cfg.category_probabilities,
            cfg.min_mixed_planar_speed,
            cfg.standing_probability,
        )
        super().__init__(cfg, env)
        self._vx_values = torch.tensor(cfg.vx_values, dtype=torch.float32, device=self.device)
        self._vy_values = torch.tensor(cfg.vy_values, dtype=torch.float32, device=self.device)
        self._yaw_values = torch.tensor(cfg.yaw_values, dtype=torch.float32, device=self.device)
        self._category_cdf = torch.tensor(
            cfg.category_probabilities, dtype=torch.float32, device=self.device
        ).cumsum(dim=0)
        self._category_cdf[-1] = 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        commands, modes = sample_wtw_walking_commands(
            len(env_ids_tensor),
            self._vx_values,
            self._vy_values,
            self._yaw_values,
            self._category_cdf,
            self.cfg.standing_probability,
        )
        self.vel_command_b[env_ids_tensor] = commands
        self.is_heading_env[env_ids_tensor] = False
        self.is_standing_env[env_ids_tensor] = modes == WTW_MODE_STAND


@configclass
class WTWWalkingVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`WTWWalkingVelocityCommand`."""

    class_type: type = WTWWalkingVelocityCommand
    vx_values: tuple[float, ...] = (-0.75, -0.50, -0.25, 0.25, 0.50, 0.75)
    vy_values: tuple[float, ...] = (-0.40, -0.25, 0.25, 0.40)
    yaw_values: tuple[float, ...] = (-0.50, -0.30, 0.30, 0.50)
    category_probabilities: tuple[float, ...] = (0.45, 0.20, 0.20, 0.15)
    min_mixed_planar_speed: float = 0.25
    standing_probability: float = 0.20


def build_wtw_command(base_velocity: torch.Tensor) -> torch.Tensor:
    """Expand base velocity into STAND or the source policy's nominal trot command."""

    if base_velocity.ndim != 2 or base_velocity.shape[1] != 3:
        raise ValueError(f"base_velocity must have shape (N, 3), got {tuple(base_velocity.shape)}.")
    command = torch.zeros(
        (base_velocity.shape[0], WTW_COMMAND_DIM), dtype=base_velocity.dtype, device=base_velocity.device
    )
    command[:, :3] = base_velocity
    walking = torch.any(torch.abs(base_velocity) > 1.0e-6, dim=1)
    command[walking, 4] = 2.5
    command[walking, 5] = 0.5
    command[walking, 8] = 0.5
    command[walking, 9] = 0.08
    command[~walking, 8] = 1.0
    command[:, 12] = 0.25
    command[:, 13] = 0.40
    return command


def _require_matrix(value: torch.Tensor, batch_size: int, width: int, label: str) -> None:
    if value.shape != (batch_size, width):
        raise ValueError(f"{label} must have shape ({batch_size}, {width}), got {tuple(value.shape)}.")


def build_wtw_observation_frame(
    *,
    projected_gravity: torch.Tensor,
    base_velocity_command: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    episode_steps: torch.Tensor,
    step_dt: float,
    default_joint_pos: torch.Tensor | None = None,
    command_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct one exact 70-D WTW frame; reset rows are all zero."""

    batch_size = projected_gravity.shape[0]
    _require_matrix(projected_gravity, batch_size, 3, "projected_gravity")
    _require_matrix(base_velocity_command, batch_size, 3, "base_velocity_command")
    _require_matrix(joint_pos, batch_size, WTW_ACTION_DIM, "joint_pos")
    _require_matrix(joint_vel, batch_size, WTW_ACTION_DIM, "joint_vel")
    _require_matrix(current_action, batch_size, WTW_ACTION_DIM, "current_action")
    _require_matrix(previous_action, batch_size, WTW_ACTION_DIM, "previous_action")
    if episode_steps.shape != (batch_size,):
        raise ValueError(f"episode_steps must have shape ({batch_size},), got {tuple(episode_steps.shape)}.")
    if not math.isclose(float(step_dt), 0.02, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"WTW observation requires a 0.02 s policy step, got {step_dt}.")

    if default_joint_pos is None:
        default_joint_pos = joint_pos.new_tensor(WTW_DEFAULT_JOINT_POS)
    if command_scales is None:
        command_scales = joint_pos.new_tensor(WTW_COMMAND_SCALES)
    if default_joint_pos.shape != (WTW_ACTION_DIM,):
        raise ValueError(f"default_joint_pos must have shape ({WTW_ACTION_DIM},).")
    if command_scales.shape != (WTW_COMMAND_DIM,):
        raise ValueError(f"command_scales must have shape ({WTW_COMMAND_DIM},).")

    command = build_wtw_command(base_velocity_command)
    gait_index = torch.remainder(episode_steps.to(joint_pos.dtype) * step_dt * command[:, 4], 1.0)
    foot_phase = torch.stack(
        (
            gait_index + command[:, 5] + command[:, 6] + command[:, 7],
            gait_index + command[:, 6],
            gait_index + command[:, 7],
            gait_index + command[:, 5],
        ),
        dim=-1,
    )
    clock = torch.sin(2.0 * math.pi * foot_phase)

    frame = torch.cat(
        (
            projected_gravity,
            command * command_scales,
            joint_pos - default_joint_pos,
            joint_vel * 0.05,
            current_action.clamp(-WTW_ACTION_CLIP, WTW_ACTION_CLIP),
            previous_action.clamp(-WTW_ACTION_CLIP, WTW_ACTION_CLIP),
            clock,
        ),
        dim=-1,
    ).clamp(-WTW_OBSERVATION_CLIP, WTW_OBSERVATION_CLIP)
    if frame.shape != (batch_size, WTW_FRAME_DIM):
        raise RuntimeError(f"WTW frame width changed unexpectedly: {tuple(frame.shape)}.")

    reset_rows = episode_steps == 0
    return torch.where(reset_rows.unsqueeze(1), torch.zeros_like(frame), frame)


def wtw_observation_frame(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Read simulator state and return the reset-safe WTW observation frame."""

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    cache = getattr(env, "_wtw_observation_constants", None)
    cache_signature = (joint_pos.device, joint_pos.dtype)
    if cache is None or cache[0] != cache_signature:
        cache = (
            cache_signature,
            joint_pos.new_tensor(WTW_DEFAULT_JOINT_POS),
            joint_pos.new_tensor(WTW_COMMAND_SCALES),
        )
        env._wtw_observation_constants = cache

    return build_wtw_observation_frame(
        projected_gravity=asset.data.projected_gravity_b,
        base_velocity_command=env.command_manager.get_command(command_name),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        current_action=env.action_manager.action,
        previous_action=env.action_manager.prev_action,
        episode_steps=env.episode_length_buf,
        step_dt=env.step_dt,
        default_joint_pos=cache[1],
        command_scales=cache[2],
    )
