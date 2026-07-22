"""Isaac-independent validation and sampling for the mixed short-stair task."""

from __future__ import annotations

import math

import torch


MODE_STAND = 0
MODE_PURE_VX = 1
MODE_PURE_VY = 2
MODE_COMBINED = 3
MODE_TURN = 4
MODE_ASCENT = 5
MODE_DESCENT = 6


def _validate_bins(
    name: str,
    values: tuple[float, ...],
    minimum_magnitude: float,
    *,
    require_positive: bool = False,
) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values.")
    if any(abs(value) + 1.0e-9 < minimum_magnitude for value in values):
        raise ValueError(f"Every {name} value must have magnitude >= {minimum_magnitude}.")
    if require_positive and any(value <= 0.0 for value in values):
        raise ValueError(f"Every {name} value must be positive.")


def validate_mixed_short_stairs_command_spec(
    vx_values: tuple[float, ...],
    vy_values: tuple[float, ...],
    ascent_speed_values: tuple[float, ...],
    descent_speed_values: tuple[float, ...],
    probabilities: tuple[float, float, float, float, float],
    minimum_translation_speed: float,
    arm_motion_probability: float,
    turn_angle_rad: float,
) -> None:
    """Validate flat modes, stair speed bins, arm subset, and relative turn angle."""
    if minimum_translation_speed <= 0.0:
        raise ValueError("minimum_translation_speed must be positive.")
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("Mixed short-stair command probabilities must be finite and non-negative.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("Mixed short-stair command probabilities must sum to 1.0.")
    if not 0.0 <= arm_motion_probability <= 1.0:
        raise ValueError("arm_motion_probability must stay inside [0, 1].")
    if not math.isfinite(turn_angle_rad) or turn_angle_rad <= 0.0 or turn_angle_rad > math.pi:
        raise ValueError("turn_angle_rad must stay inside (0, pi].")

    _validate_bins("vx_values", vx_values, minimum_translation_speed)
    _validate_bins("vy_values", vy_values, minimum_translation_speed)
    _validate_bins("ascent_speed_values", ascent_speed_values, 1.0e-6, require_positive=True)
    _validate_bins("descent_speed_values", descent_speed_values, 1.0e-6, require_positive=True)


def sample_flat_mixed_short_stairs_commands(
    count: int,
    vx_values: torch.Tensor,
    vy_values: torch.Tensor,
    category_cdf: torch.Tensor,
    arm_motion_probability: float,
    turn_angle_rad: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample flat commands, modes, the standing-arm subset, and relative turn deltas."""
    if count < 0:
        raise ValueError("count must be non-negative.")
    commands = torch.zeros((count, 3), dtype=vx_values.dtype, device=vx_values.device)
    modes = torch.zeros(count, dtype=torch.long, device=vx_values.device)
    arm_motion = torch.zeros(count, dtype=torch.bool, device=vx_values.device)
    turn_delta = torch.zeros(count, dtype=vx_values.dtype, device=vx_values.device)
    if count == 0:
        return commands, modes, arm_motion, turn_delta

    modes = torch.bucketize(
        torch.rand(count, device=vx_values.device), category_cdf, right=False
    )
    pure_vx = modes == MODE_PURE_VX
    pure_vy = modes == MODE_PURE_VY
    combined = modes == MODE_COMBINED
    turn = modes == MODE_TURN
    sample_vx = pure_vx | combined
    sample_vy = pure_vy | combined

    vx_count = int(sample_vx.sum().item())
    if vx_count:
        vx_indices = torch.randint(0, len(vx_values), (vx_count,), device=vx_values.device)
        commands[sample_vx, 0] = vx_values[vx_indices]
    vy_count = int(sample_vy.sum().item())
    if vy_count:
        vy_indices = torch.randint(0, len(vy_values), (vy_count,), device=vy_values.device)
        commands[sample_vy, 1] = vy_values[vy_indices]

    standing = modes == MODE_STAND
    arm_motion[standing] = (
        torch.rand(int(standing.sum().item()), device=vx_values.device)
        < arm_motion_probability
    )
    turn_count = int(turn.sum().item())
    if turn_count:
        turn_delta[turn] = torch.where(
            torch.rand(turn_count, device=vx_values.device) < 0.5,
            -torch.full((turn_count,), turn_angle_rad, device=vx_values.device),
            torch.full((turn_count,), turn_angle_rad, device=vx_values.device),
        )
    return commands, modes, arm_motion, turn_delta
