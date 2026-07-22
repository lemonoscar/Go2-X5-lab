"""Isaac-independent validation and sampling for stratified planar commands."""

from __future__ import annotations

import math

import torch


def validate_planar_command_spec(
    vx_values: tuple[float, ...],
    vy_values: tuple[float, ...],
    probabilities: tuple[float, float, float, float],
) -> None:
    """Validate standing, pure-vx, pure-vy, and combined command inputs."""
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("planar command category probabilities must be finite and non-negative.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("planar command category probabilities must sum to 1.0.")

    _validate_bins("vx_values", vx_values, probabilities[1] + probabilities[3])
    _validate_bins("vy_values", vy_values, probabilities[2] + probabilities[3])


def _validate_bins(name: str, values: tuple[float, ...], active_probability: float) -> None:
    if active_probability > 0.0 and not values:
        raise ValueError(f"{name} must not be empty when its command category is enabled.")
    if any(not math.isfinite(value) or value == 0.0 for value in values):
        raise ValueError(f"{name} must contain only finite non-zero values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values.")


def sample_stratified_planar_commands(
    count: int,
    vx_values: torch.Tensor,
    vy_values: torch.Tensor,
    category_cdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact-bin ``[vx, vy, 0]`` commands and the standing mask."""
    if count < 0:
        raise ValueError("count must be non-negative.")
    commands = torch.zeros((count, 3), dtype=vx_values.dtype, device=vx_values.device)
    if count == 0:
        return commands, torch.zeros(0, dtype=torch.bool, device=vx_values.device)

    category = torch.bucketize(
        torch.rand(count, device=vx_values.device),
        category_cdf,
        right=False,
    )
    pure_vx = category == 1
    pure_vy = category == 2
    combined = category == 3
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
    return commands, category == 0
