# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Fixed 10 cm short-stair terrain used by the mixed flat/stair task."""

from __future__ import annotations

from dataclasses import MISSING
import math

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _box(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))


def resolve_short_stair_step_count(difficulty: float, minimum_steps: int, maximum_steps: int) -> int:
    """Map the two-row terrain difficulty to an exact two- or three-step route."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}.")
    if minimum_steps < 1 or maximum_steps < minimum_steps:
        raise ValueError("Short-stair step bounds must satisfy 1 <= minimum_steps <= maximum_steps.")
    interpolated = minimum_steps + difficulty * (maximum_steps - minimum_steps)
    return min(maximum_steps, max(minimum_steps, math.floor(interpolated + 0.5)))


def mixed_short_stairs_terrain(
    difficulty: float,
    cfg: "MixedShortStairsTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat approach followed by a fixed-size ascent or descent."""
    step_count = resolve_short_stair_step_count(
        difficulty, cfg.minimum_steps, cfg.maximum_steps
    )
    dimensions = (
        cfg.route_width,
        cfg.approach_length,
        cfg.step_height,
        cfg.step_depth,
        cfg.minimum_landing_length,
    )
    if min(dimensions) <= 0.0:
        raise ValueError("All short-stair route dimensions must be positive.")

    size_x, size_y = cfg.size
    start_x, center_y = cfg.start_position
    stair_start_x = start_x + cfg.approach_length
    stair_end_x = stair_start_x + step_count * cfg.step_depth
    if start_x < 0.0 or stair_end_x + cfg.minimum_landing_length > size_x:
        raise ValueError("Short-stair route does not fit inside the terrain tile in X.")
    if center_y - cfg.route_width / 2.0 < 0.0 or center_y + cfg.route_width / 2.0 > size_y:
        raise ValueError("Short-stair route does not fit inside the terrain tile in Y.")

    floor_depth = 0.10
    bottom_z = -0.02
    meshes: list[trimesh.Trimesh] = [
        _box(
            (size_x, size_y, floor_depth),
            (size_x / 2.0, size_y / 2.0, -floor_depth / 2.0),
        )
    ]
    total_height = step_count * cfg.step_height

    if cfg.descent:
        # The origin is on the elevated approach. Each following 25 cm tread
        # is one 10 cm lower; the final tread is the base floor itself.
        platform_height = total_height - bottom_z
        meshes.append(
            _box(
                (stair_start_x, cfg.route_width, platform_height),
                (
                    stair_start_x / 2.0,
                    center_y,
                    bottom_z + platform_height / 2.0,
                ),
            )
        )
        for step_index in range(step_count - 1):
            top_z = (step_count - step_index - 1) * cfg.step_height
            height = top_z - bottom_z
            x_min = stair_start_x + step_index * cfg.step_depth
            meshes.append(
                _box(
                    (cfg.step_depth, cfg.route_width, height),
                    (x_min + cfg.step_depth / 2.0, center_y, bottom_z + height / 2.0),
                )
            )
        origin_z = total_height
    else:
        # Ascending flight: h, 2h, ... Nh, followed by a long top landing.
        for step_index in range(step_count):
            top_z = (step_index + 1) * cfg.step_height
            height = top_z - bottom_z
            x_min = stair_start_x + step_index * cfg.step_depth
            meshes.append(
                _box(
                    (cfg.step_depth, cfg.route_width, height),
                    (x_min + cfg.step_depth / 2.0, center_y, bottom_z + height / 2.0),
                )
            )
        landing_height = total_height - bottom_z
        meshes.append(
            _box(
                (size_x - stair_end_x, cfg.route_width, landing_height),
                (
                    (stair_end_x + size_x) / 2.0,
                    center_y,
                    bottom_z + landing_height / 2.0,
                ),
            )
        )
        origin_z = 0.0

    return meshes, np.array((start_x, center_y, origin_z), dtype=np.float64)


@configclass
class MixedShortStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`mixed_short_stairs_terrain`."""

    function = mixed_short_stairs_terrain

    start_position: tuple[float, float] = (1.0, 3.0)
    route_width: float = 2.0
    approach_length: float = 1.25
    step_height: float = 0.10
    step_depth: float = 0.25
    minimum_steps: int = 2
    maximum_steps: int = 3
    minimum_landing_length: float = 1.25
    descent: bool = False

    proportion: float = MISSING
