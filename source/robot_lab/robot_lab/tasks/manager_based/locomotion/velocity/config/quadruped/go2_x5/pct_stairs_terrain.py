# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Programmatic straight-stair proxy for the first PCT stair flight."""

from __future__ import annotations

from dataclasses import MISSING
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


# Local XY waypoints relative to the terrain origin.  The route deliberately
# contains no platform turn: approach, one measured PCT-length flight, and a
# short straight exit on the top platform.
PCT_STRAIGHT_STAIR_PATH_POINTS_XY = (
    (0.0, 0.0),
    (0.0, 0.05),
    (0.0, 3.90),
    (0.0, 4.60),
)
PCT_STRAIGHT_STAIR_PATH_HEIGHT_FRACTIONS = (0.0, 0.0, 1.0, 1.0)

# Regularized dimensions of the deployed PCT first flight.  Ray casts through
# the scan show ten approximately 0.157 m levels between progress 0.45 m and
# 3.15 m.  The remaining centerline length is the flat lower/upper landing.
# The 0.86 m width is the clear tread band and excludes the stairwell walls.
PCT_MEASURED_STAIR_CENTERLINE_RUN_M = 3.8510632252535135
PCT_REGULAR_STAIR_FLIGHT_RISE_M = 1.57
PCT_REGULAR_STAIR_WIDTH_M = 0.86
PCT_REGULAR_STAIR_COUNT = 10
PCT_MEASURED_FIRST_RISER_PROGRESS_M = 0.525
PCT_MEASURED_LAST_RISER_PROGRESS_M = 3.095
PCT_REGULAR_STAIR_TREAD_M = (
    PCT_MEASURED_LAST_RISER_PROGRESS_M - PCT_MEASURED_FIRST_RISER_PROGRESS_M
) / (PCT_REGULAR_STAIR_COUNT - 1)
PCT_REGULAR_STAIR_FLIGHT_RUN_M = PCT_REGULAR_STAIR_TREAD_M * PCT_REGULAR_STAIR_COUNT
PCT_REGULAR_STAIR_RISER_M = PCT_REGULAR_STAIR_FLIGHT_RISE_M / PCT_REGULAR_STAIR_COUNT
PCT_REGULAR_STAIR_APPROACH_M = PCT_MEASURED_FIRST_RISER_PROGRESS_M
PCT_REGULAR_STAIR_PLATFORM_GATE_M = 0.05 + PCT_MEASURED_STAIR_CENTERLINE_RUN_M
PCT_REGULAR_STAIR_PATH_LENGTH_M = PCT_REGULAR_STAIR_PLATFORM_GATE_M + 0.70
PCT_REGULAR_STAIR_TOP_EXIT_M = (
    PCT_REGULAR_STAIR_PATH_LENGTH_M
    - PCT_REGULAR_STAIR_APPROACH_M
    - PCT_REGULAR_STAIR_FLIGHT_RUN_M
)
PCT_REGULAR_STAIR_PATH_POINTS_XY = (
    (0.0, 0.0),
    (0.0, PCT_REGULAR_STAIR_APPROACH_M - 0.05),
    *(
        (0.0, PCT_REGULAR_STAIR_APPROACH_M + index * PCT_REGULAR_STAIR_TREAD_M)
        for index in range(PCT_REGULAR_STAIR_COUNT)
    ),
    (0.0, PCT_REGULAR_STAIR_PLATFORM_GATE_M),
    (0.0, PCT_REGULAR_STAIR_PATH_LENGTH_M),
)
PCT_REGULAR_STAIR_PATH_HEIGHT_FRACTIONS = (
    0.0,
    0.0,
    *((index + 1) / PCT_REGULAR_STAIR_COUNT for index in range(PCT_REGULAR_STAIR_COUNT)),
    1.0,
    1.0,
)

# Symmetric regular-box bridge used to test a complete ascent and descent.
# Both flights retain the dimensions measured from the PCT first flight; only
# scan noise is removed.  The 1.50 m landing gives the 0.70 m-long robot enough
# room to place all four feet before entering the descending flight.
PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M = 1.50
PCT_REGULAR_UP_DOWN_BOTTOM_EXIT_M = 0.80
PCT_REGULAR_UP_DOWN_ASCENT_END_M = (
    PCT_REGULAR_STAIR_APPROACH_M + PCT_REGULAR_STAIR_FLIGHT_RUN_M
)
PCT_REGULAR_UP_DOWN_DESCENT_START_M = (
    PCT_REGULAR_UP_DOWN_ASCENT_END_M + PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M
)
PCT_REGULAR_UP_DOWN_DESCENT_END_M = (
    PCT_REGULAR_UP_DOWN_DESCENT_START_M + PCT_REGULAR_STAIR_FLIGHT_RUN_M
)
PCT_REGULAR_UP_DOWN_COMPLETION_GATE_M = PCT_REGULAR_UP_DOWN_DESCENT_END_M + 0.40
PCT_REGULAR_UP_DOWN_PATH_LENGTH_M = (
    PCT_REGULAR_UP_DOWN_DESCENT_END_M + PCT_REGULAR_UP_DOWN_BOTTOM_EXIT_M
)
PCT_REGULAR_UP_DOWN_PATH_POINTS_XY = (
    (0.0, 0.0),
    (0.0, PCT_REGULAR_STAIR_APPROACH_M - 0.05),
    *(
        (0.0, PCT_REGULAR_STAIR_APPROACH_M + index * PCT_REGULAR_STAIR_TREAD_M)
        for index in range(PCT_REGULAR_STAIR_COUNT)
    ),
    (0.0, PCT_REGULAR_UP_DOWN_ASCENT_END_M),
    (0.0, PCT_REGULAR_UP_DOWN_DESCENT_START_M),
    *(
        (
            0.0,
            PCT_REGULAR_UP_DOWN_DESCENT_START_M
            + (index + 1) * PCT_REGULAR_STAIR_TREAD_M,
        )
        for index in range(PCT_REGULAR_STAIR_COUNT)
    ),
    (0.0, PCT_REGULAR_UP_DOWN_PATH_LENGTH_M),
)
PCT_REGULAR_UP_DOWN_PATH_HEIGHT_FRACTIONS = (
    0.0,
    0.0,
    *((index + 1) / PCT_REGULAR_STAIR_COUNT for index in range(PCT_REGULAR_STAIR_COUNT)),
    1.0,
    1.0,
    *(
        (PCT_REGULAR_STAIR_COUNT - index - 1) / PCT_REGULAR_STAIR_COUNT
        for index in range(PCT_REGULAR_STAIR_COUNT)
    ),
    0.0,
)


def _box(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))


def _normalized_riser_factors(count: int, variation: float) -> np.ndarray:
    """Return deterministic positive riser factors with unit mean."""
    if variation <= 0.0:
        return np.ones(count, dtype=np.float64)
    phase = np.arange(count, dtype=np.float64)
    factors = 1.0 + variation * np.sin(phase * 1.7 + 0.35)
    return factors / np.mean(factors)


def _resolve_pct_collision_mesh_path(configured_path: str) -> Path:
    """Resolve the source PCT collision mesh without copying assets outside the two task repos."""
    if configured_path:
        path = Path(configured_path).expanduser()
    elif os.environ.get("ROBOT_LAB_PCT_COLLISION_PLY"):
        path = Path(os.environ["ROBOT_LAB_PCT_COLLISION_PLY"]).expanduser()
    else:
        path = next(
            (
                parent / "arm-vla-grasp-sim" / "dataset" / "3dgs_collision.ply"
                for parent in Path(__file__).resolve().parents
                if (parent / "arm-vla-grasp-sim" / "dataset" / "3dgs_collision.ply").is_file()
            ),
            Path("/home/lemon/research/arm-vla-grasp-sim/dataset/3dgs_collision.ply"),
        )
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"PCT collision mesh not found at {path}. Set ROBOT_LAB_PCT_COLLISION_PLY explicitly."
        )
    return path


@lru_cache(maxsize=2)
def _load_canonical_pct_first_flight(collision_mesh_path: str) -> trimesh.Trimesh:
    """Crop the first PCT flight and express it as cross-track/progress/height coordinates."""
    mesh = trimesh.load(collision_mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"PCT collision mesh contains no triangle faces: {collision_mesh_path}")

    # The scene USD applies rotateZ=180 degrees to the source PLY.
    centers_world = mesh.triangles_center.copy()
    centers_world[:, :2] *= -1.0
    entrance_xy = np.array((1.50, 5.70), dtype=np.float64)
    landing_xy = np.array((1.9202, 9.52807), dtype=np.float64)
    tangent = landing_xy - entrance_xy
    flight_length = float(np.linalg.norm(tangent))
    tangent /= flight_length
    # Use the right-hand normal so the world-XY -> canonical-(cross, progress)
    # transform has determinant +1.  The opposite normal mirrors the mesh and
    # reverses triangle winding, which makes the training collision surface
    # physically different from the source PCT mesh.
    normal = np.array((tangent[1], -tangent[0]), dtype=np.float64)
    relative_xy = centers_world[:, :2] - entrance_xy
    progress = relative_xy @ tangent
    cross_track = relative_xy @ normal

    # The scan's walkable surface is about -0.13 m at the entrance and 1.44 m
    # at the landing.  Keep the noisy tread/riser band while excluding ceilings
    # and other floors that overlap the stairwell in XY.
    surface_height = -0.13 + 1.57 * np.clip(progress / flight_length, 0.0, 1.0)
    keep_faces = (
        (progress >= -0.80)
        & (progress <= flight_length + 0.80)
        & (np.abs(cross_track) <= 0.65)
        & (np.abs(centers_world[:, 2] - surface_height) <= 0.32)
    )
    face_ids = np.flatnonzero(keep_faces)
    if len(face_ids) < 100:
        raise ValueError(
            f"PCT first-flight crop is unexpectedly small ({len(face_ids)} faces): {collision_mesh_path}"
        )
    cropped = mesh.submesh([face_ids], append=True, repair=False)

    vertices_world = cropped.vertices.copy()
    vertices_world[:, :2] *= -1.0
    relative_vertices_xy = vertices_world[:, :2] - entrance_xy
    canonical_vertices = np.empty_like(vertices_world)
    canonical_vertices[:, 0] = relative_vertices_xy @ normal
    canonical_vertices[:, 1] = relative_vertices_xy @ tangent
    canonical_vertices[:, 2] = vertices_world[:, 2] + 0.13
    cropped.vertices = canonical_vertices
    return cropped


@lru_cache(maxsize=8)
def _load_canonical_pct_first_flight_volume(
    collision_mesh_path: str,
    cross_track_half_width: float,
    minimum_height: float,
    maximum_height: float,
) -> trimesh.Trimesh:
    """Crop the exact local collision volume around the first PCT flight.

    Unlike the legacy surface-band crop, this keeps every source triangle whose
    bounding box intersects the local stair corridor.  Risers, undersides and
    nearby collision faces therefore remain identical to the deployed scene.
    """
    mesh = trimesh.load(collision_mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"PCT collision mesh contains no triangle faces: {collision_mesh_path}")

    vertices_world = mesh.vertices.copy()
    vertices_world[:, :2] *= -1.0
    entrance_xy = np.array((1.50, 5.70), dtype=np.float64)
    landing_xy = np.array((1.9202, 9.52807), dtype=np.float64)
    tangent = landing_xy - entrance_xy
    flight_length = float(np.linalg.norm(tangent))
    tangent /= flight_length
    normal = np.array((tangent[1], -tangent[0]), dtype=np.float64)

    triangle_vertices = vertices_world[mesh.faces]
    relative_xy = triangle_vertices[:, :, :2] - entrance_xy
    progress = relative_xy @ tangent
    cross_track = relative_xy @ normal
    height = triangle_vertices[:, :, 2]
    keep_faces = (
        (np.max(progress, axis=1) >= -0.80)
        & (np.min(progress, axis=1) <= flight_length + 0.80)
        & (np.max(cross_track, axis=1) >= -cross_track_half_width)
        & (np.min(cross_track, axis=1) <= cross_track_half_width)
        & (np.max(height, axis=1) >= minimum_height)
        & (np.min(height, axis=1) <= maximum_height)
    )
    face_ids = np.flatnonzero(keep_faces)
    if len(face_ids) < 500:
        raise ValueError(
            f"PCT first-flight volume crop is unexpectedly small ({len(face_ids)} faces): "
            f"{collision_mesh_path}"
        )

    cropped = mesh.submesh([face_ids], append=True, repair=False)
    cropped_vertices_world = cropped.vertices.copy()
    cropped_vertices_world[:, :2] *= -1.0
    relative_vertices_xy = cropped_vertices_world[:, :2] - entrance_xy
    canonical_vertices = np.empty_like(cropped_vertices_world)
    canonical_vertices[:, 0] = relative_vertices_xy @ normal
    canonical_vertices[:, 1] = relative_vertices_xy @ tangent
    canonical_vertices[:, 2] = cropped_vertices_world[:, 2] + 0.13
    cropped.vertices = canonical_vertices
    return cropped


def pct_straight_stairs_terrain(
    difficulty: float,
    cfg: "PctStraightStairsTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one straight PCT-sized stair flight in a terrain tile."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}.")
    if cfg.flight_steps <= 0:
        raise ValueError("A PCT stair flight requires a positive number of steps.")

    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    riser_factors = _normalized_riser_factors(cfg.flight_steps, cfg.riser_variation)
    tread = cfg.flight_run / cfg.flight_steps

    start_x, start_y = cfg.start_position
    flight_start_y = start_y + cfg.approach_length
    flight_end_y = flight_start_y + cfg.flight_run
    route_end_y = flight_end_y + cfg.top_platform_exit_length

    if start_x - cfg.route_width / 2.0 < 0.0 or start_x + cfg.route_width / 2.0 > cfg.size[0]:
        raise ValueError("PCT stair route exceeds the terrain tile in X.")
    if start_y < 0.0 or route_end_y + cfg.top_platform_margin > cfg.size[1]:
        raise ValueError("PCT straight stair route exceeds the terrain tile in Y.")

    floor_depth = 0.10
    meshes: list[trimesh.Trimesh] = [
        _box(
            (cfg.size[0], cfg.size[1], floor_depth),
            (cfg.size[0] / 2.0, cfg.size[1] / 2.0, -floor_depth / 2.0),
        )
    ]

    bottom_z = -0.02
    flight_top = 0.0
    for step_index, factor in enumerate(riser_factors):
        flight_top += step_height * float(factor)
        y_min = flight_start_y + step_index * tread
        y_max = y_min + tread
        height = flight_top - bottom_z
        meshes.append(
            _box(
                (cfg.route_width, tread, height),
                (start_x, (y_min + y_max) / 2.0, bottom_z + height / 2.0),
            )
        )

    top_y_min = flight_end_y - cfg.top_platform_margin
    top_y_max = route_end_y + cfg.top_platform_margin
    top_height = flight_top - bottom_z
    meshes.append(
        _box(
            (cfg.route_width, top_y_max - top_y_min, top_height),
            (start_x, (top_y_min + top_y_max) / 2.0, bottom_z + top_height / 2.0),
        )
    )

    return meshes, np.array((start_x, start_y, 0.0), dtype=np.float64)


def pct_regular_up_down_stairs_terrain(
    difficulty: float,
    cfg: "PctRegularUpDownStairsTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate two symmetric box stair flights joined by a flat top landing."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}.")
    if cfg.flight_steps <= 0:
        raise ValueError("A PCT up/down route requires a positive number of steps per flight.")
    if min(
        cfg.route_width,
        cfg.approach_length,
        cfg.flight_run,
        cfg.step_height,
        cfg.middle_platform_length,
        cfg.bottom_exit_length,
    ) <= 0.0:
        raise ValueError("All PCT up/down route dimensions must be positive.")

    tread = cfg.flight_run / cfg.flight_steps
    start_x, start_y = cfg.start_position
    ascent_start_y = start_y + cfg.approach_length
    ascent_end_y = ascent_start_y + cfg.flight_run
    descent_start_y = ascent_end_y + cfg.middle_platform_length
    descent_end_y = descent_start_y + cfg.flight_run
    route_end_y = descent_end_y + cfg.bottom_exit_length

    if start_x - cfg.route_width / 2.0 < 0.0 or start_x + cfg.route_width / 2.0 > cfg.size[0]:
        raise ValueError("PCT up/down stair route exceeds the terrain tile in X.")
    if start_y < 0.0 or route_end_y + cfg.route_margin > cfg.size[1]:
        raise ValueError("PCT up/down stair route exceeds the terrain tile in Y.")

    floor_depth = 0.10
    bottom_z = -0.02
    meshes: list[trimesh.Trimesh] = [
        _box(
            (cfg.size[0], cfg.size[1], floor_depth),
            (cfg.size[0] / 2.0, cfg.size[1] / 2.0, -floor_depth / 2.0),
        )
    ]

    # Ascending flight: h, 2h, ..., Nh.
    for step_index in range(cfg.flight_steps):
        top_z = (step_index + 1) * cfg.step_height
        y_min = ascent_start_y + step_index * tread
        height = top_z - bottom_z
        meshes.append(
            _box(
                (cfg.route_width, tread, height),
                (start_x, y_min + tread / 2.0, bottom_z + height / 2.0),
            )
        )

    flight_top = cfg.flight_steps * cfg.step_height
    platform_height = flight_top - bottom_z
    meshes.append(
        _box(
            (cfg.route_width, cfg.middle_platform_length, platform_height),
            (
                start_x,
                (ascent_end_y + descent_start_y) / 2.0,
                bottom_z + platform_height / 2.0,
            ),
        )
    )

    # Descending flight mirrors the ascent: Nh, ..., 2h, h, then floor.
    for step_index in range(cfg.flight_steps):
        top_z = (cfg.flight_steps - step_index) * cfg.step_height
        y_min = descent_start_y + step_index * tread
        height = top_z - bottom_z
        meshes.append(
            _box(
                (cfg.route_width, tread, height),
                (start_x, y_min + tread / 2.0, bottom_z + height / 2.0),
            )
        )

    return meshes, np.array((start_x, start_y, 0.0), dtype=np.float64)


def pct_scanned_straight_stairs_terrain(
    difficulty: float,
    cfg: "PctScannedStraightStairsTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the aligned first-flight crop from the real PCT collision mesh."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}.")

    collision_path = _resolve_pct_collision_mesh_path(cfg.collision_mesh_path)
    if cfg.scan_crop_mode == "surface_band":
        scanned_flight = _load_canonical_pct_first_flight(os.fspath(collision_path)).copy()
    elif cfg.scan_crop_mode == "local_volume":
        scanned_flight = _load_canonical_pct_first_flight_volume(
            os.fspath(collision_path),
            cfg.scan_crop_cross_track_half_width,
            cfg.scan_crop_height_range[0],
            cfg.scan_crop_height_range[1],
        ).copy()
    else:
        raise ValueError(f"Unsupported PCT scan crop mode: {cfg.scan_crop_mode}")
    target_rise = cfg.scan_target_rise_range[0] + difficulty * (
        cfg.scan_target_rise_range[1] - cfg.scan_target_rise_range[0]
    )
    vertices = scanned_flight.vertices.copy()
    vertices[:, 2] *= target_rise / cfg.scan_total_rise
    vertices[:, 0] += cfg.start_position[0]
    vertices[:, 1] += cfg.start_position[1] + cfg.approach_length
    scanned_flight.vertices = vertices

    floor_depth = 0.10
    floor = _box(
        (cfg.size[0], cfg.size[1], floor_depth),
        (cfg.size[0] / 2.0, cfg.size[1] / 2.0, -floor_depth / 2.0),
    )
    flight_end_y = cfg.start_position[1] + cfg.approach_length + cfg.flight_run
    top_y_min = flight_end_y - cfg.top_platform_margin
    top_y_max = flight_end_y + cfg.top_platform_exit_length + cfg.top_platform_margin
    top_platform = _box(
        (cfg.route_width, top_y_max - top_y_min, target_rise + 0.02),
        (
            cfg.start_position[0],
            (top_y_min + top_y_max) / 2.0,
            -0.02 + (target_rise + 0.02) / 2.0,
        ),
    )
    bounds = scanned_flight.bounds
    if bounds[0, 0] < 0.0 or bounds[1, 0] > cfg.size[0]:
        raise ValueError(f"PCT scan crop exceeds terrain tile in X: {bounds[:, 0].tolist()}")
    if bounds[0, 1] < 0.0 or bounds[1, 1] > cfg.size[1]:
        raise ValueError(f"PCT scan crop exceeds terrain tile in Y: {bounds[:, 1].tolist()}")

    start_x, start_y = cfg.start_position
    meshes = []
    if cfg.scan_include_auxiliary_floor:
        meshes.append(floor)
    meshes.append(scanned_flight)
    if cfg.scan_include_auxiliary_top_platform:
        meshes.append(top_platform)
    return meshes, np.array((start_x, start_y, 0.0), dtype=np.float64)


@configclass
class PctStraightStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`pct_straight_stairs_terrain`."""

    function = pct_straight_stairs_terrain

    start_position: tuple[float, float] = (2.0, 0.80)
    route_width: float = 1.0
    # The deployed PCT path has only about 0.049 m between its first point and
    # the calibrated stair entrance.  Training must therefore learn a near-
    # standing start instead of relying on a long flat run-up.
    approach_length: float = 0.05
    flight_run: float = 3.85
    flight_steps: int = 20
    step_height_range: tuple[float, float] = (0.0175, 0.086)
    top_platform_exit_length: float = 0.70
    top_platform_margin: float = 0.30
    riser_variation: float = 0.0

    # Keep the inherited dataclass field explicit for clearer config validation.
    proportion: float = MISSING


@configclass
class PctRegularUpDownStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`pct_regular_up_down_stairs_terrain`."""

    function = pct_regular_up_down_stairs_terrain

    start_position: tuple[float, float] = (6.0, 1.0)
    route_width: float = PCT_REGULAR_STAIR_WIDTH_M
    approach_length: float = PCT_REGULAR_STAIR_APPROACH_M
    flight_run: float = PCT_REGULAR_STAIR_FLIGHT_RUN_M
    flight_steps: int = PCT_REGULAR_STAIR_COUNT
    step_height: float = PCT_REGULAR_STAIR_RISER_M
    middle_platform_length: float = PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M
    bottom_exit_length: float = PCT_REGULAR_UP_DOWN_BOTTOM_EXIT_M
    route_margin: float = 0.40

    proportion: float = MISSING


@configclass
class PctScannedStraightStairsTerrainCfg(PctStraightStairsTerrainCfg):
    """Configuration for the cropped real-PCT first-flight collision surface."""

    function = pct_scanned_straight_stairs_terrain

    start_position: tuple[float, float] = (2.0, 1.20)
    top_platform_margin: float = 0.55
    collision_mesh_path: str = ""
    scan_total_rise: float = 1.57
    scan_target_rise_range: tuple[float, float] = (0.35, 1.57)
    scan_crop_mode: str = "surface_band"
    scan_crop_cross_track_half_width: float = 1.0
    scan_crop_height_range: tuple[float, float] = (-0.8, 2.0)
    scan_include_auxiliary_floor: bool = True
    scan_include_auxiliary_top_platform: bool = True
