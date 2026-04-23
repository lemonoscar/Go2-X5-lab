"""Shared Layer 1 floor-material profiles for scene preview and environment setup.

Supports:
- Flat terrain with various materials (concrete, wood, tile, grass)
- Undulating terrain with Perlin noise height variation
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, check_file_path

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext

DEFAULT_FLOOR_MATERIAL_TYPES = ("concrete", "wood", "tile", "grass")
DEFAULT_TERRAIN_TYPES = ("flat", "undulating")

LAYER1_FLOOR_SIZE = (120.0, 120.0)
LAYER1_FLOOR_THICKNESS = 0.002
LAYER1_FLOOR_VISUAL_Z = 0.0005

# Undulating terrain settings
UNDULATING_HEIGHT_SCALE = 0.08  # Maximum height variation in meters
UNDULATING_FREQUENCY = 0.5      # Perlin noise frequency
HEIGHTFIELD_SIZE = 256          # Heightfield resolution (rows/cols)


@dataclass(frozen=True)
class FloorMaterialProfile:
    """Visual and physical properties for a Layer 1 floor category."""

    name: str
    mdl_paths: tuple[str, ...]
    fallback_color: tuple[float, float, float]
    fallback_roughness: float
    fallback_metallic: float
    ground_plane_color: tuple[float, float, float]
    static_friction: float
    dynamic_friction: float
    texture_scale: tuple[float, float] | None = None
    albedo_brightness: float | None = None


FLOOR_MATERIAL_PROFILES = {
    "concrete": FloorMaterialProfile(
        name="concrete",
        mdl_paths=(
            f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Concrete/Concrete_Precast.mdl",
            f"{ISAACLAB_NUCLEUS_DIR}/Materials/vMaterials_2/Concrete/Concrete_Precast.mdl",
        ),
        fallback_color=(0.50, 0.50, 0.48),
        fallback_roughness=0.95,
        fallback_metallic=0.0,
        ground_plane_color=(0.34, 0.34, 0.33),
        static_friction=1.00,
        dynamic_friction=0.92,
        texture_scale=(0.12, 0.12),
    ),
    "wood": FloorMaterialProfile(
        name="wood",
        mdl_paths=(
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak.mdl",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber.mdl",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks.mdl",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks.mdl",
        ),
        fallback_color=(0.49, 0.34, 0.21),
        fallback_roughness=0.62,
        fallback_metallic=0.0,
        ground_plane_color=(0.27, 0.18, 0.11),
        static_friction=0.82,
        dynamic_friction=0.74,
        texture_scale=(0.18, 0.18),
        albedo_brightness=1.05,
    ),
    "tile": FloorMaterialProfile(
        name="tile",
        mdl_paths=(
            f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        ),
        fallback_color=(0.86, 0.85, 0.81),
        fallback_roughness=0.24,
        fallback_metallic=0.0,
        ground_plane_color=(0.60, 0.60, 0.56),
        static_friction=0.72,
        dynamic_friction=0.64,
        texture_scale=(0.25, 0.25),
        albedo_brightness=1.0,
    ),
    "grass": FloorMaterialProfile(
        name="grass",
        mdl_paths=(
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Nature/Grass.mdl",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Outdoor/Grass.mdl",
            f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Grass.mdl",
        ),
        fallback_color=(0.24, 0.42, 0.18),
        fallback_roughness=0.98,
        fallback_metallic=0.0,
        ground_plane_color=(0.12, 0.20, 0.09),
        static_friction=1.15,
        dynamic_friction=1.05,
        texture_scale=(0.10, 0.10),
        albedo_brightness=1.1,
    ),
}


def resolve_floor_material_type(
    floor_material: str | None,
    available_types: Sequence[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Resolve an explicit or random floor material choice."""

    if available_types is None:
        available_types = DEFAULT_FLOOR_MATERIAL_TYPES
    available_types = tuple(available_types)
    if not available_types:
        raise ValueError("At least one floor material type must be available.")

    for floor_type in available_types:
        if floor_type not in FLOOR_MATERIAL_PROFILES:
            raise ValueError(f"Unsupported floor material type: {floor_type}")

    if floor_material in (None, "", "random"):
        chooser = rng if rng is not None else random
        return chooser.choice(available_types)

    if floor_material not in available_types:
        raise ValueError(
            f"Floor material '{floor_material}' is not available. "
            f"Available types: {', '.join(available_types)}"
        )
    return floor_material


def get_floor_material_profile(floor_material: str) -> FloorMaterialProfile:
    """Return the profile for the requested floor material."""

    try:
        return FLOOR_MATERIAL_PROFILES[floor_material]
    except KeyError as exc:
        raise ValueError(f"Unsupported floor material type: {floor_material}") from exc


def _find_available_mdl_path(profile: FloorMaterialProfile) -> str | None:
    """Return the first reachable MDL path for the floor profile, if any."""

    for mdl_path in profile.mdl_paths:
        if check_file_path(mdl_path) != 0:
            return mdl_path
    return None


def build_floor_visual_material_cfg(floor_material: str):
    """Create a visual material config with MDL-first fallback behavior."""

    profile = get_floor_material_profile(floor_material)
    mdl_path = _find_available_mdl_path(profile)
    if mdl_path is not None:
        material_kwargs = {"mdl_path": mdl_path, "project_uvw": True}
        if profile.texture_scale is not None:
            material_kwargs["texture_scale"] = profile.texture_scale
        if profile.albedo_brightness is not None:
            material_kwargs["albedo_brightness"] = profile.albedo_brightness
        return sim_utils.MdlFileCfg(**material_kwargs), f"MDL:{mdl_path.rsplit('/', 1)[-1]}"

    return (
        sim_utils.PreviewSurfaceCfg(
            diffuse_color=profile.fallback_color,
            roughness=profile.fallback_roughness,
            metallic=profile.fallback_metallic,
        ),
        "PreviewSurface fallback",
    )


def build_floor_physics_material_cfg(floor_material: str) -> sim_utils.RigidBodyMaterialCfg:
    """Create physics material settings for the chosen floor type."""

    profile = get_floor_material_profile(floor_material)
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=profile.static_friction,
        dynamic_friction=profile.dynamic_friction,
        restitution=0.0,
    )


def generate_perlin_noise_terrain(
    size: int = HEIGHTFIELD_SIZE,
    scale: float = UNDULATING_HEIGHT_SCALE,
    frequency: float = UNDULATING_FREQUENCY,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a heightfield using Perlin-like noise for undulating terrain.

    This creates smooth, rolling terrain with specified height variation.

    Args:
        size: Resolution of the heightfield (size x size).
        scale: Maximum height variation in meters.
        frequency: Noise frequency (higher = more bumps).
        seed: Random seed for reproducibility.

    Returns:
        2D numpy array of height values.
    """
    rng = np.random.RandomState(seed)

    # Generate noise using multiple octaves for natural-looking terrain
    heights = np.zeros((size, size))

    def smooth_noise(x, y, seed_offset=0, grid_size=64):
        """Generate smooth noise using value interpolation."""
        rng_local = np.random.RandomState(seed + seed_offset if seed is not None else None)
        noise = rng_local.randn(grid_size + 1, grid_size + 1)

        # Scale to grid coordinates
        scale_x = x * grid_size / size
        scale_y = y * grid_size / size

        # Clamp to valid range
        scale_x = max(0, min(grid_size - 0.01, scale_x))
        scale_y = max(0, min(grid_size - 0.01, scale_y))

        x0, y0 = int(scale_x), int(scale_y)
        x1, y1 = min(x0 + 1, grid_size), min(y0 + 1, grid_size)

        fx, fy = scale_x - x0, scale_y - y0

        # Interpolate
        top = noise[x0, y0] * (1 - fx) + noise[x1, y0] * fx
        bottom = noise[x0, y1] * (1 - fx) + noise[x1, y1] * fx
        return top * (1 - fy) + bottom * fy

    # Combine multiple octaves for natural-looking terrain
    octaves = [
        (1.0, frequency * 1.0, 0, 64),
        (0.5, frequency * 2.0, 1000, 64),
        (0.25, frequency * 4.0, 2000, 64),
    ]

    for amplitude, freq, seed_offset, grid_size in octaves:
        for i in range(size):
            for j in range(size):
                heights[i, j] += amplitude * smooth_noise(
                    i, j, seed_offset, grid_size
                )

    # Normalize to [0, scale]
    heights = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
    heights = heights * scale

    return heights


def create_undulating_terrain_collision(
    prim_path: str,
    size: tuple[float, float] = LAYER1_FLOOR_SIZE,
    height_scale: float = UNDULATING_HEIGHT_SCALE,
    frequency: float = UNDULATING_FREQUENCY,
    resolution: int = HEIGHTFIELD_SIZE,
    seed: int | None = None,
) -> None:
    """Create an undulating terrain using heightfield collision.

    Args:
        prim_path: USD prim path for the terrain.
        size: (width, length) of the terrain in meters.
        height_scale: Maximum height variation.
        frequency: Noise frequency.
        resolution: Heightfield resolution (rows/cols).
        seed: Random seed.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    # Generate heightfield data
    heights = generate_perlin_noise_terrain(
        size=resolution,
        scale=height_scale,
        frequency=frequency,
        seed=seed,
    )

    # Create the heightfield prim
    terrain_geom = UsdGeom.Mesh.Define(stage, prim_path)

    # Create vertices for the heightfield mesh
    width, length = size
    dx = width / resolution
    dy = length / resolution

    vertices = []
    face_vertex_counts = []
    face_vertex_indices = []

    for i in range(resolution):
        for j in range(resolution):
            x = (i - resolution / 2) * dx
            y = (j - resolution / 2) * dy
            z = heights[i, j]
            vertices.append((x, y, z))

    # Create faces (quads converted to triangles)
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            # First triangle
            face_vertex_indices.extend([idx, idx + 1, idx + resolution])
            face_vertex_counts.append(3)
            # Second triangle
            face_vertex_indices.extend([idx + 1, idx + resolution + 1, idx + resolution])
            face_vertex_counts.append(3)

    # Set mesh attributes
    terrain_geom.CreatePointsAttr().Set(vertices)
    terrain_geom.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    terrain_geom.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Compute normals for smooth shading
    UsdGeom.PrimvarsAPI(terrain_geom).CreatePrimvar(
        "st", UsdGeom.Tokens.variability, Sdf.ValueTypeNames.Float2Array, 1
    )

    # Apply subdivision for smooth appearance
    UsdGeom.Mesh(terrain_geom).CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.uniform)
    UsdGeom.Mesh(terrain_geom).CreateSubdivisionLevelAttr().Set(1)


def create_undulating_terrain_visual(
    prim_path: str,
    floor_material: str,
    size: tuple[float, float] = LAYER1_FLOOR_SIZE,
    height_scale: float = UNDULATING_HEIGHT_SCALE,
    frequency: float = UNDULATING_FREQUENCY,
    resolution: int = HEIGHTFIELD_SIZE,
    seed: int | None = None,
) -> None:
    """Create visual mesh for undulating terrain with material.

    Args:
        prim_path: USD prim path for the visual terrain.
        floor_material: Material type (concrete, wood, tile, grass).
        size: (width, length) of the terrain in meters.
        height_scale: Maximum height variation.
        frequency: Noise frequency.
        resolution: Mesh resolution.
        seed: Random seed.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    # Generate heightfield data
    heights = generate_perlin_noise_terrain(
        size=resolution,
        scale=height_scale,
        frequency=frequency,
        seed=seed,
    )

    # Create the visual mesh
    visual_geom = UsdGeom.Mesh.Define(stage, prim_path)

    # Create vertices
    width, length = size
    dx = width / resolution
    dy = length / resolution

    vertices = []
    face_vertex_counts = []
    face_vertex_indices = []
    uvs = []

    for i in range(resolution):
        for j in range(resolution):
            x = (i - resolution / 2) * dx
            y = (j - resolution / 2) * dy
            z = heights[i, j]
            vertices.append((x, y, z))
            # UV coordinates for texture mapping
            uvs.append((i / resolution * 10.0, j / resolution * 10.0))

    # Create faces
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            face_vertex_indices.extend([idx, idx + 1, idx + resolution])
            face_vertex_counts.append(3)
            face_vertex_indices.extend([idx + 1, idx + resolution + 1, idx + resolution])
            face_vertex_counts.append(3)

    visual_geom.CreatePointsAttr().Set(vertices)
    visual_geom.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    visual_geom.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Set UVs for texture
    primvars_api = UsdGeom.PrimvarsAPI(visual_geom)
    st_attr = primvars_api.CreatePrimvar(
        "st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.vertex
    )
    st_attr.Set(uvs)

    # Apply material
    visual_material, _ = build_floor_visual_material_cfg(floor_material)

    # Bind material to the mesh
    from isaaclab.sim import spawn_utils
    spawn_utils.bind_material_to_prim(
        stage.GetPrimAtPath(prim_path),
        visual_material,
    )

    # Enable smooth shading
    UsdGeom.Mesh(visual_geom).CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.uniform)


def resolve_terrain_type(
    terrain: str | None,
    available_types: Sequence[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Resolve an explicit or random terrain type choice.

    Args:
        terrain: "flat", "undulating", "random", or None.
        available_types: Available terrain types.
        rng: Random number generator.

    Returns:
        Resolved terrain type.
    """
    if available_types is None:
        available_types = DEFAULT_TERRAIN_TYPES
    available_types = tuple(available_types)

    if terrain in (None, "", "random"):
        chooser = rng if rng is not None else random
        return chooser.choice(available_types)

    if terrain not in available_types:
        raise ValueError(
            f"Terrain '{terrain}' not available. Options: {available_types}"
        )

    return terrain


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""

    terrain_type: str = "flat"  # "flat" or "undulating"
    floor_material: str = "concrete"
    height_scale: float = UNDULATING_HEIGHT_SCALE
    frequency: float = UNDULATING_FREQUENCY
    resolution: int = HEIGHTFIELD_SIZE
    seed: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "terrain_type": self.terrain_type,
            "floor_material": self.floor_material,
            "height_scale": self.height_scale,
            "frequency": self.frequency,
            "resolution": self.resolution,
            "seed": self.seed,
        }
