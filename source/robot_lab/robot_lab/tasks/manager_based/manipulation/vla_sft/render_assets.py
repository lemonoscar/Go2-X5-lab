"""Local PBR floor and HDRI asset helpers for Layer 1 scene rendering."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import isaaclab.sim as sim_utils


DATA_ROOT = Path(__file__).resolve().parents[5] / "data"
VLA_RENDER_MANIFEST_PATH = DATA_ROOT / "Materials" / "manifests" / "vla_render_asset_manifest.json"
VLA_RENDER_REGISTRY_PATH = DATA_ROOT / "Materials" / "manifests" / "vla_render_asset_registry.json"

DEFAULT_HDRI_PROFILE_TYPES = ("daylight_soft", "dusk_warm")


@dataclass(frozen=True)
class LocalFloorPBRAsset:
    """Resolved local floor material asset."""

    category: str
    asset_id: str
    asset_dir: str
    color_texture: str | None
    roughness_texture: str | None
    normal_texture: str | None
    metallic_texture: str | None
    ao_texture: str | None
    orm_texture: str | None
    texture_scale: tuple[float, float]
    bump_factor: float = 1.0
    ao_to_diffuse: float = 0.25


@dataclass(frozen=True)
class LocalHDRIProfile:
    """Resolved local HDRI lighting profile."""

    name: str
    asset_id: str
    texture_file: str
    intensity: float
    fallback_color: tuple[float, float, float]
    visible_in_primary_ray: bool = True
    texture_format: str = "automatic"


def _as_tuple_2(values: Sequence[float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if values is None:
        return default
    if len(values) != 2:
        raise ValueError(f"Expected 2 values, received {values}.")
    return float(values[0]), float(values[1])


def _as_tuple_3(values: Sequence[float] | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if values is None:
        return default
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, received {values}.")
    return float(values[0]), float(values[1]), float(values[2])


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    if not VLA_RENDER_MANIFEST_PATH.is_file():
        return {}
    return json.loads(VLA_RENDER_MANIFEST_PATH.read_text())


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    if not VLA_RENDER_REGISTRY_PATH.is_file():
        return {}
    return json.loads(VLA_RENDER_REGISTRY_PATH.read_text())


def clear_render_asset_caches() -> None:
    """Clear cached manifest/registry data."""

    _load_manifest.cache_clear()
    _load_registry.cache_clear()


def available_hdri_profiles() -> tuple[str, ...]:
    """Return the known HDRI profile names from the manifest."""

    manifest = _load_manifest()
    profiles = manifest.get("hdri_profiles", {})
    if not profiles:
        return DEFAULT_HDRI_PROFILE_TYPES
    return tuple(sorted(profiles.keys()))


def resolve_hdri_profile_name(
    hdri_profile: str | None,
    available_profiles: Sequence[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Resolve an explicit or random HDRI profile selection."""

    if available_profiles is None:
        available_profiles = available_hdri_profiles()
    available_profiles = tuple(available_profiles)
    if not available_profiles:
        raise ValueError("At least one HDRI profile must be available.")

    if hdri_profile in ("off", "none"):
        return "off"

    if hdri_profile in (None, "", "random"):
        chooser = rng if rng is not None else random
        return chooser.choice(available_profiles)

    if hdri_profile not in available_profiles:
        raise ValueError(
            f"HDRI profile '{hdri_profile}' is not available. "
            f"Available profiles: {', '.join(available_profiles)}"
        )
    return hdri_profile


def _resolve_data_path(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return str((DATA_ROOT / relative_path).resolve())


def _get_hdri_manifest_entry(profile_name: str) -> dict | None:
    manifest = _load_manifest()
    profiles = manifest.get("hdri_profiles", {})
    return profiles.get(profile_name)


def get_local_hdri_profile(profile_name: str) -> LocalHDRIProfile | None:
    """Return the local HDRI profile if the asset has been downloaded."""

    manifest_entry = _get_hdri_manifest_entry(profile_name)
    if manifest_entry is None:
        return None

    registry = _load_registry()
    profile_data = registry.get("hdri_profiles", {}).get(profile_name)
    if not profile_data:
        return None

    texture_file = _resolve_data_path(profile_data.get("texture_file"))
    if texture_file is None or not Path(texture_file).is_file():
        return None

    return LocalHDRIProfile(
        name=profile_name,
        asset_id=str(profile_data["asset_id"]),
        texture_file=texture_file,
        intensity=float(profile_data.get("intensity", manifest_entry.get("intensity", 1200.0))),
        fallback_color=_as_tuple_3(manifest_entry.get("fallback_color"), (0.75, 0.75, 0.75)),
        visible_in_primary_ray=bool(profile_data.get("visible_in_primary_ray", True)),
        texture_format=str(profile_data.get("texture_format", "automatic")),
    )


def build_hdri_dome_light_cfg(
    hdri_profile: str | None,
    available_profiles: Sequence[str] | None = None,
    rng: random.Random | None = None,
    fallback_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    fallback_intensity: float = 3000.0,
) -> tuple[sim_utils.DomeLightCfg, str, str]:
    """Create a dome-light config with local HDRI-first fallback behavior."""

    chosen_profile = resolve_hdri_profile_name(hdri_profile, available_profiles=available_profiles, rng=rng)
    if chosen_profile == "off":
        cfg = sim_utils.DomeLightCfg(color=fallback_color, intensity=fallback_intensity)
        return cfg, chosen_profile, "DomeLight neutral"

    manifest_entry = _get_hdri_manifest_entry(chosen_profile) or {}
    local_profile = get_local_hdri_profile(chosen_profile)

    dome_color = _as_tuple_3(manifest_entry.get("fallback_color"), fallback_color)
    dome_intensity = float(manifest_entry.get("intensity", fallback_intensity))

    if local_profile is not None:
        cfg = sim_utils.DomeLightCfg(
            color=local_profile.fallback_color,
            intensity=local_profile.intensity,
            texture_file=local_profile.texture_file,
            texture_format=local_profile.texture_format,
            visible_in_primary_ray=local_profile.visible_in_primary_ray,
        )
        return cfg, chosen_profile, f"HDRI:{local_profile.asset_id}"

    cfg = sim_utils.DomeLightCfg(color=dome_color, intensity=dome_intensity)
    return cfg, chosen_profile, f"DomeLight fallback:{chosen_profile}"


def list_local_floor_assets(floor_material: str) -> list[LocalFloorPBRAsset]:
    """Return all locally downloaded PBR assets for the requested floor category."""

    registry = _load_registry()
    assets = registry.get("floor_materials", {}).get(floor_material, [])
    resolved_assets: list[LocalFloorPBRAsset] = []
    for asset in assets:
        color_texture = _resolve_data_path(asset.get("color_texture"))
        if color_texture is None or not Path(color_texture).is_file():
            continue

        resolved_assets.append(
            LocalFloorPBRAsset(
                category=floor_material,
                asset_id=str(asset["asset_id"]),
                asset_dir=str(asset.get("asset_dir", "")),
                color_texture=color_texture,
                roughness_texture=_resolve_data_path(asset.get("roughness_texture")),
                normal_texture=_resolve_data_path(asset.get("normal_texture")),
                metallic_texture=_resolve_data_path(asset.get("metallic_texture")),
                ao_texture=_resolve_data_path(asset.get("ao_texture")),
                orm_texture=_resolve_data_path(asset.get("orm_texture")),
                texture_scale=_as_tuple_2(asset.get("texture_scale"), (0.25, 0.25)),
                bump_factor=float(asset.get("bump_factor", 1.0)),
                ao_to_diffuse=float(asset.get("ao_to_diffuse", 0.25)),
            )
        )
    return resolved_assets


def choose_local_floor_asset(
    floor_material: str,
    rng: random.Random | None = None,
) -> LocalFloorPBRAsset | None:
    """Choose one local floor asset for the requested category, if available."""

    assets = list_local_floor_assets(floor_material)
    if not assets:
        return None
    chooser = rng if rng is not None else random
    return chooser.choice(assets)


def bind_local_floor_material(
    prim_path: str,
    floor_asset: LocalFloorPBRAsset,
    rng: random.Random | None = None,
    material_path: str | None = None,
) -> str:
    """Bind a local OmniPBR material to an existing floor prim."""
    try:
        from isaacsim.core.experimental.materials import OmniPbrMaterial

        material_path = material_path or f"/World/Looks/Floor_{floor_asset.category}_{floor_asset.asset_id}"
        material = OmniPbrMaterial(material_path)
        material.set_input_values("project_uvw", True)
        material.set_input_values("texture_scale", list(floor_asset.texture_scale))
        material.set_input_values("texture_rotate", float((rng.uniform(-180.0, 180.0) if rng is not None else 0.0)))
        material.set_input_values("texture_translate", [
            float(rng.random() if rng is not None else 0.0),
            float(rng.random() if rng is not None else 0.0),
        ])
        material.set_input_values("diffuse_color_constant", [1.0, 1.0, 1.0])
        material.set_input_values("diffuse_texture", floor_asset.color_texture)
        material.set_input_values("metallic_constant", 0.0)
        material.set_input_values("specular_level", 0.45)

        if floor_asset.roughness_texture:
            material.set_input_values("reflection_roughness_texture_influence", 1.0)
            material.set_input_values("reflectionroughness_texture", floor_asset.roughness_texture)
        else:
            material.set_input_values("reflection_roughness_constant", 0.85)

        if floor_asset.orm_texture:
            material.set_input_values("enable_ORM_texture", True)
            material.set_input_values("ORM_texture", floor_asset.orm_texture)
        else:
            material.set_input_values("enable_ORM_texture", False)
            if floor_asset.metallic_texture:
                material.set_input_values("metallic_texture_influence", 1.0)
                material.set_input_values("metallic_texture", floor_asset.metallic_texture)
            if floor_asset.ao_texture:
                material.set_input_values("ao_to_diffuse", floor_asset.ao_to_diffuse)
                material.set_input_values("ao_texture", floor_asset.ao_texture)

        if floor_asset.normal_texture:
            material.set_input_values("bump_factor", floor_asset.bump_factor)
            material.set_input_values("normalmap_texture", floor_asset.normal_texture)

        sim_utils.bind_visual_material(prim_path, material_path, stronger_than_descendants=True)
        return f"LocalPBR:{floor_asset.asset_id}"
    except Exception:
        return f"LocalPBR fallback:{floor_asset.asset_id}"
