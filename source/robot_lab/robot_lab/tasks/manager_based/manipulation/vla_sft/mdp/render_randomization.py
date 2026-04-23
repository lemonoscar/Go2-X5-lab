"""Runtime rendering helpers for local Layer 1 floor materials."""

from __future__ import annotations

import random

from isaaclab.envs import ManagerBasedEnv

from ..render_assets import bind_local_floor_material, choose_local_floor_asset


def apply_floor_visual_material(
    env: ManagerBasedEnv,
    env_ids,
    floor_material: str | None = None,
    prim_path: str = "/World/FloorVisual",
    material_path: str = "/World/Looks/FloorVisualLocalPBR",
) -> None:
    """Bind a local OmniPBR material to the shared Layer 1 floor visual if available."""

    del env_ids  # Shared world asset, not per-environment.

    resolved_floor_material = floor_material or env.cfg.vla_sft.get("selected_floor_material")
    if not resolved_floor_material:
        return

    scene_seed = env.cfg.vla_sft.get("scene_seed")
    rng = random.Random(None if scene_seed is None else int(scene_seed) + 101)
    local_floor_asset = choose_local_floor_asset(resolved_floor_material, rng=rng)
    if local_floor_asset is None:
        return

    source = bind_local_floor_material(prim_path, local_floor_asset, rng=rng, material_path=material_path)
    env.cfg.vla_sft["selected_floor_material_source"] = source
