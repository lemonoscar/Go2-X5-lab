# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
MDP functions for scene-based randomization in VLA-SFT data collection.

This module provides custom MDP terms for:
    - Scene-specific object pose sampling
    - Multi-object clutter generation
    - Table surface spawning
    - Color/texture randomization
"""

from __future__ import annotations

import torch
from typing import List, Optional

from isaaclab.envs.mdp import ManagerTermBase, ManagerTermEnvCfg
from isaaclab.managers import SceneEntityCfg


def reset_object_pose_from_scene(
    env: ManagerTermBase,
    scene_config: dict,
    env_ids: Optional[torch.Tensor] = None,
) -> None:
    """Reset object pose based on scene configuration.

    This function samples object position and orientation from the
    scene configuration and applies it to the target object.

    Args:
        env: The environment instance.
        scene_config: Scene configuration dict with position_range and orientation_range.
        env_ids: Environment IDs to reset (resets all if None).
    """
    # Get the object asset
    object_cfg = SceneEntityCfg("object")
    object_articulation = env.scene[object_cfg.name]

    if env_ids is None:
        env_ids = object_articulation.write_data_to_sim_accumulate

    num_resets = len(env_ids)

    # Sample position from scene config
    pos_range = scene_config.get("position_range", {})
    x_range = pos_range.get("x", (-0.1, 0.1))
    y_range = pos_range.get("y", (-0.2, 0.0))
    z_range = pos_range.get("z", (0.02, 0.05))

    # Sample orientation from scene config
    ori_range = scene_config.get("orientation_range", {})
    yaw_range = ori_range.get("yaw", (-3.14159, 3.14159))

    # Generate random poses
    device = object_articulation.data.root_pos_w.device
    positions = torch.zeros((num_resets, 3), device=device)
    positions[:, 0] = torch.rand(num_resets, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    positions[:, 1] = torch.rand(num_resets, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    positions[:, 2] = torch.rand(num_resets, device=device) * (z_range[1] - z_range[0]) + z_range[0]

    # Yaw orientation (quaternion: w, x, y, z)
    yaws = torch.rand(num_resets, device=device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    orientations = torch.zeros((num_resets, 4), device=device)
    orientations[:, 0] = torch.cos(yaws * 0.5)  # w
    orientations[:, 3] = torch.sin(yaws * 0.5)  # z (yaw around z-axis)

    # Write to simulation
    object_articulation.write_root_pose_to_sim(positions, orientations, env_ids)


def reset_object_color(
    env: ManagerTermBase,
    color_options: List[tuple],
    env_ids: Optional[torch.Tensor] = None,
) -> None:
    """Reset object visual color from a set of options.

    Args:
        env: The environment instance.
        color_options: List of (r, g, b) color tuples.
        env_ids: Environment IDs to reset (resets all if None).
    """
    object_cfg = SceneEntityCfg("object")
    object_articulation = env.scene[object_cfg.name]

    if env_ids is None:
        env_ids = object_articulation.write_data_to_sim_accumulate

    num_resets = len(env_ids)

    # Randomly select colors
    device = object_articulation.data.root_pos_w.device
    color_indices = torch.randint(0, len(color_options), (num_resets,), device=device)

    # Apply colors (this would need visual material manipulation)
    # For now, this is a placeholder for color randomization
    # In Isaac Lab, you would typically modify the visual material properties


def spawn_clutter_objects(
    env: ManagerTermBase,
    clutter_count: int,
    clutter_types: List[str],
    clutter_size_range: tuple,
    position_range: dict,
) -> None:
    """Spawn clutter/distractor objects in the scene.

    Args:
        env: The environment instance.
        clutter_count: Number of clutter objects to spawn.
        clutter_types: List of clutter object type names.
        clutter_size_range: (min, max) size range for clutter objects.
        position_range: Dict with 'x', 'y', 'z' tuples for position sampling.
    """
    # This is a placeholder for clutter spawning
    # In a full implementation, this would:
    # 1. Create clutter object assets dynamically
    # 2. Sample positions for each clutter object
    # 3. Ensure no collision with target object
    # 4. Spawn into the scene
    pass


def spawn_table_surface(
    env: ManagerTermBase,
    table_height: float = 0.74,
    table_position: tuple = (0.0, 0.4, 0.0),
) -> None:
    """Spawn a table surface for elevated grasping tasks.

    Args:
        env: The environment instance.
        table_height: Height of the table surface.
        table_position: (x, y, z) position of table center.
    """
    # This is a placeholder for table spawning
    # In a full implementation, this would create a table mesh
    # at the specified position with the specified height
    pass


class SceneResetTermCfg(ManagerTermEnvCfg):
    """Configuration for scene-based reset term."""

    def __post_init__(self):
        # Set the function to our custom reset
        self.func = reset_object_pose_from_scene


# Helper functions for working with scene configs
def get_scene_config_from_registry(scene_id: str, registry) -> dict:
    """Extract scene configuration dict from registry scene config.

    Args:
        scene_id: Scene identifier.
        registry: BasicGraspSceneRegistry instance.

    Returns:
        Dictionary with position_range, orientation_range, etc.
    """
    scene = registry.get_scene(scene_id)

    return {
        "position_range": scene.position_range,
        "orientation_range": scene.orientation_range,
        "object_types": scene.object_types,
        "object_size_range": scene.object_size_range,
        "clutter_enabled": scene.clutter_enabled,
        "clutter_count": scene.sample_clutter_count() if scene.clutter_enabled else 0,
        "clutter_object_types": scene.clutter_object_types,
        "clutter_position_range": scene.clutter_position_range,
        "clutter_min_separation": scene.clutter_min_separation,
        "clutter_target_separation": scene.clutter_target_separation,
        "distractor_size_range": getattr(scene, "distractor_size_range", scene.object_size_range),
        "table_position": scene.table_position,
        "table_size": scene.table_size,
        "table_layouts": scene.get_table_layouts() if hasattr(scene, "get_table_layouts") else [],
        "floor_material_types": scene.floor_material_types,
        "scene_type": scene.scene_type,
    }
