# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Scene Manager for VLA-SFT data collection.

The VLASSceneManager coordinates scene selection, randomization,
and instruction generation for data collection episodes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..scenes.basic_grasp import BasicGraspSceneConfig, BasicGraspSceneRegistry
    from ..scenes.mobile_grasp import MobileGraspSceneConfig, MobileGraspSceneRegistry


class VLASSceneManager:
    """Manager for VLA-SFT scene selection and randomization.

    This class provides a unified interface for:
    - Sampling scenes from different layers
    - Getting scene-specific randomization parameters
    - Generating instructions for data collection

    Example:
        >>> manager = VLASSceneManager(seed=42)
        >>> scene_config = manager.sample_scene(layer="basic")
        >>> instruction = manager.generate_instruction(scene_config, "cube")
    """

    # Layer names
    LAYER_BASIC = "basic"
    LAYER_MOBILE = "mobile"
    LAYER_INTERACTION = "interaction"
    LAYER_OOD = "ood"

    # Valid layers
    VALID_LAYERS = {LAYER_BASIC, LAYER_MOBILE, LAYER_INTERACTION, LAYER_OOD}

    def __init__(
        self,
        seed: Optional[int] = None,
        enable_basic_layer: bool = True,
        enable_mobile_layer: bool = True,
    ):
        """Initialize the scene manager.

        Args:
            seed: Random seed for reproducible sampling.
            enable_basic_layer: Whether to enable Layer 1 (basic grasp).
            enable_mobile_layer: Whether to enable Layer 2 (mobile grasp).
        """
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed) if seed is not None else np.random

        # Layer registries
        self._basic_registry: Optional[BasicGraspSceneRegistry] = None
        self._mobile_registry: Optional[MobileGraspSceneRegistry] = None

        # Enable layers
        if enable_basic_layer:
            from ..scenes.basic_grasp import BasicGraspSceneRegistry
            self._basic_registry = BasicGraspSceneRegistry(seed=seed)

        if enable_mobile_layer:
            from ..scenes.mobile_grasp import MobileGraspSceneRegistry
            self._mobile_registry = MobileGraspSceneRegistry(seed=seed)

    def sample_scene(
        self,
        layer: str = LAYER_BASIC,
        scene_type: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> Any:
        """Sample a scene configuration.

        Args:
            layer: Scene layer to sample from.
            scene_type: Optional scene type filter (e.g., "a1_ground_grasp").
            scene_id: If specified, return this exact scene instead of sampling.

        Returns:
            Scene configuration object.

        Raises:
            ValueError: If layer is invalid or not enabled.
            KeyError: If scene_id not found.
        """
        if layer not in self.VALID_LAYERS:
            raise ValueError(
                f"Invalid layer '{layer}'. Must be one of: {self.VALID_LAYERS}"
            )

        if scene_id is not None:
            # Get specific scene by ID
            return self.get_scene_by_id(scene_id)

        # Sample scene from layer
        if layer == self.LAYER_BASIC:
            if self._basic_registry is None:
                raise ValueError("Basic layer is not enabled.")
            return self._basic_registry.sample_scene(scene_type)
        elif layer == self.LAYER_MOBILE:
            if self._mobile_registry is None:
                raise ValueError("Mobile layer is not enabled.")
            return self._mobile_registry.sample_scene(scene_type)
        else:
            raise ValueError(
                f"Layer '{layer}' is not yet implemented. "
                f"Currently available layers: Basic (1), Mobile (2)."
            )

    def get_scene_by_id(self, scene_id: str) -> Any:
        """Get a scene by its ID across all registries.

        Args:
            scene_id: Scene identifier (e.g., "a1_ground_grasp_005", "b1_open_floor_003").

        Returns:
            Scene configuration object.

        Raises:
            KeyError: If scene_id not found in any registry.
        """
        # Try basic registry
        if self._basic_registry is not None:
            try:
                return self._basic_registry.get_scene(scene_id)
            except KeyError:
                pass

        # Try mobile registry
        if self._mobile_registry is not None:
            try:
                return self._mobile_registry.get_scene(scene_id)
            except KeyError:
                pass

        raise KeyError(
            f"Scene '{scene_id}' not found in any enabled layer."
        )

    def list_scenes(
        self,
        layer: Optional[str] = None,
        scene_type: Optional[str] = None,
    ) -> List[str]:
        """List available scene IDs.

        Args:
            layer: Filter by layer (returns all if None).
            scene_type: Filter by scene type (returns all if None).

        Returns:
            List of scene IDs.
        """
        if layer is not None and layer not in self.VALID_LAYERS:
            raise ValueError(f"Invalid layer: {layer}")

        all_scenes = []

        if layer == self.LAYER_BASIC or layer is None:
            if self._basic_registry is not None:
                all_scenes.extend(self._basic_registry.list_scenes(scene_type))

        if layer == self.LAYER_MOBILE or layer is None:
            if self._mobile_registry is not None:
                all_scenes.extend(self._mobile_registry.list_scenes(scene_type))

        return all_scenes

    def get_randomization_params(
        self,
        scene_config: Any,
    ) -> Dict[str, Any]:
        """Get randomization parameters for a scene.

        Args:
            scene_config: Scene configuration object.

        Returns:
            Dictionary with randomization parameters. For basic scenes:
            - object_pose, object_color, object_type, clutter_count
            For mobile scenes:
            - target_pose, base_init_pose, obstacles (if applicable)
        """
        params = {}

        # Check if this is a mobile grasp scene
        if hasattr(scene_config, "sample_target_pose"):
            # Mobile grasp scene
            target_pos, target_angle, target_distance = scene_config.sample_target_pose(self._rng)
            base_pos, base_yaw = scene_config.sample_base_init_pose(self._rng)

            params["target_position"] = target_pos
            params["target_angle"] = target_angle
            params["target_distance"] = target_distance
            params["base_init_position"] = base_pos
            params["base_init_yaw"] = base_yaw

            # Sample obstacles for B3 scenes
            if hasattr(scene_config, "sample_obstacles") and scene_config.navigation_required:
                obstacles = scene_config.sample_obstacles(target_pos, self._rng)
                params["obstacles"] = obstacles

            # Sample occluder for B4 scenes
            if hasattr(scene_config, "sample_occluder"):
                occluder = scene_config.sample_occluder(target_pos, self._rng)
                params["occluder"] = occluder

            # Sample object type and color
            if hasattr(scene_config, "sample_object_type"):
                params["object_type"] = scene_config.sample_object_type(self._rng)
            if hasattr(scene_config, "sample_object_color"):
                params["object_color"] = scene_config.sample_object_color(self._rng)

        else:
            # Basic grasp scene
            params["object_pose"] = scene_config.sample_object_pose(self._rng)
            params["object_color"] = scene_config.sample_object_color(self._rng)
            params["object_type"] = scene_config.sample_object_type(self._rng)

            if hasattr(scene_config, "sample_clutter_count"):
                params["clutter_count"] = scene_config.sample_clutter_count(self._rng)

        return params

    def generate_instruction(
        self,
        scene_config: Any,
        object_type: Optional[str] = None,
    ) -> str:
        """Generate an instruction for a scene.

        Args:
            scene_config: Scene configuration object.
            object_type: Override object type (uses sampled if None).

        Returns:
            Instruction string.
        """
        if object_type is None:
            object_type = scene_config.sample_object_type(self._rng)

        return scene_config.generate_instruction(object_type, self._rng)

    @property
    def total_scenes(self) -> int:
        """Total number of available scenes across all layers."""
        count = 0
        if self._basic_registry is not None:
            count += self._basic_registry.total_scenes
        if self._mobile_registry is not None:
            count += self._mobile_registry.total_scenes
        return count

    def get_scene_counts(self) -> Dict[str, int]:
        """Get count of scenes by type."""
        counts = {}
        if self._basic_registry is not None:
            counts.update(self._basic_registry.get_scene_counts())
        if self._mobile_registry is not None:
            counts.update(self._mobile_registry.get_scene_counts())
        return counts


class VLADataBuffer:
    """Buffer for storing VLA-format trajectory data during collection.

    This class temporarily stores episode data before writing to disk.
    """

    def __init__(self, buffer_size: int = 1000):
        """Initialize the data buffer.

        Args:
            buffer_size: Maximum number of timesteps to buffer.
        """
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        """Clear all buffered data."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.instructions = []
        self.metadata = {}

    def add_step(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Add a single timestep to the buffer.

        Args:
            observation: Observation dict with images, proprio, etc.
            action: Action array.
            reward: Reward value.
            done: Done flag.
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def set_instruction(self, instruction: str):
        """Set the instruction for the current episode.

        Args:
            instruction: Natural language instruction string.
        """
        self.instructions.append(instruction)

    def set_metadata(self, key: str, value: Any):
        """Set metadata for the current episode.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value

    def get_episode_data(self) -> Dict[str, Any]:
        """Get all buffered episode data.

        Returns:
            Dictionary with observations, actions, rewards, instructions, metadata.
        """
        return {
            "observations": self.observations,
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "instructions": self.instructions[0] if self.instructions else "",
            "metadata": self.metadata,
        }

    def __len__(self) -> int:
        """Return number of buffered timesteps."""
        return len(self.observations)
