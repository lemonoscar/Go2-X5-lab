# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Mobile grasp scene definitions for Layer 2 of VLA-SFT data collection.

This module defines scene configurations for mobile grasping tasks:
    - B1: Open floor navigation grasp
    - B2: Constrained table approach
    - B3: Obstacle avoidance navigation
    - B4: Partial occlusion grasp
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MobileGraspSceneConfig:
    """Base configuration for mobile grasp scenes.

    Attributes:
        scene_id: Unique identifier for the scene
        scene_type: Type of scene (b1_open_floor, b2_table_approach, etc.)
        layer: Layer identifier ("mobile")
        target_distance_range: (min, max) distance to target in meters
        target_angle_range: (min, max) angle to target in radians
        navigation_required: Whether base navigation is needed
        base_init_pose_range: Dict for base initial pose sampling
        obstacles: List of obstacle configurations
        occluder_config: Optional occlusion configuration
        instructions: List of instruction templates
    """

    scene_id: str = "mobile_grasp_000"
    scene_type: str = "b1_open_floor"
    layer: str = "mobile"

    # Target configuration
    target_distance_range: Tuple[float, float] = (1.5, 3.0)
    target_angle_range: Tuple[float, float] = (-1.57, 1.57)
    target_height_range: Tuple[float, float] = (0.02, 0.05)

    # Base configuration
    base_init_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
    )

    # Navigation settings
    navigation_required: bool = True
    path_planning_required: bool = False
    approach_constrained: bool = False

    # Obstacles
    obstacles: List[Dict] = field(default_factory=list)

    # Occlusion
    occluder_config: Optional[Dict] = None

    # Object types (inherit from basic)
    object_types: List[str] = field(
        default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"]
    )
    object_size_range: Tuple[float, float] = (0.04, 0.06)

    # Instructions
    instructions: List[str] = field(
        default_factory=lambda: [
            "navigate to the {object} and pick it up",
            "go to the {object} and grasp it",
            "move towards the {object} and collect it",
        ]
    )

    def sample_target_pose(
        self,
        rng: Optional[random.Random] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Sample target position and angle.

        Args:
            rng: Random number generator

        Returns:
            (position, angle, distance) where position is (3,) and
            angle is the yaw angle from base to target, distance is linear distance.
        """
        if rng is None:
            rng = random

        # Sample distance and angle
        distance = rng.uniform(*self.target_distance_range)
        angle = rng.uniform(*self.target_angle_range)
        height = rng.uniform(*self.target_height_range)

        # Compute position (assuming base at origin)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)

        pos = np.array([x, y, height])

        return pos, angle, distance

    def sample_base_init_pose(
        self,
        rng: Optional[random.Random] = None,
    ) -> Tuple[np.ndarray, float]:
        """Sample base initial pose.

        Args:
            rng: Random number generator

        Returns:
            (position, yaw) where position is (2,) for (x, y)
        """
        if rng is None:
            rng = random

        pos = np.array([
            rng.uniform(*self.base_init_position_range["x"]),
            rng.uniform(*self.base_init_position_range["y"]),
        ])
        yaw = rng.uniform(*self.base_init_position_range["yaw"])

        return pos, yaw

    def generate_instruction(
        self,
        object_type: str,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Generate instruction for this scene."""
        if rng is None:
            rng = random
        template = rng.choice(self.instructions)
        return template.format(object=object_type)


@dataclass
class MobileGraspSceneB1(MobileGraspSceneConfig):
    """B1: Open floor navigation grasp.

    Target object at distance with no obstacles.
    Focus on long-range navigation and base positioning.
    """

    scene_type: str = "b1_open_floor"

    target_distance_range: Tuple[float, float] = (1.5, 3.0)
    target_angle_range: Tuple[float, float] = (-1.57, 1.57)

    base_init_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
    )

    navigation_required: bool = True
    path_planning_required: bool = False
    approach_constrained: bool = False

    obstacles: List[Dict] = field(default_factory=list)

    instructions: List[str] = field(
        default_factory=lambda: [
            "navigate to the {object} and pick it up",
            "go to the {object} and grasp it",
            "move towards the {object} and collect it",
        ]
    )


@dataclass
class MobileGraspSceneB2(MobileGraspSceneConfig):
    """B2: Constrained table approach.

    Target on table with limited space around it.
    Focus on precise base positioning.
    """

    scene_type: str = "b2_table_approach"

    # Table configuration
    table_position: Tuple[float, float, float] = (0.0, 0.5, 0.0)
    table_size: Tuple[float, float, float] = (0.8, 0.6, 0.74)
    table_leg_positions: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.34, 0.17),
            (0.34, 0.83),
            (-0.34, 0.17),
            (-0.34, 0.83),
        ]
    )

    # Target on table
    target_distance_range: Tuple[float, float] = (0.8, 1.2)
    target_angle_range: Tuple[float, float] = (-0.5, 0.5)

    target_height_range: Tuple[float, float] = (0.76, 0.80)

    base_init_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, -0.2),
            "y": (-0.5, 0.2),
            "yaw": (0.0, 0.5),
        }
    )

    navigation_required: bool = True
    approach_constrained: bool = True
    min_table_distance: float = 0.3

    instructions: List[str] = field(
        default_factory=lambda: [
            "approach the table and pick up the {object}",
            "go to the table and grasp the {object}",
            "move to the table and collect the {object}",
        ]
    )


@dataclass
class MobileGraspSceneB3(MobileGraspSceneConfig):
    """B3: Obstacle avoidance navigation.

    Static obstacles in the scene requiring path planning.
    Focus on navigation and obstacle avoidance.
    """

    scene_type: str = "b3_obstacle_avoidance"

    # Obstacle templates
    obstacle_templates: List[Dict] = field(
        default_factory=lambda: [
            {
                "type": "box",
                "size": (0.3, 0.3, 0.3),
                "position_range": {"x": (0.3, 0.8), "y": (0.1, 0.4)},
            },
            {
                "type": "box",
                "size": (0.2, 0.2, 0.2),
                "position_range": {"x": (0.3, 0.8), "y": (0.1, 0.4)},
            },
            {
                "type": "cylinder_barrier",
                "size": (0.15, 0.3),
                "position_range": {"x": (0.5, 0.9), "y": (0.2, 0.6)},
            },
        ]
    )

    obstacle_count_range: Tuple[int, int] = (2, 4)

    target_distance_range: Tuple[float, float] = (1.5, 2.5)
    target_angle_range: Tuple[float, float] = (-1.0, 1.0)

    target_height_range: Tuple[float, float] = (0.02, 0.05)

    base_init_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.0),
            "y": (-0.5, 0.5),
            "yaw": (0.0, 0.5),
        }
    )

    navigation_required: bool = True
    path_planning_required: bool = True

    def sample_obstacles(
        self,
        target_pos: np.ndarray,
        rng: Optional[random.Random] = None,
    ) -> List[Dict]:
        """Sample obstacle positions avoiding the target area.

        Args:
            target_pos: Target object position
            rng: Random number generator

        Returns:
            List of obstacle configurations
        """
        if rng is None:
            rng = random

        obstacles = []
        count = rng.randint(*self.obstacle_count_range)

        for _ in range(count):
            template = rng.choice(self.obstacle_templates)
            obs_type = template["type"]
            size = template["size"]
            pos_range = template["position_range"]

            # Sample position
            pos = np.array([
                rng.uniform(*pos_range["x"]),
                rng.uniform(*pos_range["y"]),
                0.0,  # on ground
            ])

            # Avoid placing too close to target
            if np.linalg.norm(pos[:2] - target_pos[:2]) < 0.3:
                # Try a different position
                pos = np.array([
                    rng.uniform(*pos_range["x"]),
                    rng.uniform(*pos_range["y"]),
                    0.0,
                ])

            obstacle = {
                "type": obs_type,
                "size": size,
                "position": pos,
            }
            obstacles.append(obstacle)

        return obstacles


@dataclass
class MobileGraspSceneB4(MobileGraspSceneConfig):
    """B4: Partial occlusion grasp.

    Target partially occluded by an obstacle.
    Focus on viewpoint adjustment and inference.
    """

    scene_type: str = "b4_partial_occlusion"

    # Occluder configuration
    occluder_type: str = "tall_block"
    occluder_size: Tuple[float, float, float] = (0.2, 0.2, 0.3)
    occluder_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.6, 0.8),
            "y": (-0.1, 0.1),
        }
    )

    # Target is behind/near occluder
    occlusion_ratio_range: Tuple[float, float] = (0.3, 0.6)

    target_distance_range: Tuple[float, float] = (0.8, 1.5)
    target_angle_range: Tuple[float, float] = (-0.5, 0.5)
    target_height_range: Tuple[float, float] = (0.02, 0.05)

    base_init_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (0.0, 0.5),
        }
    )

    navigation_required: bool = True

    def sample_occluder(
        self,
        target_pos: np.ndarray,
        rng: Optional[random.Random] = None,
    ) -> Dict:
        """Sample occluder position based on target.

        Args:
            target_pos: Target object position
            rng: Random number generator

        Returns:
            Occluder configuration
        """
        if rng is None:
            rng = random

        # Place occluder between base and target
        direction = target_pos[:2]
        direction = direction / np.linalg.norm(direction)

        # Occluder at 40-70% of the distance
        distance = np.linalg.norm(target_pos[:2]) * rng.uniform(0.4, 0.7)

        occluder_pos = target_pos.copy()
        occluder_pos[:2] = direction * distance
        occluder_pos[2] = 0.0

        return {
            "type": self.occluder_type,
            "size": self.occluder_size,
            "position": occluder_pos,
        }


class MobileGraspSceneRegistry:
    """Registry for mobile grasp scene configurations."""

    _SCENE_TEMPLATES = {
        "b1_open_floor": MobileGraspSceneB1,
        "b2_table_approach": MobileGraspSceneB2,
        "b3_obstacle_avoidance": MobileGraspSceneB3,
        "b4_partial_occlusion": MobileGraspSceneB4,
    }

    _SCENE_COUNTS = {
        "b1_open_floor": 10,
        "b2_table_approach": 10,
        "b3_obstacle_avoidance": 10,
        "b4_partial_occlusion": 10,
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the scene registry."""
        self._rng = random.Random(seed)
        self._scenes: Dict[str, MobileGraspSceneConfig] = {}
        self._initialize_scenes()

    def _initialize_scenes(self):
        """Initialize all scene configurations."""
        scene_idx = 0
        for scene_type, scene_class in self._SCENE_TEMPLATES.items():
            count = self._SCENE_COUNTS[scene_type]
            for i in range(count):
                scene_id = f"{scene_type}_{i:03d}"
                scene = scene_class(scene_id=scene_id)
                self._scenes[scene_id] = scene
                scene_idx += 1

    def list_scenes(self, scene_type: Optional[str] = None) -> List[str]:
        """List available scene IDs."""
        if scene_type is None:
            return list(self._scenes.keys())
        return [sid for sid in self._scenes.keys() if sid.startswith(scene_type)]

    def get_scene(self, scene_id: str) -> MobileGraspSceneConfig:
        """Get a scene configuration by ID."""
        if scene_id not in self._scenes:
            available = self.list_scenes()
            raise KeyError(
                f"Scene '{scene_id}' not found. "
                f"Available: {available}"
            )
        return self._scenes[scene_id]

    def sample_scene(self, scene_type: Optional[str] = None) -> MobileGraspSceneConfig:
        """Sample a random scene configuration."""
        candidates = self.list_scenes(scene_type)
        if not candidates:
            raise ValueError(f"No scenes for type: {scene_type}")
        scene_id = self._rng.choice(candidates)
        return self.get_scene(scene_id)

    @property
    def total_scenes(self) -> int:
        return len(self._scenes)

    def get_scene_counts(self) -> Dict[str, int]:
        """Get count of scenes by type."""
        counts = {}
        for scene_type in self._SCENE_TEMPLATES.keys():
            counts[scene_type] = len(self.list_scenes(scene_type))
        return counts


# Singleton instance
_default_registry: Optional[MobileGraspSceneRegistry] = None


def get_default_registry() -> MobileGraspSceneRegistry:
    """Get the default scene registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MobileGraspSceneRegistry()
    return _default_registry
