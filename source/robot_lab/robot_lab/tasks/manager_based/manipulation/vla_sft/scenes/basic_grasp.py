# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Basic grasp scene definitions for Layer 1 of VLA-SFT data collection.

This module defines scene configurations for basic grasping tasks:
    - A1: Single object ground grasp
    - A2: Single object table grasp
    - A3: Multi-object simple clutter
    - A4: Multi-height table clutter
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MIN_WORKSPACE_X = 0.80
TABLE_SURFACE_MARGIN = 0.07


@dataclass
class BasicGraspSceneConfig:
    """Base configuration for basic grasp scenes.

    Attributes:
        scene_id: Unique identifier for the scene
        scene_type: Type of scene (a1_ground, a2_table, a3_clutter)
        object_types: List of object type names
        object_size_range: (min, max) size range for objects (meters)
        position_range: Dict with 'x', 'y', 'z' tuples for position sampling
        orientation_range: Dict with 'roll', 'pitch', 'yaw' tuples for orientation
        base_init_pose: Dict for base initial pose sampling
        clutter_enabled: Whether to add distractor objects
        clutter_count_range: (min, max) number of distractor objects
        instructions: List of instruction templates for this scene
    """

    scene_id: str = "basic_grasp_000"
    scene_type: str = "a1_ground"
    layer: str = "basic"

    # Object configuration
    object_types: List[str] = field(default_factory=lambda: ["cube"])
    object_size_range: Tuple[float, float] = (0.03, 0.06)

    # Position randomization
    position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.15, 0.15),
            "y": (-0.25, 0.0),
            "z": (0.02, 0.05),
        }
    )

    # Orientation randomization (radians)
    orientation_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-3.14159, 3.14159),
        }
    )

    # Base initial pose
    base_init_pose: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )

    # Clutter configuration
    clutter_enabled: bool = False
    clutter_count_range: Tuple[int, int] = (0, 0)
    clutter_object_types: List[str] = field(default_factory=list)

    # Instruction templates (use {object} as placeholder)
    instructions: List[str] = field(
        default_factory=lambda: [
            "pick up the {object} from the ground",
            "grasp the {object}",
        ]
    )

    # Visual configuration
    object_colors: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.82, 0.16, 0.12),  # Red
            (0.12, 0.52, 0.82),  # Blue
            (0.22, 0.72, 0.22),  # Green
            (0.82, 0.72, 0.12),  # Yellow
        ]
    )
    clutter_position_range: Optional[Dict[str, Tuple[float, float]]] = None
    clutter_min_separation: float = 0.14
    clutter_target_separation: float = 0.16
    preview_camera_eye: Tuple[float, float, float] = (2.35, -2.0, 1.4)
    preview_camera_target: Tuple[float, float, float] = (0.70, 0.0, 0.18)
    table_position: Optional[Tuple[float, float, float]] = None
    table_size: Optional[Tuple[float, float, float]] = None
    table_layouts: List[Dict[str, Any]] = field(default_factory=list)
    table_surface_margin: float = TABLE_SURFACE_MARGIN
    floor_material_types: List[str] = field(
        default_factory=lambda: ["concrete", "wood", "tile", "grass"]
    )

    def sample_object_pose(
        self, rng: Optional[random.Random] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random object position and orientation.

        Args:
            rng: Random number generator (uses global if None)

        Returns:
            (position, orientation_quat) where position is (3,) and
            orientation_quat is (4,) in (w,x,y,z) format.
        """
        if rng is None:
            rng = random

        # Sample position
        pos = np.array([
            rng.uniform(*self.position_range["x"]),
            rng.uniform(*self.position_range["y"]),
            rng.uniform(*self.position_range["z"]),
        ])

        # Sample orientation (Euler angles)
        roll = rng.uniform(*self.orientation_range["roll"])
        pitch = rng.uniform(*self.orientation_range["pitch"])
        yaw = rng.uniform(*self.orientation_range["yaw"])

        # Convert to quaternion (w, x, y, z)
        # Using small angle approximation for roll/pitch when near zero
        quat = self._euler_to_quat(roll, pitch, yaw)

        return pos, quat

    def sample_base_pose(
        self, rng: Optional[random.Random] = None
    ) -> Tuple[np.ndarray, float]:
        """Sample a random base initial pose.

        Args:
            rng: Random number generator (uses global if None)

        Returns:
            (position, yaw) where position is (2,) for (x, y)
        """
        if rng is None:
            rng = random

        pos = np.array([
            rng.uniform(*self.base_init_pose["x"]),
            rng.uniform(*self.base_init_pose["y"]),
        ])
        yaw = rng.uniform(*self.base_init_pose["yaw"])

        return pos, yaw

    def sample_object_color(
        self, rng: Optional[random.Random] = None
    ) -> Tuple[float, float, float]:
        """Sample a random object color.

        Args:
            rng: Random number generator (uses global if None)

        Returns:
            (r, g, b) color tuple in [0, 1] range.
        """
        if rng is None:
            rng = random
        return rng.choice(self.object_colors)

    def sample_object_type(
        self, rng: Optional[random.Random] = None
    ) -> str:
        """Sample a random object type.

        Args:
            rng: Random number generator (uses global if None)

        Returns:
            Object type name string.
        """
        if rng is None:
            rng = random
        return rng.choice(self.object_types)

    def sample_object_size(
        self, rng: Optional[random.Random] = None
    ) -> float:
        """Sample a target object size scalar in meters."""
        if rng is None:
            rng = random
        return rng.uniform(*self.object_size_range)

    def sample_clutter_count(
        self, rng: Optional[random.Random] = None
    ) -> int:
        """Sample number of clutter objects.

        Args:
            rng: Random number generator (uses global if None)

        Returns:
            Number of clutter objects.
        """
        if not self.clutter_enabled:
            return 0
        if rng is None:
            rng = random
        min_count, max_count = self.clutter_count_range
        return rng.randint(min_count, max_count + 1)

    def sample_floor_material(
        self, rng: Optional[random.Random] = None
    ) -> str:
        """Sample a floor material type for this scene."""
        if rng is None:
            rng = random
        return rng.choice(self.floor_material_types)

    def get_table_layouts(self) -> List[Dict[str, Any]]:
        """Return all tables that belong to this scene."""
        if self.table_layouts:
            return [dict(layout) for layout in self.table_layouts]
        if self.table_position is not None and self.table_size is not None:
            return [
                {
                    "name": "table_0",
                    "position": self.table_position,
                    "size": self.table_size,
                    "color": (0.60, 0.40, 0.20),
                }
            ]
        return []

    def sample_tabletop_position(
        self,
        table_layout: Dict[str, Any],
        rng: Optional[random.Random] = None,
        z_offset_range: Tuple[float, float] = (0.004, 0.012),
    ) -> np.ndarray:
        """Sample an (x, y, z) point on top of a table surface."""
        if rng is None:
            rng = random

        center_x, center_y, center_z = table_layout["position"]
        size_x, size_y, size_z = table_layout["size"]
        margin = min(self.table_surface_margin, size_x * 0.22, size_y * 0.22)
        x_min = center_x - size_x * 0.5 + margin
        x_max = center_x + size_x * 0.5 - margin
        y_min = center_y - size_y * 0.5 + margin
        y_max = center_y + size_y * 0.5 - margin

        if x_min > x_max:
            x_min = x_max = center_x
        if y_min > y_max:
            y_min = y_max = center_y

        return np.array(
            [
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max),
                center_z + size_z + rng.uniform(*z_offset_range),
            ]
        )

    def surface_height_for_position(self, pos: np.ndarray | Tuple[float, float, float]) -> float:
        """Return the support surface z-height for the given position."""
        pos_arr = np.asarray(pos, dtype=float)
        table_layouts = self.get_table_layouts()
        candidate_heights: List[Tuple[float, float]] = []

        for table_layout in table_layouts:
            center_x, center_y, center_z = table_layout["position"]
            size_x, size_y, size_z = table_layout["size"]
            if abs(pos_arr[0] - center_x) <= size_x * 0.5 and abs(pos_arr[1] - center_y) <= size_y * 0.5:
                top_z = center_z + size_z
                candidate_heights.append((abs(pos_arr[2] - top_z), top_z))

        if candidate_heights:
            return min(candidate_heights, key=lambda item: item[0])[1]

        if float(pos_arr[2]) > 0.2 and table_layouts:
            nearest_table = min(
                table_layouts,
                key=lambda layout: abs(pos_arr[2] - (layout["position"][2] + layout["size"][2])),
            )
            return float(nearest_table["position"][2] + nearest_table["size"][2])

        return 0.0

    def sample_clutter_type(
        self, rng: Optional[random.Random] = None
    ) -> str:
        """Sample a clutter object type."""
        if rng is None:
            rng = random
        candidates = self.clutter_object_types or self.object_types
        return rng.choice(candidates)

    def sample_clutter_size(
        self, rng: Optional[random.Random] = None
    ) -> float:
        """Sample a clutter object size scalar in meters."""
        if rng is None:
            rng = random
        size_range = getattr(self, "distractor_size_range", self.object_size_range)
        return rng.uniform(*size_range)

    def sample_clutter_orientation(
        self,
        rng: Optional[random.Random] = None,
    ) -> np.ndarray:
        """Sample a planar orientation for clutter objects."""
        if rng is None:
            rng = random
        yaw = rng.uniform(*self.orientation_range["yaw"])
        return self._euler_to_quat(0.0, 0.0, yaw)

    @staticmethod
    def _sample_position_from_range(
        pos_range: Dict[str, Tuple[float, float]],
        rng: random.Random,
    ) -> np.ndarray:
        """Sample a random position from a range dict."""
        return np.array(
            [
                rng.uniform(*pos_range["x"]),
                rng.uniform(*pos_range["y"]),
                rng.uniform(*pos_range["z"]),
            ]
        )

    @staticmethod
    def _accept_position(
        candidate: np.ndarray,
        target_pos: np.ndarray,
        positions: List[np.ndarray],
        min_sep: float,
        target_sep: float,
    ) -> bool:
        """Return whether a candidate satisfies spacing constraints."""
        if np.linalg.norm(candidate[:2] - target_pos[:2]) < target_sep:
            return False
        if any(np.linalg.norm(candidate[:2] - existing[:2]) < min_sep for existing in positions):
            return False
        return True

    def _generate_jittered_candidates(
        self,
        pos_range: Dict[str, Tuple[float, float]],
        count: int,
        rng: random.Random,
    ) -> List[np.ndarray]:
        """Generate irregular fallback candidates without visible grid alignment."""
        x_min, x_max = pos_range["x"]
        y_min, y_max = pos_range["y"]
        z_min, z_max = pos_range["z"]

        x_points = max(3, int(np.ceil(np.sqrt((count + 2) * 1.6))))
        y_points = max(3, int(np.ceil((count + 2) * 1.6 / x_points)))
        cell_x = (x_max - x_min) / x_points
        cell_y = (y_max - y_min) / y_points

        candidates: List[np.ndarray] = []
        for ix in range(x_points):
            for iy in range(y_points):
                x_center = x_min + (ix + 0.5) * cell_x
                y_center = y_min + (iy + 0.5) * cell_y
                jitter_x = rng.uniform(-0.42 * cell_x, 0.42 * cell_x)
                jitter_y = rng.uniform(-0.42 * cell_y, 0.42 * cell_y)
                candidates.append(
                    np.array(
                        [
                            np.clip(x_center + jitter_x, x_min, x_max),
                            np.clip(y_center + jitter_y, y_min, y_max),
                            rng.uniform(z_min, z_max),
                        ]
                    )
                )

        extra_random = max(count * 3, 12)
        for _ in range(extra_random):
            candidates.append(self._sample_position_from_range(pos_range, rng))

        rng.shuffle(candidates)
        return candidates

    def sample_clutter_positions(
        self,
        count: int,
        target_pos: np.ndarray,
        rng: Optional[random.Random] = None,
    ) -> List[np.ndarray]:
        """Sample clutter positions with minimum spacing from each other and the target."""
        if count <= 0:
            return []
        if rng is None:
            rng = random

        pos_range = self.clutter_position_range or self.position_range
        min_sep = self.clutter_min_separation
        target_sep = max(min_sep, self.clutter_target_separation)
        positions: List[np.ndarray] = []

        for _ in range(count * 200):
            if len(positions) >= count:
                break
            candidate = self._sample_position_from_range(pos_range, rng)
            if self._accept_position(candidate, target_pos, positions, min_sep, target_sep):
                positions.append(candidate)

        if len(positions) < count:
            for candidate in self._generate_jittered_candidates(pos_range, count, rng):
                if len(positions) >= count:
                    break
                if self._accept_position(candidate, target_pos, positions, min_sep, target_sep):
                    positions.append(candidate)

        return positions[:count]

    def generate_instruction(
        self,
        object_type: str,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Generate a natural language instruction for this scene.

        Args:
            object_type: The type of target object (e.g., "cube", "sphere")
            rng: Random number generator (uses global if None)

        Returns:
            Instruction string with placeholders replaced.
        """
        if rng is None:
            rng = random
        template = rng.choice(self.instructions)
        return template.format(object=object_type)

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion (w, x, y, z).

        Uses ZYX convention (yaw-pitch-roll).
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz])


@dataclass
class BasicGraspSceneA1(BasicGraspSceneConfig):
    """A1: Single object ground grasp.

    Target object on the ground with minimal obstacles.
    Focus on basic grasp primitive learning.
    """

    scene_type: str = "a1_ground_grasp"

    object_types: List[str] = field(default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"])
    object_size_range: Tuple[float, float] = (0.04, 0.09)

    position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (MIN_WORKSPACE_X, 1.08),
            "y": (-0.22, 0.22),
            "z": (0.025, 0.05),
        }
    )

    base_init_pose: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )

    clutter_enabled: bool = False

    instructions: List[str] = field(
        default_factory=lambda: [
            "pick up the {object} from the ground",
            "grasp the {object}",
            "collect the {object} from the floor",
        ]
    )
    preview_camera_target: Tuple[float, float, float] = (0.96, 0.0, 0.16)


@dataclass
class BasicGraspSceneA2(BasicGraspSceneConfig):
    """A2: Single object table grasp.

    Target object on a table at fixed height.
    Focus on grasp at different elevations.
    """

    scene_type: str = "a2_table_grasp"
    table_height: float = 0.74

    object_types: List[str] = field(
        default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"]
    )

    object_size_range: Tuple[float, float] = (0.04, 0.10)

    position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.94, 1.32),
            "y": (-0.22, 0.22),
            "z": (0.76, 0.80),  # On top of table
        }
    )

    base_init_pose: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )

    clutter_enabled: bool = False

    instructions: List[str] = field(
        default_factory=lambda: [
            "pick up the {object} from the table",
            "grasp the {object} on the table",
            "take the {object} from the table",
        ]
    )
    preview_camera_target: Tuple[float, float, float] = (1.14, 0.0, 0.76)
    table_position: Tuple[float, float, float] = (1.20, 0.0, 0.0)
    table_size: Tuple[float, float, float] = (0.80, 0.60, 0.74)
    floor_material_types: List[str] = field(
        default_factory=lambda: ["concrete", "wood", "tile"]
    )


@dataclass
class BasicGraspSceneA3(BasicGraspSceneConfig):
    """A3: Multi-object simple clutter.

    Target object among 3-5 distractor objects.
    Focus on grasp with mild occlusion/clutter.
    """

    scene_type: str = "a3_simple_clutter"

    # Target object configuration
    object_types: List[str] = field(default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"])
    object_size_range: Tuple[float, float] = (0.04, 0.10)

    # Allow target on ground or low table
    position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (MIN_WORKSPACE_X + 0.02, 1.10),
            "y": (-0.24, 0.24),
            "z": (0.025, 0.05),
        }
    )

    base_init_pose: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )

    # Clutter configuration
    clutter_enabled: bool = True
    clutter_count_range: Tuple[int, int] = (7, 10)
    clutter_object_types: List[str] = field(
        default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"]
    )
    clutter_position_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (MIN_WORKSPACE_X, 1.34),
            "y": (-0.48, 0.48),
            "z": (0.025, 0.05),
        }
    )
    clutter_min_separation: float = 0.15
    clutter_target_separation: float = 0.18

    # Distractors are slightly smaller
    distractor_size_range: Tuple[float, float] = (0.025, 0.08)

    instructions: List[str] = field(
        default_factory=lambda: [
            "pick up the {object} among the other objects on the ground",
            "grasp the {object} from the cluttered floor",
            "find and collect the {object} from the ground",
        ]
    )
    preview_camera_target: Tuple[float, float, float] = (1.00, 0.0, 0.16)


@dataclass
class BasicGraspSceneA4(BasicGraspSceneConfig):
    """A4: Multi-height table clutter.

    Different-height tables hold a mix of objects. The target must be selected
    and grasped from a cluttered tabletop workspace.
    """

    scene_type: str = "a4_multi_height_table_clutter"

    object_types: List[str] = field(default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"])
    object_size_range: Tuple[float, float] = (0.04, 0.10)
    clutter_enabled: bool = True
    clutter_count_range: Tuple[int, int] = (9, 13)
    clutter_object_types: List[str] = field(
        default_factory=lambda: ["cube", "sphere", "cylinder", "bowl", "cup"]
    )
    distractor_size_range: Tuple[float, float] = (0.03, 0.09)
    clutter_min_separation: float = 0.16
    clutter_target_separation: float = 0.20
    preview_camera_eye: Tuple[float, float, float] = (2.95, -2.45, 1.65)
    preview_camera_target: Tuple[float, float, float] = (1.26, 0.0, 0.80)
    floor_material_types: List[str] = field(
        default_factory=lambda: ["concrete", "wood", "tile"]
    )
    table_layouts: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "low_left",
                "position": (1.08, -0.86, 0.0),
                "size": (0.52, 0.48, 0.52),
                "color": (0.54, 0.39, 0.24),
            },
            {
                "name": "mid_center",
                "position": (1.42, 0.0, 0.0),
                "size": (0.76, 0.58, 0.72),
                "color": (0.62, 0.46, 0.28),
            },
            {
                "name": "high_right",
                "position": (1.18, 0.86, 0.0),
                "size": (0.56, 0.48, 0.92),
                "color": (0.48, 0.35, 0.22),
            },
        ]
    )

    instructions: List[str] = field(
        default_factory=lambda: [
            "pick up the {object} from the cluttered tables",
            "find the {object} across the different-height tables and grasp it",
            "collect the {object} from the mixed tabletop clutter",
        ]
    )

    def sample_object_pose(
        self, rng: Optional[random.Random] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the target object on top of one of the tables."""
        if rng is None:
            rng = random

        target_table = rng.choice(self.get_table_layouts())
        pos = self.sample_tabletop_position(target_table, rng)
        yaw = rng.uniform(*self.orientation_range["yaw"])
        quat = self._euler_to_quat(0.0, 0.0, yaw)
        return pos, quat

    def sample_clutter_positions(
        self,
        count: int,
        target_pos: np.ndarray,
        rng: Optional[random.Random] = None,
    ) -> List[np.ndarray]:
        """Sample clutter positions across multiple table surfaces."""
        if count <= 0:
            return []
        if rng is None:
            rng = random

        tables = self.get_table_layouts()
        positions: List[np.ndarray] = []

        for _ in range(count * 240):
            if len(positions) >= count:
                break
            candidate = self.sample_tabletop_position(rng.choice(tables), rng)
            if self._accept_position(
                candidate,
                target_pos,
                positions,
                self.clutter_min_separation,
                self.clutter_target_separation,
            ):
                positions.append(candidate)

        if len(positions) < count:
            fallback_points: List[np.ndarray] = []
            for table_layout in tables:
                samples_per_table = max(10, count * 4)
                for _ in range(samples_per_table):
                    fallback_points.append(self.sample_tabletop_position(table_layout, rng))

            rng.shuffle(fallback_points)
            for candidate in fallback_points:
                if len(positions) >= count:
                    break
                if self._accept_position(
                    candidate,
                    target_pos,
                    positions,
                    self.clutter_min_separation,
                    self.clutter_target_separation,
                ):
                    positions.append(candidate)

        return positions[:count]


class BasicGraspSceneRegistry:
    """Registry for all basic grasp scene configurations.

    This class manages the available scene configurations and provides
    methods to sample scenes by type or ID.
    """

    # Scene templates for each type
    _SCENE_TEMPLATES = {
        "a1_ground_grasp": BasicGraspSceneA1,
        "a2_table_grasp": BasicGraspSceneA2,
        "a3_simple_clutter": BasicGraspSceneA3,
        "a4_multi_height_table_clutter": BasicGraspSceneA4,
    }

    # Number of scenes per type (for MVP)
    _SCENE_COUNTS = {
        "a1_ground_grasp": 10,
        "a2_table_grasp": 10,
        "a3_simple_clutter": 10,
        "a4_multi_height_table_clutter": 10,
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the scene registry.

        Args:
            seed: Random seed for reproducible scene sampling.
        """
        self._rng = random.Random(seed)
        self._scenes: Dict[str, BasicGraspSceneConfig] = {}
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
        """List available scene IDs.

        Args:
            scene_type: Filter by scene type (returns all if None)

        Returns:
            List of scene IDs.
        """
        if scene_type is None:
            return list(self._scenes.keys())
        return [sid for sid in self._scenes.keys() if sid.startswith(scene_type)]

    def get_scene(self, scene_id: str) -> BasicGraspSceneConfig:
        """Get a scene configuration by ID.

        Args:
            scene_id: Scene identifier

        Returns:
            Scene configuration

        Raises:
            KeyError: If scene_id not found
        """
        if scene_id not in self._scenes:
            available = self.list_scenes()
            raise KeyError(
                f"Scene '{scene_id}' not found. "
                f"Available scenes: {available}"
            )
        return self._scenes[scene_id]

    def sample_scene(
        self, scene_type: Optional[str] = None
    ) -> BasicGraspSceneConfig:
        """Sample a random scene configuration.

        Args:
            scene_type: Scene type to sample from (samples from all if None)

        Returns:
            Scene configuration
        """
        candidates = self.list_scenes(scene_type)
        if not candidates:
            raise ValueError(f"No scenes available for type: {scene_type}")
        scene_id = self._rng.choice(candidates)
        return self.get_scene(scene_id)

    @property
    def total_scenes(self) -> int:
        """Total number of registered scenes."""
        return len(self._scenes)

    def get_scene_counts(self) -> Dict[str, int]:
        """Get count of scenes by type."""
        counts = {}
        for scene_type in self._SCENE_TEMPLATES.keys():
            counts[scene_type] = len(self.list_scenes(scene_type))
        return counts


# Singleton instance for convenient access
_default_registry: Optional[BasicGraspSceneRegistry] = None


def get_default_registry() -> BasicGraspSceneRegistry:
    """Get the default scene registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = BasicGraspSceneRegistry()
    return _default_registry
