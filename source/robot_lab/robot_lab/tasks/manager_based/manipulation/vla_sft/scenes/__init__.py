# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Scene definitions for VLA-SFT data collection."""

from .basic_grasp import (
    BasicGraspSceneConfig,
    BasicGraspSceneA1,
    BasicGraspSceneA2,
    BasicGraspSceneA3,
    BasicGraspSceneA4,
    BasicGraspSceneRegistry,
)
from .mobile_grasp import (
    MobileGraspSceneConfig,
    MobileGraspSceneB1,
    MobileGraspSceneB2,
    MobileGraspSceneB3,
    MobileGraspSceneB4,
    MobileGraspSceneRegistry,
)

__all__ = [
    # Layer 1: Basic Grasp
    "BasicGraspSceneConfig",
    "BasicGraspSceneA1",
    "BasicGraspSceneA2",
    "BasicGraspSceneA3",
    "BasicGraspSceneA4",
    "BasicGraspSceneRegistry",
    # Layer 2: Mobile Grasp
    "MobileGraspSceneConfig",
    "MobileGraspSceneB1",
    "MobileGraspSceneB2",
    "MobileGraspSceneB3",
    "MobileGraspSceneB4",
    "MobileGraspSceneRegistry",
]
