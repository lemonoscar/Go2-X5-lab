# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
VLA-SFT data collection tasks for Go2-X5 mobile manipulation.

This module provides scene configurations and utilities for collecting
demonstration data to train Vision-Language-Action models via supervised
fine-tuning (SFT).

Scene Layers:
    - Layer 1 (Basic Grasp): Single object grasping tasks
    - Layer 2 (Mobile Grasp): Navigation + grasping coupling
    - Layer 3 (Interaction): Contact-rich manipulation
    - Layer 4 (OOD): Generalization test scenes

Usage:
    >>> from robot_lab.tasks.manager_based.manipulation.vla_sft import VLASSceneManager
    >>> manager = VLASSceneManager()
    >>> scene_config = manager.sample_scene(layer="basic")
    >>> mobile_scene = manager.sample_scene(layer="mobile")
"""

__version__ = "1.0.0"

__all__ = [
    "VLASSceneManager",
    "VLADataBuffer",
    # Layer 1
    "BasicGraspSceneConfig",
    "BasicGraspSceneRegistry",
    # Layer 2
    "MobileGraspSceneConfig",
    "MobileGraspSceneRegistry",
]
