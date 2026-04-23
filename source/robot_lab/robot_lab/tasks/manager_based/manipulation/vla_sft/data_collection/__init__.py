# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Data collection utilities for VLA-SFT training.

This module provides tools for collecting demonstration data
in the VLA format compatible with SimpleVLA-RL.
"""

from .scene_manager import VLASSceneManager
from .instruction_generator import InstructionGenerator

__all__ = [
    "VLASSceneManager",
    "InstructionGenerator",
]
