# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration files for VLA-SFT environments."""

from .basic_cfg import Go2X5VLASBasicEnvCfg, Go2X5VLASBasicEnvCfg_PLAY
from .mobile_cfg import (
    Go2X5VLASMobileEnvCfg,
    Go2X5VLASMobileEnvCfg_PLAY,
)

__all__ = [
    # Layer 1: Basic Grasp
    "Go2X5VLASBasicEnvCfg",
    "Go2X5VLASBasicEnvCfg_PLAY",
    # Layer 2: Mobile Grasp
    "Go2X5VLASMobileEnvCfg",
    "Go2X5VLASMobileEnvCfg_PLAY",
]
