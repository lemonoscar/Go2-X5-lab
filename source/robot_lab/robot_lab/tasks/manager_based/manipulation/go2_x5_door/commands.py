# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""High-level command contract for Go2-X5 door-opening prototypes."""

from robot_lab.tasks.manager_based.manipulation.go2_x5_tabletop.commands import (  # noqa: F401
    ARM_JOINT_POS_DIM,
    CMD_VEL_DIM,
    COMMAND_SLICES,
    GRIPPER_DIM,
    HIGH_LEVEL_COMMAND_DIM,
    HighLevelCommandSpec,
    split_high_level_action,
)
