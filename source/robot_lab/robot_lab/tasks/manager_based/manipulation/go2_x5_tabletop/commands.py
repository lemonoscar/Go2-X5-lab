# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""High-level command contract for Go2-X5 tabletop manipulation prototypes."""

from __future__ import annotations

from dataclasses import dataclass


CMD_VEL_DIM = 3
ARM_JOINT_POS_DIM = 6
GRIPPER_DIM = 1
HIGH_LEVEL_COMMAND_DIM = CMD_VEL_DIM + ARM_JOINT_POS_DIM + GRIPPER_DIM

COMMAND_SLICES = {
    "cmd_vel": slice(0, 3),
    "arm_joint_pos": slice(3, 9),
    "gripper": slice(9, 10),
}


@dataclass(frozen=True)
class HighLevelCommandSpec:
    """Action contract consumed by the prototype Isaac Lab task."""

    cmd_vel_dim: int = CMD_VEL_DIM
    arm_joint_pos_dim: int = ARM_JOINT_POS_DIM
    gripper_dim: int = GRIPPER_DIM

    @property
    def action_dim(self) -> int:
        return HIGH_LEVEL_COMMAND_DIM


def split_high_level_action(action):
    """Return named views into a ``cmd_vel + arm_joint_pos + gripper`` action."""

    return {name: action[..., indices] for name, indices in COMMAND_SLICES.items()}
