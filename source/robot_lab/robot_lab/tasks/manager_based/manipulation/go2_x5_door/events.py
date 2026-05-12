# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reset and randomization event helpers for Go2-X5 door-opening prototypes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from robot_lab.tasks.manager_based.manipulation.go2_x5_tabletop.events import reset_staged_task_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_door_joint(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Reset the articulated door/cabinet hinge to the closed pose."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    door = env.scene["door"]
    joint_pos = door.data.default_joint_pos[env_ids].clone()
    joint_vel = door.data.default_joint_vel[env_ids].clone()
    joint_ids, _ = door.find_joints(["door_left_joint", "door_right_joint"], preserve_order=True)
    joint_pos[:, joint_ids] = 0.0
    joint_vel[:, joint_ids] = 0.0
    door.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_from_stage_snapshot_placeholder(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Placeholder for future reset-from-handle-contact or door-open snapshots."""

    del env, env_ids
