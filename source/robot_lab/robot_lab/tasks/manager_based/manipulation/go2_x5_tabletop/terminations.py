# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for the Go2-X5 tabletop prototype."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from .observations import ee_to_object_distance, gripper_opening, object_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tabletop_success(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.88,
    max_eef_object_distance: float = 0.14,
    close_threshold: float = 0.018,
) -> torch.Tensor:
    lifted = object_height(env).squeeze(-1) > minimal_height
    near_ee = ee_to_object_distance(env).squeeze(-1) < max_eef_object_distance
    closed = gripper_opening(env).squeeze(-1) < close_threshold
    return lifted & near_ee & closed


def robot_fallen(env: ManagerBasedRLEnv, minimum_height: float = 0.18) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return robot.data.root_pos_w[:, 2] < minimum_height
