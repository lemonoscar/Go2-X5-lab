# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for the Go2-X5 door-opening prototype."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from .observations import door_hinge_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def door_open_success(env: ManagerBasedRLEnv, target_angle: float = 0.7) -> torch.Tensor:
    return door_hinge_angle(env).squeeze(-1) > target_angle


def robot_fallen(env: ManagerBasedRLEnv, minimum_height: float = 0.18) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return robot.data.root_pos_w[:, 2] < minimum_height
