# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward placeholders for the Go2-X5 tabletop prototype."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from .observations import ee_to_object_distance, gripper_opening, object_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_object(env: ManagerBasedRLEnv, std: float = 0.15) -> torch.Tensor:
    distance = ee_to_object_distance(env).squeeze(-1)
    return 1.0 - torch.tanh(distance / std)


def gripper_close_near_object(env: ManagerBasedRLEnv, close_threshold: float = 0.018, std: float = 0.15) -> torch.Tensor:
    near_object = reaching_object(env, std=std)
    closed = (gripper_opening(env).squeeze(-1) < close_threshold).float()
    return near_object * closed


def object_lifted(env: ManagerBasedRLEnv, minimal_height: float = 0.86) -> torch.Tensor:
    return (object_height(env).squeeze(-1) > minimal_height).float()


def base_stability(env: ManagerBasedRLEnv, roll_pitch_std: float = 0.35, vertical_vel_std: float = 0.35) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    vertical_vel = torch.abs(robot.data.root_lin_vel_b[:, 2])
    return torch.exp(-(tilt / roll_pitch_std) ** 2) * torch.exp(-(vertical_vel / vertical_vel_std) ** 2)


def staged_progress_placeholder(env: ManagerBasedRLEnv) -> torch.Tensor:
    stage = getattr(env, "_go2_x5_stage", None)
    if stage is None:
        return torch.zeros(env.num_envs, device=env.device)
    return stage.float()
