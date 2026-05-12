# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward placeholders for the Go2-X5 door-opening prototype."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from .observations import door_hinge_angle, ee_to_handle_vector

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def door_angle_reward(env: ManagerBasedRLEnv, target_angle: float = 0.7) -> torch.Tensor:
    hinge = door_hinge_angle(env).squeeze(-1)
    return torch.clamp(hinge / target_angle, min=0.0, max=1.0)


def handle_alignment_reward(env: ManagerBasedRLEnv, std: float = 0.20) -> torch.Tensor:
    distance = torch.norm(ee_to_handle_vector(env), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def base_stability_reward(
    env: ManagerBasedRLEnv,
    roll_pitch_std: float = 0.35,
    vertical_vel_std: float = 0.35,
) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    vertical_vel = torch.abs(robot.data.root_lin_vel_b[:, 2])
    return torch.exp(-(tilt / roll_pitch_std) ** 2) * torch.exp(-(vertical_vel / vertical_vel_std) ** 2)
