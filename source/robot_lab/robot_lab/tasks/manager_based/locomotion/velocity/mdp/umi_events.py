# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import SceneEntityCfg

from .events import reset_root_state_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


UMI_STAGE1_EVENT_DEFAULTS = {
    "umi_transport_root_state": {
        "pose_range": {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "z": (0.0, 0.1),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "z": (-0.25, 0.25),
            "roll": (-0.25, 0.25),
            "pitch": (-0.25, 0.25),
            "yaw": (-0.25, 0.25),
        },
    },
    "umi_push_robot": {
        "velocity_range": {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
        },
    },
    "umi_force_disturbance": {
        "force_range": (-8.0, 8.0),
        "torque_range": (-2.5, 2.5),
    },
}


def umi_transport_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Transport-style root reset used as the UMI minimal reset event."""
    return reset_root_state_uniform(
        env,
        env_ids=env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )


def umi_push_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Velocity impulse event used for UMI push disturbances."""
    return core_mdp.push_by_setting_velocity(
        env,
        env_ids=env_ids,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )


def umi_force_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """External wrench disturbance used as the optional UMI force event."""
    return core_mdp.apply_external_force_torque(
        env,
        env_ids=env_ids,
        force_range=force_range,
        torque_range=torque_range,
        asset_cfg=asset_cfg,
    )
