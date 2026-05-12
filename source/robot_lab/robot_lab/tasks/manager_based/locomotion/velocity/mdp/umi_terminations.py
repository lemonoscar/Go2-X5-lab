# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import SceneEntityCfg

from .terminations import (
    root_ang_vel_xy_above_maximum,
    root_height_above_maximum,
    root_lin_vel_z_above_maximum,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


UMI_STAGE1_TERMINATION_DEFAULTS = {
    "umi_illegal_contact": {"threshold": 1.0},
    "umi_bad_orientation": {"limit_angle": 1.0},
    "umi_root_height_below_minimum": {"minimum_height": 0.18},
    "umi_root_height_above_maximum": {"maximum_height": 0.65},
    "umi_root_lin_vel_z_above_maximum": {"maximum_speed": 3.0},
    "umi_root_ang_vel_xy_above_maximum": {"maximum_speed": 8.0},
}


def umi_illegal_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
):
    return core_mdp.illegal_contact(env, threshold=threshold, sensor_cfg=sensor_cfg)


def umi_bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    return core_mdp.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)


def umi_root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    return core_mdp.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)


def umi_root_height_above_maximum(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    return root_height_above_maximum(env, maximum_height=maximum_height, asset_cfg=asset_cfg)


def umi_root_lin_vel_z_above_maximum(
    env: ManagerBasedRLEnv,
    maximum_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    return root_lin_vel_z_above_maximum(env, maximum_speed=maximum_speed, asset_cfg=asset_cfg)


def umi_root_ang_vel_xy_above_maximum(
    env: ManagerBasedRLEnv,
    maximum_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    return root_ang_vel_xy_above_maximum(env, maximum_speed=maximum_speed, asset_cfg=asset_cfg)
