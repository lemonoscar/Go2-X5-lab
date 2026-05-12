# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for Go2-X5 door-opening prototypes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from robot_lab.tasks.manager_based.manipulation.go2_x5_tabletop.observations import (
    arm_joint_pos_command,
    camera_rgb,
    cmd_vel_command,
    gripper_command,
    last_high_level_action,
    stage_one_hot,
    student_proprio,
    time_in_stage,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def door_hinge_angle(
    env: ManagerBasedEnv,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_left_joint"], preserve_order=True),
) -> torch.Tensor:
    door: Articulation = env.scene[door_cfg.name]
    joint_ids, _ = door.find_joints(door_cfg.joint_names, preserve_order=door_cfg.preserve_order)
    return door.data.joint_pos[:, joint_ids[:1]]


def door_hinge_velocity(
    env: ManagerBasedEnv,
    door_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_left_joint"], preserve_order=True),
) -> torch.Tensor:
    door: Articulation = env.scene[door_cfg.name]
    joint_ids, _ = door.find_joints(door_cfg.joint_names, preserve_order=door_cfg.preserve_order)
    return door.data.joint_vel[:, joint_ids[:1]]


def handle_pose_in_robot_root_frame(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
) -> torch.Tensor:
    """Approximate handle pose for the scaffold until a dedicated handle frame is added."""

    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]
    handle_pos_w = door.data.root_pos_w[:, :3] + torch.tensor([0.0, -0.38, 0.78], device=env.device)
    handle_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, handle_pos_w)
    hinge = door_hinge_angle(env)
    return torch.cat([handle_pos_b, hinge], dim=-1)


def ee_to_handle_vector(env: ManagerBasedEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ee_pos_w = robot.data.body_pos_w[:, robot.find_bodies("arm_link6")[0][0], :]
    door: Articulation = env.scene["door"]
    handle_pos_w = door.data.root_pos_w[:, :3] + torch.tensor([0.0, -0.38, 0.78], device=env.device)
    return handle_pos_w - ee_pos_w


@configclass
class ObservationsCfg:
    """Teacher/student observation split for the door-opening prototype."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=student_proprio)
        cmd_vel = ObsTerm(func=cmd_vel_command)
        arm_joint_pos = ObsTerm(func=arm_joint_pos_command)
        gripper = ObsTerm(func=gripper_command)
        hinge_angle = ObsTerm(func=door_hinge_angle)
        stage = ObsTerm(func=stage_one_hot, params={"num_stages": 4})
        time_in_stage = ObsTerm(func=time_in_stage)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentVisualCfg(ObsGroup):
        rgb = ObsTerm(func=camera_rgb, params={"camera_cfg": SceneEntityCfg("dog_camera")})
        wrist_rgb = ObsTerm(func=camera_rgb, params={"camera_cfg": SceneEntityCfg("arm_camera")})
        proprio = ObsTerm(func=student_proprio)
        high_level_action = ObsTerm(func=last_high_level_action)
        stage = ObsTerm(func=stage_one_hot, params={"num_stages": 4})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        proprio = ObsTerm(func=student_proprio)
        hinge_angle = ObsTerm(func=door_hinge_angle)
        hinge_velocity = ObsTerm(func=door_hinge_velocity)
        handle_pose = ObsTerm(func=handle_pose_in_robot_root_frame)
        ee_to_handle = ObsTerm(func=ee_to_handle_vector)
        high_level_action = ObsTerm(func=last_high_level_action)
        stage = ObsTerm(func=stage_one_hot, params={"num_stages": 4})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    student: StudentVisualCfg = StudentVisualCfg()
    critic: CriticCfg = CriticCfg()
