# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for Go2-X5 tabletop manipulation prototypes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation, RigidObject
import isaaclab.envs.mdp as core_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .actions import ARM_JOINT_NAMES, DOG_JOINT_NAMES, GRIPPER_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _joint_state(env: ManagerBasedEnv, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene["robot"]
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    return robot.data.joint_pos[:, joint_ids], robot.data.joint_vel[:, joint_ids]


def student_proprio(env: ManagerBasedEnv) -> torch.Tensor:
    """Compact proprio state for a visual student policy."""

    robot: Articulation = env.scene["robot"]
    dog_pos, dog_vel = _joint_state(env, DOG_JOINT_NAMES)
    arm_pos, arm_vel = _joint_state(env, ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)
    return torch.cat(
        [
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            dog_pos - robot.data.default_joint_pos[:, robot.find_joints(DOG_JOINT_NAMES, preserve_order=True)[0]],
            dog_vel,
            arm_pos - robot.data.default_joint_pos[
                :, robot.find_joints(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES, preserve_order=True)[0]
            ],
            arm_vel,
            last_high_level_action(env),
        ],
        dim=-1,
    )


def camera_rgb(env: ManagerBasedEnv, camera_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return normalized RGB from an Isaac Lab camera sensor."""

    camera = env.scene[camera_cfg.name]
    rgb = camera.data.output["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if not torch.is_floating_point(rgb):
        rgb = rgb.float() / 255.0
    return rgb


def last_high_level_action(env: ManagerBasedEnv, expected_dim: int = 10) -> torch.Tensor:
    """Return the last high-level action, padded or clipped to the command contract."""

    action = core_mdp.last_action(env)
    if action.shape[-1] == expected_dim:
        return action
    if action.shape[-1] > expected_dim:
        return action[:, :expected_dim]
    padding = torch.zeros(action.shape[0], expected_dim - action.shape[-1], device=action.device, dtype=action.dtype)
    return torch.cat([action, padding], dim=-1)


def cmd_vel_command(env: ManagerBasedEnv) -> torch.Tensor:
    return last_high_level_action(env)[:, :3]


def arm_joint_pos_command(env: ManagerBasedEnv) -> torch.Tensor:
    return last_high_level_action(env)[:, 3:9]


def gripper_command(env: ManagerBasedEnv) -> torch.Tensor:
    return last_high_level_action(env)[:, 9:10]


def stage_one_hot(env: ManagerBasedEnv, num_stages: int = 4) -> torch.Tensor:
    stage = getattr(env, "_go2_x5_stage", None)
    if stage is None:
        stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    return F.one_hot(stage.clamp(0, num_stages - 1), num_classes=num_stages).float()


def time_in_stage(env: ManagerBasedEnv) -> torch.Tensor:
    value = getattr(env, "_go2_x5_time_in_stage", None)
    if value is None:
        value = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    return value[:, None].float()


def object_position_in_robot_root_frame(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w[:, :3])
    return object_pos_b


def object_height(env: ManagerBasedEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2:3]


def ee_to_object_vector(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return obj.data.root_pos_w[:, :3] - ee_pos_w


def ee_to_object_distance(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.norm(ee_to_object_vector(env), dim=-1, keepdim=True)


def gripper_opening(env: ManagerBasedEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES, preserve_order=True)
    return robot.data.joint_pos[:, joint_ids].mean(dim=-1, keepdim=True)


@configclass
class ObservationsCfg:
    """Teacher/student observation split for the tabletop prototype."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=student_proprio)
        cmd_vel = ObsTerm(func=cmd_vel_command)
        arm_joint_pos = ObsTerm(func=arm_joint_pos_command)
        gripper = ObsTerm(func=gripper_command)
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
        object_position = ObsTerm(func=object_position_in_robot_root_frame)
        object_height = ObsTerm(func=object_height)
        ee_to_object = ObsTerm(func=ee_to_object_vector)
        gripper_opening = ObsTerm(func=gripper_opening)
        high_level_action = ObsTerm(func=last_high_level_action)
        stage = ObsTerm(func=stage_one_hot, params={"num_stages": 4})
        time_in_stage = ObsTerm(func=time_in_stage)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    student: StudentVisualCfg = StudentVisualCfg()
    critic: CriticCfg = CriticCfg()
