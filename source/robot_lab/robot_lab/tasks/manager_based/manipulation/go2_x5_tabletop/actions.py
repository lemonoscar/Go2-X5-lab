# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Action terms for Go2-X5 high-level manipulation prototypes."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp import AbsBinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as loco_mdp


DOG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]


@configclass
class LowLevelPolicyObservationsCfg(ObsGroup):
    """DogOnly-style low-level observations for the frozen leg policy."""

    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, scale=1.0)
    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"}, scale=1.0)
    joint_pos = ObsTerm(
        func=loco_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)},
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=loco_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(func=loco_mdp.last_action, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


class BaseVelocityHoldAction(ActionTerm):
    """Accept a 3-D base command while holding the dog joints at their default pose.

    This is a visibility/smoke-test action term for the tabletop reach scaffold. It
    preserves the high-level ``cmd_vel`` action slot without requiring a trained
    low-level leg policy just to load the scene.
    """

    cfg: "BaseVelocityHoldActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "BaseVelocityHoldActionCfg", env):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        if len(self._joint_ids) == 0:
            raise ValueError(f"No joints matched joint_names={cfg.joint_names} for asset '{cfg.asset_name}'.")
        self._raw_actions = self._processed_actions = self._asset.data.default_joint_pos.new_zeros((self.num_envs, 3))

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions[:, :3]
        self._processed_actions[:] = self._raw_actions

    def apply_actions(self):
        targets = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
        self._processed_actions[:] = self._raw_actions


@configclass
class BaseVelocityHoldActionCfg(ActionTermCfg):
    """Configuration for the tabletop reach base-command hold action."""

    class_type: type = BaseVelocityHoldAction
    asset_name: str = "robot"
    joint_names: list[str] | str | None = None
    preserve_order: bool = True


@configclass
class HighLevelActionsCfg:
    """10-D task action: ``cmd_vel(3) + arm_joint_pos(6) + gripper(1)``.

    The arm and gripper commands are direct controller targets and do not enter a
    learned leg action head. The base command slot is held for the future DogOnly
    low-level policy connection, while this reach smoke task holds leg joints.
    """

    cmd_vel = BaseVelocityHoldActionCfg(
        asset_name="robot",
        joint_names=DOG_JOINT_NAMES,
        preserve_order=True,
    )
    arm_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale={
            "arm_joint1": 1.2,
            "arm_joint2": 1.2,
            "arm_joint3": 1.2,
            "arm_joint4": 0.8,
            "arm_joint5": 0.7,
            "arm_joint6": 0.7,
        },
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )
    gripper = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={"arm_joint7": 0.044, "arm_joint8": 0.044},
        close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
        threshold=0.022,
        positive_threshold=True,
    )
