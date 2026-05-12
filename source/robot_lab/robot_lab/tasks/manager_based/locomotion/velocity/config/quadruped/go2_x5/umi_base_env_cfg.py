# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import umi_events, umi_rewards, umi_terminations
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.umi_commands import (
    UmiLocomotion6dCommandsCfg,
    build_umi_joint_position_action_kwargs,
    build_umi_phase1_randomization_specs,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.umi_observations import UmiLocomotion6dObservationsCfg

from .rough_env_cfg import Go2X5RoughEnvCfg


@configclass
class UmiGo2X5BaseEnvCfg(Go2X5RoughEnvCfg):
    """UMI phase-1 base env for unified 18-DoF locomotion6d."""

    def __post_init__(self):
        super().__post_init__()

        self.observations = UmiLocomotion6dObservationsCfg()
        self.commands = UmiLocomotion6dCommandsCfg()
        self.actions.joint_pos = mdp.JointPositionActionCfg(**build_umi_joint_position_action_kwargs())
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.scene.height_scanner = None

        self._configure_umi_randomization()
        self._configure_umi_rewards()
        self._configure_umi_terminations()

        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

    def _configure_umi_randomization(self):
        specs = build_umi_phase1_randomization_specs()
        self.events.randomize_rigid_body_material = EventTerm(**copy.deepcopy(specs["material_friction"]))
        self.events.randomize_rigid_body_mass_base = EventTerm(**copy.deepcopy(specs["base_mass"]))
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = EventTerm(**copy.deepcopy(specs["base_com"]))
        self.events.randomize_actuator_gains = EventTerm(**copy.deepcopy(specs["actuator_gains"]))
        self.events.randomize_reset_base = EventTerm(
            func=umi_events.umi_transport_root_state,
            mode="reset",
            params=copy.deepcopy(umi_events.UMI_STAGE1_EVENT_DEFAULTS["umi_transport_root_state"]),
        )
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

    def _configure_umi_rewards(self):
        for attr_name, reward_term in list(vars(self.rewards).items()):
            if attr_name.startswith("_") or callable(reward_term):
                continue
            setattr(self.rewards, attr_name, None)

        joint_cfg = SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True)
        base_body_cfg = SceneEntityCfg("robot", body_names=[self.base_link_name], preserve_order=True)
        feet_body_cfg = SceneEntityCfg("robot", body_names=[self.foot_link_name])
        feet_contact_cfg = SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])
        non_foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"])

        self.rewards.umi_track_lin_vel_xy_exp = RewTerm(
            func=umi_rewards.umi_track_lin_vel_xy_exp,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_track_lin_vel_xy_exp"],
            params={"command_name": "locomotion6d", "tracking_sigma": 0.25, "power": 2.0},
        )
        self.rewards.umi_track_yaw_exp = RewTerm(
            func=umi_rewards.umi_track_yaw_exp,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_track_yaw_exp"],
            params={"command_name": "locomotion6d", "tracking_sigma": 0.25, "power": 2.0},
        )
        self.rewards.umi_track_z_height_exp = RewTerm(
            func=umi_rewards.umi_track_z_height_exp,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_track_z_height_exp"],
            params={
                "command_name": "locomotion6d",
                "z_height_sigma": 0.01,
                "asset_cfg": base_body_cfg,
                "sensor_cfg": None,
            },
        )
        self.rewards.umi_track_gravity_exp = RewTerm(
            func=umi_rewards.umi_track_gravity_exp,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_track_gravity_exp"],
            params={"command_name": "locomotion6d", "gravity_sigma": 0.05, "asset_cfg": base_body_cfg},
        )
        self.rewards.umi_action_rate_l2 = RewTerm(
            func=umi_rewards.umi_action_rate_l2,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_action_rate_l2"],
            params={"power": 2.0},
        )
        self.rewards.umi_joint_acc_l2 = RewTerm(
            func=umi_rewards.umi_joint_acc_l2,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_joint_acc_l2"],
            params={"asset_cfg": joint_cfg},
        )
        self.rewards.umi_joint_power = RewTerm(
            func=umi_rewards.umi_joint_power,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_joint_power"],
            params={"power": 2.0, "asset_cfg": joint_cfg},
        )
        self.rewards.umi_joint_torques_l2 = RewTerm(
            func=umi_rewards.umi_joint_torques_l2,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_joint_torques_l2"],
            params={"power": 2.0, "asset_cfg": joint_cfg},
        )
        self.rewards.umi_joint_pos_limits = RewTerm(
            func=umi_rewards.umi_joint_pos_limits,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_joint_pos_limits"],
            params={"penalty_scale": 0.9, "asset_cfg": joint_cfg},
        )
        self.rewards.umi_undesired_contacts = RewTerm(
            func=umi_rewards.umi_undesired_contacts,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_undesired_contacts"],
            params={"threshold": 0.1, "sensor_cfg": non_foot_contact_cfg},
        )
        self.rewards.umi_feet_drag_penalty = RewTerm(
            func=umi_rewards.umi_feet_drag_penalty,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_feet_drag_penalty"],
            params={
                "asset_cfg": feet_body_cfg,
                "penalty_feet_drag_height": 0.10,
                "feet_drag_sigma": 0.05,
            },
        )
        self.rewards.umi_feet_air_time = RewTerm(
            func=umi_rewards.umi_feet_air_time,
            weight=umi_rewards.UMI_STAGE1_REWARD_DEFAULT_WEIGHTS["umi_feet_air_time"],
            params={"command_name": "locomotion6d", "sensor_cfg": feet_contact_cfg, "threshold": 0.5},
        )

    def _configure_umi_terminations(self):
        self.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None

        non_foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"])
        base_body_cfg = SceneEntityCfg("robot", body_names=[self.base_link_name], preserve_order=True)

        self.terminations.umi_illegal_contact = DoneTerm(
            func=umi_terminations.umi_illegal_contact,
            params={
                "threshold": umi_terminations.UMI_STAGE1_TERMINATION_DEFAULTS["umi_illegal_contact"]["threshold"],
                "sensor_cfg": non_foot_contact_cfg,
            },
        )
        self.terminations.umi_bad_orientation = DoneTerm(
            func=umi_terminations.umi_bad_orientation,
            params={
                "limit_angle": umi_terminations.UMI_STAGE1_TERMINATION_DEFAULTS["umi_bad_orientation"]["limit_angle"],
                "asset_cfg": base_body_cfg,
            },
        )
