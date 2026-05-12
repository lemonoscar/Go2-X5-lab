# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .umi_base_env_cfg import UmiGo2X5BaseEnvCfg


@configclass
class UmiGo2X5ExtremeLocomotion6dEnvCfg(UmiGo2X5BaseEnvCfg):
    """Rough/extreme locomotion6d scaffold reserved for the later UMI stages."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.scene.terrain.max_init_terrain_level = 0
        self.commands.locomotion6d.rel_standing_envs = 0.0
        self.commands.locomotion6d.resampling_time_range = (4.0, 4.0)
        self.rewards.umi_track_z_height_exp.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.events.randomize_push_robot = EventTerm(
            func=mdp.umi_push_robot,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params=copy.deepcopy(mdp.UMI_STAGE1_EVENT_DEFAULTS["umi_push_robot"]),
        )
        self.events.randomize_apply_external_force_torque = EventTerm(
            func=mdp.umi_force_disturbance,
            mode="interval",
            interval_range_s=(8.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name], preserve_order=True),
                **copy.deepcopy(mdp.UMI_STAGE1_EVENT_DEFAULTS["umi_force_disturbance"]),
            },
        )
