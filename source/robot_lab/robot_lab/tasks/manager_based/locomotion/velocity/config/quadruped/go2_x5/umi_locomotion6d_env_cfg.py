# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .umi_base_env_cfg import UmiGo2X5BaseEnvCfg


@configclass
class UmiGo2X5Locomotion6dEnvCfg(UmiGo2X5BaseEnvCfg):
    """Phase-1 flat locomotion6d task shell for UMI migration."""

    def __post_init__(self):
        super().__post_init__()

        # Phase 1 starts from a flat minimal training setup.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.commands.locomotion6d.rel_standing_envs = 0.05
        self.commands.locomotion6d.resampling_time_range = (4.0, 4.0)
        self.rewards.umi_track_z_height_exp.params["sensor_cfg"] = None
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
