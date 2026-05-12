# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Go2-X5 door-opening prototype task registration."""

import gymnasium as gym


gym.register(
    id="RobotLab-Isaac-Go2-X5-Door-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.scene_cfg:Go2X5DoorEnvCfg"},
)

gym.register(
    id="RobotLab-Isaac-Go2-X5-Door-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.scene_cfg:Go2X5DoorEnvCfg_PLAY"},
)
