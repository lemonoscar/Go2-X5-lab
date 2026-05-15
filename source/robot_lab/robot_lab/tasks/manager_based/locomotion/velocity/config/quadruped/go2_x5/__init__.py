# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2X5FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2X5RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5RoughTrainerCfg",
    },
)

# UMI phase-1 skeleton tasks. The RSL-RL entry points intentionally target the future
# UMI runner module so later agents can land PPO without renaming the task ids.
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-UMI-Locomotion6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.umi_locomotion6d_env_cfg:UmiGo2X5Locomotion6dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.umi_rsl_rl_ppo_cfg:Go2X5UmiLocomotion6dPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-UMI-Extreme-Locomotion6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.umi_extreme_locomotion6d_env_cfg:UmiGo2X5ExtremeLocomotion6dEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.umi_rsl_rl_ppo_cfg:Go2X5UmiExtremeLocomotion6dPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5FoundationFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5FoundationFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5FoundationFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5ArmUnlockFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5ArmUnlockFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5ArmUnlockFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmLocomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5ArmLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5ArmLocomotionFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5ArmLocomotionFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-DogOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5DogOnlyFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-DogOnlyArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyArmFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5DogOnlyArmFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-DogOnlyRecover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyRecoverFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyRecoverFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5DogOnlyRecoverFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2-X5-DogOnlyCrawl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyCrawlFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyCrawlFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5DogOnlyCrawlFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5DogOnlyRoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyHard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyHardRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyHardRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5DogOnlyHardRoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5RobustRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5RobustRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5RobustRoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-ArmWarmup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5ArmWarmupRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5ArmWarmupRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:Go2X5ArmWarmupRoughTrainerCfg",
    },
)
