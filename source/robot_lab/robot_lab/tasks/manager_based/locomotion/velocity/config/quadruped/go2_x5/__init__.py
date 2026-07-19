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
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyStairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.train_route_env_cfg:Go2X5DogOnlyStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyStairsPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyRoughStairsVx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rough_stairs_vx_env_cfg:Go2X5DogOnlyRoughStairsVxEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyRoughStairsVxPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pct_stairs_env_cfg:Go2X5DogOnlyPctStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyPctStairsPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsHard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pct_stairs_env_cfg:Go2X5DogOnlyPctStairsHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyPctStairsHardPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstSteps-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pct_stairs_env_cfg:Go2X5DogOnlyPctStairsFirstStepsEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyPctStairsFirstStepsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstRise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pct_stairs_env_cfg:Go2X5DogOnlyPctStairsFirstRiseEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5DogOnlyPctStairsFirstRisePPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstRiseExact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:Go2X5DogOnlyPctStairsFirstRiseExactEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFirstRiseExactPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstRiseExactScan1m-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFirstRiseExactScan1mEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFirstRiseExactScan1mPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstRiseExactHighStep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFirstRiseExactHighStepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFirstRiseExactHighStepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstStepsExactHighStep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFirstStepsExactHighStepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFirstStepsExactHighStepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsSecondRiseExactHighStep-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsSecondRiseExactHighStepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsSecondRiseExactHighStepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsSecondRiseExactSlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsSecondRiseExactSlowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsSecondRiseExactSlowPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightExactSlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightExactSlowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightExactSlowPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledSlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledSlowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledSlowPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledUpright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledUprightEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledUprightPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledSafeSurvival-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledDeploymentSpeed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledDeploymentSpeedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledDeploymentSpeedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledRearSupport-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledStableCompletion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledStableCompletionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledStableCompletionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledTopLanding-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledTopLandingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledTopLandingPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledPlatformProgress-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledPlatformEntry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledPlatformEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledPlatformEntryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularStairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:Go2X5DogOnlyPctRegularStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularAscentCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctRegularAscentCurriculumEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctRegularAscentCurriculumPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularAscentRepair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctRegularAscentRepairEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctRegularAscentRepairPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularAscentSim2Real-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctRegularAscentSim2RealEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctRegularAscentSim2RealPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularUpDownStairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctRegularUpDownStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctRegularUpDownStairsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularDescentStart-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctRegularDescentStartEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctRegularDescentStartPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledCoverage-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pct_stairs_env_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledCoverageEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "Go2X5DogOnlyPctStairsFullFlightProfiledCoveragePPORunnerCfg"
        ),
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
