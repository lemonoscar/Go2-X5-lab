# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2X5RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "go2_x5_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2X5FlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.save_interval = 1000
        self.experiment_name = "go2_x5_flat"


@configclass
class Go2X5FoundationFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 24
        self.max_iterations = 8000
        self.save_interval = 500
        self.experiment_name = "go2_x5_foundation_flat"
        self.algorithm.entropy_coef = 0.01


@configclass
class Go2X5ArmUnlockFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 5000
        self.save_interval = 250
        self.experiment_name = "go2_x5_arm_unlock_flat"
        self.algorithm.entropy_coef = 0.003
        self.algorithm.learning_rate = 5.0e-4


@configclass
class Go2X5ArmLocomotionFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_arm_locomotion_flat"
        self.algorithm.entropy_coef = 0.0025
        self.algorithm.learning_rate = 2.0e-4


@configclass
class Go2X5DogOnlyFlatPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_flat"
        self.algorithm.entropy_coef = 0.0025
        self.algorithm.learning_rate = 2.0e-4


@configclass
class Go2X5DogOnlyArmFlatPPORunnerCfg(Go2X5DogOnlyFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_arm_flat"
        self.algorithm.entropy_coef = 0.0025
        self.algorithm.learning_rate = 2.0e-4


@configclass
class Go2X5DogOnlyRecoverFlatPPORunnerCfg(Go2X5DogOnlyFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 4000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_recover_flat"
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.learning_rate = 1.0e-4


@configclass
class Go2X5DogOnlyCrawlFlatPPORunnerCfg(Go2X5DogOnlyRecoverFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 8000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_crawl_flat"
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.learning_rate = 1.0e-4


@configclass
class Go2X5DogOnlyRoughPPORunnerCfg(Go2X5DogOnlyFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 8000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_rough"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 1.0e-4


@configclass
class Go2X5DogOnlyHardRoughPPORunnerCfg(Go2X5DogOnlyRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 18000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_hard_rough"
        self.algorithm.entropy_coef = 0.0025
        self.algorithm.learning_rate = 7.5e-5


@configclass
class Go2X5DogOnlyStairsPPORunnerCfg(Go2X5DogOnlyRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_dog_only_stairs"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 5.0e-5


@configclass
class Go2X5DogOnlyRoughStairsVxPPORunnerCfg(Go2X5DogOnlyRoughPPORunnerCfg):
    """Long, low-learning-rate continuation for the unified vx/stairs policy."""

    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 10000
        self.save_interval = 100
        self.experiment_name = "go2_x5_dog_only_rough_stairs_vx"
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.learning_rate = 1.0e-5


@configclass
class Go2X5DogOnlyPctStairsPPORunnerCfg(Go2X5DogOnlyRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "go2_x5_dog_only_pct_stairs"
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.learning_rate = 2.5e-5


@configclass
class Go2X5DogOnlyPctStairsHardPPORunnerCfg(Go2X5DogOnlyPctStairsPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "go2_x5_dog_only_pct_stairs_hard"
        self.algorithm.learning_rate = 1.0e-5


@configclass
class Go2X5DogOnlyPctStairsFirstStepsPPORunnerCfg(Go2X5DogOnlyPctStairsHardPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 250
        self.experiment_name = "go2_x5_dog_only_pct_stairs_first_steps"


@configclass
class Go2X5DogOnlyPctStairsFirstRisePPORunnerCfg(Go2X5DogOnlyPctStairsFirstStepsPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100
        self.experiment_name = "go2_x5_dog_only_pct_stairs_first_rise"


@configclass
class Go2X5DogOnlyPctStairsFirstRiseExactPPORunnerCfg(
    Go2X5DogOnlyPctStairsFirstRisePPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 150
        self.experiment_name = "go2_x5_dog_only_pct_stairs_first_rise_exact"


@configclass
class Go2X5DogOnlyPctStairsFirstRiseExactScan1mPPORunnerCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_first_rise_exact_scan1m"


@configclass
class Go2X5DogOnlyPctStairsFirstRiseExactHighStepPPORunnerCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactScan1mPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_first_rise_exact_high_step"


@configclass
class Go2X5DogOnlyPctStairsFirstStepsExactHighStepPPORunnerCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactHighStepPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 250
        self.experiment_name = "go2_x5_dog_only_pct_stairs_first_steps_exact_high_step"


@configclass
class Go2X5DogOnlyPctStairsSecondRiseExactHighStepPPORunnerCfg(
    Go2X5DogOnlyPctStairsFirstRiseExactHighStepPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 200
        self.experiment_name = "go2_x5_dog_only_pct_stairs_second_rise_exact_high_step"


@configclass
class Go2X5DogOnlyPctStairsSecondRiseExactSlowPPORunnerCfg(
    Go2X5DogOnlyPctStairsSecondRiseExactHighStepPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_second_rise_exact_slow"


@configclass
class Go2X5DogOnlyPctStairsFullFlightExactSlowPPORunnerCfg(
    Go2X5DogOnlyPctStairsSecondRiseExactSlowPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 48
        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_exact_slow"
        self.algorithm.entropy_coef = 0.003
        self.algorithm.learning_rate = 2.5e-5


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledSlowPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightExactSlowPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_slow"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledUprightPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledSlowPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_upright"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledUprightPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_safe_survival"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledDeploymentSpeedPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_deployment_speed"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledDeploymentSpeedPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_rear_support"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledStableCompletionPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 500
        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_stable_completion"
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 1.0e-5


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledTopLandingPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_top_landing"
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 1.0e-5


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledRearSupportPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_platform_progress"
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 5.0e-6


@configclass
class Go2X5DogOnlyPctRegularUpDownStairsPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.save_interval = 100
        self.experiment_name = "go2_x5_dog_only_pct_regular_up_down_stairs"


@configclass
class Go2X5DogOnlyPctRegularAscentCurriculumPPORunnerCfg(
    Go2X5DogOnlyPctRegularUpDownStairsPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_regular_ascent_curriculum"


@configclass
class Go2X5DogOnlyPctRegularAscentRepairPPORunnerCfg(
    Go2X5DogOnlyPctRegularAscentCurriculumPPORunnerCfg
):
    """Low-rate R1 continuation from the unified model on exact ascent."""

    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 48
        self.max_iterations = 2000
        self.save_interval = 100
        self.experiment_name = "go2_x5_dog_only_pct_regular_ascent_repair"
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 1.0e-5


@configclass
class Go2X5DogOnlyPctRegularDescentStartPPORunnerCfg(
    Go2X5DogOnlyPctRegularUpDownStairsPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_regular_descent_start"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledPlatformEntryPPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledPlatformProgressPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_platform_entry"


@configclass
class Go2X5DogOnlyPctStairsFullFlightProfiledCoveragePPORunnerCfg(
    Go2X5DogOnlyPctStairsFullFlightProfiledSafeSurvivalPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_x5_dog_only_pct_stairs_full_flight_profiled_coverage"


@configclass
class Go2X5RobustRoughPPORunnerCfg(Go2X5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 6000
        self.save_interval = 250
        self.experiment_name = "go2_x5_robust_rough"
        self.algorithm.entropy_coef = 0.004


@configclass
class Go2X5ArmWarmupRoughPPORunnerCfg(Go2X5RobustRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 32
        self.max_iterations = 4000
        self.save_interval = 250
        self.experiment_name = "go2_x5_arm_warmup_rough"
        self.algorithm.entropy_coef = 0.003
