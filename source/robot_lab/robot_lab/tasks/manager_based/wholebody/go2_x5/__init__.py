"""Go2-X5 RoboDuet WholeBody task registration."""

import gymnasium as gym


gym.register(
    id="RobotLab-Isaac-Go2-X5-WholeBody-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:Go2X5WholeBodyEnvCfg"},
)
