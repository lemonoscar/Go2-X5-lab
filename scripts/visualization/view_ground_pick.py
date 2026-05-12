# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""View the Go2-X5 ground-pick scene with zero actions."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="View the Go2-X5 ground-pick scene.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-GroundPick-Go2-X5-Play-v0",
    help="Registered Gym task ID to launch.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--low_level_policy_path",
    type=str,
    default=None,
    help="Optional path to the frozen Go2 locomotion policy. Falls back to GO2_X5_LOW_LEVEL_POLICY_PATH.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--live",
    action="store_true",
    default=False,
    help="Continuously step zero actions. By default, keep the scene paused for interactive inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.low_level_policy_path:
    os.environ["GO2_X5_LOW_LEVEL_POLICY_PATH"] = args_cli.low_level_policy_path

if not os.environ.get("GO2_X5_LOW_LEVEL_POLICY_PATH"):
    raise ValueError(
        "A frozen Go2 locomotion policy is required. Set GO2_X5_LOW_LEVEL_POLICY_PATH "
        "or pass --low_level_policy_path."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import robot_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # This viewer is for manual inspection only, so skip descriptor export and
    # keep a stable local log path for any simulator-side artifacts.
    env_cfg.export_io_descriptors = False
    env_cfg.log_dir = os.path.join(ROOT_DIR, "logs", "visualization", "view_ground_pick")
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Viewing task: {args_cli.task}")
    print(f"[INFO] Low-level policy: {os.environ['GO2_X5_LOW_LEVEL_POLICY_PATH']}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    if args_cli.live:
        print("[INFO] Live stepping enabled: the scene will keep advancing with zero actions.")
    else:
        print("[INFO] Interactive preview mode: physics stays paused so you can orbit, select, and inspect the scene.")

    env.reset()
    try:
        while simulation_app.is_running():
            if args_cli.live:
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    env.step(actions)
            else:
                env.unwrapped.sim.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
