# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the Go2-X5 door-opening prototype environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "robot_lab"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the Go2-X5 door-opening prototype env.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Go2-X5-Door-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--low_level_policy_path", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.low_level_policy_path:
    os.environ["GO2_X5_LOW_LEVEL_POLICY_PATH"] = args_cli.low_level_policy_path
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import robot_lab.tasks  # noqa: F401


def _keys(value):
    if isinstance(value, dict):
        return {key: _keys(item) for key, item in value.items()}
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.export_io_descriptors = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        action_shape = env.action_space.shape
        if len(action_shape) == 1:
            action_shape = (env.unwrapped.num_envs, action_shape[0])
        actions = torch.zeros(action_shape, device=env.unwrapped.device)
        reward = terminated = truncated = None
        for _ in range(args_cli.steps):
            obs, reward, terminated, truncated, info = env.step(actions)
        print(f"observation keys: {_keys(obs)}")
        print(f"action dim: {actions.shape[-1]}")
        print(f"reward shape: {tuple(reward.shape) if hasattr(reward, 'shape') else type(reward).__name__}")
        print(
            "done shape: "
            f"{tuple(terminated.shape) if hasattr(terminated, 'shape') else type(terminated).__name__}"
        )
        print(
            "truncated shape: "
            f"{tuple(truncated.shape) if hasattr(truncated, 'shape') else type(truncated).__name__}"
        )
        print(f"info keys: {sorted(info.keys()) if isinstance(info, dict) else type(info).__name__}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
