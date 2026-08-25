#!/usr/bin/env python3
"""Headless startup smoke for the strict Go2-X5 WholeBody inference task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


TASK_ID = "RobotLab-Isaac-Go2-X5-WholeBody-v0"
REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--manifest", type=Path)
parser.add_argument("--model-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

SOURCE_ROOT = REPO_ROOT / "source" / "robot_lab"
sys.path.insert(0, os.fspath(SOURCE_ROOT))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: E402,F401


def main() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be positive")
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    if args_cli.manifest is not None:
        env_cfg.actions.whole_body.manifest_path = os.fspath(args_cli.manifest.resolve())
    if args_cli.model_root is not None:
        env_cfg.actions.whole_body.model_root = os.fspath(args_cli.model_root.resolve())
    env_cfg.export_io_descriptors = False

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        raw_env = env.unwrapped
        if raw_env.num_envs != 1:
            raise AssertionError(f"expected one environment, got {raw_env.num_envs}")
        if raw_env.action_manager.total_action_dim != 10:
            raise AssertionError(
                f"expected public action dim 10, got {raw_env.action_manager.total_action_dim}"
            )
        if abs(float(raw_env.step_dt) - 0.02) > 1.0e-9:
            raise AssertionError(f"expected 20 ms control period, got {raw_env.step_dt}")

        env.reset()
        term = raw_env.action_manager.get_term("whole_body")
        action = term.processed_actions.clone()
        high_contact_steps = 0
        for step in range(1, args_cli.steps + 1):
            observations, rewards, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(observations["policy"]).all():
                raise FloatingPointError("WholeBody smoke produced non-finite policy observations")
            if not torch.isfinite(rewards).all():
                raise FloatingPointError("WholeBody smoke produced non-finite rewards")
            if bool((terminated | truncated).any().item()):
                raise AssertionError("WholeBody smoke reset unexpectedly")
            if not torch.isfinite(raw_env.scene["robot"].data.joint_pos_target).all():
                raise FloatingPointError("WholeBody smoke produced non-finite joint targets")
            high_contact_steps = high_contact_steps + 1 if term.diagnostics["contact"] else 0
            failure = None
            if not term.diagnostics["ik_solver_ok"] or term.diagnostics["ik_hold"]:
                failure = "IK solver entered hold mode"
            elif term.diagnostics["fallen"]:
                failure = "controller became physically unstable"
            elif high_contact_steps >= 25:
                failure = "non-foot contact exceeded 25 N for 0.5 s"
            if failure is not None:
                print(f"FAIL step={step} diagnostics={term.diagnostics}", flush=True)
                raise AssertionError(f"WholeBody smoke {failure} at step {step}")

        print(
            f"PASS task={TASK_ID} envs=1 action_dim=10 step_dt={raw_env.step_dt} "
            f"steps={args_cli.steps} diagnostics={term.diagnostics}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
