#!/usr/bin/env python3
"""Evaluate a DogOnly checkpoint on deterministic flat-ground vx tracking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip


TASK_ID = "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyRoughStairsVx-v0"
DEFAULT_SPEEDS = tuple(round(index * 0.1, 1) for index in range(8))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
parser.add_argument("--settle-seconds", type=float, default=2.0)
parser.add_argument("--hold-seconds", type=float, default=5.0)
parser.add_argument("--post-stop-seconds", type=float, default=5.0)
parser.add_argument("--absolute-tolerance", type=float, default=0.1)
parser.add_argument("--relative-tolerance", type=float, default=0.1)
parser.add_argument("--lateral-tolerance", type=float, default=0.1)
parser.add_argument("--yaw-tolerance", type=float, default=0.1)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_if_present(owner, name: str, value) -> None:
    if owner is not None and hasattr(owner, name):
        setattr(owner, name, value)


def _configure_flat_deterministic(env_cfg: ManagerBasedRLEnvCfg, duration_s: float) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 4.0
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.episode_length_s = max(duration_s + 10.0, 90.0)
    # The training task reserves contact buffers for 1024 parallel robots.
    # Keeping those capacities in a one-robot evaluator wastes roughly a GiB
    # and can OOM when a training process is resident on the same GPU.
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 5 * 2**12
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False

    for name in ("terrain_levels", "command_levels_lin_vel", "command_levels_ang_vel"):
        _set_if_present(env_cfg.curriculum, name, None)

    events = env_cfg.events
    for name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
    ):
        _set_if_present(events, name, None)

    if events.randomize_reset_base is not None:
        events.randomize_reset_base.params = {
            "pose_range": {
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "velocity_range": {
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        }
    if events.randomize_reset_joints is not None:
        events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)

    env_cfg.sim2sim_action_delay_range = (0, 0)
    env_cfg.sim2sim_action_hold_prob = 0.0
    env_cfg.sim2sim_action_noise_std = 0.0
    env_cfg.sim2sim_obs_delay_steps = 0


def _summarize_segment(
    name: str,
    command_vx: float,
    vx_samples: list[float],
    vy_samples: list[float],
    wz_samples: list[float],
    terminated: bool,
    required_for_overall: bool = True,
) -> dict[str, object]:
    if not vx_samples:
        raise RuntimeError(f"No samples collected for segment {name}.")

    mean_vx = sum(vx_samples) / len(vx_samples)
    mean_vy = sum(vy_samples) / len(vy_samples)
    mean_wz = sum(wz_samples) / len(wz_samples)
    rmse_vx = math.sqrt(sum((value - command_vx) ** 2 for value in vx_samples) / len(vx_samples))
    tolerance = max(
        args_cli.absolute_tolerance,
        args_cli.relative_tolerance * abs(command_vx),
    )
    checks = {
        "mean_vx": abs(mean_vx - command_vx) <= tolerance,
        "rmse_vx": rmse_vx <= tolerance,
        "mean_vy": abs(mean_vy) <= args_cli.lateral_tolerance,
        "mean_wz": abs(mean_wz) <= args_cli.yaw_tolerance,
        "no_termination": not terminated,
    }
    return {
        "name": name,
        "required_for_overall": required_for_overall,
        "command_vx_mps": command_vx,
        "tolerance_mps": tolerance,
        "mean_vx_mps": mean_vx,
        "rmse_vx_mps": rmse_vx,
        "mean_vy_mps": mean_vy,
        "mean_wz_radps": mean_wz,
        "within_vx_tolerance_fraction": sum(
            abs(value - command_vx) <= tolerance for value in vx_samples
        )
        / len(vx_samples),
        "max_abs_vx_error_mps": max(abs(value - command_vx) for value in vx_samples),
        "sample_count": len(vx_samples),
        "terminated": terminated,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_outputs(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results = payload["results"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "name",
            "required_for_overall",
            "command_vx_mps",
            "tolerance_mps",
            "mean_vx_mps",
            "rmse_vx_mps",
            "mean_vy_mps",
            "mean_wz_radps",
            "within_vx_tolerance_fraction",
            "max_abs_vx_error_mps",
            "sample_count",
            "terminated",
            "passed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result[name] for name in fieldnames})

    lines = [
        "# Go2-X5 flat vx tracking report",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Overall pass: **{payload['passed']}**",
        f"- Passed required segments: {payload['passed_count']}/{payload['required_count']}",
        "- The post-maximum-speed stop segment is a reported braking diagnostic, not a steady-state gate.",
        "- vx acceptance: both mean error and RMSE <= max(0.1 m/s, 10% command)",
        "- leakage acceptance: |mean vy| <= 0.1 m/s and |mean wz| <= 0.1 rad/s",
        "",
        "| segment | required | cmd vx | mean vx | RMSE vx | mean vy | mean wz | pass |",
        "|---|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        lines.append(
            "| {name} | {required_for_overall} | {command_vx_mps:.2f} | {mean_vx_mps:.3f} | "
            "{rmse_vx_mps:.3f} | {mean_vy_mps:.3f} | {mean_wz_radps:.3f} | {passed} |".format(
                **result
            )
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args_cli.settle_seconds < 0.0 or args_cli.hold_seconds <= 0.0:
        raise ValueError("settle-seconds must be non-negative and hold-seconds must be positive")

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args_cli.output_dir).expanduser().resolve()

    total_duration_s = len(args_cli.speeds) * (
        args_cli.settle_seconds + args_cli.hold_seconds
    ) + args_cli.post_stop_seconds
    _configure_flat_deterministic(env_cfg, total_duration_s)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = args_cli.device or agent_cfg.device
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.log_dir = os.fspath(checkpoint.parent)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.fspath(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    command_term = raw_env.command_manager.get_term("base_velocity")
    obs = env.get_observations()
    if obs["policy"].shape[-1] != 260:
        raise RuntimeError(f"Expected 260 policy observations, got {obs['policy'].shape[-1]}")

    dt = float(raw_env.step_dt)
    results: list[dict[str, object]] = []

    def run_segment(
        name: str,
        command_vx: float,
        settle_s: float,
        hold_s: float,
        required_for_overall: bool = True,
    ) -> dict[str, object]:
        settle_steps = round(settle_s / dt)
        hold_steps = max(1, round(hold_s / dt))
        vx_samples: list[float] = []
        vy_samples: list[float] = []
        wz_samples: list[float] = []
        terminated = False

        for step in range(settle_steps + hold_steps):
            command_term.vel_command_b[:] = 0.0
            command_term.vel_command_b[:, 0] = command_vx
            command_term.is_standing_env[:] = command_vx == 0.0
            obs = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                if actions.shape[-1] != 12:
                    raise RuntimeError(f"Expected 12 policy actions, got {actions.shape[-1]}")
                _, _, dones, _ = env.step(actions)

            if step >= settle_steps:
                robot = raw_env.scene["robot"]
                vx_samples.append(float(robot.data.root_lin_vel_b[0, 0].item()))
                vy_samples.append(float(robot.data.root_lin_vel_b[0, 1].item()))
                wz_samples.append(float(robot.data.root_ang_vel_b[0, 2].item()))
            if bool(dones[0].item()):
                terminated = True
                break

        result = _summarize_segment(
            name,
            command_vx,
            vx_samples,
            vy_samples,
            wz_samples,
            terminated,
            required_for_overall,
        )
        print(
            f"[vx] {name}: cmd={command_vx:.2f} mean={result['mean_vx_mps']:.3f} "
            f"rmse={result['rmse_vx_mps']:.3f} vy={result['mean_vy_mps']:.3f} "
            f"wz={result['mean_wz_radps']:.3f} pass={result['passed']}"
        )
        return result

    try:
        for index, command_vx in enumerate(args_cli.speeds):
            results.append(
                run_segment(
                    f"vx_{index:02d}_{command_vx:.1f}",
                    float(command_vx),
                    args_cli.settle_seconds,
                    args_cli.hold_seconds,
                )
            )
        if args_cli.post_stop_seconds > 0.0:
            results.append(
                run_segment(
                    "post_max_speed_stop",
                    0.0,
                    0.0,
                    args_cli.post_stop_seconds,
                    required_for_overall=False,
                )
            )
    finally:
        env.close()

    required_results = [result for result in results if result["required_for_overall"]]
    passed_count = sum(bool(result["passed"]) for result in required_results)
    payload = {
        "task": args_cli.task,
        "checkpoint": os.fspath(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "seed": args_cli.seed,
        "terrain": "deterministic_plane",
        "control_dt_s": dt,
        "absolute_tolerance_mps": args_cli.absolute_tolerance,
        "relative_tolerance": args_cli.relative_tolerance,
        "lateral_tolerance_mps": args_cli.lateral_tolerance,
        "yaw_tolerance_radps": args_cli.yaw_tolerance,
        "passed_count": passed_count,
        "required_count": len(required_results),
        "passed": passed_count == len(required_results),
        "results": results,
    }
    _write_outputs(output_dir, payload)
    print(f"[vx] report={output_dir / 'report.md'} overall_pass={payload['passed']}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
