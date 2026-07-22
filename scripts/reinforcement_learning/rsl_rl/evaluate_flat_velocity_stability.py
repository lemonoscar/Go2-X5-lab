#!/usr/bin/env python3
"""Benchmark a DogOnly flat checkpoint's vx/vy/wz tracking and stopping stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

from flat_velocity_stability_metrics import (  # isort: skip
    MetricThresholds,
    analyze_samples,
    build_schedule,
    write_benchmark_artifacts,
)


TASK_ID = "RobotLab-Isaac-Velocity-Flat-Go2-X5-DogOnly-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--profile", choices=("quick", "full", "planar"), default="quick")
parser.add_argument("--settle-seconds", type=float, default=2.0)
parser.add_argument("--hold-seconds", type=float, default=3.0)
parser.add_argument("--stop-seconds", type=float, default=1.5)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--policy-action-warmup-steps", type=int, default=50)
parser.add_argument("--steady-fraction", type=float, default=0.50)
parser.add_argument("--gain-min", type=float, default=0.70)
parser.add_argument("--gain-max", type=float, default=1.30)
parser.add_argument("--relative-rmse-limit", type=float, default=0.30)
parser.add_argument("--linear-absolute-floor", type=float, default=0.04)
parser.add_argument("--yaw-absolute-floor", type=float, default=0.08)
parser.add_argument("--zero-linear-rmse-limit", type=float, default=0.08)
parser.add_argument("--zero-yaw-rmse-limit", type=float, default=0.10)
parser.add_argument("--max-tilt-rad", type=float, default=0.35)
parser.add_argument("--max-base-height-std", type=float, default=0.05)
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--real-time", action="store_true")
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
from isaaclab.utils import math as math_utils
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
    """Remove evaluation confounders without changing the task's control contract."""

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 4.0
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    _set_if_present(env_cfg.scene.terrain, "max_init_terrain_level", None)
    env_cfg.episode_length_s = max(float(duration_s) + 10.0, 60.0)

    # A one-robot evaluator does not need the large contact buffers used for training.
    _set_if_present(env_cfg.sim.physx, "gpu_max_rigid_contact_count", 2**20)
    _set_if_present(env_cfg.sim.physx, "gpu_max_rigid_patch_count", 5 * 2**12)

    observations = getattr(env_cfg, "observations", None)
    _set_if_present(getattr(observations, "policy", None), "enable_corruption", False)
    _set_if_present(getattr(observations, "critic", None), "enable_corruption", False)

    curriculum = getattr(env_cfg, "curriculum", None)
    for name in (
        "terrain_levels",
        "command_levels_lin_vel",
        "command_levels_ang_vel",
        "command_range",
        "arm_command_range",
        "reward_weights",
    ):
        _set_if_present(curriculum, name, None)

    events = getattr(env_cfg, "events", None)
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

    reset_base = getattr(events, "randomize_reset_base", None)
    if reset_base is not None:
        reset_base.params = {
            "pose_range": {
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
            "velocity_range": {
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        }
    reset_joints = getattr(events, "randomize_reset_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)

    _set_if_present(env_cfg, "sim2sim_action_delay_range", (0, 0))
    _set_if_present(env_cfg, "sim2sim_action_hold_prob", 0.0)
    _set_if_present(env_cfg, "sim2sim_action_noise_std", 0.0)
    _set_if_present(env_cfg, "sim2sim_obs_delay_steps", 0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args_cli.policy_action_warmup_steps < 0:
        raise ValueError("--policy-action-warmup-steps must be non-negative")

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args_cli.output_dir).expanduser().resolve()

    thresholds = MetricThresholds(
        gain_min=args_cli.gain_min,
        gain_max=args_cli.gain_max,
        relative_rmse_limit=args_cli.relative_rmse_limit,
        linear_absolute_floor=args_cli.linear_absolute_floor,
        yaw_absolute_floor=args_cli.yaw_absolute_floor,
        zero_linear_rmse_limit=args_cli.zero_linear_rmse_limit,
        zero_yaw_rmse_limit=args_cli.zero_yaw_rmse_limit,
        max_tilt_rad=args_cli.max_tilt_rad,
        max_base_height_std_m=args_cli.max_base_height_std,
    )
    schedule = build_schedule(
        args_cli.profile,
        settle_s=args_cli.settle_seconds,
        hold_s=args_cli.hold_seconds,
        stop_s=args_cli.stop_seconds,
        repeats=args_cli.repeats,
    )
    total_duration_s = sum(segment.duration_s for segment in schedule)
    expected_evaluated_segments = sum(segment.evaluate for segment in schedule)
    _configure_flat_deterministic(env_cfg, total_duration_s)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = args_cli.device or agent_cfg.device
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.log_dir = os.fspath(checkpoint.parent)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name != "OnPolicyRunner":
        env.close()
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.fspath(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    command_term = raw_env.command_manager.get_term("base_velocity")
    termination_manager = raw_env.termination_manager
    termination_names = termination_manager.active_terms
    observations = env.get_observations()
    policy_obs_dim = int(observations["policy"].shape[-1])
    if policy_obs_dim != 260:
        env.close()
        raise RuntimeError(
            f"DogOnly Flat benchmark requires 260 policy observations, got {policy_obs_dim}. "
            "Use the task and evaluator that match this checkpoint."
        )

    dt = float(raw_env.step_dt)
    samples: list[dict[str, object]] = []
    benchmark_time_s = 0.0
    global_step = 0
    terminated_early = False
    termination_event: dict[str, object] | None = None
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"

    try:
        with samples_path.open("w", encoding="utf-8", buffering=1) as stream:
            for segment_index, segment in enumerate(schedule):
                segment_steps = max(1, round(segment.duration_s / dt))
                print(
                    f"[flat-stability] segment={segment.name} kind={segment.kind} steps={segment_steps} "
                    f"cmd=({segment.vx:.3f}, {segment.vy:.3f}, {segment.wz:.3f})"
                )
                for segment_step in range(segment_steps):
                    wall_start = time.perf_counter()
                    command_term.vel_command_b[:, 0] = segment.vx
                    command_term.vel_command_b[:, 1] = segment.vy
                    command_term.vel_command_b[:, 2] = segment.wz
                    command_term.is_standing_env[:] = (
                        abs(segment.vx) + abs(segment.vy) + abs(segment.wz) <= 1.0e-6
                    )

                    observations = env.get_observations()
                    with torch.inference_mode():
                        actions = policy(observations)
                        if actions.shape[-1] != 12:
                            raise RuntimeError(
                                f"DogOnly Flat benchmark requires 12 policy actions, got {actions.shape[-1]}. "
                                "Use the task and evaluator that match this checkpoint."
                            )
                        if args_cli.policy_action_warmup_steps > 0:
                            action_scale = min(
                                1.0,
                                float(global_step + 1) / float(args_cli.policy_action_warmup_steps),
                            )
                            actions = actions * action_scale
                        else:
                            action_scale = 1.0
                        _, _, dones, _ = env.step(actions)

                    robot = raw_env.scene["robot"]
                    roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
                    roll = math_utils.wrap_to_pi(roll)
                    pitch = math_utils.wrap_to_pi(pitch)
                    yaw = math_utils.wrap_to_pi(yaw)
                    base_pos = robot.data.root_pos_w[0]
                    done = bool(dones[0].item())
                    fired_terminations = [
                        name
                        for name in termination_names
                        if bool(termination_manager.get_term(name)[0].item())
                    ]
                    row = {
                        "time_s": benchmark_time_s,
                        "global_step": global_step,
                        "segment_index": segment_index,
                        "segment_name": segment.name,
                        "segment_kind": segment.kind,
                        "segment_time_s": (segment_step + 1) * dt,
                        "evaluate": segment.evaluate,
                        "cmd_vx": segment.vx,
                        "cmd_vy": segment.vy,
                        "cmd_wz": segment.wz,
                        "measured_vx": float(robot.data.root_lin_vel_b[0, 0].item()),
                        "measured_vy": float(robot.data.root_lin_vel_b[0, 1].item()),
                        "measured_wz": float(robot.data.root_ang_vel_b[0, 2].item()),
                        "base_x": float(base_pos[0].item()),
                        "base_y": float(base_pos[1].item()),
                        "base_z": float(base_pos[2].item()),
                        "base_roll": float(roll[0].item()),
                        "base_pitch": float(pitch[0].item()),
                        "base_yaw": float(yaw[0].item()),
                        "action_abs_mean": float(actions[0].abs().mean().item()),
                        "action_abs_max": float(actions[0].abs().max().item()),
                        "policy_action_scale": action_scale,
                        "done": done,
                        "termination_terms": fired_terminations,
                    }
                    samples.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

                    if global_step % max(1, args_cli.print_every) == 0:
                        print(
                            f"[tracking] t={benchmark_time_s:.2f}s "
                            f"cmd=({segment.vx:.2f},{segment.vy:.2f},{segment.wz:.2f}) "
                            f"meas=({row['measured_vx']:.2f},{row['measured_vy']:.2f},{row['measured_wz']:.2f})"
                        )
                    benchmark_time_s += dt
                    global_step += 1
                    if args_cli.real_time:
                        time.sleep(max(0.0, dt - (time.perf_counter() - wall_start)))
                    if done:
                        terminated_early = True
                        termination_event = {
                            "segment_name": segment.name,
                            "segment_kind": segment.kind,
                            "segment_time_s": row["segment_time_s"],
                            "benchmark_time_s": benchmark_time_s,
                            "terms": fired_terminations,
                        }
                        print(
                            f"[flat-stability] environment terminated during {segment.name}: "
                            f"{fired_terminations}"
                        )
                        break
                if terminated_early:
                    break
    finally:
        env.close()

    summary = analyze_samples(
        samples,
        thresholds=thresholds,
        steady_fraction=args_cli.steady_fraction,
        expected_evaluated_segments=expected_evaluated_segments,
    )
    metadata = {
        "checkpoint": os.fspath(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "task": args_cli.task,
        "profile": args_cli.profile,
        "repeats": args_cli.repeats,
        "seed": args_cli.seed,
        "terrain": "deterministic_plane",
        "control_dt_s": dt,
        "policy_action_warmup_steps": args_cli.policy_action_warmup_steps,
        "terminated_early": terminated_early,
        "termination_event": termination_event,
        "policy_observation_dim": policy_obs_dim,
        "policy_action_dim": 12,
        "schedule": [segment.to_dict() for segment in schedule],
    }
    write_benchmark_artifacts(output_dir, samples, summary, metadata)
    print(
        f"[flat-stability] report={output_dir / 'report.md'} "
        f"commands={summary['passed_command_segments']}/{summary['command_segments']} "
        f"stops={summary['passed_stop_segments']}/{summary['stop_segments']} "
        f"overall_pass={summary['passed']}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
