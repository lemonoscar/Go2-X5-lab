#!/usr/bin/env python3
"""Run the deterministic 60-second Go2-X5 WholeBody inference contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher


TASK_ID = "RobotLab-Isaac-Go2-X5-WholeBody-v0"
REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--print-every", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

SOURCE_ROOT = (REPO_ROOT / "source" / "robot_lab").resolve()
sys.path.insert(0, os.fspath(SOURCE_ROOT))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import robot_lab  # noqa: E402
import robot_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _command(step: int, canonical: torch.Tensor) -> tuple[str, torch.Tensor]:
    command = canonical.clone()
    if step < 250:
        return "canonical_stand", command
    if step < 500:
        command[0] = 0.30
        return "vx_positive", command
    if step < 750:
        command[0] = -0.20
        return "vx_negative", command
    if step < 1000:
        command[1] = 0.15
        return "vy_positive", command
    if step < 1250:
        command[1] = -0.15
        return "vy_negative", command
    if step < 1500:
        command[2] = 0.30
        return "wz_positive", command
    if step < 1750:
        command[2] = -0.30
        return "wz_negative", command
    if step < 2000:
        command[3:6] += torch.tensor([0.04, 0.03, 0.04], device=command.device)
        return "tcp_position_step", command
    if step < 2250:
        command[6:9] = torch.tensor([0.25, -0.20, 0.30], device=command.device)
        return "tcp_orientation_step", command
    if step < 2375:
        command[9] = 1.0
        return "gripper_open", command
    if step < 2500:
        command[9] = 0.0
        return "gripper_close", command
    if step < 2550:
        return "finite_clip", torch.tensor(
            [3.0, -2.0, 4.0, 3.0, 3.0, 3.0, 2.0, -2.0, 2.0, 5.0],
            device=command.device,
        )
    if step == 2550:
        return "nonfinite_reject", torch.full((10,), float("nan"), device=command.device)
    if step < 2750:
        command[3:6] = torch.tensor([0.0, 0.77, 0.0], device=command.device)
        command[6:9] = torch.tensor([1.0, 1.0, 1.0], device=command.device)
        return "ik_extreme_target", command
    return "canonical_recovery", command


def main() -> None:
    if args_cli.task != TASK_ID:
        raise ValueError(f"This replay is locked to {TASK_ID}, got {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    if unwrapped.action_manager.total_action_dim != 10:
        raise ValueError(
            f"WholeBody environment action dim must be 10, got {unwrapped.action_manager.total_action_dim}"
        )
    env.reset()
    action_term = unwrapped.action_manager.get_term("whole_body")
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]
    nonfoot_body_ids = [
        index for index, name in enumerate(contact_sensor.body_names) if not name.endswith("_foot")
    ]
    canonical = action_term.processed_actions[0].clone()
    rows: list[dict] = []
    numerical_failures = 0
    unexpected_resets = 0
    start = time.perf_counter()

    for step in range(3000):
        segment, action = _command(step, canonical)
        _, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
        diagnostics = dict(action_term.diagnostics)
        targets = robot.data.joint_pos_target[0]
        finite = bool(torch.isfinite(targets).all().item())
        numerical_failures += int(not finite)
        reset = bool((terminated | truncated).any().item())
        unexpected_resets += int(reset)
        projected_gravity = robot.data.projected_gravity_b[0]
        fallen = bool(robot.data.root_pos_w[0, 2] < 0.28 or projected_gravity[2] > -0.5)
        nonfoot_contact_n = float(
            torch.linalg.vector_norm(
                contact_sensor.data.net_forces_w[0, nonfoot_body_ids], dim=-1
            ).max().item()
        )
        contact = nonfoot_contact_n >= 25.0
        row = {
            "step": step,
            "time_s": step * 0.02,
            "segment": segment,
            "original_command": action.detach().cpu().tolist(),
            "applied_command": action_term.processed_actions[0].detach().cpu().tolist(),
            "clipped_mask": diagnostics.get("clipped_mask", torch.zeros(10, dtype=torch.bool))
            .detach()
            .cpu()
            .tolist(),
            "command_rejected": diagnostics.get("command_rejected", False),
            "ik_hold": diagnostics.get("ik_hold", False),
            "stalled": diagnostics.get("stalled", False),
            "ik_position_error_m": diagnostics.get("ik_position_error_m"),
            "ik_orientation_error_rad": diagnostics.get("ik_orientation_error_rad"),
            "ik_command_position_error_m": diagnostics.get("ik_command_position_error_m"),
            "ik_command_orientation_error_rad": diagnostics.get("ik_command_orientation_error_rad"),
            "controller_time_ms": diagnostics.get("controller_time_ms"),
            "fallen": fallen,
            "contact": contact,
            "max_nonfoot_contact_n": nonfoot_contact_n,
            "base_position_world": robot.data.root_pos_w[0].detach().cpu().tolist(),
            "joint_targets_finite": finite,
            "unexpected_reset": reset,
        }
        rows.append(row)
        if step % args_cli.print_every == 0:
            print(
                f"step={step:04d}/3000 segment={segment} z={row['base_position_world'][2]:.3f} "
                f"fallen={int(fallen)} ik_hold={int(row['ik_hold'])} stalled={int(row['stalled'])}"
            )

    elapsed = time.perf_counter() - start
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args_cli.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=True) + "\n")
    segments = sorted({row["segment"] for row in rows})
    controller_times_ms = sorted(
        float(row["controller_time_ms"])
        for row in rows
        if row["controller_time_ms"] is not None
    )
    summary = {
        "task": TASK_ID,
        "steps": len(rows),
        "simulated_seconds": len(rows) * 0.02,
        "wall_seconds": elapsed,
        "controller_steps_per_wall_second": len(rows) / elapsed,
        "segments": segments,
        "numerical_failures": numerical_failures,
        "unexpected_resets": unexpected_resets,
        "fallen_steps": sum(row["fallen"] for row in rows),
        "contact_steps": sum(row["contact"] for row in rows),
        "ik_hold_steps": sum(row["ik_hold"] for row in rows),
        "stalled_steps": sum(row["stalled"] for row in rows),
        "rejected_packets": sum(row["command_rejected"] for row in rows),
        "controller_time_ms_mean": sum(controller_times_ms) / len(controller_times_ms),
        "controller_time_ms_p95": controller_times_ms[round(0.95 * (len(controller_times_ms) - 1))],
        "controller_time_ms_max": controller_times_ms[-1],
        "contract_pass": numerical_failures == 0 and unexpected_resets == 0 and len(rows) == 3000,
        "stability_pass": not any(row["fallen"] for row in rows),
    }
    (args_cli.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args_cli.output_dir / "REPORT.md").write_text(
        "# Go2-X5 WholeBody 60 秒回放\n\n"
        f"- Contract pass: `{summary['contract_pass']}`\n"
        f"- Stability pass: `{summary['stability_pass']}`\n"
        f"- Steps: `{summary['steps']}` / simulated `{summary['simulated_seconds']:.1f} s`\n"
        f"- Unexpected resets: `{summary['unexpected_resets']}`\n"
        f"- Non-finite targets: `{summary['numerical_failures']}`\n"
        f"- Fallen / IK hold / stalled steps: `{summary['fallen_steps']}` / "
        f"`{summary['ik_hold_steps']}` / `{summary['stalled_steps']}`\n"
        f"- Non-foot contact steps (>=25 N): `{summary['contact_steps']}`\n"
        f"- Rejected packets: `{summary['rejected_packets']}`\n"
        f"- Controller time mean / p95 / max: `{summary['controller_time_ms_mean']:.3f}` / "
        f"`{summary['controller_time_ms_p95']:.3f}` / `{summary['controller_time_ms_max']:.3f}` ms\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()
    if not summary["contract_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
