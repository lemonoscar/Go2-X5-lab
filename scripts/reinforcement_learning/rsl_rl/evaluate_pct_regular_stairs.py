"""Evaluate a DogOnly checkpoint on regularized one-way/up-down or scanned PCT stairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip


TASK_ID = "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularStairs-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--mode",
    choices=("nominal", "robust"),
    default="robust",
    help="nominal removes domain randomization; robust keeps the task's small deployment randomization.",
)
parser.add_argument("--output", type=str, default=None, help="JSON result path.")
parser.add_argument(
    "--geometry-only",
    action="store_true",
    help="Inspect the selected terrain, then exit without loading a policy.",
)
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

from rl_utils import ActionDelayWrapper
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.pct_stairs_terrain import (
    PCT_MEASURED_FIRST_RISER_PROGRESS_M,
    PCT_MEASURED_LAST_RISER_PROGRESS_M,
    PCT_MEASURED_STAIR_CENTERLINE_RUN_M,
    PCT_REGULAR_STAIR_APPROACH_M,
    PCT_REGULAR_STAIR_COUNT,
    PCT_REGULAR_STAIR_FLIGHT_RISE_M,
    PCT_REGULAR_STAIR_RISER_M,
    PCT_REGULAR_STAIR_TREAD_M,
    PCT_REGULAR_STAIR_WIDTH_M,
    PCT_REGULAR_UP_DOWN_DESCENT_END_M,
    PCT_REGULAR_UP_DOWN_DESCENT_START_M,
    PCT_REGULAR_UP_DOWN_PATH_LENGTH_M,
    PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _disable_domain_randomization(env_cfg: ManagerBasedRLEnvCfg) -> None:
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False

    events = env_cfg.events
    reset_params = events.randomize_reset_base.params
    if "path_progress_height_anchors" in reset_params:
        reset_params["lateral_offset_range"] = (0.0, 0.0)
        reset_params["forward_offset_range"] = (0.0, 0.0)
        reset_params["height_offset_range"] = (0.0, 0.0)
        reset_params["roll_range"] = (0.0, 0.0)
        reset_params["pitch_jitter_range"] = (0.0, 0.0)
        reset_params["yaw_jitter_range"] = (math.pi / 2.0, math.pi / 2.0)
        reset_params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    else:
        events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (math.pi / 2.0, math.pi / 2.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    events.randomize_rigid_body_material.params["static_friction_range"] = (1.0, 1.0)
    events.randomize_rigid_body_material.params["dynamic_friction_range"] = (1.0, 1.0)
    events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
    events.randomize_rigid_body_mass_base = None
    events.randomize_rigid_body_mass_others = None
    events.randomize_com_positions = None
    events.randomize_actuator_gains = None
    env_cfg.sim2sim_action_hold_prob = 0.0
    env_cfg.sim2sim_action_noise_std = 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return (1.0 - alpha) * ordered[lower] + alpha * ordered[upper]


def _inspect_pct_terrain_geometry(env_cfg: ManagerBasedRLEnvCfg) -> dict[str, object]:
    """Identify the selected PCT terrain and verify regular-box dimensions when present."""
    generator_cfg = env_cfg.scene.terrain.terrain_generator
    if generator_cfg is None:
        raise RuntimeError("The task does not contain a terrain generator.")

    if "pct_regular_up_down_stairs" in generator_cfg.sub_terrains:
        if set(generator_cfg.sub_terrains) != {"pct_regular_up_down_stairs"}:
            raise RuntimeError("The regular up/down comparison must contain exactly one terrain.")
        stair_cfg = generator_cfg.sub_terrains["pct_regular_up_down_stairs"]
        meshes, origin = stair_cfg.function(1.0, stair_cfg)
        ascent_start = 1
        platform_index = ascent_start + PCT_REGULAR_STAIR_COUNT
        descent_start = platform_index + 1
        ascent_meshes = meshes[ascent_start:platform_index]
        platform_mesh = meshes[platform_index]
        descent_meshes = meshes[
            descent_start : descent_start + PCT_REGULAR_STAIR_COUNT
        ]
        if len(ascent_meshes) != PCT_REGULAR_STAIR_COUNT:
            raise RuntimeError(
                f"Expected {PCT_REGULAR_STAIR_COUNT} ascent boxes, got {len(ascent_meshes)}."
            )
        if len(descent_meshes) != PCT_REGULAR_STAIR_COUNT:
            raise RuntimeError(
                f"Expected {PCT_REGULAR_STAIR_COUNT} descent boxes, got {len(descent_meshes)}."
            )

        ascent_tops = [float(mesh.bounds[1, 2]) for mesh in ascent_meshes]
        descent_tops = [float(mesh.bounds[1, 2]) for mesh in descent_meshes]
        expected_ascent_tops = [
            (index + 1) * PCT_REGULAR_STAIR_RISER_M
            for index in range(PCT_REGULAR_STAIR_COUNT)
        ]
        expected_descent_tops = list(reversed(expected_ascent_tops))
        all_step_meshes = ascent_meshes + descent_meshes
        tread_depths = [
            float(mesh.bounds[1, 1] - mesh.bounds[0, 1]) for mesh in all_step_meshes
        ]
        clear_widths = [
            float(mesh.bounds[1, 0] - mesh.bounds[0, 0]) for mesh in all_step_meshes
        ]
        platform_length = float(platform_mesh.bounds[1, 1] - platform_mesh.bounds[0, 1])
        first_riser_progress = float(ascent_meshes[0].bounds[0, 1] - origin[1])
        descent_start_progress = float(descent_meshes[0].bounds[0, 1] - origin[1])
        descent_end_progress = float(descent_meshes[-1].bounds[1, 1] - origin[1])

        tolerance = 1.0e-6
        checks = {
            "first_riser_progress": (
                first_riser_progress,
                PCT_REGULAR_STAIR_APPROACH_M,
            ),
            "middle_platform_length": (
                platform_length,
                PCT_REGULAR_UP_DOWN_TOP_PLATFORM_M,
            ),
            "descent_start_progress": (
                descent_start_progress,
                PCT_REGULAR_UP_DOWN_DESCENT_START_M,
            ),
            "descent_end_progress": (
                descent_end_progress,
                PCT_REGULAR_UP_DOWN_DESCENT_END_M,
            ),
            "minimum_tread_depth": (min(tread_depths), PCT_REGULAR_STAIR_TREAD_M),
            "maximum_tread_depth": (max(tread_depths), PCT_REGULAR_STAIR_TREAD_M),
            "minimum_clear_width": (min(clear_widths), PCT_REGULAR_STAIR_WIDTH_M),
            "maximum_clear_width": (max(clear_widths), PCT_REGULAR_STAIR_WIDTH_M),
        }
        checks.update(
            {
                f"ascent_top_{index}": (actual, expected)
                for index, (actual, expected) in enumerate(
                    zip(ascent_tops, expected_ascent_tops)
                )
            }
        )
        checks.update(
            {
                f"descent_top_{index}": (actual, expected)
                for index, (actual, expected) in enumerate(
                    zip(descent_tops, expected_descent_tops)
                )
            }
        )
        mismatches = [
            f"{name}: generated={actual:.9f}, expected={expected:.9f}"
            for name, (actual, expected) in checks.items()
            if abs(actual - expected) > tolerance
        ]
        if mismatches:
            raise RuntimeError("Regular up/down stair geometry mismatch: " + "; ".join(mismatches))

        return {
            "passed": True,
            "terrain_variant": "regular_box_up_down",
            "origin_xyz_m": [float(value) for value in origin],
            "steps_per_flight": PCT_REGULAR_STAIR_COUNT,
            "riser_height_m": PCT_REGULAR_STAIR_RISER_M,
            "tread_depth_m": tread_depths[0],
            "clear_width_m": clear_widths[0],
            "middle_platform_length_m": platform_length,
            "ascent_step_top_heights_m": ascent_tops,
            "descent_step_top_heights_m": descent_tops,
            "descent_end_progress_m": descent_end_progress,
            "path_length_m": PCT_REGULAR_UP_DOWN_PATH_LENGTH_M,
        }

    if "pct_regular_stairs" not in generator_cfg.sub_terrains:
        if set(generator_cfg.sub_terrains) != {"pct_scanned_first_flight"}:
            raise RuntimeError(
                "Expected exactly one pct_regular_stairs or pct_scanned_first_flight sub-terrain, "
                f"got {sorted(generator_cfg.sub_terrains)}."
            )
        stair_cfg = generator_cfg.sub_terrains["pct_scanned_first_flight"]
        if abs(float(stair_cfg.proportion) - 1.0) > 1.0e-9:
            raise RuntimeError("The scanned PCT comparison terrain must have proportion=1.0.")
        return {
            "passed": True,
            "terrain_variant": "pct_scanned_collision",
            "generator_function": stair_cfg.function.__name__,
            "start_position_xy_m": [float(value) for value in stair_cfg.start_position],
            "flat_approach_m": float(stair_cfg.approach_length),
            "configured_flight_run_m": float(stair_cfg.flight_run),
            "configured_route_width_m": float(stair_cfg.route_width),
            "scan_crop_mode": stair_cfg.scan_crop_mode,
            "scan_target_rise_range_m": [
                float(value) for value in stair_cfg.scan_target_rise_range
            ],
            "scan_include_auxiliary_floor": bool(stair_cfg.scan_include_auxiliary_floor),
            "scan_include_auxiliary_top_platform": bool(
                stair_cfg.scan_include_auxiliary_top_platform
            ),
        }

    stair_cfg = generator_cfg.sub_terrains["pct_regular_stairs"]
    meshes, origin = stair_cfg.function(1.0, stair_cfg)
    step_meshes = meshes[1 : 1 + PCT_REGULAR_STAIR_COUNT]
    if len(step_meshes) != PCT_REGULAR_STAIR_COUNT:
        raise RuntimeError(
            f"Expected {PCT_REGULAR_STAIR_COUNT} generated steps, got {len(step_meshes)}."
        )

    top_heights_m = [float(mesh.bounds[1, 2]) for mesh in step_meshes]
    tread_depths_m = [float(mesh.bounds[1, 1] - mesh.bounds[0, 1]) for mesh in step_meshes]
    clear_widths_m = [float(mesh.bounds[1, 0] - mesh.bounds[0, 0]) for mesh in step_meshes]
    first_riser_progress_m = float(step_meshes[0].bounds[0, 1] - origin[1])
    riser_heights_m = [top_heights_m[0]] + [
        top_heights_m[index] - top_heights_m[index - 1]
        for index in range(1, len(top_heights_m))
    ]

    tolerance = 1.0e-6
    checks = {
        "first_riser_progress": (
            first_riser_progress_m,
            PCT_REGULAR_STAIR_APPROACH_M,
        ),
        "first_step_top": (top_heights_m[0], PCT_REGULAR_STAIR_RISER_M),
        "last_step_top": (top_heights_m[-1], PCT_REGULAR_STAIR_FLIGHT_RISE_M),
        "minimum_tread_depth": (min(tread_depths_m), PCT_REGULAR_STAIR_TREAD_M),
        "maximum_tread_depth": (max(tread_depths_m), PCT_REGULAR_STAIR_TREAD_M),
        "minimum_clear_width": (min(clear_widths_m), PCT_REGULAR_STAIR_WIDTH_M),
        "maximum_clear_width": (max(clear_widths_m), PCT_REGULAR_STAIR_WIDTH_M),
        "minimum_riser_height": (min(riser_heights_m), PCT_REGULAR_STAIR_RISER_M),
        "maximum_riser_height": (max(riser_heights_m), PCT_REGULAR_STAIR_RISER_M),
    }
    mismatches = [
        f"{name}: generated={actual:.9f}, expected={expected:.9f}"
        for name, (actual, expected) in checks.items()
        if abs(actual - expected) > tolerance
    ]
    if mismatches:
        raise RuntimeError("Regular stair mesh geometry mismatch: " + "; ".join(mismatches))

    return {
        "passed": True,
        "terrain_variant": "regular_box",
        "origin_xyz_m": [float(value) for value in origin],
        "flat_approach_to_first_riser_m": first_riser_progress_m,
        "tread_depth_m": tread_depths_m[0],
        "clear_width_m": clear_widths_m[0],
        "riser_heights_m": riser_heights_m,
        "step_top_heights_m": top_heights_m,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    terrain_geometry_check = _inspect_pct_terrain_geometry(env_cfg)
    print("[GEOMETRY] " + json.dumps(terrain_geometry_check, ensure_ascii=False))
    if args_cli.geometry_only:
        if args_cli.output:
            output = Path(args_cli.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {"task": args_cli.task, "terrain_geometry_check": terrain_geometry_check},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[GEOMETRY] wrote {output}")
        return

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args_cli.num_envs <= 0 or args_cli.episodes <= 0:
        raise ValueError("--num_envs and --episodes must be positive")

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = args_cli.device or agent_cfg.device
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.terrain_generator.num_cols = args_cli.num_envs
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.log_dir = os.fspath(checkpoint.parent)
    if args_cli.mode == "nominal":
        _disable_domain_randomization(env_cfg)

    base_command_cfg = env_cfg.commands.base_velocity
    path_points = tuple(base_command_cfg.path_points_xy)
    path_length_m = sum(
        math.hypot(
            float(path_points[index][0] - path_points[index - 1][0]),
            float(path_points[index][1] - path_points[index - 1][1]),
        )
        for index in range(1, len(path_points))
    )
    total_rise_m = float(base_command_cfg.maximum_total_rise)
    platform_gate_progress_m = float(base_command_cfg.completion_progress_ratio) * path_length_m
    completion_height_ratio = float(base_command_cfg.completion_height_ratio)
    completion_return_height_tolerance = getattr(
        base_command_cfg, "completion_return_height_tolerance", None
    )
    completion_peak_height_ratio = float(
        getattr(base_command_cfg, "completion_peak_height_ratio", 0.0)
    )

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    delay_range = getattr(env_cfg, "sim2sim_action_delay_range", (0, 0))
    hold_prob = getattr(env_cfg, "sim2sim_action_hold_prob", 0.0)
    action_noise = getattr(env_cfg, "sim2sim_action_noise_std", 0.0)
    if max(delay_range) > 0 or hold_prob > 0.0 or action_noise > 0.0:
        gym_env = ActionDelayWrapper(
            gym_env,
            delay_steps_range=tuple(delay_range),
            hold_prob=hold_prob,
            action_noise_std=action_noise,
        )
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.fspath(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    policy_obs_shape = tuple(obs["policy"].shape)
    if policy_obs_shape[-1] != 260:
        raise RuntimeError(f"Expected 260 policy observations, got {policy_obs_shape[-1]}")

    raw_env = env.unwrapped
    command_term = raw_env.command_manager.get_term("base_velocity")
    termination_manager = raw_env.termination_manager
    term_names = termination_manager.active_terms
    contact_cfg = termination_manager.get_term_cfg("illegal_contact").params["sensor_cfg"]
    contact_sensor = raw_env.scene.sensors[contact_cfg.name]
    num_envs = args_cli.num_envs
    device = raw_env.device

    peak_progress = torch.zeros(num_envs, device=device)
    peak_height_gain = torch.zeros(num_envs, device=device)
    last_height_gain = torch.zeros(num_envs, device=device)
    start_progress = torch.zeros(num_envs, device=device)
    start_height_gain = torch.zeros(num_envs, device=device)
    peak_tilt_deg = torch.zeros(num_envs, device=device)
    peak_critical_contact = torch.zeros(num_envs, device=device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_records: list[dict[str, object]] = []

    max_episode_steps = math.ceil(env_cfg.episode_length_s / raw_env.step_dt) + 2
    max_total_steps = max_episode_steps * (math.ceil(args_cli.episodes / num_envs) + 2)
    total_steps = 0

    try:
        while len(episode_records) < args_cli.episodes and total_steps < max_total_steps:
            progress = command_term.path_progress_m
            height_gain = command_term.height_gain_m
            capture_start = episode_steps <= 1
            start_progress = torch.where(capture_start, progress, start_progress)
            start_height_gain = torch.where(capture_start, height_gain, start_height_gain)
            projected_up = torch.clamp(
                -raw_env.scene["robot"].data.projected_gravity_b[:, 2], -1.0, 1.0
            )
            tilt_deg = torch.rad2deg(torch.acos(projected_up))
            forces = contact_sensor.data.net_forces_w_history[:, :, contact_cfg.body_ids, :]
            critical_contact = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))

            peak_progress = torch.maximum(peak_progress, progress)
            peak_height_gain = torch.maximum(peak_height_gain, height_gain)
            last_height_gain.copy_(height_gain)
            peak_tilt_deg = torch.maximum(peak_tilt_deg, tilt_deg)
            peak_critical_contact = torch.maximum(peak_critical_contact, critical_contact)
            episode_steps += 1

            with torch.inference_mode():
                actions = policy(obs)
                if actions.shape[-1] != 12:
                    raise RuntimeError(f"Expected 12 policy actions, got {actions.shape[-1]}")
                obs, _, dones, _ = env.step(actions)
            total_steps += 1

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for env_id_tensor in done_ids:
                env_id = int(env_id_tensor.item())
                fired_terms = [
                    name
                    for name in term_names
                    if bool(termination_manager.get_term(name)[env_id].item())
                ]
                episode_records.append(
                    {
                        "episode": len(episode_records),
                        "env_id": env_id,
                        "success": "pct_path_completed" in fired_terms,
                        "termination_terms": fired_terms,
                        "steps": int(episode_steps[env_id].item()),
                        "duration_s": float(episode_steps[env_id].item() * raw_env.step_dt),
                        "start_progress_m": float(start_progress[env_id].item()),
                        "start_height_gain_m": float(start_height_gain[env_id].item()),
                        "peak_progress_m": float(peak_progress[env_id].item()),
                        "peak_height_gain_m": float(peak_height_gain[env_id].item()),
                        "final_height_gain_m": float(last_height_gain[env_id].item()),
                        "peak_tilt_deg": float(peak_tilt_deg[env_id].item()),
                        "peak_critical_contact_n": float(peak_critical_contact[env_id].item()),
                    }
                )
                if len(episode_records) >= args_cli.episodes:
                    break

            if done_ids.numel() > 0:
                peak_progress[done_ids] = 0.0
                peak_height_gain[done_ids] = 0.0
                last_height_gain[done_ids] = 0.0
                start_progress[done_ids] = 0.0
                start_height_gain[done_ids] = 0.0
                peak_tilt_deg[done_ids] = 0.0
                peak_critical_contact[done_ids] = 0.0
                episode_steps[done_ids] = 0
    finally:
        env.close()

    if len(episode_records) < args_cli.episodes:
        raise RuntimeError(
            f"Evaluation stopped after {len(episode_records)} episodes and {total_steps} steps"
        )

    records = episode_records[: args_cli.episodes]
    successes = [bool(record["success"]) for record in records]
    start_progress_values = [float(record["start_progress_m"]) for record in records]
    start_height_values = [float(record["start_height_gain_m"]) for record in records]
    progress_values = [float(record["peak_progress_m"]) for record in records]
    height_values = [float(record["peak_height_gain_m"]) for record in records]
    final_height_values = [float(record["final_height_gain_m"]) for record in records]
    tilt_values = [float(record["peak_tilt_deg"]) for record in records]
    contact_values = [float(record["peak_critical_contact_n"]) for record in records]
    platform_gate_reached = [
        progress >= platform_gate_progress_m
        and height
        >= (
            completion_peak_height_ratio
            if completion_return_height_tolerance is not None
            else completion_height_ratio
        )
        * total_rise_m
        for progress, height in zip(progress_values, height_values)
    ]
    top_reached = [
        height
        >= (
            completion_peak_height_ratio
            if completion_return_height_tolerance is not None
            else completion_height_ratio
        )
        * total_rise_m
        for height in height_values
    ]
    returned_to_final_height = [
        True
        if completion_return_height_tolerance is None
        else abs(height) <= float(completion_return_height_tolerance)
        for height in final_height_values
    ]
    completion_state_reached = [
        gate and returned
        for gate, returned in zip(platform_gate_reached, returned_to_final_height)
    ]
    termination_counts = {
        name: sum(name in record["termination_terms"] for record in records) for name in term_names
    }
    stage_reach_rates = {
        f"{int(fraction * 100)}%": sum(
            progress >= fraction * platform_gate_progress_m
            for progress in progress_values
        )
        / len(records)
        for fraction in (0.25, 0.50, 0.75, 1.0)
    }

    summary = {
        "task": args_cli.task,
        "mode": args_cli.mode,
        "checkpoint": os.fspath(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "seed": args_cli.seed,
        "num_envs": num_envs,
        "episodes": len(records),
        "policy_observation_dim": policy_obs_shape[-1],
        "policy_action_dim": 12,
        "geometry": {
            "terrain_variant": terrain_geometry_check["terrain_variant"],
            "measured_centerline_run_m": PCT_MEASURED_STAIR_CENTERLINE_RUN_M,
            "first_riser_progress_m": PCT_MEASURED_FIRST_RISER_PROGRESS_M,
            "last_riser_progress_m": PCT_MEASURED_LAST_RISER_PROGRESS_M,
            "total_rise_m": total_rise_m,
            "platform_gate_progress_m": platform_gate_progress_m,
            "completion_height_ratio": completion_height_ratio,
            "completion_peak_height_ratio": completion_peak_height_ratio,
            "completion_return_height_tolerance_m": completion_return_height_tolerance,
            "path_length_m": path_length_m,
            "forward_velocity_mps": float(base_command_cfg.forward_velocity),
            "episode_length_s": float(env_cfg.episode_length_s),
            "terrain_geometry_check": terrain_geometry_check,
        },
        "results": {
            "success_count": sum(successes),
            "success_rate": sum(successes) / len(records),
            "platform_gate_reach_count": sum(platform_gate_reached),
            "platform_gate_reach_rate": sum(platform_gate_reached) / len(records),
            "top_reach_count": sum(top_reached),
            "top_reach_rate": sum(top_reached) / len(records),
            "completion_state_reach_count": sum(completion_state_reached),
            "completion_state_reach_rate": sum(completion_state_reached) / len(records),
            "mean_start_progress_m": _mean(start_progress_values),
            "mean_start_height_gain_m": _mean(start_height_values),
            "mean_completion_to_platform_gate": _mean(
                [min(progress / platform_gate_progress_m, 1.0) for progress in progress_values]
            ),
            "mean_peak_progress_m": _mean(progress_values),
            "median_peak_progress_m": _percentile(progress_values, 0.50),
            "p10_peak_progress_m": _percentile(progress_values, 0.10),
            "p90_peak_progress_m": _percentile(progress_values, 0.90),
            "mean_peak_height_gain_m": _mean(height_values),
            "mean_final_height_gain_m": _mean(final_height_values),
            "mean_peak_tilt_deg": _mean(tilt_values),
            "max_peak_tilt_deg": max(tilt_values),
            "mean_peak_critical_contact_n": _mean(contact_values),
            "max_peak_critical_contact_n": max(contact_values),
            "stage_reach_rates": stage_reach_rates,
            "termination_counts": termination_counts,
        },
        "episodes_data": records,
    }

    output = (
        Path(args_cli.output).expanduser().resolve()
        if args_cli.output
        else Path("logs/evaluation/pct_regular_stairs")
        / checkpoint.stem
        / f"{args_cli.mode}_seed{args_cli.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[RESULT] " + json.dumps(summary["results"], ensure_ascii=False))
    print(f"[RESULT] wrote {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
