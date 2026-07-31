#!/usr/bin/env python3
"""Evaluate the exported Walk These Ways Go2 policy on the Go2-X5 flat model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "RobotLab-Isaac-Velocity-Flat-Go2-X5-WTW-PD40-v0"
LEGACY_TASK_ID = "RobotLab-Isaac-Velocity-Flat-Go2-X5-DogOnly-v0"
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_RUN_DIR = (
    WORKSPACE_ROOT
    / "walk-these-ways-go2"
    / "runs"
    / "gait-conditioned-agility"
    / "pretrain-go2"
    / "train"
    / "142238.667503"
    / "checkpoints"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=TASK_ID)
parser.add_argument("--body-path", type=str, default=os.fspath(DEFAULT_RUN_DIR / "body_latest.jit"))
parser.add_argument(
    "--adaptation-path",
    type=str,
    default=os.fspath(DEFAULT_RUN_DIR / "adaptation_module_latest.jit"),
)
parser.add_argument(
    "--manifest-path",
    type=str,
    default=None,
    help="Required provenance/ABI manifest for a fine-tuned JIT pair; omitted only for the original golden JITs.",
)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument(
    "--profile",
    choices=("walking", "nominal", "quick", "full", "planar"),
    default="walking",
)
parser.add_argument("--settle-seconds", type=float, default=2.0)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=None,
    help="Command duration; defaults to 20 s for walking and 3 s for legacy profiles.",
)
parser.add_argument(
    "--stop-seconds",
    type=float,
    default=None,
    help="Diagnostic stop duration; defaults to 2 s for walking and 1.5 s for legacy profiles.",
)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument(
    "--policy-action-warmup-steps",
    type=int,
    default=0,
    help="Optional takeover ramp. Zero matches the source WTW training/deployment action path.",
)
parser.add_argument("--leg-stiffness", type=float, default=40.0)
parser.add_argument("--leg-damping", type=float, default=1.0)
parser.add_argument("--spawn-height", type=float, default=0.30)
parser.add_argument("--max-arm-tracking-error-rad", type=float, default=0.15)
parser.add_argument("--max-gripper-tracking-error-m", type=float, default=0.005)
parser.add_argument(
    "--arm-pose",
    type=float,
    nargs=6,
    default=(0.0, 0.3, 0.5, 0.0, 0.0, 0.0),
    metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    help="Fixed absolute arm joint pose in radians; the policy never observes or controls it.",
)
parser.add_argument("--steady-fraction", type=float, default=0.50)
parser.add_argument("--gain-min", type=float, default=0.85)
parser.add_argument("--gain-max", type=float, default=1.15)
parser.add_argument("--relative-rmse-limit", type=float, default=0.15)
parser.add_argument("--linear-absolute-floor", type=float, default=0.05)
parser.add_argument("--yaw-absolute-floor", type=float, default=0.08)
parser.add_argument("--zero-linear-rmse-limit", type=float, default=0.05)
parser.add_argument("--zero-yaw-rmse-limit", type=float, default=0.08)
parser.add_argument("--max-tilt-rad", type=float, default=0.25)
parser.add_argument("--max-base-height-std", type=float, default=0.03)
parser.add_argument("--max-wz-harmonic-amplitude", type=float, default=0.15)
parser.add_argument("--max-torque-saturation-rate", type=float, default=0.01)
parser.add_argument("--max-action-clip-rate", type=float, default=0.001)
parser.add_argument("--print-every", type=int, default=50)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--disable-fabric",
    action="store_true",
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The development environment may have another Go2-X5-lab editable install.
# Resolve this checkout explicitly before registering its Gym tasks.
SOURCE_ROOT = (REPO_ROOT / "source" / "robot_lab").resolve()
sys.path.insert(0, os.fspath(SOURCE_ROOT))

import gymnasium as gym
import torch

import robot_lab

robot_lab_path = Path(robot_lab.__file__).resolve()
if SOURCE_ROOT not in robot_lab_path.parents:
    raise RuntimeError(
        f"robot_lab resolved to the wrong checkout: {robot_lab_path}; expected it below {SOURCE_ROOT}"
    )

import robot_lab.tasks  # noqa: F401, E402
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "rsl_rl"))
from flat_velocity_stability_metrics import (  # noqa: E402
    CommandSegment,
    MetricThresholds,
    analyze_samples,
    build_schedule,
    write_benchmark_artifacts,
)
from wtw_policy_adapter import (  # noqa: E402
    ACTION_CLIP,
    ACTION_DIM,
    ACTION_SCALES,
    DEFAULT_GRIPPER_JOINT_POS,
    DEFAULT_JOINT_POS,
    HISTORY_LENGTH,
    OBSERVATION_DIM,
    POLICY_DT_S,
    GRIPPER_JOINT_NAMES,
    WTW_JOINT_NAMES,
    WTWPolicyAdapter,
    make_walking_command,
)
from wtw_evaluation_metrics import (  # noqa: E402
    WALKING_COMMANDS,
    augment_summary_with_wtw_metrics,
    source_planar_deadzone,
)


ARM_JOINT_NAMES = tuple(f"arm_joint{index}" for index in range(1, 7))
FULL_MANIPULATOR_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_if_present(owner, name: str, value) -> None:
    if owner is not None and hasattr(owner, name):
        setattr(owner, name, value)


def _classify_termination(terms: list[str]) -> str:
    if any("contact" in term for term in terms):
        return "nonfoot_contact_unclassified"
    if any("orientation" in term for term in terms):
        return "fall_orientation"
    if any("height" in term for term in terms):
        return "fall_height"
    if "truncated" in terms:
        return "truncated"
    return "termination_unclassified"


def _failure_phase(segment: CommandSegment) -> str:
    return "startup_takeover" if segment.kind == "settle" else segment.kind


def _build_walking_schedule(
    *,
    settle_s: float,
    hold_s: float,
    stop_s: float,
    repeats: int,
) -> list[CommandSegment]:
    schedule = [CommandSegment("initial_settle", settle_s, kind="settle", evaluate=False)]
    for repeat in range(1, repeats + 1):
        for name, vx, vy, wz in WALKING_COMMANDS:
            schedule.append(CommandSegment(f"{name}_r{repeat}", hold_s, vx, vy, wz))
            schedule.append(CommandSegment(f"stop_after_{name}_r{repeat}", stop_s, kind="stop"))
    return schedule


def _configure_environment(
    env_cfg,
    *,
    arm_pose: tuple[float, ...],
    spawn_height: float,
    trial_duration_s: float,
) -> None:
    """Make the existing DogOnly task a deterministic WTW deployment shell."""

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 4.0
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    _set_if_present(env_cfg.scene.terrain, "max_init_terrain_level", None)
    env_cfg.episode_length_s = max(float(trial_duration_s) + 10.0, 60.0)
    _set_if_present(env_cfg, "export_io_descriptors", False)

    # Match the source deployment controller and pose contract.
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, spawn_height)
    joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
    joint_pos.update(dict(zip(WTW_JOINT_NAMES, DEFAULT_JOINT_POS, strict=True)))
    joint_pos.update(dict(zip(ARM_JOINT_NAMES, arm_pose, strict=True)))
    joint_pos.update(
        dict(zip(GRIPPER_JOINT_NAMES, DEFAULT_GRIPPER_JOINT_POS, strict=True))
    )
    env_cfg.scene.robot.init_state.joint_pos = joint_pos

    leg_action = env_cfg.actions.joint_pos
    leg_action.joint_names = list(WTW_JOINT_NAMES)
    leg_action.preserve_order = True
    leg_action.scale = {
        ".*_hip_joint": 0.125,
        ".*_thigh_joint": 0.25,
        ".*_calf_joint": 0.25,
    }
    leg_action.clip = {".*": (-10.0, 10.0)}

    external_arm_action = env_cfg.actions.arm_joint_pos
    external_arm_action.joint_names = list(ARM_JOINT_NAMES)
    external_arm_action.preserve_order = True
    external_gripper_action = getattr(env_cfg.actions, "gripper_joint_pos", None)
    if external_gripper_action is None:
        external_gripper_action = locomotion_mdp.ArmCommandPositionActionCfg(
            asset_name="robot",
            joint_names=list(GRIPPER_JOINT_NAMES),
            command_name="gripper_joint_pos",
            preserve_order=True,
        )
        env_cfg.actions.gripper_joint_pos = external_gripper_action
    else:
        external_gripper_action.joint_names = list(GRIPPER_JOINT_NAMES)
        external_gripper_action.command_name = "gripper_joint_pos"
        external_gripper_action.preserve_order = True

    for actuator_name in ("legs_hip_thigh", "legs_calf"):
        actuator = env_cfg.scene.robot.actuators[actuator_name]
        actuator.stiffness = args_cli.leg_stiffness
        actuator.damping = args_cli.leg_damping

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

    base_command = env_cfg.commands.base_velocity
    base_command.resampling_time_range = (1.0e9, 1.0e9)
    base_command.heading_command = False
    _set_if_present(base_command, "rel_standing_envs", 1.0)
    _set_if_present(base_command, "rel_heading_envs", 0.0)
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
        _set_if_present(base_command.ranges, name, (0.0, 0.0))

    arm_command = env_cfg.commands.arm_joint_pos
    arm_command.joint_names = list(ARM_JOINT_NAMES)
    arm_command.preserve_order = True
    arm_command.position_range = [(0.0, 0.0)] * len(ARM_JOINT_NAMES)
    arm_command.use_default_offset = True
    arm_command.resampling_time_range = (1.0e9, 1.0e9)
    gripper_command = getattr(env_cfg.commands, "gripper_joint_pos", None)
    if gripper_command is None:
        gripper_command = locomotion_mdp.ArmJointPositionCommandCfg(
            asset_name="robot",
            joint_names=list(GRIPPER_JOINT_NAMES),
            resampling_time_range=(1.0e9, 1.0e9),
            position_range=[(0.0, 0.0)] * len(GRIPPER_JOINT_NAMES),
            use_default_offset=True,
            clip_to_joint_limits=False,
            preserve_order=True,
        )
        env_cfg.commands.gripper_joint_pos = gripper_command
    else:
        gripper_command.joint_names = list(GRIPPER_JOINT_NAMES)
        gripper_command.preserve_order = True
        gripper_command.position_range = [(0.0, 0.0)] * len(GRIPPER_JOINT_NAMES)
        gripper_command.use_default_offset = True
        gripper_command.clip_to_joint_limits = False
        gripper_command.resampling_time_range = (1.0e9, 1.0e9)

    env_cfg.scene.contact_forces.history_length = env_cfg.decimation

    _set_if_present(env_cfg, "sim2sim_action_delay_range", (0, 0))
    _set_if_present(env_cfg, "sim2sim_action_hold_prob", 0.0)
    _set_if_present(env_cfg, "sim2sim_action_noise_std", 0.0)
    _set_if_present(env_cfg, "sim2sim_obs_delay_steps", 0)
    _set_if_present(env_cfg.sim.physx, "gpu_max_rigid_contact_count", 2**20)
    _set_if_present(env_cfg.sim.physx, "gpu_max_rigid_patch_count", 5 * 2**12)


def _live_mass_com(robot) -> dict[str, object]:
    """Read the instantiated articulation's total mass and whole-body CoM."""

    masses = robot.root_physx_view.get_masses().to(robot.device)[0]
    local_com = robot.root_physx_view.get_coms().to(robot.device)[0, :, :3]
    body_pos_w = robot.data.body_pos_w[0]
    body_quat_w = robot.data.body_quat_w[0]
    if not (len(masses) == len(local_com) == len(body_pos_w) == len(robot.body_names)):
        raise RuntimeError("PhysX mass/CoM tensors do not match the articulation body count")

    body_com_w = body_pos_w + math_utils.quat_apply(body_quat_w, local_com)
    total_mass = masses.sum()
    whole_com_w = (body_com_w * masses.unsqueeze(-1)).sum(dim=0) / total_mass
    whole_com_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w[0],
        whole_com_w - robot.data.root_pos_w[0],
    )
    return {
        "total_mass_kg": float(total_mass.item()),
        "whole_body_com_world_m": whole_com_w.tolist(),
        "whole_body_com_base_m": whole_com_b.tolist(),
        "body_masses_kg": {
            name: float(mass.item()) for name, mass in zip(robot.body_names, masses, strict=True)
        },
        "body_local_com_m": {
            name: value.tolist()
            for name, value in zip(robot.body_names, local_com, strict=True)
        },
    }


def main() -> None:
    if args_cli.task not in (TASK_ID, LEGACY_TASK_ID):
        raise ValueError(
            f"this evaluator supports only {TASK_ID} and legacy {LEGACY_TASK_ID}, got {args_cli.task}"
        )
    if args_cli.repeats <= 0:
        raise ValueError("--repeats must be positive")
    hold_seconds = args_cli.hold_seconds
    if hold_seconds is None:
        hold_seconds = 20.0 if args_cli.profile == "walking" else 3.0
    stop_seconds = args_cli.stop_seconds
    if stop_seconds is None:
        stop_seconds = 2.0 if args_cli.profile == "walking" else 1.5
    if min(args_cli.settle_seconds, hold_seconds, stop_seconds) <= 0.0:
        raise ValueError("--settle-seconds, --hold-seconds, and --stop-seconds must be positive")
    if args_cli.policy_action_warmup_steps < 0:
        raise ValueError("--policy-action-warmup-steps must be non-negative")
    if args_cli.leg_stiffness <= 0.0 or args_cli.leg_damping < 0.0:
        raise ValueError("leg stiffness must be positive and damping must be non-negative")
    if args_cli.spawn_height <= 0.18:
        raise ValueError("--spawn-height must be above the task's 0.18 m minimum root height")
    if args_cli.max_arm_tracking_error_rad <= 0.0:
        raise ValueError("--max-arm-tracking-error-rad must be positive")
    if args_cli.max_gripper_tracking_error_m <= 0.0:
        raise ValueError("--max-gripper-tracking-error-m must be positive")
    if min(
        args_cli.max_wz_harmonic_amplitude,
        args_cli.max_torque_saturation_rate,
        args_cli.max_action_clip_rate,
    ) <= 0.0:
        raise ValueError("harmonic, torque saturation, and action clip thresholds must be positive")

    body_path = Path(args_cli.body_path).expanduser().resolve()
    adaptation_path = Path(args_cli.adaptation_path).expanduser().resolve()
    manifest_path = (
        Path(args_cli.manifest_path).expanduser().resolve()
        if args_cli.manifest_path is not None
        else None
    )
    for model_path in (body_path, adaptation_path):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if args_cli.profile == "walking":
        schedule = _build_walking_schedule(
            settle_s=args_cli.settle_seconds,
            hold_s=hold_seconds,
            stop_s=stop_seconds,
            repeats=args_cli.repeats,
        )
    elif args_cli.profile == "nominal":
        schedule = [
            CommandSegment("initial_settle", args_cli.settle_seconds, kind="settle", evaluate=False)
        ]
        for repeat in range(1, args_cli.repeats + 1):
            schedule.extend(
                (
                    CommandSegment(
                        f"vx_pos_050_r{repeat}",
                        hold_seconds,
                        vx=0.50,
                    ),
                    CommandSegment(
                        f"stop_after_vx_pos_050_r{repeat}",
                        stop_seconds,
                        kind="stop",
                    ),
                )
            )
    else:
        schedule = build_schedule(
            args_cli.profile,
            settle_s=args_cli.settle_seconds,
            hold_s=hold_seconds,
            stop_s=stop_seconds,
            repeats=args_cli.repeats,
        )
    evaluated_segments = schedule[1:]
    if len(evaluated_segments) % 2 != 0:
        raise RuntimeError("velocity benchmark schedule must contain command/stop pairs")
    command_stop_pairs = list(zip(evaluated_segments[::2], evaluated_segments[1::2], strict=True))
    if any(command.kind != "command" or stop.kind != "stop" for command, stop in command_stop_pairs):
        raise RuntimeError("velocity benchmark schedule does not alternate command and stop segments")

    arm_pose = tuple(float(value) for value in args_cli.arm_pose)
    trial_duration_s = args_cli.settle_seconds + hold_seconds + stop_seconds
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_environment(
        env_cfg,
        arm_pose=arm_pose,
        spawn_height=args_cli.spawn_height,
        trial_duration_s=trial_duration_s,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = os.fspath(output_dir / "isaaclab")

    torch.manual_seed(args_cli.seed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    dt = float(raw_env.step_dt)
    physics_dt = float(raw_env.physics_dt)
    if abs(dt - POLICY_DT_S) > 1.0e-9:
        env.close()
        raise RuntimeError(f"WTW policy requires dt={POLICY_DT_S:.6f}s, environment has {dt:.6f}s")
    if env.action_space.shape[-1] != ACTION_DIM:
        env.close()
        raise RuntimeError(f"WTW policy requires {ACTION_DIM} actions, got {env.action_space.shape[-1]}")

    leg_joint_ids, leg_joint_names = robot.find_joints(list(WTW_JOINT_NAMES), preserve_order=True)
    arm_joint_ids, arm_joint_names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    gripper_joint_ids, gripper_joint_names = robot.find_joints(
        list(GRIPPER_JOINT_NAMES), preserve_order=True
    )
    if tuple(leg_joint_names) != WTW_JOINT_NAMES:
        env.close()
        raise RuntimeError(f"leg state order mismatch: {leg_joint_names}")
    if tuple(arm_joint_names) != ARM_JOINT_NAMES:
        env.close()
        raise RuntimeError(f"arm state order mismatch: {arm_joint_names}")
    if tuple(gripper_joint_names) != GRIPPER_JOINT_NAMES:
        env.close()
        raise RuntimeError(f"gripper state order mismatch: {gripper_joint_names}")

    leg_action_term = raw_env.action_manager.get_term("joint_pos")
    if tuple(leg_action_term._joint_names) != WTW_JOINT_NAMES:
        env.close()
        raise RuntimeError(f"leg action order mismatch: {leg_action_term._joint_names}")
    runtime_action_scales = torch.as_tensor(
        leg_action_term._scale,
        dtype=torch.float32,
        device=raw_env.device,
    ).reshape(-1)
    expected_action_scales = torch.tensor(
        ACTION_SCALES,
        dtype=torch.float32,
        device=raw_env.device,
    )
    if runtime_action_scales.shape != expected_action_scales.shape or not torch.allclose(
        runtime_action_scales,
        expected_action_scales,
        atol=1.0e-7,
        rtol=0.0,
    ):
        env.close()
        raise RuntimeError(
            f"leg action scale mismatch: expected {list(ACTION_SCALES)}, "
            f"got {runtime_action_scales.tolist()}"
        )

    external_arm_action_term = raw_env.action_manager.get_term("arm_joint_pos")
    external_gripper_action_term = raw_env.action_manager.get_term("gripper_joint_pos")
    for label, action_term, expected_names in (
        ("arm", external_arm_action_term, ARM_JOINT_NAMES),
        ("gripper", external_gripper_action_term, GRIPPER_JOINT_NAMES),
    ):
        if tuple(action_term._joint_names) != expected_names:
            env.close()
            raise RuntimeError(
                f"external {label} action order mismatch: {action_term._joint_names}"
            )
        if (
            action_term.action_dim != 0
            or action_term.raw_actions.shape[-1] != 0
            or action_term.processed_actions.shape[-1] != 0
        ):
            env.close()
            raise RuntimeError(
                f"external {label} action term must consume zero policy dimensions"
            )

    arm_target = torch.tensor(arm_pose, dtype=torch.float32, device=raw_env.device).unsqueeze(0)
    gripper_target = torch.tensor(
        DEFAULT_GRIPPER_JOINT_POS, dtype=torch.float32, device=raw_env.device
    ).unsqueeze(0)
    arm_limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids]
    if torch.any(arm_target[0] < arm_limits[:, 0]) or torch.any(arm_target[0] > arm_limits[:, 1]):
        env.close()
        raise ValueError(f"--arm-pose is outside the soft joint limits: {arm_pose}")
    gripper_limits = robot.data.joint_pos_limits[0, gripper_joint_ids]
    if torch.any(gripper_target[0] < gripper_limits[:, 0]) or torch.any(
        gripper_target[0] > gripper_limits[:, 1]
    ):
        env.close()
        raise ValueError(
            f"fixed gripper target is outside the hard joint limits: {DEFAULT_GRIPPER_JOINT_POS}"
        )

    default_leg_pos = robot.data.default_joint_pos[0, leg_joint_ids]
    expected_leg_pos = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=raw_env.device)
    if not torch.allclose(default_leg_pos, expected_leg_pos, atol=1.0e-7, rtol=0.0):
        env.close()
        raise RuntimeError(
            f"runtime default leg pose does not match WTW q0: {default_leg_pos.tolist()}"
        )
    default_gripper_pos = robot.data.default_joint_pos[0, gripper_joint_ids]
    if not torch.allclose(default_gripper_pos, gripper_target[0], atol=1.0e-7, rtol=0.0):
        env.close()
        raise RuntimeError(
            "runtime default gripper pose does not match fixed external target: "
            f"{default_gripper_pos.tolist()}"
        )

    adapter = WTWPolicyAdapter.from_jit_paths(
        body_path=body_path,
        adaptation_path=adaptation_path,
        manifest_path=manifest_path,
        device=raw_env.device,
    )
    if adapter.manifest is not None:
        manifest_controller = adapter.manifest["controller"]
        runtime_controller = {
            "leg_stiffness": args_cli.leg_stiffness,
            "leg_damping": args_cli.leg_damping,
            "spawn_height_m": args_cli.spawn_height,
        }
        for name, runtime_value in runtime_controller.items():
            if not math.isclose(
                runtime_value,
                float(manifest_controller[name]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                env.close()
                raise ValueError(
                    f"runtime {name}={runtime_value} does not match manifest controller "
                    f"value {manifest_controller[name]}"
                )
        if tuple(float(value) for value in manifest_controller["gripper_target_m"]) != tuple(
            DEFAULT_GRIPPER_JOINT_POS
        ):
            env.close()
            raise ValueError(
                "runtime gripper target does not match manifest controller gripper_target_m"
            )
    base_command_term = raw_env.command_manager.get_term("base_velocity")
    arm_command_term = raw_env.command_manager.get_term("arm_joint_pos")
    gripper_command_term = raw_env.command_manager.get_term("gripper_joint_pos")
    for label, command_term, expected_names in (
        ("arm", arm_command_term, ARM_JOINT_NAMES),
        ("gripper", gripper_command_term, GRIPPER_JOINT_NAMES),
    ):
        command_joint_names = tuple(
            robot.joint_names[index] for index in command_term.joint_ids
        )
        if command_joint_names != expected_names:
            env.close()
            raise RuntimeError(
                f"external {label} command order mismatch: {command_joint_names}"
            )
        if command_term.command_buffer.shape[-1] != len(expected_names):
            env.close()
            raise RuntimeError(
                f"external {label} command has an unexpected target dimension"
            )
    termination_manager = raw_env.termination_manager
    termination_names = tuple(termination_manager.active_terms)
    contact_sensor = raw_env.scene.sensors["contact_forces"]
    if contact_sensor.cfg.history_length != raw_env.cfg.decimation:
        env.close()
        raise RuntimeError(
            "contact force history must cover one full control decimation: "
            f"history={contact_sensor.cfg.history_length}, decimation={raw_env.cfg.decimation}"
        )
    contact_body_names = tuple(contact_sensor.body_names)
    foot_sensor_ids = [
        index for index, name in enumerate(contact_body_names) if name.lower().endswith("foot")
    ]
    if len(foot_sensor_ids) != 4:
        env.close()
        raise RuntimeError(
            f"expected four foot contact sensors, got {[contact_body_names[index] for index in foot_sensor_ids]}"
        )
    foot_names = [contact_body_names[index] for index in foot_sensor_ids]
    foot_body_ids, resolved_foot_names = robot.find_bodies(foot_names, preserve_order=True)
    if tuple(resolved_foot_names) != tuple(foot_names):
        env.close()
        raise RuntimeError(f"foot state order mismatch: expected {foot_names}, got {resolved_foot_names}")
    nonfoot_sensor_ids = [
        index for index, name in enumerate(contact_body_names) if name not in foot_names
    ]

    env.reset(seed=args_cli.seed)
    if contact_sensor.data.net_forces_w_history.shape[1] != raw_env.cfg.decimation:
        env.close()
        raise RuntimeError("contact force tensor history does not match configured decimation")
    mass_com = _live_mass_com(robot)
    zero_policy_command = make_walking_command(0.0, batch_size=1, device=raw_env.device)
    action_scales = torch.tensor(ACTION_SCALES, dtype=torch.float32, device=raw_env.device)
    q0 = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=raw_env.device)
    leg_effort_limits = robot.data.joint_effort_limits[0, leg_joint_ids]

    samples: list[dict[str, object]] = []
    termination_events: list[dict[str, object]] = []
    benchmark_time_s = 0.0
    global_step = 0
    samples_path = output_dir / "samples.jsonl"

    def set_external_commands(segment: CommandSegment) -> None:
        base_command_term.vel_command_b[:, 0] = segment.vx
        base_command_term.vel_command_b[:, 1] = segment.vy
        base_command_term.vel_command_b[:, 2] = segment.wz
        if hasattr(base_command_term, "is_heading_env"):
            base_command_term.is_heading_env[:] = False
        if hasattr(base_command_term, "is_standing_env"):
            base_command_term.is_standing_env[:] = (
                abs(segment.vx) + abs(segment.vy) + abs(segment.wz) <= 1.0e-6
            )
        arm_command_term.command_buffer[:] = arm_target
        gripper_command_term.command_buffer[:] = gripper_target

    def run_segment(
        segment: CommandSegment,
        *,
        segment_index: int,
        trial_index: int,
        trial_step: int,
        policy_command: torch.Tensor,
        next_policy_command: torch.Tensor,
        stream,
    ) -> tuple[bool, int]:
        nonlocal benchmark_time_s, global_step

        steps = max(1, round(segment.duration_s / dt))
        print(
            f"[wtw-go2x5] trial={trial_index} segment={segment.name} steps={steps} "
            f"cmd=({segment.vx:.3f}, {segment.vy:.3f}, {segment.wz:.3f})"
        )
        for segment_step in range(steps):
            wall_start = time.perf_counter()
            set_external_commands(segment)
            raw_action = adapter.infer_raw()
            clipped_action = raw_action.clamp(-ACTION_CLIP, ACTION_CLIP)
            if args_cli.policy_action_warmup_steps > 0:
                action_scale = min(
                    1.0,
                    float(trial_step + 1) / float(args_cli.policy_action_warmup_steps),
                )
            else:
                action_scale = 1.0
            applied_action = clipped_action * action_scale
            # Do not wrap simulator stepping in torch.inference_mode(): Isaac Lab
            # mutates its state buffers during later explicit resets.
            _, _, terminated, truncated, _ = env.step(applied_action)

            done = bool((terminated[0] | truncated[0]).item())
            fired_terminations = [
                name
                for name in termination_names
                if bool(termination_manager.get_term(name)[0].item())
            ]
            if bool(truncated[0].item()) and not fired_terminations:
                fired_terminations.append("truncated")

            roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
            roll = math_utils.wrap_to_pi(roll)
            pitch = math_utils.wrap_to_pi(pitch)
            yaw = math_utils.wrap_to_pi(yaw)
            base_pos = robot.data.root_pos_w[0]
            leg_pos = robot.data.joint_pos[:, leg_joint_ids]
            leg_vel = robot.data.joint_vel[:, leg_joint_ids]
            measured_arm = robot.data.joint_pos[:, arm_joint_ids]
            measured_gripper = robot.data.joint_pos[:, gripper_joint_ids]
            arm_error = measured_arm - arm_target
            gripper_error = measured_gripper - gripper_target
            leg_target = q0.unsqueeze(0) + action_scales.unsqueeze(0) * applied_action
            computed_torque = robot.data.computed_torque[:, leg_joint_ids]
            applied_torque = robot.data.applied_torque[:, leg_joint_ids]
            torque_saturated = torch.abs(computed_torque - applied_torque) > 1.0e-5
            contact_history = contact_sensor.data.net_forces_w_history[0]
            contact_force_history_n = torch.linalg.norm(contact_history, dim=-1)
            foot_force_n = contact_force_history_n[:, foot_sensor_ids].amax(dim=0)
            foot_slip_mps = torch.linalg.norm(
                robot.data.body_lin_vel_w[0, foot_body_ids, :2], dim=-1
            )
            foot_slip_mps = torch.where(foot_force_n > 1.0, foot_slip_mps, 0.0)
            foot_impulse_n_s = contact_force_history_n[:, foot_sensor_ids].sum(dim=0) * float(
                physics_dt
            )
            nonfoot_peak_force_n = contact_force_history_n[:, nonfoot_sensor_ids].amax(dim=0)
            nonfoot_contact_bodies = [
                contact_body_names[sensor_id]
                for sensor_id, force in zip(nonfoot_sensor_ids, nonfoot_peak_force_n, strict=True)
                if float(force.item()) > 1.0
            ]
            deadzone_vx, deadzone_vy = source_planar_deadzone(segment.vx, segment.vy)
            row = {
                "time_s": benchmark_time_s,
                "global_step": global_step,
                "trial_index": trial_index,
                "trial_step": trial_step,
                "segment_index": segment_index,
                "segment_name": segment.name,
                "segment_kind": segment.kind,
                "segment_time_s": (segment_step + 1) * dt,
                "evaluate": segment.evaluate,
                "cmd_vx": segment.vx,
                "cmd_vy": segment.vy,
                "cmd_wz": segment.wz,
                "raw_command": [segment.vx, segment.vy, segment.wz],
                "source_deadzone_command": [deadzone_vx, deadzone_vy, segment.wz],
                "wtw_command": policy_command[0].tolist(),
                "measured_vx": float(robot.data.root_lin_vel_b[0, 0].item()),
                "measured_vy": float(robot.data.root_lin_vel_b[0, 1].item()),
                "measured_wz": float(robot.data.root_ang_vel_b[0, 2].item()),
                "base_x": float(base_pos[0].item()),
                "base_y": float(base_pos[1].item()),
                "base_z": float(base_pos[2].item()),
                "base_roll": float(roll[0].item()),
                "base_pitch": float(pitch[0].item()),
                "base_yaw": float(yaw[0].item()),
                "leg_joint_pos": leg_pos[0].tolist(),
                "leg_joint_vel": leg_vel[0].tolist(),
                "leg_position_target": leg_target[0].tolist(),
                "raw_policy_action": raw_action[0].tolist(),
                "clipped_policy_action": clipped_action[0].tolist(),
                "applied_policy_action": applied_action[0].tolist(),
                "raw_action_clipped": (raw_action[0].abs() > ACTION_CLIP).tolist(),
                "action_abs_mean": float(applied_action[0].abs().mean().item()),
                "action_abs_max": float(applied_action[0].abs().max().item()),
                "policy_action_scale": action_scale,
                "leg_computed_torque_nm": computed_torque[0].tolist(),
                "leg_applied_torque_nm": applied_torque[0].tolist(),
                "leg_effort_limit_nm": leg_effort_limits.tolist(),
                "leg_torque_saturated": torque_saturated[0].tolist(),
                "foot_contact_force_n": dict(zip(foot_names, foot_force_n.tolist(), strict=True)),
                "foot_contact_slip_mps": dict(zip(foot_names, foot_slip_mps.tolist(), strict=True)),
                "foot_contact_impulse_n_s": dict(
                    zip(foot_names, foot_impulse_n_s.tolist(), strict=True)
                ),
                "nonfoot_contact_bodies": nonfoot_contact_bodies,
                "nonfoot_contact_peak_force_n": dict(
                    zip(
                        (contact_body_names[index] for index in nonfoot_sensor_ids),
                        nonfoot_peak_force_n.tolist(),
                        strict=True,
                    )
                ),
                "arm_command": arm_target[0].tolist(),
                "arm_joint_pos": measured_arm[0].tolist(),
                "arm_joint_error": arm_error[0].tolist(),
                "arm_tracking_rmse_rad": float(torch.sqrt(torch.mean(arm_error[0].square())).item()),
                "arm_tracking_max_abs_rad": float(arm_error[0].abs().max().item()),
                "gripper_command_m": gripper_target[0].tolist(),
                "gripper_joint_pos_m": measured_gripper[0].tolist(),
                "gripper_joint_error_m": gripper_error[0].tolist(),
                "gripper_tracking_rmse_m": float(
                    torch.sqrt(torch.mean(gripper_error[0].square())).item()
                ),
                "gripper_tracking_max_abs_m": float(gripper_error[0].abs().max().item()),
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
                "done": done,
                "state_is_post_auto_reset": done,
                "termination_terms": fired_terminations,
            }
            samples.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

            if global_step % max(1, args_cli.print_every) == 0:
                print(
                    f"[tracking] t={benchmark_time_s:.2f}s "
                    f"cmd=({segment.vx:.2f},{segment.vy:.2f},{segment.wz:.2f}) "
                    f"meas=({row['measured_vx']:.2f},{row['measured_vy']:.2f},"
                    f"{row['measured_wz']:.2f}) arm_rmse={row['arm_tracking_rmse_rad']:.3f}"
                )

            benchmark_time_s += dt
            global_step += 1
            trial_step += 1
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.perf_counter() - wall_start)))
            if done:
                event = {
                    "trial_index": trial_index,
                    "segment_name": segment.name,
                    "segment_kind": segment.kind,
                    "segment_time_s": row["segment_time_s"],
                    "benchmark_time_s": benchmark_time_s,
                    "terms": fired_terminations,
                    "failure_phase": _failure_phase(segment),
                    "failure_class": _classify_termination(fired_terminations),
                }
                termination_events.append(event)
                print(f"[wtw-go2x5] terminated during {segment.name}: {fired_terminations}")
                return True, trial_step

            observation_command = (
                next_policy_command if segment_step == steps - 1 else policy_command
            )
            adapter.advance(
                projected_gravity=robot.data.projected_gravity_b,
                command=observation_command,
                joint_pos=leg_pos,
                joint_vel=leg_vel,
                applied_action=applied_action,
            )
        return False, trial_step

    try:
        with samples_path.open("w", encoding="utf-8", buffering=1) as stream:
            for trial_index, (command_segment, stop_segment) in enumerate(
                command_stop_pairs, start=1
            ):
                env.reset(seed=args_cli.seed)
                adapter.reset(num_envs=1)
                trial_step = 0
                command_policy_command = make_walking_command(
                    command_segment.vx,
                    command_segment.vy,
                    command_segment.wz,
                    batch_size=1,
                    device=raw_env.device,
                )
                settle_segment = CommandSegment(
                    name=f"settle_before_{command_segment.name}",
                    duration_s=args_cli.settle_seconds,
                    vx=command_segment.vx,
                    vy=command_segment.vy,
                    wz=command_segment.wz,
                    kind="settle",
                    evaluate=False,
                )
                fell, trial_step = run_segment(
                    settle_segment,
                    segment_index=-trial_index,
                    trial_index=trial_index,
                    trial_step=trial_step,
                    policy_command=command_policy_command,
                    next_policy_command=command_policy_command,
                    stream=stream,
                )
                if fell:
                    continue
                fell, trial_step = run_segment(
                    command_segment,
                    segment_index=schedule.index(command_segment),
                    trial_index=trial_index,
                    trial_step=trial_step,
                    policy_command=command_policy_command,
                    next_policy_command=zero_policy_command,
                    stream=stream,
                )
                if fell:
                    continue
                run_segment(
                    stop_segment,
                    segment_index=schedule.index(stop_segment),
                    trial_index=trial_index,
                    trial_step=trial_step,
                    policy_command=zero_policy_command,
                    next_policy_command=zero_policy_command,
                    stream=stream,
                )
    finally:
        env.close()

    expected_evaluated_segments = sum(segment.evaluate for segment in schedule)
    summary = analyze_samples(
        samples,
        thresholds=thresholds,
        steady_fraction=args_cli.steady_fraction,
        expected_evaluated_segments=expected_evaluated_segments,
    )
    summary = augment_summary_with_wtw_metrics(
        samples,
        summary,
        sample_dt_s=dt,
        steady_fraction=args_cli.steady_fraction,
        max_wz_harmonic_amplitude=args_cli.max_wz_harmonic_amplitude,
        max_torque_saturation_rate=args_cli.max_torque_saturation_rate,
        max_action_clip_rate=args_cli.max_action_clip_rate,
        expected_command_segments=sum(
            segment.evaluate and segment.kind == "command" for segment in schedule
        ),
    )
    valid_arm_rows = [row for row in samples if not row["state_is_post_auto_reset"]]
    arm_step_rmse = [float(row["arm_tracking_rmse_rad"]) for row in valid_arm_rows]
    arm_max_abs = [float(row["arm_tracking_max_abs_rad"]) for row in valid_arm_rows]
    if arm_step_rmse:
        arm_p95_index = max(0, round(0.95 * (len(arm_step_rmse) - 1)))
        arm_joint_rmse = {
            name: math.sqrt(
                sum(float(row["arm_joint_error"][index]) ** 2 for row in valid_arm_rows)
                / len(valid_arm_rows)
            )
            for index, name in enumerate(ARM_JOINT_NAMES)
        }
        arm_joint_max_abs = {
            name: max(abs(float(row["arm_joint_error"][index])) for row in valid_arm_rows)
            for index, name in enumerate(ARM_JOINT_NAMES)
        }
        arm_fixture = {
            "valid_sample_count": len(valid_arm_rows),
            "step_rmse_mean_rad": sum(arm_step_rmse) / len(arm_step_rmse),
            "step_rmse_p95_rad": sorted(arm_step_rmse)[arm_p95_index],
            "joint_max_abs_rad": max(arm_max_abs),
            "per_joint_rmse_rad": arm_joint_rmse,
            "per_joint_max_abs_rad": arm_joint_max_abs,
            "joint_max_abs_limit_rad": args_cli.max_arm_tracking_error_rad,
            "status": (
                "passed"
                if max(arm_max_abs) <= args_cli.max_arm_tracking_error_rad
                else "tracking_error_exceeded"
            ),
            "passed": max(arm_max_abs) <= args_cli.max_arm_tracking_error_rad,
        }
    else:
        arm_fixture = {
            "valid_sample_count": 0,
            "step_rmse_mean_rad": None,
            "step_rmse_p95_rad": None,
            "joint_max_abs_rad": None,
            "joint_max_abs_limit_rad": args_cli.max_arm_tracking_error_rad,
            "status": "insufficient_pre_reset_samples",
            "passed": False,
        }
    gripper_step_rmse = [float(row["gripper_tracking_rmse_m"]) for row in valid_arm_rows]
    gripper_max_abs = [float(row["gripper_tracking_max_abs_m"]) for row in valid_arm_rows]
    if gripper_step_rmse:
        gripper_p95_index = max(0, round(0.95 * (len(gripper_step_rmse) - 1)))
        gripper_joint_rmse = {
            name: math.sqrt(
                sum(float(row["gripper_joint_error_m"][index]) ** 2 for row in valid_arm_rows)
                / len(valid_arm_rows)
            )
            for index, name in enumerate(GRIPPER_JOINT_NAMES)
        }
        gripper_joint_max_abs = {
            name: max(
                abs(float(row["gripper_joint_error_m"][index])) for row in valid_arm_rows
            )
            for index, name in enumerate(GRIPPER_JOINT_NAMES)
        }
        gripper_fixture = {
            "valid_sample_count": len(valid_arm_rows),
            "target_m": list(DEFAULT_GRIPPER_JOINT_POS),
            "step_rmse_mean_m": sum(gripper_step_rmse) / len(gripper_step_rmse),
            "step_rmse_p95_m": sorted(gripper_step_rmse)[gripper_p95_index],
            "joint_max_abs_m": max(gripper_max_abs),
            "per_joint_rmse_m": gripper_joint_rmse,
            "per_joint_max_abs_m": gripper_joint_max_abs,
            "joint_max_abs_limit_m": args_cli.max_gripper_tracking_error_m,
            "status": (
                "passed"
                if max(gripper_max_abs) <= args_cli.max_gripper_tracking_error_m
                else "tracking_error_exceeded"
            ),
            "passed": max(gripper_max_abs) <= args_cli.max_gripper_tracking_error_m,
        }
    else:
        gripper_fixture = {
            "valid_sample_count": 0,
            "target_m": list(DEFAULT_GRIPPER_JOINT_POS),
            "step_rmse_mean_m": None,
            "step_rmse_p95_m": None,
            "joint_max_abs_m": None,
            "joint_max_abs_limit_m": args_cli.max_gripper_tracking_error_m,
            "status": "insufficient_pre_reset_samples",
            "passed": False,
        }
    walking_metrics_passed = bool(summary["passed"])
    summary["arm_fixture"] = arm_fixture
    summary["gripper_fixture"] = gripper_fixture
    summary["passed"] = bool(
        walking_metrics_passed and arm_fixture["passed"] and gripper_fixture["passed"]
    )
    summary["overall_failure_reasons"] = []
    if not walking_metrics_passed:
        summary["overall_failure_reasons"].append("walking_only_metrics")
    if not arm_fixture["passed"]:
        summary["overall_failure_reasons"].append(f"arm_fixture_{arm_fixture['status']}")
    if not gripper_fixture["passed"]:
        summary["overall_failure_reasons"].append(
            f"gripper_fixture_{gripper_fixture['status']}"
        )
    metadata = {
        "checkpoint": os.fspath(body_path),
        "body_path": os.fspath(body_path),
        "body_sha256": _sha256(body_path),
        "adaptation_path": os.fspath(adaptation_path),
        "adaptation_sha256": _sha256(adaptation_path),
        "known_checkpoint_contract_verified": True,
        "checkpoint_verification_mode": (
            "continuation_manifest" if manifest_path is not None else "original_golden_hash_and_zero_history"
        ),
        "manifest_path": os.fspath(manifest_path) if manifest_path is not None else None,
        "manifest": adapter.manifest,
        "task": args_cli.task,
        "profile": args_cli.profile,
        "repeats": args_cli.repeats,
        "seed": args_cli.seed,
        "terrain": "deterministic_plane",
        "control_dt_s": dt,
        "policy_action_warmup_steps": args_cli.policy_action_warmup_steps,
        "policy_observation_dim": OBSERVATION_DIM,
        "policy_history_length": HISTORY_LENGTH,
        "policy_action_dim": ACTION_DIM,
        "policy_joint_order": list(WTW_JOINT_NAMES),
        "policy_default_joint_pos": list(DEFAULT_JOINT_POS),
        "policy_action_scales": list(ACTION_SCALES),
        "fixed_arm_pose_rad": list(arm_pose),
        "fixed_gripper_pose_m": list(DEFAULT_GRIPPER_JOINT_POS),
        "manipulator_joint_order": list(FULL_MANIPULATOR_JOINT_NAMES),
        "manipulator_control": "external_absolute_position_command; zero policy dimensions",
        "external_action_dims": {
            "arm_joint_pos": external_arm_action_term.action_dim,
            "gripper_joint_pos": external_gripper_action_term.action_dim,
        },
        "leg_pd": {
            "stiffness": args_cli.leg_stiffness,
            "damping": args_cli.leg_damping,
        },
        "spawn_height_m": args_cli.spawn_height,
        "robot_lab_module": os.fspath(robot_lab_path),
        "robot": mass_com,
        "independent_reset_per_command": True,
        "continued_after_termination": True,
        "walking_only_acceptance": args_cli.profile == "walking",
        "stop_segments_are_diagnostic_only": True,
        "source_planar_deadzone_is_reported_not_applied": True,
        "gait_cycle_metrics": {
            "frequency_hz": 2.5,
            "integer_cycle_window": True,
            "max_wz_harmonic_amplitude": args_cli.max_wz_harmonic_amplitude,
            "max_torque_saturation_rate": args_cli.max_torque_saturation_rate,
            "max_action_clip_rate": args_cli.max_action_clip_rate,
        },
        "contact_logging": {
            "sensor_body_order": list(contact_body_names),
            "foot_body_order": foot_names,
            "contact_threshold_n": 1.0,
            "history_length": contact_sensor.cfg.history_length,
            "history_window_s": contact_sensor.cfg.history_length * physics_dt,
            "history_covers_one_policy_step": bool(
                contact_sensor.cfg.history_length == raw_env.cfg.decimation
            ),
            "impulse_estimator": (
                "sum(one-decimation contact-force history magnitudes) * physics_dt"
            ),
            "physics_dt_s": physics_dt,
        },
        "done_row_state_note": "Isaac Lab auto-resets before state sampling; done rows are post-reset snapshots.",
        "terminated_early": bool(termination_events),
        "termination_event": termination_events[0] if termination_events else None,
        "termination_events": termination_events,
        "schedule": [segment.to_dict() for segment in schedule],
        "execution_protocol": {
            "settle_s_per_command": args_cli.settle_seconds,
            "hold_s": hold_seconds,
            "stop_s": stop_seconds,
        },
    }
    write_benchmark_artifacts(output_dir, samples, summary, metadata)
    if arm_fixture["valid_sample_count"]:
        arm_result_text = (
            f"- 固定臂有效样本数：`{arm_fixture['valid_sample_count']}`；step RMSE mean/p95："
            f"`{arm_fixture['step_rmse_mean_rad']:.4f}` / "
            f"`{arm_fixture['step_rmse_p95_rad']:.4f} rad`；最大单关节误差："
            f"`{arm_fixture['joint_max_abs_rad']:.4f} rad`；fixture pass："
            f"**{arm_fixture['passed']}**。\n"
        )
    else:
        arm_result_text = (
            "- 固定臂没有 termination auto-reset 前的有效样本；"
            "fixture status：`insufficient_pre_reset_samples`；fixture pass：**False**。\n"
        )
    if gripper_fixture["valid_sample_count"]:
        gripper_result_text = (
            f"- 固定夹爪 target：`{list(DEFAULT_GRIPPER_JOINT_POS)} m`；step RMSE mean/p95："
            f"`{gripper_fixture['step_rmse_mean_m']:.5f}` / "
            f"`{gripper_fixture['step_rmse_p95_m']:.5f} m`；最大单关节误差："
            f"`{gripper_fixture['joint_max_abs_m']:.5f} m`；fixture pass："
            f"**{gripper_fixture['passed']}**。\n"
        )
    else:
        gripper_result_text = (
            "- 固定夹爪没有 termination auto-reset 前的有效样本；"
            "fixture status：`insufficient_pre_reset_samples`；fixture pass：**False**。\n"
        )
    with (output_dir / "report.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## WTW / 固定机械臂补充\n\n"
            f"- WTW spawn height：`{args_cli.spawn_height:.3f} m`；"
            f"腿部 PD：`{args_cli.leg_stiffness:g}/{args_cli.leg_damping:g}`；"
            f"action warmup：`{args_cli.policy_action_warmup_steps}` steps。\n"
            f"{arm_result_text}"
            f"{gripper_result_text}"
            "- 正式 walking pass 只统计 command segment；stop/zero-command 单独保留为诊断，"
            "不影响 walking-only pass rate。\n"
            "- 每步样本记录 computed/applied torque、逐关节饱和、逐足 force/slip/impulse、"
            "非足接触 body，以及逐关节机械臂 target/actual/error。\n"
            "- done 行的机器人状态是 Isaac Lab 自动复位后的快照；"
            "终止前趋势应读取前一行。\n"
        )
    walking_result = summary["walking_only"]
    print(
        f"[wtw-go2x5] report={output_dir / 'report.md'} "
        f"walking={walking_result['passed_command_segments']}/"
        f"{walking_result['expected_command_segments']} "
        f"stops_diagnostic={summary['passed_stop_segments']}/{summary['stop_segments']} "
        f"overall_pass={summary['passed']}"
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
