#!/usr/bin/env python3
"""Import a trusted RoboDuet 019999 run into the ignored WholeBody model store."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pickle
import shutil
import sys
import types

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "robot_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if "robot_lab" not in sys.modules:
    robot_lab = types.ModuleType("robot_lab")
    robot_lab.__path__ = [str(SOURCE_ROOT / "robot_lab")]
    sys.modules["robot_lab"] = robot_lab

from robot_lab.go2_x5_wholebody.manifest import load_manifest, sha256  # noqa: E402
from robot_lab.go2_x5_wholebody.models import ArmActorCritic, DogActorCritic  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "models" / "go2_x5_wholebody" / "019999"
DEFAULT_MANIFEST = SOURCE_ROOT / "data" / "Policies" / "go2_x5_wholebody" / "019999.yaml"
URDF = SOURCE_ROOT / "data" / "Robots" / "go2_x5" / "go2_x5.urdf"
URDF_SHA256 = "8947e24c4c7e3c8074fe7b727c68d0addbc4ca376bce2bf05261573681ce7807"


def _checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TypeError(f"Expected raw tensor state_dict: {path}")
    return state


def _cfg_value(cfg: dict, *keys):
    value = cfg
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"parameters.pkl is missing Cfg.{'.'.join(keys)}")
        value = value[key]
    return value


def _validate_parameters(parameters: dict) -> None:
    cfg = parameters.get("Cfg")
    if not isinstance(cfg, dict):
        raise TypeError("parameters.pkl must contain a mapping named 'Cfg'")
    expected = {
        ("dog", "dog_num_observations"): 56,
        ("dog", "dog_num_privileged_obs"): 2,
        ("dog", "dog_num_observation_history"): 30,
        ("dog", "dog_num_obs_history"): 1680,
        ("dog", "dog_actions"): 12,
        ("arm", "arm_num_observations"): 20,
        ("arm", "arm_num_privileged_obs"): 9,
        ("arm", "arm_num_observation_history"): 30,
        ("arm", "arm_num_obs_history"): 600,
        ("arm", "num_actions_arm_cd"): 8,
        ("env", "observe_vel"): False,
        ("env", "observe_clock_inputs"): True,
        ("env", "observe_two_prev_actions"): False,
        ("use_rot6d",): False,
        ("normalization", "clip_observations"): 100.0,
        ("normalization", "clip_actions"): 10.0,
    }
    for keys, expected_value in expected.items():
        actual = _cfg_value(cfg, *keys)
        if actual != expected_value:
            raise ValueError(f"Cfg.{'.'.join(keys)} expected {expected_value!r}, got {actual!r}")


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"Refusing to overwrite different artifact: {destination}")
        return
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise IOError(f"Copied artifact checksum mismatch: {destination}")


def _manifest(dog: Path, arm: Path, parameters: Path, source_run_id: str) -> dict:
    return {
        "schema_version": 1,
        "policy": {"name": "Go2-x5-wholebody", "checkpoint_iteration": 19999},
        "source": {
            "run_id": source_run_id,
            "network_commit": "fe16a0666648de22ec1e2c57f94ec759b8587553",
            "ik_commit": "2e86749d86ee4150073197fab7d2e5d56f8c07e7",
            "runtime_profile_commit": "f22cea50351fb15c2c2f84e4ca9a906a0a4e1c5c",
            "parameters_sha256": sha256(parameters),
            "trusted_pickle_import_only": True,
        },
        "network": {
            "dog": {
                "observation_dim": 56,
                "history_length": 30,
                "history_dim": 1680,
                "privileged_dim": 2,
                "actor_input_dim": 1682,
                "action_dim": 12,
            },
            "arm": {
                "observation_dim": 20,
                "history_length": 30,
                "history_dim": 600,
                "privileged_dim": 9,
                "old_history_dim": 580,
                "old_history_latent_dim": 128,
                "actor_input_dim": 157,
                "action_dim": 8,
            },
            "observe_vel": False,
            "observe_clock_inputs": True,
            "observe_two_prev_actions": False,
            "use_rot6d": False,
            "inference_arm_body_plan_tanh": False,
        },
        "runtime": {
            "profile": "go2_x5_joint40k_pd40_v1",
            "control_dt": 0.02,
            "sim_dt": 0.005,
            "decimation": 4,
            "num_envs": 1,
            "action_dim": 10,
            "observation_clip": 100.0,
            "action_clip": 10.0,
            "urdf_sha256": URDF_SHA256,
            "physics": {
                "solver_type": "TGS",
                "solver_position_iterations": 8,
                "solver_velocity_iterations": 1,
                "contact_offset_m": 0.01,
                "rest_offset_m": 0.0,
                "bounce_threshold_velocity_m_s": 0.5,
                "max_depenetration_velocity_m_s": 1.0,
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
            },
            "asset": {
                "total_mass_kg": 20.076596,
                "mass_tolerance_kg": 0.01,
                "whole_body_com_base_m": [0.02587665, -0.000253943, 0.022855602],
                "com_tolerance_m": 0.005,
            },
            "gait": {
                "frequency_hz": 3.0,
                "phase": 0.5,
                "offset": 0.0,
                "bound": 0.0,
                "duty": 0.5,
                "stand_threshold": 0.1,
            },
            "action_scales": {
                "hip": 0.125,
                "thigh": 0.25,
                "calf": 0.25,
                "arm": 0.5,
            },
            "pd": {
                "legs": {"stiffness": 40.0, "damping": 1.0},
                "arm_joint1": {"stiffness": 40.0, "damping": 3.0},
                "arm_joint2_3": {"stiffness": 70.0, "damping": 15.0},
                "arm_joint4_6": {"stiffness": 25.0, "damping": 2.0},
                "gripper": {"stiffness": 50.0, "damping": 20.0, "friction": 0.05},
            },
            "ik_position_drive": {
                "profile": "roboduet_go2_x5_ik_v1",
                "stiffness": [10000.0, 10000.0, 10000.0, 4000.0, 2000.0, 1000.0],
                "damping": [140.0, 140.0, 140.0, 70.0, 42.0, 28.0],
                "effort": [20.0, 20.0, 15.0, 7.0, 5.0, 5.0],
                "velocity": 3.0,
            },
            "tcp": {
                "public_frame": "arm_eef_link",
                "legacy_training_point_m": 0.1,
                "urdf_eef_offset_m": 0.08657,
                "ground_z_offset_m": 0.38,
            },
            "joint_names": {
                "dog": [
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                ],
                "arm": [f"arm_joint{index}" for index in range(1, 7)],
                "gripper": ["arm_joint7", "arm_joint8"],
            },
            "joint_limits": {
                "lower": [
                    -1.0472, -1.5708, -2.7227,
                    -1.0472, -1.5708, -2.7227,
                    -1.0472, -0.5236, -2.7227,
                    -1.0472, -0.5236, -2.7227,
                    -3.14159265359, 0.0, 0.0,
                    -1.57079632679, -1.57079632679, -1.57079632679,
                    0.0, 0.0,
                ],
                "upper": [
                    1.0472, 3.4907, -0.83776,
                    1.0472, 3.4907, -0.83776,
                    1.0472, 4.5379, -0.83776,
                    1.0472, 4.5379, -0.83776,
                    3.14159265359, 3.66519, 3.14159265359,
                    1.57079632679, 1.57079632679, 1.57079632679,
                    0.044, 0.044,
                ],
                "effort": [23.7, 23.7, 45.43] * 4 + [30.0] * 6 + [20.0] * 2,
                "velocity": [30.1, 30.1, 15.7] * 4 + [10.0] * 6 + [1.0] * 2,
            },
        },
        "artifacts": {
            "dog": {
                "relative_path": "checkpoints_dog/ac_weights_019999.pt",
                "sha256": sha256(dog),
                "size_bytes": dog.stat().st_size,
            },
            "arm": {
                "relative_path": "checkpoints_arm/ac_weights_019999.pt",
                "sha256": sha256(arm),
                "size_bytes": arm.stat().st_size,
            },
        },
        "dependencies": {
            "pin": "2.7.0",
            "pin-pink": "3.1.0",
            "qpsolvers": "4.8.2",
            "quadprog": ">=0.1.12,<0.2",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a trusted RoboDuet run; runtime never reads parameters.pkl."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-run-id",
        help="Canonical source run ID when run_dir is only a local staging directory.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    dog_source = run_dir / "checkpoints_dog" / "ac_weights_019999.pt"
    arm_source = run_dir / "checkpoints_arm" / "ac_weights_019999.pt"
    parameters_source = run_dir / "parameters.pkl"
    for path in (dog_source, arm_source, parameters_source, URDF):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(URDF) != URDF_SHA256:
        raise ValueError(f"Go2-X5 URDF SHA256 mismatch: {sha256(URDF)}")

    # The source run is explicitly trusted by the caller. Pickle is confined to
    # this one-time importer and never enters the runtime package.
    with parameters_source.open("rb") as stream:
        parameters = pickle.load(stream)
    _validate_parameters(parameters)
    DogActorCritic().load_state_dict(_checkpoint(dog_source), strict=True)
    ArmActorCritic().load_state_dict(_checkpoint(arm_source), strict=True)

    dog_destination = args.output_root / "checkpoints_dog" / dog_source.name
    arm_destination = args.output_root / "checkpoints_arm" / arm_source.name
    _copy_verified(dog_source, dog_destination)
    _copy_verified(arm_source, arm_destination)
    source_run_id = args.source_run_id or run_dir.name
    manifest = _manifest(dog_destination, arm_destination, parameters_source, source_run_id)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    load_manifest(args.manifest_out)
    print(f"Imported Dog: {dog_destination} ({sha256(dog_destination)})")
    print(f"Imported Arm: {arm_destination} ({sha256(arm_destination)})")
    print(f"Wrote manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
