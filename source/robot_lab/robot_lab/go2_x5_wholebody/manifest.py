"""Safe manifest loading and immutable artifact verification for WholeBody policy bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
URDF_SHA256 = "8947e24c4c7e3c8074fe7b727c68d0addbc4ca376bce2bf05261573681ce7807"


@dataclass(frozen=True)
class ArtifactPaths:
    dog_checkpoint: Path
    arm_checkpoint: Path
    urdf: Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"WholeBody manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise TypeError(f"WholeBody manifest must be a mapping: {manifest_path}")
    _validate_contract(manifest)
    return manifest


def _value(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"WholeBody manifest is missing '{dotted_key}'")
        value = value[key]
    return value


def _validate_contract(manifest: dict[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "policy.name": "Go2-x5-wholebody",
        "policy.checkpoint_iteration": 19999,
        "network.dog.observation_dim": 56,
        "network.dog.history_length": 30,
        "network.dog.history_dim": 1680,
        "network.dog.privileged_dim": 2,
        "network.dog.actor_input_dim": 1682,
        "network.dog.action_dim": 12,
        "network.arm.observation_dim": 20,
        "network.arm.history_length": 30,
        "network.arm.history_dim": 600,
        "network.arm.privileged_dim": 9,
        "network.arm.old_history_dim": 580,
        "network.arm.old_history_latent_dim": 128,
        "network.arm.actor_input_dim": 157,
        "network.arm.action_dim": 8,
        "network.observe_vel": False,
        "network.observe_clock_inputs": True,
        "network.observe_two_prev_actions": False,
        "network.use_rot6d": False,
        "network.inference_arm_body_plan_tanh": False,
        "runtime.profile": "go2_x5_joint40k_pd40_v1",
        "runtime.control_dt": 0.02,
        "runtime.sim_dt": 0.005,
        "runtime.decimation": 4,
        "runtime.num_envs": 1,
        "runtime.action_dim": 10,
        "runtime.observation_clip": 100.0,
        "runtime.action_clip": 10.0,
        "runtime.urdf_sha256": URDF_SHA256,
        "runtime.pd.legs.stiffness": 40.0,
        "runtime.pd.legs.damping": 1.0,
        "runtime.pd.arm_joint1.stiffness": 40.0,
        "runtime.pd.arm_joint1.damping": 3.0,
        "runtime.pd.arm_joint2_3.stiffness": 70.0,
        "runtime.pd.arm_joint2_3.damping": 15.0,
        "runtime.pd.arm_joint4_6.stiffness": 25.0,
        "runtime.pd.arm_joint4_6.damping": 2.0,
        "runtime.pd.gripper.stiffness": 50.0,
        "runtime.pd.gripper.damping": 20.0,
        "runtime.pd.gripper.friction": 0.05,
        "runtime.gait.frequency_hz": 3.0,
        "runtime.gait.phase": 0.5,
        "runtime.gait.offset": 0.0,
        "runtime.gait.bound": 0.0,
        "runtime.gait.duty": 0.5,
        "runtime.gait.stand_threshold": 0.1,
        "runtime.action_scales.hip": 0.125,
        "runtime.action_scales.thigh": 0.25,
        "runtime.action_scales.calf": 0.25,
        "runtime.action_scales.arm": 0.5,
        "runtime.tcp.public_frame": "arm_eef_link",
        "runtime.tcp.legacy_training_point_m": 0.1,
        "runtime.tcp.urdf_eef_offset_m": 0.08657,
        "runtime.tcp.ground_z_offset_m": 0.38,
    }
    for dotted_key, expected_value in expected.items():
        actual = _value(manifest, dotted_key)
        if actual != expected_value:
            raise ValueError(
                f"WholeBody manifest mismatch for '{dotted_key}': "
                f"expected {expected_value!r}, got {actual!r}"
            )
    for name in ("dog", "arm"):
        artifact = _value(manifest, f"artifacts.{name}")
        if not isinstance(artifact, dict):
            raise TypeError(f"artifacts.{name} must be a mapping")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"artifacts.{name}.sha256 must contain a complete SHA256")
        if not isinstance(artifact.get("relative_path"), str):
            raise ValueError(f"artifacts.{name}.relative_path must be a string")
    for name in ("lower", "upper", "effort", "velocity"):
        values = _value(manifest, f"runtime.joint_limits.{name}")
        if (
            not isinstance(values, list)
            or len(values) != 20
            or any(not isinstance(value, (int, float)) for value in values)
        ):
            raise ValueError(f"runtime.joint_limits.{name} must contain 20 numeric values")
    expected_joint_names = {
        "dog": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
        "arm": [f"arm_joint{index}" for index in range(1, 7)],
        "gripper": ["arm_joint7", "arm_joint8"],
    }
    if _value(manifest, "runtime.joint_names") != expected_joint_names:
        raise ValueError("runtime.joint_names does not match the frozen FL/FR/RL/RR/Arm/Gripper order")
    asset = _value(manifest, "runtime.asset")
    if (
        not isinstance(asset, dict)
        or not isinstance(asset.get("whole_body_com_base_m"), list)
        or len(asset["whole_body_com_base_m"]) != 3
        or any(not isinstance(value, (int, float)) for value in asset["whole_body_com_base_m"])
    ):
        raise ValueError("runtime.asset.whole_body_com_base_m must contain three numeric values")
    for name in ("total_mass_kg", "mass_tolerance_kg", "com_tolerance_m"):
        if not isinstance(asset.get(name), (int, float)) or asset[name] <= 0.0:
            raise ValueError(f"runtime.asset.{name} must be positive")


def resolve_artifacts(
    manifest: dict[str, Any],
    model_root: str | Path,
    urdf_path: str | Path,
) -> ArtifactPaths:
    root = Path(model_root).expanduser().resolve()
    paths: dict[str, Path] = {}
    for name in ("dog", "arm"):
        artifact = _value(manifest, f"artifacts.{name}")
        path = (root / artifact["relative_path"]).resolve()
        if root not in path.parents:
            raise ValueError(f"artifacts.{name}.relative_path escapes model_root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"WholeBody {name} checkpoint does not exist: {path}")
        actual_digest = sha256(path)
        if actual_digest != artifact["sha256"]:
            raise ValueError(
                f"WholeBody {name} checkpoint SHA256 mismatch: "
                f"expected {artifact['sha256']}, got {actual_digest}"
            )
        expected_size = artifact.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(
                f"WholeBody {name} checkpoint size mismatch: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
        paths[name] = path

    urdf = Path(urdf_path).expanduser().resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"Go2-X5 URDF does not exist: {urdf}")
    actual_urdf_digest = sha256(urdf)
    if actual_urdf_digest != _value(manifest, "runtime.urdf_sha256"):
        raise ValueError(
            "Go2-X5 URDF SHA256 mismatch: "
            f"expected {_value(manifest, 'runtime.urdf_sha256')}, got {actual_urdf_digest}"
        )
    return ArtifactPaths(paths["dog"], paths["arm"], urdf)
