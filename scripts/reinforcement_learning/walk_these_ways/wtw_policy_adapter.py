"""Isaac-independent adapter for the exported Walk These Ways Go2 policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch


OBSERVATION_DIM = 70
HISTORY_LENGTH = 30
OBSERVATION_HISTORY_DIM = OBSERVATION_DIM * HISTORY_LENGTH
LATENT_DIM = 2
ACTION_DIM = 12
COMMAND_DIM = 15
POLICY_DT_S = 0.02
ACTION_CLIP = 10.0
OBSERVATION_CLIP = 100.0
MANIFEST_FORMAT_VERSION = 1
DEFAULT_LEG_STIFFNESS = 40.0
DEFAULT_LEG_DAMPING = 1.0
DEFAULT_SPAWN_HEIGHT_M = 0.30
GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")
DEFAULT_GRIPPER_JOINT_POS = (0.044, 0.044)

WTW_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
DEFAULT_JOINT_POS = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)
ACTION_SCALES = (0.125, 0.25, 0.25) * 4
# Fixed feed-forward action obtained from the steady-cycle mean of actor1500 at
# 0.5 m/s.  Unlike a zero action, it supplies the gravity torques needed by the
# front-loaded Go2-X5 to keep all four feet down at 40/1 leg PD gains.
STAND_ACTION = (
    -1.0105,
    -0.2757,
    0.4758,
    1.5379,
    -0.1350,
    0.4137,
    -1.2827,
    -0.1131,
    -0.0578,
    1.4618,
    0.0613,
    0.0940,
)
COMMAND_SCALES = (2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 0.3, 0.3, 1.0, 1.0, 1.0)

KNOWN_BODY_SHA256 = "7b6e604e2147742a89ef50d91e7ee501023331b2589d1c3143a9d2ba858db7b5"
KNOWN_ADAPTATION_SHA256 = "0e091f829dcfbedd4ccca6752863e1e2feca105f79da07d04e3545b8815dcc13"
ZERO_HISTORY_LATENT = (-1.538343906402588, -1.3359832763671875)
ZERO_HISTORY_ACTION = (
    -0.5988616943359375,
    0.17977777123451233,
    0.03785664960741997,
    0.5643272399902344,
    0.33564773201942444,
    0.16656625270843506,
    -0.7482739090919495,
    0.03009459376335144,
    -0.05163779854774475,
    1.0494321584701538,
    0.20368297398090363,
    -0.5978571772575378,
)


def make_walking_command(
    vx: float,
    vy: float = 0.0,
    wz: float = 0.0,
    *,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return the 15-D command used by the exported policy's nominal trot."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    command = (
        vx,
        vy,
        wz,
        0.0,  # body-height offset
        2.5,  # gait frequency
        0.5,  # phase
        0.0,  # offset
        0.0,  # bound
        0.5,  # stance duration
        0.08,  # foot-swing height
        0.0,  # body pitch
        0.0,  # body roll
        0.25,  # stance width
        0.4,  # stance length
        0.0,  # auxiliary reward command
    )
    return torch.tensor(command, dtype=torch.float32, device=device).repeat(batch_size, 1)


def make_standing_command(
    *,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return the Go2-X5 STAND command; its leg action is handled outside the actor."""
    command = make_walking_command(0.0, batch_size=batch_size, device=device)
    command[:, 4] = 0.0  # freeze the gait clock
    command[:, 5:8] = 0.0  # repurpose the former pronk tuple as STAND
    command[:, 8] = 1.0  # all-stance semantics
    command[:, 9] = 0.0  # no swing in STAND
    return command


def make_two_state_command(
    vx: float,
    vy: float = 0.0,
    wz: float = 0.0,
    *,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Select STAND for zero velocity and nominal trot for any walking command."""
    if max(abs(vx), abs(vy), abs(wz)) <= 1.0e-6:
        return make_standing_command(batch_size=batch_size, device=device)
    return make_walking_command(vx, vy, wz, batch_size=batch_size, device=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {label}.path must be a non-empty string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _manifest_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"manifest {label}.sha256 must be a lowercase SHA-256 hex digest")
    return value


def _require_mapping(owner: dict[str, Any], key: str) -> dict[str, Any]:
    value = owner.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"manifest {key} must be an object")
    return value


def _check_manifest_sequence(actual: object, expected: Sequence[object], label: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError(f"manifest {label} must contain {len(expected)} values")
    if all(isinstance(value, str) for value in expected):
        matches = tuple(actual) == tuple(expected)
    else:
        try:
            matches = all(
                math.isclose(float(value), float(reference), abs_tol=1.0e-7)
                for value, reference in zip(actual, expected, strict=True)
            )
        except (TypeError, ValueError):
            matches = False
    if not matches:
        raise ValueError(f"manifest {label} mismatch: expected {list(expected)}, got {actual}")


def load_and_validate_manifest(
    manifest_path: str | Path,
    *,
    body_path: str | Path,
    adaptation_path: str | Path,
) -> dict[str, Any]:
    """Load a continuation bundle manifest and bind it to the supplied JIT files."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read WTW manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("WTW manifest root must be an object")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"unsupported WTW manifest format_version: expected {MANIFEST_FORMAT_VERSION}, "
            f"got {manifest.get('format_version')}"
        )
    if not isinstance(manifest.get("stage"), str) or not manifest["stage"]:
        raise ValueError("manifest stage must be a non-empty string")
    if not isinstance(manifest.get("git_commit"), str) or not manifest["git_commit"]:
        raise ValueError("manifest git_commit must be a non-empty string")

    supplied_paths = {
        "body": Path(body_path).expanduser().resolve(),
        "adaptation": Path(adaptation_path).expanduser().resolve(),
    }
    for label, supplied_path in supplied_paths.items():
        artifact = _require_mapping(manifest, label)
        declared_path = _manifest_path(manifest_path, artifact.get("path"), label)
        if declared_path != supplied_path:
            raise ValueError(
                f"manifest {label}.path resolves to {declared_path}, but evaluator supplied {supplied_path}"
            )
        expected_sha = _manifest_sha256(artifact.get("sha256"), label)
        actual_sha = _sha256(supplied_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{label} checkpoint SHA-256 mismatch: expected {expected_sha}, got {actual_sha} ({supplied_path})"
            )

    abi = _require_mapping(manifest, "abi")
    expected_scalars = {
        "observation_dim": OBSERVATION_DIM,
        "history_length": HISTORY_LENGTH,
        "history_dim": OBSERVATION_HISTORY_DIM,
        "latent_dim": LATENT_DIM,
        "action_dim": ACTION_DIM,
    }
    for key, expected in expected_scalars.items():
        if abi.get(key) != expected:
            raise ValueError(f"manifest ABI {key} mismatch: expected {expected}, got {abi.get(key)}")
    try:
        policy_dt_matches = math.isclose(
            float(abi.get("policy_dt_s")), POLICY_DT_S, abs_tol=1.0e-9
        )
    except (TypeError, ValueError):
        policy_dt_matches = False
    if not policy_dt_matches:
        raise ValueError(
            f"manifest ABI policy_dt_s mismatch: expected {POLICY_DT_S}, got {abi.get('policy_dt_s')}"
        )
    _check_manifest_sequence(abi.get("joint_order"), WTW_JOINT_NAMES, "ABI joint_order")
    _check_manifest_sequence(abi.get("default_joint_pos"), DEFAULT_JOINT_POS, "ABI default_joint_pos")
    _check_manifest_sequence(abi.get("action_scales"), ACTION_SCALES, "ABI action_scales")

    controller = _require_mapping(manifest, "controller")
    expected_controller = {
        "leg_stiffness": DEFAULT_LEG_STIFFNESS,
        "leg_damping": DEFAULT_LEG_DAMPING,
        "spawn_height_m": DEFAULT_SPAWN_HEIGHT_M,
    }
    for key, expected in expected_controller.items():
        try:
            matches = math.isclose(float(controller.get(key)), expected, abs_tol=1.0e-9)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"manifest controller {key} mismatch: expected {expected}, got {controller.get(key)}")
    _check_manifest_sequence(
        controller.get("gripper_target_m"),
        DEFAULT_GRIPPER_JOINT_POS,
        "controller.gripper_target_m",
    )

    parent = _require_mapping(manifest, "parent_checkpoint")
    if not isinstance(parent.get("path"), str) or not parent["path"]:
        raise ValueError("manifest parent_checkpoint.path must be a non-empty string")
    _manifest_sha256(parent.get("sha256"), "parent_checkpoint")
    rsl_checkpoint = _require_mapping(manifest, "rsl_checkpoint")
    if not isinstance(rsl_checkpoint.get("path"), str) or not rsl_checkpoint["path"]:
        raise ValueError("manifest rsl_checkpoint.path must be a non-empty string")
    _manifest_sha256(rsl_checkpoint.get("sha256"), "rsl_checkpoint")
    if not isinstance(rsl_checkpoint.get("iteration"), int) or rsl_checkpoint["iteration"] < 0:
        raise ValueError("manifest rsl_checkpoint.iteration must be a non-negative integer")
    return manifest


class WTWPolicyAdapter:
    """Own the policy history/action/clock state for one batched rollout."""

    def __init__(
        self,
        adaptation_module: torch.nn.Module,
        body: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.adaptation_module = adaptation_module.to(self.device).eval()
        self.body = body.to(self.device).eval()
        self._validate_model_contract()
        self.reset()

    @classmethod
    def from_jit_paths(
        cls,
        *,
        body_path: str | Path,
        adaptation_path: str | Path,
        manifest_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        verify_known_checkpoint: bool = True,
    ) -> "WTWPolicyAdapter":
        """Load two exported modules using either a bundle manifest or the original golden hashes."""
        body_path = Path(body_path).expanduser().resolve()
        adaptation_path = Path(adaptation_path).expanduser().resolve()
        manifest = None
        if manifest_path is not None:
            manifest = load_and_validate_manifest(
                manifest_path,
                body_path=body_path,
                adaptation_path=adaptation_path,
            )
        elif verify_known_checkpoint:
            cls._check_hash(body_path, KNOWN_BODY_SHA256, "body")
            cls._check_hash(adaptation_path, KNOWN_ADAPTATION_SHA256, "adaptation")

        adapter = cls(
            torch.jit.load(str(adaptation_path), map_location=device),
            torch.jit.load(str(body_path), map_location=device),
            device=device,
        )
        adapter.manifest = manifest
        if manifest_path is None and verify_known_checkpoint:
            adapter._validate_zero_history_golden()
        return adapter

    @staticmethod
    def _check_hash(path: Path, expected: str, label: str) -> None:
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{label} checkpoint SHA-256 mismatch: expected {expected}, got {actual} ({path})")

    def _run_modules(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            latent = self.adaptation_module(history)
            if not isinstance(latent, torch.Tensor) or latent.shape != (history.shape[0], LATENT_DIM):
                shape = getattr(latent, "shape", None)
                raise ValueError(f"adaptation output must have shape (N, {LATENT_DIM}), got {shape}")
            body_input = torch.cat((history, latent), dim=-1)
            action = self.body(body_input)
            if not isinstance(action, torch.Tensor) or action.shape != (history.shape[0], ACTION_DIM):
                shape = getattr(action, "shape", None)
                raise ValueError(f"body output must have shape (N, {ACTION_DIM}), got {shape}")
            if not torch.isfinite(latent).all() or not torch.isfinite(action).all():
                raise ValueError("policy produced non-finite latent values or actions")
        return latent, action

    def _validate_model_contract(self) -> None:
        history = torch.zeros(1, OBSERVATION_HISTORY_DIM, dtype=torch.float32, device=self.device)
        self._run_modules(history)

    def _validate_zero_history_golden(self) -> None:
        history = torch.zeros(1, OBSERVATION_HISTORY_DIM, dtype=torch.float32, device=self.device)
        latent, action = self._run_modules(history)
        expected_latent = torch.tensor(ZERO_HISTORY_LATENT, dtype=torch.float32, device=self.device).unsqueeze(0)
        expected_action = torch.tensor(ZERO_HISTORY_ACTION, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not torch.allclose(latent, expected_latent, rtol=1e-5, atol=1e-5):
            raise ValueError("adaptation module failed the known zero-history golden check")
        if not torch.allclose(action, expected_action, rtol=1e-5, atol=1e-5):
            raise ValueError("body module failed the known zero-history golden check")

    def reset(self, num_envs: int = 1) -> None:
        """Reset all rollout state; the first inference sees an all-zero history."""
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs = num_envs
        self.observation_history = torch.zeros(
            num_envs, OBSERVATION_HISTORY_DIM, dtype=torch.float32, device=self.device
        )
        self.previous_action = torch.zeros(num_envs, ACTION_DIM, dtype=torch.float32, device=self.device)
        self.gait_index = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.clock_inputs = torch.zeros(num_envs, 4, dtype=torch.float32, device=self.device)

    def infer(self) -> torch.Tensor:
        """Infer a clipped action without changing rollout state."""
        return self.infer_raw().clamp(-ACTION_CLIP, ACTION_CLIP)

    def infer_raw(self) -> torch.Tensor:
        """Infer the unclipped network output without changing rollout state."""
        _, action = self._run_modules(self.observation_history)
        return action

    def _as_batch(self, value: torch.Tensor | Sequence[float], width: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        expected = (self.num_envs, width)
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
        return tensor

    def advance(
        self,
        *,
        projected_gravity: torch.Tensor | Sequence[float],
        command: torch.Tensor | Sequence[float],
        joint_pos: torch.Tensor | Sequence[float],
        joint_vel: torch.Tensor | Sequence[float],
        applied_action: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        """Append one post-physics observation and advance action/gait state."""
        gravity = self._as_batch(projected_gravity, 3, "projected_gravity")
        command = self._as_batch(command, COMMAND_DIM, "command")
        joint_pos = self._as_batch(joint_pos, ACTION_DIM, "joint_pos")
        joint_vel = self._as_batch(joint_vel, ACTION_DIM, "joint_vel")
        applied_action = self._as_batch(applied_action, ACTION_DIM, "applied_action").clamp(
            -ACTION_CLIP, ACTION_CLIP
        )

        self.gait_index = torch.remainder(self.gait_index + POLICY_DT_S * command[:, 4], 1.0)
        phase = command[:, 5]
        offset = command[:, 6]
        bound = command[:, 7]
        foot_phase = torch.stack(
            (
                self.gait_index + phase + offset + bound,
                self.gait_index + offset,
                self.gait_index + bound,
                self.gait_index + phase,
            ),
            dim=-1,
        )
        self.clock_inputs = torch.sin(2.0 * math.pi * foot_phase)

        q0 = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=self.device)
        command_scales = torch.tensor(COMMAND_SCALES, dtype=torch.float32, device=self.device)
        observation = torch.cat(
            (
                gravity,
                command * command_scales,
                joint_pos - q0,
                joint_vel * 0.05,
                applied_action,
                self.previous_action,
                self.clock_inputs,
            ),
            dim=-1,
        )
        if not torch.isfinite(observation).all():
            raise ValueError("constructed observation contains non-finite values")
        observation = observation.clamp(-OBSERVATION_CLIP, OBSERVATION_CLIP)
        self.observation_history = torch.cat(
            (self.observation_history[:, OBSERVATION_DIM:], observation),
            dim=-1,
        )
        self.previous_action = applied_action.clone()
        return observation
