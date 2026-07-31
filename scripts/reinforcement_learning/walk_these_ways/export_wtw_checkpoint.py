#!/usr/bin/env python3
"""Export a Go2-X5 WTW RSL-RL checkpoint as two JIT modules plus a bound manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from torch import nn

from wtw_policy_adapter import (
    ACTION_DIM,
    ACTION_SCALES,
    DEFAULT_GRIPPER_JOINT_POS,
    DEFAULT_JOINT_POS,
    DEFAULT_LEG_DAMPING,
    DEFAULT_LEG_STIFFNESS,
    DEFAULT_SPAWN_HEIGHT_M,
    HISTORY_LENGTH,
    LATENT_DIM,
    MANIFEST_FORMAT_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_HISTORY_DIM,
    POLICY_DT_S,
    WTW_JOINT_NAMES,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
KNOWN_PARENT_SHA256 = "1f4218009a9d269ffb54b9034b6a488b09062fda6d1115cd4ac7943a70a81c43"
DEFAULT_PARENT_CHECKPOINT = (
    REPO_ROOT.parent
    / "walk-these-ways-go2"
    / "runs"
    / "gait-conditioned-agility"
    / "pretrain-go2"
    / "train"
    / "142238.667503"
    / "checkpoints"
    / "ac_weights_last.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ELU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _submodule_state(
    model_state: dict[str, torch.Tensor],
    *,
    prefix: str,
    target: nn.Module,
) -> dict[str, torch.Tensor]:
    state = {
        key.removeprefix(prefix): value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    expected = set(target.state_dict())
    if set(state) != expected:
        raise ValueError(
            f"checkpoint {prefix.rstrip('.')} key mismatch: "
            f"missing={sorted(expected.difference(state))}, unexpected={sorted(set(state).difference(expected))}"
        )
    return state


def _git_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("critic_calibration", "r0_actor_continuation"), required=True)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--expected-parent-sha256", default=KNOWN_PARENT_SHA256)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--git-commit", default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    parent_path = args.parent_checkpoint.expanduser().resolve()
    for path in (checkpoint_path, parent_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent_sha256 = _sha256(parent_path)
    if parent_sha256 != args.expected_parent_sha256:
        raise ValueError(
            f"parent checkpoint SHA-256 mismatch: expected {args.expected_parent_sha256}, got {parent_sha256}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("RSL checkpoint must contain a model_state_dict object")
    checkpoint_iteration = checkpoint.get("iter")
    if not isinstance(checkpoint_iteration, int) or checkpoint_iteration < 0:
        raise ValueError("RSL checkpoint iter must be a non-negative integer")
    if args.iteration is not None and args.iteration != checkpoint_iteration:
        raise ValueError(
            f"requested iteration {args.iteration} does not match checkpoint iter {checkpoint_iteration}"
        )
    model_state = checkpoint["model_state_dict"]
    unexpected_model_keys = [
        key
        for key in model_state
        if key != "std"
        and not key.startswith("adaptation_module.")
        and not key.startswith("actor_body.")
        and not key.startswith("critic.")
    ]
    if unexpected_model_keys:
        raise ValueError(f"unexpected WTW model_state_dict keys: {sorted(unexpected_model_keys)}")

    adaptation = _make_mlp(OBSERVATION_HISTORY_DIM, (256, 128), LATENT_DIM).eval()
    body = _make_mlp(OBSERVATION_HISTORY_DIM + LATENT_DIM, (512, 256, 128), ACTION_DIM).eval()
    adaptation.load_state_dict(
        _submodule_state(model_state, prefix="adaptation_module.", target=adaptation),
        strict=True,
    )
    body.load_state_dict(_submodule_state(model_state, prefix="actor_body.", target=body), strict=True)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adaptation_path = output_dir / "adaptation_module.jit"
    body_path = output_dir / "body.jit"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (adaptation_path, body_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing export artifacts: {existing}")

    scripted_adaptation = torch.jit.script(adaptation)
    scripted_body = torch.jit.script(body)
    torch.jit.save(scripted_adaptation, os.fspath(adaptation_path))
    torch.jit.save(scripted_body, os.fspath(body_path))

    generator = torch.Generator(device="cpu").manual_seed(0)
    histories = torch.randn(11, OBSERVATION_HISTORY_DIM, generator=generator)
    histories[0].zero_()
    with torch.inference_mode():
        expected_latent = adaptation(histories)
        actual_latent = scripted_adaptation(histories)
        body_inputs = torch.cat((histories, expected_latent), dim=-1)
        expected_action = body(body_inputs)
        actual_action = scripted_body(body_inputs)
    latent_diff = float(torch.max(torch.abs(expected_latent - actual_latent)).item())
    action_diff = float(torch.max(torch.abs(expected_action - actual_action)).item())
    if latent_diff > 1.0e-6 or action_diff > 1.0e-6:
        raise RuntimeError(
            f"JIT parity failed: adaptation max_abs_diff={latent_diff}, body max_abs_diff={action_diff}"
        )

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "stage": args.stage,
        "parent_checkpoint": {
            "path": os.fspath(parent_path),
            "sha256": parent_sha256,
        },
        "rsl_checkpoint": {
            "path": os.fspath(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "iteration": checkpoint_iteration,
        },
        "adaptation": {
            "path": adaptation_path.name,
            "sha256": _sha256(adaptation_path),
        },
        "body": {
            "path": body_path.name,
            "sha256": _sha256(body_path),
        },
        "abi": {
            "observation_dim": OBSERVATION_DIM,
            "history_length": HISTORY_LENGTH,
            "history_dim": OBSERVATION_HISTORY_DIM,
            "latent_dim": LATENT_DIM,
            "action_dim": ACTION_DIM,
            "joint_order": list(WTW_JOINT_NAMES),
            "default_joint_pos": list(DEFAULT_JOINT_POS),
            "action_scales": list(ACTION_SCALES),
            "policy_dt_s": POLICY_DT_S,
        },
        "controller": {
            "leg_stiffness": DEFAULT_LEG_STIFFNESS,
            "leg_damping": DEFAULT_LEG_DAMPING,
            "spawn_height_m": DEFAULT_SPAWN_HEIGHT_M,
            "gripper_target_m": list(DEFAULT_GRIPPER_JOINT_POS),
        },
        "git_commit": _git_commit(args.git_commit),
        "verification": {
            "history_samples": len(histories),
            "adaptation_max_abs_diff": latent_diff,
            "body_max_abs_diff": action_diff,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported WTW continuation bundle: {manifest_path}")


if __name__ == "__main__":
    main()
