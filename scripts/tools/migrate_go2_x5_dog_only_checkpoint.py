#!/usr/bin/env python3

"""Migrate whole-body Go2-X5 flat checkpoints to the dog-only architecture.

Source route checkpoints:
- obs dim = 259
- action dim = 18 (12 dog + 6 arm)

Target dog-only checkpoints:
- obs dim = 260 (adds 1-d gripper command placeholder)
- action dim = 12 (dog joints only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_OBS_DIM = 259
NEW_OBS_DIM = 260
OLD_ACTION_DIM = 18
NEW_ACTION_DIM = 12


def _expand_obs_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[1] != OLD_OBS_DIM:
        raise ValueError(f"Expected first-layer obs dim {OLD_OBS_DIM}, got {tuple(weight.shape)}")
    expanded = weight.new_zeros((weight.shape[0], NEW_OBS_DIM))
    expanded[:, :OLD_OBS_DIM] = weight
    return expanded


def _shrink_action_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[0] != OLD_ACTION_DIM:
        raise ValueError(f"Expected actor output dim {OLD_ACTION_DIM}, got {tuple(weight.shape)}")
    return weight[:NEW_ACTION_DIM].clone()


def _shrink_action_bias(bias: torch.Tensor) -> torch.Tensor:
    if bias.shape[0] != OLD_ACTION_DIM:
        raise ValueError(f"Expected actor output bias dim {OLD_ACTION_DIM}, got {tuple(bias.shape)}")
    return bias[:NEW_ACTION_DIM].clone()


def _shrink_std(std: torch.Tensor) -> torch.Tensor:
    if std.shape[0] != OLD_ACTION_DIM:
        raise ValueError(f"Expected std dim {OLD_ACTION_DIM}, got {tuple(std.shape)}")
    return std[:NEW_ACTION_DIM].clone()


def migrate_checkpoint(input_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(input_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    state = checkpoint["model_state_dict"]

    required_keys = ["actor.0.weight", "critic.0.weight", "actor.6.weight", "actor.6.bias"]
    for key in required_keys:
        if key not in state:
            raise KeyError(f"Missing required key '{key}' in checkpoint.")

    actor_in = state["actor.0.weight"].shape[1]
    critic_in = state["critic.0.weight"].shape[1]
    actor_out = state["actor.6.weight"].shape[0]
    if actor_in != OLD_OBS_DIM or critic_in != OLD_OBS_DIM or actor_out != OLD_ACTION_DIM:
        raise ValueError(
            "This does not look like a whole-body route checkpoint. "
            f"Observed dims: actor_in={actor_in}, critic_in={critic_in}, actor_out={actor_out}."
        )

    state["actor.0.weight"] = _expand_obs_weight(state["actor.0.weight"])
    state["critic.0.weight"] = _expand_obs_weight(state["critic.0.weight"])
    state["actor.6.weight"] = _shrink_action_weight(state["actor.6.weight"])
    state["actor.6.bias"] = _shrink_action_bias(state["actor.6.bias"])

    if "std" in state:
        state["std"] = _shrink_std(state["std"])
    if "log_std" in state:
        state["log_std"] = _shrink_std(state["log_std"])

    infos = checkpoint.get("infos", {})
    if not isinstance(infos, dict):
        infos = {"legacy_infos": infos}
    infos["go2_x5_dog_only_migration"] = {
        "source_obs_dim": OLD_OBS_DIM,
        "target_obs_dim": NEW_OBS_DIM,
        "source_action_dim": OLD_ACTION_DIM,
        "target_action_dim": NEW_ACTION_DIM,
        "actor_first_layer": "copied + zero gripper column",
        "actor_hidden_layers": "copied",
        "actor_output_layer": "dog rows copied, arm rows dropped",
        "critic_first_layer": "copied + zero gripper column",
        "critic_hidden_layers": "copied",
    }
    checkpoint["infos"] = infos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Whole-body checkpoint path.")
    parser.add_argument("--output", type=Path, default=None, help="Output checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_name(
        f"{input_path.stem}_dogonly{input_path.suffix}"
    )
    migrate_checkpoint(input_path, output_path)
    print(f"[INFO] Migrated checkpoint written to: {output_path}")


if __name__ == "__main__":
    main()
