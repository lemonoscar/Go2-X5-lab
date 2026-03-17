#!/usr/bin/env python3

"""Expand legacy Go2-X5 route checkpoints to the arm-aware route architecture.

Legacy route checkpoints were trained with:
- policy obs dim = 235
- critic obs dim = 235
- action dim = 12

The updated staged route keeps arm dimensions present from the start:
- policy/critic obs dim = 259
- action dim = 18

This script copies the learned leg-related weights into the new layout and
zero-initializes the new arm-related channels/outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_OBS_DIM = 235
NEW_OBS_DIM = 259
OLD_ACTION_DIM = 12
NEW_ACTION_DIM = 18

# Legacy route observation layout
OLD_BASE = slice(0, 12)
OLD_JOINT_POS = slice(12, 24)
OLD_JOINT_VEL = slice(24, 36)
OLD_ACTIONS = slice(36, 48)
OLD_HEIGHT = slice(48, 235)

# New route observation layout
NEW_BASE = slice(0, 12)
NEW_JOINT_POS_DOG = slice(12, 24)
NEW_JOINT_POS_ARM = slice(24, 30)
NEW_JOINT_VEL_DOG = slice(30, 42)
NEW_JOINT_VEL_ARM = slice(42, 48)
NEW_ACTIONS_DOG = slice(48, 60)
NEW_ACTIONS_ARM = slice(60, 66)
NEW_HEIGHT = slice(66, 253)
NEW_ARM_COMMAND = slice(253, 259)


def _expand_obs_weight(weight: torch.Tensor) -> torch.Tensor:
    if tuple(weight.shape) != (weight.shape[0], OLD_OBS_DIM):
        raise ValueError(f"Expected first-layer obs weight with dim {OLD_OBS_DIM}, got {tuple(weight.shape)}")

    expanded = weight.new_zeros((weight.shape[0], NEW_OBS_DIM))
    expanded[:, NEW_BASE] = weight[:, OLD_BASE]
    expanded[:, NEW_JOINT_POS_DOG] = weight[:, OLD_JOINT_POS]
    expanded[:, NEW_JOINT_VEL_DOG] = weight[:, OLD_JOINT_VEL]
    expanded[:, NEW_ACTIONS_DOG] = weight[:, OLD_ACTIONS]
    expanded[:, NEW_HEIGHT] = weight[:, OLD_HEIGHT]
    return expanded


def _expand_action_weight(weight: torch.Tensor) -> torch.Tensor:
    if tuple(weight.shape) != (OLD_ACTION_DIM, weight.shape[1]):
        raise ValueError(f"Expected actor output weight with dim {OLD_ACTION_DIM}, got {tuple(weight.shape)}")

    expanded = weight.new_zeros((NEW_ACTION_DIM, weight.shape[1]))
    expanded[:OLD_ACTION_DIM] = weight
    return expanded


def _expand_action_bias(bias: torch.Tensor) -> torch.Tensor:
    if tuple(bias.shape) != (OLD_ACTION_DIM,):
        raise ValueError(f"Expected actor output bias with dim {OLD_ACTION_DIM}, got {tuple(bias.shape)}")

    expanded = bias.new_zeros((NEW_ACTION_DIM,))
    expanded[:OLD_ACTION_DIM] = bias
    return expanded


def _expand_std(std: torch.Tensor, arm_std: float) -> torch.Tensor:
    if tuple(std.shape) != (OLD_ACTION_DIM,):
        raise ValueError(f"Expected std tensor with dim {OLD_ACTION_DIM}, got {tuple(std.shape)}")

    expanded = std.new_full((NEW_ACTION_DIM,), arm_std)
    expanded[:OLD_ACTION_DIM] = std
    return expanded


def migrate_checkpoint(input_path: Path, output_path: Path, arm_std: float) -> None:
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
            "This does not look like a legacy route checkpoint. "
            f"Observed dims: actor_in={actor_in}, critic_in={critic_in}, actor_out={actor_out}."
        )

    state["actor.0.weight"] = _expand_obs_weight(state["actor.0.weight"])
    state["critic.0.weight"] = _expand_obs_weight(state["critic.0.weight"])
    state["actor.6.weight"] = _expand_action_weight(state["actor.6.weight"])
    state["actor.6.bias"] = _expand_action_bias(state["actor.6.bias"])

    if "std" in state:
        state["std"] = _expand_std(state["std"], arm_std)
    if "log_std" in state:
        state["log_std"] = _expand_std(state["log_std"], float(torch.log(torch.tensor(arm_std)).item()))

    infos = checkpoint.get("infos", {})
    if not isinstance(infos, dict):
        infos = {"legacy_infos": infos}
    infos["go2_x5_route_migration"] = {
        "source_obs_dim": OLD_OBS_DIM,
        "target_obs_dim": NEW_OBS_DIM,
        "source_action_dim": OLD_ACTION_DIM,
        "target_action_dim": NEW_ACTION_DIM,
        "arm_action_init": "zeros",
        "arm_obs_init": "zeros",
        "arm_std": arm_std,
    }
    checkpoint["infos"] = infos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Legacy checkpoint path.")
    parser.add_argument("--output", type=Path, default=None, help="Output checkpoint path.")
    parser.add_argument(
        "--arm-std",
        type=float,
        default=0.25,
        help="Initial exploration std for the newly added arm action dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_name(
        f"{input_path.stem}_armdims{input_path.suffix}"
    )

    migrate_checkpoint(input_path, output_path, arm_std=args.arm_std)
    print(f"[INFO] Migrated checkpoint written to: {output_path}")


if __name__ == "__main__":
    main()
