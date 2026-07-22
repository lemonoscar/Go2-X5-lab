"""Configuration and sampler tests for the model_19750 planar specialization tasks."""

from __future__ import annotations

import pytest
import torch

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
MDP_DIR = REPO_ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp"
sys.path.insert(0, str(MDP_DIR))

from planar_command_utils import sample_stratified_planar_commands, validate_planar_command_spec


def test_planar_sampler_rejects_invalid_probability_and_bins_before_env_creation() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_planar_command_spec(
            (-0.7, 0.7),
            (-0.2, 0.2),
            (0.20, 0.35, 0.30, 0.25),
        )
    with pytest.raises(ValueError, match="non-zero"):
        validate_planar_command_spec(
            (-0.7, 0.7),
            (-0.2, 0.0, 0.2),
            (0.10, 0.35, 0.30, 0.25),
        )


def test_planar_sampler_preserves_small_lateral_bins_and_all_categories() -> None:
    torch.manual_seed(7)
    count = 20000
    commands, standing = sample_stratified_planar_commands(
        count,
        torch.tensor((-0.7, -0.15, 0.15, 0.7)),
        torch.tensor((-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)),
        torch.tensor((0.10, 0.45, 0.75, 1.0)),
    )
    vx = commands[:, 0]
    vy = commands[:, 1]

    assert torch.all(commands[:, 2] == 0.0)
    assert torch.any(standing)
    assert torch.any((vx != 0.0) & (vy == 0.0))
    assert torch.any((vx == 0.0) & (vy != 0.0))
    assert torch.any((vx != 0.0) & (vy != 0.0))
    assert {-0.05, 0.05}.issubset(set(round(value, 2) for value in vy.tolist()))
