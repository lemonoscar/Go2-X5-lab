"""Geometry and configuration tests for the mixed flat/short-stair task."""

from __future__ import annotations

import math
import importlib.util

import numpy as np
import pytest
import torch

from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
MDP_DIR = REPO_ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp"
TERRAIN_FILE = (
    REPO_ROOT
    / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5"
    / "mixed_short_stairs_terrain.py"
)
sys.path.insert(0, str(MDP_DIR))

from mixed_short_stairs_utils import (
    MODE_COMBINED,
    MODE_PURE_VX,
    MODE_PURE_VY,
    MODE_STAND,
    MODE_TURN,
    sample_flat_mixed_short_stairs_commands,
    validate_mixed_short_stairs_command_spec,
)


def _load_terrain_module_without_isaac():
    isaaclab_module = types.ModuleType("isaaclab")
    terrains_module = types.ModuleType("isaaclab.terrains")
    utils_module = types.ModuleType("isaaclab.utils")
    terrains_module.SubTerrainBaseCfg = object
    utils_module.configclass = lambda cls: cls
    isaaclab_module.terrains = terrains_module
    isaaclab_module.utils = utils_module
    sys.modules.setdefault("isaaclab", isaaclab_module)
    sys.modules.setdefault("isaaclab.terrains", terrains_module)
    sys.modules.setdefault("isaaclab.utils", utils_module)

    spec = importlib.util.spec_from_file_location("mixed_short_stairs_terrain_unit", TERRAIN_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TERRAIN_MODULE = _load_terrain_module_without_isaac()
MixedShortStairsTerrainCfg = TERRAIN_MODULE.MixedShortStairsTerrainCfg
mixed_short_stairs_terrain = TERRAIN_MODULE.mixed_short_stairs_terrain
resolve_short_stair_step_count = TERRAIN_MODULE.resolve_short_stair_step_count


def _terrain_cfg(*, descent: bool) -> MixedShortStairsTerrainCfg:
    cfg = MixedShortStairsTerrainCfg()
    cfg.proportion = 1.0
    cfg.descent = descent
    cfg.size = (6.0, 6.0)
    return cfg


@pytest.mark.parametrize(("difficulty", "expected_steps"), ((0.0, 2), (1.0, 3)))
def test_ascent_has_exact_riser_and_tread_geometry(
    difficulty: float, expected_steps: int
) -> None:
    cfg = _terrain_cfg(descent=False)
    meshes, origin = mixed_short_stairs_terrain(difficulty, cfg)

    assert resolve_short_stair_step_count(difficulty, 2, 3) == expected_steps
    np.testing.assert_allclose(origin, (1.0, 3.0, 0.0), atol=1.0e-9)
    step_meshes = meshes[1 : 1 + expected_steps]
    assert len(step_meshes) == expected_steps
    np.testing.assert_allclose(
        [mesh.bounds[1, 2] for mesh in step_meshes],
        np.arange(1, expected_steps + 1) * 0.10,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        [mesh.bounds[1, 0] - mesh.bounds[0, 0] for mesh in step_meshes],
        np.full(expected_steps, 0.25),
        atol=1.0e-9,
    )
    assert step_meshes[0].bounds[0, 0] - origin[0] == pytest.approx(1.25)
    assert meshes[-1].bounds[1, 0] - meshes[-1].bounds[0, 0] >= 1.25


@pytest.mark.parametrize(("difficulty", "expected_steps"), ((0.0, 2), (1.0, 3)))
def test_descent_origin_and_surfaces_encode_exact_drops(
    difficulty: float, expected_steps: int
) -> None:
    cfg = _terrain_cfg(descent=True)
    meshes, origin = mixed_short_stairs_terrain(difficulty, cfg)

    total_height = expected_steps * 0.10
    np.testing.assert_allclose(origin, (1.0, 3.0, total_height), atol=1.0e-9)
    top_platform = meshes[1]
    assert top_platform.bounds[1, 2] == pytest.approx(total_height)
    assert top_platform.bounds[1, 0] - origin[0] == pytest.approx(1.25)

    lower_treads = meshes[2:]
    surface_heights = [total_height]
    surface_heights.extend(mesh.bounds[1, 2] for mesh in lower_treads)
    surface_heights.append(0.0)
    np.testing.assert_allclose(
        np.diff(surface_heights), np.full(expected_steps, -0.10), atol=1.0e-9
    )
    np.testing.assert_allclose(
        [mesh.bounds[1, 0] - mesh.bounds[0, 0] for mesh in lower_treads],
        np.full(expected_steps - 1, 0.25),
        atol=1.0e-9,
    )


def test_flat_command_sampler_matches_modes_deadband_and_turn_contract() -> None:
    vx_values = (-0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40)
    vy_values = (-0.20, -0.15, -0.10, 0.10, 0.15, 0.20)
    ascent_values = (0.18, 0.22, 0.25)
    descent_values = (0.12, 0.16, 0.20)
    probabilities = (0.25, 0.25, 0.20, 0.10, 0.20)
    validate_mixed_short_stairs_command_spec(
        vx_values,
        vy_values,
        ascent_values,
        descent_values,
        probabilities,
        0.10,
        0.40,
        math.pi / 4.0,
    )

    torch.manual_seed(17)
    commands, modes, arm_motion, turn_delta = sample_flat_mixed_short_stairs_commands(
        20000,
        torch.tensor(vx_values),
        torch.tensor(vy_values),
        torch.tensor(probabilities).cumsum(dim=0),
        0.40,
        math.pi / 4.0,
    )
    for mode, probability in enumerate(probabilities):
        assert torch.mean((modes == mode).float()).item() == pytest.approx(
            probability, abs=0.015
        )
    assert torch.all(commands[modes == MODE_STAND] == 0.0)
    assert torch.all(commands[modes == MODE_TURN] == 0.0)
    assert torch.all(commands[modes == MODE_PURE_VX, 1:] == 0.0)
    assert torch.all(commands[modes == MODE_PURE_VY][:, (0, 2)] == 0.0)
    assert torch.all(commands[modes == MODE_COMBINED, :2] != 0.0)
    moving_components = torch.abs(commands[:, :2][commands[:, :2] != 0.0])
    assert torch.min(moving_components).item() == pytest.approx(0.10)
    assert torch.all(~arm_motion | (modes == MODE_STAND))
    assert torch.mean(arm_motion[modes == MODE_STAND].float()).item() == pytest.approx(
        0.40, abs=0.02
    )
    assert sorted(torch.unique(turn_delta[modes == MODE_TURN]).tolist()) == pytest.approx(
        [-math.pi / 4.0, math.pi / 4.0]
    )


def test_command_spec_rejects_a_hidden_low_speed_bin() -> None:
    with pytest.raises(ValueError, match="magnitude >= 0.1"):
        validate_mixed_short_stairs_command_spec(
            (-0.4, 0.05, 0.4),
            (-0.2, 0.2),
            (0.18, 0.22, 0.25),
            (0.12, 0.16, 0.20),
            (0.25, 0.25, 0.20, 0.10, 0.20),
            0.10,
            0.40,
            math.pi / 4.0,
        )
