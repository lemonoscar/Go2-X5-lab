from __future__ import annotations

from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "robot_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if "robot_lab" not in sys.modules:
    robot_lab = types.ModuleType("robot_lab")
    robot_lab.__path__ = [str(SOURCE_ROOT / "robot_lab")]
    sys.modules["robot_lab"] = robot_lab

from robot_lab.go2_x5_wholebody.ik import (  # noqa: E402
    ARM_JOINT_NAMES,
    PINNED_DEPENDENCIES,
    POLICY_ARM_ZERO,
    Go2X5IK,
    arm_dof_indices,
)


URDF = REPO_ROOT / "source" / "robot_lab" / "data" / "Robots" / "go2_x5" / "go2_x5.urdf"


def test_dependency_drift_fails_fast() -> None:
    versions = {**PINNED_DEPENDENCIES, "qpsolvers": "4.11.0", "quadprog": "0.1.13"}
    with patch("robot_lab.go2_x5_wholebody.ik.metadata.version", side_effect=versions.__getitem__):
        try:
            Go2X5IK(str(URDF))
        except ImportError as error:
            assert "qpsolvers" in str(error) and "4.8.2" in str(error)
        else:
            raise AssertionError("dependency drift was accepted")


def test_name_mapping_rejects_missing_and_duplicates() -> None:
    names = ["leg", *ARM_JOINT_NAMES, "arm_joint7", "arm_joint8"]
    assert arm_dof_indices(names) == (1, 2, 3, 4, 5, 6)
    for invalid in ([*names, "arm_joint1"], names[:-3]):
        try:
            arm_dof_indices(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid arm joint mapping was accepted")


def test_target_coordinate_convention_and_solver_limits() -> None:
    ik = Go2X5IK(str(URDF))
    half_yaw = np.deg2rad(45.0)
    target = ik.target_in_base(
        base_position_world=[1.0, 2.0, 0.5],
        base_quaternion_xyzw=[0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)],
        ground_height_world=0.1,
        target_lpy=[0.5, 0.0, 0.0],
        target_rpy=[0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(target.translation, [0.5, 0.0, -0.02], atol=1.0e-12)
    np.testing.assert_allclose(target.rotation, np.eye(3), atol=1.0e-12)

    exact = ik.forward_kinematics(POLICY_ARM_ZERO)
    command, status = ik.step(POLICY_ARM_ZERO, exact)
    assert status.solver_ok and not status.held
    np.testing.assert_allclose(command, POLICY_ARM_ZERO, atol=2.0e-5)

    moved = exact.copy()
    moved.translation += np.array([0.01, -0.005, 0.005])
    measured = POLICY_ARM_ZERO.copy()
    for _ in range(150):
        command, status = ik.step(measured, moved)
        assert status.solver_ok, status.message
        assert np.max(np.abs(command - measured)) <= 3.0 * ik.dt + 1.0e-9
        measured = command
    position_error, _ = ik.pose_error(measured, moved)
    assert position_error <= 0.002


def test_failure_holds_last_legal_command() -> None:
    ik = Go2X5IK(str(URDF))
    target = ik.forward_kinematics(POLICY_ARM_ZERO)
    valid, status = ik.step(POLICY_ARM_ZERO, target)
    assert status.solver_ok
    with patch.object(ik.pink, "solve_ik", return_value=np.full(6, np.nan)):
        held, held_status = ik.step(POLICY_ARM_ZERO, target)
    assert held_status.held and not held_status.solver_ok
    np.testing.assert_allclose(held, valid)
    nan_held, nan_status = ik.step(np.full(6, np.nan), target)
    assert nan_status.held and not nan_status.solver_ok
    np.testing.assert_allclose(nan_held, valid)
