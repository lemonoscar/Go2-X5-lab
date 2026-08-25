from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "robot_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if "robot_lab" not in sys.modules:
    robot_lab = types.ModuleType("robot_lab")
    robot_lab.__path__ = [str(SOURCE_ROOT / "robot_lab")]
    sys.modules["robot_lab"] = robot_lab

from robot_lab.go2_x5_wholebody.controller import (  # noqa: E402
    ARM_POLICY_ZERO,
    DOG_DEFAULT_JOINT_POS,
    RobotState,
    WholeBodyController,
    canonical_command,
)
from robot_lab.go2_x5_wholebody.ik import Go2X5IK  # noqa: E402
from robot_lab.go2_x5_wholebody.models import (  # noqa: E402
    ArmActorCritic,
    DogActorCritic,
    load_actor_critics,
)


GOLDEN = REPO_ROOT / "test" / "fixtures" / "go2_x5_wholebody" / "019999_golden.json.gz"
MODEL_ROOT = REPO_ROOT / "models" / "go2_x5_wholebody" / "019999"
URDF = SOURCE_ROOT / "data" / "Robots" / "go2_x5" / "go2_x5.urdf"


class FakeIK:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, object]] = []

    def reset(self) -> None:
        self.calls.clear()

    def target_in_base(self, **kwargs):
        return kwargs

    def step(self, measured_q, target):
        self.calls.append((np.asarray(measured_q), target))
        return np.asarray(ARM_POLICY_ZERO, dtype=np.float64) + 0.05, {"ok": True}


def _zero_models() -> tuple[DogActorCritic, ArmActorCritic]:
    dog, arm = DogActorCritic(), ArmActorCritic()
    for model in (dog, arm):
        for parameter in model.parameters():
            parameter.data.zero_()
    dog.actor_body[-1].bias.data.copy_(torch.arange(12, dtype=torch.float32) / 10.0)
    arm.actor_body[-1].bias.data[-2:] = torch.tensor([2.0, -2.0])
    return dog, arm


def _state() -> RobotState:
    return RobotState(
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        dog_joint_pos=torch.tensor([DOG_DEFAULT_JOINT_POS]),
        dog_joint_vel=torch.zeros(1, 12),
        arm_joint_pos=torch.tensor([ARM_POLICY_ZERO]),
        base_roll_pitch=torch.tensor([[0.1, -0.2]]),
        base_position_world=torch.tensor([[1.0, 2.0, 0.34]]),
        base_quaternion_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        ground_height_world=torch.zeros(1),
    )


def test_reset_first_frame_and_source_execution_order() -> None:
    dog, arm = _zero_models()
    ik = FakeIK()
    controller = WholeBodyController(dog, arm, ik)
    output = controller.step(canonical_command(), _state())

    assert output.arm_observation.shape == (1, 20)
    assert output.dog_observation.shape == (1, 56)
    torch.testing.assert_close(output.arm_observation[0, :12], torch.zeros(12))
    torch.testing.assert_close(output.arm_observation[0, 12:15], torch.tensor([0.5, 0.2, 0.0]))
    torch.testing.assert_close(output.arm_observation[0, 15:18], torch.tensor([0.1, 0.5, 0.0]))
    torch.testing.assert_close(output.body_plan, torch.tensor([[0.3, -0.4]]))
    torch.testing.assert_close(output.dog_observation[0, 42:44], output.body_plan[0])
    torch.testing.assert_close(output.dog_observation[0, 52:56], torch.ones(4))
    torch.testing.assert_close(controller.arm_history[0, :-20], torch.zeros(580))
    torch.testing.assert_close(controller.dog_history[0, :-56], torch.zeros(1624))
    assert len(ik.calls) == 1

    second = controller.step(canonical_command(), _state())
    torch.testing.assert_close(second.arm_observation[0, 6:12], torch.full((6,), 0.1))
    torch.testing.assert_close(
        second.dog_observation[0, 27:39], torch.arange(12, dtype=torch.float32) / 10.0
    )


def test_command_clipping_and_nonfinite_packet_hold() -> None:
    dog, arm = _zero_models()
    controller = WholeBodyController(dog, arm, FakeIK())
    command = torch.tensor([3.0, -2.0, 4.0, 3.0, 3.0, 3.0, 2.0, -2.0, 2.0, 5.0])
    clipped = controller.step(command, _state())
    assert not clipped.command.rejected
    assert clipped.command.clipped_mask.all()
    torch.testing.assert_close(clipped.command.applied[0, :3], torch.tensor([1.5, -0.3, 1.5]))
    assert torch.linalg.vector_norm(clipped.command.applied[0, 3:6]) <= 0.770001
    torch.testing.assert_close(clipped.gripper_joint_target, torch.full((1, 2), 0.044))

    rejected = controller.step(torch.full((10,), float("nan")), _state())
    assert rejected.command.rejected
    torch.testing.assert_close(rejected.command.applied, clipped.command.applied)
    torch.testing.assert_close(rejected.gripper_joint_target, clipped.gripper_joint_target)


def test_shape_error_rejects_before_history_mutation() -> None:
    dog, arm = _zero_models()
    controller = WholeBodyController(dog, arm, FakeIK())
    before = controller.dog_history.clone()
    try:
        controller.step(torch.zeros(9), _state())
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("9-D command was accepted")
    torch.testing.assert_close(controller.dog_history, before)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_019999_roboduet_golden_trace() -> None:
    dog_checkpoint = MODEL_ROOT / "checkpoints_dog" / "ac_weights_019999.pt"
    arm_checkpoint = MODEL_ROOT / "checkpoints_arm" / "ac_weights_019999.pt"
    if not dog_checkpoint.is_file() or not arm_checkpoint.is_file():
        pytest.skip("external RoboDuet 019999 checkpoints are not installed")

    with gzip.open(GOLDEN, "rt", encoding="utf-8") as stream:
        golden = json.load(stream)
    assert golden["schema_version"] == 1
    assert golden["source"]["run_id"] == "dummy-w11894zo_seed5806"
    assert (
        golden["source"]["network_commit"]
        == "fe16a0666648de22ec1e2c57f94ec759b8587553"
    )
    assert golden["source"]["ik_commit"] == "2e86749d86ee4150073197fab7d2e5d56f8c07e7"
    assert _sha256(dog_checkpoint) == golden["source"]["dog_checkpoint_sha256"]
    assert _sha256(arm_checkpoint) == golden["source"]["arm_checkpoint_sha256"]
    assert _sha256(URDF) == golden["source"]["urdf_sha256"]

    dog, arm = load_actor_critics(dog_checkpoint, arm_checkpoint)
    controller = WholeBodyController(dog, arm, Go2X5IK(str(URDF)))
    atol = golden["tolerance"]["absolute"]
    rtol = golden["tolerance"]["relative"]

    def check(actual: torch.Tensor, expected: list[float]) -> None:
        torch.testing.assert_close(
            actual.detach().cpu().reshape(-1),
            torch.tensor(expected, dtype=torch.float32),
            atol=atol,
            rtol=rtol,
        )

    for step in golden["steps"]:
        state = step["state"]
        output = controller.step(
            torch.tensor(step["command"], dtype=torch.float32),
            RobotState(
                projected_gravity=torch.tensor([state["projected_gravity"]]),
                dog_joint_pos=torch.tensor([state["dog_joint_pos"]]),
                dog_joint_vel=torch.tensor([state["dog_joint_vel"]]),
                arm_joint_pos=torch.tensor([state["arm_joint_pos"]]),
                base_roll_pitch=torch.tensor([state["base_roll_pitch"]]),
                base_position_world=torch.tensor([state["base_position_world"]]),
                base_quaternion_xyzw=torch.tensor([state["base_quaternion_xyzw"]]),
                ground_height_world=torch.tensor(state["ground_height_world"]),
            ),
        )
        expected = step["expected"]
        check(output.arm_observation, expected["arm_observation"])
        check(controller.arm_history, expected["arm_history"])
        check(arm.adaptation_module(controller.arm_history), expected["arm_adaptation_latent"])
        check(
            arm.actor_history_encoder(controller.arm_history[:, :-20]),
            expected["arm_history_latent"],
        )
        check(output.learned_arm_action, expected["learned_arm_action"])
        check(output.body_plan, expected["body_plan"])
        check(output.dog_observation, expected["dog_observation"])
        check(controller.dog_history, expected["dog_history"])
        check(dog.adaptation_module(controller.dog_history), expected["dog_adaptation_latent"])
        check(output.dog_action, expected["dog_action"])
        check(output.arm_joint_target, expected["arm_joint_target"])
        check(output.applied_arm_action, expected["applied_arm_action"])
        assert output.ik_status[0].position_error_m == pytest.approx(
            expected["ik_position_error_m"], abs=atol
        )
        assert output.ik_status[0].orientation_error_rad == pytest.approx(
            expected["ik_orientation_error_rad"], abs=atol
        )
