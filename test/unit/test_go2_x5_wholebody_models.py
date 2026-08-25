from __future__ import annotations

import io
from pathlib import Path
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "robot_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if "robot_lab" not in sys.modules:
    robot_lab = types.ModuleType("robot_lab")
    robot_lab.__path__ = [str(SOURCE_ROOT / "robot_lab")]
    sys.modules["robot_lab"] = robot_lab

from robot_lab.go2_x5_wholebody.models import ArmActorCritic, DogActorCritic  # noqa: E402


def test_checkpoint_shapes_and_full_trainable_paths() -> None:
    dog = DogActorCritic()
    arm = ArmActorCritic()
    assert dog.adaptation_module[0].in_features == 1680
    assert dog.actor_body[0].in_features == 1682
    assert dog.actor_body[-1].out_features == 12
    assert arm.adaptation_module[0].in_features == 600
    assert arm.actor_history_encoder[0].in_features == 580
    assert arm.actor_body[0].in_features == 157
    assert arm.actor_body[-1].out_features == 8

    dog_history = torch.randn(2, 1680, requires_grad=True)
    arm_history = torch.randn(2, 600, requires_grad=True)
    dog_loss = dog.act_student(dog_history).sum() + dog.evaluate(
        dog_history, torch.randn(2, 2)
    ).sum()
    arm_loss = arm.inference_mean(arm_history).sum() + arm.evaluate(
        arm_history, torch.randn(2, 9)
    ).sum()
    (dog_loss + arm_loss).backward()
    assert dog.actor_body[0].weight.grad is not None
    assert dog.critic_body[0].weight.grad is not None
    assert arm.actor_history_encoder[0].weight.grad is not None
    assert arm.critic_body[0].weight.grad is not None


def test_state_dict_roundtrip_preserves_all_keys() -> None:
    for model in (DogActorCritic(), ArmActorCritic()):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        clone = type(model)()
        clone.load_state_dict(state, strict=True)
        assert tuple(clone.state_dict()) == tuple(model.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(clone.state_dict()[key], value)


def test_arm_inference_intentionally_bypasses_training_tanh() -> None:
    arm = ArmActorCritic()
    for parameter in arm.parameters():
        parameter.data.zero_()
    arm.std.data.fill_(0.1)
    arm.actor_body[-1].bias.data[-2:] = torch.tensor([2.0, -2.0])
    history = torch.zeros(1, 600)
    inference = arm.inference_mean(history)
    arm.update_distribution(history)
    assert arm.distribution is not None
    torch.testing.assert_close(inference[0, -2:], torch.tensor([2.0, -2.0]))
    torch.testing.assert_close(
        arm.distribution.mean[0, -2:], torch.tanh(torch.tensor([2.0, -2.0]))
    )
