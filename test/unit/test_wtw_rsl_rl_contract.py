"""Pure-Torch contract tests for WTW RSL-RL checkpoint continuation."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "source"
    / "robot_lab"
    / "robot_lab"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "quadruped"
    / "go2_x5"
    / "agents"
    / "wtw_rsl_rl.py"
)
SOURCE_CHECKPOINT_DIR = (
    REPO_ROOT.parent
    / "walk-these-ways-go2"
    / "runs"
    / "gait-conditioned-agility"
    / "pretrain-go2"
    / "train"
    / "142238.667503"
    / "checkpoints"
)


def _load_wtw_module():
    """Load the module without importing Isaac Sim for its config-only dependencies."""

    class _RunnerCfg:
        pass

    class _AlgorithmCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab_utils = types.ModuleType("isaaclab.utils")
    fake_isaaclab_utils.configclass = lambda cls: cls
    fake_isaaclab.utils = fake_isaaclab_utils
    fake_isaaclab_rl = types.ModuleType("isaaclab_rl")
    fake_isaaclab_rl_rsl_rl = types.ModuleType("isaaclab_rl.rsl_rl")
    fake_isaaclab_rl_rsl_rl.RslRlOnPolicyRunnerCfg = _RunnerCfg
    fake_isaaclab_rl_rsl_rl.RslRlPpoAlgorithmCfg = _AlgorithmCfg
    fake_isaaclab_rl.rsl_rl = fake_isaaclab_rl_rsl_rl

    replacements = {
        "isaaclab": fake_isaaclab,
        "isaaclab.utils": fake_isaaclab_utils,
        "isaaclab_rl": fake_isaaclab_rl,
        "isaaclab_rl.rsl_rl": fake_isaaclab_rl_rsl_rl,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location("_wtw_rsl_rl_under_test", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


wtw = _load_wtw_module()


def _observations(batch_size: int = 4) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, wtw.WTW_OBSERVATION_HISTORY_DIM),
            "critic": torch.randn(batch_size, wtw.WTW_CRITIC_OBSERVATION_DIM),
        },
        batch_size=[batch_size],
    )


def _model(**kwargs):
    return wtw.WTWActorCritic(
        _observations(),
        {"policy": ["policy"], "critic": ["critic"]},
        wtw.WTW_ACTION_DIM,
        **kwargs,
    )


def _synthetic_raw_checkpoint(model, path: Path) -> dict[str, torch.Tensor]:
    checkpoint = {}
    actor_index = 1
    for key, value in model.state_dict().items():
        if key.startswith("adaptation_module.") or key.startswith("actor_body."):
            checkpoint[key] = torch.full_like(value, actor_index / 100.0)
            actor_index += 1
    checkpoint["std"] = torch.full((wtw.WTW_ACTION_DIM,), 0.715)
    checkpoint["critic_body.0.weight"] = torch.randn(512, 2102)
    checkpoint["critic_body.0.bias"] = torch.randn(512)
    torch.save(checkpoint, path)
    return checkpoint


def test_actor_critic_shapes_freeze_flags_and_fresh_std() -> None:
    model = _model(freeze_adaptation=True, freeze_actor_body=False, freeze_std=False)
    critic_only_model = _model(freeze_adaptation=True, freeze_actor_body=True, freeze_std=True)
    obs = _observations(batch_size=4)

    actions = model.act_inference(obs)
    values = model.evaluate(obs)

    assert actions.shape == (4, wtw.WTW_ACTION_DIM)
    assert values.shape == (4, 1)
    torch.testing.assert_close(model.std, torch.full((wtw.WTW_ACTION_DIM,), 0.20))
    assert all(not parameter.requires_grad for parameter in model.adaptation_module.parameters())
    assert all(parameter.requires_grad for parameter in model.actor_body.parameters())
    assert all(parameter.requires_grad for parameter in model.critic.parameters())
    assert model.std.requires_grad
    assert all(not parameter.requires_grad for parameter in critic_only_model.adaptation_module.parameters())
    assert all(not parameter.requires_grad for parameter in critic_only_model.actor_body.parameters())
    assert not critic_only_model.std.requires_grad
    assert all(parameter.requires_grad for parameter in critic_only_model.critic.parameters())


@pytest.mark.parametrize("incorrect_action_dim", (14, 20))
def test_actor_head_rejects_gripper_or_full_manipulator_action_dims(
    incorrect_action_dim: int,
) -> None:
    """The six arm and two gripper targets remain zero-dimensional external actions."""

    with pytest.raises(ValueError, match="WTW actor requires 12 actions"):
        wtw.WTWActorCritic(
            _observations(),
            {"policy": ["policy"], "critic": ["critic"]},
            incorrect_action_dim,
        )


def test_selective_raw_loader_keeps_new_critic_and_resets_std(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = _model(freeze_adaptation=False)
    critic_before = {key: value.clone() for key, value in model.critic.state_dict().items()}
    raw_path = tmp_path / "ac_weights_raw.pt"
    source = _synthetic_raw_checkpoint(model, raw_path)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    model.load_raw_actor_checkpoint(raw_path, expected_sha256=digest)

    for key, value in model.state_dict().items():
        if key.startswith("adaptation_module.") or key.startswith("actor_body."):
            torch.testing.assert_close(value, source[key])
    for key, value in model.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key])
    torch.testing.assert_close(model.std, torch.full_like(model.std, 0.20))


def test_selective_raw_loader_rejects_incomplete_or_unknown_actor_state(tmp_path: Path) -> None:
    model = _model()
    raw_path = tmp_path / "bad_raw.pt"
    source = _synthetic_raw_checkpoint(model, raw_path)
    source.pop("actor_body.6.bias")
    source["encoder.unreviewed"] = torch.zeros(1)
    torch.save(source, raw_path)

    with pytest.raises(ValueError, match="raw actor checkpoint key mismatch"):
        model.load_raw_actor_checkpoint(raw_path)


@pytest.mark.skipif(
    not (SOURCE_CHECKPOINT_DIR / "ac_weights_last.pt").is_file(),
    reason="WTW source checkpoint is not present in this workspace",
)
def test_raw_checkpoint_actor_matches_exported_jit_on_random_history() -> None:
    model = _model(
        raw_checkpoint_path=SOURCE_CHECKPOINT_DIR / "ac_weights_last.pt",
        raw_checkpoint_sha256=wtw.WTW_RAW_CHECKPOINT_SHA256,
    ).eval()
    adaptation_jit = torch.jit.load(str(SOURCE_CHECKPOINT_DIR / "adaptation_module_latest.jit"), map_location="cpu")
    body_jit = torch.jit.load(str(SOURCE_CHECKPOINT_DIR / "body_latest.jit"), map_location="cpu")
    generator = torch.Generator().manual_seed(20260731)
    history = torch.randn(11, wtw.WTW_OBSERVATION_HISTORY_DIM, generator=generator)

    with torch.inference_mode():
        expected_latent = adaptation_jit(history)
        expected_action = body_jit(torch.cat((history, expected_latent), dim=-1))
        actual_latent = model.adaptation_module(history)
        actual_action = model.actor_body(torch.cat((history, actual_latent), dim=-1))

    torch.testing.assert_close(actual_latent, expected_latent, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(actual_action, expected_action, rtol=0.0, atol=1.0e-6)


def test_runner_builds_wtw_policy_and_saves_exportable_keys(tmp_path: Path) -> None:
    template = _model()
    raw_path = tmp_path / "ac_weights_raw.pt"
    _synthetic_raw_checkpoint(template, raw_path)

    class _FakeEnv:
        num_envs = 4
        num_actions = wtw.WTW_ACTION_DIM

        def get_observations(self):
            return _observations(batch_size=self.num_envs)

    train_cfg = {
        "num_steps_per_env": 2,
        "save_interval": 1,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "WTWActorCritic",
            "raw_checkpoint_path": str(raw_path),
            "raw_checkpoint_sha256": None,
            "freeze_adaptation": True,
            "freeze_actor_body": False,
            "freeze_std": False,
            "init_noise_std": 0.20,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.10,
            "entropy_coef": 0.001,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 5.0e-5,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.005,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "normalize_advantage_per_mini_batch": False,
        },
    }
    runner = wtw.WTWOnPolicyRunner(_FakeEnv(), train_cfg, log_dir=None, device="cpu")
    runner.logger_type = "tensorboard"
    checkpoint_path = tmp_path / "model_0.pt"
    runner.save(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"model_state_dict", "optimizer_state_dict", "iter", "infos"}
    model_keys = set(checkpoint["model_state_dict"])
    assert any(key.startswith("adaptation_module.") for key in model_keys)
    assert any(key.startswith("actor_body.") for key in model_keys)
    assert any(key.startswith("critic.") for key in model_keys)
    assert "std" in model_keys
    assert not any(key.startswith("critic_body.") for key in model_keys)
    assert wtw.Go2X5WtwPD40PPORunnerCfg.class_name == "WTWOnPolicyRunner"
    assert wtw.Go2X5WtwPD40PPORunnerCfg.init_at_random_ep_len is False
