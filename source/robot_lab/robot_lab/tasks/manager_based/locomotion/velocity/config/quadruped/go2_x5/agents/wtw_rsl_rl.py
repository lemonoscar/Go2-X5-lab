# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL policy and runner for WTW checkpoint continuation on Go2-X5."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner


WTW_OBSERVATION_HISTORY_DIM = 2100
WTW_LATENT_DIM = 2
WTW_ACTION_DIM = 12
WTW_CRITIC_OBSERVATION_DIM = 260
WTW_RAW_CHECKPOINT_SHA256 = "1f4218009a9d269ffb54b9034b6a488b09062fda6d1115cd4ac7943a70a81c43"
WTW_RAW_CHECKPOINT_PATH = (
    "../walk-these-ways-go2/runs/gait-conditioned-agility/pretrain-go2/train/"
    "142238.667503/checkpoints/ac_weights_last.pt"
)


def _make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ELU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _group_dim(obs: TensorDict, obs_groups: dict[str, list[str]], group_name: str) -> int:
    dimension = 0
    for observation_name in obs_groups[group_name]:
        observation = obs[observation_name]
        if observation.ndim != 2:
            raise ValueError(
                f"WTW {group_name} observation '{observation_name}' must be 2-D, got {tuple(observation.shape)}"
            )
        dimension += observation.shape[-1]
    return dimension


class WTWActorCritic(nn.Module):
    """WTW adaptation/body actor with a new full-state Go2-X5 critic."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        raw_checkpoint_path: str | Path | None = None,
        raw_checkpoint_sha256: str | None = None,
        freeze_adaptation: bool = True,
        freeze_actor_body: bool = False,
        freeze_std: bool = False,
        init_noise_std: float = 0.20,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        adaptation_hidden_dims: list[int] = [256, 128],
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected WTWActorCritic arguments: {sorted(kwargs)}")
        if num_actions != WTW_ACTION_DIM:
            raise ValueError(f"WTW actor requires {WTW_ACTION_DIM} actions, got {num_actions}")
        if _group_dim(obs, obs_groups, "policy") != WTW_OBSERVATION_HISTORY_DIM:
            raise ValueError(f"WTW actor requires a {WTW_OBSERVATION_HISTORY_DIM}-D policy observation")
        if _group_dim(obs, obs_groups, "critic") != WTW_CRITIC_OBSERVATION_DIM:
            raise ValueError(f"WTW critic requires a {WTW_CRITIC_OBSERVATION_DIM}-D critic observation")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("WTW continuation requires actor and critic observation normalization to be disabled")
        if adaptation_hidden_dims != [256, 128] or actor_hidden_dims != [512, 256, 128]:
            raise ValueError("WTW adaptation/body dimensions must preserve the source checkpoint ABI")
        if activation.lower() != "elu":
            raise ValueError("WTW adaptation/body activation must be ELU")
        if noise_std_type != "scalar" or state_dependent_std:
            raise ValueError("WTW continuation supports only state-independent scalar action noise")
        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be positive, got {init_noise_std}")

        self.obs_groups = obs_groups
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()

        self.adaptation_module = _make_mlp(
            WTW_OBSERVATION_HISTORY_DIM, adaptation_hidden_dims, WTW_LATENT_DIM
        )
        self.actor_body = _make_mlp(
            WTW_OBSERVATION_HISTORY_DIM + WTW_LATENT_DIM, actor_hidden_dims, WTW_ACTION_DIM
        )
        self.critic = _make_mlp(WTW_CRITIC_OBSERVATION_DIM, critic_hidden_dims, 1)
        self.std = nn.Parameter(torch.full((WTW_ACTION_DIM,), float(init_noise_std)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        if raw_checkpoint_path is not None:
            self.load_raw_actor_checkpoint(raw_checkpoint_path, expected_sha256=raw_checkpoint_sha256)

        self.set_freeze_flags(
            adaptation=freeze_adaptation,
            actor_body=freeze_actor_body,
            action_std=freeze_std,
        )

    def load_raw_actor_checkpoint(
        self, checkpoint_path: str | Path, *, expected_sha256: str | None = None
    ) -> None:
        """Strictly migrate only the deployable actor from a raw WTW checkpoint."""
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"WTW raw checkpoint not found: {checkpoint_path}")
        if expected_sha256 is not None:
            actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"WTW raw checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
                )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"WTW raw checkpoint must be a state dict, got {type(checkpoint).__name__}")

        target_state = self.state_dict()
        actor_keys = {
            key
            for key in target_state
            if key.startswith("adaptation_module.") or key.startswith("actor_body.")
        }
        missing_keys = actor_keys.difference(checkpoint)
        ignored_keys = {
            key for key in checkpoint if key == "std" or key.startswith("critic_body.")
        }
        unexpected_keys = set(checkpoint).difference(actor_keys).difference(ignored_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "WTW raw actor checkpoint key mismatch: "
                f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
            )

        adaptation_state = {
            key.removeprefix("adaptation_module."): checkpoint[key]
            for key in actor_keys
            if key.startswith("adaptation_module.")
        }
        actor_body_state = {
            key.removeprefix("actor_body."): checkpoint[key]
            for key in actor_keys
            if key.startswith("actor_body.")
        }
        self.adaptation_module.load_state_dict(adaptation_state, strict=True)
        self.actor_body.load_state_dict(actor_body_state, strict=True)

    def set_freeze_flags(self, *, adaptation: bool, actor_body: bool, action_std: bool) -> None:
        for parameter in self.adaptation_module.parameters():
            parameter.requires_grad_(not adaptation)
        for parameter in self.actor_body.parameters():
            parameter.requires_grad_(not actor_body)
        self.std.requires_grad_(not action_std)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> None:
        raise NotImplementedError

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["critic"]], dim=-1)

    def _actor_mean(self, observation_history: torch.Tensor) -> torch.Tensor:
        latent = self.adaptation_module(observation_history)
        return self.actor_body(torch.cat((observation_history, latent), dim=-1))

    def _update_distribution(self, observation_history: torch.Tensor) -> None:
        mean = self._actor_mean(observation_history)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            return self.std
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        del kwargs
        observation_history = self.get_actor_obs(obs)
        self._update_distribution(observation_history)
        return self.distribution.sample()  # type: ignore[union-attr]

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self._actor_mean(self.get_actor_obs(obs))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        del kwargs
        return self.critic(self.get_critic_obs(obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


class WTWOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that constructs the checkpoint-compatible WTW actor."""

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name != "WTWActorCritic":
            raise ValueError(f"WTWOnPolicyRunner requires WTWActorCritic, got {policy_class_name}")
        actor_critic = WTWActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        if algorithm_class_name != "PPO":
            raise ValueError(f"WTWOnPolicyRunner requires PPO, got {algorithm_class_name}")
        algorithm = PPO(
            actor_critic,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm


@configclass
class WTWActorCriticCfg:
    class_name: str = "WTWActorCritic"
    raw_checkpoint_path: str = WTW_RAW_CHECKPOINT_PATH
    raw_checkpoint_sha256: str = WTW_RAW_CHECKPOINT_SHA256
    freeze_adaptation: bool = True
    freeze_actor_body: bool = False
    freeze_std: bool = False
    init_noise_std: float = 0.20
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    adaptation_hidden_dims: list[int] = [256, 128]
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False


@configclass
class Go2X5WtwPD40PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "WTWOnPolicyRunner"
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 100
    experiment_name = "go2_x5_wtw_pd40_r0"
    init_at_random_ep_len = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 10.0
    policy = WTWActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
