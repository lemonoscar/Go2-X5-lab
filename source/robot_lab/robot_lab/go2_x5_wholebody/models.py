"""RoboDuet Dog/Arm actor-critic definitions with checkpoint-compatible keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(dimensions: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class DogActorCritic(nn.Module):
    """Exact trainable architecture used by the RoboDuet Go2-X5 Dog checkpoint."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int = 56,
        num_privileged_obs: int = 2,
        num_obs_history: int = 1680,
        num_actions: int = 12,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        self.adaptation_module = _mlp((num_obs_history, 256, 128, num_privileged_obs))
        self.actor_body = _mlp((num_obs_history + num_privileged_obs, 512, 256, 128, num_actions))
        self.critic_body = _mlp((num_obs_history + num_privileged_obs, 512, 256, 128, 1))
        self.std = nn.Parameter(torch.ones(num_actions))
        self.distribution: Normal | None = None

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.act_student(observation_history)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Dog action distribution has not been initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Dog action distribution has not been initialized.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Dog action distribution has not been initialized.")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_distribution(self, observation_history: torch.Tensor) -> None:
        mean = self.act_student(observation_history)
        self.distribution = Normal(mean, mean * 0.0 + self.std, validate_args=False)

    def act(self, observation_history: torch.Tensor, **_: Any) -> torch.Tensor:
        self.update_distribution(observation_history)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Dog action distribution has not been initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, observation_history: torch.Tensor) -> torch.Tensor:
        latent = self.adaptation_module(observation_history)
        return self.actor_body(torch.cat((observation_history, latent), dim=-1))

    def act_teacher(
        self, observation_history: torch.Tensor, privileged_info: torch.Tensor
    ) -> torch.Tensor:
        return self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))

    def evaluate(
        self, observation_history: torch.Tensor, privileged_observations: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        return self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))

    def get_student_latent(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.adaptation_module(observation_history)


class ArmActorCritic(nn.Module):
    """Exact trainable architecture used by the RoboDuet Go2-X5 Arm checkpoint."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int = 20,
        num_privileged_obs: int = 9,
        num_obs_history: int = 600,
        num_actions: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        self.adaptation_module = _mlp((num_obs_history, 256, 128, num_privileged_obs))
        old_history_dim = num_obs_history - num_obs
        self.actor_history_encoder = _mlp((old_history_dim, 512, 256, 128))
        self.actor_body = _mlp((num_obs + num_privileged_obs + 128, 512, 256, 128, num_actions))
        self.critic_history_encoder = _mlp((old_history_dim, 512, 256, 128))
        self.critic_body = _mlp((num_obs + num_privileged_obs + 128, 512, 256, 128, 1))
        self.std = nn.Parameter(0.1 * torch.ones(num_actions))
        self.distribution: Normal | None = None

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.inference_mean(observation_history)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Arm action distribution has not been initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Arm action distribution has not been initialized.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Arm action distribution has not been initialized.")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def inference_mean(self, observation_history: torch.Tensor) -> torch.Tensor:
        """Reproduce ``scripts/load_policy.py``: history encoder, with no final tanh."""
        current = observation_history[..., -self.num_obs :]
        latent = self.adaptation_module(observation_history)
        history_latent = self.actor_history_encoder(observation_history[..., : -self.num_obs])
        return self.actor_body(torch.cat((current, latent, history_latent), dim=-1))

    def update_distribution(self, observation_history: torch.Tensor) -> None:
        """Preserve the training path, where the final two body-plan values use tanh."""
        mean = self.inference_mean(observation_history)
        mean = torch.cat((mean[..., :-2], torch.tanh(mean[..., -2:])), dim=-1)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observation_history: torch.Tensor, **_: Any) -> torch.Tensor:
        self.update_distribution(observation_history)
        assert self.distribution is not None
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Arm action distribution has not been initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.inference_mean(observation_history)

    def act_teacher(
        self, observation_history: torch.Tensor, privileged_info: torch.Tensor
    ) -> torch.Tensor:
        current = observation_history[..., -self.num_obs :]
        history_latent = self.actor_history_encoder(observation_history[..., : -self.num_obs])
        return self.actor_body(torch.cat((current, privileged_info, history_latent), dim=-1))

    def evaluate(
        self, observation_history: torch.Tensor, privileged_observations: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        current = observation_history[..., -self.num_obs :]
        history_latent = self.critic_history_encoder(observation_history[..., : -self.num_obs])
        return self.critic_body(torch.cat((current, privileged_observations, history_latent), dim=-1))

    def get_student_latent(self, observation_history: torch.Tensor) -> torch.Tensor:
        return self.adaptation_module(observation_history)


def _state_dict(path: str | Path, device: torch.device | str) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"WholeBody checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility for the original raw state dict.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError(f"Expected a non-empty raw state_dict in {checkpoint_path}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise TypeError(f"Checkpoint is not a raw tensor state_dict: {checkpoint_path}")
    return checkpoint


def load_actor_critics(
    dog_checkpoint: str | Path,
    arm_checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[DogActorCritic, ArmActorCritic]:
    """Load both full raw state dicts without dropping critic or training parameters."""
    dog = DogActorCritic().to(device)
    arm = ArmActorCritic().to(device)
    dog.load_state_dict(_state_dict(dog_checkpoint, device), strict=True)
    arm.load_state_dict(_state_dict(arm_checkpoint, device), strict=True)
    dog.eval()
    arm.eval()
    return dog, arm
