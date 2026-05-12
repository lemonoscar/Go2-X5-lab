# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reset and randomization event helpers for Go2-X5 tabletop prototypes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_staged_task_state(env: ManagerBasedEnv, env_ids: torch.Tensor | None, num_stages: int = 4) -> None:
    """Initialize lightweight stage buffers used by the prototype observation terms."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_go2_x5_stage"):
        env._go2_x5_stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._go2_x5_time_in_stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._go2_x5_num_stages = int(num_stages)
    env._go2_x5_stage[env_ids] = 0
    env._go2_x5_time_in_stage[env_ids] = 0.0


def reset_from_stage_snapshot_placeholder(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Placeholder for future reset-from-later-stage snapshot loading."""

    del env, env_ids
