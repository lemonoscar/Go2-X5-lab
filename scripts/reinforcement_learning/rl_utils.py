# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils


class ActionDelayWrapper(gym.Wrapper):
    """Applies action delay/hold noise to simulate control timing uncertainty."""

    def __init__(
        self,
        env: gym.Env,
        delay_steps_range: tuple[int, int] = (0, 0),
        hold_prob: float = 0.0,
        action_noise_std: float = 0.0,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", None)
        self.delay_steps_range = (max(0, int(delay_steps_range[0])), max(0, int(delay_steps_range[1])))
        self.hold_prob = max(0.0, float(hold_prob))
        self.action_noise_std = max(0.0, float(action_noise_std))

        self._buffer = None
        self._buffer_idx = 0
        self._delay_steps = None
        self._last_action = None
        self._pending_reset = True
        self._startup_age = None
        self._startup_steps = round(getattr(env.unwrapped.cfg, "startup_hold_seconds", 0.) / env.unwrapped.step_dt)

    def _init_buffers(self, actions: torch.Tensor):
        if self._buffer is not None:
            return
        if self.num_envs is None:
            self.num_envs = actions.shape[0]
        max_delay = max(self.delay_steps_range)
        buffer_shape = (max_delay + 1, self.num_envs) + actions.shape[1:]
        self._buffer = torch.zeros(buffer_shape, device=actions.device, dtype=actions.dtype)
        self._startup_age = torch.zeros(self.num_envs, device=actions.device, dtype=torch.long)
        self._delay_steps = torch.zeros(self.num_envs, device=actions.device, dtype=torch.long)
        self._last_action = torch.zeros((self.num_envs,) + actions.shape[1:], device=actions.device, dtype=actions.dtype)

    def _sample_delays(self, env_ids: torch.Tensor):
        if self._delay_steps is None:
            return
        if self.delay_steps_range[0] == self.delay_steps_range[1]:
            self._delay_steps[env_ids] = self.delay_steps_range[0]
        else:
            low, high = self.delay_steps_range
            self._delay_steps[env_ids] = torch.randint(
                low=low,
                high=high + 1,
                size=(env_ids.numel(),),
                device=self._delay_steps.device,
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._pending_reset = True
        return obs, info

    def step(self, actions: torch.Tensor):
        if not torch.is_tensor(actions):
            raise ValueError("ActionDelayWrapper expects torch.Tensor actions.")
        self._init_buffers(actions)
        if self._pending_reset:
            env_ids = torch.arange(self.num_envs, device=actions.device)
            self._sample_delays(env_ids)
            self._buffer.zero_()
            self._last_action.zero_()
            self._buffer_idx = 0
            self._startup_age.zero_()
            self._pending_reset = False

        if self.hold_prob > 0.0:
            hold_mask = torch.rand(self.num_envs, device=actions.device) < self.hold_prob
            if hold_mask.any():
                view_shape = (self.num_envs,) + (1,) * (actions.dim() - 1)
                actions = torch.where(hold_mask.view(view_shape), self._last_action, actions)

        if self.action_noise_std > 0.0:
            actions = actions + torch.randn_like(actions) * self.action_noise_std

        write_idx = self._buffer_idx
        self._buffer[write_idx] = actions
        env_ids = torch.arange(self.num_envs, device=actions.device)
        read_idx = (write_idx - self._delay_steps) % self._buffer.shape[0]
        delayed_actions = self._buffer[read_idx, env_ids]
        self._buffer_idx = (write_idx + 1) % self._buffer.shape[0]
        self._last_action = actions

        if self._startup_steps:
            delayed_actions = torch.where((self._startup_age < self._startup_steps)[:, None], 0., delayed_actions)
        self._startup_age += 1
        obs, reward, terminated, truncated, info = self.env.step(delayed_actions)
        done = terminated | truncated
        if not torch.is_tensor(done):
            done = torch.as_tensor(done, device=actions.device)
        if done.any():
            done_ids = torch.where(done)[0]
            self._startup_age[done_ids] = 0
            self._buffer[:, done_ids] = 0.0
            self._last_action[done_ids] = 0.0
            self._sample_delays(done_ids)
        return obs, reward, terminated, truncated, info


def camera_follow(env):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    # camera_pos[2] = torch.clamp(camera_pos[2], min=0.1)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )
