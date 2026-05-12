from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Go2X5GroundPickSmokeEnvCfg:
    image_size: int = 224
    max_episode_steps: int = 160
    dt: float = 0.1
    target_lift_height: float = 0.16
    success_hold_steps: int = 3
    dog_view_x_limit: float = 1.2
    dog_view_y_limit: float = 0.6
    object_spawn_x_range: tuple[float, float] = (0.42, 0.72)
    object_spawn_y_range: tuple[float, float] = (-0.14, 0.14)


class Go2X5GroundPickSmokeEnv:
    """A lightweight smoke environment that mirrors the Go2-X5 VLA contract.

    This is intentionally not a physics-faithful Isaac environment. It exists to validate the
    end-to-end SimpleVLA-RL pipeline with the same observation and action contract that the
    future Isaac ground-pick task will use.
    """

    requires_isaac_app = False

    _BASE_LOW = np.array([-0.6, -0.4, -0.6], dtype=np.float32)
    _BASE_HIGH = np.array([0.6, 0.4, 0.6], dtype=np.float32)
    _ARM_LOW = np.array([-2.618, 0.0, 0.0, -1.5708, -1.5708, -1.5708], dtype=np.float32)
    _ARM_HIGH = np.array([3.14, 3.14, 3.14, 1.5708, 1.5708, 1.5708], dtype=np.float32)
    _GRIPPER_LOW = 0.0
    _GRIPPER_HIGH = 0.044

    def __init__(
        self,
        task_name: str,
        task_id: int,
        trial_id: int,
        trial_seed: int,
        image_size: int = 224,
        max_episode_steps: int = 160,
        instruction: str = "pick up the red block from the ground",
        config: Any = None,
        **kwargs,
    ) -> None:
        del task_id, kwargs
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config
        self.cfg = Go2X5GroundPickSmokeEnvCfg(image_size=image_size, max_episode_steps=max_episode_steps)
        self.instruction = instruction
        self._rng = np.random.default_rng(trial_seed)
        self._success = False
        self._success_streak = 0
        self._episode_step = 0

        self.base_pos = np.zeros(2, dtype=np.float32)
        self.base_yaw = 0.0
        self.arm_joints = np.zeros(6, dtype=np.float32)
        self.gripper_open = self._GRIPPER_HIGH
        self.object_world = np.zeros(2, dtype=np.float32)
        self.object_height = 0.0
        self.object_grasped = False

    def reset(self):
        self._rng = np.random.default_rng(self.trial_seed + self.trial_id)
        self._success = False
        self._success_streak = 0
        self._episode_step = 0
        self.base_pos[:] = 0.0
        self.base_yaw = 0.0
        self.arm_joints[:] = 0.0
        self.gripper_open = self._GRIPPER_HIGH
        self.object_world[0] = self._rng.uniform(*self.cfg.object_spawn_x_range)
        self.object_world[1] = self._rng.uniform(*self.cfg.object_spawn_y_range)
        self.object_height = 0.0
        self.object_grasped = False
        return self._make_obs()

    def close(self):
        return None

    def get_instruction(self) -> str:
        return self.instruction

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1] != 10:
            raise ValueError(f"Expected 10D action, got shape {action.shape}")

        self._episode_step += 1
        base_cmd = np.clip(action[:3], self._BASE_LOW, self._BASE_HIGH)
        arm_target = np.clip(action[3:9], self._ARM_LOW, self._ARM_HIGH)
        gripper_target = float(np.clip(action[9], self._GRIPPER_LOW, self._GRIPPER_HIGH))

        self.base_pos += base_cmd[:2] * self.cfg.dt
        self.base_yaw += float(base_cmd[2] * self.cfg.dt)
        self.arm_joints = 0.55 * self.arm_joints + 0.45 * arm_target
        self.gripper_open = gripper_target

        eef_xy, eef_z = self._compute_eef_pose()
        object_rel = self.object_world - self.base_pos

        close_in_xy = np.linalg.norm(eef_xy - object_rel) < 0.08
        low_enough = eef_z < 0.09
        closed_gripper = self.gripper_open < 0.012

        if not self.object_grasped and close_in_xy and low_enough and closed_gripper:
            self.object_grasped = True

        if self.object_grasped:
            if self.gripper_open > 0.02:
                self.object_grasped = False
            else:
                self.object_world = self.base_pos + eef_xy
                self.object_height = max(0.0, eef_z - 0.015)

        if not self.object_grasped:
            self.object_height = max(0.0, self.object_height - 0.04)

        lifted = self.object_grasped and self.object_height >= self.cfg.target_lift_height
        if lifted:
            self._success_streak += 1
        else:
            self._success_streak = 0
        self._success = self._success_streak >= self.cfg.success_hold_steps

        terminated = self._success
        truncated = self._episode_step >= self.cfg.max_episode_steps
        reward = 1.0 if self._success else 0.0
        info = {
            "success": self._success,
            "object_height": float(self.object_height),
            "object_grasped": bool(self.object_grasped),
        }
        return self._make_obs(), reward, terminated, truncated, info

    def _compute_eef_pose(self):
        joint1, joint2, joint3, joint4, joint5, joint6 = self.arm_joints
        forward = 0.38 + 0.14 * np.tanh(0.45 * joint2 + 0.35 * joint3)
        lateral = 0.18 * np.tanh(0.6 * joint1 + 0.15 * joint4)
        lift = 0.03 + 0.14 * (0.5 + 0.5 * np.tanh(0.5 * joint2 - 0.35 * joint3 - 0.25 * joint5 + 0.15 * joint6))
        return np.array([forward, lateral], dtype=np.float32), float(lift)

    def _make_obs(self):
        return {
            "dog_camera_image": self._render_dog_camera(),
            "arm_camera_image": self._render_arm_camera(),
        }

    def _render_dog_camera(self):
        size = self.cfg.image_size
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:] = np.array([214, 207, 191], dtype=np.uint8)

        # horizon / floor bands
        image[: size // 3] = np.array([180, 205, 225], dtype=np.uint8)
        image[size // 3 :] = np.array([188, 170, 140], dtype=np.uint8)

        object_rel = self.object_world - self.base_pos
        obj_u = int(size * 0.5 + (object_rel[1] / self.cfg.dog_view_y_limit) * size * 0.28)
        obj_v = int(size * 0.92 - (object_rel[0] / self.cfg.dog_view_x_limit) * size * 0.68)
        self._draw_circle(image, obj_u, obj_v, max(4, size // 28), (215, 48, 48))

        eef_xy, _ = self._compute_eef_pose()
        eef_u = int(size * 0.5 + (eef_xy[1] / self.cfg.dog_view_y_limit) * size * 0.28)
        eef_v = int(size * 0.92 - (eef_xy[0] / self.cfg.dog_view_x_limit) * size * 0.68)
        self._draw_circle(image, eef_u, eef_v, max(4, size // 34), (32, 124, 229))

        # base marker
        self._draw_rect(image, int(size * 0.44), int(size * 0.82), int(size * 0.56), int(size * 0.92), (58, 58, 58))
        if self.object_grasped:
            self._draw_circle(image, eef_u, max(0, eef_v - size // 18), max(4, size // 30), (65, 175, 84))
        return image

    def _render_arm_camera(self):
        size = self.cfg.image_size
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:] = np.array([42, 44, 52], dtype=np.uint8)

        eef_xy, eef_z = self._compute_eef_pose()
        object_rel = self.object_world - self.base_pos
        obj_dx = object_rel[0] - eef_xy[0]
        obj_dy = object_rel[1] - eef_xy[1]
        obj_dz = self.object_height - eef_z

        center_u = int(size * 0.5 + np.clip(obj_dy / 0.3, -1.0, 1.0) * size * 0.22)
        center_v = int(size * 0.62 - np.clip(obj_dz / 0.25, -1.0, 1.0) * size * 0.30)
        obj_radius = max(5, size // 24)

        # depth shading from forward distance
        depth_alpha = float(np.clip(1.0 - abs(obj_dx) / 0.4, 0.25, 1.0))
        object_color = tuple(int(c * depth_alpha) for c in (235, 76, 66))
        self._draw_circle(image, center_u, center_v, obj_radius, object_color)

        gripper_color = (102, 255, 178) if self.gripper_open > 0.02 else (255, 214, 102)
        grip_gap = max(4, int((self.gripper_open / self._GRIPPER_HIGH) * size * 0.035))
        self._draw_rect(image, center_u - obj_radius - 12, center_v - grip_gap - 4, center_u - obj_radius + 2, center_v - grip_gap + 4, gripper_color)
        self._draw_rect(image, center_u + obj_radius - 2, center_v + grip_gap - 4, center_u + obj_radius + 12, center_v + grip_gap + 4, gripper_color)

        # arm frame guide
        self._draw_rect(image, 0, int(size * 0.72), size, size, (30, 32, 38))
        if self.object_grasped:
            self._draw_circle(image, center_u, max(0, center_v - size // 10), max(4, size // 28), (76, 175, 80))
        return image

    @staticmethod
    def _draw_circle(image, center_u, center_v, radius, color):
        h, w = image.shape[:2]
        u = np.arange(w)[None, :]
        v = np.arange(h)[:, None]
        mask = (u - center_u) ** 2 + (v - center_v) ** 2 <= radius ** 2
        image[mask] = np.array(color, dtype=np.uint8)

    @staticmethod
    def _draw_rect(image, u0, v0, u1, v1, color):
        h, w = image.shape[:2]
        u0 = max(0, min(w, u0))
        u1 = max(0, min(w, u1))
        v0 = max(0, min(h, v0))
        v1 = max(0, min(h, v1))
        image[v0:v1, u0:u1] = np.array(color, dtype=np.uint8)
