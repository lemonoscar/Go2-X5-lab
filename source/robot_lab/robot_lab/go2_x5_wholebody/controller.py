"""Simulator-independent 10-D RoboDuet whole-body inference controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .models import ArmActorCritic, DogActorCritic


DOG_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
ARM_JOINT_NAMES = tuple(f"arm_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")

DOG_DEFAULT_JOINT_POS = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)
DOG_ACTION_SCALE = (0.125, 0.25, 0.25) * 4
ARM_POLICY_ZERO = (0.0, 0.8, 0.8, 0.0, 0.0, 0.0)

DOG_FRAME_DIM = 56
ARM_FRAME_DIM = 20
HISTORY_LENGTH = 30
ACTION_CLIP = 10.0
OBSERVATION_CLIP = 100.0
CONTROL_DT = 0.02


def canonical_command(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    length = torch.tensor(0.5, device=device, dtype=dtype)
    polar = torch.tensor(0.2, device=device, dtype=dtype)
    return torch.stack(
        (
            length.new_zeros(()),
            length.new_zeros(()),
            length.new_zeros(()),
            length * torch.cos(polar),
            length.new_zeros(()),
            length * torch.sin(polar),
            length.new_tensor(0.1),
            length.new_tensor(0.5),
            length.new_zeros(()),
            length.new_zeros(()),
        )
    )


@dataclass(frozen=True)
class RobotState:
    """One batched state snapshot, already mapped to RoboDuet joint order."""

    projected_gravity: torch.Tensor
    dog_joint_pos: torch.Tensor
    dog_joint_vel: torch.Tensor
    arm_joint_pos: torch.Tensor
    base_roll_pitch: torch.Tensor
    base_position_world: torch.Tensor
    base_quaternion_xyzw: torch.Tensor
    ground_height_world: torch.Tensor

    @property
    def num_envs(self) -> int:
        return int(self.dog_joint_pos.shape[0])


@dataclass(frozen=True)
class CommandReport:
    original: torch.Tensor
    applied: torch.Tensor
    clipped_mask: torch.Tensor
    rejected: bool
    message: str = ""


@dataclass(frozen=True)
class WholeBodyOutput:
    dog_joint_target: torch.Tensor
    arm_joint_target: torch.Tensor
    gripper_joint_target: torch.Tensor
    dog_action: torch.Tensor
    learned_arm_action: torch.Tensor
    applied_arm_action: torch.Tensor
    body_plan: torch.Tensor
    dog_observation: torch.Tensor
    arm_observation: torch.Tensor
    command: CommandReport
    ik_status: tuple[Any, ...]


class WholeBodyController:
    """Run the Arm actor, body planner, Dog actor and single-env Pink IK in source order."""

    def __init__(
        self,
        dog: DogActorCritic,
        arm: ArmActorCritic,
        ik: Any,
        *,
        num_envs: int = 1,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if ik is not None and num_envs != 1:
            raise ValueError("Pink WholeBody IK currently supports exactly one environment.")
        self.dog = dog
        self.arm = arm
        self.ik = ik
        self.num_envs = num_envs
        parameter = next(dog.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        if next(arm.parameters()).device != self.device:
            raise ValueError("Dog and Arm models must be on the same device.")
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if not hasattr(self, "dog_history"):
            self.dog_history = torch.zeros(
                self.num_envs, DOG_FRAME_DIM * HISTORY_LENGTH, device=self.device, dtype=self.dtype
            )
            self.arm_history = torch.zeros(
                self.num_envs, ARM_FRAME_DIM * HISTORY_LENGTH, device=self.device, dtype=self.dtype
            )
            self.previous_dog_action = torch.zeros(self.num_envs, 12, device=self.device, dtype=self.dtype)
            self.previous_arm_action = torch.zeros(self.num_envs, 6, device=self.device, dtype=self.dtype)
            self.gait_phase = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
            self.last_valid_command = canonical_command(self.device, self.dtype).repeat(self.num_envs, 1)
        self.dog_history[env_ids] = 0.0
        self.arm_history[env_ids] = 0.0
        self.previous_dog_action[env_ids] = 0.0
        self.previous_arm_action[env_ids] = 0.0
        self.gait_phase[env_ids] = 0.0
        self.last_valid_command[env_ids] = canonical_command(self.device, self.dtype)
        if self.ik is not None and hasattr(self.ik, "reset"):
            self.ik.reset()

    @staticmethod
    def _shape(tensor: torch.Tensor, name: str, expected: tuple[int, int]) -> None:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")

    def _state_on_model_device(self, state: RobotState) -> RobotState:
        expected = state.num_envs
        if expected != self.num_envs:
            raise ValueError(f"RobotState contains {expected} envs, controller expects {self.num_envs}.")
        converted = RobotState(
            projected_gravity=state.projected_gravity.to(self.device, self.dtype),
            dog_joint_pos=state.dog_joint_pos.to(self.device, self.dtype),
            dog_joint_vel=state.dog_joint_vel.to(self.device, self.dtype),
            arm_joint_pos=state.arm_joint_pos.to(self.device, self.dtype),
            base_roll_pitch=state.base_roll_pitch.to(self.device, self.dtype),
            base_position_world=state.base_position_world.to(self.device, self.dtype),
            base_quaternion_xyzw=state.base_quaternion_xyzw.to(self.device, self.dtype),
            ground_height_world=state.ground_height_world.to(self.device, self.dtype),
        )
        self._shape(converted.projected_gravity, "projected_gravity", (expected, 3))
        self._shape(converted.dog_joint_pos, "dog_joint_pos", (expected, 12))
        self._shape(converted.dog_joint_vel, "dog_joint_vel", (expected, 12))
        self._shape(converted.arm_joint_pos, "arm_joint_pos", (expected, 6))
        self._shape(converted.base_roll_pitch, "base_roll_pitch", (expected, 2))
        self._shape(converted.base_position_world, "base_position_world", (expected, 3))
        self._shape(converted.base_quaternion_xyzw, "base_quaternion_xyzw", (expected, 4))
        if tuple(converted.ground_height_world.shape) != (expected,):
            raise ValueError(
                "ground_height_world must have shape "
                f"({expected},), got {tuple(converted.ground_height_world.shape)}"
            )
        if not torch.isfinite(converted.ground_height_world).all():
            raise ValueError("ground_height_world contains NaN or Inf")
        return converted

    def _command(self, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, CommandReport]:
        original = torch.as_tensor(command, device=self.device, dtype=self.dtype)
        if original.ndim == 1:
            original = original.unsqueeze(0)
        if tuple(original.shape) != (self.num_envs, 10):
            raise ValueError(
                f"WholeBody command must have shape (10,) or ({self.num_envs}, 10), got {tuple(original.shape)}"
            )
        if not torch.isfinite(original).all():
            applied = self.last_valid_command.clone()
            report = CommandReport(
                original=original.clone(),
                applied=applied.clone(),
                clipped_mask=torch.zeros_like(applied, dtype=torch.bool),
                rejected=True,
                message="NaN/Inf command rejected; holding the last valid command.",
            )
            return applied, self._cartesian_to_lpy(applied[:, 3:6]), report

        applied = original.clone()
        applied[:, 0] = applied[:, 0].clamp(-1.5, 1.5)
        applied[:, 1] = applied[:, 1].clamp(-0.3, 0.3)
        applied[:, 2] = applied[:, 2].clamp(-1.5, 1.5)
        lpy = self._cartesian_to_lpy(applied[:, 3:6])
        lpy[:, 0] = lpy[:, 0].clamp(0.3, 0.77)
        lpy[:, 1] = lpy[:, 1].clamp(-0.45 * torch.pi, 0.45 * torch.pi)
        lpy[:, 2] = lpy[:, 2].clamp(-0.5 * torch.pi, 0.5 * torch.pi)
        applied[:, 3:6] = self._lpy_to_cartesian(lpy)
        applied[:, 6] = applied[:, 6].clamp(-0.45 * torch.pi, 0.45 * torch.pi)
        applied[:, 7] = applied[:, 7].clamp(-torch.pi / 3.0, torch.pi / 3.0)
        applied[:, 8] = applied[:, 8].clamp(-75.0 * torch.pi / 180.0, 75.0 * torch.pi / 180.0)
        applied[:, 9] = applied[:, 9].clamp(0.0, 1.0)
        clipped_mask = ~torch.isclose(applied, original, rtol=0.0, atol=1.0e-7)
        self.last_valid_command.copy_(applied)
        return applied, lpy, CommandReport(
            original=original.clone(),
            applied=applied.clone(),
            clipped_mask=clipped_mask,
            rejected=False,
        )

    @staticmethod
    def _cartesian_to_lpy(position: torch.Tensor) -> torch.Tensor:
        xy = torch.linalg.vector_norm(position[:, :2], dim=-1)
        return torch.stack(
            (
                torch.linalg.vector_norm(position, dim=-1),
                torch.atan2(position[:, 2], xy),
                torch.atan2(position[:, 1], position[:, 0]),
            ),
            dim=-1,
        )

    @staticmethod
    def _lpy_to_cartesian(lpy: torch.Tensor) -> torch.Tensor:
        length, polar, azimuth = lpy.unbind(dim=-1)
        planar = length * torch.cos(polar)
        return torch.stack(
            (planar * torch.cos(azimuth), planar * torch.sin(azimuth), length * torch.sin(polar)),
            dim=-1,
        )

    def _clock(self, velocity_command: torch.Tensor) -> torch.Tensor:
        self.gait_phase.remainder_(1.0).add_(3.0 * CONTROL_DT).remainder_(1.0)
        phase = self.gait_phase
        foot_phase = torch.stack((phase + 0.5, phase, phase, phase + 0.5), dim=-1)
        standing = torch.linalg.vector_norm(velocity_command, dim=-1) < 0.1
        foot_phase[standing] = 0.25
        return torch.sin(2.0 * torch.pi * foot_phase)

    @staticmethod
    def _append(history: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        return torch.cat((history[:, frame.shape[-1] :], frame), dim=-1)

    @torch.no_grad()
    def step(self, command: torch.Tensor, state: RobotState) -> WholeBodyOutput:
        state = self._state_on_model_device(state)
        applied_command, arm_lpy, command_report = self._command(command)
        arm_rpy = applied_command[:, 6:9]

        arm_zero = state.arm_joint_pos.new_tensor(ARM_POLICY_ZERO)
        arm_observation = torch.cat(
            (
                state.arm_joint_pos - arm_zero,
                self.previous_arm_action,
                arm_lpy,
                arm_rpy,
                state.base_roll_pitch,
            ),
            dim=-1,
        ).clamp(-OBSERVATION_CLIP, OBSERVATION_CLIP)
        if arm_observation.shape[-1] != ARM_FRAME_DIM:
            raise RuntimeError(f"Arm observation must be {ARM_FRAME_DIM}D, got {arm_observation.shape}")
        self.arm_history = self._append(self.arm_history, arm_observation)
        learned_arm_action = self.arm.inference_mean(self.arm_history)
        if tuple(learned_arm_action.shape) != (self.num_envs, 8) or not torch.isfinite(learned_arm_action).all():
            raise FloatingPointError(f"Arm actor returned invalid output {tuple(learned_arm_action.shape)}")

        body_plan = learned_arm_action[:, 6:8] * 0.4
        body_plan[:, 0] = body_plan[:, 0].clamp(-0.4, 0.3)
        body_plan[:, 1] = body_plan[:, 1].clamp(-0.4, 0.4)
        dog_command = torch.cat((applied_command[:, :3], body_plan), dim=-1)
        dog_command_scale = dog_command.new_tensor((2.0, 2.0, 0.25, 1.0, 1.0))
        clock = self._clock(applied_command[:, :3])

        dog_zero = state.dog_joint_pos.new_tensor(DOG_DEFAULT_JOINT_POS)
        dog_observation = torch.cat(
            (
                state.projected_gravity,
                state.dog_joint_pos - dog_zero,
                state.dog_joint_vel * 0.05,
                self.previous_dog_action,
                dog_command * dog_command_scale,
                arm_lpy,
                arm_rpy,
                state.base_roll_pitch,
                clock,
            ),
            dim=-1,
        ).clamp(-OBSERVATION_CLIP, OBSERVATION_CLIP)
        if dog_observation.shape[-1] != DOG_FRAME_DIM:
            raise RuntimeError(f"Dog observation must be {DOG_FRAME_DIM}D, got {dog_observation.shape}")
        self.dog_history = self._append(self.dog_history, dog_observation)
        dog_action = self.dog.act_student(self.dog_history)
        if tuple(dog_action.shape) != (self.num_envs, 12) or not torch.isfinite(dog_action).all():
            raise FloatingPointError(f"Dog actor returned invalid output {tuple(dog_action.shape)}")

        arm_targets: list[torch.Tensor] = []
        statuses: list[Any] = []
        for env_index in range(self.num_envs):
            if self.ik is None:
                arm_target = arm_zero + learned_arm_action[env_index, :6].clamp(-ACTION_CLIP, ACTION_CLIP) * 0.5
                status = None
            else:
                target = self.ik.target_in_base(
                    base_position_world=state.base_position_world[env_index].detach().cpu().numpy(),
                    base_quaternion_xyzw=state.base_quaternion_xyzw[env_index].detach().cpu().numpy(),
                    ground_height_world=float(state.ground_height_world[env_index].item()),
                    target_lpy=arm_lpy[env_index].detach().cpu().numpy(),
                    target_rpy=arm_rpy[env_index].detach().cpu().numpy(),
                )
                command_q, status = self.ik.step(
                    state.arm_joint_pos[env_index].detach().cpu().numpy(), target
                )
                arm_target = torch.as_tensor(command_q, device=self.device, dtype=self.dtype)
            if tuple(arm_target.shape) != (6,) or not torch.isfinite(arm_target).all():
                raise FloatingPointError("IK produced an invalid six-joint target.")
            arm_targets.append(arm_target)
            statuses.append(status)

        arm_joint_target = torch.stack(arm_targets)
        applied_arm_action = ((arm_joint_target - arm_zero) / 0.5).clamp(-ACTION_CLIP, ACTION_CLIP)
        arm_joint_target = arm_zero + 0.5 * applied_arm_action
        applied_dog_action = dog_action.clamp(-ACTION_CLIP, ACTION_CLIP)
        dog_scale = dog_action.new_tensor(DOG_ACTION_SCALE)
        dog_joint_target = dog_zero + applied_dog_action * dog_scale
        gripper_joint_target = (0.044 * applied_command[:, 9:10]).repeat(1, 2)

        self.previous_arm_action.copy_(applied_arm_action)
        self.previous_dog_action.copy_(dog_action)
        return WholeBodyOutput(
            dog_joint_target=dog_joint_target,
            arm_joint_target=arm_joint_target,
            gripper_joint_target=gripper_joint_target,
            dog_action=dog_action,
            learned_arm_action=learned_arm_action,
            applied_arm_action=applied_arm_action,
            body_plan=body_plan,
            dog_observation=dog_observation,
            arm_observation=arm_observation,
            command=command_report,
            ik_status=tuple(statuses),
        )
