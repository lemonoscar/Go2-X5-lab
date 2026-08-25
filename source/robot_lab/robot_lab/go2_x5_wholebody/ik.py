"""Optional Pinocchio + Pink IK, copied from the RoboDuet Go2-X5 inference path."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib import metadata
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np


ARM_JOINT_NAMES = tuple(f"arm_joint{index}" for index in range(1, 7))
POLICY_ARM_ZERO = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)
PINNED_DEPENDENCIES = {"pin": "2.7.0", "pin-pink": "3.1.0", "qpsolvers": "4.8.2"}


@dataclass(frozen=True)
class IKStatus:
    solver_ok: bool
    held: bool
    success: bool
    stalled: bool
    position_error_m: float
    orientation_error_rad: float
    command_position_error_m: float
    command_orientation_error_rad: float
    message: str = ""


def _dependencies() -> tuple[Any, Any, Any, Any]:
    actual = {name: metadata.version(name) for name in (*PINNED_DEPENDENCIES, "quadprog")}
    mismatches = {
        name: (expected, actual[name])
        for name, expected in PINNED_DEPENDENCIES.items()
        if actual[name] != expected
    }
    quadprog_version = tuple(int(value) for value in re.findall(r"\d+", actual["quadprog"])[:3])
    if not (quadprog_version >= (0, 1, 12) and quadprog_version < (0, 2)):
        mismatches["quadprog"] = (">=0.1.12,<0.2", actual["quadprog"])
    if mismatches:
        raise ImportError(f"Go2-X5 WholeBody IK dependency mismatch: {mismatches}")
    try:
        pin = import_module("pinocchio")
        if not hasattr(pin, "Model"):
            # Some Isaac workstations also contain the unrelated PyPI
            # ``pinocchio==0.1`` package. The ``pin`` wheel's real bindings live
            # under cmeel.prefix and must win before Pink imports Pinocchio.
            cmeel_paths = [path for path in sys.path if "cmeel.prefix" in path]
            if not cmeel_paths:
                raise ImportError(
                    f"'{getattr(pin, '__file__', 'pinocchio')}' is not the Pinocchio robotics module"
                )
            preferred = cmeel_paths[0]
            sys.path.remove(preferred)
            sys.path.insert(0, preferred)
            for name in tuple(sys.modules):
                if name == "pinocchio" or name.startswith("pinocchio."):
                    del sys.modules[name]
            pin = import_module("pinocchio")
            if not hasattr(pin, "Model"):
                raise ImportError("cmeel Pinocchio bindings do not expose pinocchio.Model")
        pink = import_module("pink")
        tasks = import_module("pink.tasks")
        import_module("qpsolvers")
        import_module("quadprog")
    except ImportError as error:
        raise ImportError(
            "Go2-X5 WholeBody IK requires the optional 'wholebody-ik' dependencies: "
            "pin, pin-pink, qpsolvers and quadprog."
        ) from error
    return pin, pink, tasks.FrameTask, tasks.PostureTask


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"base quaternion must be finite xyzw[4], got {q}")
    norm = np.linalg.norm(q)
    if norm < 1.0e-12:
        raise ValueError("base quaternion has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def arm_dof_indices(dof_names: Iterable[str]) -> tuple[int, ...]:
    names = list(dof_names)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate simulator DOF names: {duplicates}")
    missing = [name for name in ARM_JOINT_NAMES if name not in names]
    if missing:
        raise ValueError(f"missing Go2-X5 arm DOFs: {missing}; simulator has {names}")
    indices = tuple(names.index(name) for name in ARM_JOINT_NAMES)
    if len(set(indices)) != len(ARM_JOINT_NAMES):
        raise ValueError(f"non-unique Go2-X5 arm mapping: {indices}")
    return indices


class Go2X5IK:
    """One-step float64 velocity IK, warm-started from measured arm joints."""

    def __init__(
        self,
        urdf_path: str,
        dt: float = 0.02,
        policy_zero: Sequence[float] = POLICY_ARM_ZERO,
        velocity_limit: float = 3.0,
    ) -> None:
        if dt <= 0.0 or velocity_limit <= 0.0:
            raise ValueError("dt and velocity_limit must be positive")
        self.pin, self.pink, FrameTask, PostureTask = _dependencies()
        self.dt = float(dt)
        self.policy_zero = self._vector(policy_zero, "policy_zero")

        full_model = self.pin.buildModelFromUrdf(urdf_path)
        missing = [name for name in ARM_JOINT_NAMES if not full_model.existJointName(name)]
        if missing:
            raise ValueError(f"URDF is missing required arm joints: {missing}")
        if full_model.getFrameId("arm_eef_link") >= full_model.nframes:
            raise ValueError("URDF is missing official arm_eef_link frame")

        reference = self.pin.neutral(full_model)
        for name, value in zip(ARM_JOINT_NAMES, self.policy_zero, strict=True):
            joint = full_model.joints[full_model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"{name} must be scalar, got nq={joint.nq}, nv={joint.nv}")
            reference[joint.idx_q] = value
        keep = set(ARM_JOINT_NAMES)
        locked = [joint_id for joint_id, name in enumerate(full_model.names) if joint_id and name not in keep]
        self.model = self.pin.buildReducedModel(full_model, locked, reference)
        if tuple(self.model.names[1:]) != ARM_JOINT_NAMES or self.model.nq != 6 or self.model.nv != 6:
            raise ValueError(
                f"reduced IK mapping mismatch: names={list(self.model.names)}, "
                f"nq={self.model.nq}, nv={self.model.nv}"
            )
        if self.model.getFrameId("arm_eef_link") >= self.model.nframes:
            raise ValueError("arm_eef_link disappeared from reduced URDF model")

        self.model.velocityLimit[:] = velocity_limit
        self.data = self.model.createData()
        self.frame_task = FrameTask(
            "arm_eef_link", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0e-6, gain=1.0
        )
        self.posture_task = PostureTask(cost=1.0e-4, lm_damping=1.0e-8, gain=1.0)
        self.posture_task.set_target(self.policy_zero)
        self.tasks = [self.frame_task, self.posture_task]
        self.reset()

    def reset(self) -> None:
        self.last_valid_command: np.ndarray | None = None
        self._target: Any | None = None
        self._best_error = np.inf
        self._steps_without_improvement = 0

    @staticmethod
    def _vector(values: Sequence[float], label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError(f"{label} must be finite float64[6], got {vector}")
        return vector.copy()

    def target_in_base(
        self,
        base_position_world: Sequence[float],
        base_quaternion_xyzw: Sequence[float],
        ground_height_world: float,
        target_lpy: Sequence[float],
        target_rpy: Sequence[float],
    ) -> Any:
        base_position = np.asarray(base_position_world, dtype=np.float64)
        lpy = np.asarray(target_lpy, dtype=np.float64)
        rpy = np.asarray(target_rpy, dtype=np.float64)
        if base_position.shape != (3,) or lpy.shape != (3,) or rpy.shape != (3,):
            raise ValueError("base_position, target_lpy and target_rpy must each have shape (3,)")
        if not np.isfinite(np.concatenate((base_position, lpy, rpy, [ground_height_world]))).all():
            raise ValueError("target transform inputs must be finite")

        rotation_world_base = _quaternion_xyzw_to_rotation(base_quaternion_xyzw)
        base_yaw = np.arctan2(rotation_world_base[1, 0], rotation_world_base[0, 0])
        length, polar, azimuth = lpy
        offset_yaw_frame = np.array(
            [
                length * np.cos(polar) * np.cos(azimuth),
                length * np.cos(polar) * np.sin(azimuth),
                length * np.sin(polar),
            ],
            dtype=np.float64,
        )
        rotation_world_yaw = _rotation_z(base_yaw)
        position_world_target = np.array(
            [base_position[0], base_position[1], float(ground_height_world) + 0.38], dtype=np.float64
        ) + rotation_world_yaw @ offset_yaw_frame
        roll, pitch, yaw = rpy
        rotation_world_target = (
            rotation_world_yaw @ _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
        )
        return self.pin.SE3(
            rotation_world_base.T @ rotation_world_target,
            rotation_world_base.T @ (position_world_target - base_position),
        )

    def forward_kinematics(self, q: Sequence[float]) -> Any:
        q_array = self._vector(q, "q")
        self.pin.forwardKinematics(self.model, self.data, q_array)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.model.getFrameId("arm_eef_link")].copy()

    def pose_error(self, q: Sequence[float], target: Any) -> tuple[float, float]:
        actual = self.forward_kinematics(q)
        position_error = float(np.linalg.norm(actual.translation - target.translation))
        orientation_error = float(np.linalg.norm(self.pin.log3(actual.rotation.T @ target.rotation)))
        return position_error, orientation_error

    def _update_stall_state(
        self, target: Any, position_error: float, orientation_error: float
    ) -> bool:
        if position_error <= 0.002 and orientation_error <= np.deg2rad(1.0):
            self._steps_without_improvement = 0
            return False
        target_changed = self._target is None
        if self._target is not None:
            target_changed = (
                np.linalg.norm(target.translation - self._target.translation) > 0.005
                or np.linalg.norm(self.pin.log3(self._target.rotation.T @ target.rotation)) > np.deg2rad(0.5)
            )
        if target_changed:
            self._target = target.copy()
            self._best_error = np.inf
            self._steps_without_improvement = 0
        combined_error = position_error + 0.1 * orientation_error
        if combined_error < self._best_error - 1.0e-6:
            self._best_error = combined_error
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1
        return self._steps_without_improvement >= max(1, round(0.5 / self.dt))

    def step(self, measured_q: Sequence[float], target: Any) -> tuple[np.ndarray, IKStatus]:
        current_position_error = np.inf
        current_orientation_error = np.inf
        stalled = False
        try:
            measured = self._vector(measured_q, "measured_q")
            if not np.isfinite(target.translation).all() or not np.isfinite(target.rotation).all():
                raise ValueError("IK target contains non-finite values")
            lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
            if np.any(measured < lower - 1.0e-6) or np.any(measured > upper + 1.0e-6):
                raise ValueError(f"measured joints violate position limits: {measured}")
            measured = np.clip(measured, lower, upper)
            if self.last_valid_command is None:
                self.last_valid_command = measured.copy()

            current_position_error, current_orientation_error = self.pose_error(measured, target)
            stalled = self._update_stall_state(target, current_position_error, current_orientation_error)
            configuration = self.pink.Configuration(self.model, self.data, measured.copy())
            self.frame_task.set_target(target)
            velocity = self.pink.solve_ik(
                configuration,
                self.tasks,
                self.dt,
                solver="quadprog",
                damping=1.0e-8,
                safety_break=True,
            )
            command = self.pin.integrate(self.model, measured, velocity * self.dt)
            if not np.isfinite(command).all():
                raise FloatingPointError("IK returned a non-finite joint command")
            if np.any(command < lower - 1.0e-9) or np.any(command > upper + 1.0e-9):
                raise ValueError(f"IK command violates position limits: {command}")
            command = np.clip(command, lower, upper)
            command_position_error, command_orientation_error = self.pose_error(command, target)
            self.last_valid_command = command.copy()
            return command, IKStatus(
                solver_ok=True,
                held=False,
                success=current_position_error <= 0.002
                and current_orientation_error <= np.deg2rad(1.0),
                stalled=stalled,
                position_error_m=current_position_error,
                orientation_error_rad=current_orientation_error,
                command_position_error_m=command_position_error,
                command_orientation_error_rad=command_orientation_error,
            )
        except Exception as error:
            command = self.policy_zero.copy() if self.last_valid_command is None else self.last_valid_command.copy()
            try:
                command_position_error, command_orientation_error = self.pose_error(command, target)
            except Exception:
                command_position_error, command_orientation_error = np.inf, np.inf
            return command, IKStatus(
                solver_ok=False,
                held=True,
                success=current_position_error <= 0.002
                and current_orientation_error <= np.deg2rad(1.0),
                stalled=stalled,
                position_error_m=current_position_error,
                orientation_error_rad=current_orientation_error,
                command_position_error_m=command_position_error,
                command_orientation_error_rad=command_orientation_error,
                message=f"{type(error).__name__}: {error}",
            )
