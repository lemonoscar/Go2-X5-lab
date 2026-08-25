"""Thin Isaac Lab ActionTerm for the 10-D Go2-X5 WholeBody controller."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import time
from typing import Any

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

from robot_lab.assets.go2_x5 import GO2_X5_URDF_PATH
from robot_lab.go2_x5_wholebody.controller import (
    ARM_JOINT_NAMES,
    DOG_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    RobotState,
    WholeBodyController,
    canonical_command,
)
from robot_lab.go2_x5_wholebody.ik import Go2X5IK
from robot_lab.go2_x5_wholebody.manifest import load_manifest, resolve_artifacts
from robot_lab.go2_x5_wholebody.models import load_actor_critics


_REPO_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[5]
    / "data"
    / "Policies"
    / "go2_x5_wholebody"
    / "019999.yaml"
)
DEFAULT_MODEL_ROOT = _REPO_ROOT / "models" / "go2_x5_wholebody" / "019999"


def _unique_joint_ids(asset: Articulation, names: tuple[str, ...]) -> list[int]:
    ids, resolved = asset.find_joints(list(names), preserve_order=True)
    if tuple(resolved) != names or len(set(ids)) != len(names):
        raise ValueError(f"Go2-X5 joint mapping mismatch: requested={names}, resolved={resolved}, ids={ids}")
    return list(ids)


def _exclude_overlapping_finger_pair() -> None:
    """Disable only the known arm_link7/arm_link8 self-collision pair."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not root.IsValid():
        raise RuntimeError("WholeBody collision setup cannot find /World/envs/env_0/Robot")
    links = {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.GetName() in {"arm_link7", "arm_link8"}}
    if set(links) != {"arm_link7", "arm_link8"}:
        raise RuntimeError(f"WholeBody collision setup found links {sorted(links)}")
    relationship = UsdPhysics.FilteredPairsAPI.Apply(links["arm_link7"]).CreateFilteredPairsRel()
    relationship.AddTarget(links["arm_link8"].GetPath())


class Go2X5WholeBodyAction(ActionTerm):
    """Map one 10-D Cartesian command to 12 leg, 6 arm and 2 gripper targets."""

    cfg: "Go2X5WholeBodyActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "Go2X5WholeBodyActionCfg", env: Any) -> None:
        super().__init__(cfg, env)
        if self.num_envs != 1:
            raise ValueError("RobotLab-Isaac-Go2-X5-WholeBody-v0 requires num_envs=1 for Pink IK.")
        if abs(float(env.step_dt) - 0.02) > 1.0e-9:
            raise ValueError(f"WholeBody controller period must be 0.02 s, got {env.step_dt}")

        self._dog_ids = _unique_joint_ids(self._asset, DOG_JOINT_NAMES)
        self._arm_ids = _unique_joint_ids(self._asset, ARM_JOINT_NAMES)
        self._gripper_ids = _unique_joint_ids(self._asset, GRIPPER_JOINT_NAMES)
        self._joint_ids = self._dog_ids + self._arm_ids + self._gripper_ids
        _exclude_overlapping_finger_pair()
        self._contact_sensor = env.scene["contact_forces"]
        self._nonfoot_body_ids = [
            index
            for index, name in enumerate(self._contact_sensor.body_names)
            if not name.endswith("_foot")
        ]
        if not self._nonfoot_body_ids:
            raise ValueError("WholeBody contact diagnostics found no non-foot bodies")

        manifest = load_manifest(cfg.manifest_path)
        artifacts = resolve_artifacts(manifest, cfg.model_root, cfg.urdf_path)
        dog, arm = load_actor_critics(
            artifacts.dog_checkpoint, artifacts.arm_checkpoint, device=cfg.model_device
        )
        self._controller = WholeBodyController(
            dog,
            arm,
            Go2X5IK(str(artifacts.urdf), dt=0.02),
            num_envs=self.num_envs,
        )
        self._validate_articulation(manifest)

        reset_command = canonical_command(self.device).repeat(self.num_envs, 1)
        self._raw_actions = reset_command.clone()
        self._processed_actions = reset_command.clone()
        default_targets = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._joint_targets = default_targets.clone()
        self._last_output = None
        self._diagnostics: dict[str, Any] = {}

    @property
    def action_dim(self) -> int:
        return 10

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics

    def _validate_articulation(self, manifest: dict[str, Any]) -> None:
        expected_limits = manifest["runtime"]["joint_limits"]
        checks = (
            (
                "position",
                self._asset.data.joint_pos_limits[0, self._joint_ids],
                torch.tensor(
                    list(zip(expected_limits["lower"], expected_limits["upper"], strict=True)),
                    device=self.device,
                ),
                1.0e-5,
            ),
            (
                "effort",
                self._asset.data.joint_effort_limits[0, self._joint_ids],
                torch.tensor(expected_limits["effort"], device=self.device),
                1.0e-3,
            ),
            (
                "velocity",
                self._asset.data.joint_vel_limits[0, self._joint_ids],
                torch.tensor(expected_limits["velocity"], device=self.device),
                1.0e-3,
            ),
        )
        for label, actual, expected, atol in checks:
            expected = expected.to(dtype=actual.dtype)
            if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
                raise ValueError(
                    f"WholeBody live {label} limits mismatch: "
                    f"expected {expected.tolist()}, got {actual.tolist()}"
                )

        pd = manifest["runtime"]["pd"]
        expected_stiffness = torch.tensor(
            [pd["legs"]["stiffness"]] * 12
            + [pd["arm_joint1"]["stiffness"]]
            + [pd["arm_joint2_3"]["stiffness"]] * 2
            + [pd["arm_joint4_6"]["stiffness"]] * 3
            + [pd["gripper"]["stiffness"]] * 2,
            device=self.device,
        )
        expected_damping = torch.tensor(
            [pd["legs"]["damping"]] * 12
            + [pd["arm_joint1"]["damping"]]
            + [pd["arm_joint2_3"]["damping"]] * 2
            + [pd["arm_joint4_6"]["damping"]] * 3
            + [pd["gripper"]["damping"]] * 2,
            device=self.device,
        )
        expected_friction = torch.tensor(
            [0.0] * 18 + [pd["gripper"]["friction"]] * 2,
            device=self.device,
        )
        actuator_checks = (
            (
                "stiffness",
                self._asset.data.default_joint_stiffness[0, self._joint_ids],
                expected_stiffness,
            ),
            (
                "damping",
                self._asset.data.default_joint_damping[0, self._joint_ids],
                expected_damping,
            ),
            (
                "friction",
                self._asset.data.joint_friction_coeff[0, self._joint_ids],
                expected_friction,
            ),
        )
        for label, actual, expected in actuator_checks:
            expected = expected.to(dtype=actual.dtype)
            if not torch.allclose(actual, expected, atol=1.0e-5, rtol=0.0):
                raise ValueError(
                    f"WholeBody live {label} mismatch: expected {expected.tolist()}, got {actual.tolist()}"
                )
        masses = self._asset.root_physx_view.get_masses().to(self.device)[0]
        total_mass = float(masses.sum().item())
        expected_mass = manifest["runtime"]["asset"]["total_mass_kg"]
        tolerance = manifest["runtime"]["asset"].get("mass_tolerance_kg", 0.01)
        if abs(total_mass - expected_mass) > tolerance:
            raise ValueError(
                f"WholeBody live mass mismatch: expected {expected_mass}±{tolerance} kg, got {total_mass} kg"
            )

        local_com = self._asset.root_physx_view.get_coms().to(self.device)[0, :, :3]
        body_pos_w = self._asset.data.body_pos_w[0]
        body_quat_w = self._asset.data.body_quat_w[0]
        if not (len(masses) == len(local_com) == len(body_pos_w) == len(self._asset.body_names)):
            raise RuntimeError("WholeBody PhysX mass/COM tensors do not match the articulation body count")
        body_com_w = body_pos_w + quat_apply(body_quat_w, local_com)
        whole_com_w = (body_com_w * masses.unsqueeze(-1)).sum(dim=0) / masses.sum()
        whole_com_b = quat_apply_inverse(
            self._asset.data.root_quat_w[0],
            whole_com_w - self._asset.data.root_pos_w[0],
        )
        expected_com = whole_com_b.new_tensor(manifest["runtime"]["asset"]["whole_body_com_base_m"])
        com_tolerance = float(manifest["runtime"]["asset"].get("com_tolerance_m", 0.005))
        com_error = torch.linalg.vector_norm(whole_com_b - expected_com).item()
        if com_error > com_tolerance:
            raise ValueError(
                "WholeBody live COM mismatch: "
                f"expected {expected_com.tolist()}±{com_tolerance} m, "
                f"got {whole_com_b.tolist()} (error {com_error:.6f} m)"
            )

    def _robot_state(self) -> RobotState:
        root_quat_wxyz = self._asset.data.root_quat_w
        roll, pitch, _ = euler_xyz_from_quat(root_quat_wxyz)
        root_quat_xyzw = root_quat_wxyz[:, (1, 2, 3, 0)]
        return RobotState(
            projected_gravity=self._asset.data.projected_gravity_b,
            dog_joint_pos=self._asset.data.joint_pos[:, self._dog_ids],
            dog_joint_vel=self._asset.data.joint_vel[:, self._dog_ids],
            arm_joint_pos=self._asset.data.joint_pos[:, self._arm_ids],
            base_roll_pitch=torch.stack((roll, pitch), dim=-1),
            base_position_world=self._asset.data.root_pos_w,
            base_quaternion_xyzw=root_quat_xyzw,
            ground_height_world=self._env.scene.env_origins[:, 2],
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        started = time.perf_counter()
        output = self._controller.step(actions, self._robot_state())
        controller_time_ms = 1000.0 * (time.perf_counter() - started)
        self._processed_actions.copy_(output.command.applied.to(self.device))
        self._joint_targets = torch.cat(
            (
                output.dog_joint_target,
                output.arm_joint_target,
                output.gripper_joint_target,
            ),
            dim=-1,
        ).to(self.device)
        if not torch.isfinite(self._joint_targets).all():
            raise FloatingPointError("WholeBody produced non-finite articulation targets.")
        status = output.ik_status[0]
        projected_gravity = self._asset.data.projected_gravity_b[0]
        fallen = bool(self._asset.data.root_pos_w[0, 2] < 0.28 or projected_gravity[2] > -0.5)
        max_nonfoot_contact_n = float(
            torch.linalg.vector_norm(
                self._contact_sensor.data.net_forces_w[0, self._nonfoot_body_ids], dim=-1
            ).max().item()
        )
        self._diagnostics = {
            "original_command": output.command.original.to(self.device),
            "applied_command": output.command.applied.to(self.device),
            "command_rejected": output.command.rejected,
            "command_message": output.command.message,
            "clipped_mask": output.command.clipped_mask.to(self.device),
            "ik_hold": bool(status.held),
            "stalled": bool(status.stalled),
            "ik_position_error_m": float(status.position_error_m),
            "ik_orientation_error_rad": float(status.orientation_error_rad),
            "ik_command_position_error_m": float(status.command_position_error_m),
            "ik_command_orientation_error_rad": float(status.command_orientation_error_rad),
            "fallen": fallen,
            "contact": max_nonfoot_contact_n >= 25.0,
            "max_nonfoot_contact_n": max_nonfoot_contact_n,
            "controller_time_ms": controller_time_ms,
        }
        self._last_output = output

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        reset_ids = [] if env_ids is None else [int(value) for value in env_ids]
        if reset_ids not in ([0], []):
            raise ValueError(f"WholeBody IK only supports resetting env 0, got {reset_ids}")
        self._controller.reset()
        reset_command = canonical_command(self.device)
        self._raw_actions[:] = reset_command
        self._processed_actions[:] = reset_command
        self._joint_targets[:] = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._diagnostics = {}


@configclass
class Go2X5WholeBodyActionCfg(ActionTermCfg):
    class_type: type = Go2X5WholeBodyAction
    asset_name: str = "robot"
    manifest_path: str = str(DEFAULT_MANIFEST)
    model_root: str = os.environ.get("GO2_X5_WHOLEBODY_MODEL_DIR", str(DEFAULT_MODEL_ROOT))
    urdf_path: str = GO2_X5_URDF_PATH
    model_device: str = "cpu"
