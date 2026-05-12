#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math
import os
import platform
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def _to_float_list(value: Any, expected_len: int) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        seq = value
    elif hasattr(value, "tolist"):
        seq = value.tolist()
    else:
        return None

    out = []
    for i in range(min(len(seq), expected_len)):
        try:
            out.append(float(seq[i]))
        except (TypeError, ValueError):
            return None
    if len(out) < expected_len:
        out.extend([0.0] * (expected_len - len(out)))
    return out


def _prepend_env_path(env_name: str, value: str) -> None:
    if not value:
        return
    current = os.environ.get(env_name, "")
    entries = [p for p in current.split(os.pathsep) if p]
    if value in entries:
        return
    entries.insert(0, value)
    os.environ[env_name] = os.pathsep.join(entries)


def _prepend_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _default_arch_lib_dir(sdk_root: str) -> str:
    machine = platform.machine().lower()
    arch = "aarch64" if ("aarch64" in machine or "arm64" in machine) else "x86_64"
    return os.path.join(sdk_root, "lib", arch)


class ArxBackend:
    """Adapter over different ARX Python SDK APIs."""

    def __init__(
        self,
        model: str,
        interface_name: str,
        urdf_path: str,
        logger,
        enable_background_send_recv: bool,
        controller_dt: float,
        init_to_home: bool,
        require_initial_state: bool,
    ):
        self.logger = logger
        self.backend_name = "none"
        self.arm = None
        self.arm_api = None
        self.joint_count = 6
        self._background_send_recv = enable_background_send_recv
        self._controller_dt = max(0.0, float(controller_dt))
        self._joint_cmd = None
        self._gain = None
        self._last_gain_kp = None
        self._last_gain_kd = None
        self._force_send_recv_once = False
        self._init_backend(
            model=model,
            interface_name=interface_name,
            urdf_path=urdf_path,
            init_to_home=init_to_home,
            require_initial_state=require_initial_state,
        )

    def _init_backend(
        self,
        model: str,
        interface_name: str,
        urdf_path: str,
        init_to_home: bool,
        require_initial_state: bool,
    ) -> None:
        # Preferred backend: real-stanford/arx5-sdk python bindings.
        try:
            import arx5_interface as arx5  # type: ignore

            self.arm_api = arx5
            dof = 6
            try:
                robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
                dof = int(robot_config.joint_dof)
                controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                    "joint_controller", dof
                )
                controller_config.background_send_recv = bool(self._background_send_recv)
                if self._controller_dt > 0.0:
                    controller_config.controller_dt = float(self._controller_dt)
                self.arm = arx5.Arx5JointController(robot_config, controller_config, interface_name)
            except Exception:
                self.arm = arx5.Arx5JointController(model, interface_name)
                self._force_send_recv_once = True
                try:
                    dof = int(self.arm.get_robot_config().joint_dof)
                except Exception:
                    dof = 6

            self.joint_count = dof
            self._joint_cmd = arx5.JointState(self.joint_count)
            self._gain = arx5.Gain(self.joint_count)
            if init_to_home and hasattr(self.arm, "reset_to_home"):
                self.arm.reset_to_home()

            self.backend_name = "arx5_interface"
            self.logger.info(
                f"ARX backend ready: {self.backend_name}, dof={self.joint_count}, background_send_recv={self._background_send_recv}"
            )
            if require_initial_state and not self.connection_ok():
                raise RuntimeError("ARX initial state invalid (all-zero state), check CAN wiring/interface.")
            return
        except Exception as ex:  # pylint: disable=broad-except
            self.logger.warning(f"Load arx5_interface (Arx5JointController) failed: {ex}")

        # Compatibility backend: legacy Arm wrapper (if provided by custom package).
        try:
            from arx5_interface import Arm  # type: ignore

            try:
                self.arm = Arm(model, interface_name, urdf_path) if urdf_path else Arm(model, interface_name)
            except TypeError:
                self.arm = Arm(model, interface_name)
            self.backend_name = "arx5_interface_arm"
            self.logger.info(f"ARX backend ready: {self.backend_name}")
            if require_initial_state and not self.connection_ok():
                raise RuntimeError("ARX initial state invalid for Arm wrapper backend.")
            return
        except Exception as ex:  # pylint: disable=broad-except
            self.logger.warning(f"Load arx5_interface Arm wrapper failed: {ex}")

        # Fallback: custom wrapper package.
        try:
            from arx_x5_sdk import ArxInterface, ArmType  # type: ignore

            arm_type = getattr(ArmType, "X5_2025", None)
            if str(model).upper().startswith("L5"):
                arm_type = getattr(ArmType, "L5_2025", arm_type)
            self.arm = ArxInterface(interface_name, arm_type) if arm_type is not None else ArxInterface(interface_name)
            self.backend_name = "arx_x5_sdk"
            self.logger.info(f"ARX backend ready: {self.backend_name}")
            return
        except Exception as ex:  # pylint: disable=broad-except
            self.logger.warning(f"Load arx_x5_sdk failed: {ex}")

        self.backend_name = "none"
        self.arm = None
        self.logger.error("No ARX Python SDK backend available.")

    def _call_first(self, names: Sequence[str], *args) -> Tuple[bool, Any]:
        if self.arm is None:
            return False, None
        for name in names:
            fn = getattr(self.arm, name, None)
            if not callable(fn):
                continue
            try:
                return True, fn(*args)
            except TypeError:
                continue
            except Exception:
                continue
        return False, None

    def _read_vector(self, method_names: Sequence[str], attr_names: Sequence[str], expected_len: int) -> Optional[List[float]]:
        ok, val = self._call_first(method_names)
        if ok:
            out = _to_float_list(val, expected_len)
            if out is not None:
                return out
        if self.arm is None:
            return None
        for attr in attr_names:
            if not hasattr(self.arm, attr):
                continue
            out = _to_float_list(getattr(self.arm, attr), expected_len)
            if out is not None:
                return out
        return None

    def connection_ok(self) -> bool:
        for _ in range(3):
            if self.backend_name == "arx5_interface" and self.arm is not None and not self._background_send_recv:
                try:
                    if hasattr(self.arm, "send_recv_once"):
                        self.arm.send_recv_once()
                except Exception:
                    pass
            q, dq, _ = self.read_state(self.joint_count)
            if q is None:
                time.sleep(0.02)
                continue
            q_all_zero = all(abs(v) < 1e-8 for v in q)
            if dq is None:
                if not q_all_zero:
                    return True
            else:
                dq_all_zero = all(abs(v) < 1e-8 for v in dq)
                if not (q_all_zero and dq_all_zero):
                    return True
            time.sleep(0.02)
        return False

    def _update_gain_if_needed(self, kp: Optional[Sequence[float]], kd: Optional[Sequence[float]]) -> None:
        if self.backend_name != "arx5_interface" or self.arm is None or self._gain is None:
            return
        if kp is None or kd is None:
            return
        if len(kp) < self.joint_count or len(kd) < self.joint_count:
            return
        kp_local = [float(x) for x in kp[: self.joint_count]]
        kd_local = [float(x) for x in kd[: self.joint_count]]
        if self._last_gain_kp == kp_local and self._last_gain_kd == kd_local:
            return
        try:
            arr_kp = self._gain.kp()
            arr_kd = self._gain.kd()
            for i in range(self.joint_count):
                arr_kp[i] = kp_local[i]
                arr_kd[i] = kd_local[i]
            self.arm.set_gain(self._gain)
            self._last_gain_kp = kp_local
            self._last_gain_kd = kd_local
        except Exception:
            return

    def send_joint_command(
        self,
        q: Sequence[float],
        speed: float,
        dq: Optional[Sequence[float]] = None,
        tau: Optional[Sequence[float]] = None,
        kp: Optional[Sequence[float]] = None,
        kd: Optional[Sequence[float]] = None,
    ) -> bool:
        if self.arm is None:
            return False
        cmd = [float(x) for x in q]

        if self.backend_name == "arx5_interface":
            try:
                if self._joint_cmd is None:
                    self._joint_cmd = self.arm_api.JointState(self.joint_count)  # type: ignore
                self._update_gain_if_needed(kp=kp, kd=kd)
                pos = self._joint_cmd.pos()
                for i in range(min(self.joint_count, len(cmd))):
                    pos[i] = cmd[i]
                if dq is not None and hasattr(self._joint_cmd, "vel"):
                    vel = self._joint_cmd.vel()
                    for i in range(min(self.joint_count, len(dq))):
                        vel[i] = float(dq[i])
                if tau is not None and hasattr(self._joint_cmd, "torque"):
                    torque = self._joint_cmd.torque()
                    for i in range(min(self.joint_count, len(tau))):
                        torque[i] = float(tau[i])
                self.arm.set_joint_cmd(self._joint_cmd)
                if (not self._background_send_recv or self._force_send_recv_once) and hasattr(self.arm, "send_recv_once"):
                    self.arm.send_recv_once()
                return True
            except Exception:
                return False

        ok, _ = self._call_first(("step_joints",), cmd)
        if ok:
            return True
        for args in ((cmd,), (cmd, speed), (cmd, speed, False), (cmd, speed, False, 30.0)):
            ok, _ = self._call_first(("set_joints",), *args)
            if ok:
                return True
        ok, _ = self._call_first(("set_joint_positions", "set_joint_position"), cmd)
        return ok

    def read_state(self, expected_len: int) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
        if self.backend_name == "arx5_interface" and self.arm is not None:
            try:
                state = self.arm.get_joint_state()
                q = _to_float_list(state.pos(), expected_len)
                dq = _to_float_list(state.vel(), expected_len)
                tau = _to_float_list(state.torque(), expected_len)
                return q, dq, tau
            except Exception:
                return None, None, None

        q = self._read_vector(
            ("get_joints", "get_joint_positions", "get_joint_pos", "get_q"),
            ("joints", "joint_positions", "q"),
            expected_len,
        )
        dq = self._read_vector(
            ("get_joint_velocities", "get_joint_vel", "get_dq"),
            ("joint_velocities", "dq"),
            expected_len,
        )
        tau = self._read_vector(
            ("get_joint_torques", "get_joint_tau", "get_tau"),
            ("joint_torques", "tau"),
            expected_len,
        )
        return q, dq, tau

    def stop(self) -> None:
        if self.arm is None:
            return
        if self.backend_name == "arx5_interface":
            try:
                if hasattr(self.arm, "set_to_damping"):
                    self.arm.set_to_damping()
                    return
            except Exception:
                pass
        self._call_first(("stop_current_motor_cmd", "stop", "shutdown"))


class ArxX5BridgeNode(Node):
    def __init__(self):
        super().__init__("arx_x5_bridge")

        self.declare_parameter("model", "X5")
        self.declare_parameter("interface_name", "can0")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("joint_count", 6)
        self.declare_parameter("cmd_topic", "/arx_x5/joint_cmd")
        self.declare_parameter("state_topic", "/arx_x5/joint_state")
        self.declare_parameter("publish_rate_hz", 200.0)
        self.declare_parameter("command_speed", 0.4)
        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("sdk_root", "")
        self.declare_parameter("sdk_python_path", "")
        self.declare_parameter("sdk_lib_path", "")
        self.declare_parameter("require_sdk", True)
        self.declare_parameter("require_initial_state", True)
        self.declare_parameter("enable_background_send_recv", True)
        self.declare_parameter("controller_dt", 0.002)
        self.declare_parameter("init_to_home", False)

        self.model = str(self.get_parameter("model").value)
        self.interface_name = str(self.get_parameter("interface_name").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.joint_count = int(self.get_parameter("joint_count").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.command_speed = float(self.get_parameter("command_speed").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.sdk_root = str(self.get_parameter("sdk_root").value) or os.environ.get("ARX5_SDK_ROOT", "")
        self.sdk_python_path = str(self.get_parameter("sdk_python_path").value)
        self.sdk_lib_path = str(self.get_parameter("sdk_lib_path").value)
        self.require_sdk = bool(self.get_parameter("require_sdk").value)
        self.require_initial_state = bool(self.get_parameter("require_initial_state").value)
        self.enable_background_send_recv = bool(self.get_parameter("enable_background_send_recv").value)
        self.controller_dt = float(self.get_parameter("controller_dt").value)
        self.init_to_home = bool(self.get_parameter("init_to_home").value)

        self._prepare_sdk_environment()

        self.last_q = [0.0] * self.joint_count
        self.last_dq = [0.0] * self.joint_count
        self.last_tau = [0.0] * self.joint_count
        self.last_kp = [0.0] * self.joint_count
        self.last_kd = [0.0] * self.joint_count
        self.last_cmd_stamp = time.monotonic()
        self.recv_cmd = False
        self.last_stale_warn_stamp = 0.0

        self.backend = None
        if self.dry_run:
            self.get_logger().warning("ARX bridge running in dry_run mode.")
        else:
            self.backend = ArxBackend(
                model=self.model,
                interface_name=self.interface_name,
                urdf_path=self.urdf_path,
                logger=self.get_logger(),
                enable_background_send_recv=self.enable_background_send_recv,
                controller_dt=self.controller_dt,
                init_to_home=self.init_to_home,
                require_initial_state=self.require_initial_state,
            )

            if self.backend.backend_name == "none":
                msg = "ARX SDK backend not available."
                if self.require_sdk:
                    raise RuntimeError(msg)
                self.get_logger().warning(msg + " Continue in shadow-only mode.")
                self.backend = None
            else:
                if self.backend.joint_count != self.joint_count:
                    self.get_logger().warning(
                        f"joint_count override: launch={self.joint_count}, sdk={self.backend.joint_count}"
                    )
                    self.joint_count = int(self.backend.joint_count)
                    self.last_q = [0.0] * self.joint_count
                    self.last_dq = [0.0] * self.joint_count
                    self.last_tau = [0.0] * self.joint_count

        self.sub_cmd = self.create_subscription(Float32MultiArray, self.cmd_topic, self._on_cmd, 10)
        self.pub_state = self.create_publisher(Float32MultiArray, self.state_topic, 10)

        hz = self.publish_rate_hz if self.publish_rate_hz > 1e-6 else 200.0
        period = 1.0 / hz
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"ARX bridge started: model={self.model}, iface={self.interface_name}, "
            f"cmd_topic={self.cmd_topic}, state_topic={self.state_topic}, N={self.joint_count}"
        )

    def _prepare_sdk_environment(self) -> None:
        if self.sdk_root:
            if not self.sdk_python_path:
                candidate = os.path.join(self.sdk_root, "python")
                if os.path.isdir(candidate):
                    self.sdk_python_path = candidate
            if not self.sdk_lib_path:
                candidate = _default_arch_lib_dir(self.sdk_root)
                if os.path.isdir(candidate):
                    self.sdk_lib_path = candidate

        if self.sdk_python_path:
            _prepend_sys_path(self.sdk_python_path)
            self.get_logger().info(f"Use ARX python path: {self.sdk_python_path}")
        if self.sdk_lib_path:
            _prepend_env_path("LD_LIBRARY_PATH", self.sdk_lib_path)
            self.get_logger().info(f"Use ARX library path: {self.sdk_lib_path}")

    def _on_cmd(self, msg: Float32MultiArray) -> None:
        n = self.joint_count
        if len(msg.data) < n:
            self.get_logger().warning("Ignore arm cmd: insufficient data length.")
            return

        q = [float(x) for x in msg.data[:n]]
        dq = [0.0] * n
        kp = [0.0] * n
        kd = [0.0] * n
        tau = [0.0] * n
        if len(msg.data) >= 2 * n:
            dq = [float(x) for x in msg.data[n:2 * n]]
        if len(msg.data) >= 3 * n:
            kp = [float(x) for x in msg.data[2 * n:3 * n]]
        if len(msg.data) >= 4 * n:
            kd = [float(x) for x in msg.data[3 * n:4 * n]]
        if len(msg.data) >= 5 * n:
            tau = [float(x) for x in msg.data[4 * n:5 * n]]

        self.last_q = q
        self.last_dq = dq
        self.last_kp = kp
        self.last_kd = kd
        self.last_tau = tau
        self.last_cmd_stamp = time.monotonic()
        self.recv_cmd = True

    def _publish_state(self) -> None:
        msg = Float32MultiArray()
        msg.data = list(self.last_q) + list(self.last_dq) + list(self.last_tau)
        self.pub_state.publish(msg)

    def _on_timer(self) -> None:
        now = time.monotonic()
        stale = (now - self.last_cmd_stamp) > self.cmd_timeout_sec

        if stale and (now - self.last_stale_warn_stamp) > 1.0:
            self.last_stale_warn_stamp = now
            if self.recv_cmd:
                self.get_logger().warning("Arm command timeout, keep last command/state.")

        if self.recv_cmd and not stale and self.backend is not None:
            sent = self.backend.send_joint_command(
                self.last_q,
                self.command_speed,
                dq=self.last_dq,
                tau=self.last_tau,
                kp=self.last_kp,
                kd=self.last_kd,
            )
            if not sent:
                self.get_logger().warning("ARX command send failed; keep shadow state.")

        if self.backend is not None:
            q, dq, tau = self.backend.read_state(self.joint_count)
            if q is not None and len(q) == self.joint_count:
                self.last_q = q
            if dq is not None and len(dq) == self.joint_count:
                self.last_dq = dq
            if tau is not None and len(tau) == self.joint_count:
                self.last_tau = tau

        for i in range(self.joint_count):
            if not math.isfinite(self.last_q[i]):
                self.last_q[i] = 0.0
            if not math.isfinite(self.last_dq[i]):
                self.last_dq[i] = 0.0
            if not math.isfinite(self.last_tau[i]):
                self.last_tau[i] = 0.0

        self._publish_state()

    def destroy_node(self):
        if self.backend is not None:
            self.backend.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArxX5BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
