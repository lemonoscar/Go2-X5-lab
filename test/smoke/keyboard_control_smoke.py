#!/usr/bin/env python3
"""
Go2-X5 键盘控制 + 摄像头显示 (Smoke 版本 - 无需 Isaac Lab)

这个版本使用轻量级的 smoke 环境，不需要 Isaac Sim/Isaac Lab。
适合本地测试和开发。

使用方法:
    cd ~/Issac/Go2-X5-lab
    python test/smoke/keyboard_control_smoke.py

键盘控制:
    [底座 - WASD] W/S=前进/后退 A/D=左转/右转 Q/E=左移/右移
    [机械臂] I/K/J/L/U/O 控制关节 SPACE=切换关节组
    [夹爪] 1=打开 2=闭合
    [其他] R=重置 P=暂停 H=帮助 ESC=退出
"""

import argparse
from collections import deque
import sys
import time
from pathlib import Path

import numpy as np
import cv2

# Add robot_lab to path
REPO_ROOT = Path(__file__).resolve().parents[2]
robot_lab_path = REPO_ROOT / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))

# Import smoke environment
try:
    from .support.go2_x5_ground_pick_env import Go2X5GroundPickSmokeEnv, Go2X5GroundPickSmokeEnvCfg
except ImportError:
    from support.go2_x5_ground_pick_env import Go2X5GroundPickSmokeEnv, Go2X5GroundPickSmokeEnvCfg


class PlaceholderImageGenerator:
    """简单的占位图像生成器 (不需要外部依赖)"""

    def __init__(self, size: int = 224):
        self.size = size

    def create_ego_view(self, object_pos=None, base_orientation=0.0):
        """创建ego视角图像"""
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # 天空渐变
        horizon = self.size // 3
        image[:horizon, :] = [135, 206, 235]  # 天蓝
        image[horizon:, :] = [34, 139, 34]    # 草绿
        image[horizon:horizon+2] = [100, 100, 100]

        # 地平线标记
        base_u = int(self.size // 2 + np.sin(base_orientation) * 20)
        base_v = int(self.size * 0.87)
        cv2.rectangle(image, (base_u - 15, base_v - 10), (base_u + 15, base_v), (50, 50, 50), -1)

        # 箭头
        arrow_u = int(base_u + np.cos(base_orientation) * 30)
        arrow_v = base_v - 30
        cv2.line(image, (base_u, base_v), (arrow_u, arrow_v), (200, 200, 50), 3)

        return image

    def create_arm_view(self, object_dist=0.5, gripper_open=True):
        """创建arm视角图像"""
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        image[:] = [42, 44, 52]  # 深灰背景

        center = self.size // 2
        radius = int((1.0 - object_dist) * self.size // 8) + 10

        # 物体
        alpha = max(0.3, min(1.0, 0.3 + object_dist * 0.7))
        obj_color = tuple(int(c * alpha) for c in (235, 76, 66))
        cv2.circle(image, (center, center), radius, obj_color, -1)

        # 夹爪
        gripper_color = (102, 255, 178) if gripper_open else (255, 214, 102)
        gap = 25 if gripper_open else 8
        cv2.rectangle(image, (center - radius - 25, center - gap - 4),
                     (center - radius - 5, center + gap + 4), gripper_color, -1)
        cv2.rectangle(image, (center + radius + 5, center - gap - 4),
                     (center + radius + 25, center + gap + 4), gripper_color, -1)

        return image


class SmokeKeyboardController:
    """键盘控制器 (smoke 版本)"""

    def __init__(self):
        # 机械臂状态
        self.arm_joints = np.zeros(6, dtype=np.float32)
        self.gripper_open = 0.044
        self.arm_joint_idx = 0
        self.arm_delta = 0.05

        # 底座状态
        self.base_vel = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]

        # 按键状态
        self.keys = {}

        # 帮助信息
        self.help_text = """
╔════════════════════════════════════════════════════════════╗
║  Go2-X5 键盘控制 (Smoke 环境 - 无需 Isaac Lab)            ║
╠════════════════════════════════════════════════════════════╣
║  [底座] W/S=前进/后退  A/D=左转/右转  Q/E=左移/右移        ║
║  [机械臂] I/K=关节1,2  J/L=关节3,4  U/O=关节5,6           ║
║          SPACE=切换关节组                                   ║
║  [夹爪] 1=打开  2=闭合                                     ║
║  [其他] R=重置  P=暂停  H=帮助  ESC=退出                  ║
╚════════════════════════════════════════════════════════════╝
        """

    def handle_key(self, key):
        """处理按键"""
        self.keys[key] = True

        # 机械臂控制 (单次触发)
        if key == "i":
            self.arm_joints[self.arm_joint_idx] += self.arm_delta
        elif key == "k":
            self.arm_joints[self.arm_joint_idx] -= self.arm_delta
        elif key == "j":
            self.arm_joints[(self.arm_joint_idx + 1) % 6] += self.arm_delta
        elif key == "l":
            self.arm_joints[(self.arm_joint_idx + 1) % 6] -= self.arm_delta
        elif key == "u":
            self.arm_joints[(self.arm_joint_idx + 2) % 6] += self.arm_delta
        elif key == "o":
            self.arm_joints[(self.arm_joint_idx + 2) % 6] -= self.arm_delta
        elif key == " ":
            self.arm_joint_idx = (self.arm_joint_idx + 2) % 6
            print(f"\n[机械臂] 切换到关节组: {self.arm_joint_idx+1}, {self.arm_joint_idx+2}, {self.arm_joint_idx+3}")
        elif key == "1":
            self.gripper_open = 0.044
            print("\n[夹爪] 打开")
        elif key == "2":
            self.gripper_open = 0.0
            print("\n[夹爪] 闭合")

        # 限制关节角度
        self.arm_joints = np.clip(self.arm_joints, -3.14, 3.14)

    def release_key(self, key):
        """释放按键"""
        self.keys[key] = False

    def get_action(self):
        """获取当前动作"""
        # 底座速度 (持续按键)
        base_vel = np.zeros(3, dtype=np.float32)
        if self.keys.get("w", False):
            base_vel[0] = 0.3
        elif self.keys.get("s", False):
            base_vel[0] = -0.3

        if self.keys.get("a", False):
            base_vel[2] = 0.3
        elif self.keys.get("d", False):
            base_vel[2] = -0.3

        if self.keys.get("q", False):
            base_vel[1] = 0.2
        elif self.keys.get("e", False):
            base_vel[1] = -0.2

        # 组合动作: [base_vx, base_vy, base_wz, arm_joints(6), gripper]
        action = np.zeros(10, dtype=np.float32)
        action[:3] = base_vel
        action[3:9] = self.arm_joints * 0.1  # 缩放
        action[9] = self.gripper_open

        return action


class SmokeCameraViewer:
    """Smoke 环境的摄像头显示 (使用占位图像)"""

    def __init__(self):
        self.running = True
        self.paused = False
        self.frame_count = 0

        # 创建占位图像生成器
        self.img_gen = PlaceholderImageGenerator(size=224)

    def start(self):
        """启动显示"""
        import threading
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()

    def _display_loop(self):
        """显示循环"""
        cv2.namedWindow("Go2-X5 Camera View (Smoke)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Go2-X5 Camera View (Smoke)", 896, 224)

        print("[摄像头] 已启动 (Smoke 模式 - 使用占位图像)")

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            # 生成占位图像
            ego_view = self.img_gen.create_ego_view(
                object_pos=(112 + np.random.randint(-20, 20), 150 + np.random.randint(-20, 20)),
                base_orientation=np.random.uniform(-0.3, 0.3)
            )
            arm_view = self.img_gen.create_arm_view(
                object_dist=np.random.uniform(0.3, 0.6),
                gripper_open=self.gripper_open if hasattr(self, 'gripper_open') else True
            )

            # 拼接
            combined = np.hstack([ego_view, arm_view])

            # 添加标签
            cv2.putText(combined, "Ego View (Smoke)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Arm View (Smoke)", (456, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {self.frame_count}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示
            cv2.imshow("Go2-X5 Camera View (Smoke)", combined)

            self.frame_count += 1

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                self.running = False

        cv2.destroyAllWindows()

    def update(self, env_state):
        """更新显示 (同步夹爪状态)"""
        if hasattr(self, 'gripper_open'):
            self.gripper_open = env_state.get('gripper_open', 0.044) > 0.02

    def stop(self):
        """停止显示"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)


def non_blocking_keyboard():
    """非阻塞键盘输入"""
    import select
    import tty
    import termios

    def is_data():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if is_data():
                c = sys.stdin.read(1)
                yield c
            else:
                yield None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    parser = argparse.ArgumentParser(description="Go2-X5 键盘控制 (Smoke)")
    parser.add_argument("--no_cameras", action="store_true", help="不显示摄像头")
    parser.add_argument("--record", action="store_true", help="记录数据")
    parser.add_argument("--record_dir", type=str, default="./smoke_demos", help="记录目录")
    args = parser.parse_args()

    print("=" * 60)
    print("  Go2-X5 键盘控制 (Smoke 环境)")
    print("  无需 Isaac Lab - 适合本地开发")
    print("=" * 60)

    # 创建环境
    print("\n[初始化] 创建 Smoke 环境...")
    env = Go2X5GroundPickSmokeEnv(
        task_name="ground_pick",
        task_id=0,
        trial_id=0,
        trial_seed=42,
        image_size=224,
        max_episode_steps=200,
        instruction="pick up the red block from the ground",
    )

    # 重置环境
    obs = env.reset()
    print("[初始化] 环境已就绪")

    # 创建控制器
    controller = SmokeKeyboardController()

    # 创建摄像头显示
    camera_viewer = None
    if not args.no_cameras:
        try:
            camera_viewer = SmokeCameraViewer()
            camera_viewer.start()
        except Exception as e:
            print(f"[警告] 摄像头显示启动失败: {e}")

    # 数据记录
    recorded_data = []
    if args.record:
        import os
        os.makedirs(args.record_dir, exist_ok=True)
        print(f"[记录] 将保存到: {args.record_dir}")

    # 打印帮助
    print(controller.help_text)
    print("\n[控制] 开始！")
    print("提示: 在终端按键控制，摄像头窗口按 ESC 只关闭显示\n")

    # 设置非阻塞输入
    import tty
    import termios
    import select

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        step = 0
        paused = False
        running = True

        while running:
            # 非阻塞键盘输入
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                key_lower = key.lower()

                # 处理按键
                if key_lower == "\x1b":  # ESC
                    print("\n[退出]")
                    running = False
                elif key_lower == "r":
                    print("\n[重置]")
                    obs = env.reset()
                    controller.arm_joints = np.zeros(6, dtype=np.float32)
                    step = 0
                elif key_lower == "p":
                    paused = not paused
                    print(f"\n[{'暂停' if paused else '继续'}]")
                elif key_lower == "h":
                    print(controller.help_text)
                elif key_lower == "q":
                    print("\n[退出]")
                    running = False
                else:
                    controller.handle_key(key_lower)

                # 释放持续按键
                if key_lower not in ["w", "a", "s", "d", "q", "e"]:
                    controller.release_key(key_lower)

            # 释放底座按键 (如果没按)
            for k in ["w", "a", "s", "d", "q", "e"]:
                if not select.select([sys.stdin], [], [], 0)[0]:
                    controller.release_key(k)

            if not paused:
                # 获取动作
                action = controller.get_action()

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                # 记录数据
                if args.record:
                    recorded_data.append({
                        "step": step,
                        "action": action.copy(),
                        "proprio": obs.get("state", np.zeros(55)).copy() if isinstance(obs, dict) else np.zeros(55),
                    })

                # 打印状态
                if step % 10 == 0:
                    print(f"\r[步数: {step}] "
                          f"底座: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]  "
                          f"机械臂: [{action[3]:.2f}, {action[4]:.2f}, {action[5]:.2f}, ...]  "
                          f"夹爪: {'开' if action[9] > 0.02 else '合'}  "
                          f"奖励: {reward:.2f}", end="")

                # 检查结束
                if done:
                    print(f"\n\n[完成] Episode 结束, 步数: {step}, 总奖励: {reward:.2f}")
                    if args.record and recorded_data:
                        import json
                        import os
                        save_path = os.path.join(args.record_dir, f"demo_{len(recorded_data)}.json")
                        with open(save_path, 'w') as f:
                            json.dump(recorded_data, f)
                        print(f"[记录] 已保存: {save_path}")
                        recorded_data = []
                    obs = env.reset()
                    step = 0

            time.sleep(0.01)  # 避免 CPU 过载

    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # 清理
    print("\n[清理] 正在关闭...")
    if camera_viewer:
        camera_viewer.stop()
    env.close()
    print("[完成]")


if __name__ == "__main__":
    main()
