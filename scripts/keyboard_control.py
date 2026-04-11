#!/usr/bin/env python3
"""
Go2-X5 键盘控制 (最简版本)
"""

import argparse
from pathlib import Path
import sys

# Add robot_lab to path
script_dir = Path(__file__).parent.absolute()
robot_lab_path = script_dir.parent / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))

# Import AppLauncher first - this will launch Isaac Sim
from isaaclab.app import AppLauncher

# Parse args FIRST before launching app
parser = argparse.ArgumentParser(description="Go2-X5 键盘控制")
parser.add_argument("--no_cameras", action="store_true", help="不显示摄像头")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# NOW launch the app - this makes pxr available
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# AFTER app is launched, import everything else
import numpy as np
import torch
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv

from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
    Go2X5GroundPickEnvCfg_PLAY,
    Go2X5GroundPickSceneCfg,
)


def main():
    # Create environment config
    env_cfg = Go2X5GroundPickEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.time_out = None

    scene_cfg = Go2X5GroundPickSceneCfg(num_envs=1, env_spacing=4.0)

    # Create environment
    print("[初始化] 创建环境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    # Wait a bit for initialization
    for _ in range(10):
        env.step(torch.zeros(1, env.unwrapped.action_space.shape[0], device=env.unwrapped.device))

    # Base keyboard control
    base_keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.5,
        v_y_sensitivity=0.3,
        omega_z_sensitivity=0.5,
    )
    base_keyboard = Se2Keyboard(base_keyboard_cfg)

    # Arm state
    arm_joints = np.zeros(6, dtype=np.float32)
    gripper_open = 0.044
    arm_joint_idx = 0
    arm_delta = 0.05

    # Print help
    print("""
╔════════════════════════════════════════════════════════════╗
║  Go2-X5 键盘控制                                           ║
╠════════════════════════════════════════════════════════════╣
║  [底座] W/S=前进/后退  A/D=左转/右转  Q/E=左移/右移        ║
║  [机械臂] I/K=关节1  J/L=关节2  U/O=关节3  SPACE=切换    ║
║  [夹爪] 1=打开  2=闭合                                     ║
║  [其他] R=重置  P=暂停  H=帮助  ESC=退出                  ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Main loop
    print("\n[控制] 开始! (Ctrl+C 退出)")

    step = 0
    paused = False

    try:
        while simulation_app.is_running():
            # Handle arm keyboard
            if app_launcher.is_keyboard_pressed("i"):
                arm_joints[arm_joint_idx] += arm_delta
            if app_launcher.is_keyboard_pressed("k"):
                arm_joints[arm_joint_idx] -= arm_delta
            if app_launcher.is_keyboard_pressed("j"):
                arm_joints[(arm_joint_idx + 1) % 6] += arm_delta
            if app_launcher.is_keyboard_pressed("l"):
                arm_joints[(arm_joint_idx + 1) % 6] -= arm_delta
            if app_launcher.is_keyboard_pressed("u"):
                arm_joints[(arm_joint_idx + 2) % 6] += arm_delta
            if app_launcher.is_keyboard_pressed("o"):
                arm_joints[(arm_joint_idx + 2) % 6] -= arm_delta

            if app_launcher.is_keyboard_pressed("space"):
                arm_joint_idx = (arm_joint_idx + 2) % 6
                print(f"\r[机械臂] 关节: {arm_joint_idx+1}, {arm_joint_idx+2}, {arm_joint_idx+3}")

            if app_launcher.is_keyboard_pressed("1"):
                gripper_open = 0.044
                print("\r[夹爪] 打开")
            if app_launcher.is_keyboard_pressed("2"):
                gripper_open = 0.0
                print("\r[夹爪] 闭合")

            if app_launcher.is_keyboard_pressed("r"):
                print("\n[重置]")
                env.reset()
                arm_joints = np.zeros(6, dtype=np.float32)
                step = 0
                continue

            if app_launcher.is_keyboard_pressed("p"):
                paused = not paused
                print(f"\n[{'暂停' if paused else '继续'}]")

            if app_launcher.is_keyboard_pressed("h"):
                print("""
[底座] W/S=前进/后退  A/D=左转/右转  Q/E=左移/右移
[机械臂] I/K=关节1  J/L=关节2  U/O=关节3  SPACE=切换
[夹爪] 1=打开  2=闭合
[其他] R=重置  P=暂停  H=帮助  ESC=退出
                """)

            if not paused:
                # Get base command from keyboard
                base_cmd_input = base_keyboard.advance()
                base_cmd = torch.tensor([
                    [base_cmd_input[0], base_cmd_input[1], base_cmd_input[2]]
                ], device=env.unwrapped.device)

                # Construct action: [base_vel(3), arm_joints(6), gripper(1)] = 10
                action = torch.zeros(1, 10, device=env.unwrapped.device)
                action[0, 0] = base_cmd[0, 0]  # vx
                action[0, 1] = base_cmd[0, 1]  # vy
                action[0, 2] = base_cmd[0, 2]  # wz
                action[0, 3:9] = torch.tensor(arm_joints, device=env.unwrapped.device) * 0.1
                action[0, 9] = gripper_open

                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                step += 1

                if step % 30 == 0:
                    print(f"\r[步数: {step}]  动作: [{action[0,0]:.2f}, {action[0,1]:.2f}, {action[0,2]:.2f}, ...]  夹爪: {'开' if gripper_open > 0.02 else '合'}", end="")

            if app_launcher.is_keyboard_pressed("escape"):
                break

    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断")

    print("\n[完成]")
    simulation_app.close()


if __name__ == "__main__":
    main()
