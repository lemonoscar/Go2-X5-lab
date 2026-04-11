#!/usr/bin/env python3
"""
Go2-X5 键盘控制 (简化版 - 不需要 pxr)

使用 Isaac Lab 的 ManagerBasedEnv 而不是直接操作 USD
"""

import argparse
from pathlib import Path
import sys
import threading
import time

import numpy as np
import cv2
import torch

# Add robot_lab to path
script_dir = Path(__file__).parent.absolute()
robot_lab_path = script_dir.parent / "source" / "robot_lab"
if robot_lab_path.exists():
    sys.path.insert(0, str(robot_lab_path))

# Import Isaac Lab (this will also launch Isaac Sim)
try:
    from isaaclab.app import AppLauncher
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
    from isaaclab.sensors import Camera
except ImportError as e:
    print(f"Isaac Lab import failed: {e}")
    print("\n请确保已激活 Isaac Lab 环境:")
    print("  conda activate env_isaaclab")
    exit(1)

# Import configs AFTER AppLauncher will be used
from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
    Go2X5GroundPickEnvCfg_PLAY,
    Go2X5GroundPickSceneCfg,
)
from isaaclab.envs import ManagerBasedRLEnv


def main():
    parser = argparse.ArgumentParser(description="Go2-X5 键盘控制")
    parser.add_argument("--no_cameras", action="store_true", help="不显示摄像头")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Isaac Sim (this makes pxr available)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # NOW we can import other modules that depend on Isaac Sim
    # Import pxr INSIDE the function, not at module level
    try:
        from pxr import Usd, UsdGeom, Gf
        has_pxr = True
    except ImportError:
        print("Warning: pxr not available, some features may not work")
        has_pxr = True  # Actually it should be available now

    # Create environment config
    env_cfg = Go2X5GroundPickEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.time_out = None

    scene_cfg = Go2X5GroundPickSceneCfg(num_envs=1, env_spacing=4.0)

    # Create environment
    print("[初始化] 创建环境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Enable cameras
    print("[初始化] 创建摄像头...")
    dog_camera = Camera(scene_cfg.dog_camera)
    arm_camera = Camera(scene_cfg.arm_camera)

    dog_camera.create("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot/base/dog_vla_camera")
    arm_camera.create("/World/envs/env_0/Robot/arm_link6", "/World/envs/env_0/Robot/arm_link6/arm_vla_camera")

    # Reset environment
    print("[初始化] 重置环境...")
    env.reset()

    # Wait for cameras to initialize
    for _ in range(10):
        env.step(torch.zeros(1, env.unwrapped.action_space.shape[0]))
        dog_camera.update(env.sim.current_time, env.sim.dt)
        arm_camera.update(env.sim.current_time, env.sim.dt)

    # Get action space info
    action_dim = env.unwrapped.action_space.shape[0]
    print(f"[初始化] 动作空间维度: {action_dim}")

    # Base keyboard
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

    # Camera display
    camera_thread = None
    if not args.no_cameras:
        display_active = True
        latest_images = {"dog": None, "arm": None}

        def camera_display_loop():
            nonlocal display_active, latest_images
            cv2.namedWindow("Go2-X5 Cameras", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Go2-X5 Cameras", 896, 224)

            while display_active:
                if latest_images["dog"] is not None and latest_images["arm"] is not None:
                    combined = np.hstack([latest_images["dog"], latest_images["arm"]])
                    cv2.putText(combined, "Ego View", (10, 30), 0, 1, (0,255,0), 2)
                    cv2.putText(combined, "Arm View", (456, 30), 0, 1, (0,255,0), 2)
                    cv2.imshow("Go2-X5 Cameras", combined)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    display_active = False

            cv2.destroyAllWindows()

        camera_thread = threading.Thread(target=camera_display_loop, daemon=True)
        camera_thread.start()

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
    print("\n[控制] 开始!")

    step = 0
    paused = False

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

            # Construct action
            # The action space depends on the env configuration
            # For Go2-X5: [base_vel(3), arm_joints(6), gripper(1)] = 10
            action = torch.zeros(1, 10, device=env.unwrapped.device)
            action[0, 0] = base_cmd[0, 0]  # vx
            action[0, 1] = base_cmd[0, 1]  # vy
            action[0, 2] = base_cmd[0, 2]  # wz
            action[0, 3:9] = torch.tensor(arm_joints, device=env.unwrapped.device) * 0.1
            action[0, 9] = gripper_open

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            step += 1

            # Update cameras
            dog_camera.update(env.sim.current_time, env.sim.dt)
            arm_camera.update(env.sim.current_time, env.sim.dt)

            # Update display
            if not args.no_cameras and step % 2 == 0:
                try:
                    dog_rgb = dog_camera.data.output['rgb'][0].cpu().numpy()
                    arm_rgb = arm_camera.data.output['rgb'][0].cpu().numpy()
                    latest_images["dog"] = (dog_rgb * 255).astype(np.uint8)
                    latest_images["arm"] = (arm_rgb * 255).astype(np.uint8)
                except:
                    pass

            if step % 30 == 0:
                print(f"\r[步数: {step}]  动作: [{action[0,0]:.2f}, {action[0,1]:.2f}, {action[0,2]:.2f}, ...]", end="")

        if app_launcher.is_keyboard_pressed("escape"):
            break

    print("\n\n[退出] 清理中...")
    if camera_thread:
        display_active = False
        camera_thread.join(timeout=1)

    simulation_app.close()
    print("[完成]")


if __name__ == "__main__":
    main()
