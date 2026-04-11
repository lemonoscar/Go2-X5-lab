#!/usr/bin/env python3
"""
Go2-X5 键盘控制 + 摄像头显示 (简化版)

使用方法:
    cd ~/xhq_workload/Go2-X5-lab
    python scripts/keyboard_control_simple.py

键盘控制:
    [底座 - WASD] W/S=前进/后退 A/D=左转/右转 Q/E=左移/右移
    [机械臂] I/K/J/L/U/O 控制关节 SPACE=切换关节组
    [夹爪] 1=打开 2=闭合
    [其他] R=重置 P=暂停 H=帮助 ESC=退出
"""

import argparse
from pathlib import Path
import time
import threading

import numpy as np
import cv2
import torch

try:
    from isaaclab.app import AppLauncher
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
    from isaaclab.sensors import Camera
    from isaaclab.utils import convert_dict_to_backend
except ImportError:
    print("Isaac Lab not found")
    exit(1)

# Add robot_lab to path
import sys
robot_lab_path = Path(__file__).parent.parent / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))


def main():
    parser = argparse.ArgumentParser(description="Go2-X5 键盘控制")
    parser.add_argument("--no_cameras", action="store_true", help="不显示摄像头")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import configs
    from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
        Go2X5GroundPickEnvCfg_PLAY,
        Go2X5GroundPickSceneCfg,
    )
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    # Create scene config
    scene_cfg = Go2X5GroundPickSceneCfg(num_envs=1, env_spacing=4.0)

    # Spawn scene manually
    from isaaclab.sim import SimulationContext, spawn_scene
    sim = SimulationContext(sim_params={"dt": 0.005})
    spawn_scene(scene_cfg, sim)

    # Get robot
    from isaaclab.assets import Articulation
    from isaaclab.utils.assets import register_asset_to_sim

    robot_cfg = scene_cfg.robot
    robot = Articulation(cfg=robot_cfg)
    robot.initialize("/World/envs/env_0/Robot")
    robot.write_data_to_sim()

    # Create cameras
    print("\n[初始化] 创建摄像头...")
    dog_camera = Camera(scene_cfg.dog_camera)
    arm_camera = Camera(scene_cfg.arm_camera)

    dog_camera.create("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot/base/dog_vla_camera")
    arm_camera.create("/World/envs/env_0/Robot/arm_link6", "/World/envs/env_0/Robot/arm_link6/arm_vla_camera")

    # Reset
    sim.reset()
    for _ in range(10):
        sim.step()
        dog_camera.update(sim.current_time, sim.dt)
        arm_camera.update(sim.current_time, sim.dt)

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

    # Camera display
    camera_thread = None
    if not args.no_cameras:
        display_active = True
        latest_dog_img = None
        latest_arm_img = None

        def camera_display_loop():
            nonlocal display_active, latest_dog_img, latest_arm_img
            cv2.namedWindow("Cameras", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cameras", 896, 224)

            while display_active:
                if latest_dog_img is not None and latest_arm_img is not None:
                    combined = np.hstack([latest_dog_img, latest_arm_img])
                    cv2.putText(combined, "Ego View", (10, 30), 0, 1, (0,255,0), 2)
                    cv2.putText(combined, "Arm View", (456, 30), 0, 1, (0,255,0), 2)
                    cv2.imshow("Cameras", combined)

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
║  [机械臂] I/K=关节1,2  J/L=关节3,4  U/O=关节5,6           ║
║          SPACE=切换关节组                                   ║
║  [夹爪] 1=打开  2=闭合                                     ║
║  [其他] R=重置  P=暂停  H=帮助  ESC=退出                  ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Main loop
    print("\n[控制] 开始!")

    step = 0
    paused = False

    # Joint indices for arm (from env config)
    arm_joint_ids, _ = robot.find_joints(
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
        preserve_order=True
    )
    gripper_joint_ids, _ = robot.find_joints(
        ["arm_joint7", "arm_joint8"],
        preserve_order=True
    )

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
            print(f"\r[机械臂] 当前控制: J{arm_joint_idx+1}, J{(arm_joint_idx+1)%6+1}, J{(arm_joint_idx+2)%6+1}")

        if app_launcher.is_keyboard_pressed("1"):
            gripper_open = 0.044
            print("\r[夹爪] 打开")
        if app_launcher.is_keyboard_pressed("2"):
            gripper_open = 0.0
            print("\r[夹爪] 闭合")

        if app_launcher.is_keyboard_pressed("r"):
            print("\n[重置]")
            sim.reset()
            arm_joints = np.zeros(6, dtype=np.float32)
            step = 0

        if app_launcher.is_keyboard_pressed("p"):
            paused = not paused
            print(f"\n[{'暂停' if paused else '继续'}]")

        if app_launcher.is_keyboard_pressed("h"):
            print("""
[底座] W/S=前进/后退  A/D=左转/右转  Q/E=左移/右移
[机械臂] I/K=关节1,2  J/L=关节3,4  U/O=关节5,6  SPACE=切换
[夹爪] 1=打开  2=闭合
[其他] R=重置  P=暂停  H=帮助  ESC=退出
            """)

        if not paused:
            # Get base command
            base_cmd = base_keyboard.advance()
            base_vel = torch.tensor([
                [base_cmd[0], base_cmd[1], base_cmd[2]]
            ], device=sim.device)

            # Set arm joint positions
            current_arm_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
            target_arm_pos = torch.tensor(arm_joints, device=sim.device).unsqueeze(0)
            # Smooth interpolation
            new_arm_pos = current_arm_pos * 0.9 + target_arm_pos * 0.1
            robot.set_joint_position_target(
                arm_joint_ids,
                new_arm_pos,
                env_ids=None,
            )

            # Set gripper
            gripper_pos = torch.tensor([[gripper_open, gripper_open]], device=sim.device)
            robot.set_joint_position_target(
                gripper_joint_ids,
                gripper_pos,
                env_ids=None,
            )

            # Write to sim
            robot.write_data_to_sim()

            # Step sim
            sim.step()

            # Update cameras
            dog_camera.update(sim.current_time, sim.dt)
            arm_camera.update(sim.current_time, sim.dt)

            # Update display images
            if not args.no_cameras and step % 3 == 0:
                try:
                    dog_rgb = dog_camera.data.output['rgb'][0].cpu().numpy()
                    arm_rgb = arm_camera.data.output['rgb'][0].cpu().numpy()
                    latest_dog_img = (dog_rgb * 255).astype(np.uint8)
                    latest_arm_img = (arm_rgb * 255).astype(np.uint8)
                except:
                    pass

            step += 1
            if step % 30 == 0:
                arm_state = robot.data.joint_pos[:, arm_joint_ids][0].cpu().numpy()
                print(f"\r[步数: {step}]  底座: [{base_vel[0,0]:.2f}, {base_vel[0,1]:.2f}, {base_vel[0,2]:.2f}]  "
                      f"机械臂: [{arm_state[0]:.2f}, {arm_state[1]:.2f}, {arm_state[2]:.2f}, ...]  "
                      f"夹爪: {'开' if gripper_open > 0.02 else '合'}", end="")

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
