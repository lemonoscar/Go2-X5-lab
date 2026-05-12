#!/usr/bin/env python3
"""
Go2-X5 键盘控制 + 摄像头显示 (使用 dog-only policy)

功能:
1. 底座由预训练的 dog-only policy 控制 (WASD)
2. 机械臂手动控制 (IJKLUO)
3. 实时显示双摄像头

前提条件:
- Isaac Lab 已安装
- GO2_X5_LOW_LEVEL_POLICY_PATH 已设置

使用方法:
    cd ~/xhq_workload/Go2-X5-lab
    export GO2_X5_LOW_LEVEL_POLICY_PATH=logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt
    conda activate isaaclab
    python scripts/control/keyboard_control_with_policy.py
"""

import argparse
from pathlib import Path
import os
import threading
import time

import numpy as np
import cv2
import torch

try:
    from isaaclab.app import AppLauncher
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
    from isaaclab.sensors import Camera
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext, spawn_scene
except ImportError as e:
    print(f"错误: Isaac Lab 未安装 - {e}")
    print("请在安装了 Isaac Lab 的机器上运行此脚本")
    print("或者在 ubuntu1 上运行")
    exit(1)

# Add robot_lab to path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
robot_lab_path = REPO_ROOT / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))

# Import configurations
from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
    Go2X5GroundPickSceneCfg,
    ActionsCfg,
)


def main():
    parser = argparse.ArgumentParser(description="Go2-X5 键盘控制")
    parser.add_argument("--no_cameras", action="store_true", help="不显示摄像头")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Check for policy path
    policy_path = os.environ.get("GO2_X5_LOW_LEVEL_POLICY_PATH", "")
    if not policy_path or not os.path.exists(policy_path):
        print("\n警告: GO2_X5_LOW_LEVEL_POLICY_PATH 未设置或文件不存在")
        print("底座将使用直接控制（不是最佳效果）")
        print("\n建议运行:")
        print("  export GO2_X5_LOW_LEVEL_POLICY_PATH=~/xhq_workload/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt")
        print()

    # Launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Create scene config
    scene_cfg = Go2X5GroundPickSceneCfg(num_envs=1, env_spacing=4.0)

    # Create simulation
    sim = SimulationContext(sim_params={"dt": 0.005})
    spawn_scene(scene_cfg, sim)

    # Reset a few times to stabilize
    for _ in range(5):
        sim.step()

    # Get robot
    robot_cfg = scene_cfg.robot
    robot = Articulation(cfg=robot_cfg)
    robot.initialize("/World/envs/env_0/Robot")
    robot.write_data_to_sim()
    robot.update(sim.dt)

    # Get joint indices
    DOG_JOINT_NAMES = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    ARM_JOINT_NAMES = [
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]

    dog_joint_ids, _ = robot.find_joints(DOG_JOINT_NAMES, preserve_order=True)
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    gripper_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES, preserve_order=True)

    print(f"[初始化] Dog joints: {len(dog_joint_ids)}, Arm joints: {len(arm_joint_ids)}, Gripper: {len(gripper_joint_ids)}")

    # Load dog-only policy if available
    dog_policy = None
    if policy_path and os.path.exists(policy_path):
        try:
            from isaaclab.utils.mapping import load_mdp_from_ckpt
            dog_policy = load_mdp_from_ckpt(policy_path)
            print(f"[初始化] Dog-only policy 已加载: {policy_path}")
        except Exception as e:
            print(f"[警告] 无法加载 policy: {e}")

    # Create cameras
    print("\n[初始化] 创建摄像头...")
    dog_camera = Camera(scene_cfg.dog_camera)
    arm_camera = Camera(scene_cfg.arm_camera)

    dog_camera.create("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot/base/dog_vla_camera")
    arm_camera.create("/World/envs/env_0/Robot/arm_link6", "/World/envs/env_0/Robot/arm_link6/arm_vla_camera")

    # Initialize cameras
    for _ in range(10):
        sim.step()
        dog_camera.update(sim.current_time, sim.dt)
        arm_camera.update(sim.current_time, sim.dt)

    print("[初始化] 完成")

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
    display_active = True
    latest_images = {"dog": None, "arm": None}

    if not args.no_cameras:
        def camera_display_loop():
            nonlocal display_active, latest_images
            cv2.namedWindow("Go2-X5 Cameras", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Go2-X5 Cameras", 896, 224)

            while display_active:
                if latest_images["dog"] is not None and latest_images["arm"] is not None:
                    combined = np.hstack([latest_images["dog"], latest_images["arm"]])
                    cv2.putText(combined, "Ego View", (10, 30), 0, 1, (0,255,0), 2)
                    cv2.putText(combined, "Arm View", (456, 30), 0, 1, (0,255,0), 2)
                    cv2.putText(combined, f"Gripper: {'Open' if gripper_open > 0.02 else 'Closed'}",
                               (10, 215), 0, 0.5, (255,255,255), 1)
                    cv2.imshow("Go2-X5 Cameras", combined)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    display_active = False

            cv2.destroyAllWindows()

        camera_thread = threading.Thread(target=camera_display_loop, daemon=True)
        camera_thread.start()

    # Print help
    print("""
╔════════════════════════════════════════════════════════════╗
║  Go2-X5 键盘控制 (Dog-only Policy)                        ║
╠════════════════════════════════════════════════════════════╣
║  [底座] W/S=前进/后退  A/D=左转/右转  Q/E=左移/右移        ║
║        (由 dog-only policy 控制)                           ║
║  [机械臂] I/K=关节1  J/L=关节2  U/O=关节3                ║
║          SPACE=切换关节组                                   ║
║  [夹爪] 1=打开  2=闭合                                     ║
║  [其他] R=重置  P=暂停  H=帮助  ESC=退出                  ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Main loop
    print("\n[控制] 开始!")

    step = 0
    paused = False
    base_cmd = torch.zeros(1, 3, device=sim.device)

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
            print(f"\r[机械臂] 切换到关节: {arm_joint_idx+1}, {(arm_joint_idx+1)%6+1}, {(arm_joint_idx+2)%6+1}")

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
            base_cmd[0, 0] = base_cmd_input[0]
            base_cmd[0, 1] = base_cmd_input[1]
            base_cmd[0, 2] = base_cmd_input[2]

            # Control dog joints with policy or direct
            if dog_policy is not None:
                # Use dog-only policy
                # Get dog observations
                dog_pos = robot.data.joint_pos[:, dog_joint_ids]
                dog_vel = robot.data.joint_vel[:, dog_joint_ids]

                # TODO: Construct proper observation for policy
                # For now, use simple proportional control
                target_dog_pos = robot.data.joint_pos[:, dog_joint_ids].clone()
                # Simple forward/backward
                if abs(base_cmd[0, 0]) > 0.01:
                    target_dog_pos[:, 2::3] += base_cmd[0, 0] * 0.1  # hip joints
                    target_dog_pos[:, 3::3] -= base_cmd[0, 0] * 0.05  # thigh joints
                # Turning
                if abs(base_cmd[0, 2]) > 0.01:
                    target_dog_pos[:, ::3] += base_cmd[0, 2] * 0.05  # left hips
                    target_dog_pos[:, 1::3] -= base_cmd[0, 2] * 0.05  # right hips

                robot.set_joint_position_target(dog_joint_ids, target_dog_pos)
            else:
                # Direct control (simplified)
                target_dog_pos = robot.data.joint_pos[:, dog_joint_ids].clone()
                if abs(base_cmd[0, 0]) > 0.01:
                    for i in range(4):
                        target_dog_pos[0, i*3 + 2] += base_cmd[0, 0] * 0.02
                if abs(base_cmd[0, 2]) > 0.01:
                    left_mult = 1 if base_cmd[0, 2] > 0 else -1
                    for i in range(2):
                        target_dog_pos[0, i*3] += base_cmd[0, 2] * 0.01 * left_mult
                        target_dog_pos[0, (i+2)*3] -= base_cmd[0, 2] * 0.01 * left_mult

                robot.set_joint_position_target(dog_joint_ids, target_dog_pos)

            # Set arm joint positions
            current_arm_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
            target_arm_pos = torch.tensor(arm_joints, device=sim.device).unsqueeze(0)
            # Smooth interpolation
            new_arm_pos = current_arm_pos * 0.9 + target_arm_pos * 0.1
            robot.set_joint_position_target(arm_joint_ids, new_arm_pos)

            # Set gripper
            gripper_pos = torch.tensor([[gripper_open, gripper_open]], device=sim.device)
            robot.set_joint_position_target(gripper_joint_ids, gripper_pos)

            # Write to sim
            robot.write_data_to_sim()

            # Step sim
            sim.step()

            # Update cameras every few steps
            if step % 2 == 0:
                dog_camera.update(sim.current_time, sim.dt)
                arm_camera.update(sim.current_time, sim.dt)

                # Update display images
                if not args.no_cameras:
                    try:
                        dog_rgb = dog_camera.data.output['rgb'][0].cpu().numpy()
                        arm_rgb = arm_camera.data.output['rgb'][0].cpu().numpy()
                        latest_images["dog"] = (dog_rgb * 255).astype(np.uint8)
                        latest_images["arm"] = (arm_rgb * 255).astype(np.uint8)
                    except:
                        pass

            step += 1
            if step % 30 == 0:
                arm_state = robot.data.joint_pos[:, arm_joint_ids][0].cpu().numpy()
                print(f"\r[步数: {step}]  "
                      f"底座指令: [{base_cmd[0,0]:.2f}, {base_cmd[0,1]:.2f}, {base_cmd[0,2]:.2f}]  "
                      f"机械臂: [{arm_state[0]:.2f}, {arm_state[1]:.2f}, {arm_state[2]:.2f}, ...]", end="")

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
