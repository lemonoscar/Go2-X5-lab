#!/usr/bin/env python3
"""
Go2-X5 键盘控制 (简化稳定版)

直接使用 Isaac Sim 的 SimulationContext，避免复杂的 ManagerBasedRLEnv
"""

import argparse
import os
from pathlib import Path
import sys
import time

# ============ 在任何导入之前设置 policy 路径 ============
policy_paths = [
    "/home/lemon/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt",
]
for p in policy_paths:
    if os.path.exists(p):
        os.environ["GO2_X5_LOW_LEVEL_POLICY_PATH"] = p
        print(f"[设置 Policy] {p}")
        break
# ====================================================================

# 添加 robot_lab 到路径
REPO_ROOT = Path(__file__).resolve().parents[2]
robot_lab_path = REPO_ROOT / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))

# 导入 AppLauncher 并启动 Isaac Sim
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 启动后，导入其他模块
import torch
import numpy as np
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import spawn_scene
from isaaclab.assets import Articulation

# 导入配置 - 直接导入不触发 __init__.py 的扫描
from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
    Go2X5GroundPickSceneCfg,
)


def main():
    print("=" * 60)
    print("  Go2-X5 键盘控制")
    print("=" * 60)

    # 创建场景配置
    scene_cfg = Go2X5GroundPickSceneCfg(num_envs=1, env_spacing=4.0)

    # 创建仿真
    sim = SimulationContext(sim_params={"dt": 0.005})
    spawn_scene(scene_cfg, sim)

    # 获取机器人
    robot_cfg = scene_cfg.robot
    robot = Articulation(cfg=robot_cfg)
    robot.initialize("/World/envs/env_0/Robot")
    robot.write_data_to_sim()

    # 等待稳定
    print("[初始化] 等待仿真稳定...")
    for _ in range(20):
        sim.step()
    robot.update(sim.dt)

    print("[初始化] 完成")

    # 获取关节 ID
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

    print(f"[信息] Dog 关节: {len(dog_joint_ids)}, Arm 关节: {len(arm_joint_ids)}, Gripper: {len(gripper_joint_ids)}")

    # 键盘控制
    base_keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.5,
        v_y_sensitivity=0.3,
        omega_z_sensitivity=0.5,
    )
    base_keyboard = Se2Keyboard(base_keyboard_cfg)

    # 状态变量
    arm_joints = np.zeros(6, dtype=np.float32)
    gripper_open = 0.044
    arm_joint_idx = 0
    arm_delta = 0.05

    # 打印说明
    print("\n" + "=" * 60)
    print("控制说明:")
    print("  点击 Isaac Sim 窗口获得焦点")
    print("  [底座] W/S=前进/后退  A/D=左转/右转")
    print("  [机械臂] I/K/J/L/U/O 控制关节  SPACE=切换")
    print("  [夹爪] 1=打开  2=闭合")
    print("  [其他] R=重置  ESC=退出")
    print("=" * 60 + "\n")

    step = 0
    auto_move = True  # 默认自动前进，方便测试

    try:
        while simulation_app.is_running():
            # 获取键盘输入
            base_cmd_input = base_keyboard.advance()
            base_cmd = torch.tensor([
                [base_cmd_input[0], base_cmd_input[1], base_cmd_input[2]]
            ], device=sim.device)

            # 处理机械臂键盘
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

            if app_launcher.is_keyboard_pressed("m"):
                auto_move = not auto_move
                print(f"\r[自动移动] {'开启' if auto_move else '关闭'}")

            if app_launcher.is_keyboard_pressed("r"):
                print("\n[重置]")
                sim.reset()
                robot.write_data_to_sim()
                for _ in range(10):
                    sim.step()
                robot.update(sim.dt)
                arm_joints = np.zeros(6, dtype=np.float32)
                step = 0
                continue

            # 构造动作
            # 对于底座，我们直接设置关节位置（简化版，不用 policy）
            target_dog_pos = robot.data.joint_pos[:, dog_joint_ids].clone()

            # 简单的底座控制
            if abs(base_cmd[0, 0]) > 0.01:  # 前进/后退
                # 简单的步态：交替抬腿
                for i in range(4):
                    if base_cmd[0, 0] > 0:  # 前进
                        target_dog_pos[0, i*3 + 2] += 0.1  # 抬腿
                    else:
                        target_dog_pos[0, i*3 + 2] -= 0.1
            if abs(base_cmd[0, 2]) > 0.01:  # 转向
                # 左右腿不同
                for i in range(2):
                    target_dog_pos[0, i*3] += base_cmd[0, 2] * 0.05
                    target_dog_pos[0, (i+2)*3] -= base_cmd[0, 2] * 0.05

            # 自动移动模式
            if auto_move and abs(base_cmd[0, 0]) < 0.01 and abs(base_cmd[0, 2]) < 0.01:
                # 简单的踏步动作
                target_dog_pos = robot.data.joint_pos[:, dog_joint_ids].clone()
                phase = (step // 10) % 2
                if phase == 0:
                    target_dog_pos[0, [0, 3, 6, 9]] += 0.1
                    target_dog_pos[0, [1, 4, 7, 10]] -= 0.1
                else:
                    target_dog_pos[0, [0, 3, 6, 9]] -= 0.1
                    target_dog_pos[0, [1, 4, 7, 10]] += 0.1

            # 设置底座关节
            robot.set_joint_position_target(dog_joint_ids, target_dog_pos)

            # 设置机械臂关节
            current_arm_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
            target_arm_pos = torch.tensor(arm_joints, device=sim.device).unsqueeze(0)
            new_arm_pos = current_arm_pos * 0.9 + target_arm_pos * 0.1
            robot.set_joint_position_target(arm_joint_ids, new_arm_pos)

            # 设置夹爪
            gripper_pos = torch.tensor([[gripper_open, gripper_open]], device=sim.device)
            robot.set_joint_position_target(gripper_joint_ids, gripper_pos)

            # 写入仿真
            robot.write_data_to_sim()

            # 步进
            sim.step()
            robot.update(sim.dt)

            # 打印状态
            if step % 30 == 0:
                arm_state = robot.data.joint_pos[:, arm_joint_ids][0].cpu().numpy()
                print(f"\r[步数: {step:4d}]  "
                      f"指令: [{base_cmd[0,0]:.2f}, {base_cmd[0,1]:.2f}, {base_cmd[0,2]:.2f}]  "
                      f"机械臂: [{arm_state[0]:.2f}, {arm_state[1]:.2f}, ...]  "
                      f"自动: {'开' if auto_move else '关'}   ", end="")

            step += 1

            if app_launcher.is_keyboard_pressed("escape"):
                break

    except KeyboardInterrupt:
        print("\n\n[中断]")

    print("\n[完成]")
    simulation_app.close()


if __name__ == "__main__":
    main()
