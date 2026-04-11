#!/usr/bin/env python3
"""
Go2-X5 键盘控制测试 (最简版本)
"""

import argparse
import os
from pathlib import Path
import sys

# ============ 在导入其他模块之前设置 policy 路径 ============
policy_paths = [
    "/home/lemon/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt",
    str(Path(__file__).parent.parent / "logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt"),
]

for p in policy_paths:
    if os.path.exists(p):
        os.environ["GO2_X5_LOW_LEVEL_POLICY_PATH"] = p
        print(f"[找到 Policy] {p}")
        break
else:
    print("[警告] 未找到 dog-only policy!")
# ====================================================================

# Add robot_lab to path
script_dir = Path(__file__).parent.absolute()
robot_lab_path = script_dir.parent / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))

from isaaclab.app import AppLauncher

# Parse args
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
    Go2X5GroundPickEnvCfg_PLAY,
    Go2X5GroundPickSceneCfg,
)


def main():
    from pathlib import Path
    import os
    import tempfile

    print("=" * 60)
    print("  Go2-X5 键盘控制测试")
    print("=" * 60)

    # 自动查找并直接设置 policy 路径
    policy_paths = [
        "/home/lemon/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt",
        str(Path(__file__).parent.parent / "logs/rsl_rl/go2_x5_dog_only_flat/2026-04-01_03-51-52/exported/policy.pt"),
    ]

    policy_path = None
    for p in policy_paths:
        if os.path.exists(p):
            policy_path = p
            break

    # Create scene config
    scene_cfg = Go2X5GroundPickSceneCfg(num_envs=1, env_spacing=4.0)

    # Create environment config
    env_cfg = Go2X5GroundPickEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.time_out = None

    # 设置 log 目录 (必需)
    env_cfg.log_dir = Path(tempfile.gettempdir()) / "isaac_lab_logs"

    # 直接覆盖 policy 路径 (关键修复!)
    if policy_path:
        env_cfg.actions.base_policy.policy_path = policy_path
        print(f"[设置 Policy] {policy_path}")
    else:
        print("[警告] 未找到 dog-only policy，底座控制可能不正常")

    print("[1/5] 创建环境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("[2/5] 重置环境...")
    env.reset()

    print("[3/5] 获取机器人...")
    robot = env.scene["robot"]
    print(f"  机器人名称: {robot.name}")
    print(f"  关节数量: {robot.num_joints}")

    print("[4/5] 设置键盘控制...")
    base_keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.5,
        v_y_sensitivity=0.3,
        omega_z_sensitivity=0.5,
    )
    base_keyboard = Se2Keyboard(base_keyboard_cfg)

    # 获取关节信息
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

    print(f"  Dog关节: {len(dog_joint_ids)} 个")
    print(f"  Arm关节: {len(arm_joint_ids)} 个")
    print(f"  Gripper关节: {len(gripper_joint_ids)} 个")

    # 获取动作空间
    action_space = env.unwrapped.action_space
    print(f"  动作空间: {action_space}")

    print("[5/5] 启动控制循环...")
    print("\n" + "=" * 60)
    print("控制说明:")
    print("  点击 Isaac Sim 窗口获得焦点")
    print("  W/S - 前进/后退")
    print("  A/D - 左转/右转")
    print("  R - 重置")
    print("  ESC - 退出")
    print("=" * 60 + "\n")

    step = 0
    arm_joints = np.zeros(6, dtype=np.float32)
    gripper_open = 0.044

    # 用于自动移动的标志
    auto_move = True

    try:
        while simulation_app.is_running():
            # 获取键盘输入
            base_cmd_input = base_keyboard.advance()
            base_cmd = torch.tensor([
                [base_cmd_input[0], base_cmd_input[1], base_cmd_input[2]]
            ], device=env.unwrapped.device)

            # 构造动作
            action = torch.zeros(1, 10, device=env.unwrapped.device)
            action[0, 0] = base_cmd[0, 0]  # vx
            action[0, 1] = base_cmd[1]  # vy
            action[0, 2] = base_cmd[0, 2]  # wz
            action[0, 3:9] = torch.tensor(arm_joints, device=env.unwrapped.device) * 0.1
            action[0, 9] = gripper_open

            # 如果没有输入，自动前进一点 (测试用)
            if auto_move and abs(base_cmd[0, 0]) < 0.01 and abs(base_cmd[0, 2]) < 0.01:
                action[0, 0] = 0.2  # 自动前进

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            step += 1

            # 打印状态
            if step % 10 == 0:
                joint_pos = robot.data.joint_pos[0, :3].cpu().numpy()  # 前3个关节
                print(f"\r[步数: {step:4d}]  "
                      f"指令: [{action[0,0]:.2f}, {action[0,1]:.2f}, {action[0,2]:.2f}]  "
                      f"关节: [{joint_pos[0]:.2f}, {joint_pos[1]:.2f}, {joint_pos[2]:.2f}]  "
                      f"自动移动: {'开' if auto_move else '关'}   ", end="")

            # 检查键盘控制
            if app_launcher.is_keyboard_pressed("r"):
                print("\n[重置]")
                env.reset()
                step = 0

            if app_launcher.is_keyboard_pressed("m"):
                auto_move = not auto_move
                print(f"\n[自动移动] {'开启' if auto_move else '关闭'}")

            if app_launcher.is_keyboard_pressed("escape"):
                break

            # 防止太快
            import time
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[中断]")

    print("\n[完成]")
    simulation_app.close()


if __name__ == "__main__":
    main()
