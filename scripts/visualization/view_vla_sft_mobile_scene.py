#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
可视化 VLA-SFT Layer 2 场景。

使用方法:
    python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b1_open_floor
    python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b2_table_approach --live
    python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b3_obstacle_avoidance --num_obstacles 3
    python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b4_partial_occlusion
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# 设置路径
GO2_X5_LAB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GO2_X5_LAB_ROOT / "source" / "robot_lab"))

from isaaclab.app import AppLauncher


# 解析参数
parser = argparse.ArgumentParser(description="Visualize VLA-SFT Layer 2 (Mobile Grasp) scenes")
parser.add_argument("--scene_type", type=str, default="b1_open_floor",
                    choices=["b1_open_floor", "b2_table_approach", "b3_obstacle_avoidance", "b4_partial_occlusion"])
parser.add_argument("--object_type", type=str, default=None,
                    choices=["cube", "sphere", "cylinder", "bowl", "cup"])
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--floor_material", type=str, default="random",
                    choices=["random", "concrete", "wood", "tile", "grass"])
parser.add_argument("--live", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 现在可以导入其他模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

import robot_lab.tasks  # noqa: F401
from robot_lab.assets import GO2_X5_CFG
from robot_lab.tasks.manager_based.manipulation.vla_sft.floor_materials import (
    LAYER1_FLOOR_SIZE,
    LAYER1_FLOOR_THICKNESS,
    LAYER1_FLOOR_VISUAL_Z,
    build_floor_physics_material_cfg,
    build_floor_visual_material_cfg,
    get_floor_material_profile,
    resolve_floor_material_type,
)
from robot_lab.tasks.manager_based.manipulation.vla_sft.scenes import (
    MobileGraspSceneB1,
    MobileGraspSceneB2,
    MobileGraspSceneB3,
    MobileGraspSceneB4,
)


SCENE_CLASSES = {
    "b1_open_floor": MobileGraspSceneB1,
    "b2_table_approach": MobileGraspSceneB2,
    "b3_obstacle_avoidance": MobileGraspSceneB3,
    "b4_partial_occlusion": MobileGraspSceneB4,
}


def object_dimensions(object_type: str, size_scale: float) -> tuple[float, float]:
    """Return a coarse (radius_like, height) for a preview primitive."""
    if object_type == "sphere":
        return size_scale * 0.5, size_scale
    if object_type == "bowl":
        return size_scale * 0.62, size_scale * 0.45
    if object_type == "cup":
        return size_scale * 0.36, size_scale * 1.20
    if object_type == "cylinder":
        return size_scale * 0.42, size_scale * 1.10
    return size_scale * 0.5, size_scale


def create_object(prim_path: str, object_type: str, size_scale: float, color: tuple, pos: tuple, quat: tuple):
    """Create a simple preview object using built-in Isaac Lab primitive spawners."""
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), metallic=0.0)
    radius_like, height = object_dimensions(object_type, size_scale)

    if object_type == "sphere":
        spawn_cfg = sim_utils.SphereCfg(
            radius=radius_like,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            visual_material=visual_material,
        )
    elif object_type in {"cylinder", "cup", "bowl"}:
        spawn_cfg = sim_utils.CylinderCfg(
            radius=radius_like,
            height=height,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            visual_material=visual_material,
        )
    else:
        spawn_cfg = sim_utils.CuboidCfg(
            size=(size_scale, size_scale, size_scale),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            visual_material=visual_material,
        )

    spawn_cfg.func(prim_path, spawn_cfg, translation=tuple(float(v) for v in pos), orientation=tuple(float(v) for v in quat))


def create_table(prim_path: str, pos: tuple, size: tuple, color: tuple[float, float, float] = (0.6, 0.4, 0.2)):
    """Create a static preview table."""
    table_cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
    )
    table_cfg.func(
        prim_path,
        table_cfg,
        translation=(float(pos[0]), float(pos[1]), float(pos[2]) + float(size[2]) / 2.0),
    )


def create_obstacle(prim_path: str, obs_type: str, size: tuple, pos: tuple, color: tuple[float, float, float] = (0.4, 0.4, 0.45)):
    """Create an obstacle for navigation scenarios."""
    if obs_type == "cylinder_barrier":
        # Cylinder obstacle
        obstacle_cfg = sim_utils.CylinderCfg(
            radius=size[0],
            height=size[1],
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        )
        obstacle_cfg.func(
            prim_path,
            obstacle_cfg,
            translation=(float(pos[0]), float(pos[1]), float(size[1]) / 2.0),
        )
    else:
        # Box obstacle
        obstacle_cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        )
        obstacle_cfg.func(
            prim_path,
            obstacle_cfg,
            translation=(float(pos[0]), float(pos[1]), float(size[2]) / 2.0),
        )


def create_light():
    """Create a dome light for scene preview."""
    light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    light_cfg.func("/World/DomeLight", light_cfg)


def create_floor(floor_material: str, floor_visual_material):
    """Create the physical ground and a thin visual overlay for the chosen floor material."""
    floor_profile = get_floor_material_profile(floor_material)
    ground_cfg = sim_utils.GroundPlaneCfg(
        color=floor_profile.ground_plane_color,
        size=LAYER1_FLOOR_SIZE,
        physics_material=build_floor_physics_material_cfg(floor_material),
    )
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    floor_cfg = sim_utils.CuboidCfg(
        size=(LAYER1_FLOOR_SIZE[0], LAYER1_FLOOR_SIZE[1], LAYER1_FLOOR_THICKNESS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=floor_visual_material,
    )
    floor_cfg.func("/World/FloorVisual", floor_cfg, translation=(0.0, 0.0, LAYER1_FLOOR_VISUAL_Z))


def create_robot(base_pos: tuple, base_yaw: float):
    """Spawn the real Go2-X5 robot asset at specified base position."""
    robot_cfg = GO2_X5_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(float(base_pos[0]), float(base_pos[1]), 0.38),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            "arm_joint1": 0.0,
            "arm_joint2": 1.6,
            "arm_joint3": 1.2,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "arm_joint7": 0.044,
            "arm_joint8": 0.044,
        },
        joint_vel={".*": 0.0},
    )
    return Articulation(cfg=robot_cfg)


def resolve_preview_camera(scene_config, base_pos: tuple, target_pos: tuple):
    """Get a stable preview camera framing for the mobile grasp workspace."""
    # Position camera to see both robot and target
    mid_x = (base_pos[0] + target_pos[0]) / 2
    mid_y = (base_pos[1] + target_pos[1]) / 2

    # Camera offset from the midpoint
    offset_distance = 4.0
    eye = (mid_x + offset_distance, mid_y - offset_distance, 2.5)
    target = (mid_x, mid_y, 0.3)

    return eye, target


def main():
    """主函数。"""
    # 设置随机种子
    rng = random.Random(args.seed)

    # 获取场景配置
    scene_class = SCENE_CLASSES[args.scene_type]
    scene_config = scene_class()

    # 采样参数
    obj_type = args.object_type or rng.choice(scene_config.object_types)

    # Sample target pose
    target_pos, target_angle, target_distance = scene_config.sample_target_pose(rng)

    # Sample base init pose
    base_pos, base_yaw = scene_config.sample_base_init_pose(rng)

    # Sample color
    obj_color = tuple(rng.uniform(0.3, 0.9) for _ in range(3))

    # Generate instruction
    instruction = scene_config.generate_instruction(obj_type, rng)

    # Sample obstacles for B3
    obstacles = []
    if hasattr(scene_config, "sample_obstacles") and scene_config.navigation_required:
        obstacles = scene_config.sample_obstacles(target_pos, rng)

    # Sample occluder for B4
    occluder = None
    if hasattr(scene_config, "sample_occluder"):
        occluder = scene_config.sample_occluder(target_pos, rng)

    # Floor material
    floor_type = resolve_floor_material_type(
        args.floor_material,
        available_types=("concrete", "wood", "tile", "grass"),
        rng=rng,
    )
    floor_visual_material, floor_visual_source = build_floor_visual_material_cfg(floor_type)

    # Camera setup
    camera_eye, camera_target = resolve_preview_camera(scene_config, base_pos, target_pos)

    # 打印信息
    print("=" * 60)
    print("VLA-SFT Layer 2 Scene Viewer (Mobile Grasp)")
    print("=" * 60)
    print(f"Scene Type: {args.scene_type}")
    print(f"Object Type: {obj_type}")
    print(f"Target Position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    print(f"Target Angle: {target_angle:.3f} rad ({target_angle*57.3:.1f} deg)")
    print(f"Target Distance: {target_distance:.3f} m")
    print(f"Base Position: ({base_pos[0]:.3f}, {base_pos[1]:.3f})")
    print(f"Base Yaw: {base_yaw:.3f} rad ({base_yaw*57.3:.1f} deg)")
    print(f"Object Color: {tuple(round(c, 2) for c in obj_color)}")
    print(f"Instruction: {instruction}")
    print(f"Floor Material: {floor_type}")
    print(f"Preview Camera Eye: {tuple(round(v, 2) for v in camera_eye)}")
    print(f"Preview Camera Target: {tuple(round(v, 2) for v in camera_target)}")

    if obstacles:
        print(f"Obstacles: {len(obstacles)}")
    if occluder:
        print(f"Occluder: {occluder['type']} at ({occluder['position'][0]:.3f}, {occluder['position'][1]:.3f})")

    # Check for table in B2
    if hasattr(scene_config, "table_position") and scene_config.table_position:
        print(f"Table: pos={scene_config.table_position}, size={scene_config.table_size}")

    print("=" * 60)

    # 创建新的 stage
    sim_utils.create_new_stage()

    # 初始化 SimulationContext
    sim = SimulationContext(SimulationCfg(dt=0.01, device=args.device))

    # 使用 floor profile 创建地面
    create_floor(floor_type, floor_visual_material)

    # 添加光照
    create_light()

    # 添加桌子 (B2 场景)
    if hasattr(scene_config, "table_position") and scene_config.table_position:
        create_table(
            "/World/Table",
            scene_config.table_position,
            scene_config.table_size,
        )

    # 添加障碍物 (B3 场景)
    for i, obs in enumerate(obstacles):
        create_obstacle(
            f"/World/Obstacle_{i}",
            obs["type"],
            obs["size"],
            obs["position"],
        )

    # 添加遮挡物 (B4 场景)
    if occluder:
        create_obstacle(
            "/World/Occluder",
            occluder["type"],
            occluder["size"],
            occluder["position"],
            color=(0.5, 0.3, 0.2),
        )

    # 添加目标物体
    # Use default quaternion (no rotation)
    obj_quat = (1.0, 0.0, 0.0, 0.0)
    obj_size = rng.uniform(*scene_config.object_size_range)
    create_object("/World/TargetObject", obj_type, obj_size, obj_color, target_pos, obj_quat)

    # 加载真实机器人资产
    robot = create_robot(base_pos, base_yaw)
    print("[INFO] Robot loaded from GO2_X5_CFG")

    # 设置相机
    sim.set_camera_view(eye=list(camera_eye), target=list(camera_target))
    sim.reset()

    for _ in range(10):
        sim.step()

    # 打印状态
    print("[INFO] Scene loaded successfully!")
    if args.live:
        print("[INFO] Live mode - physics running. Press Ctrl+C to exit.")
    else:
        print("[INFO] View mode - use mouse to inspect. Press Ctrl+C to exit.")

    # 主循环
    try:
        while simulation_app.is_running():
            if args.live:
                sim.step(render=True)
            else:
                sim.render()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        sim.stop()
        sim.clear()


if __name__ == "__main__":
    main()
    simulation_app.close()
