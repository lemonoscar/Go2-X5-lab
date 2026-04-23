#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
可视化 VLA-SFT Layer 1 场景。

使用方法:
    python scripts/tools/view_vla_sft_scene.py --scene_type a1_ground_grasp
    python scripts/tools/view_vla_sft_scene.py --scene_type a2_table_grasp --live
    python scripts/tools/view_vla_sft_scene.py --scene_type a3_simple_clutter --num_clutter 5
    python scripts/tools/view_vla_sft_scene.py --scene_type a4_multi_height_table_clutter
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# 设置路径
GO2_X5_LAB_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(GO2_X5_LAB_ROOT / "source" / "robot_lab"))

from isaaclab.app import AppLauncher


# 解析参数
parser = argparse.ArgumentParser(description="Visualize VLA-SFT Layer 1 scenes")
parser.add_argument("--scene_type", type=str, default="a1_ground_grasp",
                    choices=["a1_ground_grasp", "a2_table_grasp", "a3_simple_clutter", "a4_multi_height_table_clutter"])
parser.add_argument("--object_type", type=str, default=None,
                    choices=["cube", "sphere", "cylinder", "bowl", "cup"])
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--enable_table", action="store_true")
parser.add_argument("--num_clutter", type=int, default=None)
parser.add_argument("--floor_material", type=str, default="random",
                    choices=["random", "concrete", "wood", "tile", "grass"])
parser.add_argument("--terrain_type", type=str, default="flat",
                    choices=["flat", "undulating"])
parser.add_argument("--height_scale", type=float, default=0.08,
                    help="Maximum height variation for undulating terrain (meters)")
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
    HEIGHTFIELD_SIZE,
    UNDULATING_FREQUENCY,
    build_floor_physics_material_cfg,
    build_floor_visual_material_cfg,
    create_undulating_terrain_collision,
    create_undulating_terrain_visual,
    generate_perlin_noise_terrain,
    get_floor_material_profile,
    resolve_floor_material_type,
)
from robot_lab.tasks.manager_based.manipulation.vla_sft.render_assets import (
    bind_local_floor_material,
    choose_local_floor_asset,
)
from robot_lab.tasks.manager_based.manipulation.vla_sft.scenes import (
    BasicGraspSceneA1,
    BasicGraspSceneA2,
    BasicGraspSceneA3,
    BasicGraspSceneA4,
)


SCENE_CLASSES = {
    "a1_ground_grasp": BasicGraspSceneA1,
    "a2_table_grasp": BasicGraspSceneA2,
    "a3_simple_clutter": BasicGraspSceneA3,
    "a4_multi_height_table_clutter": BasicGraspSceneA4,
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


def place_object_on_surface(scene_config, object_type: str, size_scale: float, pos):
    """Lift an object so its base rests on the ground or table surface."""
    pos = pos.copy()
    _, height = object_dimensions(object_type, size_scale)
    surface_z = 0.0
    if hasattr(scene_config, "surface_height_for_position"):
        surface_z = float(scene_config.surface_height_for_position(pos))
    elif scene_config.table_position is not None and scene_config.table_size is not None and float(pos[2]) > 0.2:
        surface_z = float(scene_config.table_position[2]) + float(scene_config.table_size[2])
    pos[2] = max(float(pos[2]), surface_z + height * 0.5 + 0.003)
    return pos


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
    """Create a static preview table aligned with the Go2-X5 front workspace."""
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


def create_tables(scene_config):
    """Create all tables defined by the scene configuration."""
    table_layouts = scene_config.get_table_layouts() if hasattr(scene_config, "get_table_layouts") else []
    for index, table_layout in enumerate(table_layouts):
        create_table(
            f"/World/Table_{index}",
            table_layout["position"],
            table_layout["size"],
            color=table_layout.get("color", (0.6, 0.4, 0.2)),
        )


def create_light(light_cfg):
    """Create a dome light for scene preview."""
    light_cfg.func("/World/DomeLight", light_cfg)


def build_neutral_light_cfg():
    """Use a neutral dome light without any background texture."""
    return sim_utils.DomeLightCfg(color=(0.74, 0.74, 0.72), intensity=2600.0)


def create_floor(floor_material: str, floor_visual_material, local_floor_asset=None, rng: random.Random | None = None,
                terrain_type: str = "flat", height_scale: float = 0.08):
    """Create the physical ground and a thin visual overlay for the chosen floor material.

    Args:
        floor_material: Material type (concrete, wood, tile, grass).
        floor_visual_material: Visual material configuration.
        local_floor_asset: Optional local PBR asset.
        rng: Random number generator.
        terrain_type: "flat" or "undulating".
        height_scale: Maximum height variation for undulating terrain.
    """
    floor_profile = get_floor_material_profile(floor_material)

    if terrain_type == "undulating":
        # Create undulating terrain with collision and visual meshes
        try:
            import omni.usd
            from pxr import UsdPhysics

            # Generate and create the collision heightfield
            terrain_seed = rng.randint(0, 10000) if rng else None

            # Create visual terrain mesh
            create_undulating_terrain_visual(
                prim_path="/World/FloorVisual",
                floor_material=floor_material,
                size=LAYER1_FLOOR_SIZE,
                height_scale=height_scale,
                frequency=UNDULATING_FREQUENCY,
                resolution=HEIGHTFIELD_SIZE,
                seed=terrain_seed,
            )

            # For now, use a flat ground plane for physics (heightfield collision can be added later)
            ground_cfg = sim_utils.GroundPlaneCfg(
                color=floor_profile.ground_plane_color,
                size=LAYER1_FLOOR_SIZE,
                physics_material=build_floor_physics_material_cfg(floor_material),
            )
            ground_cfg.func("/World/GroundPlane", ground_cfg)

            print(f"[INFO] Created undulating terrain (height_scale={height_scale:.3f})")

        except Exception as e:
            print(f"[WARNING] Failed to create undulating terrain, falling back to flat: {e}")
            # Fallback to flat terrain
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
    else:
        # Flat terrain
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

    if local_floor_asset is not None:
        bind_local_floor_material("/World/FloorVisual", local_floor_asset, rng=rng)


def create_robot():
    """Spawn the real Go2-X5 robot asset in a ready-to-grasp pose."""
    robot_cfg = GO2_X5_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
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


def resolve_preview_camera(scene_config, obj_pos: tuple[float, float, float]):
    """Get a stable preview camera framing for the workspace."""
    eye = getattr(scene_config, "preview_camera_eye", (2.2, -1.9, 1.35))
    target = getattr(scene_config, "preview_camera_target", None)
    if target is None:
        target = (float(obj_pos[0]), float(obj_pos[1]), max(0.14, float(obj_pos[2]) + 0.14))
    return eye, target


def main():
    """主函数。"""
    # 设置随机种子
    rng = random.Random(args.seed)

    # 获取场景配置
    scene_class = SCENE_CLASSES[args.scene_type]
    scene_config = scene_class()

    # 采样参数
    obj_type = args.object_type or scene_config.sample_object_type(rng)
    obj_pos, obj_quat = scene_config.sample_object_pose(rng)
    obj_size = scene_config.sample_object_size(rng)
    obj_pos = place_object_on_surface(scene_config, obj_type, obj_size, obj_pos)
    obj_color = scene_config.sample_object_color(rng)
    instruction = scene_config.generate_instruction(obj_type, rng)
    num_clutter = args.num_clutter if args.num_clutter is not None else scene_config.sample_clutter_count(rng)
    camera_eye, camera_target = resolve_preview_camera(scene_config, obj_pos)
    floor_type = resolve_floor_material_type(
        args.floor_material,
        available_types=getattr(scene_config, "floor_material_types", ("concrete", "wood", "tile", "grass")),
        rng=rng,
    )
    floor_visual_material, floor_visual_source = build_floor_visual_material_cfg(floor_type)
    local_floor_asset = choose_local_floor_asset(floor_type, rng=rng)
    if local_floor_asset is not None:
        floor_visual_source = f"LocalPBR:{local_floor_asset.asset_id}"
    light_cfg = build_neutral_light_cfg()

    # 打印信息
    print("=" * 60)
    print("VLA-SFT Scene Viewer")
    print("=" * 60)
    print(f"Scene Type: {args.scene_type}")
    print(f"Object Type: {obj_type}")
    print(f"Object Size: {obj_size:.3f} m")
    print(f"Object Position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
    print(f"Object Color: {tuple(round(c, 2) for c in obj_color)}")
    print(f"Instruction: {instruction}")
    print(f"Floor Material: {floor_type}")
    print(f"Floor Material Source: {floor_visual_source}")
    print(f"Terrain Type: {args.terrain_type}")
    if args.terrain_type == "undulating":
        print(f"Height Scale: {args.height_scale:.3f} m")
    print("Light Source: neutral dome light")
    print(f"Preview Camera Eye: {tuple(round(v, 2) for v in camera_eye)}")
    print(f"Preview Camera Target: {tuple(round(v, 2) for v in camera_target)}")
    table_layouts = scene_config.get_table_layouts() if hasattr(scene_config, "get_table_layouts") else []
    if table_layouts:
        print(f"Tables: {len(table_layouts)}")
    if num_clutter > 0:
        print(f"Clutter Objects: {num_clutter}")
    print("=" * 60)

    # 创建新的 stage
    sim_utils.create_new_stage()

    # 初始化 SimulationContext
    sim = SimulationContext(SimulationCfg(dt=0.01, device=args.device))

    # 使用 floor profile 创建地面
    create_floor(floor_type, floor_visual_material, local_floor_asset=local_floor_asset, rng=rng,
                terrain_type=args.terrain_type, height_scale=args.height_scale)

    # 添加光照
    create_light(light_cfg)

    # 添加桌子 (如果需要)
    should_create_table = args.enable_table or bool(table_layouts)
    if should_create_table:
        create_tables(scene_config)

    # 添加目标物体
    create_object("/World/TargetObject", obj_type, obj_size, obj_color, obj_pos, obj_quat)

    # 添加 clutter
    if num_clutter > 0:
        clutter_positions = scene_config.sample_clutter_positions(num_clutter, obj_pos, rng)
        for i, c_pos in enumerate(clutter_positions):
            c_type = scene_config.sample_clutter_type(rng)
            c_size = scene_config.sample_clutter_size(rng)
            c_pos = place_object_on_surface(scene_config, c_type, c_size, c_pos)
            c_color = scene_config.sample_object_color(rng)
            c_quat = scene_config.sample_clutter_orientation(rng)

            create_object(
                f"/World/Clutter_{i}",
                c_type,
                c_size,
                c_color,
                c_pos,
                c_quat,
            )

    # 加载真实机器人资产
    robot = create_robot()
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
