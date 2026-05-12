# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab config scaffold for a Go2-X5 tabletop manipulation task."""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

from robot_lab.assets import GO2_X5_CFG

from . import events, rewards, terminations
from .actions import HighLevelActionsCfg
from .gr00t_assets import get_first_gr00t_asset_path, get_gr00t_asset_path
from .observations import ObservationsCfg


TABLE_CENTER = (0.86, 0.0)
TABLE_DEPTH = 0.72
TABLE_WIDTH = 1.50
TABLE_THICKNESS = 0.038
TABLETOP_Z = 0.674
TABLE_SIZE = (TABLE_DEPTH, TABLE_WIDTH, TABLE_THICKNESS)
TABLE_LEG_WIDTH = 0.035
TABLE_LEG_EDGE_DISTANCE = 0.055
TABLE_LEG_HEIGHT = TABLETOP_Z - TABLE_THICKNESS

TRAY_CENTER = (0.74, 0.0)
TRAY_SIZE = (0.34, 0.34, 0.015)
TRAY_RAIL_THICKNESS = 0.015
TRAY_RAIL_HEIGHT = 0.055
TRAY_BASE_BOTTOM_Z = TABLETOP_Z + 0.002
TRAY_TOP_Z = TRAY_BASE_BOTTOM_Z + TRAY_SIZE[2]
OBJECT_BASE_Z = TRAY_TOP_Z + 0.002
TRAIN_OBJECT_CENTER_Z = OBJECT_BASE_Z + 0.04
TARGET_MARKER_POS = (0.62, -0.12, TRAY_TOP_Z + 0.006)
PLAY_NUM_ENVS = 16
GR00T_TABLE_SPACING = 5.4
GR00T_TABLE_MDL = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl"


def _asset_mode() -> str:
    value = os.environ.get("GO2_X5_TABLETOP_ASSET_MODE", "auto").lower()
    return value if value in {"primitive", "gr00t", "auto"} else "auto"


def _table_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.95, dynamic_friction=0.75, restitution=0.02)


def _object_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.75, dynamic_friction=0.60, restitution=0.02)


def _collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.0025, rest_offset=0.0)


def _use_gr00t_mdl_materials(preset: str) -> bool:
    value = os.environ.get("GO2_X5_TABLETOP_USE_GR00T_MDL", "auto").lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return preset == "play" and _asset_mode() != "primitive"


def _table_visual_material(preset: str):
    if _use_gr00t_mdl_materials(preset) and NVIDIA_NUCLEUS_DIR:
        return sim_utils.MdlFileCfg(mdl_path=GR00T_TABLE_MDL, project_uvw=True, texture_scale=(0.18, 0.18))
    color = (0.54, 0.38, 0.22) if preset == "play" else (0.50, 0.44, 0.36)
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.64)


def _leg_visual_material() -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.29, 0.30, 0.30), metallic=0.25, roughness=0.38)


def _make_table_cfg(preset: str = "train") -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLETOP_Z - TABLE_SIZE[2] * 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=_collision_props(),
            physics_material=_table_material(),
            visual_material=_table_visual_material(preset),
        ),
    )


def _make_table_leg_cfg(name: str, x_offset: float, y_offset: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER[0] + x_offset, TABLE_CENTER[1] + y_offset, TABLE_LEG_HEIGHT * 0.5)
        ),
        spawn=sim_utils.CylinderCfg(
            radius=TABLE_LEG_WIDTH * 0.5,
            height=TABLE_LEG_HEIGHT,
            axis="Z",
            collision_props=_collision_props(),
            physics_material=_table_material(),
            visual_material=_leg_visual_material(),
        ),
    )


def _make_tray_cfg(preset: str = "train") -> AssetBaseCfg:
    use_gr00t = preset == "play" and _asset_mode() != "primitive"
    tray_usd = get_gr00t_asset_path("simple_tray") if use_gr00t else None
    if tray_usd is not None:
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.70, 0.0, TABLETOP_Z + 0.01)),
            spawn=sim_utils.UsdFileCfg(usd_path=tray_usd, collision_props=_collision_props()),
        )

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TRAY_CENTER[0], TRAY_CENTER[1], TRAY_BASE_BOTTOM_Z + TRAY_SIZE[2] * 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=TRAY_SIZE,
            collision_props=_collision_props(),
            physics_material=_table_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.17, 0.18, 0.17), roughness=0.50),
        ),
    )


def _make_tray_rail_cfg(prim_name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=_collision_props(),
            physics_material=_table_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.16, 0.18), roughness=0.55),
        ),
    )


def _object_center_z(base_z: float, shape: str, size: tuple[float, ...]) -> float:
    if shape == "sphere":
        return base_z + size[0]
    return base_z + size[-1] * 0.5


def _play_object_spec() -> dict:
    return {"shape": "cylinder", "size": (0.032, 0.095), "mass": 0.16, "color": (0.18, 0.42, 0.82)}


def _make_shape_spawn(spec: dict, color_scale: float = 1.0):
    shape = spec["shape"]
    size = spec["size"]
    color = tuple(min(1.0, max(0.0, channel * color_scale)) for channel in spec["color"])
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=MassPropertiesCfg(mass=spec["mass"]),
        collision_props=_collision_props(),
        physics_material=_object_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.42),
    )
    if shape in {"cube", "cuboid"}:
        return sim_utils.CuboidCfg(size=size, **common)
    if shape == "cylinder":
        return sim_utils.CylinderCfg(radius=size[0], height=size[1], axis="Z", **common)
    if shape == "sphere":
        return sim_utils.SphereCfg(radius=size[0], **common)
    if shape == "cone":
        return sim_utils.ConeCfg(radius=size[0], height=size[1], axis="Z", **common)
    raise ValueError(f"Unsupported tabletop object shape: {shape}")


def _make_object_cfg(preset: str = "train") -> RigidObjectCfg:
    use_gr00t = preset == "play" and _asset_mode() != "primitive"
    object_usd = (
        get_first_gr00t_asset_path(("simple_bottle", "grab_bottle", "grab_waterbottle", "grab_apple", "simple_cube"))
        if use_gr00t
        else None
    )
    common = dict(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.70, 0.0, TRAIN_OBJECT_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    if object_usd is not None:
        return RigidObjectCfg(
            **common,
            spawn=sim_utils.UsdFileCfg(
                usd_path=object_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.18),
                collision_props=_collision_props(),
            ),
        )

    if preset == "play":
        spec = _play_object_spec()
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(TRAY_CENTER[0], TRAY_CENTER[1], _object_center_z(OBJECT_BASE_Z, spec["shape"], spec["size"])),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=_make_shape_spawn(spec),
        )

    return RigidObjectCfg(
        **common,
        spawn=sim_utils.CuboidCfg(
            size=(0.055, 0.055, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.15),
            collision_props=_collision_props(),
            physics_material=_object_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.16, 0.12), metallic=0.0),
        ),
    )


def _make_target_marker_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReachTarget",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_MARKER_POS),
        spawn=sim_utils.CylinderCfg(
            radius=0.032,
            height=0.006,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.75, 0.22), emissive_color=(0.02, 0.10, 0.03), roughness=0.35
            ),
        ),
    )


def _make_overview_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 20.0),
        ),
    )


def _tray_front_back_rail(name: str, sign: float) -> AssetBaseCfg:
    return _make_tray_rail_cfg(
        name,
        (
            TRAY_CENTER[0],
            TRAY_CENTER[1] + sign * (TRAY_SIZE[1] * 0.5 + TRAY_RAIL_THICKNESS * 0.5),
            TRAY_TOP_Z + TRAY_RAIL_HEIGHT * 0.5,
        ),
        (TRAY_SIZE[0] + 2.0 * TRAY_RAIL_THICKNESS, TRAY_RAIL_THICKNESS, TRAY_RAIL_HEIGHT),
    )


def _tray_left_right_rail(name: str, sign: float) -> AssetBaseCfg:
    return _make_tray_rail_cfg(
        name,
        (
            TRAY_CENTER[0] + sign * (TRAY_SIZE[0] * 0.5 + TRAY_RAIL_THICKNESS * 0.5),
            TRAY_CENTER[1],
            TRAY_TOP_Z + TRAY_RAIL_HEIGHT * 0.5,
        ),
        (TRAY_RAIL_THICKNESS, TRAY_SIZE[1], TRAY_RAIL_HEIGHT),
    )


def _configure_scene_assets(scene: "Go2X5TabletopSceneCfg", preset: str) -> None:
    scene.table = _make_table_cfg(preset)
    leg_x = TABLE_DEPTH * 0.5 - TABLE_LEG_EDGE_DISTANCE
    leg_y = TABLE_WIDTH * 0.5 - TABLE_LEG_EDGE_DISTANCE
    scene.table_leg_fl = _make_table_leg_cfg("TableLegFL", -leg_x, -leg_y)
    scene.table_leg_fr = _make_table_leg_cfg("TableLegFR", -leg_x, leg_y)
    scene.table_leg_rl = _make_table_leg_cfg("TableLegRL", leg_x, -leg_y)
    scene.table_leg_rr = _make_table_leg_cfg("TableLegRR", leg_x, leg_y)
    scene.tray = _make_tray_cfg(preset)
    scene.tray_front = _tray_front_back_rail("TrayFront", 1.0)
    scene.tray_back = _tray_front_back_rail("TrayBack", -1.0)
    scene.tray_left = _tray_left_right_rail("TrayLeft", -1.0)
    scene.tray_right = _tray_left_right_rail("TrayRight", 1.0)
    scene.object = _make_object_cfg(preset)
    scene.target_marker = _make_target_marker_cfg()
    if preset == "play":
        scene.overview_camera = _make_overview_camera_cfg()


@configclass
class Go2X5TabletopSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING

    table = _make_table_cfg("train")
    table_leg_fl = _make_table_leg_cfg(
        "TableLegFL", -(TABLE_DEPTH * 0.5 - TABLE_LEG_EDGE_DISTANCE), -(TABLE_WIDTH * 0.5 - TABLE_LEG_EDGE_DISTANCE)
    )
    table_leg_fr = _make_table_leg_cfg(
        "TableLegFR", -(TABLE_DEPTH * 0.5 - TABLE_LEG_EDGE_DISTANCE), TABLE_WIDTH * 0.5 - TABLE_LEG_EDGE_DISTANCE
    )
    table_leg_rl = _make_table_leg_cfg(
        "TableLegRL", TABLE_DEPTH * 0.5 - TABLE_LEG_EDGE_DISTANCE, -(TABLE_WIDTH * 0.5 - TABLE_LEG_EDGE_DISTANCE)
    )
    table_leg_rr = _make_table_leg_cfg(
        "TableLegRR", TABLE_DEPTH * 0.5 - TABLE_LEG_EDGE_DISTANCE, TABLE_WIDTH * 0.5 - TABLE_LEG_EDGE_DISTANCE
    )
    tray = _make_tray_cfg("train")
    tray_front = _tray_front_back_rail("TrayFront", 1.0)
    tray_back = _tray_front_back_rail("TrayBack", -1.0)
    tray_left = _tray_left_right_rail("TrayLeft", -1.0)
    tray_right = _tray_left_right_rail("TrayRight", 1.0)
    object = _make_object_cfg("train")
    target_marker = _make_target_marker_cfg()
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link6",
                name="end_effector",
                offset=OffsetCfg(pos=(0.08657, 0.0, 0.0)),
            ),
        ],
    )
    dog_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/dog_vla_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.30, 0.0, 0.16), rot=(-0.3799, 0.5963, 0.5963, -0.3799), convention="ros"),
    )
    arm_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.08657, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(
            color=(0.42, 0.43, 0.40),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.98, 0.95, 0.88), intensity=2000.0),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/fill_light",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.94, 0.86), intensity=650.0, angle=2.5),
    )


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_stage = EventTerm(func=events.reset_staged_task_state, mode="reset", params={"num_stages": 4})
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    randomize_object_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 16,
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=rewards.reaching_object, params={"std": 0.15}, weight=2.0)
    gripper_close = RewTerm(func=rewards.gripper_close_near_object, weight=1.0)
    object_lifted = RewTerm(func=rewards.object_lifted, params={"minimal_height": 0.86}, weight=6.0)
    base_stability = RewTerm(func=rewards.base_stability, weight=0.5)
    staged_progress = RewTerm(func=rewards.staged_progress_placeholder, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(func=terminations.robot_fallen, params={"minimum_height": 0.18})
    success = DoneTerm(func=terminations.tabletop_success)


@configclass
class Go2X5TabletopEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2X5TabletopSceneCfg = Go2X5TabletopSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: HighLevelActionsCfg = HighLevelActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        visual_preset = os.environ.get("GO2_X5_TABLETOP_VISUAL_PRESET", "train").lower()
        if visual_preset not in {"train", "play"}:
            visual_preset = "train"
        _configure_scene_assets(self.scene, visual_preset)
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
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
        self.export_io_descriptors = True
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.65, 0.0, 0.65)


@configclass
class Go2X5TabletopEnvCfg_PLAY(Go2X5TabletopEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        visual_preset = os.environ.get("GO2_X5_TABLETOP_VISUAL_PRESET", "play").lower()
        if visual_preset not in {"train", "play"}:
            visual_preset = "play"
        _configure_scene_assets(self.scene, visual_preset)
        if visual_preset == "play":
            self.events.reset_object_position.params["pose_range"]["x"] = (0.0, 0.0)
            self.events.reset_object_position.params["pose_range"]["y"] = (0.0, 0.0)
            self.events.reset_object_position.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.scene.num_envs = PLAY_NUM_ENVS
        self.scene.env_spacing = GR00T_TABLE_SPACING
        self.observations.policy.enable_corruption = False
        self.viewer.eye = (14.0, -14.0, 10.0)
        self.viewer.lookat = (0.78, 0.0, 0.72)
