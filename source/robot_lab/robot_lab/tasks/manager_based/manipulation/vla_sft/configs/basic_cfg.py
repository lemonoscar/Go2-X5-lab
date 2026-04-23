# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Environment configuration for VLA-SFT Layer 1 (Basic Grasp) data collection.

This module extends the ground_pick environment with scene-based randomization
and VLA-specific data collection features.
"""

from __future__ import annotations

import random
from dataclasses import MISSING, field
from typing import Dict, List, Optional, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.assets import GO2_X5_CFG
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as loco_mdp

# Import MDP functions from ground_pick (via vla_sft.mdp re-export)
from ..mdp import (
    AbsBinaryJointPositionActionCfg,
    JointPositionActionCfg,
    action_rate_l2,
    apply_floor_visual_material,
    gripper_closed_around_object,
    gripper_opening,
    joint_vel_l2,
    object_ee_distance,
    object_height,
    object_is_lifted,
    object_position_in_robot_root_frame,
    reset_root_state_uniform,
    reset_scene_to_default,
    root_height_below_minimum,
    success_bonus,
    stable_base_bonus,
    time_out,
    ground_pick_success,
    ee_to_object_vector,
)
from ..floor_materials import (
    DEFAULT_FLOOR_MATERIAL_TYPES,
    DEFAULT_TERRAIN_TYPES,
    LAYER1_FLOOR_SIZE,
    LAYER1_FLOOR_THICKNESS,
    LAYER1_FLOOR_VISUAL_Z,
    UNDULATING_HEIGHT_SCALE,
    UNDULATING_FREQUENCY,
    HEIGHTFIELD_SIZE,
    build_floor_physics_material_cfg,
    build_floor_visual_material_cfg,
    get_floor_material_profile,
    resolve_floor_material_type,
    resolve_terrain_type,
)


# Joint names
DOG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

ARM_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6",
]

GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]


@configclass
class Go2X5VLASBasicSceneCfg(InteractiveSceneCfg):
    """Scene configuration for VLA-SFT Basic Grasp environments.

    Extends the ground_pick scene with support for:
    - Multiple object types (cube, sphere, cylinder, bowl, cup)
    - Clutter objects (distractors)
    - Table surface for elevated grasps
    - Scene-based randomization
    """

    robot: ArticulationCfg = MISSING

    # Target object (main object to grasp)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.56, 0.0, 0.04),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.82, 0.16, 0.12),
                metallic=0.0,
            ),
        ),
    )

    # Clutter objects (distractors) - initially disabled
    clutter_objects: List[RigidObjectCfg] = field(default_factory=list)

    # Table surface for elevated grasps (optional)
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=None,  # Disabled by default, enabled for A2 scenes
    )

    # End effector frame
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

    # Dog camera (body-mounted)
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
        offset=CameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.16),
            rot=(-0.3799, 0.5963, 0.5963, -0.3799),
            convention="ros",
        ),
    )

    # Arm camera (wrist-mounted)
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
        offset=CameraCfg.OffsetCfg(
            pos=(0.08657, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    ground_visual = AssetBaseCfg(
        prim_path="/World/FloorVisual",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, LAYER1_FLOOR_VISUAL_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(LAYER1_FLOOR_SIZE[0], LAYER1_FLOOR_SIZE[1], LAYER1_FLOOR_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.50, 0.50, 0.48),
                roughness=0.95,
                metallic=0.0,
            ),
        ),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )


@configclass
class ActionsCfg:
    """Action configuration for Go2-X5 VLA-SFT.

    Note: base_policy using pre-trained locomotion policy is not yet
    implemented. For now, use direct joint control.
    """

    # Arm joint control
    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale={
            "arm_joint1": 1.2,
            "arm_joint2": 1.2,
            "arm_joint3": 1.2,
            "arm_joint4": 0.8,
            "arm_joint5": 0.7,
            "arm_joint6": 0.7,
        },
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )

    # Gripper control
    gripper_action = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={"arm_joint7": 0.044, "arm_joint8": 0.044},
        close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
        threshold=0.022,
        positive_threshold=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for policy learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, scale=1.0)
        dog_joint_pos = ObsTerm(
            func=loco_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        dog_joint_vel = ObsTerm(
            func=loco_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES)},
            noise=Unoise(n_min=-1.0, n_max=1.0),
            scale=0.05,
        )
        arm_joint_pos = ObsTerm(
            func=loco_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)},
            scale=1.0,
        )
        arm_joint_vel = ObsTerm(
            func=loco_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)},
            scale=0.05,
        )
        object_position = ObsTerm(func=object_position_in_robot_root_frame)
        object_height = ObsTerm(func=object_height)
        ee_to_object = ObsTerm(func=ee_to_object_vector)
        gripper_opening = ObsTerm(func=gripper_opening)
        actions = ObsTerm(func=loco_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for scene randomization."""

    apply_floor_visual = EventTerm(
        func=apply_floor_visual_material,
        mode="prestartup",
    )

    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.08, 0.08),
                "y": (-0.18, 0.18),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration for training."""

    reaching_object = RewTerm(
        func=object_ee_distance,
        params={"std": 0.12},
        weight=2.5,
    )
    grasp_closure = RewTerm(
        func=gripper_closed_around_object,
        params={"distance_std": 0.10, "close_threshold": 0.018},
        weight=2.0,
    )
    lifting_object = RewTerm(
        func=object_is_lifted,
        params={"minimal_height": 0.08},
        weight=8.0,
    )
    success_bonus = RewTerm(
        func=success_bonus,
        params={
            "minimal_height": 0.12,
            "max_eef_object_distance": 0.14,
            "close_threshold": 0.018,
        },
        weight=20.0,
    )
    stable_base = RewTerm(
        func=stable_base_bonus,
        params={"roll_pitch_std": 0.35, "vertical_vel_std": 0.35},
        weight=1.0,
    )
    action_rate = RewTerm(func=action_rate_l2, weight=-1.0e-4)
    arm_joint_vel = RewTerm(
        func=joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)},
        weight=-1.0e-4,
    )


@configclass
class TerminationsCfg:
    """Termination configuration."""

    time_out = DoneTerm(func=time_out, time_out=True)
    robot_fall = DoneTerm(
        func=root_height_below_minimum,
        params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
    )
    object_dropped = DoneTerm(
        func=root_height_below_minimum,
        params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(
        func=ground_pick_success,
        params={
            "minimal_height": 0.12,
            "max_eef_object_distance": 0.14,
            "close_threshold": 0.018,
        },
    )


@configclass
class Go2X5VLASBasicEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for VLA-SFT Layer 1 (Basic Grasp) environment.

    This environment supports scene-based data collection with:
    - Multiple object types and poses
    - Clutter objects
    - Table surfaces
    - VLA-specific camera outputs
    """

    scene: Go2X5VLASBasicSceneCfg = field(
        default_factory=lambda: Go2X5VLASBasicSceneCfg(
            num_envs=64,
            env_spacing=4.0,
            replicate_physics=True,
        )
    )

    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: EventCfg = field(default_factory=EventCfg)

    commands = None
    curriculum = None

    # VLA-SFT specific settings
    vla_sft: Dict[str, any] = field(
        default_factory=lambda: {
            "scene_layer": "basic",
            "enable_scene_randomization": True,
            "floor_material": "random",
            "floor_material_types": list(DEFAULT_FLOOR_MATERIAL_TYPES),
            "terrain_type": "flat",  # "flat" or "undulating"
            "terrain_types": list(DEFAULT_TERRAIN_TYPES),
            "undulating_height_scale": UNDULATING_HEIGHT_SCALE,
            "undulating_frequency": UNDULATING_FREQUENCY,
            "data_collection": {
                "output_dir": "./data/vla_sft/basic",
                "save_images": True,
                "save_frequency": 1,
            },
        }
    )

    def __post_init__(self):
        """Post-initialization configuration."""
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024

        # Robot configuration
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

        floor_rng = random.Random(self.vla_sft.get("scene_seed"))
        available_floor_types = self.vla_sft.get("floor_material_types") or list(DEFAULT_FLOOR_MATERIAL_TYPES)
        selected_floor_material = resolve_floor_material_type(
            self.vla_sft.get("floor_material"),
            available_types=available_floor_types,
            rng=floor_rng,
        )
        floor_profile = get_floor_material_profile(selected_floor_material)
        floor_visual_material, floor_visual_source = build_floor_visual_material_cfg(selected_floor_material)

        # Resolve terrain type (flat or undulating)
        available_terrain_types = self.vla_sft.get("terrain_types") or list(DEFAULT_TERRAIN_TYPES)
        selected_terrain_type = resolve_terrain_type(
            self.vla_sft.get("terrain_type"),
            available_types=available_terrain_types,
            rng=floor_rng,
        )

        self.scene.ground.spawn = GroundPlaneCfg(
            size=LAYER1_FLOOR_SIZE,
            color=floor_profile.ground_plane_color,
            physics_material=build_floor_physics_material_cfg(selected_floor_material),
        )
        self.scene.ground_visual.init_state = AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, LAYER1_FLOOR_VISUAL_Z)
        )
        self.scene.ground_visual.spawn = sim_utils.CuboidCfg(
            size=(LAYER1_FLOOR_SIZE[0], LAYER1_FLOOR_SIZE[1], LAYER1_FLOOR_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=floor_visual_material,
        )

        self.vla_sft["selected_floor_material"] = selected_floor_material
        self.vla_sft["selected_floor_material_source"] = floor_visual_source
        self.vla_sft["selected_terrain_type"] = selected_terrain_type

        self.export_io_descriptors = True
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)


@configclass
class Go2X5VLASBasicEnvCfg_PLAY(Go2X5VLASBasicEnvCfg):
    """Play configuration for single environment interaction."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
