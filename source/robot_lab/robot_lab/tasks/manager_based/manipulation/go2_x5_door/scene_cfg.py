# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab config scaffold for a Go2-X5 door-opening task."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robot_lab.assets import GO2_X5_CFG

from . import events, rewards, terminations
from .actions import HighLevelActionsCfg
from .observations import ObservationsCfg


@configclass
class Go2X5DoorSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING

    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.95, 0.28, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"door_left_joint": 0.0, "door_right_joint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                stiffness=30.0,
                damping=4.0,
            )
        },
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
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_stage = EventTerm(func=events.reset_staged_task_state, mode="reset", params={"num_stages": 4})
    reset_door = EventTerm(func=events.reset_door_joint, mode="reset")
    randomize_door_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door"),
            "static_friction_range": (0.5, 1.1),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.03),
            "num_buckets": 16,
        },
    )


@configclass
class RewardsCfg:
    door_angle = RewTerm(func=rewards.door_angle_reward, params={"target_angle": 0.7}, weight=4.0)
    handle_alignment = RewTerm(func=rewards.handle_alignment_reward, params={"std": 0.20}, weight=2.0)
    base_stability = RewTerm(func=rewards.base_stability_reward, weight=0.5)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(func=terminations.robot_fallen, params={"minimum_height": 0.18})
    success = DoneTerm(func=terminations.door_open_success, params={"target_angle": 0.7})


@configclass
class Go2X5DoorEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2X5DoorSceneCfg = Go2X5DoorSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: HighLevelActionsCfg = HighLevelActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
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
        self.viewer.lookat = (0.75, 0.2, 0.65)


@configclass
class Go2X5DoorEnvCfg_PLAY(Go2X5DoorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
