# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

GO2_X5_URDF_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/go2_x5/go2_x5.urdf"
GO2_X5_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/go2_x5/go2_x5.usd"

_GO2_X5_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)

_GO2_X5_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
)

if os.path.exists(GO2_X5_USD_PATH):
    _GO2_X5_SPAWN_CFG = sim_utils.UsdFileCfg(
        usd_path=GO2_X5_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=_GO2_X5_RIGID_PROPS,
        articulation_props=_GO2_X5_ARTICULATION_PROPS,
    )
else:
    _GO2_X5_SPAWN_CFG = sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=GO2_X5_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=_GO2_X5_RIGID_PROPS,
        articulation_props=_GO2_X5_ARTICULATION_PROPS,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    )

GO2_X5_CFG = ArticulationCfg(
    spawn=_GO2_X5_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            # Go2-X5 default pose aligned to the UMI-on-Legs Go2-ARX5 controller offset.
            "FR_hip_joint": 0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FL_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            # UMI-on-Legs env_go2ARX5 default arm offset.
            "arm_joint1": 0.0,
            "arm_joint2": 0.3,
            "arm_joint3": 0.5,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=35.278,
            saturation_effort=35.278,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=44.4,
            saturation_effort=44.4,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "arm_joint12": DCMotorCfg(
            joint_names_expr=["arm_joint1", "arm_joint2"],
            effort_limit=20.0,
            saturation_effort=20.0,
            velocity_limit=3.0,
            stiffness=100.0,
            damping=3.0,
            friction=0.0,
        ),
        "arm_joint3": DCMotorCfg(
            joint_names_expr=["arm_joint3"],
            effort_limit=15.0,
            saturation_effort=15.0,
            velocity_limit=3.0,
            stiffness=100.0,
            damping=3.0,
            friction=0.0,
        ),
        "arm_joint4": DCMotorCfg(
            joint_names_expr=["arm_joint4"],
            effort_limit=7.0,
            saturation_effort=7.0,
            velocity_limit=3.0,
            stiffness=20.0,
            damping=2.0,
            friction=0.0,
        ),
        "arm_joint5": DCMotorCfg(
            joint_names_expr=["arm_joint5"],
            effort_limit=5.0,
            saturation_effort=5.0,
            velocity_limit=3.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.0,
        ),
        "arm_joint6": DCMotorCfg(
            joint_names_expr=["arm_joint6"],
            effort_limit=5.0,
            saturation_effort=5.0,
            velocity_limit=3.0,
            stiffness=5.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC motor.
"""
