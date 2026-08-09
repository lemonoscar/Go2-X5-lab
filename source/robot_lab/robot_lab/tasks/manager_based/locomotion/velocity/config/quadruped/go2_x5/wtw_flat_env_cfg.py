"""Deterministic Go2-X5 environment for WTW PD40/1 continuation training."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .train_route_env_cfg import (
    ARM_LOCKED_DEFAULT_RANGE,
    FLAT_FOUNDATION_TERRAIN_CFG,
    Go2X5DogOnlyFlatEnvCfg,
)


@configclass
class WTWPolicyObservationCfg(ObsGroup):
    """One 70-D compound frame with a single oldest-first 30-frame history."""

    wtw_frame = ObsTerm(
        func=mdp.wtw_observation_frame,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(mdp.WTW_JOINT_NAMES), preserve_order=True
            ),
        },
        clip=(-mdp.WTW_OBSERVATION_CLIP, mdp.WTW_OBSERVATION_CLIP),
        history_length=mdp.WTW_HISTORY_LENGTH,
        flatten_history_dim=True,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2X5WTWFlatEnvCfg(Go2X5DogOnlyFlatEnvCfg):
    """Warm-start the 12-D WTW actor on the full fixed-arm Go2-X5 asset."""

    expected_policy_observation_dim: int = mdp.WTW_FRAME_DIM * mdp.WTW_HISTORY_LENGTH
    expected_critic_observation_dim: int = 260
    expected_action_dim: int = mdp.WTW_ACTION_DIM

    def __post_init__(self):
        super().__post_init__()

        # Exact target timing and nominal Go2-X5 PD controller.
        self.scene.num_envs = 512
        self.episode_length_s = 20.0
        self.decimation = 8
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.history_length = self.decimation
        for actuator_name in ("legs_hip_thigh", "legs_calf"):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.stiffness = 40.0
            actuator.damping = 1.0

        # Generate the flat collision mesh locally. Isaac Lab's built-in "plane"
        # type fetches its USD from NVIDIA S3, which makes headless training depend
        # on external asset availability despite the terrain being featureless.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_FOUNDATION_TERRAIN_CFG.copy()
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.visual_material = None
        self.scene.sky_light.spawn.texture_file = None

        # Source WTW default pose. The arm remains the real, fixed payload.
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(dict(zip(mdp.WTW_JOINT_NAMES, mdp.WTW_DEFAULT_JOINT_POS, strict=True)))
        joint_pos.update(
            dict(
                zip(
                    mdp.WTW_GRIPPER_JOINT_NAMES,
                    mdp.WTW_GRIPPER_DEFAULT_JOINT_POS,
                    strict=True,
                )
            )
        )
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            pos=(0.0, 0.0, 0.30), joint_pos=joint_pos
        )

        # WTW controls only FL/FR/RL/RR leg joints. ArmCommandPositionAction remains
        # a zero-dimensional external action term inherited from DogOnly.
        self.actions.joint_pos.joint_names = list(mdp.WTW_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
        }
        self.actions.joint_pos.clip = {".*": (-mdp.WTW_ACTION_CLIP, mdp.WTW_ACTION_CLIP)}
        self.commands.arm_joint_pos.position_range = ARM_LOCKED_DEFAULT_RANGE
        self.commands.arm_joint_pos.resampling_time_range = (20.0, 20.0)
        self.commands.gripper_joint_pos = mdp.ArmJointPositionCommandCfg(
            asset_name="robot",
            joint_names=list(mdp.WTW_GRIPPER_JOINT_NAMES),
            resampling_time_range=(20.0, 20.0),
            position_range=[(0.0, 0.0)] * len(mdp.WTW_GRIPPER_JOINT_NAMES),
            use_default_offset=True,
            clip_to_joint_limits=False,
            preserve_order=True,
        )
        self.actions.gripper_joint_pos = mdp.ArmCommandPositionActionCfg(
            asset_name="robot",
            joint_names=list(mdp.WTW_GRIPPER_JOINT_NAMES),
            command_name="gripper_joint_pos",
            preserve_order=True,
        )

        self.commands.base_velocity = mdp.WTWWalkingVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            standing_probability=0.20,
            ranges=mdp.WTWWalkingVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.75, 0.75),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-0.50, 0.50),
                heading=None,
            ),
        )

        # The actor sees exactly one 30x70 WTW term. The inherited critic group
        # intentionally stays at the existing 260-D full-state DogOnly contract.
        self.observations.policy = WTWPolicyObservationCfg()

        # R0 isolates the fixed payload/controller: command sampling is the only
        # stochastic task input. Reset pose and velocity remain deterministic.
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.sim2sim_action_delay_range = (0, 0)
        self.sim2sim_action_hold_prob = 0.0
        self.sim2sim_action_noise_std = 0.0
        self.sim2sim_obs_delay_steps = 0

        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.reward_weights = None
        self.curriculum.arm_command_range = None

        # Preserve walking quality while giving the explicit STAND branch the
        # inherited joint-stillness and four-foot-contact objectives.
        self.rewards.base_height_l2.params["target_height"] = 0.30

        # Thigh/calf contacts remain a penalty. Only clear base/arm contacts,
        # overturning, and low base height end an episode during continuation.
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base", "arm_.*"]
        self.terminations.terrain_out_of_bounds = None
        self.terminations.root_height_above_maximum = None
        self.terminations.root_lin_vel_z_above_maximum = None
        self.terminations.root_ang_vel_xy_above_maximum = None

        self._validate_wtw_contract()

    def _validate_wtw_contract(self) -> None:
        if not math.isclose(self.sim.dt * self.decimation, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("WTW task must run at a 0.02 s policy period.")
        if tuple(self.actions.joint_pos.joint_names) != mdp.WTW_JOINT_NAMES:
            raise ValueError("WTW joint action order must be FL, FR, RL, RR.")
        if self.observations.policy.wtw_frame.history_length != mdp.WTW_HISTORY_LENGTH:
            raise ValueError("WTW policy history must contain exactly 30 compound frames.")
        if self.observations.policy.wtw_frame.flatten_history_dim is not True:
            raise ValueError("WTW policy history must be flattened oldest-first.")
        if self.actions.arm_joint_pos.class_type is not mdp.ArmCommandPositionAction:
            raise ValueError("WTW arm control must remain outside the policy action head.")
        if self.actions.gripper_joint_pos.class_type is not mdp.ArmCommandPositionAction:
            raise ValueError("WTW gripper control must remain outside the policy action head.")
        if tuple(self.actions.gripper_joint_pos.joint_names) != mdp.WTW_GRIPPER_JOINT_NAMES:
            raise ValueError("WTW gripper action must preserve arm_joint7/arm_joint8 order.")
        if self.actions.gripper_joint_pos.command_name != "gripper_joint_pos":
            raise ValueError("WTW gripper action must use the fixed gripper command term.")
        if not math.isclose(
            self.commands.base_velocity.standing_probability,
            0.20,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("WTW continuation must sample the STAND branch at 20%.")
        if self.rewards.stand_still is None or self.rewards.feet_contact_without_cmd is None:
            raise ValueError("WTW STAND training rewards must remain enabled.")
        if (
            self.commands.base_velocity is None
            or self.commands.arm_joint_pos is None
            or self.commands.gripper_joint_pos is None
        ):
            raise ValueError("WTW evaluator requires base, fixed-arm, and fixed-gripper command terms.")
        if tuple(self.commands.gripper_joint_pos.joint_names) != mdp.WTW_GRIPPER_JOINT_NAMES:
            raise ValueError("WTW gripper command must preserve arm_joint7/arm_joint8 order.")
        if not self.commands.gripper_joint_pos.use_default_offset:
            raise ValueError("WTW gripper command must hold the articulation default pose.")
        gripper_ranges = tuple(tuple(value) for value in self.commands.gripper_joint_pos.position_range)
        if gripper_ranges != ((0.0, 0.0), (0.0, 0.0)):
            raise ValueError("WTW gripper command offset must remain fixed at zero.")
        gripper_init = tuple(
            self.scene.robot.init_state.joint_pos[name] for name in mdp.WTW_GRIPPER_JOINT_NAMES
        )
        if gripper_init != mdp.WTW_GRIPPER_DEFAULT_JOINT_POS:
            raise ValueError("WTW gripper must initialize open at 0.044 m per finger.")
        if self.scene.contact_forces is None:
            raise ValueError("WTW evaluator requires the full-body contact sensor.")
        if self.scene.contact_forces.history_length != self.decimation:
            raise ValueError("WTW contact history must cover one complete policy interval.")
        if self.scene.terrain.terrain_type != "generator" or self.scene.terrain.terrain_generator is None:
            raise ValueError("WTW flat terrain must be generated locally without a remote plane USD.")
        if self.scene.terrain.use_terrain_origins is not False:
            raise ValueError("WTW flat terrain must place environments on the scene grid.")
        if self.scene.terrain.visual_material is not None:
            raise ValueError("WTW flat terrain must not require a remote visual material.")
        if self.scene.sky_light.spawn.texture_file is not None:
            raise ValueError("WTW headless training must not require a remote sky texture.")
        for actuator_name in ("legs_hip_thigh", "legs_calf"):
            actuator = self.scene.robot.actuators[actuator_name]
            if actuator.stiffness != 40.0 or actuator.damping != 1.0:
                raise ValueError(f"{actuator_name} must use fixed PD 40/1.")
