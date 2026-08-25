"""Single-environment flat Go2-X5 WholeBody inference task."""

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab.envs import mdp as core_mdp

from robot_lab.go2_x5_wholebody.controller import (
    ARM_JOINT_NAMES,
    ARM_POLICY_ZERO,
    DOG_DEFAULT_JOINT_POS,
    DOG_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.go2_x5.wtw_flat_env_cfg import (
    Go2X5WTWFlatEnvCfg,
)

from .actions import Go2X5WholeBodyActionCfg


@configclass
class WholeBodyActionsCfg:
    whole_body = Go2X5WholeBodyActionCfg()


@configclass
class WholeBodyObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=core_mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2X5WholeBodyEnvCfg(Go2X5WTWFlatEnvCfg):
    """Inference-only shell whose Gym action is exactly the public 10-D command."""

    expected_action_dim: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**12

        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(dict(zip(DOG_JOINT_NAMES, DOG_DEFAULT_JOINT_POS, strict=True)))
        joint_pos.update(dict(zip(ARM_JOINT_NAMES, ARM_POLICY_ZERO, strict=True)))
        joint_pos.update(dict(zip(GRIPPER_JOINT_NAMES, (0.0, 0.0), strict=True)))
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            pos=(0.0, 0.0, 0.34), joint_pos=joint_pos
        )
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.actuators = {
            "legs_hip_thigh": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                effort_limit=23.7,
                effort_limit_sim=23.7,
                saturation_effort=23.7,
                velocity_limit=30.1,
                velocity_limit_sim=30.1,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
            ),
            "legs_calf": DCMotorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit=45.43,
                effort_limit_sim=45.43,
                saturation_effort=45.43,
                velocity_limit=15.7,
                velocity_limit_sim=15.7,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
            ),
            "arm_joint1": DCMotorCfg(
                joint_names_expr=["arm_joint1"], effort_limit=30.0, effort_limit_sim=30.0,
                saturation_effort=30.0, velocity_limit=10.0, velocity_limit_sim=10.0,
                stiffness=40.0, damping=3.0, friction=0.0,
            ),
            "arm_joint23": DCMotorCfg(
                joint_names_expr=["arm_joint2", "arm_joint3"], effort_limit=30.0,
                effort_limit_sim=30.0, saturation_effort=30.0, velocity_limit=10.0,
                velocity_limit_sim=10.0, stiffness=70.0, damping=15.0, friction=0.0,
            ),
            "arm_joint456": DCMotorCfg(
                joint_names_expr=["arm_joint4", "arm_joint5", "arm_joint6"], effort_limit=30.0,
                effort_limit_sim=30.0, saturation_effort=30.0, velocity_limit=10.0,
                velocity_limit_sim=10.0, stiffness=25.0, damping=2.0, friction=0.0,
            ),
            "gripper": DCMotorCfg(
                joint_names_expr=["arm_joint7", "arm_joint8"], effort_limit=20.0,
                effort_limit_sim=20.0, saturation_effort=20.0, velocity_limit=1.0,
                velocity_limit_sim=1.0, stiffness=50.0, damping=20.0, friction=0.05,
            ),
        }

        self.actions = WholeBodyActionsCfg()
        self.observations = WholeBodyObservationsCfg()
        self.commands.base_velocity = None
        self.commands.arm_joint_pos = None
        self.commands.gripper_joint_pos = None
        deterministic_reset_events = {"randomize_reset_joints", "randomize_reset_base"}
        for name in dir(self.events):
            value = getattr(self.events, name)
            if (
                not name.startswith("_")
                and name not in deterministic_reset_events
                and value is not None
                and hasattr(value, "func")
            ):
                setattr(self.events, name, None)
        for name in dir(self.rewards):
            value = getattr(self.rewards, name)
            if not name.startswith("_") and value is not None and hasattr(value, "weight"):
                setattr(self.rewards, name, None)
        for name in dir(self.terminations):
            if name != "time_out" and not name.startswith("_"):
                value = getattr(self.terminations, name)
                if value is not None and hasattr(value, "func"):
                    setattr(self.terminations, name, None)
        for name in dir(self.curriculum):
            if not name.startswith("_"):
                value = getattr(self.curriculum, name)
                if value is not None and hasattr(value, "func"):
                    setattr(self.curriculum, name, None)

        if abs(self.sim.dt * self.decimation - 0.02) > 1.0e-9:
            raise ValueError("WholeBody task must run at 50 Hz.")
        if self.scene.num_envs != 1:
            raise ValueError("WholeBody Pink IK task must use exactly one environment.")
        if any(
            value is not None
            for value in (
                self.commands.base_velocity,
                self.commands.arm_joint_pos,
                self.commands.gripper_joint_pos,
            )
        ):
            raise ValueError("WholeBody public actions must be the only command source.")
