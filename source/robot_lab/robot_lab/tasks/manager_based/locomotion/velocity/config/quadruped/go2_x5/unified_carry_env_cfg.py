"""New 26000 continuation tasks. Original DogOnly task contracts stay intact."""
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.unified_carry import (
    UnifiedArmCommandCfg, UnifiedVelocityCommandCfg, four_beat_reward,
    standing_cost, unified_features,
)
from .train_route_env_cfg import (
    Go2X5DogOnlyRoughEnvCfg, DOG_ONLY_ROUGH_CHECKPOINT_JOINT_POS,
    DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT,
)
from .agents.rsl_rl_ppo_cfg import Go2X5DogOnlyRoughPPORunnerCfg


@configclass
class UnifiedCarryFlatEnvCfg(Go2X5DogOnlyRoughEnvCfg):
    """Stage 1: flat locomotion + small arm disturbance, before payload tasks."""
    def __post_init__(self):
        super().__post_init__()
        joints = dict(self.scene.robot.init_state.joint_pos)
        joints.update(DOG_ONLY_ROUGH_CHECKPOINT_JOINT_POS)
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            joint_pos=joints, pos=(0, 0, DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT))
        self.scene.num_envs = 1024
        self.scene.terrain.terrain_type = 'plane'
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        # Headless physics must not wait for remote HDR/MDL visual downloads.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35))
        self.scene.light.spawn.texture_file = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.episode_length_s = 70.0
        self.commands.base_velocity = UnifiedVelocityCommandCfg(
            asset_name='robot', joint_names=self.joint_names,
            resampling_time_range=(5.0, 8.0), rel_standing_envs=0.35,
            rel_heading_envs=0.0, heading_command=False, debug_vis=False,
            ranges=UnifiedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.75), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.3, 0.3)))
        self.commands.arm_joint_pos = UnifiedArmCommandCfg(debug_vis=False)
        for group in (self.observations.policy, self.observations.critic):
            group.unified_features = ObsTerm(func=unified_features, clip=(-100.0, 100.0))
        # Remove trot and motion-suppressing standing shaping in this new task only.
        self.rewards.feet_gait = None
        self.rewards.stand_still = None
        self.rewards.feet_contact_without_cmd = None
        self.rewards.flat_orientation_l2.weight = -0.1
        self.rewards.joint_pos_penalty.weight = -0.05
        self.rewards.joint_pos_penalty.params['stand_still_scale'] = 1.0
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.four_beat = RewTerm(func=four_beat_reward, weight=0.3)
        self.rewards.standing_position = RewTerm(func=standing_cost, weight=-0.6, params={'kind': 'position'})
        self.rewards.standing_velocity = RewTerm(func=standing_cost, weight=-0.4, params={'kind': 'velocity'})
        self.rewards.standing_posture = RewTerm(func=standing_cost, weight=-0.4, params={'kind': 'posture'})
        self.rewards.base_height_l2.params['target_height'] = DOG_ONLY_ROUGH_CHECKPOINT_BASE_HEIGHT
        # Preserve the source controller (40/1 and 0.25 action scaling), 50 Hz policy.
        self.sim.dt = 0.0025
        self.decimation = 8
        self.events.randomize_reset_base.params['pose_range'].update(z=(0, 0), roll=(0, 0), pitch=(0, 0))
        self.events.randomize_reset_base.params['velocity_range'] = {k: (0, 0) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
        # First stage deliberately excludes carrying: no claim of full task acceptance.
        self.disable_zero_weight_rewards()


@configclass
class UnifiedBaselineFlatEnvCfg(UnifiedCarryFlatEnvCfg):
    """Original network dimensions, identical 26000 controller, no arm excitation."""
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.unified_features = None
        self.observations.critic.unified_features = None
        self.commands.arm_joint_pos.amplitude = 0.0
        self.observations.policy.enable_corruption = False


@configclass
class UnifiedCarryPPORunnerCfg(Go2X5DogOnlyRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'go2_x5_26000_unified_stage1'
        self.max_iterations = 1500
        self.save_interval = 100
        self.algorithm.learning_rate = 5e-5
        self.algorithm.schedule = 'fixed'
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.01
