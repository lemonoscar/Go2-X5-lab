"""Isolated unified DogOnly continuation terms; original tasks are unchanged."""
import math
import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from .unified_contract import ShortHistory, filter_command, gait_frequency, gait_swing_mask


class UnifiedVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.requested = self.vel_command_b.clone()
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.anchor = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw_anchor = self.phase.clone()
        self.was_standing = torch.zeros_like(self.is_standing_env)
        self.motion_released = torch.zeros_like(self.is_standing_env)
        self.history = ShortHistory(self.num_envs, self.device)
        self.joint_ids = self.robot.find_joints(cfg.joint_names, preserve_order=True)[0]
        self.arm_ids = self.robot.find_joints([f'arm_joint{i}' for i in range(1, 7)], preserve_order=True)[0]
        self.metrics['arm_stow_error'] = self.phase.clone()
        self.metrics['waiting_for_stow'] = self.phase.clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.was_standing[ids] = False
        self.motion_released[ids] = False
        self.phase[ids] = 0
        self.history.reset(ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Explicit pure-axis bins, including pure yaw, alongside mixed commands.
        category = torch.randint(0, 4, (len(ids),), device=self.device)
        c = self.vel_command_b[ids].clone()
        for axis in range(3):
            mask = category == axis
            c[mask] *= torch.nn.functional.one_hot(torch.tensor(axis, device=self.device), 3)
        c = filter_command(c)
        c[self.is_standing_env[ids]] = 0
        # Small positive/negative samples in a dead zone legitimately become standing.
        self.is_standing_env[ids] = (c == 0).all(-1)
        self.requested[ids] = c
        self.vel_command_b[ids] = c
        self.time_left[ids[self.is_standing_env[ids]]] = self.cfg.standing_seconds

    def _update_command(self):
        self.vel_command_b.copy_(self.requested)
        # Withhold locomotion until both measured arm state and arm target are stowed.
        q = self.robot.data.joint_pos[:, self.arm_ids]
        q0 = self.robot.data.default_joint_pos[:, self.arm_ids]
        arm = self._env.command_manager.get_term('arm_joint_pos')
        error = (q - q0).abs().amax(-1)
        target_ready = (arm.command - q0).abs().amax(-1) < 0.03
        # Measured baseline static PD error is 0.05--0.069 rad. A 0.06 gate
        # permanently suppressed valid walking commands. Use measured tolerance
        # and hysteresis, so normal tracking ripple does not flicker the mode.
        ready = (error < self.cfg.stow_tolerance) & target_ready
        ready &= self.robot.data.joint_vel[:, self.arm_ids].abs().amax(-1) < 0.15
        self.motion_released |= ready
        self.motion_released &= (self.requested != 0).any(-1) & target_ready & (error < 0.20)
        self.metrics['arm_stow_error'].copy_((q-q0).abs().amax(-1))
        self.metrics['waiting_for_stow'].copy_(((self.requested != 0).any(-1) & ~self.motion_released).float())
        self.vel_command_b[~self.motion_released] = 0
        standing = (self.vel_command_b == 0).all(-1)
        entered = standing & ~self.was_standing
        self.anchor[entered] = self.robot.data.root_pos_w[entered]
        yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)[2]
        self.yaw_anchor[entered] = yaw[entered]
        frequency = gait_frequency(self.vel_command_b, self.cfg.gait_frequency_min, self.cfg.gait_frequency_max)
        self.phase = torch.where(standing, 0.0, (self.phase + self._env.step_dt * frequency) % 1.0)
        self.was_standing.copy_(standing)


@configclass
class UnifiedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UnifiedVelocityCommand
    gait_frequency_min: float = 0.7
    gait_frequency_max: float = 2.0
    swing_fraction: float = 0.20
    standing_seconds: float = 65.0
    stow_tolerance: float = 0.10
    joint_names: list[str] = []


class UnifiedArmCommand(CommandTerm):
    """Rate-limited arm excitation for the first stabilization curriculum.

    This initial stage is deliberately a joint-space disturbance curriculum,
    not a claim of world-TCP task completion. Full task trajectories are a
    separate acceptance gate, documented in the runbook.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.ids = self.robot.find_joints([f'arm_joint{i}' for i in range(1, 7)], preserve_order=True)[0]
        self.target = self.robot.data.default_joint_pos[:, self.ids].clone()
        self.offset = torch.zeros_like(self.target)
        self.clock = torch.zeros(self.num_envs, device=self.device)
        self.ready_time = self.clock.clone()

    @property
    def command(self):
        return self.target

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        self.offset[env_ids] = (torch.rand(len(env_ids), 6, device=self.device) * 2 - 1) * self.cfg.amplitude

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.target[ids] = self.robot.data.default_joint_pos[ids][:, self.ids]
        self.clock[ids] = 0
        self.ready_time[ids] = 0
        return super().reset(env_ids)

    def _update_command(self):
        base = self._env.command_manager.get_term('base_velocity')
        requested_stand = (base.requested == 0).all(-1)
        settled = self.robot.data.root_lin_vel_b[:, :2].norm(dim=-1) < 0.1
        self.ready_time = torch.where(requested_stand & settled, self.ready_time + self._env.step_dt, 0)
        active = requested_stand & (self.ready_time >= 0.5)
        self.clock += self._env.step_dt
        # Slow out-and-back movement with bounded target velocity. No joint teleports.
        envelope = 0.5 - 0.5 * torch.cos(2 * math.pi * self.clock / 12.0)
        default = self.robot.data.default_joint_pos[:, self.ids]
        desired = default + self.offset * envelope[:, None] * active[:, None]
        limits = self.robot.data.soft_joint_pos_limits[:, self.ids]
        desired = torch.max(torch.min(desired, limits[..., 1]), limits[..., 0])
        step = self.cfg.speed * self._env.step_dt
        self.target += (desired - self.target).clamp(-step, step)


@configclass
class UnifiedArmCommandCfg(CommandTermCfg):
    class_type: type = UnifiedArmCommand
    asset_name: str = 'robot'
    amplitude: float = 0.15
    speed: float = 0.35
    resampling_time_range: tuple[float, float] = (12.0, 12.0)


def unified_features(env):
    term = env.command_manager.get_term('base_velocity')
    robot = term.robot
    q = robot.data.joint_pos[:, term.joint_ids] - robot.data.default_joint_pos[:, term.joint_ids]
    dq = robot.data.joint_vel[:, term.joint_ids]
    frame = torch.cat([robot.data.root_ang_vel_b, robot.data.projected_gravity_b, q, dq,
                       env.action_manager.action[:, :12]], dim=-1)
    history = term.history.update(frame, env.common_step_counter)
    standing = (term.command == 0).all(-1)
    delta = quat_apply_inverse(robot.data.root_quat_w, term.anchor - robot.data.root_pos_w)[:, :2]
    yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2] - term.yaw_anchor
    phase = 2 * math.pi * term.phase
    features = torch.stack([phase.sin(), phase.cos(), standing.float(),
                            delta[:, 0], delta[:, 1], yaw.sin(), yaw.cos() - 1], dim=-1)
    features[:, :2] *= (~standing)[:, None]
    features[:, 3:] *= standing[:, None]
    return torch.cat([features, history], dim=-1)


def standing_cost(env, kind):
    term = env.command_manager.get_term('base_velocity')
    robot = term.robot
    standing = (term.command == 0).all(-1)
    if kind == 'position':
        cost = ((robot.data.root_pos_w[:, :2] - term.anchor[:, :2]).norm(dim=-1) / 0.05).square()
    elif kind == 'velocity':
        cost = (robot.data.root_lin_vel_b[:, :2].norm(dim=-1) / 0.1).square()
    elif kind == 'posture':
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        # euler_xyz_from_quat may return angles in [0,2pi).
        roll = torch.atan2(roll.sin(), roll.cos())
        pitch = torch.atan2(pitch.sin(), pitch.cos())
        dyaw = yaw - term.yaw_anchor
        dyaw = torch.atan2(dyaw.sin(), dyaw.cos())
        cost = ((roll.abs() - math.radians(5)).clamp_min(0) / math.radians(5)).square()
        cost += ((pitch.abs() - math.radians(10)).clamp_min(0) / math.radians(5)).square()
        cost += (dyaw / math.radians(5)).square()
    else:
        raise ValueError(kind)
    return cost.clamp_max(25) * standing


def four_beat_reward(env, balanced=False):
    """WTW-style swing-force/stance-speed shaping, soft four-beat schedule."""
    term = env.command_manager.get_term('base_velocity')
    if not hasattr(term, 'feet_ids'):
        names = ['RL_foot', 'FL_foot', 'RR_foot', 'FR_foot']
        term.feet_ids = term.robot.find_bodies(names, preserve_order=True)[0]
        term.sensor_ids = env.scene['contact_forces'].find_bodies(names, preserve_order=True)[0]
    swing = gait_swing_mask(term.phase, term.cfg.swing_fraction)
    sensor = env.scene['contact_forces']
    force = sensor.data.net_forces_w[:, term.sensor_ids].norm(dim=-1)
    velocity = term.robot.data.body_lin_vel_w[:, term.feet_ids].norm(dim=-1)
    score = torch.where(swing, torch.exp(-(force / 30).square()), torch.exp(-(velocity / 0.2).square()))
    moving = (term.command != 0).any(-1)
    upright = (-term.robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
    if balanced:
        # Give the scheduled swing and all support feet equal influence.
        swing_score = (score * swing).sum(-1) / swing.sum(-1).clamp_min(1)
        stance_score = (score * ~swing).sum(-1) / (~swing).sum(-1).clamp_min(1)
        value = torch.where(swing.any(-1), 0.5 * (swing_score + stance_score), stance_score)
    else:
        value = score.mean(-1)
    return value * moving * upright


def moving_action_rate(env):
    """Soft leg-target smoothing only while a locomotion command is applied."""
    moving = (env.command_manager.get_term('base_velocity').command != 0).any(-1)
    delta = env.action_manager.action - env.action_manager.prev_action
    return delta.square().sum(-1) * moving
