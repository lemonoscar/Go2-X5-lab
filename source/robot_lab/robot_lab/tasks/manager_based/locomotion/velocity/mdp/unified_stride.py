"""Measured contact events for the isolated flat gait-repair experiment."""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse


class ShortStepCost(ManagerTermBase):
    """Hysteretic contacts; bounded costs, never a hard ban on recovery steps."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.sensor = env.scene['contact_forces']
        names = ['RL_foot', 'FL_foot', 'RR_foot', 'FR_foot']
        self.feet = self.robot.find_bodies(names, preserve_order=True)[0]
        self.feet_sensor = self.sensor.find_bodies(names, preserve_order=True)[0]
        self.contact = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.bool)
        self.duration = torch.zeros(env.num_envs, 4, device=env.device)
        self.lift_position = torch.zeros(env.num_envs, 4, 2, device=env.device)
        self.valid_swing = torch.zeros_like(self.contact)
        self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.initialized[ids] = False
        self.valid_swing[ids] = False
        self.duration[ids] = 0

    def __call__(self, env):
        term = env.command_manager.get_term('base_velocity')
        force = self.sensor.data.net_forces_w[:, self.feet_sensor].norm(dim=-1)
        contact = torch.where(self.contact, force > 3., force > 10.)
        moving = (term.command != 0).any(-1)
        valid = self.initialized[:, None] & moving[:, None]
        touchdown = contact & ~self.contact & valid
        liftoff = ~contact & self.contact & valid
        delta = self.robot.data.body_pos_w[:, self.feet] - self.robot.data.root_pos_w[:, None]
        q = self.robot.data.root_quat_w[:, None].expand(-1, 4, -1)
        position = quat_apply_inverse(q.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, 4, 3)[..., :2]
        command = term.command[:, :2]
        speed = command.norm(dim=-1)
        direction = command / speed.clamp_min(1.e-6)[:, None]
        excursion = ((position - self.lift_position) * direction[:, None]).sum(-1)
        # At vx=.5: 0.14s minimum swing, 0.22s stance, 0.15m body-relative excursion.
        min_air = (0.18 - 0.08 * speed).clamp(0.12, 0.18)[:, None]
        target = (0.30 * speed).clamp_max(0.20)[:, None]
        air_cost = ((min_air - self.duration) / min_air).clamp(0, 1)
        stance_cost = ((0.22 - self.duration) / 0.22).clamp(0, 1)
        stride_cost = ((target - excursion) / target.clamp_min(0.03)).clamp(0, 1)
        stride_cost *= (speed > 0.1)[:, None]  # No translational stride demand for pure yaw.
        cost = ((air_cost + stride_cost) * touchdown * self.valid_swing + stance_cost * liftoff).sum(-1)
        # De-emphasize the preference during balance recovery, but never disable tracking.
        recovery = (self.robot.data.projected_gravity_b[:, :2].norm(dim=-1) < 0.20)
        cost *= torch.where(recovery, 1., 0.25)
        self.lift_position[liftoff] = position[liftoff]
        self.valid_swing |= liftoff
        self.valid_swing &= ~touchdown & moving[:, None]
        changed = contact != self.contact
        self.duration = torch.where(changed | ~valid, 0., self.duration) + env.step_dt
        self.contact.copy_(contact)
        self.initialized[:] = True
        # RewardManager multiplies by dt: this is a per-event cost, independent of policy rate.
        return cost / env.step_dt
