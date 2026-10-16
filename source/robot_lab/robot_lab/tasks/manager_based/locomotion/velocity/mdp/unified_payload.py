"""Rigidly held TCP payload and terrain curriculum, without payload labels in observations."""
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .unified_carry import UnifiedVelocityCommand, UnifiedVelocityCommandCfg, UnifiedArmCommand, UnifiedArmCommandCfg
from .unified_contract import combine_payload_inertia
from .utils import is_env_assigned_to_terrain


class PayloadVelocityCommand(UnifiedVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        rough = ~is_env_assigned_to_terrain(self._env, 'flat')[ids]
        self.requested[ids[(~rough) & (torch.rand(len(ids), device=self.device) < 0.7)]] = 0
        stopped = (self.requested[ids] == 0).all(-1)
        self.requested[ids[rough & stopped], 0] = 0.25
        self.vel_command_b[ids] = self.requested[ids]
        self.is_standing_env[ids] = (self.requested[ids] == 0).all(-1)


@configclass
class PayloadVelocityCommandCfg(UnifiedVelocityCommandCfg):
    class_type: type = PayloadVelocityCommand


class PayloadArmCommand(UnifiedArmCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.offset[ids[~is_env_assigned_to_terrain(self._env, 'flat')[ids]]] = 0


@configclass
class PayloadArmCommandCfg(UnifiedArmCommandCfg):
    class_type: type = PayloadArmCommand


def randomize_tcp_payload(env, env_ids, mass_range=(0.005, 0.45), empty_probability=0.2):
    """Equivalent rigid sphere at grasp TCP; no grasp/slip/contact simulation.

    Restore cached unloaded properties before each draw. Existing startup DR is
    retained. COM frame orientation is preserved; inertia uses that same frame.
    """
    robot = env.scene['robot']; view = robot.root_physx_view
    ids = torch.arange(env.num_envs) if env_ids is None else env_ids.cpu()
    if not hasattr(env, '_tcp_payload_original'):
        body = robot.find_bodies('arm_link6', preserve_order=True)[0]
        if len(body) != 1:
            raise RuntimeError('Expected exactly one arm_link6 payload body')
        env._tcp_payload_original = (body[0], view.get_masses().clone(), view.get_coms().clone(), view.get_inertias().clone())
        env.tcp_payload_mass = torch.zeros(env.num_envs, device=env.device)
    body, mass0, com0, inertia0 = env._tcp_payload_original
    mass, com, inertia = view.get_masses(), view.get_coms(), view.get_inertias()
    payload = torch.empty(len(ids)).uniform_(*mass_range)
    payload[torch.rand(len(ids)) < empty_probability] = 0
    tcp = com0[ids, body, :3].new_tensor([0.15757, 0., 0.]).expand(len(ids), -1)
    delta_body = tcp - com0[ids, body, :3]
    # PhysX COM pose quaternion is xyzw, Isaac utilities use wxyz.
    q = com0[ids, body, 3:7][:, [3, 0, 1, 2]]
    delta_com = quat_apply_inverse(q, delta_body)
    total, combined = combine_payload_inertia(mass0[ids, body], inertia0[ids, body].reshape(-1, 3, 3), payload, delta_com)
    mass[ids, body] = total
    com[ids, body] = com0[ids, body]
    com[ids, body, :3] += delta_body * (payload / total)[:, None]
    inertia[ids, body] = combined.flatten(1)
    view.set_masses(mass, ids); view.set_coms(com, ids); view.set_inertias(inertia, ids)
    env.tcp_payload_mass[ids.to(env.device)] = payload.to(env.device)


def patch_complete(env):
    xy = env.scene['robot'].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return xy.abs().amax(-1) > 6.0


def payload_terrain_levels(env, env_ids):
    # Before the first reset, root state is not located at its terrain origin.
    if env.common_step_counter == 0:
        return env.scene.terrain.terrain_levels.float().mean()
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    flat = is_env_assigned_to_terrain(env, 'flat')[ids]
    completed = patch_complete(env)[ids]
    failed = env.termination_manager.terminated[ids]
    env.scene.terrain.update_env_origins(ids, completed & ~flat & ~failed, failed & ~flat)
    return env.scene.terrain.terrain_levels.float().mean()


class TerrainFocusVelocityCommand(PayloadVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        flat = is_env_assigned_to_terrain(self._env, 'flat')[ids]
        self.requested[ids[flat]] = 0
        forward = ~flat & (torch.rand(len(ids), device=self.device) < 0.8)
        self.requested[ids[forward]] = self.requested.new_tensor([0.5, 0., 0.])
        self.vel_command_b[ids] = self.requested[ids]
        self.is_standing_env[ids] = flat


@configclass
class TerrainFocusVelocityCommandCfg(PayloadVelocityCommandCfg):
    class_type: type = TerrainFocusVelocityCommand


class FullRangeArmCommand(PayloadArmCommand):
    """Full soft joint limits, with continuous out/hold/back targets and bounded speed.

    Joint-space stress coverage, not a collision-certified TCP workspace planner.
    """
    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        limits = self.robot.data.soft_joint_pos_limits[ids][:, self.ids]
        sample = torch.rand(len(ids), 6, device=self.device)
        # Half of joint targets are exact soft-limit endpoints; half cover the interior.
        sample = torch.where(sample < .25, 0., torch.where(sample > .75, 1., (sample-.25)*2))
        target = limits[..., 0] + sample * (limits[..., 1] - limits[..., 0])
        self.offset[ids] = target - self.robot.data.default_joint_pos[ids][:, self.ids]
        self.offset[ids[~is_env_assigned_to_terrain(self._env, 'flat')[ids]]] = 0
        self.clock[ids] = 0

    def _update_command(self):
        from .unified_contract import full_arm_envelope
        base = self._env.command_manager.get_term('base_velocity')
        standing = (base.requested == 0).all(-1)
        settled = self.robot.data.root_lin_vel_b[:, :2].norm(dim=-1) < .1
        self.ready_time = torch.where(standing & settled, self.ready_time + self._env.step_dt, 0.)
        active = standing & (self.ready_time >= .5)
        self.clock += self._env.step_dt * active
        self.clock[~standing] = 0
        default = self.robot.data.default_joint_pos[:, self.ids]
        desired = default + self.offset * full_arm_envelope(self.clock)[:, None] * active[:, None]
        limits = self.robot.data.soft_joint_pos_limits[:, self.ids]
        desired = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
        step = self.cfg.speed * self._env.step_dt
        self.target += (desired - self.target).clamp(-step, step)


@configclass
class FullRangeArmCommandCfg(PayloadArmCommandCfg):
    class_type: type = FullRangeArmCommand
    speed: float = 0.75
    # Pause/resume of the envelope must not resample a new endpoint mid-excursion.
    resampling_time_range: tuple[float, float] = (65., 65.)
