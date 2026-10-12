"""Initial flat command/standing gate. Does not claim TCP or carrying acceptance."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--task', default='RobotLab-Isaac-Velocity-Flat-Go2-X5-UnifiedBaseline-v0')
p.add_argument('--seconds', type=float, default=65)
p.add_argument('--num_envs', type=int, default=16)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import robot_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat


def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = args.seconds + 10
    cfg.commands.base_velocity.resampling_time_range = (10000., 10000.)
    env = gym.make(args.task, cfg=cfg)
    state = torch.load(args.checkpoint, map_location=env.unwrapped.device, weights_only=True)['model_state_dict']
    def actor(obs):
        for i in (0,2,4,6):
            obs = torch.nn.functional.linear(obs, state[f'actor.{i}.weight'], state[f'actor.{i}.bias'])
            if i != 6:
                obs = torch.nn.functional.elu(obs)
        return obs
    commands = torch.tensor([[0,0,0],[.75,0,0],[-.5,0,0],[0,.2,0],[0,-.2,0],[0,0,.3],[0,0,-.3],
                             [.5,.1,.15]], device=env.unwrapped.device)
    target = commands[torch.arange(args.num_envs,device=commands.device) % len(commands)]
    obs,_ = env.reset()
    term = env.unwrapped.command_manager.get_term('base_velocity')
    term.requested.copy_(target)
    term.time_left[:] = 10000
    term._update_command()
    obs = env.unwrapped.observation_manager.compute()
    robot = env.unwrapped.scene['robot']
    anchor = robot.data.root_pos_w[:,:2].clone()
    failures = torch.zeros(args.num_envs, dtype=torch.bool, device=commands.device)
    velocity_mae = torch.zeros(args.num_envs,3,device=commands.device)
    max_drift = torch.zeros(args.num_envs,device=commands.device)
    steps = int(args.seconds/env.unwrapped.step_dt)
    with (out/'trace.jsonl').open('w') as f, torch.inference_mode():
        for step in range(steps):
            action=actor(obs['policy'])
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite policy action')
            obs,_,terminated,truncated,_ = env.step(action)
            failures |= terminated | truncated
            velocity = torch.cat([robot.data.root_lin_vel_b[:,:2],robot.data.root_ang_vel_b[:,2:3]],-1)
            velocity_mae += (velocity-target).abs()/steps
            drift = (robot.data.root_pos_w[:,:2]-anchor).norm(dim=-1)
            max_drift = torch.maximum(max_drift,drift)
            if step%10==0:
                f.write(json.dumps({'t':step*env.unwrapped.step_dt,'command':target.tolist(),
                    'velocity':velocity.tolist(),'position':robot.data.root_pos_w.tolist(),
                    'quaternion_wxyz':robot.data.root_quat_w.tolist(),'drift':drift.tolist(),
                    'terminated':terminated.tolist()})+'\n')
            term.requested.copy_(target)
            term.time_left[:] = 10000
        result={'stage':'flat_command_diagnostic_only','seconds':args.seconds,'commands':target.tolist(),
                'terminated_or_reset':failures.tolist(),'velocity_mae':velocity_mae.tolist(),
                'max_xy_displacement':max_drift.tolist(),
                'not_validated':['world_TCP','physical_grasp','payload','rough_terrain','MuJoCo']}
        (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    env.close()

try:
    main()
finally:
    app.close()
