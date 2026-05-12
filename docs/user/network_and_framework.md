# Network And Framework Notes

## Framework Stack

`Go2-X5-lab` uses Isaac Lab ManagerBased environments.

```text
Gym task ID
  -> isaaclab.envs:ManagerBasedRLEnv
  -> env_cfg_entry_point
  -> scene / actions / observations / rewards / events / terminations
  -> RSL-RL or CusRL runner config
```

The local extension package is installed from `source/robot_lab`.

## RSL-RL Runner Configs

Go2-X5 locomotion runner configs live in:

```text
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/agents/rsl_rl_ppo_cfg.py
```

Typical current network shape for DogOnly PPO:

| Field | Value |
| --- | --- |
| Actor input | `260` |
| Critic input | `260` |
| Actor output | `12` |
| Hidden dims | `[512, 256, 128]` |
| Activation | `ELU` |

See `docs/development/02_current_rl_policy.md` for the full contract.

## DogOnly PPO Contract

The active low-level Go2-X5 design separates the base policy from arm and gripper command channels:

- PPO outputs only leg actions.
- Arm joint targets are command-driven.
- Gripper is an explicit high-level command slot.
- The policy can observe arm state and arm commands to stabilize the base under arm motion.

This contract is important for VLA / UMI integration because the high-level task policy can issue:

```text
cmd_vel(3) + arm_joint_pos_or_pose(6) + gripper(1)
```

without owning direct low-level leg joints.

## Sim2sim And Randomization

The locomotion stack includes support for:

- Action delay and action hold.
- Action noise.
- Observation delay.
- IMU / projected gravity noise.
- Friction, mass, COM, actuator gain, external force, and push randomization.

Training-time randomization is configured in environment configs. Replay code can disable or freeze randomization for deterministic policy inspection.

## Task Families

- Go2 and baseline Go2-X5 locomotion: standard velocity tracking.
- Staged Go2-X5 training: foundation, rough transfer, DogOnly, arm warmup.
- UMI locomotion6D: separate 18-DoF prototype path.
- Manipulation prototypes: high-level 10-D action contract and visual / privileged observation split.
