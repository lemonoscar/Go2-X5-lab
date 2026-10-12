# 26000 unified continuation

Status: stage 1 implemented, runtime verification pending. This is not full carrying or TCP acceptance.

The new task is `RobotLab-Isaac-Velocity-Flat-Go2-X5-UnifiedCarry-v0`.
It retains the source 260 input prefix and adds 7 mode/phase/anchor values plus
four previous 54-D proprioceptive frames (oldest first): 483 inputs, 12 leg actions.
History frames contain angular velocity, projected gravity, 18 relative joint
positions, 18 joint velocities, and 12 previous leg actions. Reset fills history
with the first measured frame; repeated reads do not advance history.

The actor has the same hidden layers. Migration copies the original 260 columns,
zeroes new columns, and tests deterministic output equality; Adam state is discarded.
The new update counter starts at zero with source iteration 26000 retained in metadata.
Policy period remains 0.02 seconds; source default joint pose, 40/1 leg PD and 0.25
leg action scale are preserved. The original task and 260x12 deployment profile remain available.

Commands have independent inclusive dead zones vx +/-0.1, vy/wz +/-0.05.
Full limits are vx -0.5/+0.75, vy +/-0.2, wz +/-0.3. The normalized quadratic
budget is <=1, then dead zones are reapplied. Pure axes and combined commands
are sampled explicitly. Zero commands select standing. Movement waits for measured
arm and target return to the transport pose. Stage 1 arm excitation is bounded
joint-space motion, amplitude 0.15 rad and target slew <=0.35 rad/s.
This is not world-space IK or a complete manipulation controller.

Standing anchors are held throughout a standing interval, including arm changes.
The phase and reward are disabled while standing. Existing trot shaping and rigid
stand-still joint penalties are removed only in this task. The four-beat reward is
soft and lower-weight than velocity tracking. Standing may adjust feet and posture.

## Commands

Use scripts/checkpoints/migrate_go2_x5_unified_checkpoint.py SOURCE DESTINATION,
then train.py with the new task, --resume --checkpoint DESTINATION --no_load_optimizer.
The output checkpoint and its JSON manifest must stay together. Original profiles
must reject the new shape instead of silently dropping history.

Remote runtime: source /data1/duanzhibo/isaacsim/setup_conda_env.sh before invoking
/data4/duanzhibo/miniconda3/envs/isaaclab/bin/python. All write/cache/log paths must
be redirected under /data4/duanzhibo/xhq_workload. Source package PYTHONPATH must
point to the isolated checkout, not the old editable install.

## Remaining required gates

- Original-model flat diagnostic, rough-terrain baseline and fixed per-level speed criteria.
- Stage 1 learning smoke then measured regressions against the original checkpoint.
- Accurate terminal-link equivalent load (mass, COM and inertia), then real gripping evidence
  over 5–450 g; loading only the base is not acceptable.
- Dataset-backed full arm trajectories, actual world-TCP compensation and grasp/transport posture checks.
- 60-second standing tests, 10-second extended dwells, all agreed position/velocity/attitude/TCP thresholds.
- Isolated MuJoCo 483-input profile with exactly matching history, phase, frames and control timing.

Do not claim carrying, world-TCP or sim2sim acceptance from the stage 1 command diagnostic.
