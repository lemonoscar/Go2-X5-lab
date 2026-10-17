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

## 舒缓摆腿配置（待续训验证）

任务：`RobotLab-Isaac-Velocity-Flat-Go2-X5-UnifiedSmoothWalk-v0`。
从第一阶段 `model_1499.pt` 续训；483 维观测、12 维腿部动作不变，无需再次扩展网络。
原 `UnifiedCarry` 任务保留原参数，便于对照。

- 四拍参考摆动占比：0.20 → 0.24；相位间隔仍为 0.25，参考摆动窗口不重叠。
- 频率公式：`0.55 + 1.05 * clamp(norm(vxy)/0.75 + abs(wz)/0.6, 0, 1)` Hz；站立相位为零。
  高速前进为 1.6 Hz（参考摆动 0.15 秒）；后退 -0.5 为 1.25 Hz；侧移 0.2 为 0.83 Hz；转向 0.3 为 1.075 Hz。
- 新增仅在实际行走指令非零时生效的动作变化平方和惩罚，权重 -0.005；站立及等待收臂时为零。
  这是轻量平滑项，不是落足次数惩罚，也不能保证减少实际碎步。
- 原速度范围、逐轴死区、站立奖励、机械臂控制、速度跟踪奖励均继承第一阶段。
- 独立实验目录名：`go2_x5_unified_smooth_walk`；继承学习率 5e-5，额外执行 2000 次更新。

续训入口（在已配置的 IsaacLab 环境和仓库根目录中执行）：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Go2-X5-UnifiedSmoothWalk-v0 \
  --resume --checkpoint /absolute/path/to/model_1499.pt --no_load_optimizer \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 2000 \
  --seed 42 --run_name smooth-swing
```

此变更只完成配置，未证明动作已经改善。续训对照应记录实际足端接触周期、重复落足、滑移、
速度误差和底座姿态，并回归站立动臂位移。不得通过修改播放速度展示“变慢”；
评估循环每步只能由环境推进一次命令相位，不应额外调用 `_update_command()`。

## 第二阶段：恢复原步频，负载与地形适应

任务 `RobotLab-Isaac-Velocity-Rough-Go2-X5-UnifiedPayloadTerrain-v0` 从第一阶段
`model_1499.pt` 开始；不用低频实验 `model_3498.pt`。继承原 0.7–2 Hz、20% 摆动占比、
速度范围与 483→12 网络，不包含 smooth-walk 新增的动作变化惩罚。

- 每次 reset 抽样 20% 空载，其余为 5–450 g 均匀分布。
- 负载建模为刚性夹持、半径 15 mm 的等效球，中心在 arm_link6 局部 +X 0.15757 m；
  同时更新该连杆质量、质心和惯量（平行轴定理），保存并恢复未加载参数，避免多次 reset 累加。
  不模拟独立物体的碰撞、抓取滑脱或夹爪接触，不向 actor/critic 提供精确负载标签。
- 50% 地形列为平地，其中 70% 指令采样为站立动臂；其余地形只收臂行走。
  动臂仍是第一阶段的小幅关节扰动，完整世界 TCP 作业轨迹尚未接入。
- 另外 50% 包括起伏（1–6 cm）、障碍块（2–10 cm）、坡面（坡度 0–0.25）、
  上下台阶（3–10 cm）。四档难度从最低档起，根据走出本块区域或跌倒升降档。
- 16 m 地形块；离任一水平轴中心 6 m 时以 timeout 结束，避免跨地形列后仍做平地作业。
  站立回合仍为 70 秒。地形难度上限是本阶段训练配置，不是已验收能力。
- 首轮额外 2000 次 PPO 更新、1024 环境、学习率 5e-5、重新初始化优化器。

通过动力学读回和启动检查不代表通过负载/粗糙地形验收。训练后需分空载及负载档、
分地形难度评估速度跟踪、跌倒、站立动臂位移，并回归前后/左右/转向各轴；
完整作业范围与 MuJoCo 验收仍是后续独立步骤。

## 2026-10-16: measured stride repair (experimental)

Task `RobotLab-Isaac-Velocity-Flat-Go2-X5-UnifiedStrideRepair-v0` resumes accepted
stage1 `model_1499.pt`, fresh optimizer, 2000 additional updates, 1024 envs, seed42.
Flat only; half empty, otherwise 5--150 g equivalent rigid TCP payload. Full command
ranges/dead zones and 35% standing sampling (plus dead-zone zeros) remain intact.
The existing small joint-space arm excitation remains; this is not full TCP-work acceptance.

Reference frequency remains 0.7--2 Hz. Swing fraction .24; balanced swing/support
reward .8 replaces the original .3 average. Measured Schmitt contact thresholds
10/3 N drive bounded short-swing, short-stance and signed body-relative foot
excursion event costs, weight -.025 per event (divide by dt before RewardManager).
At .5 m/s: minimum air .14 s, stance .22 s, excursion .15 m; at .75 m/s:
minimum air .12 s, excursion capped .20 m. Pure yaw has no translation excursion
cost. Standing disables all event costs. Tilt above approximately 11.5 degrees
reduces event cost to 25% to allow recovery. Existing air-time reward is removed.
Yaw tracking weight3.5, orientation-.3, slip-.12; action-rate penalty unchanged.
No new actor observations, no payload label, no hard foot locking.

Contact regression: `python test/test_unified_stride.py`; also run existing
contract/payload checks and a 96-env simulator probe before training. Probe must
print STRIDE_PROBE_COMPLETE and show finite rewards/observations. A normal launch
is not evidence of improved gait. Compare actual contacts/stride, vx/wz error,
slip and tilt, startup failures and 60s standing arm drift against stage1 before
advancing to 450g and rough terrain. Count resets from time zero, not only after settling.

## 2026-10-17: separate terrain-focus and full-range arm run

`RobotLab-Isaac-Velocity-Rough-Go2-X5-UnifiedTerrainFocus-v0`: resume payload-terrain
`2026-10-14_23-26-18_payload-terrain-original-gait/model_3498.pt` with fresh optimizer,
4096 envs, 2000 additional updates. Keep the concurrent stride-repair run untouched.
20% flat stationary arm work, 20% rough, 20% blocks, 20% ascending stairs and 20%
descending stairs. Seven risers per outward stair flight, .4m tread, height
curriculum .03--.10m. 80% of rough commands are vx=.5, vy=wz=0; remaining draws
retain the original full-range mixture/dead zones. Original frequency/rewards,
0/5--450g rigid equivalent payload and terrain curriculum are retained.

Arm targets cover the entire simulator soft joint range (90% of hard range).
Each joint draw: 25% lower endpoint, 25% upper endpoint, 50% uniform interior.
32s cycle: 12s outward, 4s endpoint hold, 12s return, 4s stowed hold. Target slew
cap .75rad/s; envelope pauses unless stationary and settled; walking requests
return to stow under the existing gate. No hard-limit forcing or instantaneous
pose reset. This is maximal joint-space stress coverage, NOT collision-certified
workspace trajectories or completed physical manipulation acceptance. Existing
small-amplitude tasks and the currently running stride experiment remain unchanged.

### 2026-10-17 task-arm repair (not yet dynamically accepted)

`UnifiedTaskArm` resumes terrain `model_5497.pt` with the same 483 observations / 12 leg actions. The previous full-limit sampler could withdraw its target when base speed rose; the new finite task player completes its path independently of base speed, including the checked return before walking. No load label is observed.

The initial repair allocates 60% flat manipulation and 10% each rough, blocks, upstairs and downstairs replay. Flat starts empty; terrain retains 0--450 g. Expansion is gated by independent evaluation, not elapsed training steps. There are 24 initial and 24 main-region candidate paths, six action families weighted 25/25/15/15/10/10. Arcs are free-space exercises, not door-opening demonstrations. Main-region paths are not enabled in the initial stage. Library bounds are candidate bounds, not a workspace reachability claim.

Geometry was extracted from the composed training USD joint frames and collision mesh bounds. `scripts/tools/build_task_arm_library.py` uses conservative link OBB rejection along the joint path, plus torso, floor and the head-camera envelope. The camera base-frame box is x=[.29779,.35285], y=±.045005, z=[.086,.131] m: housing plus 30 mm forward / 20 mm upward. Planning adds 15 mm clearance; the actual-state OBB guard uses 3 mm. Adjacent/housing OBB exclusions are explicit; PhysX self-collision is enabled to retain actual geometry interactions. These sampled guards do not certify continuous hardware safety. Finger targets remain at the existing configuration; no real object/grasp contact is simulated.

Provisional arm DC-motor constraints are 15 Nm (J1--3), 3 Nm (J4--6), 3 rad/s. Gains are 250/8 and 80/3 respectively, with inherited gain randomization; all leg gains are preserved. These are simulation choices based on the supplied URDF, not verified hardware ratings. Quintic path durations bound target joint speed/acceleration at .4 rad/s and .8 rad/s², and TCP speed at .06 m/s. Runtime soft limits are checked before reset. A new-task-only one-second zero-leg-action startup is implemented in the same delay wrapper for training and evaluation.

Acceptance must pair base drift (5 cm goal, 10 cm hard bound), speed and attitude with actual arm/TCP execution. Neither reward growth nor reduced arm movement establishes success. Preserve first failures and report achieved motion, payload and all cases. The original tasks and checkpoint histories remain unchanged.
