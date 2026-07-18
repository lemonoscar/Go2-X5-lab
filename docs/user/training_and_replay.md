# Training And Replay

## List Tasks

```bash
python scripts/maintenance/list_envs.py
```

## Foundation Flat Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --headless
```

Output goes to:

```text
logs/rsl_rl/go2_x5_foundation_flat/<run>/
```

## Rough Transfer Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-Robust-v0 \
  --headless \
  --resume \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/<run>/model_<iter>.pt
```

## Arm Warmup Training

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-ArmWarmup-v0 \
  --headless \
  --resume \
  --checkpoint=logs/rsl_rl/go2_x5_robust_rough/<run>/model_<iter>.pt
```

## DogOnly Rough Curriculum Training

This task keeps the DogOnly PPO action head at `12` leg actions, keeps the arm fixed through the command-driven arm action term, and trains on generated rough terrain with the live `height_scan` observation and terrain-level curriculum.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0 \
  --headless \
  --resume \
  --checkpoint=logs/rsl_rl/go2_x5_dog_only_flat/2026-04-18_18-46-33_dog_only_recover_from15000/model_18250.pt
```

Output goes to:

```text
logs/rsl_rl/go2_x5_dog_only_rough/<run>/
```

## PCT Stair Adaptation Training

This is an isolated continuation stage for narrow up/down stairs. It keeps the DogOnly `260`-D observation and `12`-D action contract and restores the default pose used by the source rough checkpoint.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyStairs-v0 \
  --headless \
  --resume \
  --checkpoint=logs/rsl_rl/go2_x5_dog_only_rough/2026-05-12_08-16-24/model_26000.pt \
  --no_load_optimizer
```

The task defaults to 1024 environments, 6000 continuation updates, a `5e-5` learning rate, and a separate output family:

```text
logs/rsl_rl/go2_x5_dog_only_stairs/<run>/
```

Use `--num_envs`, `--max_iterations`, and `--seed` for resource-limited smoke or repeatable evaluation runs. Do not replace the production PCT checkpoint until the generated-stair, flat/rough regression, and scanned-PCT acceptance gates in `docs/train/dogonly_ppo_stair_adaptation.md` all pass.

## PCT Straight-Stair V2 Training

Use the V2 task to learn only the PCT first straight flight after the first stair-adaptation checkpoint has shown useful progress but before it becomes unstable. The first gate is exactly 250 updates from `model_26250.pt` with a fresh optimizer:

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairs-v0 \
  --headless \
  --device=cuda:0 \
  --num_envs=512 \
  --max_iterations=250 \
  --seed=0 \
  --resume \
  --checkpoint=logs/rsl_rl/go2_x5_dog_only_stairs/2026-07-14_02-48-37_pct_stairs_seed0/model_26250.pt \
  --no_load_optimizer \
  --run_name=pct_straight_v2_seed0_gate250
```

Output goes to:

```text
logs/rsl_rl/go2_x5_dog_only_pct_stairs/<run>/
```

Do not continue directly past the first 250-update checkpoint. First inspect `Curriculum/terrain_levels/completion_rate`, centerline progress/height metrics, body-contact and orientation terminations, then run the scanned-PCT five-entry-condition gate for the first straight flight. Platform turning and the second flight are outside this stage. The detailed thresholds are recorded in `docs/train/dogonly_ppo_pct_stairs_v2.md`.

If the procedural-only gate reaches the top but does not transfer to the scanned PCT first flight, use the scan-mixed high-difficulty gate. Replace `<GATE250_CHECKPOINT>` with the actual first-gate file (the runner may save the inclusive last iteration as `model_26499.pt`):

```bash
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairs-v0 \
  --headless \
  --device=cuda:0 \
  --num_envs=512 \
  --max_iterations=250 \
  --seed=0 \
  --resume \
  --checkpoint=<GATE250_CHECKPOINT> \
  --no_load_optimizer \
  --run_name=pct_straight_scanmix_high_seed0_gate250 \
  env.scene.terrain.max_init_terrain_level=9
```

The max-level override is intentional: a new Isaac process does not restore per-environment terrain levels from a checkpoint, so omitting it would restart the curriculum at level 0 and waste the high-difficulty adaptation gate.

## PCT Full-Height Hard-Only Long Training

This isolated task removes difficulty progression and trains every environment on the full `1.57 m` first-flight rise. Eighteen of twenty terrain columns use the unscaled target scan; the other two use same-height procedural stairs. It continues from the current real-PCT progress candidate with a fresh optimizer:

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsHard-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs/2026-07-14_07-05-50_pct_straight_scan80_easycurriculum_footlift3_seed0_final250/model_26850.pt \
  --no_load_optimizer \
  --run_name pct_hard_only_scan90_seed0_long1000
```

The runner uses an initial learning rate of `1e-5`, saves every 50 updates, and writes only TensorBoard/checkpoint data under:

```text
logs/rsl_rl/go2_x5_dog_only_pct_stairs_hard/<run>/
```

Do not add a terrain-level override: this task intentionally has one full-height row. Monitor scan progress/height/completion together with illegal contact, hip/thigh contact, and value loss; reward alone is not a pass condition.

The command above has now been run as the reproducible hard-only experiment
`2026-07-14_08-58-11_pct_hard_only_scan90_seed0_long1000`. It was deliberately stopped after about 108 updates:
the target-scan safe completion rate stayed at `0%`, bad-orientation termination stabilized near `84%`, and scan progress/height
plateaued near `14.7% / 12.4%`. In real PCT, `model_26950.pt` reached only `0.2414 m`, versus `0.6261 m` for the source
`model_26850.pt`, and never raised root z above the initial `0.3693 m`.

Keep this command for reproduction, but do not resume that run to 1000 updates and do not deploy its `model_26900.pt` or
`model_26950.pt`. The current real-PCT candidate remains the source `model_26850.pt`. The next training revision must pass a
short first-steps gate on the same full-height geometry before full-flight continuation.

## PCT Full-Height First-Steps Bootstrap

Stage A keeps every terrain cell on the unscaled full-height target scan, but trains only the measured first `1.00 m` scan segment
(`1.05 m` including the approach offset, `0.314 m` target rise). It is an isolated bootstrap and must start from the original
`model_26850.pt`, not a hard-only regressed checkpoint:

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstSteps-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 250 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs/2026-07-14_07-05-50_pct_straight_scan80_easycurriculum_footlift3_seed0_final250/model_26850.pt \
  --no_load_optimizer \
  --run_name pct_first_steps_scan100_seed0_gate250
```

Do not continue automatically into Stage B. First require target-scan completion `>=50%`, path/height ratios `>=85%`,
bad orientation `<10%`, illegal contact `<5%`, and a real-PCT root-z increase. The Stage-A checkpoint is not a deployment model.

The initial `1.05 m / 0.314 m` Stage-A target produced no safe completion and was stopped. The approved first-rise fallback keeps
the same full-height scan but uses the measured `0.65 m / 0.141 m` goal:

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFirstRise-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 100 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs/2026-07-14_07-05-50_pct_straight_scan80_easycurriculum_footlift3_seed0_final250/model_26850.pt \
  --no_load_optimizer \
  --run_name pct_first_rise_scan100_seed0_gate100
```

This fallback also starts from the original `model_26850.pt`; it must not resume the failed FirstSteps run.

## PCT Full-Flight Profiled Deployment-Speed Training

Use this isolated task for the complete measured first flight at the real no-Float PCT command speed (`0.25 m/s`). It keeps the
exact full-height collision, measured non-linear height profile, 40-second episode, 48-step rollout, `260 -> 12` policy contract,
and the safe-progress/termination gates. It has no terrain difficulty levels and saves every 100 updates.

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledDeploymentSpeed-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_coverage/2026-07-14_12-24-16_pct_full_profiled_coverage25bottom_pitch18_clear08_grace200_term500_tilt4_48_from26999_seed0_long1000/model_27100.pt \
  --no_load_optimizer \
  --run_name pct_full_profiled_deployment_speed025_term500_tilt4_48_fromcoverage27100_seed0_long1000
```

Treat each 100-update checkpoint as one decision point. Do not change the task after only 20--30 healthy updates. At each decision
point, compare path progress, height gain, completion, orientation failure, contact failure, and value loss, then run the external
PCT stair smoke at an explicit `--pct-carry-max-linear-velocity 0.25` with a 20-second stall timeout. Continue another 100 updates
only when that real-PCT result improves; training reward alone is not an acceptance signal.

## PCT Rear-Support 1000-Update Continuation

Use this task after the deployment-speed run has reached the repeatable `0.79 m` transition plateau. It changes only the reward by
adding a progress-gated rear-foot catch-up/support term with weight `3.0`; geometry, measured height profile, bottom reset, command
speed, episode length, terminations, PPO shape, and all inherited reward weights remain unchanged.

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledRearSupport-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_deployment_speed/2026-07-14_13-30-44_pct_full_profiled_deployment_speed025_term500_tilt4_48_fromcoverage27100_seed0_long1000/model_28099.pt \
  --no_load_optimizer \
  --run_name pct_full_profiled_rear_support_w3_lag38_height10_from28099_seed0_long1000
```

The runner uses `48` rollout steps, `1000` continuation updates, a fresh optimizer at `2.5e-5`, entropy coefficient `0.003`,
and saves every `100` updates. Ordinary early fluctuations are not a stop condition. Check every 100-update point for NaN, value
loss, path/height, rear-support reward, orientation and contacts; perform real PCT no-Float comparisons at least at `+500` and
`+1000`. A checkpoint advances only if its physical progress and root height both exceed the source model's
`0.792373 m / 0.440427 m` under the same `0.25 m/s`, root z=`0.172 m`, 20-second-stall protocol.

## PCT Stable-Completion 500-Update Consolidation

Use this follow-up only after the rear-support run has learned substantial first-flight progress but begins trading safety for
base/head contact. It keeps the same PCT geometry and rear-foot reward, but strengthens dense posture/contact/clearance costs,
keeps the hard failure cost unchanged, doubles the safety-gated completion bonus, and lowers PPO learning/exploration rates.

The selected source is `model_29000.pt`, not the later `model_29098.pt`: under the same real no-Float PCT protocol the former
reached `3.459380 m / root z 1.715964 m` with zero critical contact, while the latter reached farther only by producing
`700.65 N @ Head_lower` and `47.255 deg` tilt.

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledStableCompletion-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 500 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_rear_support/2026-07-14_19-38-06_pct_full_profiled_rear_support_w3_lag38_height10_from28099_seed0_long1000/model_29000.pt \
  --no_load_optimizer \
  --run_name pct_stable_completion_orient8_contact4_clear6_bonus500_from29000_seed0_long500
```

The run saves every 100 updates under
`logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_stable_completion/`. Do not run a GUI replay concurrently on the
same 8 GB GPU. At `+100/+200/+300/+400/+500`, inspect path/height together with bad orientation, illegal contact,
base/head/arm contact, completion, and value loss; a reward increase alone is not acceptance.

## PCT Top-Landing 1000-Update Continuation

V6.8 was stopped at `+100`: its `model_29100.pt` reached `3.882698 m / root z 2.051168 m` in real PCT, but produced
`37.580 deg` tilt and `1415.36 N @ FR_hip`, so it is not a valid continuation source. Use the last safe candidate
`model_29000.pt` for this isolated last-riser coverage run.

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledTopLanding-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_rear_support/2026-07-14_19-38-06_pct_full_profiled_rear_support_w3_lag38_height10_from28099_seed0_long1000/model_29000.pt \
  --no_load_optimizer \
  --run_name pct_top_landing_gate30_bottom40_top60_bonus500_from29000_seed0_long1000
```

The task keeps the exact real PCT first flight and `0.25 m/s` command. Reset sampling is 40% full bottom starts and 60% measured
last-riser starts; only positive progress reward uses a 30-degree world-level tilt gate. Completion remains 20 degrees, hard failure
remains 32 degrees, and critical contact remains 35 N. Inspect every 100 updates and accept a checkpoint only after real-PCT trials;
the final gate is at least four successes from nominal and X/Y `+/-5 cm` entry conditions.

V6.9 was rejected at `+100`: its real nominal trial reached only `2.990142 m / root z 1.596187 m`, below the safe source's
`3.459380 m / 1.715964 m`. The `39.22 N @ Head_lower` peak is now treated as a minor warning rather than the primary failure.

## PCT Forward-Priority Platform Continuation

Use V6.10 when the required outcome is reaching and dwelling on the first platform. It fixes the training completion point from
`4.324 m` to the real `3.902 m` gate, uses bottom starts only, and accepts 35--50 N as a reported minor contact. It still rejects
contact at or above 50 N and posture beyond 32 degrees.

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledPlatformProgress-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_rear_support/2026-07-14_19-38-06_pct_full_profiled_rear_support_w3_lag38_height10_from28099_seed0_long1000/model_29000.pt \
  --no_load_optimizer \
  --run_name pct_platform_progress_target3902_height985_gate30_contact50_forward12_from29000_seed0_long1000
```

The formal run is stored under
`logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_platform_progress/2026-07-14_23-33-16_pct_platform_progress_target3902_height985_gate30_contact50_forward12_from29000_seed0_long1000/`.
At each 100-update checkpoint, test nominal real PCT first. Run all five entry offsets only after nominal platform success.

V6.10 `model_29100/29200` both crossed the physical `3.902 m` entrance line, but neither ever combined that progress with
root z `>=1.85 m`; their best line-crossing heights were about `1.77 m`, with zero top-dwell frames. V6.11 therefore keeps every
V6.10 setting and moves only the training completion point `0.25 m` into the platform, to `4.15 m`.

## PCT Platform-Entry 1000-Update Continuation

```bash
cd /home/lemon/research/Issac/Go2-X5-lab

env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairsFullFlightProfiledPlatformEntry-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 512 \
  --max_iterations 1000 \
  --seed 0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/go2_x5_dog_only_pct_stairs_full_flight_profiled_platform_progress/2026-07-14_23-56-26_pct_platform_progress_target3902_height985_gate30_contact50_forward12_from29100_seed0_remaining900/model_29200.pt \
  --no_load_optimizer \
  --run_name pct_platform_entry_target4150_height985_gate30_contact50_forward12_from29200_seed0_long1000
```

This task changes no reward, terrain, observation, action, posture, or contact setting from V6.10. It uses a fresh optimizer and
saves every 100 updates. Stop at each checkpoint for the nominal real-PCT trial; run the other four entry offsets only after the
nominal trial reaches platform height and dwells there for 25 frames.

## Regular-Box Staged Up/Down Training

Keep the final bottom-start task fixed. First train the isolated ascent curriculum from `model_29200.pt`:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularAscentCurriculum-v0 \
  --resume \
  --checkpoint /home/lemon/research/Issac/Go2-X5-lab/logs/rsl_rl/Rough/model_29200.pt \
  --no_load_optimizer \
  --num_envs 512 \
  --max_iterations 1000 \
  --run_name regular_ascent_curriculum_from29200_seed0_long1000 \
  --seed 0 \
  --device cuda:0 \
  --headless
```

Every 100-update checkpoint must be evaluated on the separate fixed `0.157 m` task; curriculum-level completion is not an exact
height acceptance result:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/evaluate_pct_regular_stairs.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularStairs-v0 \
  --checkpoint <ASCENT_CHECKPOINT> \
  --num_envs 32 \
  --episodes 32 \
  --mode robust \
  --seed 0 \
  --device cuda:0 \
  --headless \
  --output <ASCENT_EVALUATION_JSON>
```

After an exact-ascent checkpoint reaches at least `26/32`, train the top-platform reset task. Its success means only that the
descending flight and bottom gate were completed from a top start:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularDescentStart-v0 \
  --resume \
  --checkpoint <BEST_ASCENT_CHECKPOINT> \
  --no_load_optimizer \
  --num_envs 512 \
  --max_iterations 1000 \
  --run_name regular_descent_start_seed0_long1000 \
  --seed 0 \
  --device cuda:0 \
  --headless
```

Only after the local descent gate reaches at least `26/32` should its best actor continue on
`RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctRegularUpDownStairs-v0`. Final acceptance remains a bottom start, multiple seeds,
at least 32 robust episodes per seed, and at least 80% complete ascent-platform-descent success. The evaluator records each
episode's start progress and height so top-start results cannot be confused with full-route results.

## Unified Rough/Stairs/Vx 10k Training

The unified task keeps DogOnly `260 -> 12`, restores the checkpoint pose used by `model_26250.pt`, trains explicit up/down stair
columns, and treats any vx error up to `0.1 m/s` as acceptable. After the requested maximum was reduced to `0.7 m/s`, the
repository includes the accepted 50-update continuation checkpoint. Start the remaining 9950 updates with:

```bash
env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyRoughStairsVx-v0 \
  --num_envs 1024 \
  --max_iterations 9950 \
  --seed 0 \
  --resume \
  --checkpoint logs/rsl_rl/go2_x5_dog_only_rough_stairs_vx/2026-07-18_10-49-13_curriculum10k_v2_from26250_seed0/model_26300.pt \
  --run_name curriculum10k_v4_vx070_rehearsal35_from26300_seed0 \
  --logger tensorboard \
  --device cuda:0 \
  --headless
```

Bundled starting checkpoint:

```text
logs/rsl_rl/go2_x5_dog_only_rough_stairs_vx/2026-07-18_10-49-13_curriculum10k_v2_from26250_seed0/model_26300.pt
SHA-256: 43cdeb52a1ef3f2d562cd481b52343a3e424600bfd9319ccec199eb2693e93ac
```

The checkpoint contains actor, critic, optimizer, and global iteration `26300`. Because the command deliberately omits
`--no_load_optimizer`, it retains that optimizer state and ends at global iteration `36250`; the next new gate is
`model_26400.pt`. No training process is left running by this repository handoff.

The final seed-0 handoff evaluation passed all eight steady-state segments from `0.0` through `0.7 m/s`. At `0.7 m/s`, measured
mean/RMSE were `0.618/0.083 m/s`; the diagnostic stop immediately after the maximum-speed segment also passed with
`0.070 m/s` RMSE.

The earlier `10-26-35_curriculum10k_from26250_seed0` launch was stopped at about 27 updates because its first speed-bin promotion
occurred before the intended 250-update warm-up. V2 produced `model_26300.pt`; it passed all retained `0.0--0.7 m/s` points.
The short V3 rehearsal launch was stopped without a checkpoint when the maximum-speed requirement changed. A V4 startup audit
then reached iteration `26317` before being stopped at the user's request; it produced no newer checkpoint and is not a candidate.
The bundled `model_26300.pt` remains the clean continuation point.

Evaluate a candidate on deterministic flat terrain over `vx=0.0,0.1,...,0.7 m/s`, including zero speed and the
post-maximum-speed stopping diagnostic:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
/home/lemon/miniconda3/envs/env_isaaclab/bin/python -B \
  scripts/reinforcement_learning/rsl_rl/evaluate_vx_tracking.py \
  --checkpoint <CHECKPOINT> \
  --output-dir <OUTPUT_DIR> \
  --seed 0 \
  --device cuda:0 \
  --headless
```

The evaluator requires both mean vx error and vx RMSE to remain within `max(0.1 m/s, 10% * vx_cmd)`, with
`|mean vy| <= 0.1 m/s`, `|mean wz| <= 0.1 rad/s`, and no reset.

## Standard Replay

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/<run>/model_<iter>.pt \
  --num_envs=1
```

## PCT Straight-Stair Task Replay

Replay a PCT-stairs checkpoint with the same centerline command generator and logged training-time environment overrides:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyPctStairs-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_dog_only_pct_stairs/<run>/model_<iter>.pt \
  --num_envs=1 \
  --seed=0 \
  --device=cuda:0 \
  --real-time \
  --debug_interval=50
```

The replay keeps running and resets completed or failed episodes until the window is closed or `Ctrl+C` is pressed. Do not add `--base_cmd` or `--keyboard`: this task owns its path-following velocity command and those manual modes are only available to tasks with generic velocity ranges.

## Fixed Base Command Replay

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/<run>/model_<iter>.pt \
  --num_envs=1 \
  --base_cmd 0.0 0.0 0.0
```

## Keyboard Replay

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/<run>/model_<iter>.pt \
  --num_envs=1 \
  --keyboard
```

Keyboard details are in `docs/user/keyboard_control.md`.

## Custom Map Replay

```bash
python scripts/reinforcement_learning/rsl_rl/play_cs.py \
  --task=<TASK_ID> \
  --checkpoint=<PATH_TO_MODEL> \
  --map=<PATH_TO_USD_MAP>
```

## Checkpoint Migration

Old Go2-X5 route checkpoints may need shape migration before resuming into newer arm-aware or DogOnly configs.

```bash
python scripts/checkpoints/migrate_go2_x5_route_checkpoint.py \
  --input logs/rsl_rl/go2_x5_foundation_flat/<run>/model_<iter>.pt
```

DogOnly checkpoint migration:

```bash
python scripts/checkpoints/migrate_go2_x5_dog_only_checkpoint.py \
  --input <OLD_CHECKPOINT>
```

When optimizer state no longer matches the new network shape, resume with:

```bash
--no_load_optimizer
```

## Training Records

Reward weights, PPO settings, and phase notes are stored locally in `docs/train`.
