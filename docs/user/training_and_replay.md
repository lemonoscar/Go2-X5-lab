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

## Standard Replay

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/<run>/model_<iter>.pt \
  --num_envs=1
```

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
