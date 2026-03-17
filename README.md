# Go2-X5-lab

`Go2-X5-lab` 是一个面向 `Unitree Go2` 与 `Go2-X5` 的 Isaac Lab 扩展仓库，聚焦于四足 locomotion、带机械臂形态的 locomotion 训练，以及面向迁移鲁棒性的 sim2sim 配置。


| Go2 | Go2-X5 |
| --- | --- |
| <img src="./docs/imgs/unitree_go2.png" alt="Unitree Go2" width="100%"> | <img src="./docs/imgs/go2-x5.png" alt="Go2-X5" width="100%"> |
| 纯四足速度跟踪与地形 locomotion | 四足底盘 + 机械臂形态，支持 flat / rough / foundation / robust |

## 概览

这个仓库覆盖的核心内容包括：

- `Go2` 平地与粗糙地形速度跟踪训练 / 回放
- `Go2-X5` 平地与粗糙地形 locomotion 训练 / 回放
- `Go2-X5` foundation / robust 训练路线
- Isaac Lab 中面向 sim2sim 的延迟、噪声与动力学扰动建模
- Go2 / Go2-X5 资产的 URDF / MJCF 转 USD 工作流

## 快速开始

### 1. 安装

建议使用以下版本组合：

- Python `3.11`
- Isaac Lab `2.3.0`
- Isaac Sim `5.1.0`

安装扩展：

```bash
python -m pip install -e source/robot_lab
```

### 2. 检查任务注册

```bash
python scripts/tools/list_envs.py
```

如果输出中能看到下面列出的 6 个任务，说明扩展安装和任务注册正常。

### 3. 直接训练

Go2-X5 foundation flat：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --headless
```

### 4. 直接回放

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000.pt \
  --num_envs=1
```

如果需要键盘控制单机器人，可追加：

```bash
--keyboard
```

## 支持的任务

当前注册的任务如下：

| 机器人 | Task ID | 任务说明 |
| --- | --- | --- |
| Go2 | `RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0` | 平地速度跟踪 |
| Go2 | `RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` | 粗糙地形速度跟踪 |
| Go2-X5 | `RobotLab-Isaac-Velocity-Flat-Go2-X5-v0` | 平地 locomotion + arm 形态配置 |
| Go2-X5 | `RobotLab-Isaac-Velocity-Rough-Go2-X5-v0` | 粗糙地形 locomotion + arm 形态配置 |
| Go2-X5 | `RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0` | foundation flat 路线，保留 arm 维度但固定默认位姿 |
| Go2-X5 | `RobotLab-Isaac-Velocity-Rough-Go2-X5-Robust-v0` | P2a rough transfer 路线，保留 arm 维度但继续锁在默认位姿 |

相关环境注册入口位于：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5`

## 常用脚本

主要脚本入口如下：

- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`
- `scripts/reinforcement_learning/rsl_rl/play_cs.py`
- `scripts/reinforcement_learning/cusrl/train.py`
- `scripts/reinforcement_learning/cusrl/play.py`
- `scripts/tools/list_envs.py`
- `scripts/tools/migrate_go2_x5_route_checkpoint.py`
- `scripts/tools/convert_urdf.py`
- `scripts/tools/convert_mjcf.py`
- `scripts/tools/clean_trash.py`

用途说明：

- `rsl_rl/train.py`：RSL-RL 训练入口
- `rsl_rl/play.py`：标准 checkpoint 回放入口
- `rsl_rl/play_cs.py`：加载指定 USD 地图的定制回放入口
- `cusrl/train.py` / `cusrl/play.py`：CusRL 工作流入口
- `migrate_go2_x5_route_checkpoint.py`：把旧的 12-action route checkpoint 迁移到新的 arm-aware route 架构
- `convert_urdf.py` / `convert_mjcf.py`：机器人描述文件转换为 USD

## 训练与回放示例

### RSL-RL 训练

Go2 rough：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless
```

Go2-X5 foundation flat：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --headless
```

说明：

- 这个阶段的 policy 仍然包含 arm 相关观测和动作维度。
- 但 `arm_joint_pos` 命令被固定在默认位姿，目的是先学稳底盘，不丢掉后续开 arm 的网络形状。

Go2-X5 P2a rough transfer 从 foundation checkpoint 热启动：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-Robust-v0 \
  --headless \
  --resume \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000.pt
```

说明：

- 这个阶段是纯 `flat -> rough` 迁移：rough terrain 打开，但 arm 仍然锁在默认位姿。
- policy 维度保持不变，因此后续如果要做 `P2b` 再开 arm，不需要再改网络结构。

如果你手里已经有旧架构的 `model_8000.pt`（旧 route 是 `235 -> 12`），先迁移再 resume：

```bash
python scripts/tools/migrate_go2_x5_route_checkpoint.py \
  --input logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000.pt
```

默认会输出：

```bash
logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000_armdims.pt
```

然后用迁移后的 checkpoint 接着训练，并显式跳过旧 optimizer 状态：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Go2-X5-Robust-v0 \
  --headless \
  --resume \
  --no_load_optimizer \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000_armdims.pt
```

### RSL-RL 回放

通用写法：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=<TASK_ID> \
  --checkpoint=<PATH_TO_MODEL>
```

示例：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000.pt \
  --num_envs=1
```

单机器人键盘控制：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0 \
  --checkpoint=logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/model_8000.pt \
  --num_envs=1 \
  --keyboard
```

### 指定地图回放

```bash
python scripts/reinforcement_learning/rsl_rl/play_cs.py \
  --task=<TASK_ID> \
  --checkpoint=<PATH_TO_MODEL> \
  --map=<PATH_TO_USD_MAP>
```

### CusRL

```bash
python scripts/reinforcement_learning/cusrl/train.py \
  --task=<TASK_ID> \
  --headless
```

```bash
python scripts/reinforcement_learning/cusrl/play.py \
  --task=<TASK_ID>
```

## 机器人资产

仓库内包含以下机器人描述文件：

- Go2
  - `source/robot_lab/data/Robots/unitree/go2_description/urdf/go2_description.urdf`
- Go2-X5
  - `source/robot_lab/data/Robots/go2_x5/go2_x5.urdf`
  - `source/robot_lab/data/Robots/go2_x5/go2_x5.mujoco.urdf`

如果需要将描述文件转换为 USD，可使用：

```bash
python scripts/tools/convert_urdf.py <INPUT_URDF> <OUTPUT_USD> --headless
```

```bash
python scripts/tools/convert_mjcf.py <INPUT_MJCF> <OUTPUT_USD> --headless
```

## Sim2sim

这里所说的 `sim2sim`，指的是一组面向迁移鲁棒性的扰动建模与资产准备能力，主要包括：

- 动作延迟
- 动作保持概率
- 动作噪声
- 观测延迟
- 摩擦扰动
- 外力 / 推扰动
- actuator gain drift

相关配置与运行逻辑主要位于：

- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/flat_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/train_route_env_cfg.py`

当前这部分工作的重点是：

1. 在 Isaac Lab 训练与回放中显式建模时序延迟、噪声与动力学扰动。
2. 使用 Go2-X5 的 MuJoCo 形式资产支持跨仿真器对齐与导出。

## 输出目录

训练与运行结果主要写入以下目录：

- `logs/`
  - checkpoint、导出模型、视频等训练产物
- `outputs/`
  - Hydra / Isaac Lab 运行输出

测试已有模型时，通常从 `logs/` 中选择 checkpoint，并传给 `play.py` 的 `--checkpoint`。
