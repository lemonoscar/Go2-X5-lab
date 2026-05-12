# Go2-X5-lab 文档说明

本文档是 `docs/` 目录的中文入口，用于说明当前文档体系、各目录用途、推送规则和后续维护要求。

## 总体原则

`docs/` 目录按用途拆分，不再混放开发草稿、训练记录、用户指南和历史报告。

- 给 Codex 使用的当前开发资料放在本地目录 `development/`，不推送远端。
- 给开发者运行项目使用的说明放在 `user/`。
- 当前 DogOnly PPO 训练记录放在 `train/`，仅本地保存。
- 过期方案、旧检查报告、非当前训练路线放在 `history/`，仅本地保存。
- 图片和运行截图统一放在 `appends/`。

## 目录说明

| 目录 | 用途 | 推送规则 |
| --- | --- | --- |
| `development/` | Codex 读取的本地开发文档，包括仓库地图、当前 RL policy、VLA-RL 联合接口、工作规范和 `PLANS.md`。 | 忽略推送 |
| `user/` | 给开发者使用的文档，包括环境初始化、框架说明、训练回放、键盘控制、相机可视化和 VLA-SFT 使用说明。 | 可推送 |
| `train/` | 当前 DogOnly PPO 的训练规范、奖励权重、阶段记录和 checkpoint 说明。 | 忽略推送 |
| `history/` | 已过期或非当前路线的开发文档、旧报告、旧训练路线和设计草案。 | 忽略推送 |
| `appends/images/` | README 和文档使用的稳定图片。 | 可推送 |
| `appends/media/` | 本地运行截图、视觉检查图和临时媒体输出。 | 忽略推送 |

## Development 文档

`development/` 是 Codex 修改项目时必须优先读取的目录，当前按以下顺序组织：

1. `01_repository_map.md`：仓库主要目录和各模块职责。
2. `02_current_rl_policy.md`：当前 DogOnly PPO 的输入、输出、任务 ID、配置路径和 reward 设计边界。
3. `03_vla_rl_integration.md`：VLA-RL 联合部分的任务接口、10 维高层 action contract、tabletop/door/VLA-SFT/UMI 相关路径。
4. `04_workflow.md`：工作规范，尤其是代码修改前必须先写计划。
5. `PLANS.md`：代码修改计划文件，必须等待人工审核通过后才能实施。

关键规则：以后任何代码修改都必须先在 `docs/development/PLANS.md` 中写计划，状态标记为 `Pending Human Review`，等待人工明确批准后再开始实现。Codex 不可以自己审核、自己批准、自己直接改代码。

代码修改完成并验证后，还必须做两件收尾工作：

1. 更新相关 `docs/development` 文档，保证当前架构、policy、VLA-RL 接口或工作规范是最新的。
2. 将已执行的计划从 `docs/development/PLANS.md` 复制一份到 `docs/history`，使用类似 `archive_plan_YYYY-MM-DD_<short_slug>.md` 的文件名，并在首行写入过期/历史归档提示。

## User 文档

`user/` 只保留开发者需要直接阅读和执行的文档：

- `environment.md`：环境初始化、安装和任务注册检查。
- `network_and_framework.md`：Isaac Lab、ManagerBased 环境、RSL-RL 和 DogOnly PPO 框架说明。
- `training_and_replay.md`：训练、resume、回放、自定义地图回放和 checkpoint 迁移命令。
- `keyboard_control.md`：键盘控制说明。
- `camera_visualization.md`：相机可视化和测试脚本说明。
- `vla_sft.md`：VLA-SFT 用户入口。
- `vla_sft_layer1_usage.md`、`vla_sft_layer2_usage.md`：VLA-SFT 分层使用说明。
- `vla_sft_visualization.md`：VLA-SFT 场景可视化说明。
- `vla_sft_scene_parameters.yaml`：VLA-SFT 场景参数。

`user/` 下不再保留 `vla_sft/` 二级目录；必要内容已经扁平化到 `docs/user`。

## Train 文档

`train/` 只记录当前 DogOnly PPO 相关训练内容：

- `dogonly_ppo_training_standard.md`：当前训练记录规范。
- `dogonly_ppo_p1_foundation_flat.md`：P1 foundation flat 训练记录。
- `dogonly_ppo_p2_rough_transfer_arm_warmup.md`：P2 rough transfer / arm warmup 训练记录。

非当前 DogOnly PPO 模型的训练阶段文档已经迁入：

```text
docs/history/non_dogonly_paper_driven_training_route/
```

## History 文档

`history/` 存放不再作为当前指导依据的文档。所有文件首行必须包含过期提示。

当前命名规则：

- `archive_*`：归档的旧方案、旧报告、旧日志或旧设计稿。
- `non_dogonly_*`：非当前 DogOnly PPO 路线的训练资料。

这些文档只用于追溯，不作为当前开发或训练依据。

## Appends 目录

图片和媒体统一放在 `appends/`：

- `appends/images/`：稳定图片，例如 root `README.md` 中展示 Go2 / Go2-X5 的图片。
- `appends/media/`：运行截图、视觉检查输出、临时 media 文件。

不要再新增 `docs/imgs` 或 `docs/media`。

## 推送范围

当前文档推送白名单由 `.gitignore` 控制，主要包括：

- 根目录 `README.md`
- 根目录 `requirements.txt`
- `docs/README.md`
- `docs/Readme_CN.md`
- `docs/user/**`
- `docs/appends/images/**`

默认忽略：

- `docs/train/**`
- `docs/development/**`
- `docs/history/**`
- `docs/appends/media/**`
- 其他临时 Markdown/TXT 文档

## 维护要求

新增或修改文档时按以下规则放置：

1. 架构、策略接口、Codex 工作规范：放入 `development/`。
2. 安装、运行、训练、回放、可视化命令：放入 `user/`。
3. 训练奖励、权重、checkpoint、实验记录：放入 `train/`。
4. 过期方案、旧日志、一次性报告：放入 `history/` 并在首行写过期提示。
5. 图片和媒体：放入 `appends/images` 或 `appends/media`。
6. 已执行的 `PLANS.md` 计划：复制归档到 `history/archive_plan_YYYY-MM-DD_<short_slug>.md`。
