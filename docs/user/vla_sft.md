# Go2-X5 VLA-SFT 使用入口

> 项目：Go2-X5 VLA-SFT 数据采集
> 当前归类：用户侧参数、可视化和数据采集使用文档

## 文档索引

| 文档 | 描述 | 状态 |
|------|------|------|
| [vla_sft_scene_parameters.yaml](vla_sft_scene_parameters.yaml) | 场景参数配置文件 | 完成 |
| [vla_sft_layer1_usage.md](vla_sft_layer1_usage.md) | Layer 1 使用指南 | 完成 |
| [vla_sft_layer2_usage.md](vla_sft_layer2_usage.md) | Layer 2 使用指南 | 完成 |
| [vla_sft_visualization.md](vla_sft_visualization.md) | 场景可视化指南 | 完成 |

长篇场景设计草案已归档到 `../history/archive_vla_sft_scene_design.md`，不再作为用户运行入口。

## 实现进度

### Layer 1: 基础抓取场景 (Basic Grasp)

| 组件 | 文件 | 状态 |
|------|------|------|
| 场景定义 | `source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft/scenes/basic_grasp.py` | 完成 |
| 场景配置 | `source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft/configs/basic_cfg.py` | 完成 |
| 场景随机化 | `source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft/mdp/scene_randomization.py` | 完成 |
| 场景管理器 | `source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft/data_collection/scene_manager.py` | 完成 |
| 指令生成器 | `source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft/data_collection/instruction_generator.py` | 完成 |

代码统计：约 2000 行 Python 代码。

## 快速开始

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft import VLASSceneManager

manager = VLASSceneManager(seed=42)

scene = manager.sample_scene(layer="basic")
params = manager.get_randomization_params(scene)
instruction = manager.generate_instruction(scene)

print(f"Scene: {scene.scene_id}")
print(f"Instruction: {instruction}")
```

---

## 运行建议

1. 先阅读 `vla_sft_layer1_usage.md` 或 `vla_sft_layer2_usage.md`。
2. 使用 `vla_sft_visualization.md` 中的命令检查场景是否能正确渲染。
3. 修改参数时同步更新 `vla_sft_scene_parameters.yaml`。
