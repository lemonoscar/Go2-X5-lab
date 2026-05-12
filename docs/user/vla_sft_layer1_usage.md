# VLA-SFT Layer 1 使用指南

> **版本**: 1.0
> **日期**: 2026-04-11
> **状态**: Layer 1 (Basic Grasp) 已实现

---

## 目录结构

```
source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft/
├── __init__.py                    # 模块入口
├── scenes/                        # 场景定义
│   ├── __init__.py
│   └── basic_grasp.py            # Layer 1 场景配置 (A1-A3)
├── configs/                       # 环境配置
│   ├── __init__.py
│   └── basic_cfg.py              # Basic Grasp 环境配置
├── mdp/                          # MDP 函数
│   ├── __init__.py
│   └── scene_randomization.py   # 场景随机化
├── data_collection/              # 数据采集工具
│   ├── __init__.py
│   ├── scene_manager.py         # 场景管理器
│   └── instruction_generator.py # 指令生成器
```

相关测试已迁移到 `test/unit/test_vla_sft.py`。

---

## 快速开始

### 1. 场景管理器使用

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft import VLASSceneManager

# 初始化场景管理器
manager = VLASSceneManager(seed=42)

# 采样一个场景
scene = manager.sample_scene(layer="basic", scene_type="a1_ground_grasp")
print(f"Scene ID: {scene.scene_id}")
print(f"Scene Type: {scene.scene_type}")

# 获取随机化参数
params = manager.get_randomization_params(scene)
print(f"Object Position: {params['object_pose']}")
print(f"Object Color: {params['object_color']}")

# 生成指令
instruction = manager.generate_instruction(scene)
print(f"Instruction: {instruction}")
```

### 2. 直接使用场景配置

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft.scenes import (
    BasicGraspSceneA1,
    BasicGraspSceneA2,
    BasicGraspSceneA3,
    BasicGraspSceneRegistry,
)

# 使用预定义场景
scene = BasicGraspSceneA1()
pos, quat = scene.sample_object_pose()
color = scene.sample_object_color()

# 使用场景注册表
registry = BasicGraspSceneRegistry()
scene = registry.get_scene("a1_ground_grasp_005")
```

### 3. 指令生成器使用

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft.data_collection import InstructionGenerator

gen = InstructionGenerator(seed=42)

# 生成指令
instruction = gen.generate("basic_grasp", {"object": "cube"})
print(instruction)  # "pick up the cube from the ground"

# 生成多个变体
instructions = [
    gen.generate("basic_grasp", {"object": "cube"})
    for _ in range(10)
]
```

---

## 场景类型说明

### A1: 单物体地面抓取 (`a1_ground_grasp`)

- **描述**: 目标物体在地面上，无障碍物
- **物体**: cube, sphere, cylinder
- **位置范围**: x∈[-0.15, 0.15], y∈[-0.25, 0.0], z∈[0.02, 0.05]
- **数量**: 10 个场景

### A2: 单物体桌面抓取 (`a2_table_grasp`)

- **描述**: 目标物体在桌面上，固定高度
- **物体**: cube, sphere, cylinder, bowl, cup
- **桌面高度**: 0.74 m
- **数量**: 10 个场景

### A3: 多物体简单 clutter (`a3_simple_clutter`)

- **描述**: 目标物体周围有 3-5 个干扰物体
- **物体**: 同 A1 + 干扰物
- **Clutter 数量**: 3-5 个
- **数量**: 10 个场景

---

## 运行测试

```bash
cd <repo-root>
python test/unit/test_vla_sft.py
```

预期输出:
```
============================================================
VLA-SFT Layer 1 Tests
============================================================
Testing BasicGraspSceneRegistry...
  Scene counts: {'a1_ground_grasp': 10, 'a2_table_grasp': 10, 'a3_simple_clutter': 10}
  Total scenes: 30
  Sampled scene: a1_ground_grasp_000, type: a1_ground_grasp
  Got scene: a1_ground_grasp_005
  ✓ BasicGraspSceneRegistry tests passed!
...
All tests passed! ✓
============================================================
```

---

## 下一步工作

1. **Layer 2 实现** - 移动抓取场景
2. **Expert Policy** - 专家策略用于数据采集
3. **Data Collector** - 完整的数据采集 pipeline
4. **环境集成** - 与 Isaac Lab 环境集成

---

## API 参考

### VLASSceneManager

```python
class VLASSceneManager:
    def __init__(self, seed: Optional[int] = None, enable_basic_layer: bool = True)
    def sample_scene(self, layer: str = "basic", scene_type: Optional[str] = None) -> SceneConfig
    def list_scenes(self, layer: Optional[str] = None, scene_type: Optional[str] = None) -> List[str]
    def get_randomization_params(self, scene_config: SceneConfig) -> Dict[str, Any]
    def generate_instruction(self, scene_config: SceneConfig, object_type: Optional[str] = None) -> str
```

### BasicGraspSceneConfig

```python
@dataclass
class BasicGraspSceneConfig:
    scene_id: str
    scene_type: str
    object_types: List[str]
    object_size_range: Tuple[float, float]
    position_range: Dict[str, Tuple[float, float]]
    orientation_range: Dict[str, Tuple[float, float]]

    def sample_object_pose(self, rng: Optional[random.Random] = None) -> Tuple[np.ndarray, np.ndarray]
    def sample_object_color(self, rng: Optional[random.Random] = None) -> Tuple[float, float, float]
    def sample_object_type(self, rng: Optional[random.Random] = None) -> str
    def generate_instruction(self, object_type: str, rng: Optional[random.Random] = None) -> str
```

---

## 问题排查

### 导入错误

如果遇到导入错误，确保 `robot_lab` 在 Python 路径中：

```bash
export PYTHONPATH="<repo-root>/source/robot_lab:$PYTHONPATH"
```

### 依赖问题

确保 Isaac Lab 已正确安装：

```bash
cd /path/to/IsaacLab
pip install -e .
```
