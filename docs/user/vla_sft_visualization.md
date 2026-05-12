# VLA-SFT 场景可视化指南

> **版本**: 1.0
> **日期**: 2026-04-11

---

## 前置条件

确保你已经安装了 Isaac Lab 和 Isaac Sim：

```bash
# 设置 Isaac Sim 路径
export ISAACSIM_PATH=/path/to/isaac-sim  # 或 Isaac Lab 自带的 Omniverse
source $ISAACSIM_PATH/setup.sh
```

---

## 基本使用

### 查看 A1 场景 (地面抓取)

```bash
cd <repo-root>
python scripts/visualization/view_vla_sft_scene.py --scene_type a1_ground_grasp
```

### 查看 A2 场景 (桌面抓取)

```bash
python scripts/visualization/view_vla_sft_scene.py --scene_type a2_table_grasp
```

### 查看 A3 场景 (Clutter 抓取)

```bash
python scripts/visualization/view_vla_sft_scene.py --scene_type a3_simple_clutter --num_clutter 5
```

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scene_type` | 场景类型: a1_ground_grasp, a2_table_grasp, a3_simple_clutter | a1_ground_grasp |
| `--object_type` | 物体类型: cube, sphere, cylinder, bowl, cup (None=随机) | None |
| `--seed` | 随机种子 | None |
| `--enable_table` | 强制显示桌子 | False |
| `--num_clutter` | clutter 物体数量 | 0 |
| `--live` | 启动物理仿真 | False |
| `--headless` | 无头模式 (不显示 UI) | False |

---

## 使用示例

### 示例 1: 查看红色方块地面抓取

```bash
python scripts/visualization/view_vla_sft_scene.py \
    --scene_type a1_ground_grasp \
    --object_type cube \
    --seed 42
```

输出:
```
============================================================
VLA-SFT Scene Viewer
============================================================
Scene Type: a1_ground_grasp
Object Type: cube
Object Position: (0.056, -0.124, 0.038)
Object Orientation: (1.000, 0.000, 0.000, -0.045)
Object Color: RGB(0.82, 0.16, 0.12)
Instruction: pick up the cube from the ground
============================================================
```

### 示例 2: 查看桌面抓取 + 物理仿真

```bash
python scripts/visualization/view_vla_sft_scene.py \
    --scene_type a2_table_grasp \
    --object_type sphere \
    --live
```

### 示例 3: 查看 Clutter 场景

```bash
python scripts/visualization/view_vla_sft_scene.py \
    --scene_type a3_simple_clutter \
    --num_clutter 5 \
    --seed 123
```

### 示例 4: 指定物体颜色 (通过种子)

```bash
# 不同的种子会产生不同的物体颜色和位置
python scripts/visualization/view_vla_sft_scene.py --seed 1
python scripts/visualization/view_vla_sft_scene.py --seed 2
python scripts/visualization/view_vla_sft_scene.py --seed 3
```

---

## 交互操作

### 鼠标控制

| 操作 | 功能 |
|------|------|
| 左键拖拽 | 旋转视角 |
| 右键拖拽 | 平移视角 |
| 滚轮 | 缩放 |
| Ctrl+左键 | 选择物体 |

### 键盘控制

| 按键 | 功能 |
|------|------|
| Ctrl+C | 退出 |
| ESC | 退出 (某些 Isaac Sim 版本) |

---

## 故障排查

### 问题 1: 找不到 Isaac Sim

```bash
ValueError: IsaacSim package could not be found.
```

**解决方法**: 设置正确的 Isaac Sim 路径

```bash
export ISAACSIM_PATH=/home/user/OmniKit/isaac-sim
source $ISAACSIM_PATH/setup.sh
```

### 问题 2: 没有显示机器人

```
[WARNING] Go2-X5 USD file not found at ...
[WARNING] Robot will not be displayed
```

**解决方法**: 确保 Go2-X5 的 USD 文件存在。机器人只是不显示，不影响场景查看。

### 问题 3: 窗口无响应

**解决方法**: 使用 `--live` 参数启动物理仿真，或检查显卡驱动。

---

## 批量生成场景截图

你可以使用这个脚本批量生成场景截图用于数据集预览：

```bash
#!/bin/bash
# 生成不同场景的截图

SCENES=("a1_ground_grasp" "a2_table_grasp" "a3_simple_clutter")
OBJECTS=("cube" "sphere" "cylinder")

for scene in "${SCENES[@]}"; do
    for obj in "${OBJECTS[@]}"; do
        for seed in {0..4}; do
            OUTPUT="sim_docs/screenshots/${scene}_${obj}_seed${seed}.png"
            python scripts/visualization/view_vla_sft_scene.py \
                --scene_type $scene \
                --object_type $obj \
                --seed $seed \
                --headless &
            sleep 2  # 等待渲染完成
        done
    done
done
```

---

## 与数据采集集成

这个可视化脚本使用与数据采集相同的场景配置，所以你看到的场景就是实际采集时使用的场景。

要开始数据采集，请参考 `vla_sft_layer1_usage.md`。
