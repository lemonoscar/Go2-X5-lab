# Go2-X5 相机可视化测试脚本

## 概述

这些脚本用于测试和可视化 Go2-X5 机器人的两个相机：
- **Dog Camera** (ego-view): 安装在机器狗头部
- **Arm Camera** (wrist view): 安装在机械臂末端

## 使用方法

### 1. 主可视化脚本

```bash
python scripts/visualize_camera.py [选项]
```

**命令行参数:**
```
--dog_cam_pos x y z       # 狗相机相对base的位置 (默认: 0.30 0.0 0.16)
--dog_cam_rot x y z w     # 狗相机旋转四元数 (默认: -0.3799 0.5963 0.5963 -0.3799)
--arm_cam_pos x y z       # 臂相机相对arm_link6的位置 (默认: 0.08657 0.0 0.0)
--arm_cam_rot x y z w     # 臂相机旋转四元数 (默认: 0.5 -0.5 0.5 -0.5)
--save_images             # 保存相机图像到 camera_output/ 目录
--no_render               # 无头模式（不显示GUI）
```

### 2. 预设配置脚本

| 脚本 | 描述 | 臂相机朝向 |
|------|------|-----------|
| `test_cam_current.sh` | 当前配置 | 向后看（180°旋转） |
| `test_cam_forward.sh` | **推荐** | 向前看 |
| `test_cam_downward.sh` | 俯视配置 | 向下45° |
| `test_cam_elevated.sh` | 抬高配置 | 向前下30°，位置上移3cm |

### 3. 运行示例

```bash
# 测试当前配置（在Isaac Sim中可视化）
./scripts/test_cam_current.sh

# 测试推荐配置（相机向前看）
./scripts/test_cam_forward.sh

# 保存相机图像到文件
./scripts/test_cam_forward.sh --save_images

# 无头模式运行（不显示GUI，只保存图像）
./scripts/test_cam_forward.sh --save_images --no_render

# 自定义参数测试
python scripts/visualize_camera.py \
    --arm_cam_pos 0.10 0.0 0.02 \
    --arm_cam_rot 0 0 0 1 \
    --save_images
```

## 相机位置参考

### Dog Camera (ego-view)
- 父级: `Robot/base`
- 默认位置: (0.30, 0.0, 0.16) - 前方30cm，上方16cm
- 用途: 环境导航、物体定位

### Arm Camera (wrist view)
- 父级: `Robot/arm_link6`
- 默认位置: (0.08657, 0.0, 0.0) - 末端执行器中心
- 参考点:
  - 左夹爪: (0.08657, 0.0249, 0.0)
  - 右夹爪: (0.08657, -0.0249, 0.0)
  - 末端中心: (0.08657, 0.0, 0.0)
- 用途: 精确抓取、物体操作

## 四元数速查

```
# 向前看 (无旋转)
0 0 0 1

# 向下45°
0.383 0 0 0.924

# 向下30°
0.259 0 0 0.966

# 绕Y轴旋转180° (向后看)
0 1 0 0

# 当前配置 (绕(1,0,1)轴转180°)
0.5 -0.5 0.5 -0.5
```

## 输出

运行脚本后会：
1. 打印相机配置信息
2. 打印世界坐标系中的相机位置
3. 在Isaac Sim中显示相机位置（小金字塔形状）
4. 如果使用 `--save_images`，保存图像到 `camera_output/` 目录

## 调试提示

1. **相机看不到物体**: 检查旋转四元数，确保相机朝向正确
2. **相机被夹爪遮挡**: 尝试抬高相机位置 (增加z值)
3. **相机视角太窄**: 修改配置中的 `focal_length` 参数
