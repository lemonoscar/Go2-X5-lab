# Go2-X5 键盘控制使用说明

## 前提条件

- 已安装 Isaac Lab
- Go2-X5-lab 项目已克隆

## 运行方法

```bash
cd <repo-root>

# 激活 Isaac Lab 环境
conda activate isaaclab

# 运行键盘控制
python scripts/control/keyboard_control_simple.py

# 如果只想看3D窗口，不要摄像头显示
python scripts/control/keyboard_control_simple.py --no_cameras

# 无头模式 (不显示3D窗口)
python scripts/control/keyboard_control_simple.py --headless
```

## 键盘控制

### 底座控制 (Go2 狗)

| 按键 | 功能 |
|------|------|
| W | 前进 |
| S | 后退 |
| A | 左转 |
| D | 右转 |
| Q | 向左平移 |
| E | 向右平移 |

### 机械臂控制

| 按键 | 功能 |
|------|------|
| I/K | 控制关节1 (增/减) |
| J/L | 控制关节2 (增/减) |
| U/O | 控制关节3 (增/减) |
| SPACE | 切换控制的关节组 |

### 夹爪控制

| 按键 | 功能 |
|------|------|
| 1 | 打开夹爪 |
| 2 | 闭合夹爪 |

### 其他

| 按键 | 功能 |
|------|------|
| R | 重置环境 |
| P | 暂停/继续 |
| H | 显示帮助 |
| ESC | 退出 |

## 摄像头显示

运行后会弹出一个窗口显示两个摄像头：
- **Ego View**: 机器人底座摄像头 (向前看)
- **Arm View**: 机械臂手腕摄像头 (向下看)

在摄像头窗口中按 ESC 关闭摄像头显示 (不会退出程序)。

## 常见问题

### 1. 找不到 Isaac Lab

确保已激活 Isaac Lab conda 环境：
```bash
conda activate isaaclab
```

### 2. 摄像头窗口黑屏

等待几秒钟，摄像头需要时间初始化。

### 3. 控制没有反应

确保焦点在 Isaac Sim 窗口上，点击一下 3D 视图。

### 4. 机械臂不动

- 确保已切换到正确的关节组 (按 SPACE)
- 机械臂控制是增量的，需要多次按键才能看到明显变化

## 下一步

熟悉键盘控制后，可以：
1. 手动采集演示数据
2. 测试不同的抓取策略
3. 录制训练数据
