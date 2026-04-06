# Go2-X5 DogOnly PPO 设计文档

## 1. 设计目标

当前 `Go2-X5` 的低层策略不再采用“PPO 直接输出全身 18 个关节动作”的 whole-body joint policy。

当前主设计范式是：

- **PPO 只负责机器狗下肢 12 维动作**
- **机械臂 6 维目标由独立命令接口提供**
- **夹爪 1 维作为上层任务接口显式保留，但当前低层不直接驱动**
- **低层 PPO 的职责是：在上层 arm/抓取命令存在时，保持底盘稳定、完成移动、抑制漂移**

这意味着当前低层策略本质上是：

**mobile base stabilizer + locomotion controller**

而不是：

**whole-body joint-level motion generator**


## 2. 为什么不继续用 18 维 whole-body PPO

旧思路的问题在于，一个 PPO 需要同时学会：

- 四足底盘 locomotion
- 机械臂关节级 tracking
- 底盘与机械臂的耦合补偿
- 静止 / 移动 / 大范围 arm 姿态下的稳定性

这会带来 3 个直接问题：

1. `credit assignment` 太差  
   arm 命令变化引起的重心扰动，会经过接触、姿态、步态再反馈到底盘稳定，链路太长。

2. PPO 容量被错误占用  
   机械臂本来可以由传统位置控制器跟踪，但旧设计让 PPO 也去学 arm joint tracking。

3. 上层策略接口不干净  
   面向 UMI / VLA 时，上层应该输出“任务命令”，而不是直接接管 18 个底层关节。

因此当前设计改为：

- **高层决定 task command**
- **低层 PPO 只负责 base / legs**
- **机械臂保持命令驱动，不进入 PPO action head**


## 3. 当前网络设计范式

### 3.1 总体结构

当前 `DogOnly` 策略的结构是：

- 输入：`260` 维
- 输出：`12` 维
- actor hidden dims：`[512, 256, 128]`
- critic hidden dims：`[512, 256, 128]`
- activation：`ELU`

对应代码：

- [train_route_env_cfg.py](/home/lemon/Issac/Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/train_route_env_cfg.py)
- [rsl_rl_ppo_cfg.py](/home/lemon/Issac/Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/agents/rsl_rl_ppo_cfg.py)

逻辑上可写成：

```text
policy_input(260)
  -> Linear(260, 512) -> ELU
  -> Linear(512, 256) -> ELU
  -> Linear(256, 128) -> ELU
  -> Linear(128, 12)
```

critic 则是：

```text
critic_input(260)
  -> Linear(260, 512) -> ELU
  -> Linear(512, 256) -> ELU
  -> Linear(256, 128) -> ELU
  -> Linear(128, 1)
```


## 4. 当前输入接口

### 4.1 物理输入不是纯 10 维命令，而是“命令 + 状态”

当前低层 PPO **不是**只看高层命令。

当前真正输入的是：

- 机器人当前状态观测
- 加上高层命令接口

也就是说，当前范式是：

```text
policy_input = robot_state_obs + task_command_obs
```

而不是：

```text
policy_input = task_command_only
```

### 4.2 当前 260 维输入组成

当前 `DogOnly` 任务中，策略输入由以下部分组成：

| 项 | 维度 | 含义 |
| --- | --- | --- |
| `base_lin_vel` | 3 | 机身线速度 |
| `base_ang_vel` | 3 | 机身角速度 |
| `projected_gravity` | 3 | 机身姿态重力投影 |
| `velocity_commands` | 3 | 上层给定底盘速度命令 |
| `joint_pos` | 18 | 全身关节相对默认位姿，含腿和机械臂 |
| `joint_vel` | 18 | 全身关节速度，含腿和机械臂 |
| `actions` | 18 | 历史动作观测，当前为 12 维腿动作 + 6 维零 padding |
| `height_scan` | 187 | 地形高度扫描 |
| `arm_joint_command` | 6 | 上层 arm pose 命令 |
| `gripper_command` | 1 | gripper 占位接口 |

合计：

```text
3 + 3 + 3 + 3 + 18 + 18 + 18 + 187 + 6 + 1 = 260
```

### 4.3 关于 `joint_pos / joint_vel`

当前 `joint_pos` 和 `joint_vel` **包含全身关节**，不是只有腿。

也就是：

- 12 条腿关节
- 6 条机械臂关节

都包含在 observation 里。

这是刻意保留的，因为低层 PPO 虽然不输出 arm 动作，但仍然必须知道：

- arm 当前在哪里
- arm 当前速度有多大
- arm 命令和当前状态之间有多大差值


## 5. 当前输出接口

### 5.1 PPO 输出

当前 PPO 输出只有：

- `12` 维腿部动作

对应关节为：

- `FR_hip_joint`
- `FR_thigh_joint`
- `FR_calf_joint`
- `FL_hip_joint`
- `FL_thigh_joint`
- `FL_calf_joint`
- `RR_hip_joint`
- `RR_thigh_joint`
- `RR_calf_joint`
- `RL_hip_joint`
- `RL_thigh_joint`
- `RL_calf_joint`

### 5.2 整机最终控制信号

虽然 PPO 只输出 12 维，但整机控制链依然可以表达：

- 腿：`12`
- 机械臂：`6`
- gripper：`1`

只是后面这 `6 + 1` 不再由 PPO 学出来，而是走上层命令 / 独立控制器。


## 6. 新的封装方式

### 6.1 当前封装边界

当前 `DogOnly` 版本的关键封装是：

1. **策略 head 只对狗腿开放**
2. **arm command 通过 command manager 输入**
3. **arm action term 不消耗 PPO 动作维度**
4. **gripper 在 observation 中显式占位**
5. **历史动作观测保留旧的 18 维布局，便于迁移旧 backbone**

对应实现：

- [actions.py](/home/lemon/Issac/Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/actions.py)
- [observations.py](/home/lemon/Issac/Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py)
- [train_route_env_cfg.py](/home/lemon/Issac/Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/train_route_env_cfg.py)

### 6.2 `ArmCommandPositionAction` 的意义

`ArmCommandPositionAction` 的设计目的不是替代 arm controller，而是明确系统边界：

- PPO 不负责输出 arm 关节动作
- arm 关节直接跟踪 `arm_joint_pos` 命令
- PPO 只从观测中看到 arm 当前状态和 arm 命令

这使得系统职责更清晰：

- arm tracking：由命令驱动
- base stabilization：由 PPO 学习

### 6.3 `last_action_with_padding`

当前策略迁移时，没有直接把历史动作观测从 18 维砍成 12 维，而是保留成：

- 前 12 维：腿动作历史
- 后 6 维：arm padding = 0

这样做的目的是：

- 尽量复用旧 checkpoint 的前层权重
- 减少从 whole-body policy 迁到 dog-only policy 时的结构突变

这是一个**迁移友好型封装**，不是最终必须长期保留的唯一形式。


## 7. 面向 UMI VLA 的底层策略范式

### 7.1 上下层职责划分

面向 UMI / VLA 时，推荐的分层关系是：

#### 上层：UMI VLA / 任务策略

上层输出：

- `cmd_vel(3)`
- `arm_pose(6)`
- `gripper(1)`

也就是一个 **10 维任务命令接口**。

#### 下层：DogOnly PPO

下层接收：

- 当前机器人状态观测
- 上层任务命令

下层输出：

- `12` 维腿动作

#### arm / gripper 控制层

- `arm_pose(6)` 由独立 tracking controller 跟踪
- `gripper(1)` 由 gripper 控制器解释

### 7.2 这套范式为什么适合 UMI VLA

UMI / VLA 更适合输出：

- 意图
- 目标
- task condition

而不是直接输出底层 18 个关节的 joint command。

因此当前低层 PPO 更适合作为：

**task-conditioned locomotion stabilizer**

它的职责是：

- 读取上层的 body/arm task command
- 在机械臂运动、重心变化、静止或移动的情况下稳住底盘
- 生成腿部控制

### 7.3 推荐的 UMI VLA 低层接口

建议把当前系统抽象成下面这个标准接口：

#### 上层输入给低层

```text
cmd_vel:        3
arm_pose:       6
gripper:        1
robot_state:  250
-------------------
total:        260
```

#### 低层输出

```text
dog_joint_action: 12
```

#### 非 PPO 输出

```text
arm_target:     6
gripper_target: 1
```

这意味着：

- 上层 UMI / VLA 不需要理解狗腿控制细节
- 下层 PPO 不需要学抓取语义和 arm 规划


## 8. 当前范式的能力边界

当前设计能做的事情：

- 在 arm 命令存在时训练底盘稳定性
- 学习静止 / 低速移动条件下的 base-arm coupling
- 作为 UMI VLA 下层稳定器使用

当前设计**不**负责的事情：

- 机械臂复杂操作规划
- whole-body joint-level 统一生成
- 末端抓取策略本身

所以如果未来要做更高层任务，应该继续沿着：

- `VLA/UMI -> task command`
- `DogOnly PPO -> locomotion stabilization`
- `arm/gripper controller -> manipulation execution`

而不是回到 “一个 PPO 输出全身所有关节”。


## 9. 面向 sim2real / rl_ras_n 的接口要求

当前这版策略如果要导出到 `rl_ras_n` / MuJoCo / real deployment，部署侧必须同步满足：

- observation dim = `260`
- action dim = `12`
- `actions` 观测必须补零到 `18`
- `height_scan` 必须显式占位到 `187`
- `arm_joint_command` 必须保留 `6`
- `gripper_command` 必须保留 `1`

也就是说，部署侧不能只改模型文件，不改接口配置。

需要同步检查：

- `policy/go2_x5/robot_lab/config.yaml`
- MuJoCo / real runtime 的 observation builder
- action scale 是否已经切到 12 维腿动作
- arm 输出是否已经从 PPO head 剥离


## 10. 后续推荐演进

当前设计已经把结构方向校正到更合理的范式，但后续还可以继续加强：

1. 在 observation 中显式加入 `arm_tracking_error`
2. 在 observation 中显式加入 `arm_command_delta`
3. 把 `gripper_command` 从占位接口发展成真正的 task-conditioned 通道
4. 在 UMI / VLA 接口层明确消息协议：
   - `cmd_vel`
   - `arm_pose`
   - `gripper`
   - `mode / task id`

这四点会让当前低层策略更适合作为一个长期稳定的 UMI VLA 基座。
