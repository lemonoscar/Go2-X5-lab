# Go2-X5 与 UMI-on-Legs 参数对齐和修改建议书

## 1. 目标

这份建议书的目标不是直接改代码，而是先把 `umi-on-legs` 中能作为“对齐基准”的参数抽出来，和当前 `Go2-X5-lab` 做逐项对照，再给出一套可执行的修改方案。

你的核心诉求可以拆成 4 类：

1. 机器狗站姿要更低，更接近蹲姿，不要现在这种偏高、偏容易摔的默认姿态。
2. 抬脚要更高，不能是贴地拖着走。
3. 机械臂默认姿势希望固定到 `0, 1.57, 1.57, 0, 0, 0`。
4. `kp scale` 以及相关控制器参数，希望尽量和 `umi-on-legs` 同步。

结论先写在前面：

- 你的判断基本成立。`Go2-X5-lab` 当前默认资产初值、控制刚度和一部分 locomotion 奖励，确实和 `umi-on-legs` 的设计风格不一致。
- 但有一点要单独说明：我没有在 `umi-on-legs` 的训练配置里找到机械臂默认姿势 `0, 1.57, 1.57, 0, 0, 0` 这个值。`umi-on-legs` 训练配置里能找到的 arm offset 是 `0.0, 0.3, 0.5, 0.0, 0.0, 0.0`，而 `Go2-X5-lab` 的 `play.py` 内置 pose 序列里有一个近似值 `0, 1.57, 1.27, 0, 0, 0`。所以如果你坚持 `0, 1.57, 1.57, 0, 0, 0`，这更像是“新的产品决策”，不是严格意义上的“和 umi 配置同步”。

## 2. UMI-on-Legs 可作为基准的参数

### 2.1 控制器 scale / kp / kd

`umi-on-legs` 明确把这三个量视为新机器人最重要的参数，见：

- `umi-on-legs/mani-centric-wbc/docs/wbc.md:67`

实际配置见：

- `umi-on-legs/mani-centric-wbc/config/env/env_go2ARX5.yaml:33-122`

提取结果如下：

### 腿部控制器

- `scale`: 12 个腿关节全部是 `0.25`
- `kp`: 12 个腿关节全部是 `40.0`
- `kd`: 12 个腿关节全部是 `1.0`

### 机械臂控制器

- `scale`: 6 个臂关节也全部是 `0.25`
- `kp`: `[100, 100, 100, 20, 20, 5]`
- `kd`: `[3, 3, 3, 2, 1, 0.5]`

### 机器人默认 offset

`umi-on-legs/mani-centric-wbc/config/env/env_go2ARX5.yaml:98-122`

腿部 offset：

- FR: `[0.1, 0.8, -1.5]`
- FL: `[-0.1, 0.8, -1.5]`
- RR: `[0.1, 1.0, -1.5]`
- RL: `[-0.1, 1.0, -1.5]`

机械臂 offset：

- `[0.0, 0.3, 0.5, 0.0, 0.0, 0.0]`

这说明 UMI 的腿部姿态不是“全腿统一 0.8 / -1.5”，后腿比前腿更蹲一些。

### 2.2 初始 base 高度

`umi-on-legs/mani-centric-wbc/config/env/env_go2ARX5.yaml:147-151`

- `init_state.pos.z = 0.3`

这比当前 `Go2-X5-lab` 的 `0.38` 明显更低。

### 2.3 locomotion 任务里的高度目标

`umi-on-legs` locomotion 不是只盯一个固定 base 高度，而是给一个采样区间：

- `umi-on-legs/mani-centric-wbc/config/env/tasks/locomotion6d.yaml:6-10`
- `umi-on-legs/mani-centric-wbc/legged_gym/env/isaacgym/task.py:306-375`

提取结果：

- `z_height_range = [0.1, 0.4]`
- `z_height_reward_scale = 0.5`

这表示它在训练时会给 base 一个随机高度目标，而不是单一固定值。

另外在 reaching 配置里还显式用了：

- `umi-on-legs/mani-centric-wbc/config/env/constraints/root_height.yaml:1-7`

结果：

- `target_height = 0.35`

### 2.4 抬脚相关设计

`umi-on-legs` 里没有一个和你描述完全一致的“必须抬很高”的单独 foot height reward，但它有两套强相关机制：

1. `feet_air_time`
- `umi-on-legs/mani-centric-wbc/config/env/tasks/local_2d_vel.yaml:21-25`
- `umi-on-legs/mani-centric-wbc/legged_gym/env/isaacgym/task.py:229-247`
- 奖励 scale 为 `1.0`
- 逻辑是鼓励有足够长的离地时间

2. `feet_drag`
- `umi-on-legs/mani-centric-wbc/config/env/default_constraints.yaml:1-9`
- `umi-on-legs/mani-centric-wbc/config/env/constraints/feet_drag.yaml:1-10`

关键参数：

- `penalty_feet_drag_height = 0.1`
- `violation_feet_drag_height = 0.04`
- `violation_feet_drag_speed = 0.05`
- 在 `combo_go2ARX5_locomotion6d.yaml:21-27` 中，`feet_drag.penalty_weight` 被覆盖为 `-0.01`

这套设计的含义是：

- UMI 不一定要求摆腿轨迹“很高”，
- 但它明确不允许机器人在低高度、还有平面速度的时候拖脚。

### 2.5 固定臂 locomotion

`umi-on-legs/mani-centric-wbc/config/env/combo_go2ARX5_fixed_locomotion6d.yaml:29-62`

固定臂 locomotion 里，额外固定动作是 6 个零，arm offset 也被设置为全零：

- arm extra action: 全 0
- arm offset: 全 0

这说明 UMI 内部其实存在两种 arm 默认语义：

1. 全模型配置 `env_go2ARX5.yaml`：arm offset 是 `0, 0.3, 0.5, 0, 0, 0`
2. fixed-arm locomotion：arm offset 是 `0, 0, 0, 0, 0, 0`

因此如果你要说“完全对齐 umi”，必须先明确你要对齐的是哪条路线。

## 3. Go2-X5-lab 当前参数现状

### 3.1 默认资产初值

`Go2-X5-lab/source/robot_lab/robot_lab/assets/go2_x5.py:41-79`

当前值：

- base 初始高度：`0.38`
- 腿部默认 joint_pos：
  - 左右 hip: `0.0`
  - 前后 thigh: 全部 `0.8`
  - calf: 全部 `-1.5`
- 机械臂默认 joint_pos：
  - `[0, 0, 0, 0, 0, 0]`

直接看这组值就能解释你说的两个问题：

1. 站姿偏高：
   - base 高度 `0.38` 明显高于 UMI 的 `0.3`
   - 后腿没有采用 UMI 那种更蹲的 `1.0`

2. arm 默认姿态完全不对齐：
   - 现在是全零，不是你想要的 `0, 1.57, 1.57, 0, 0, 0`
   - 也不是 UMI `env_go2ARX5.yaml` 里的 `0, 0.3, 0.5, 0, 0, 0`

### 3.2 当前 PD/电机参数

同文件：

- 腿：
  - `stiffness = 25.0`
  - `damping = 0.5`
- 臂：
  - `stiffness = 20.0`
  - `damping = 0.5`

这和 UMI 差异很大：

- 腿：`25/0.5` vs `40/1`
- 臂：`20/0.5` vs `[100,100,100,20,20,5] / [3,3,3,2,1,0.5]`

当前 `Go2-X5-lab` 的控制器明显更软。

### 3.3 当前动作 scale

基础动作定义：

- `Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:127-129`

默认统一 scale：

- `0.5`

Go2-X5 路线里又被覆盖为：

- `train_route_env_cfg.py:137-144`
- `rough_env_cfg.py:110-118`
- `train_route_env_cfg.py:777-783`

当前主路线的 leg action scale：

- hip: `0.125`
- thigh: `0.25`
- calf: `0.25`

当前 arm action scale：

- rough/base 路线：`0.1`
- flat 路线局部放大到 `1.2 / 1.2 / 1.2 / 0.8 / 0.7 / 0.7`

如果按 UMI 的“controller scale = 0.25 for all joints”理解，当前 Go2-X5-lab 至少有两个偏差：

1. hip scale 只有 `0.125`
2. arm scale 完全没有统一到 UMI 风格

### 3.4 当前 base 高度目标

`Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/rough_env_cfg.py:156-163`

当前 rough 路线：

- `base_height_l2.target_height = 0.33`

dog-only 路线：

- `train_route_env_cfg.py:868`
- `base_height_l2.target_height = 0.33`

所以当前训练是在“资产初始高度 0.38，但奖励目标高度 0.33”的组合下做的。这个组合本身就不干净：

- reset 出来偏高，
- reward 再把它往 0.33 拉，
- 但腿部默认 offset 又不像 UMI 那样更蹲，
- 会导致站立几何和奖励目标之间不够一致。

### 3.5 当前抬脚相关设计

基础奖励定义：

- `velocity_env_cfg.py:537-545`
- `velocity_env_cfg.py:604-623`
- `mdp/rewards.py:503-548`

当前通用项：

- `feet_air_time.threshold = 0.5`
- `feet_height.target_height = 0.05`
- `feet_height_body.target_height = -0.3`

dog-only 路线被改成：

- `train_route_env_cfg.py:840`
- `train_route_env_cfg.py:869-870`

结果：

- `feet_air_time.weight = 0.16`
- `feet_air_time.threshold = 0.35`
- `feet_height_body.weight = -4.5`
- `feet_height_body.target_height = -0.16`

这套设计和 UMI 的差异在于：

1. Go2-X5-lab 主要靠 body-frame 脚高目标来管摆腿
2. UMI 主要靠 air-time + anti-drag

当前这套 `feet_height_body.target_height = -0.16` 不一定错，但它更像“规定摆腿几何”，不是“禁止拖脚”。两者训练出来的步态气质会不同。

### 3.6 当前 play 默认 arm pose

`Go2-X5-lab/scripts/reinforcement_learning/rsl_rl/play.py:507-510`

内置默认 pose set 里目前有：

- `[0, 1.57, 1.27, 0, 0, 0]`
- `[1.0, 0, 0, 0, 0, 0]`

所以即使不谈训练，只看 `play`，当前默认姿态也不是你要的 `0, 1.57, 1.57, 0, 0, 0`。

## 4. 差异总结

### 4.1 站姿

当前 `Go2-X5-lab` 问题：

- base init z 太高：`0.38`
- 后腿不够蹲：后腿 thigh 还是 `0.8`
- base reward target 与 reset 几何不完全一致

相对 UMI：

- UMI init z 是 `0.3`
- UMI 后腿 thigh 是 `1.0`
- UMI 的默认 offset 更像前低后蹲的承重姿态

### 4.2 抬脚

当前 `Go2-X5-lab`：

- 主要依赖 `feet_height_body = -0.16`
- 没有直接引入 UMI 风格的 `feet_drag` 约束

相对 UMI：

- UMI 用 `feet_air_time + feet_drag`
- 核心不是“脚摆到某个固定高度”，而是“不能低空滑动拖脚”

### 4.3 arm 默认姿态

当前 `Go2-X5-lab`：

- 资产默认 pose 是全零
- `play` 默认 pose 是 `0, 1.57, 1.27, 0, 0, 0`

UMI：

- 全模型配置的 arm offset 是 `0, 0.3, 0.5, 0, 0, 0`
- fixed-arm locomotion 路线是全零
- 没找到 `0, 1.57, 1.57, 0, 0, 0`

所以这一项需要你明确：你是要“按 umi 同步”，还是“采用你定义的新默认姿态”。

### 4.4 scale / kp / kd

当前 `Go2-X5-lab` 和 UMI 的差异是最大的，而且这是高优先级问题。

对比：

- 腿 scale：`0.125/0.25/0.25` vs UMI 全 `0.25`
- 腿 stiffness/damping：`25/0.5` vs UMI `40/1`
- 臂 stiffness/damping：`20/0.5` vs UMI 分关节 `[100,100,100,20,20,5] / [3,3,3,2,1,0.5]`

这说明现在 Go2-X5-lab 的控制器“风格”不是 umi 风格。

## 5. 建议的对齐方案

我建议按“先几何，再控制器，再奖励”这个顺序改，不要一次全动。

### 第一阶段：先把默认站姿对齐

建议修改：

1. `assets/go2_x5.py`
- `init_state.pos.z: 0.38 -> 0.30`

2. `assets/go2_x5.py`
- 腿部默认 joint pose 改成更接近 UMI：
  - FR: `[0.1, 0.8, -1.5]`
  - FL: `[-0.1, 0.8, -1.5]`
  - RR: `[0.1, 1.0, -1.5]`
  - RL: `[-0.1, 1.0, -1.5]`

原因：

- 这是最直接解决“站太高、不够蹲”的办法。
- 仅仅改 reward target，不改资产初值，不足以解决初始姿态几何问题。

### 第二阶段：把控制器参数对齐到 UMI 风格

建议修改：

1. 腿部 actuator
- `stiffness: 25 -> 40`
- `damping: 0.5 -> 1.0`

2. 臂部 actuator
- 不建议继续用单一 `20 / 0.5`
- 应拆成至少 3 组或 6 组 actuator，按关节配置：
  - joint1-3: `kp=100, kd=3`
  - joint4: `kp=20, kd=2`
  - joint5: `kp=20, kd=1`
  - joint6: `kp=5, kd=0.5`

3. 动作 scale
- 若目标是对齐 UMI 风格，dog-only 腿部 action scale 建议统一评估为 `0.25`
- 最少先做一个 A/B：
  - A 组：维持现在 `0.125/0.25/0.25`
  - B 组：改成全腿 `0.25`

原因：

- UMI 文档自己就把 `scale/kp/kd` 视为最关键参数。
- 当前 Go2-X5-lab 这三项没有对齐，训练出来的稳定性和真实落地气质都会偏掉。

### 第三阶段：抬脚逻辑改成“anti-drag 主导”

建议修改：

1. 在 Go2-X5-lab 引入 UMI 风格 `feet_drag` 约束/奖励
- 优先级高于单纯把 `feet_height_body` 再调大

2. dog-only 路线中，保留 `feet_air_time`，并提高它的重要性
- 当前是 `0.16`
- 可以做两组实验：`0.16` 和 `0.25`

3. `feet_height_body.target_height`
- 如果你明确想“更高的摆腿”，可以先从 `-0.16 -> -0.10` 做实验
- 不建议直接跳到非常激进的值，否则容易出现夸张抬腿和速度跟踪退化

4. 训练目标应该从“脚要高”改成“脚不能低空拖动”
- 这是 umi 的思路
- 也是更稳、更可迁移的思路

原因：

- “高抬腿”本身不是目标，清障、不拖脚、不过度耗能才是目标。
- 单独把 `feet_height_body` 提很高，容易把步态推向夸张摆腿，不一定更稳。

### 第四阶段：明确 arm 默认姿态策略

这一项必须先选一个基准。

#### 方案 A：严格按 UMI 训练配置对齐

把默认 arm offset 改成：

- `[0.0, 0.3, 0.5, 0.0, 0.0, 0.0]`

这是最接近 `umi-on-legs/config/env/env_go2ARX5.yaml` 的做法。

#### 方案 B：按你指定的新默认姿态

把默认 arm offset 改成：

- `[0.0, 1.57, 1.57, 0.0, 0.0, 0.0]`

同时同步修改：

1. `assets/go2_x5.py` 的 arm 初始 joint_pos
2. `commands.arm_joint_pos` 使用 `use_default_offset=True` 的所有任务
3. `play.py` 的 `default_pose_set_raw`
4. 任何假设“全零 arm 默认位姿”的奖励项或 warmup 逻辑

我的判断：

- 如果你是为了“机械臂看起来更正确、更像实际期望姿态”，方案 B 合理。
- 但这不是“和 umi 训练配置同步”，而是“在 Go2-X5-lab 中建立新的统一 arm home pose”。

## 6. 推荐的最终修改顺序

建议按下面顺序做，而不是一口气全改：

1. 先改资产默认几何
- base z
- 腿部默认姿态
- arm 默认姿态

2. 再改 PD 控制器
- 腿 `40/1`
- 臂按 UMI 分关节

3. 再改动作 scale
- 优先做腿部统一到 `0.25` 的实验

4. 再改足端逻辑
- 加 `feet_drag`
- 调 `feet_air_time`
- 最后才动 `feet_height_body`

5. 最后重训 dog-only
- 不建议直接在老 checkpoint 上硬迁移
- 因为默认 offset、actuator gains、action scale 一起变了，旧策略分布大概率已经不匹配

## 7. 我建议直接改的文件

### 必改

1. `Go2-X5-lab/source/robot_lab/robot_lab/assets/go2_x5.py`
- 资产默认 base 高度
- 腿部默认 joint_pos
- arm 默认 joint_pos
- actuator stiffness / damping

2. `Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/train_route_env_cfg.py`
- dog-only 路线奖励
- base height target
- feet_air_time
- feet_height_body
- arm command range 中心

3. `Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/rough_env_cfg.py`
- 基础 Go2-X5 路线 action scale
- base height target

4. `Go2-X5-lab/scripts/reinforcement_learning/rsl_rl/play.py`
- 内置 arm 默认 pose set

### 建议新增

1. 在 `mdp/rewards.py` 或 `mdp/constraints.py` 中加入 UMI 风格 `feet_drag`
2. 在 `go2_x5` 训练路线上挂进去

## 8. 风险判断

### 低风险

- 改 `play.py` 默认 arm pose
- 调文档和配置中的 arm pose set

### 中风险

- 把资产默认姿态改成更蹲
- base 初始高度从 `0.38` 改到 `0.30`

### 高风险

- 改 actuator stiffness / damping
- 改 action scale
- 同时改奖励

高风险不代表不能改，而是代表：

- 一旦修改，旧 checkpoint 基本不再是同一分布上的策略
- 最稳妥的做法是从头重训 dog-only，或者至少从非常早期 checkpoint 重新 warmup

## 9. 最终建议

如果你的目标是“让 Go2-X5-lab 真正继承 umi-on-legs 的稳定性风格”，我的建议是：

1. 先按 UMI 对齐腿部几何和控制器
- base z 改低
- 后腿更蹲
- 腿 `kp/kd` 改到 `40/1`
- 腿 scale 认真评估是否统一到 `0.25`

2. 抬脚逻辑改成 `feet_air_time + feet_drag` 主导
- 不要只靠更高的 `feet_height_body`

3. arm 默认姿态单独做产品决策
- 如果你坚持 `0, 1.57, 1.57, 0, 0, 0`，那就明确把它定义成 Go2-X5 的新 home pose
- 不要把它写成“这是 umi 里的值”，因为我在 umi 训练配置里没有找到这个数

4. 不建议继续在当前 dog-only 参数上小修小补
- 当前偏差不是一两个 reward weight 的问题
- 是“默认几何 + 控制器 + locomotion 奖励逻辑”整体风格都和 umi 不一致

## 10. 可直接执行的下一步

如果你认同这份建议书，下一步我建议直接做一版“严格对齐 umi 风格”的代码修改，范围如下：

1. 下蹲站姿对齐
2. 腿部 `kp/kd/scale` 对齐
3. 加 `feet_drag`
4. arm 默认 pose 采用你指定的 `0, 1.57, 1.57, 0, 0, 0`
5. 顺手把 `play.py` 和训练配置一起统一

这样改完之后，再给你一版新的 dog-only 训练命令和验证 checklist。
