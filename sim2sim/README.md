# Go2-X5 Sim2Sim Workspace

This workspace contains the Go2-X5 Gazebo and MuJoCo sim2sim runtime inside
`Go2-X5-lab`. It is self-contained: runtime files are loaded from this
directory, not from the old external `/home/lemon/Issac/rl_sar` checkout.

The ROS package name is still `rl_sar` for compatibility with the migrated
launch files and CMake targets.

Baseline recorded on 2026-05-12:

- Default sim2sim profile: `go2_x5/robot_lab`
- Explicit DogOnly profile: `go2_x5/dog_only_260x12`
- Policy contract: `260 -> 12`
- `actions` observation size: 18, padded from the 12 leg actions
- Active model: `policy/go2_x5/robot_lab/policy.onnx`

## Scope

- Isaac Lab training remains under `source/robot_lab` and
  `scripts/reinforcement_learning`.
- Gazebo, MuJoCo, ROS packages, C++ runtime, robot description, and sim2sim
  policy bundle live here.
- DogOnly is the default runtime contract: PPO writes only the 12 leg actions,
  while arm and gripper commands stay outside the PPO action head.

## Build

Gazebo / ROS2:

```bash
cd /home/lemon/Issac/Go2-X5-lab/sim2sim
source /opt/ros/humble/setup.bash
./build.sh
```

MuJoCo:

```bash
cd /home/lemon/Issac/Go2-X5-lab/sim2sim
./build.sh -mj
```

## Gazebo Runtime Validation

Terminal 1:

```bash
cd /home/lemon/Issac/Go2-X5-lab/sim2sim
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source install/setup.bash
ros2 launch rl_sar gazebo.launch.py rname:=go2_x5
```

Terminal 2:

```bash
cd /home/lemon/Issac/Go2-X5-lab/sim2sim
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source install/setup.bash
ros2 run rl_sar rl_sim
```

In the `rl_sim` window:

1. Press `R` to reset the simulator if needed.
2. Press `0` to enter get-up.
3. Press `1` to enter DogOnly locomotion with the fixed command.
4. Use `W/S/A/D/Q/E` to adjust velocity and yaw.
5. Press `Space` or `5` to clear commands.

Expected startup log after pressing `1`:

```text
Policy manifest OK: profile=go2_x5/robot_lab, ..., shape=260->12, action_obs=18
```

## MuJoCo Runtime Validation

```bash
cd /home/lemon/Issac/Go2-X5-lab/sim2sim
export LD_LIBRARY_PATH=/home/lemon/Issac/Go2-X5-lab/sim2sim/library/mujoco/lib:$LD_LIBRARY_PATH
./cmake_build/bin/rl_sim_mujoco go2_x5 scene_flat
```

Use the same keys as Gazebo. The same `Policy manifest OK` line should be
printed when the RL policy is loaded.

## Policy Profiles

Default DogOnly:

```bash
ros2 run rl_sar rl_sim
./cmake_build/bin/rl_sim_mujoco go2_x5 scene_flat
```

Explicit DogOnly profile:

```bash
GO2_X5_SIM2SIM_CONFIG=dog_only_260x12 ros2 run rl_sar rl_sim
GO2_X5_SIM2SIM_CONFIG=dog_only_260x12 ./cmake_build/bin/rl_sim_mujoco go2_x5 scene_flat
```

## Policy Guardrails

Every runnable profile must include `manifest.yaml`. At runtime the C++ loader
fails fast if any of these do not match:

- profile key
- `policy_mode`
- `model_name`
- observation dimension
- action dimension
- `actions` observation dimension
- model SHA256
- model forward output size

Policy model files under `policy/**/*.pt` and `policy/**/*.onnx`, downloaded
runtime libraries under `library/`, and build outputs are local artifacts and
are ignored by git.
