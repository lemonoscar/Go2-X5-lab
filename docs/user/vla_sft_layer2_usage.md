# VLA-SFT Layer 2 (Mobile Grasp) Usage Guide

## Overview

Layer 2 extends Layer 1 by adding **base navigation** to the grasping task. The robot must navigate to the target object before grasping, introducing coupling between locomotion and manipulation.

## Scene Types

### B1: Open Floor Navigation Grasp
- **Description**: Target object placed 1.5-3m away on open floor
- **Focus**: Long-range navigation and base positioning
- **Scene Count**: 10
- **Key Features**:
  - No obstacles
  - Variable target distance (1.5-3m)
  - Wide target angle range (±90°)

### B2: Constrained Table Approach
- **Description**: Target on table with limited approach space
- **Focus**: Precise base positioning near table
- **Scene Count**: 10
- **Key Features**:
  - Table at (0, 0.5, 0) with size (0.8, 0.6, 0.74)
  - Target at 0.76-0.8m height (on table)
  - Constrained base initialization

### B3: Obstacle Avoidance Navigation
- **Description**: Static obstacles requiring path planning
- **Focus**: Navigation around obstacles
- **Scene Count**: 10
- **Key Features**:
  - 2-4 randomly placed obstacles
  - Obstacle types: boxes, cylinder barriers
  - Target at 1.5-2.5m distance

### B4: Partial Occlusion Grasp
- **Description**: Target partially occluded by tall obstacle
- **Focus**: Viewpoint adjustment and inference
- **Scene Count**: 10
- **Key Features**:
  - Occluder placed between base and target
  - Occluder is a tall block (0.2 × 0.2 × 0.3m)
  - Target at 0.8-1.5m distance

## File Structure

```
vla_sft/
├── scenes/
│   ├── __init__.py          # Exports both Layer 1 and 2
│   ├── basic_grasp.py       # Layer 1 scenes
│   └── mobile_grasp.py      # Layer 2 scenes (NEW)
├── configs/
│   ├── __init__.py          # Exports both configs
│   ├── basic_cfg.py         # Layer 1 config
│   └── mobile_cfg.py        # Layer 2 config (NEW)
├── data_collection/
│   ├── scene_manager.py     # Updated for Layer 2
│   └── instruction_generator.py
└── __init__.py              # Updated exports
```

## Usage Examples

### Scene Registry

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft.scenes import MobileGraspSceneRegistry

# Create registry
registry = MobileGraspSceneRegistry(seed=42)

# Get scene counts
counts = registry.get_scene_counts()
# {'b1_open_floor': 10, 'b2_table_approach': 10, ...}

# Sample by type
b1_scene = registry.sample_scene("b1_open_floor")

# Get specific scene
scene = registry.get_scene("b3_obstacle_avoidance_005")
```

### Scene Manager (Multi-Layer)

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft import VLASSceneManager

# Create manager with both layers
manager = VLASSceneManager(seed=42, enable_basic_layer=True, enable_mobile_layer=True)

# Sample from specific layer
basic_scene = manager.sample_scene(layer="basic")
mobile_scene = manager.sample_scene(layer="mobile")

# Get randomization params
params = manager.get_randomization_params(mobile_scene)
# Returns: target_position, base_init_position, obstacles, occluder, etc.
```

### Direct Scene Usage

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft.scenes import (
    MobileGraspSceneB1,
    MobileGraspSceneB3,
)

# B1: Open floor
b1 = MobileGraspSceneB1()
target_pos, angle, distance = b1.sample_target_pose()
base_pos, base_yaw = b1.sample_base_init_pose()

# B3: Obstacle avoidance
b3 = MobileGraspSceneB3()
target_pos = b3.sample_target_pose()[0]
obstacles = b3.sample_obstacles(target_pos)
```

## Visualization

Launch the Layer 2 scene viewer:

```bash
# B1: Open floor
python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b1_open_floor

# B2: Table approach
python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b2_table_approach --live

# B3: Obstacle avoidance
python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b3_obstacle_avoidance

# B4: Partial occlusion
python scripts/visualization/view_vla_sft_mobile_scene.py --scene_type b4_partial_occlusion --seed 123
```

### Visualization Options

- `--scene_type`: B1, B2, B3, or B4
- `--object_type`: cube, sphere, cylinder, bowl, cup
- `--seed`: Random seed for reproducibility
- `--floor_material`: concrete, wood, tile, grass
- `--live`: Enable physics simulation
- `--headless`: Run without GUI

## Environment Configuration

```python
from robot_lab.tasks.manager_based.manipulation.vla_sft.configs import (
    Go2X5VLASMobileEnvCfg,
    Go2X5VLASMobileEnvCfg_PLAY,
)

# Training configuration
env_cfg = Go2X5VLASMobileEnvCfg()
# 32 environments, 12s episodes, larger workspace

# Play configuration
play_cfg = Go2X5VLASMobileEnvCfg_PLAY()
# Single environment, no observation noise
```

## Key Differences from Layer 1

| Feature | Layer 1 (Basic) | Layer 2 (Mobile) |
|---------|----------------|------------------|
| Navigation | No | Yes |
| Target Distance | 0.4-0.7m | 1.5-3m |
| Episode Length | 8s | 12s |
| Workspace | 4m spacing | 6m spacing |
| Obstacles | No | Yes (B3) |
| Occlusion | No | Yes (B4) |
| Tables | Simple | Constrained approach (B2) |

## Testing

Run Layer 2 tests:

```bash
cd <repo-root>/source/robot_lab/robot_lab/tasks/manager_based/manipulation/vla_sft
python3 -c "
from scenes.mobile_grasp import MobileGraspSceneRegistry, MobileGraspSceneB1, MobileGraspSceneB3, MobileGraspSceneB4

registry = MobileGraspSceneRegistry(seed=42)
print(f'Total scenes: {registry.total_scenes}')

b1 = MobileGraspSceneB1()
pos, angle, dist = b1.sample_target_pose()
print(f'B1 target: {dist:.2f}m at {angle:.2f}rad')

b3 = MobileGraspSceneB3()
target = b3.sample_target_pose()[0]
obstacles = b3.sample_obstacles(target)
print(f'B3 obstacles: {len(obstacles)}')
"
```

## Data Collection Output

Layer 2 data includes:

```
data/vla_sft/mobile/
├── episodes/
│   ├── episode_0001.hdf5
│   ├── episode_0002.hdf5
│   └── ...
├── images/
│   ├── dog_camera/
│   │   ├── episode_0001_step_000.png
│   │   └── ...
│   └── arm_camera/
│       └── ...
└── metadata.json
```

## Next Steps

1. Implement expert policy for navigation + grasp
2. Set up data collection pipeline
3. Implement Layer 3 (Interaction) scenes
4. Implement Layer 4 (OOD) scenes
