# Environment Initialization

## Recommended Versions

- Python `3.11`
- Isaac Sim `5.1.0`
- Isaac Lab `2.3.0`
- CUDA / PyTorch stack supplied by the Isaac Lab environment

The repository installs a local Isaac Lab extension named `robot_lab`.

## Install

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e source/robot_lab
```

Optional CusRL entrypoints:

```bash
python -m pip install -e "source/robot_lab[cusrl]"
```

## Verify Task Registration

```bash
python scripts/maintenance/list_envs.py
```

The command should list Go2, Go2-X5 locomotion, staged Go2-X5 training, UMI prototype, ground-pick, tabletop, and door task IDs.

## Common Runtime Paths

| Path | Meaning |
| --- | --- |
| `logs/rsl_rl/<experiment>/<run>/` | RSL-RL checkpoints, params, exports, and videos. |
| `outputs/<date>/` | Hydra / Isaac Lab run outputs. |
| `docs/appends/media/` | Local visual check screenshots; ignored by git. |
| `source/robot_lab/data/Robots/go2_x5` | Go2-X5 URDF, MuJoCo URDF, USD, config, and meshes. |

## Asset Conversion

URDF to USD:

```bash
python scripts/assets/convert_urdf.py <INPUT_URDF> <OUTPUT_USD> --headless
```

MJCF to USD:

```bash
python scripts/assets/convert_mjcf.py <INPUT_MJCF> <OUTPUT_USD> --headless
```

## Optional Conda Usage

If using the local Isaac Lab Conda environment:

```bash
conda run -n env_isaaclab python scripts/maintenance/list_envs.py
```

Use the same environment for training, replay, and smoke checks so Isaac Lab, Isaac Sim, PyTorch, and extension imports match.
