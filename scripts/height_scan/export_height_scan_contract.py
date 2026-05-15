#!/usr/bin/env python3
"""Export the DogOnly-Rough height-scan deployment contract.

This script is intentionally safe for the local development host. By default it
exports the Isaac Lab grid contract and deterministic placeholder alignment
samples without touching robot hardware. Pass ``--collect-samples`` when running
through the Isaac Lab launcher to collect live simulation observations.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


TASK_ID = "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0"
OBS_DIM = 260
OUTPUT_DIM = 12
HEIGHT_SCAN_DIM = 187
GRID_SIZE = (1.6, 1.0)
GRID_RESOLUTION = 0.1
GRID_ORDERING = "xy"
HEIGHT_SCAN_SLICE = (66, 253)
_SIMULATION_APP = None
OBSERVATION_SLICES = {
    "base_lin_vel": [0, 3],
    "base_ang_vel": [3, 6],
    "projected_gravity": [6, 9],
    "velocity_commands": [9, 12],
    "joint_pos": [12, 30],
    "joint_vel": [30, 48],
    "actions": [48, 66],
    "height_scan": [66, 253],
    "arm_joint_command": [253, 259],
    "gripper_command": [259, 260],
}


def _repo_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    go2_root = script_path.parents[2]
    workspace_root = go2_root.parent
    gx_real_root = workspace_root / "gx-real"
    return go2_root, workspace_root, gx_real_root


def isaac_grid_xy(
    *,
    size: tuple[float, float] = GRID_SIZE,
    resolution: float = GRID_RESOLUTION,
    ordering: str = GRID_ORDERING,
) -> np.ndarray:
    """Reproduce Isaac Lab's GridPatternCfg flattening for a yaw-aligned RayCaster.

    Isaac Lab creates x/y coordinates with ``torch.arange(-size/2, size/2,
    resolution)`` including the end point via a small epsilon, then flattens the
    meshgrid in row-major order.
    """

    if ordering not in {"xy", "yx"}:
        raise ValueError(f"unsupported grid ordering: {ordering}")
    x = np.arange(-size[0] / 2.0, size[0] / 2.0 + 1.0e-9, resolution, dtype=np.float64)
    y = np.arange(-size[1] / 2.0, size[1] / 2.0 + 1.0e-9, resolution, dtype=np.float64)
    indexing = "xy" if ordering == "xy" else "ij"
    grid_x, grid_y = np.meshgrid(x, y, indexing=indexing)
    grid_xy = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1))).astype(np.float32)
    if grid_xy.shape != (HEIGHT_SCAN_DIM, 2):
        raise RuntimeError(f"expected grid shape {(HEIGHT_SCAN_DIM, 2)}, got {grid_xy.shape}")
    return grid_xy


def build_contract(grid_xy: np.ndarray, *, source: str) -> dict[str, Any]:
    generated_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    return {
        "version": 1,
        "task": TASK_ID,
        "generated_at_utc": generated_at,
        "generated_by": Path(__file__).name,
        "grid_source": source,
        "obs_dim": OBS_DIM,
        "output_dim": OUTPUT_DIM,
        "height_scan_dim": HEIGHT_SCAN_DIM,
        "observation_slices": OBSERVATION_SLICES,
        "height_scan": {
            "dim": HEIGHT_SCAN_DIM,
            "resolution": GRID_RESOLUTION,
            "size": list(GRID_SIZE),
            "grid_shape": [17, 11],
            "clip": [-1.0, 1.0],
            "scale": 1.0,
            "offset": 0.5,
            "ray_alignment": "yaw",
            "frame": "base_yaw_aligned",
            "flatten_order": "exported_from_isaac_lab",
            "grid_ordering": GRID_ORDERING,
            "grid_xy_source": "height_scan_contract.npz:grid_xy",
        },
        "policy": {
            "input_dim": OBS_DIM,
            "output_dim": OUTPUT_DIM,
            "output": "12D leg action only",
        },
    }


def make_static_samples(num_samples: int) -> dict[str, np.ndarray]:
    if num_samples <= 0:
        raise ValueError("--static-samples must be positive")
    sample_obs = np.zeros((num_samples, OBS_DIM), dtype=np.float32)
    sample_height_scan = sample_obs[:, HEIGHT_SCAN_SLICE[0] : HEIGHT_SCAN_SLICE[1]].copy()
    return {
        "sample_obs": sample_obs,
        "sample_height_scan": sample_height_scan,
        "sample_source": np.array(["static_contract_placeholder"]),
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _policy_obs_from_reset_or_step(obs: Any) -> Any:
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def collect_simulation_samples(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Collect DogOnly-Rough observation samples from Isaac Lab.

    This function imports Isaac/Omniverse modules lazily so the static export path
    remains usable with plain Python.
    """

    from isaaclab.app import AppLauncher

    global _SIMULATION_APP
    app_launcher = AppLauncher({"headless": args.headless, "device": args.device, "enable_cameras": False})
    _SIMULATION_APP = app_launcher.app

    import gymnasium as gym
    import torch

    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env = None
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )
        env_cfg.export_io_descriptors = False
        env_cfg.log_dir = str(Path(args.gx_real_root).resolve() / "logs" / "height_scan_contract_export")
        if getattr(env_cfg.observations, "policy", None) is not None:
            env_cfg.observations.policy.enable_corruption = False
        env = gym.make(args.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        obs, _ = env.reset(seed=args.seed)
        obs_policy = _policy_obs_from_reset_or_step(obs)
        action_dim = env.action_space.shape[-1]

        obs_batches = []
        ray_hit_batches = []
        sensor_pos_batches = []
        base_pose_batches = []
        for step_index in range(max(args.steps, 1)):
            obs_np = _as_numpy(obs_policy)
            obs_batches.append(obs_np)

            sensor = unwrapped.scene.sensors.get("height_scanner")
            if sensor is not None:
                ray_hit_batches.append(_as_numpy(sensor.data.ray_hits_w))
                sensor_pos_batches.append(_as_numpy(sensor.data.pos_w))
            robot = unwrapped.scene["robot"]
            if hasattr(robot.data, "root_pos_w") and hasattr(robot.data, "root_quat_w"):
                base_pose_batches.append(
                    np.concatenate(
                        [_as_numpy(robot.data.root_pos_w), _as_numpy(robot.data.root_quat_w)],
                        axis=-1,
                    )
                )

            if step_index < args.steps - 1:
                with torch.inference_mode():
                    actions = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)
                    obs, _, _, _, _ = env.step(actions)
                    obs_policy = _policy_obs_from_reset_or_step(obs)

        sample_obs = np.concatenate(obs_batches, axis=0)[: args.samples].astype(np.float32)
        sample_height_scan = sample_obs[:, HEIGHT_SCAN_SLICE[0] : HEIGHT_SCAN_SLICE[1]].copy()
        samples = {
            "sample_obs": sample_obs,
            "sample_height_scan": sample_height_scan,
            "sample_source": np.array(["isaac_lab_runtime"]),
        }
        if ray_hit_batches:
            samples["sample_ray_hits_w"] = np.concatenate(ray_hit_batches, axis=0)[: args.samples].astype(np.float32)
        if sensor_pos_batches:
            samples["sample_sensor_pos_w"] = np.concatenate(sensor_pos_batches, axis=0)[: args.samples].astype(np.float32)
        if base_pose_batches:
            samples["sample_robot_base_pose"] = np.concatenate(base_pose_batches, axis=0)[: args.samples].astype(np.float32)
        return samples
    except Exception:
        raise


def write_outputs(
    *,
    contract_yaml: Path,
    contract_npz: Path,
    samples_npz: Path,
    contract: dict[str, Any],
    grid_xy: np.ndarray,
    samples: dict[str, np.ndarray],
) -> None:
    contract_yaml.parent.mkdir(parents=True, exist_ok=True)
    contract_npz.parent.mkdir(parents=True, exist_ok=True)
    samples_npz.parent.mkdir(parents=True, exist_ok=True)

    with contract_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(contract, handle, sort_keys=False)

    np.savez_compressed(contract_npz, grid_xy=grid_xy.astype(np.float32), **samples)
    np.savez_compressed(samples_npz, grid_xy=grid_xy.astype(np.float32), **samples)


def validate_outputs(contract_npz: Path, samples_npz: Path) -> None:
    contract_data = np.load(contract_npz, allow_pickle=False)
    samples_data = np.load(samples_npz, allow_pickle=False)
    grid_xy = contract_data["grid_xy"]
    sample_obs = samples_data["sample_obs"]
    sample_height_scan = samples_data["sample_height_scan"]
    if grid_xy.shape != (HEIGHT_SCAN_DIM, 2):
        raise RuntimeError(f"grid_xy.shape expected {(HEIGHT_SCAN_DIM, 2)}, got {grid_xy.shape}")
    if sample_obs.shape[-1] != OBS_DIM:
        raise RuntimeError(f"sample_obs.shape[-1] expected {OBS_DIM}, got {sample_obs.shape[-1]}")
    if sample_height_scan.shape[-1] != HEIGHT_SCAN_DIM:
        raise RuntimeError(
            f"sample_height_scan.shape[-1] expected {HEIGHT_SCAN_DIM}, got {sample_height_scan.shape[-1]}"
        )
    if not np.allclose(sample_obs[:, HEIGHT_SCAN_SLICE[0] : HEIGHT_SCAN_SLICE[1]], sample_height_scan, atol=1.0e-5):
        raise RuntimeError("sample_obs[:, 66:253] does not match sample_height_scan")


def parse_args(default_collect_samples: bool = False) -> argparse.Namespace:
    _, _, default_gx_real_root = _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--gx-real-root", default=str(default_gx_real_root))
    parser.add_argument("--contract-yaml", default="policies/height_scan_contract.yaml")
    parser.add_argument("--contract-npz", default="policies/height_scan_contract.npz")
    parser.add_argument(
        "--samples-npz",
        default="data/height_scan_alignment/sim_alignment_samples.npz",
    )
    parser.add_argument("--static-samples", type=int, default=1)
    parser.add_argument("--collect-samples", action="store_true", default=default_collect_samples)
    parser.add_argument("--allow-static-fallback", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--disable-fabric", action="store_true")
    return parser.parse_args()


def main(default_collect_samples: bool = False) -> None:
    args = parse_args(default_collect_samples=default_collect_samples)
    gx_real_root = Path(args.gx_real_root).expanduser().resolve()
    contract_yaml = gx_real_root / args.contract_yaml
    contract_npz = gx_real_root / args.contract_npz
    samples_npz = gx_real_root / args.samples_npz

    grid_xy = isaac_grid_xy()
    contract = build_contract(grid_xy, source="isaaclab.GridPatternCfg(resolution=0.1,size=[1.6,1.0],ordering=xy)")

    samples: dict[str, np.ndarray]
    if args.collect_samples:
        try:
            samples = collect_simulation_samples(args)
        except Exception:
            if not args.allow_static_fallback:
                raise
            print("[WARN] Isaac Lab runtime sample collection failed; writing static placeholder samples.", file=sys.stderr)
            samples = make_static_samples(args.static_samples)
    else:
        samples = make_static_samples(args.static_samples)

    write_outputs(
        contract_yaml=contract_yaml,
        contract_npz=contract_npz,
        samples_npz=samples_npz,
        contract=contract,
        grid_xy=grid_xy,
        samples=samples,
    )
    validate_outputs(contract_npz, samples_npz)
    print(f"wrote {contract_yaml}")
    print(f"wrote {contract_npz}")
    print(f"wrote {samples_npz}")
    print(f"grid_xy.shape={grid_xy.shape}")
    print(f"sample_obs.shape={samples['sample_obs'].shape}")
    print(f"sample_height_scan.shape={samples['sample_height_scan'].shape}")
    if _SIMULATION_APP is not None:
        _SIMULATION_APP.close()


if __name__ == "__main__":
    main()
