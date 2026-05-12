# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the Go2-X5 tabletop prototype environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "robot_lab"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the Go2-X5 tabletop prototype env.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Go2-X5-Tabletop-Reach-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--keep_running", action="store_true", help="Keep stepping zero actions until interrupted.")
parser.add_argument("--step_sleep", type=float, default=0.0, help="Seconds to sleep after each step.")
parser.add_argument("--pause_after_reset", action="store_true", help="Pause in GUI mode after reset for visual inspection.")
parser.add_argument("--save_rgb", action="store_true", help="Save student.rgb images after reset and after settling.")
parser.add_argument("--rgb_out_dir", type=str, default="docs/media/tabletop_play")
parser.add_argument("--physics_validate", action="store_true", help="Run tabletop physics sanity checks.")
parser.add_argument("--physics_validate_steps", type=int, default=500)
parser.add_argument("--asset_mode", choices=["primitive", "gr00t", "auto"], default="auto")
parser.add_argument("--visual_preset", choices=["train", "play"], default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.visual_preset is None:
    args_cli.visual_preset = "play" if args_cli.task.endswith("-Play-v0") else "train"
os.environ["GO2_X5_TABLETOP_ASSET_MODE"] = args_cli.asset_mode
os.environ["GO2_X5_TABLETOP_VISUAL_PRESET"] = args_cli.visual_preset

if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Usd, UsdGeom

import robot_lab.tasks  # noqa: F401


APPROX_TABLETOP_Z = 0.674
APPROX_TRAY_TOP_Z = 0.691


def _nested_shapes(value):
    if isinstance(value, dict):
        return {key: _nested_shapes(item) for key, item in value.items()}
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def _shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def _prim_path(env, name: str, env_index: int = 0) -> str:
    cfg = _asset_cfg(env, name)
    if cfg is not None and hasattr(cfg, "prim_path"):
        return _actual_prim_path(getattr(cfg, "prim_path"), env_index)
    try:
        asset = env.unwrapped.scene[name]
    except Exception:
        return "<missing>"
    cfg = getattr(asset, "cfg", None)
    return _actual_prim_path(getattr(cfg, "prim_path", getattr(asset, "prim_path", "<unknown>")), env_index)


def _asset_cfg(env, name: str):
    try:
        asset = env.unwrapped.scene[name]
        cfg = getattr(asset, "cfg", None)
        if cfg is not None:
            return cfg
    except Exception:
        pass
    scene_cfg = getattr(env.unwrapped.scene, "cfg", None)
    return getattr(scene_cfg, name, None)


def _actual_prim_path(path, env_index: int = 0) -> str:
    text = str(path)
    env_path = f"/World/envs/env_{env_index}"
    text = text.replace("{ENV_REGEX_NS}", env_path)
    text = text.replace("/World/envs/env_.*/", f"{env_path}/")
    return text


def _as_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _env_vector(value, env_index: int = 0):
    if value is None:
        return None
    if hasattr(value, "ndim"):
        if value.ndim == 3:
            return value[env_index, 0]
        if value.ndim == 2:
            return value[env_index]
        if value.ndim == 1:
            return value
    try:
        return value[env_index]
    except Exception:
        return value


def _world_pose(env, name: str, env_index: int = 0):
    try:
        asset = env.unwrapped.scene[name]
    except Exception:
        asset = None

    data = getattr(asset, "data", None) if asset is not None else None
    if data is not None and hasattr(data, "root_pos_w"):
        return {
            "pos": _as_list(_env_vector(data.root_pos_w, env_index)),
            "quat": _as_list(_env_vector(getattr(data, "root_quat_w", None), env_index)),
        }
    if data is not None and hasattr(data, "body_pos_w"):
        return {
            "pos": _as_list(_env_vector(data.body_pos_w, env_index)),
            "quat": _as_list(_env_vector(getattr(data, "body_quat_w", None), env_index)),
        }

    prim_path = _prim_path(env, name, env_index)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path) if stage is not None and prim_path not in ("<missing>", "<unknown>") else None
    if prim is not None and prim.IsValid():
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        return {"pos": [float(translation[0]), float(translation[1]), float(translation[2])]}

    cfg = _asset_cfg(env, name)
    init_state = getattr(cfg, "init_state", None)
    if init_state is not None:
        return {
            "configured_pos": _as_list(getattr(init_state, "pos", None)),
            "configured_rot": _as_list(getattr(init_state, "rot", None)),
        }
    return "<unavailable>"


def _world_bbox(env, name: str, env_index: int = 0):
    prim_path = _prim_path(env, name, env_index)
    stage = omni.usd.get_context().get_stage()
    if stage is None or prim_path in ("<missing>", "<unknown>"):
        return None
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bbox_min = aligned_box.GetMin()
        bbox_max = aligned_box.GetMax()
        return {
            "min": [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])],
            "max": [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])],
        }
    except Exception as exc:
        return {"error": type(exc).__name__}


def _make_zero_actions(env):
    action_shape = env.action_space.shape
    if len(action_shape) == 1:
        action_shape = (env.unwrapped.num_envs, action_shape[0])
    return torch.zeros(action_shape, device=env.unwrapped.device)


def _step(env, actions):
    obs, reward, terminated, truncated, info = env.step(actions)
    if not args_cli.headless:
        env.unwrapped.sim.render()
    if args_cli.step_sleep > 0.0:
        time.sleep(args_cli.step_sleep)
    return obs, reward, terminated, truncated, info


def _print_scene_report(env, obs, actions, reward, terminated, truncated, info):
    num_envs = env.unwrapped.num_envs
    last_env = max(0, num_envs - 1)
    print(f"task id: {args_cli.task}", flush=True)
    print(f"env class: {type(env.unwrapped).__name__}", flush=True)
    print(f"headless: {args_cli.headless}", flush=True)
    print(f"num envs: {num_envs}", flush=True)
    print(f"observation keys/shapes: {_nested_shapes(obs)}", flush=True)
    print(f"action space shape: {env.action_space.shape}", flush=True)
    print(f"single action dimension: {actions.shape[-1]}", flush=True)
    print(f"zero action shape: {tuple(actions.shape)}", flush=True)
    print(f"reward shape: {_shape(reward)}", flush=True)
    print(f"terminated shape: {_shape(terminated)}", flush=True)
    print(f"truncated shape: {_shape(truncated)}", flush=True)
    print(f"env origins: {_env_origin_report(env)}", flush=True)
    for env_index in sorted({0, last_env}):
        print(f"env_{env_index} prim paths:", flush=True)
        for name in ("robot", "table", "tray", "object", "target_marker"):
            print(f"  {name}: {_prim_path(env, name, env_index)}", flush=True)
            print(f"  {name} pose: {_world_pose(env, name, env_index)}", flush=True)
    print(f"asset mode: {args_cli.asset_mode}", flush=True)
    print(f"visual preset: {args_cli.visual_preset}", flush=True)
    print(f"info keys: {sorted(info.keys()) if isinstance(info, dict) else type(info).__name__}", flush=True)
    sys.stdout.flush()


def _env_origin_report(env):
    origins = getattr(env.unwrapped.scene, "env_origins", None)
    if origins is None:
        return "<unavailable>"
    origins_cpu = origins.detach().cpu()
    mins = origins_cpu.min(dim=0).values.tolist()
    maxs = origins_cpu.max(dim=0).values.tolist()
    first = origins_cpu[0].tolist()
    last = origins_cpu[-1].tolist()
    return {"first": first, "last": last, "min": mins, "max": maxs}


def _student_rgb(obs):
    if not isinstance(obs, dict):
        return None
    student = obs.get("student")
    if not isinstance(student, dict):
        return None
    return student.get("rgb")


def _write_rgb(rgb, label: str):
    if rgb is None:
        print(f"rgb save {label}: skipped; rgb tensor not found", flush=True)
        return None

    out_dir = Path(args_cli.rgb_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array = rgb[0].detach().cpu()
    if torch.is_floating_point(array):
        array = array.clamp(0.0, 1.0).mul(255.0).byte()
    else:
        array = array.byte()
    if array.shape[-1] == 4:
        array = array[..., :3]
    image = array.numpy()
    png_path = out_dir / f"{label}.png"
    try:
        from PIL import Image

        Image.fromarray(image).save(png_path)
        print(f"saved rgb {label}: {png_path}", flush=True)
        return png_path
    except Exception as exc:
        ppm_path = out_dir / f"{label}.ppm"
        with ppm_path.open("wb") as file:
            file.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
            file.write(image.tobytes())
        print(f"saved rgb {label}: {ppm_path} (PIL unavailable: {type(exc).__name__})", flush=True)
        return ppm_path


def _write_rgb_grid(rgb, label: str):
    if rgb is None or not hasattr(rgb, "shape") or rgb.shape[0] <= 1:
        return None

    out_dir = Path(args_cli.rgb_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = rgb[: min(16, rgb.shape[0])].detach().cpu()
    if torch.is_floating_point(images):
        images = images.clamp(0.0, 1.0).mul(255.0).byte()
    else:
        images = images.byte()
    if images.shape[-1] == 4:
        images = images[..., :3]

    count, height, width, channels = images.shape
    cols = 4
    rows = (count + cols - 1) // cols
    canvas = torch.zeros((rows * height, cols * width, channels), dtype=torch.uint8)
    for index in range(count):
        row = index // cols
        col = index % cols
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = images[index]

    png_path = out_dir / f"{label}_env_grid.png"
    try:
        from PIL import Image

        Image.fromarray(canvas.numpy()).save(png_path)
        print(f"saved rgb {label} env grid: {png_path}", flush=True)
        return png_path
    except Exception as exc:
        ppm_path = out_dir / f"{label}_env_grid.ppm"
        image = canvas.numpy()
        with ppm_path.open("wb") as file:
            file.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
            file.write(image.tobytes())
        print(f"saved rgb {label} env grid: {ppm_path} (PIL unavailable: {type(exc).__name__})", flush=True)
        return ppm_path


def _save_rgb(obs, label: str):
    rgb = _student_rgb(obs)
    saved = _write_rgb(rgb, label)
    _write_rgb_grid(rgb, label)
    return saved


def _configure_viewer_camera(env):
    if args_cli.headless:
        return
    origins = getattr(env.unwrapped.scene, "env_origins", None)
    if origins is None:
        eye = [14.0, -14.0, 10.0] if env.unwrapped.num_envs >= 16 else [2.4, 2.0, 1.6]
        target = [0.78, 0.0, 0.72]
    else:
        origins_cpu = origins.detach().cpu()
        center = origins_cpu.mean(dim=0)
        span = torch.max(origins_cpu.max(dim=0).values - origins_cpu.min(dim=0).values).item()
        target = [float(center[0] + 0.78), float(center[1]), 0.72]
        if env.unwrapped.num_envs >= 16:
            distance = max(10.0, span * 0.95 + 4.0)
            eye = [target[0] + distance, target[1] - distance, max(8.0, span * 0.55 + 5.0)]
        else:
            eye = [target[0] + 1.8, target[1] + 1.6, 1.5]
    try:
        env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        print(f"viewer camera eye: {eye}", flush=True)
        print(f"viewer camera target: {target}", flush=True)
    except Exception as exc:
        print(f"viewer camera setup skipped: {type(exc).__name__}: {exc}", flush=True)


def _configure_overview_camera(env):
    try:
        camera = env.unwrapped.scene["overview_camera"]
    except Exception:
        return False
    device = env.unwrapped.device
    origins = getattr(env.unwrapped.scene, "env_origins", None)
    if origins is None:
        origins = torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=device)
    else:
        origins = origins.to(device)
    eyes = origins + torch.tensor([1.72, -1.18, 1.32], dtype=torch.float32, device=device)
    targets = origins + torch.tensor([0.74, 0.00, 0.70], dtype=torch.float32, device=device)
    camera.set_world_poses_from_view(eyes, targets)
    return True


def _save_overview_rgb(env, label: str):
    try:
        camera = env.unwrapped.scene["overview_camera"]
    except Exception:
        return None
    rgb = camera.data.output.get("rgb")
    saved = _write_rgb(rgb, f"overview_{label}")
    _write_rgb_grid(rgb, f"overview_{label}")
    return saved


def _has_nonfinite(value) -> bool:
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if torch.is_tensor(value):
        return not torch.isfinite(value).all().item()
    return False


def _pose_pos(pose):
    if isinstance(pose, dict):
        return pose.get("pos") or pose.get("configured_pos")
    return None


def _support_top_z(env):
    z_values = []
    for name in ("table", "tray"):
        bbox = _world_bbox(env, name)
        if isinstance(bbox, dict) and "max" in bbox:
            z_values.append(bbox["max"][2])
    if z_values:
        return max(z_values), "usd_bbox"
    return APPROX_TRAY_TOP_Z, "fallback_constant"


def _object_velocity(env, name: str = "object"):
    try:
        obj = env.unwrapped.scene[name]
    except Exception:
        return None, None
    data = getattr(obj, "data", None)
    if data is None:
        return None, None
    lin_vel = getattr(data, "root_lin_vel_w", getattr(data, "root_lin_vel_b", None))
    ang_vel = getattr(data, "root_ang_vel_w", getattr(data, "root_ang_vel_b", None))
    return lin_vel, ang_vel


def _print_validation_check(name: str, passed: bool, detail: str):
    status = "PASS" if passed else "FAIL"
    print(f"physics validation {name}: {status} - {detail}", flush=True)
    return passed


def _run_physics_validation(env, obs, actions, reward, terminated, truncated, initial_table_pose):
    support_z, support_source = _support_top_z(env)
    object_bbox = _world_bbox(env, "object")
    object_pose = _world_pose(env, "object")
    table_pose = _world_pose(env, "table")
    table_pos_initial = _pose_pos(initial_table_pose)
    table_pos_final = _pose_pos(table_pose)
    lin_vel, ang_vel = _object_velocity(env)
    lin_norm = torch.norm(lin_vel, dim=-1).max().item() if lin_vel is not None else 0.0
    ang_norm = torch.norm(ang_vel, dim=-1).max().item() if ang_vel is not None else 0.0

    if isinstance(object_bbox, dict) and "min" in object_bbox:
        object_bottom_z = object_bbox["min"][2]
        object_height_detail = f"object bbox min z={object_bottom_z:.4f}, support top z={support_z:.4f} ({support_source})"
    else:
        object_bottom_z = _pose_pos(object_pose)[2] if _pose_pos(object_pose) is not None else -1.0
        object_height_detail = f"object root z={object_bottom_z:.4f}, support top z={support_z:.4f} ({support_source})"

    table_delta = 0.0
    if table_pos_initial is not None and table_pos_final is not None:
        table_delta = sum((float(a) - float(b)) ** 2 for a, b in zip(table_pos_initial, table_pos_final)) ** 0.5

    all_object_z_detail = "object root positions unavailable"
    all_objects_above_support = True
    try:
        obj = env.unwrapped.scene["object"]
        object_z = obj.data.root_pos_w[:, 2]
        all_objects_above_support = bool(torch.all(object_z > support_z + 0.005).item())
        all_object_z_detail = (
            f"min root z={object_z.min().item():.4f}, max root z={object_z.max().item():.4f}, "
            f"support top z={support_z:.4f} ({support_source})"
        )
    except Exception:
        pass

    checks = [
        _print_validation_check("finite_observations", not _has_nonfinite(obs), "observations contain only finite values"),
        _print_validation_check(
            "finite_rewards",
            reward is not None and torch.is_tensor(reward) and torch.isfinite(reward).all().item(),
            f"reward shape={_shape(reward)}",
        ),
        _print_validation_check("action_dim", actions.shape[-1] == 10, f"action dim={actions.shape[-1]}"),
        _print_validation_check("reward_shape", _shape(reward) == (env.unwrapped.num_envs,), f"reward shape={_shape(reward)}"),
        _print_validation_check(
            "terminated_shape",
            _shape(terminated) == (env.unwrapped.num_envs,),
            f"terminated shape={_shape(terminated)}",
        ),
        _print_validation_check(
            "truncated_shape",
            _shape(truncated) == (env.unwrapped.num_envs,),
            f"truncated shape={_shape(truncated)}",
        ),
        _print_validation_check(
            "object_above_support",
            object_bottom_z > support_z - 0.035,
            object_height_detail,
        ),
        _print_validation_check(
            "objects_above_support_all_envs",
            all_objects_above_support,
            all_object_z_detail,
        ),
        _print_validation_check(
            "object_velocity_bounded",
            lin_norm < 2.0 and ang_norm < 20.0,
            f"max linear={lin_norm:.4f} m/s max angular={ang_norm:.4f} rad/s",
        ),
        _print_validation_check("table_fixed", table_delta < 0.005, f"table position delta={table_delta:.6f} m"),
    ]
    print(f"physics validation overall: {'PASS' if all(checks) else 'FAIL'}", flush=True)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.export_io_descriptors = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        actions = _make_zero_actions(env)
        initial_table_pose = _world_pose(env, "table")
        _configure_viewer_camera(env)
        has_overview_camera = _configure_overview_camera(env)

        if args_cli.save_rgb:
            _save_rgb(obs, "reset")
            if has_overview_camera:
                _save_overview_rgb(env, "reset")

        if args_cli.pause_after_reset and not args_cli.headless:
            input("Paused after reset for visual inspection. Press Enter to step zero actions...")

        reward = terminated = truncated = info = None
        steps = max(args_cli.steps, args_cli.physics_validate_steps if args_cli.physics_validate else args_cli.steps)
        for _ in range(steps):
            obs, reward, terminated, truncated, info = _step(env, actions)
        if args_cli.save_rgb:
            _save_rgb(obs, "settled")
            if has_overview_camera:
                _save_overview_rgb(env, "settled")
        _print_scene_report(env, obs, actions, reward, terminated, truncated, info)
        if args_cli.physics_validate:
            _run_physics_validation(env, obs, actions, reward, terminated, truncated, initial_table_pose)

        if args_cli.keep_running:
            print("keep_running enabled. Stepping zero actions until interrupted.")
            try:
                while simulation_app.is_running():
                    obs, reward, terminated, truncated, info = _step(env, actions)
            except KeyboardInterrupt:
                print("Interrupted; closing environment.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
