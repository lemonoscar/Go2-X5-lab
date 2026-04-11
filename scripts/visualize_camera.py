#!/usr/bin/env python3
"""Camera visualization script for Go2-X5 robot.

This script:
1. Loads the Go2-X5 ground pick environment
2. Visualizes camera positions in 3D
3. Shows what each camera sees
4. Allows interactive camera parameter adjustment
"""

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from isaaclab.app import AppLauncher
    from isaaclab.sim import SimulationContext
except ImportError:
    print("IsaacLab not found. Please install IsaacLab first.")
    exit(1)

# Add robot_lab to path
import sys
robot_lab_path = Path(__file__).parent.parent / "source" / "robot_lab"
sys.path.insert(0, str(robot_lab_path))

from robot_lab.tasks.manager_based.manipulation.ground_pick.go2_x5_ground_pick_env_cfg import (
    Go2X5GroundPickEnvCfg_PLAY,
    Go2X5GroundPickSceneCfg,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Go2-X5 cameras")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--dog_cam_pos", type=float, nargs=3, default=[0.30, 0.0, 0.16],
                        help="Dog camera position relative to base (x,y,z)")
    parser.add_argument("--dog_cam_rot", type=float, nargs=4, default=[-0.3799, 0.5963, 0.5963, -0.3799],
                        help="Dog camera rotation quaternion (x,y,z,w)")
    parser.add_argument("--arm_cam_pos", type=float, nargs=3, default=[0.08657, 0.0, 0.0],
                        help="Arm camera position relative to arm_link6 (x,y,z)")
    parser.add_argument("--arm_cam_rot", type=float, nargs=4, default=[0.5, -0.5, 0.5, -0.5],
                        help="Arm camera rotation quaternion (x,y,z,w)")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    parser.add_argument("--save_images", action="store_true", help="Save camera images to disk")
    parser.add_argument("--output_dir", type=str, default="camera_output", help="Output directory for images")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def create_custom_scene_config(args):
    """Create scene config with custom camera positions."""

    class CustomSceneCfg(Go2X5GroundPickSceneCfg):
        def __post_init__(self):
            super().__post_init__()

            # Override dog camera
            self.dog_camera.offset.pos = tuple(args.dog_cam_pos)
            self.dog_camera.offset.rot = tuple(args.dog_cam_rot)
            print(f"\n[Dog Camera]")
            print(f"  Position: {self.dog_camera.offset.pos}")
            print(f"  Rotation (quat): {self.dog_camera.offset.rot}")

            # Override arm camera
            self.arm_camera.offset.pos = tuple(args.arm_cam_pos)
            self.arm_camera.offset.rot = tuple(args.arm_cam_rot)
            print(f"\n[Arm Camera]")
            print(f"  Position: {self.arm_camera.offset.pos}")
            print(f"  Rotation (quat): {self.arm_camera.offset.rot}")

    return CustomSceneCfg(num_envs=args.num_envs, env_spacing=4.0, replicate_physics=False)


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to euler angles (roll, pitch, yaw) in degrees."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def get_camera_world_transform(sim, prim_path):
    """Get the world transform of a camera prim."""
    from pxr import Usd, UsdGeom, Gf

    stage = sim.stage
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        return None, None

    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Extract position
    pos = world_transform.ExtractTranslation()

    # Extract rotation as quaternion
    rot = world_transform.ExtractRotationQuaternion()

    return (pos[0], pos[1], pos[2]), (rot.Real, rot.Imaginary[0], rot.Imaginary[1], rot.Imaginary[2])


def print_camera_info(args):
    """Print detailed camera information."""
    print("=" * 60)
    print("CAMERA CONFIGURATION")
    print("=" * 60)

    print("\n[Dog Camera - Ego View]")
    print(f"  Parent: Robot/base")
    print(f"  Local Position: ({args.dog_cam_pos[0]:.4f}, {args.dog_cam_pos[1]:.4f}, {args.dog_cam_pos[2]:.4f})")
    roll, pitch, yaw = quaternion_to_euler(*args.dog_cam_rot)
    print(f"  Local Rotation (quat): ({args.dog_cam_rot[0]:.4f}, {args.dog_cam_rot[1]:.4f}, {args.dog_cam_rot[2]:.4f}, {args.dog_cam_rot[3]:.4f})")
    print(f"  Local Rotation (euler): roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
    print(f"  Resolution: 224x224")
    print(f"  FOV: ~60° (focal_length=24.0)")

    print("\n[Arm Camera - Wrist View]")
    print(f"  Parent: Robot/arm_link6")
    print(f"  Local Position: ({args.arm_cam_pos[0]:.4f}, {args.arm_cam_pos[1]:.4f}, {args.arm_cam_pos[2]:.4f})")
    roll, pitch, yaw = quaternion_to_euler(*args.arm_cam_rot)
    print(f"  Local Rotation (quat): ({args.arm_cam_rot[0]:.4f}, {args.arm_cam_rot[1]:.4f}, {args.arm_cam_rot[2]:.4f}, {args.arm_cam_rot[3]:.4f})")
    print(f"  Local Rotation (euler): roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
    print(f"  Resolution: 224x224")
    print(f"  FOV: ~70° (focal_length=18.0)")

    # Reference info from URDF
    print("\n[Reference - Gripper Positions (from URDF)]")
    print(f"  End effector center: (0.08657, 0.0, 0.0) relative to arm_link6")
    print(f"  Left gripper finger: (0.08657,  0.0249, 0.0) relative to arm_link6")
    print(f"  Right gripper finger: (0.08657, -0.0249, 0.0) relative to arm_link6")

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    # Print camera info before launching
    print_camera_info(args)

    # Launch Isaac app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.respawn

    # Create simulation
    sim = SimulationContext(sim_params={"dt": 0.005})
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.4])

    # Create custom scene config
    scene_cfg = create_custom_scene_config(args)

    # Spawn scene
    from isaaclab.sim import spawn_scene
    spawn_scene(scene_cfg, sim)

    # Simulate a few steps
    sim.step()
    sim.step()
    sim.step()
    sim.step()

    # Get camera world transforms
    print("\n" + "=" * 60)
    print("WORLD COORDINATES (after spawning)")
    print("=" * 60)

    env_ns = "/World/envs/env_0"
    dog_cam_path = f"{env_ns}/Robot/base/dog_vla_camera"
    arm_cam_path = f"{env_ns}/Robot/arm_link6/arm_vla_camera"

    dog_pos, dog_rot = get_camera_world_transform(sim, dog_cam_path)
    if dog_pos:
        print(f"\nDog Camera World Position: ({dog_pos[0]:.3f}, {dog_pos[1]:.3f}, {dog_pos[2]:.3f})")
        roll, pitch, yaw = quaternion_to_euler(*dog_rot)
        print(f"Dog Camera World Rotation: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")

    arm_pos, arm_rot = get_camera_world_transform(sim, arm_cam_path)
    if arm_pos:
        print(f"\nArm Camera World Position: ({arm_pos[0]:.3f}, {arm_pos[1]:.3f}, {arm_pos[2]:.3f})")
        roll, pitch, yaw = quaternion_to_euler(*arm_rot)
        print(f"Arm Camera World Rotation: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")

    # Get camera images
    print("\n" + "=" * 60)
    print("CAPTURING CAMERA IMAGES")
    print("=" * 60)

    from isaaclab.sensors import Camera

    # Create camera sensors
    dog_camera = Camera(scene_cfg.dog_camera)
    arm_camera = Camera(scene_cfg.arm_camera)

    # Create sensors
    dog_camera.create("/World/envs/env_0/Robot/base", "/World/envs/env_0/Robot/base/dog_vla_camera")
    arm_camera.create("/World/envs/env_0/Robot/arm_link6", "/World/envs/env_0/Robot/arm_link6/arm_vla_camera")

    sim.step()

    # Get images
    dog_data = dog_camera.data
    arm_data = arm_camera.data

    print(f"\nDog Camera Output Shape: {dog_data.output.shape if hasattr(dog_data, 'output') else 'N/A'}")
    print(f"Arm Camera Output Shape: {arm_data.output.shape if hasattr(arm_data, 'output') else 'N/A'}")

    # Save images if requested
    if args.save_images:
        import cv2
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Get RGB images
        if hasattr(dog_data, 'output') and 'rgb' in dog_data.output:
            dog_rgb = dog_data.output['rgb'][0].cpu().numpy()  # Remove batch dim
            dog_rgb = (dog_rgb * 255).astype(np.uint8)
            dog_bgr = cv2.cvtColor(dog_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / "dog_camera.png"), dog_bgr)
            print(f"Saved: {output_dir / 'dog_camera.png'}")

        if hasattr(arm_data, 'output') and 'rgb' in arm_data.output:
            arm_rgb = arm_data.output['rgb'][0].cpu().numpy()
            arm_rgb = (arm_rgb * 255).astype(np.uint8)
            arm_bgr = cv2.cvtColor(arm_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / "arm_camera.png"), arm_bgr)
            print(f"Saved: {output_dir / 'arm_camera.png'}")

    print("\n" + "=" * 60)
    print("VISUALIZATION TIPS")
    print("=" * 60)
    print("\nIn the Isaac Sim window:")
    print("  1. Use mouse to rotate/pan/zoom the view")
    print("  2. Look for the camera prims (small pyramids)")
    print("  3. The camera Z-axis points in the viewing direction")
    print("\nTo adjust camera positions, use command line args:")
    print("  --dog_cam_pos 0.30 0.0 0.16")
    print("  --dog_cam_rot -0.3799 0.5963 0.5963 -0.3799")
    print("  --arm_cam_pos 0.08657 0.0 0.0")
    print("  --arm_cam_rot 0 0 0 1  # Try this for forward-looking!")

    if args.no_render:
        print("\nRunning in headless mode. Remove --no_render to see visualization.")
    else:
        print("\nKeeping visualization open. Press Ctrl+C to exit.")
        try:
            while True:
                sim.step()
        except KeyboardInterrupt:
            print("\nExiting...")

    simulation_app.close()


if __name__ == "__main__":
    main()
