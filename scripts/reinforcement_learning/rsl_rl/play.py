import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_width", type=int, default=2440, help="Width of the recorded video.")
parser.add_argument("--video_height", type=int, default=1560, help="Height of the recorded video.")
parser.add_argument("--video_fps", type=int, default=20, help="FPS of the recorded video. Lower = slower playback.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--base_cmd",
    type=float,
    nargs=3,
    default=None,
    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
    help="Override the base velocity command with fixed values during play.",
)
parser.add_argument(
    "--arm_cmd_pos_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Override arm joint position command range for play.",
)
parser.add_argument(
    "--arm_cmd_resample_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Override arm command resampling time range for play.",
)
parser.add_argument(
    "--arm_cmd_pose",
    type=float,
    nargs="+",
    default=None,
    help="Override arm command with a fixed joint offset pose (len = arm joints).",
)
parser.add_argument(
    "--arm_cmd_pose_set",
    type=float,
    nargs="+",
    action="append",
    default=None,
    help="Provide multiple arm joint poses (repeat per env).",
)
parser.add_argument(
    "--arm_cmd_pose_sequence",
    action="store_true",
    default=False,
    help="Play arm poses sequentially over time using arm_cmd_pose_set or the built-in default pose list.",
)
parser.add_argument(
    "--arm_cmd_pose_repeat",
    type=int,
    default=1,
    help="Repeat each pose across this many envs.",
)
parser.add_argument(
    "--arm_cmd_pose_absolute",
    action="store_true",
    default=False,
    help="Treat arm_cmd_pose/arm_cmd_pose_set values as absolute positions (default: offsets).",
)
parser.add_argument(
    "--arm_cmd_pose_alt",
    type=float,
    nargs="+",
    default=None,
    help="Optional alternate arm pose for back-and-forth motion (len = arm joints).",
)
parser.add_argument(
    "--arm_cmd_pose_period",
    type=float,
    default=2.0,
    help="Seconds per pose when alternating between arm_cmd_pose and arm_cmd_pose_alt.",
)
parser.add_argument(
    "--arm_cmd_sequence_pose_s",
    type=float,
    default=5.0,
    help="Seconds to command each pose when arm_cmd_pose_sequence is enabled.",
)
parser.add_argument(
    "--arm_cmd_sequence_default_s",
    type=float,
    default=5.0,
    help="Seconds to command the default arm pose between consecutive poses in arm_cmd_pose_sequence mode.",
)
parser.add_argument(
    "--arm_cmd_force_action",
    action="store_true",
    default=False,
    help="Force arm action dimensions to match the commanded arm pose exactly instead of letting whole-body RL respond.",
)
parser.add_argument(
    "--arm_cmd_schedule",
    action="store_true",
    default=False,
    help="Schedule arm commands with warmup, slow move, and hold phases.",
)
parser.add_argument(
    "--arm_cmd_warmup_s",
    type=float,
    default=2.5,
    help="Warmup seconds before applying arm commands.",
)
parser.add_argument(
    "--arm_cmd_move_s",
    type=float,
    default=1.5,
    help="Seconds to interpolate arm commands to a new target.",
)
parser.add_argument(
    "--arm_cmd_hold_s",
    type=float,
    default=2.5,
    help="Seconds to hold the target arm command before resampling.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
from rl_utils import ActionDelayWrapper, camera_follow

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp


def _disable_play_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Freeze play-time stochasticity so playback reflects the learned policy instead of domain randomization."""
    env_cfg.observations.policy.enable_corruption = False

    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None

    terrain = getattr(env_cfg.scene, "terrain", None)
    if terrain is not None:
        # Keep terrain assignment deterministic during playback instead of random sampling across the grid.
        terrain.max_init_terrain_level = 0
        if terrain.terrain_generator is not None:
            terrain.terrain_generator.num_rows = min(terrain.terrain_generator.num_rows, 5)
            terrain.terrain_generator.num_cols = min(terrain.terrain_generator.num_cols, 5)
            terrain.terrain_generator.curriculum = False

    if getattr(env_cfg, "events", None) is not None:
        reset_base_event = getattr(env_cfg.events, "randomize_reset_base", None)
        if reset_base_event is not None:
            reset_base_event.params = {
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }

        material_event = getattr(env_cfg.events, "randomize_rigid_body_material", None)
        if material_event is not None:
            material_event.params["static_friction_range"] = (1.0, 1.0)
            material_event.params["dynamic_friction_range"] = (1.0, 1.0)
            material_event.params["restitution_range"] = (0.0, 0.0)

        for term_name in (
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "randomize_push_robot",
            "push_robot",
            "randomize_actuator_gains",
        ):
            if hasattr(env_cfg.events, term_name):
                setattr(env_cfg.events, term_name, None)

    if hasattr(env_cfg, "sim2sim_action_delay_range"):
        env_cfg.sim2sim_action_delay_range = (0, 0)
    if hasattr(env_cfg, "sim2sim_action_hold_prob"):
        env_cfg.sim2sim_action_hold_prob = 0.0
    if hasattr(env_cfg, "sim2sim_action_noise_std"):
        env_cfg.sim2sim_action_noise_std = 0.0
    if hasattr(env_cfg, "sim2sim_obs_delay_steps"):
        env_cfg.sim2sim_obs_delay_steps = 0
        if env_cfg.observations.policy.base_ang_vel is not None:
            env_cfg.observations.policy.base_ang_vel.func = locomotion_mdp.base_ang_vel
            if env_cfg.observations.policy.base_ang_vel.params is not None:
                env_cfg.observations.policy.base_ang_vel.params.pop("delay_steps", None)
        if env_cfg.observations.policy.projected_gravity is not None:
            env_cfg.observations.policy.projected_gravity.func = locomotion_mdp.projected_gravity
            if env_cfg.observations.policy.projected_gravity.params is not None:
                env_cfg.observations.policy.projected_gravity.params.pop("delay_steps", None)
        if env_cfg.observations.policy.joint_pos is not None:
            env_cfg.observations.policy.joint_pos.func = locomotion_mdp.joint_pos_rel
            if env_cfg.observations.policy.joint_pos.params is not None:
                env_cfg.observations.policy.joint_pos.params.pop("delay_steps", None)
        if env_cfg.observations.policy.joint_vel is not None:
            env_cfg.observations.policy.joint_vel.func = locomotion_mdp.joint_vel_rel
            if env_cfg.observations.policy.joint_vel.params is not None:
                env_cfg.observations.policy.joint_vel.params.pop("delay_steps", None)
        if env_cfg.observations.policy.actions is not None:
            action_obs_params = env_cfg.observations.policy.actions.params
            if action_obs_params is not None and "total_action_dim" in action_obs_params:
                env_cfg.observations.policy.actions.func = locomotion_mdp.last_action_with_padding
                action_obs_params.pop("delay_steps", None)
            else:
                env_cfg.observations.policy.actions.func = locomotion_mdp.last_action
                if action_obs_params is not None:
                    action_obs_params.pop("delay_steps", None)
        if env_cfg.observations.policy.velocity_commands is not None:
            env_cfg.observations.policy.velocity_commands.func = locomotion_mdp.generated_commands
            if env_cfg.observations.policy.velocity_commands.params is None:
                env_cfg.observations.policy.velocity_commands.params = {}
            env_cfg.observations.policy.velocity_commands.params["command_name"] = "base_velocity"
            env_cfg.observations.policy.velocity_commands.params.pop("delay_steps", None)
        if getattr(env_cfg.observations.policy, "arm_joint_command", None) is not None:
            env_cfg.observations.policy.arm_joint_command.func = locomotion_mdp.generated_commands
            if env_cfg.observations.policy.arm_joint_command.params is None:
                env_cfg.observations.policy.arm_joint_command.params = {}
            env_cfg.observations.policy.arm_joint_command.params["command_name"] = "arm_joint_pos"
            env_cfg.observations.policy.arm_joint_command.params.pop("delay_steps", None)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    _disable_play_randomization(env_cfg)
    # avoid early episode termination during play/recording
    if getattr(env_cfg, "terminations", None) is not None:
        env_cfg.terminations.terrain_out_of_bounds = None
        if args_cli.video:
            env_cfg.terminations.time_out = None

    if args_cli.keyboard and args_cli.base_cmd is not None:
        raise ValueError("--base_cmd is not compatible with --keyboard.")

    controller = None
    if args_cli.keyboard:
        # Match teleop sensitivity to the task command ranges instead of pushing the policy outside training support.
        lin_vel_x_max = max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_x)
        lin_vel_y_max = max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_y)
        ang_vel_z_max = max(abs(v) for v in env_cfg.commands.base_velocity.ranges.ang_vel_z)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        print(
            "[INFO] Keyboard command limits "
            f"lin_x=+/-{lin_vel_x_max:.3f}, lin_y=+/-{lin_vel_y_max:.3f}, ang_z=+/-{ang_vel_z_max:.3f}"
        )
    fixed_base_cmd_cfg = None
    if args_cli.base_cmd is not None:
        fixed_base_cmd_cfg = tuple(args_cli.base_cmd)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (fixed_base_cmd_cfg[0], fixed_base_cmd_cfg[0])
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (fixed_base_cmd_cfg[1], fixed_base_cmd_cfg[1])
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (fixed_base_cmd_cfg[2], fixed_base_cmd_cfg[2])
        print(
            "[INFO] Play command fixed to "
            f"lin_x={fixed_base_cmd_cfg[0]:.3f}, lin_y={fixed_base_cmd_cfg[1]:.3f}, ang_z={fixed_base_cmd_cfg[2]:.3f}"
        )
    elif not args_cli.keyboard:
        print(
            "[INFO] Play uses the task command distribution "
            f"lin_x={env_cfg.commands.base_velocity.ranges.lin_vel_x}, "
            f"lin_y={env_cfg.commands.base_velocity.ranges.lin_vel_y}, "
            f"ang_z={env_cfg.commands.base_velocity.ranges.ang_vel_z}, "
            f"stand={env_cfg.commands.base_velocity.rel_standing_envs:.2f}"
        )
    print(
        "[INFO] Play randomization disabled: observation corruption, material/mass/CoM DR, "
        "reset velocity perturbations, pushes, actuator gain drift, and sim2sim delays."
    )

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=lin_vel_x_max,
            v_y_sensitivity=lin_vel_y_max,
            omega_z_sensitivity=ang_vel_z_max,
        )
        controller = Se2Keyboard(config)

    if env_cfg.commands.arm_joint_pos is not None:
        # increase arm motion amplitude for play (can be overridden via CLI)
        env_cfg.commands.arm_joint_pos.position_range = (-1.5, 3.2)
        env_cfg.commands.arm_joint_pos.resampling_time_range = (1.0, 2.0)
        if args_cli.arm_cmd_pos_range is not None:
            env_cfg.commands.arm_joint_pos.position_range = tuple(args_cli.arm_cmd_pos_range)
        if args_cli.arm_cmd_resample_range is not None:
            env_cfg.commands.arm_joint_pos.resampling_time_range = tuple(args_cli.arm_cmd_resample_range)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise ModuleNotFoundError(
                "The 'isaaclab.utils.pretrained_checkpoint' module is not available. "
                "Please use --checkpoint instead of --use_pretrained_checkpoint."
            )
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # set viewer resolution for higher quality video recording
    if args_cli.video:
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # Calculate real-time fps based on simulation dt
        step_dt = env.unwrapped.step_dt
        realtime_fps = int(1.0 / step_dt)
        # Use user-specified fps for video playback (lower = slower playback)
        video_fps = args_cli.video_fps
        video_duration = args_cli.video_length / video_fps
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": video_fps,
        }
        print("[INFO] Recording videos during training.")
        print(f"[INFO] Simulation realtime fps: {realtime_fps}, Video fps: {video_fps}")
        print(f"[INFO] Video will be {video_duration:.1f} seconds ({args_cli.video_length} frames at {video_fps} fps)")
        print(f"[INFO] Playback speed: {video_fps / realtime_fps:.2f}x (1.0 = realtime)")
        print(f"[INFO] Resolution: {args_cli.video_width}x{args_cli.video_height}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # sim2sim action timing uncertainty (optional)
    sim2sim_delay = getattr(env_cfg, "sim2sim_action_delay_range", None)
    if sim2sim_delay is not None:
        hold_prob = getattr(env_cfg, "sim2sim_action_hold_prob", 0.0)
        noise_std = getattr(env_cfg, "sim2sim_action_noise_std", 0.0)
        if max(sim2sim_delay) > 0 or hold_prob > 0.0 or noise_std > 0.0:
            env = ActionDelayWrapper(
                env,
                delay_steps_range=tuple(sim2sim_delay),
                hold_prob=hold_prob,
                action_noise_std=noise_std,
            )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    arm_term = None
    arm_pose = None
    arm_pose_alt = None
    arm_pose_steps = None
    arm_schedule = None
    arm_pose_set = None
    arm_pose_set_targets = None
    arm_pose_sequence = None
    default_arm_sequence = None
    arm_action_map = None
    base_cmd_term = None
    fixed_base_cmd = None
    default_pose_set_raw = [
        [-1.66, 2.20, 1.83, 0.09, 0.17, 0.17],
        [-1.66, 2.20, 1.83, 0.09, 0.17, 0.17],
        [0, 1.48, 1.83, 0, 1.57, 0],
        [1.53, 2.7, 2.55, -0.09, 0.0, 0.09],
        [0.0, 2.7, 2.21, -0.09, -0.09, 0.09],
    ]

    def resolve_arm_action_map(current_arm_term):
        action_term = env.unwrapped.action_manager._terms.get("joint_pos", None)
        if action_term is None:
            return None
        name_to_index = {name: i for i, name in enumerate(action_term._joint_names)}
        arm_names = list(current_arm_term.cfg.joint_names)
        arm_action_indices = [name_to_index[name] for name in arm_names if name in name_to_index]
        if len(arm_action_indices) != len(arm_names):
            print("[WARN] Could not resolve all arm joints in action mapping.")
            return None
        return {
            "indices": arm_action_indices,
            "scale": action_term._scale,
            "offset": action_term._offset,
        }

    if args_cli.arm_cmd_schedule and args_cli.arm_cmd_pose is not None:
        raise ValueError("arm_cmd_schedule is not compatible with arm_cmd_pose overrides.")
    if args_cli.arm_cmd_pose_sequence and (args_cli.arm_cmd_pose is not None or args_cli.arm_cmd_pose_alt is not None):
        raise ValueError("arm_cmd_pose_sequence is not compatible with arm_cmd_pose or arm_cmd_pose_alt.")
    if args_cli.arm_cmd_pose_sequence and args_cli.arm_cmd_schedule:
        raise ValueError("arm_cmd_pose_sequence is not compatible with arm_cmd_schedule.")
    if args_cli.arm_cmd_pose_set is not None and (
        args_cli.arm_cmd_pose is not None or args_cli.arm_cmd_schedule
    ):
        raise ValueError("arm_cmd_pose_set is not compatible with arm_cmd_pose or arm_cmd_schedule.")
    if args_cli.arm_cmd_pose is not None:
        arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
        if arm_term is None:
            raise RuntimeError("arm_cmd_pose requested but arm_joint_pos command is not active.")
        if len(args_cli.arm_cmd_pose) != len(arm_term.joint_ids):
            raise ValueError("arm_cmd_pose length must match arm joint count.")
        defaults = arm_term.asset.data.default_joint_pos[:, arm_term.joint_ids]
        pose_values = torch.tensor(args_cli.arm_cmd_pose, device=defaults.device)
        arm_pose = pose_values if args_cli.arm_cmd_pose_absolute else defaults + pose_values
        arm_action_map = resolve_arm_action_map(arm_term)
        if args_cli.arm_cmd_pose_alt is not None:
            if len(args_cli.arm_cmd_pose_alt) != len(arm_term.joint_ids):
                raise ValueError("arm_cmd_pose_alt length must match arm joint count.")
            pose_values_alt = torch.tensor(args_cli.arm_cmd_pose_alt, device=defaults.device)
            arm_pose_alt = pose_values_alt if args_cli.arm_cmd_pose_absolute else defaults + pose_values_alt
            arm_pose_steps = max(1, int(args_cli.arm_cmd_pose_period / dt))
    if args_cli.arm_cmd_schedule:
        arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
        if arm_term is None:
            raise RuntimeError("arm_cmd_schedule requested but arm_joint_pos command is not active.")
        defaults = arm_term.asset.data.default_joint_pos[:, arm_term.joint_ids]
        cmd_range = args_cli.arm_cmd_pos_range or arm_term.cfg.position_range
        warmup_steps = max(0, int(args_cli.arm_cmd_warmup_s / dt))
        move_steps = max(1, int(args_cli.arm_cmd_move_s / dt))
        hold_steps = max(1, int(args_cli.arm_cmd_hold_s / dt))
        schedule_steps = move_steps + hold_steps
        current_offsets = torch.zeros_like(defaults)
        target_offsets = torch.empty_like(defaults)
        def sample_offsets(cmd_range, template):
            if isinstance(cmd_range, (list, tuple)) and len(cmd_range) > 0 and isinstance(cmd_range[0], (list, tuple)):
                min_range = torch.tensor([r[0] for r in cmd_range], device=template.device)
                max_range = torch.tensor([r[1] for r in cmd_range], device=template.device)
                offsets = torch.empty_like(template)
                offsets.uniform_(0.0, 1.0)
                return min_range + offsets * (max_range - min_range)
            return torch.empty_like(template).uniform_(*cmd_range)
        arm_schedule = {
            "defaults": defaults,
            "cmd_range": cmd_range,
            "warmup_steps": warmup_steps,
            "move_steps": move_steps,
            "hold_steps": hold_steps,
            "schedule_steps": schedule_steps,
            "current_offsets": current_offsets,
            "target_offsets": target_offsets,
            "sample_offsets": sample_offsets,
        }
        arm_action_map = resolve_arm_action_map(arm_term)
    if args_cli.arm_cmd_pose_set is not None:
        arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
        if arm_term is None:
            raise RuntimeError("arm_cmd_pose_set requested but arm_joint_pos command is not active.")
        defaults = arm_term.asset.data.default_joint_pos[:, arm_term.joint_ids]
        poses = []
        for pose in args_cli.arm_cmd_pose_set:
            if len(pose) != len(arm_term.joint_ids):
                raise ValueError("arm_cmd_pose_set entry length must match arm joint count.")
            pose_values = torch.tensor(pose, device=defaults.device)
            poses.append(pose_values)
        arm_pose_set = torch.stack(poses, dim=0)
        repeat = max(1, int(args_cli.arm_cmd_pose_repeat))
        if repeat > 1:
            arm_pose_set = arm_pose_set.repeat_interleave(repeat, dim=0)
        env_ids = torch.arange(defaults.shape[0], device=defaults.device)
        pose_indices = env_ids % arm_pose_set.shape[0]
        pose_values = arm_pose_set[pose_indices]
        arm_pose_set_targets = pose_values if args_cli.arm_cmd_pose_absolute else defaults + pose_values
        if arm_term.cfg.clip_to_joint_limits:
            limits = arm_term.asset.data.soft_joint_pos_limits[:, arm_term.joint_ids, :]
            min_pos = limits[..., 0]
            max_pos = limits[..., 1]
            arm_pose_set_targets = torch.max(torch.min(arm_pose_set_targets, max_pos), min_pos)
        arm_action_map = resolve_arm_action_map(arm_term)
    if args_cli.arm_cmd_pose_sequence:
        arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
        if arm_term is None:
            raise RuntimeError("arm_cmd_pose_sequence requested but arm_joint_pos command is not active.")
        defaults = arm_term.asset.data.default_joint_pos[:, arm_term.joint_ids]
        pose_set_raw = args_cli.arm_cmd_pose_set if args_cli.arm_cmd_pose_set is not None else default_pose_set_raw
        poses = []
        for pose in pose_set_raw:
            if len(pose) != len(arm_term.joint_ids):
                raise ValueError("arm_cmd_pose_sequence entry length must match arm joint count.")
            pose_values = torch.tensor(pose, device=defaults.device)
            poses.append(pose_values)
        targets = torch.stack(poses, dim=0)
        if args_cli.arm_cmd_pose_set is not None and not args_cli.arm_cmd_pose_absolute:
            targets = defaults[0].unsqueeze(0) + targets
        if arm_term.cfg.clip_to_joint_limits:
            limits = arm_term.asset.data.soft_joint_pos_limits[:, arm_term.joint_ids, :]
            min_pos = limits[0, :, 0]
            max_pos = limits[0, :, 1]
            targets = torch.max(torch.min(targets, max_pos), min_pos)
        arm_pose_sequence = {
            "defaults": defaults,
            "targets": targets,
            "pose_steps": max(1, int(args_cli.arm_cmd_sequence_pose_s / dt)),
            "default_steps": max(1, int(args_cli.arm_cmd_sequence_default_s / dt)),
            "num_envs": defaults.shape[0],
        }
        arm_action_map = resolve_arm_action_map(arm_term)

    base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity", None)
    if args_cli.keyboard and base_cmd_term is None:
        raise RuntimeError("Keyboard control requires an active base_velocity command term.")
    fixed_base_cmd = None
    if base_cmd_term is not None and fixed_base_cmd_cfg is not None:
        fixed_base_cmd = torch.tensor(
            [
                fixed_base_cmd_cfg[0],
                fixed_base_cmd_cfg[1],
                fixed_base_cmd_cfg[2],
            ],
            dtype=torch.float32,
            device=base_cmd_term.device,
        ).repeat(base_cmd_term.num_envs, 1)

    # default arm sequence: wait 2s, move to extreme pose in ~2s, then hold
    if (
        env_cfg.commands.arm_joint_pos is not None
        and args_cli.arm_cmd_pose is None
        and args_cli.arm_cmd_pose_alt is None
        and not args_cli.arm_cmd_schedule
        and args_cli.arm_cmd_pose_set is None
    ):
        arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
        if arm_term is not None:
            defaults = arm_term.asset.data.default_joint_pos[:, arm_term.joint_ids]
            pose_set_raw = getattr(env_cfg, "arm_pose_set", None)
            if pose_set_raw is None:
                pose_set_raw = default_pose_set_raw
            pose_set = torch.tensor(pose_set_raw, device=defaults.device)
            if pose_set.ndim != 2 or pose_set.shape[1] != len(arm_term.joint_ids):
                raise ValueError(
                    "arm_pose_set must be a list of poses with length equal to arm joint count."
                )
            # repeat each pose across 4 envs
            pose_set = pose_set.repeat_interleave(4, dim=0)
            env_ids = torch.arange(defaults.shape[0], device=defaults.device)
            pose_indices = env_ids % pose_set.shape[0]
            targets = pose_set[pose_indices]
            if arm_term.cfg.clip_to_joint_limits:
                limits = arm_term.asset.data.soft_joint_pos_limits[:, arm_term.joint_ids, :]
                min_pos = limits[..., 0]
                max_pos = limits[..., 1]
                targets = torch.max(torch.min(targets, max_pos), min_pos)
            warmup_steps = max(0, int(4.0 / dt))
            move_steps = max(1, int(4.0 / dt))
            hold_steps = max(1, int(2.0 / dt))
            default_arm_sequence = {
                "defaults": defaults,
                "targets": targets,
                "warmup_steps": warmup_steps,
                "move_steps": move_steps,
                "hold_steps": hold_steps,
            }
            arm_action_map = resolve_arm_action_map(arm_term)
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                desired_arm_pos = None
                desired_base_cmd = fixed_base_cmd
                if args_cli.keyboard and controller is not None and base_cmd_term is not None:
                    desired_base_cmd = torch.tensor(
                        controller.advance(), dtype=torch.float32, device=base_cmd_term.device
                    ).unsqueeze(0).repeat(base_cmd_term.num_envs, 1)
                if arm_schedule is not None:
                    schedule = arm_schedule
                    if timestep < schedule["warmup_steps"]:
                        desired_arm_pos = schedule["defaults"]
                    else:
                        phase = (timestep - schedule["warmup_steps"]) % schedule["schedule_steps"]
                        if phase == 0:
                            schedule["current_offsets"] = schedule["target_offsets"].clone()
                            schedule["target_offsets"] = schedule["sample_offsets"](
                                schedule["cmd_range"], schedule["target_offsets"]
                            )
                        if phase < schedule["move_steps"]:
                            alpha = phase / schedule["move_steps"]
                            offsets = (1.0 - alpha) * schedule["current_offsets"] + alpha * schedule["target_offsets"]
                        else:
                            offsets = schedule["target_offsets"]
                        target_pos = schedule["defaults"] + offsets
                        if arm_term.cfg.clip_to_joint_limits:
                            limits = arm_term.asset.data.soft_joint_pos_limits[:, arm_term.joint_ids, :]
                            min_pos = limits[..., 0]
                            max_pos = limits[..., 1]
                            target_pos = torch.max(torch.min(target_pos, max_pos), min_pos)
                        desired_arm_pos = target_pos
                if arm_pose_sequence is not None:
                    seq = arm_pose_sequence
                    block_steps = seq["pose_steps"] + seq["default_steps"]
                    total_steps = seq["targets"].shape[0] * block_steps
                    if timestep >= total_steps:
                        desired_arm_pos = seq["defaults"]
                    else:
                        pose_idx = timestep // block_steps
                        phase = timestep % block_steps
                        if phase < seq["pose_steps"]:
                            desired_arm_pos = seq["targets"][pose_idx].unsqueeze(0).repeat(seq["num_envs"], 1)
                        else:
                            desired_arm_pos = seq["defaults"]
                if default_arm_sequence is not None:
                    seq = default_arm_sequence
                    if timestep < seq["warmup_steps"]:
                        desired_arm_pos = seq["defaults"]
                    elif timestep < seq["warmup_steps"] + seq["move_steps"]:
                        alpha = (timestep - seq["warmup_steps"]) / seq["move_steps"]
                        target_pos = seq["defaults"] + alpha * (seq["targets"] - seq["defaults"])
                        desired_arm_pos = target_pos
                    else:
                        desired_arm_pos = seq["targets"]
                if arm_pose_set_targets is not None:
                    desired_arm_pos = arm_pose_set_targets
                if arm_term is not None and arm_pose is not None:
                    if arm_pose_alt is not None and arm_pose_steps is not None:
                        select_alt = (timestep // arm_pose_steps) % 2 == 1
                        desired_arm_pos = arm_pose_alt if select_alt else arm_pose
                    else:
                        desired_arm_pos = arm_pose
                if desired_base_cmd is not None and base_cmd_term is not None:
                    base_cmd_term.vel_command_b[:] = desired_base_cmd
                    if hasattr(base_cmd_term, "is_heading_env"):
                        base_cmd_term.is_heading_env[:] = False
                    if hasattr(base_cmd_term, "is_standing_env"):
                        base_cmd_term.is_standing_env[:] = torch.linalg.norm(desired_base_cmd, dim=1) < 1.0e-6
                    if hasattr(base_cmd_term, "heading_target"):
                        base_cmd_term.heading_target[:] = 0.0
                if desired_arm_pos is not None and arm_term is not None:
                    arm_term.command_buffer[:] = desired_arm_pos
                # refresh obs so policy sees the new commands
                obs = env.get_observations()
                # agent stepping
                actions = policy(obs)
                if args_cli.arm_cmd_force_action and desired_arm_pos is not None and arm_action_map is not None:
                    idx = arm_action_map["indices"]
                    scale = arm_action_map["scale"]
                    offset = arm_action_map["offset"]
                    if torch.is_tensor(scale):
                        scale_arm = scale[:, idx]
                    else:
                        scale_arm = scale
                    if torch.is_tensor(offset):
                        offset_arm = offset[:, idx]
                    else:
                        offset_arm = offset
                    raw_arm = (desired_arm_pos - offset_arm) / scale_arm
                    actions[:, idx] = raw_arm
                # actions = torch.zeros_like(actions)
                # env stepping
                obs, _, _, _ = env.step(actions)
            timestep += 1
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    print(f"[INFO] Video recording completed. Saved {args_cli.video_length} frames.")
                    break

            if args_cli.keyboard:
                camera_follow(env)

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"[WARNING] Simulation interrupted: {e}")
    finally:
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
