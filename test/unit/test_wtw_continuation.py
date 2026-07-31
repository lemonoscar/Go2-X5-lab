"""Pure Torch and static configuration tests for WTW continuation."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WTW_MDP_FILE = (
    REPO_ROOT
    / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/wtw_continuation.py"
)
LOCOMOTION_ACTIONS_FILE = WTW_MDP_FILE.with_name("actions.py")
WTW_CFG_FILE = (
    REPO_ROOT
    / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/wtw_flat_env_cfg.py"
)
GO2_X5_INIT_FILE = WTW_CFG_FILE.parent / "__init__.py"


def _load_wtw_mdp_without_isaac():
    isaaclab_module = types.ModuleType("isaaclab")
    envs_module = types.ModuleType("isaaclab.envs")
    mdp_module = types.ModuleType("isaaclab.envs.mdp")
    managers_module = types.ModuleType("isaaclab.managers")
    utils_module = types.ModuleType("isaaclab.utils")

    class _UniformVelocityCommand:
        pass

    class _UniformVelocityCommandCfg:
        pass

    class _SceneEntityCfg:
        def __init__(self, name: str):
            self.name = name

    mdp_module.UniformVelocityCommand = _UniformVelocityCommand
    mdp_module.UniformVelocityCommandCfg = _UniformVelocityCommandCfg
    managers_module.SceneEntityCfg = _SceneEntityCfg
    utils_module.configclass = lambda cls: cls

    isaaclab_module.envs = envs_module
    isaaclab_module.managers = managers_module
    isaaclab_module.utils = utils_module
    envs_module.mdp = mdp_module
    sys.modules.setdefault("isaaclab", isaaclab_module)
    sys.modules.setdefault("isaaclab.envs", envs_module)
    sys.modules.setdefault("isaaclab.envs.mdp", mdp_module)
    sys.modules.setdefault("isaaclab.managers", managers_module)
    sys.modules.setdefault("isaaclab.utils", utils_module)

    spec = importlib.util.spec_from_file_location("wtw_continuation_unit", WTW_MDP_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wtw = _load_wtw_mdp_without_isaac()


def _load_locomotion_actions_without_isaac():
    """Load the zero-dimensional command action with minimal Isaac stubs."""

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_assets = types.ModuleType("isaaclab.assets")
    fake_articulation = types.ModuleType("isaaclab.assets.articulation")
    fake_managers = types.ModuleType("isaaclab.managers")
    fake_action_manager = types.ModuleType("isaaclab.managers.action_manager")
    fake_utils = types.ModuleType("isaaclab.utils")

    class _Articulation:
        pass

    class _ActionTermCfg:
        pass

    class _ActionTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device
            self._asset = env.scene[cfg.asset_name]

    fake_articulation.Articulation = _Articulation
    fake_managers.ActionTermCfg = _ActionTermCfg
    fake_action_manager.ActionTerm = _ActionTerm
    fake_utils.configclass = lambda cls: cls
    fake_isaaclab.assets = fake_assets
    fake_isaaclab.managers = fake_managers
    fake_isaaclab.utils = fake_utils

    replacements = {
        "isaaclab": fake_isaaclab,
        "isaaclab.assets": fake_assets,
        "isaaclab.assets.articulation": fake_articulation,
        "isaaclab.managers": fake_managers,
        "isaaclab.managers.action_manager": fake_action_manager,
        "isaaclab.utils": fake_utils,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "wtw_locomotion_actions_unit", LOCOMOTION_ACTIONS_FILE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_walking_command_spec_rejects_standing_and_hidden_low_speed_mixed_bins() -> None:
    with pytest.raises(ValueError, match="non-zero walking values"):
        wtw.validate_wtw_walking_command_spec(
            (-0.25, 0.0, 0.25),
            (-0.25, 0.25),
            (-0.3, 0.3),
            (0.45, 0.20, 0.20, 0.15),
        )
    with pytest.raises(ValueError, match="sum to 1.0"):
        wtw.validate_wtw_walking_command_spec(
            (-0.25, 0.25),
            (-0.25, 0.25),
            (-0.3, 0.3),
            (0.40, 0.20, 0.20, 0.15),
        )
    with pytest.raises(ValueError, match="mixed planar speed"):
        wtw.validate_wtw_walking_command_spec(
            (-0.10, 0.10),
            (-0.10, 0.10),
            (-0.3, 0.3),
            (0.45, 0.20, 0.20, 0.15),
        )


def test_walking_sampler_preserves_exact_bins_probabilities_and_no_standing() -> None:
    torch.manual_seed(31)
    count = 30000
    vx_values = torch.tensor((-0.75, -0.50, -0.25, 0.25, 0.50, 0.75))
    vy_values = torch.tensor((-0.40, -0.25, 0.25, 0.40))
    yaw_values = torch.tensor((-0.50, -0.30, 0.30, 0.50))
    probabilities = torch.tensor((0.45, 0.20, 0.20, 0.15))

    commands, modes = wtw.sample_wtw_walking_commands(
        count, vx_values, vy_values, yaw_values, probabilities.cumsum(dim=0)
    )

    assert not torch.any(torch.all(commands == 0.0, dim=1))
    assert torch.all(commands[modes == wtw.WTW_MODE_PURE_VX, 1:] == 0.0)
    assert torch.all(commands[modes == wtw.WTW_MODE_PURE_VY][:, (0, 2)] == 0.0)
    assert torch.all(commands[modes == wtw.WTW_MODE_PURE_YAW, :2] == 0.0)
    mixed_commands = commands[modes == wtw.WTW_MODE_MIXED]
    assert torch.all(mixed_commands != 0.0)
    assert torch.all(torch.linalg.norm(mixed_commands[:, :2], dim=1) >= 0.25)
    for mode, expected in enumerate(probabilities):
        assert torch.mean((modes == mode).float()).item() == pytest.approx(expected.item(), abs=0.012)


def test_wtw_frame_is_exact_and_reset_row_is_zero() -> None:
    q0 = torch.tensor(wtw.WTW_DEFAULT_JOINT_POS).repeat(2, 1)
    joint_offset = torch.arange(wtw.WTW_ACTION_DIM, dtype=torch.float32) * 0.01
    q0[1] += joint_offset
    joint_vel = torch.arange(wtw.WTW_ACTION_DIM, dtype=torch.float32).repeat(2, 1)
    current_action = torch.arange(-6, 6, dtype=torch.float32).repeat(2, 1)
    previous_action = -current_action

    frame = wtw.build_wtw_observation_frame(
        projected_gravity=torch.tensor(((0.0, 0.0, -1.0), (0.1, 0.2, -0.9))),
        base_velocity_command=torch.tensor(((0.25, 0.0, 0.0), (0.50, -0.25, 0.30))),
        joint_pos=q0,
        joint_vel=joint_vel,
        current_action=current_action,
        previous_action=previous_action,
        episode_steps=torch.tensor((0, 1)),
        step_dt=0.02,
    )

    assert frame.shape == (2, wtw.WTW_FRAME_DIM)
    assert torch.count_nonzero(frame[0]) == 0
    command = wtw.build_wtw_command(torch.tensor(((0.50, -0.25, 0.30),)))
    torch.testing.assert_close(frame[1, 0:3], torch.tensor((0.1, 0.2, -0.9)))
    torch.testing.assert_close(
        frame[1, 3:18], command[0] * torch.tensor(wtw.WTW_COMMAND_SCALES)
    )
    torch.testing.assert_close(frame[1, 18:30], joint_offset)
    torch.testing.assert_close(frame[1, 30:42], joint_vel[1] * 0.05)
    torch.testing.assert_close(frame[1, 42:54], current_action[1])
    torch.testing.assert_close(frame[1, 54:66], previous_action[1])
    expected_clock = torch.sin(2.0 * torch.pi * torch.tensor((0.55, 0.05, 0.05, 0.55)))
    torch.testing.assert_close(frame[1, 66:70], expected_clock)


def test_wtw_frame_rejects_wrong_policy_period_and_clips_raw_actions() -> None:
    inputs = {
        "projected_gravity": torch.tensor(((0.0, 0.0, -1.0),)),
        "base_velocity_command": torch.tensor(((0.25, 0.0, 0.0),)),
        "joint_pos": torch.tensor((wtw.WTW_DEFAULT_JOINT_POS,)),
        "joint_vel": torch.zeros(1, wtw.WTW_ACTION_DIM),
        "current_action": torch.full((1, wtw.WTW_ACTION_DIM), 100.0),
        "previous_action": torch.full((1, wtw.WTW_ACTION_DIM), -100.0),
        "episode_steps": torch.ones(1, dtype=torch.long),
    }
    frame = wtw.build_wtw_observation_frame(**inputs, step_dt=0.02)
    assert torch.all(frame[:, 42:54] == wtw.WTW_ACTION_CLIP)
    assert torch.all(frame[:, 54:66] == -wtw.WTW_ACTION_CLIP)

    with pytest.raises(ValueError, match="0.02 s policy step"):
        wtw.build_wtw_observation_frame(**inputs, step_dt=0.01)


def test_external_gripper_action_consumes_zero_policy_dims_and_tracks_two_joint_command() -> None:
    actions = _load_locomotion_actions_without_isaac()

    class _Asset:
        def __init__(self) -> None:
            self.last_target = None
            self.last_joint_ids = None

        def find_joints(self, joint_names, preserve_order: bool):
            assert preserve_order is True
            assert tuple(joint_names) == wtw.WTW_GRIPPER_JOINT_NAMES
            return [7, 8], list(joint_names)

        def set_joint_position_target(self, target, *, joint_ids) -> None:
            self.last_target = target.clone()
            self.last_joint_ids = list(joint_ids)

    target = torch.tensor(((0.044, 0.044), (0.044, 0.044), (0.044, 0.044)))
    asset = _Asset()
    command_term = types.SimpleNamespace(command=target)
    env = types.SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": asset},
        command_manager=types.SimpleNamespace(
            get_term=lambda name: command_term if name == "gripper_joint_pos" else None
        ),
    )
    cfg = types.SimpleNamespace(
        asset_name="robot",
        joint_names=list(wtw.WTW_GRIPPER_JOINT_NAMES),
        command_name="gripper_joint_pos",
        preserve_order=True,
    )

    action = actions.ArmCommandPositionAction(cfg, env)
    assert action.action_dim == 0
    assert action.raw_actions.shape == (3, 0)
    assert action.processed_actions.shape == (3, 0)
    action.process_actions(torch.empty(3, 0))
    action.apply_actions()
    torch.testing.assert_close(asset.last_target, target)
    assert asset.last_joint_ids == [7, 8]
    with pytest.raises(ValueError, match="expects zero-dim policy actions"):
        action.process_actions(torch.zeros(3, 1))


def test_wtw_task_static_contract_and_registration_are_isolated() -> None:
    cfg_source = WTW_CFG_FILE.read_text()
    registration_source = GO2_X5_INIT_FILE.read_text()

    assert "class Go2X5WTWFlatEnvCfg(Go2X5DogOnlyFlatEnvCfg)" in cfg_source
    assert "expected_policy_observation_dim: int = mdp.WTW_FRAME_DIM * mdp.WTW_HISTORY_LENGTH" in cfg_source
    assert "expected_critic_observation_dim: int = 260" in cfg_source
    assert wtw.WTW_GRIPPER_JOINT_NAMES == ("arm_joint7", "arm_joint8")
    assert wtw.WTW_GRIPPER_DEFAULT_JOINT_POS == (0.044, 0.044)
    assert "self.actions.joint_pos.joint_names = list(mdp.WTW_JOINT_NAMES)" in cfg_source
    assert "self.scene.contact_forces.history_length = self.decimation" in cfg_source
    assert "self.decimation = 8" in cfg_source
    assert 'self.scene.terrain.terrain_type = "generator"' in cfg_source
    assert "self.scene.terrain.terrain_generator = FLAT_FOUNDATION_TERRAIN_CFG.copy()" in cfg_source
    assert "self.scene.terrain.use_terrain_origins = False" in cfg_source
    assert "self.scene.terrain.visual_material = None" in cfg_source
    assert "self.scene.sky_light.spawn.texture_file = None" in cfg_source
    assert 'self.scene.terrain.terrain_type = "plane"' not in cfg_source
    assert "self.commands.base_velocity = mdp.WTWWalkingVelocityCommandCfg(" in cfg_source
    assert "self.commands.arm_joint_pos.position_range = ARM_LOCKED_DEFAULT_RANGE" in cfg_source
    assert "self.commands.gripper_joint_pos = mdp.ArmJointPositionCommandCfg(" in cfg_source
    assert "joint_names=list(mdp.WTW_GRIPPER_JOINT_NAMES)" in cfg_source
    assert "position_range=[(0.0, 0.0)] * len(mdp.WTW_GRIPPER_JOINT_NAMES)" in cfg_source
    assert "use_default_offset=True" in cfg_source
    assert "self.actions.gripper_joint_pos = mdp.ArmCommandPositionActionCfg(" in cfg_source
    assert 'command_name="gripper_joint_pos"' in cfg_source
    assert "if self.scene.contact_forces.history_length != self.decimation:" in cfg_source
    assert "if self.scene.contact_forces is None:" in cfg_source
    assert "RobotLab-Isaac-Velocity-Flat-Go2-X5-WTW-PD40-v0" in registration_source
    assert "wtw_rsl_rl:Go2X5WtwPD40PPORunnerCfg" in registration_source
