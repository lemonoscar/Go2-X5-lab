"""Unit tests for the Isaac-independent Walk These Ways policy adapter."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WTW_SCRIPTS = REPO_ROOT / "scripts" / "reinforcement_learning" / "walk_these_ways"
EVALUATOR_FILE = WTW_SCRIPTS / "evaluate_go2_x5_flat.py"
sys.path.insert(0, str(WTW_SCRIPTS))

import wtw_policy_adapter as wtw
import wtw_evaluation_metrics as metrics
import export_wtw_checkpoint as exporter


class _DummyAdaptation(torch.nn.Module):
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return torch.stack((history.sum(dim=-1), history[:, -1]), dim=-1)


class _DummyBody(torch.nn.Module):
    def forward(self, body_input: torch.Tensor) -> torch.Tensor:
        values = torch.linspace(-12.0, 12.0, wtw.ACTION_DIM, device=body_input.device)
        return values.unsqueeze(0).expand(body_input.shape[0], -1)


class _ConstantModule(torch.nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(inputs.shape[0], -1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, *, body_path: Path, adaptation_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "format_version": wtw.MANIFEST_FORMAT_VERSION,
                "stage": "r0_actor_continuation",
                "parent_checkpoint": {"path": "ac_weights_last.pt", "sha256": "1" * 64},
                "rsl_checkpoint": {"path": "model_100.pt", "sha256": "2" * 64, "iteration": 100},
                "body": {"path": body_path.name, "sha256": _file_sha256(body_path)},
                "adaptation": {
                    "path": adaptation_path.name,
                    "sha256": _file_sha256(adaptation_path),
                },
                "abi": {
                    "observation_dim": wtw.OBSERVATION_DIM,
                    "history_length": wtw.HISTORY_LENGTH,
                    "history_dim": wtw.OBSERVATION_HISTORY_DIM,
                    "latent_dim": wtw.LATENT_DIM,
                    "action_dim": wtw.ACTION_DIM,
                    "joint_order": list(wtw.WTW_JOINT_NAMES),
                    "default_joint_pos": list(wtw.DEFAULT_JOINT_POS),
                    "action_scales": list(wtw.ACTION_SCALES),
                    "policy_dt_s": wtw.POLICY_DT_S,
                },
                "controller": {
                    "leg_stiffness": wtw.DEFAULT_LEG_STIFFNESS,
                    "leg_damping": wtw.DEFAULT_LEG_DAMPING,
                    "spawn_height_m": wtw.DEFAULT_SPAWN_HEIGHT_M,
                    "gripper_target_m": list(wtw.DEFAULT_GRIPPER_JOINT_POS),
                },
                "git_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )


def test_contract_constants_and_two_state_commands() -> None:
    command = wtw.make_two_state_command(0.3, -0.1, 0.2, batch_size=2)
    stand = wtw.make_two_state_command(0.0, batch_size=2)

    assert len(wtw.WTW_JOINT_NAMES) == wtw.ACTION_DIM
    assert wtw.DEFAULT_JOINT_POS == (
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    )
    assert wtw.ACTION_SCALES == (0.125, 0.25, 0.25) * 4
    assert wtw.GRIPPER_JOINT_NAMES == ("arm_joint7", "arm_joint8")
    assert wtw.DEFAULT_GRIPPER_JOINT_POS == (0.044, 0.044)
    assert command.shape == (2, wtw.COMMAND_DIM)
    torch.testing.assert_close(
        command[0],
        torch.tensor((0.3, -0.1, 0.2, 0.0, 2.5, 0.5, 0.0, 0.0, 0.5, 0.08, 0.0, 0.0, 0.25, 0.4, 0.0)),
    )
    torch.testing.assert_close(command[0], command[1])
    torch.testing.assert_close(
        stand[0],
        torch.tensor(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.4, 0.0)
        ),
    )
    torch.testing.assert_close(stand[0], stand[1])


def test_reset_and_infer_use_zero_history_and_clip_actions() -> None:
    adapter = wtw.WTWPolicyAdapter(_DummyAdaptation(), _DummyBody())
    adapter.reset(num_envs=2)

    action = adapter.infer()

    assert adapter.observation_history.shape == (2, wtw.OBSERVATION_HISTORY_DIM)
    assert torch.count_nonzero(adapter.observation_history) == 0
    assert action.shape == (2, wtw.ACTION_DIM)
    assert action[0, 0].item() == -wtw.ACTION_CLIP
    assert action[0, -1].item() == wtw.ACTION_CLIP
    torch.testing.assert_close(action[0], action[1])
    assert adapter.infer_raw()[0, 0].item() < -wtw.ACTION_CLIP
    assert adapter.infer_raw()[0, -1].item() > wtw.ACTION_CLIP


def test_advance_builds_exact_observation_and_oldest_first_history() -> None:
    adapter = wtw.WTWPolicyAdapter(_DummyAdaptation(), _DummyBody())
    command = wtw.make_walking_command(0.3, -0.1, 0.2)
    q0 = torch.tensor(wtw.DEFAULT_JOINT_POS)
    q_offset = torch.arange(wtw.ACTION_DIM, dtype=torch.float32) * 0.01
    joint_vel = torch.arange(wtw.ACTION_DIM, dtype=torch.float32)
    action_1 = torch.arange(-6, 6, dtype=torch.float32)

    observation_1 = adapter.advance(
        projected_gravity=(0.1, 0.2, -0.9),
        command=command,
        joint_pos=q0 + q_offset,
        joint_vel=joint_vel,
        applied_action=action_1,
    )

    expected_clock = torch.sin(
        2.0
        * torch.pi
        * torch.tensor(
            (
                0.05 + 0.5,
                0.05,
                0.05,
                0.05 + 0.5,
            )
        )
    )
    torch.testing.assert_close(observation_1[0, 0:3], torch.tensor((0.1, 0.2, -0.9)))
    torch.testing.assert_close(observation_1[0, 3:18], command[0] * torch.tensor(wtw.COMMAND_SCALES))
    torch.testing.assert_close(observation_1[0, 18:30], q_offset)
    torch.testing.assert_close(observation_1[0, 30:42], joint_vel * 0.05)
    torch.testing.assert_close(observation_1[0, 42:54], action_1)
    torch.testing.assert_close(observation_1[0, 54:66], torch.zeros(wtw.ACTION_DIM))
    torch.testing.assert_close(observation_1[0, 66:70], expected_clock)
    assert torch.count_nonzero(adapter.observation_history[:, : -wtw.OBSERVATION_DIM]) == 0
    torch.testing.assert_close(adapter.observation_history[:, -wtw.OBSERVATION_DIM :], observation_1)

    action_2 = action_1 + 1.0
    observation_2 = adapter.advance(
        projected_gravity=(0.1, 0.2, -0.9),
        command=command,
        joint_pos=q0,
        joint_vel=torch.zeros(wtw.ACTION_DIM),
        applied_action=action_2,
    )

    torch.testing.assert_close(observation_2[0, 54:66], action_1)
    torch.testing.assert_close(
        adapter.observation_history[:, -2 * wtw.OBSERVATION_DIM : -wtw.OBSERVATION_DIM],
        observation_1,
    )
    torch.testing.assert_close(adapter.observation_history[:, -wtw.OBSERVATION_DIM :], observation_2)
    torch.testing.assert_close(adapter.gait_index, torch.tensor((0.1,)))


def test_advance_clips_observation_and_rejects_invalid_inputs() -> None:
    adapter = wtw.WTWPolicyAdapter(_DummyAdaptation(), _DummyBody())
    command = wtw.make_walking_command(0.0)
    command[0, 0] = 1000.0

    observation = adapter.advance(
        projected_gravity=(0.0, 0.0, -1.0),
        command=command,
        joint_pos=wtw.DEFAULT_JOINT_POS,
        joint_vel=torch.zeros(wtw.ACTION_DIM),
        applied_action=torch.zeros(wtw.ACTION_DIM),
    )
    assert observation[0, 3].item() == wtw.OBSERVATION_CLIP

    with pytest.raises(ValueError, match="joint_pos must have shape"):
        adapter.advance(
            projected_gravity=(0.0, 0.0, -1.0),
            command=wtw.make_walking_command(0.0),
            joint_pos=torch.zeros(11),
            joint_vel=torch.zeros(wtw.ACTION_DIM),
            applied_action=torch.zeros(wtw.ACTION_DIM),
        )
    bad_gravity = torch.tensor((float("nan"), 0.0, -1.0))
    with pytest.raises(ValueError, match="projected_gravity contains non-finite"):
        adapter.advance(
            projected_gravity=bad_gravity,
            command=wtw.make_walking_command(0.0),
            joint_pos=wtw.DEFAULT_JOINT_POS,
            joint_vel=torch.zeros(wtw.ACTION_DIM),
            applied_action=torch.zeros(wtw.ACTION_DIM),
        )


def test_injected_modules_must_match_exported_tensor_contract() -> None:
    class BadAdaptation(torch.nn.Module):
        def forward(self, history: torch.Tensor) -> torch.Tensor:
            return torch.zeros(history.shape[0], 3)

    with pytest.raises(ValueError, match=r"adaptation output must have shape \(N, 2\)"):
        wtw.WTWPolicyAdapter(BadAdaptation(), _DummyBody())


def test_from_jit_paths_checks_hashes_and_zero_history_golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adaptation_path = tmp_path / "adaptation.jit"
    body_path = tmp_path / "body.jit"
    adaptation = torch.jit.trace(
        _ConstantModule(wtw.ZERO_HISTORY_LATENT),
        torch.zeros(1, wtw.OBSERVATION_HISTORY_DIM),
    )
    body = torch.jit.trace(
        _ConstantModule(wtw.ZERO_HISTORY_ACTION),
        torch.zeros(1, wtw.OBSERVATION_HISTORY_DIM + wtw.LATENT_DIM),
    )
    torch.jit.save(adaptation, str(adaptation_path))
    torch.jit.save(body, str(body_path))
    monkeypatch.setattr(wtw, "KNOWN_ADAPTATION_SHA256", _file_sha256(adaptation_path))
    monkeypatch.setattr(wtw, "KNOWN_BODY_SHA256", _file_sha256(body_path))

    adapter = wtw.WTWPolicyAdapter.from_jit_paths(
        body_path=body_path,
        adaptation_path=adaptation_path,
    )
    torch.testing.assert_close(adapter.infer(), torch.tensor(wtw.ZERO_HISTORY_ACTION).unsqueeze(0))

    monkeypatch.setattr(wtw, "ZERO_HISTORY_ACTION", (0.0,) * wtw.ACTION_DIM)
    with pytest.raises(ValueError, match="zero-history golden check"):
        wtw.WTWPolicyAdapter.from_jit_paths(
            body_path=body_path,
            adaptation_path=adaptation_path,
        )

    monkeypatch.setattr(wtw, "KNOWN_BODY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        wtw.WTWPolicyAdapter.from_jit_paths(
            body_path=body_path,
            adaptation_path=adaptation_path,
        )


def test_manifest_binds_finetuned_jits_and_validates_full_abi(tmp_path: Path) -> None:
    adaptation_path = tmp_path / "adaptation.jit"
    body_path = tmp_path / "body.jit"
    manifest_path = tmp_path / "manifest.json"
    torch.jit.save(
        torch.jit.trace(_ConstantModule((0.0, 0.0)), torch.zeros(1, wtw.OBSERVATION_HISTORY_DIM)),
        str(adaptation_path),
    )
    torch.jit.save(
        torch.jit.trace(_ConstantModule((0.0,) * wtw.ACTION_DIM), torch.zeros(1, wtw.OBSERVATION_HISTORY_DIM + 2)),
        str(body_path),
    )
    _write_manifest(manifest_path, body_path=body_path, adaptation_path=adaptation_path)

    adapter = wtw.WTWPolicyAdapter.from_jit_paths(
        body_path=body_path,
        adaptation_path=adaptation_path,
        manifest_path=manifest_path,
    )
    assert adapter.manifest is not None
    assert adapter.manifest["stage"] == "r0_actor_continuation"
    assert adapter.manifest["controller"]["gripper_target_m"] == [0.044, 0.044]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["controller"]["gripper_target_m"] = [0.0, 0.0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="controller.gripper_target_m mismatch"):
        wtw.WTWPolicyAdapter.from_jit_paths(
            body_path=body_path,
            adaptation_path=adaptation_path,
            manifest_path=manifest_path,
        )

    manifest["controller"]["gripper_target_m"] = list(wtw.DEFAULT_GRIPPER_JOINT_POS)
    manifest["abi"]["joint_order"][0] = "wrong_joint"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="joint_order mismatch"):
        wtw.WTWPolicyAdapter.from_jit_paths(
            body_path=body_path,
            adaptation_path=adaptation_path,
            manifest_path=manifest_path,
        )


def test_walking_profile_excludes_source_deadzone_and_cycle_metrics_are_exact() -> None:
    for _, vx, vy, _ in metrics.WALKING_COMMANDS:
        if vx != 0.0 or vy != 0.0:
            assert math.hypot(vx, vy) >= 0.25
            assert metrics.source_planar_deadzone(vx, vy) == (vx, vy)
    assert metrics.source_planar_deadzone(0.16, 0.08) == (0.0, 0.0)

    rows = []
    for index in range(83):
        phase = 2.0 * math.pi * metrics.GAIT_FREQUENCY_HZ * wtw.POLICY_DT_S * index
        rows.append(
            {
                "cmd_vx": 0.5,
                "cmd_vy": 0.0,
                "cmd_wz": 0.0,
                "measured_vx": 0.5,
                "measured_vy": 0.0,
                "measured_wz": 0.12 * math.sin(phase),
                "leg_torque_saturated": [False] * wtw.ACTION_DIM,
                "raw_action_clipped": [False] * wtw.ACTION_DIM,
                "arm_joint_error": [0.0] * 6,
            }
        )

    result = metrics.gait_cycle_metrics(rows, sample_dt_s=wtw.POLICY_DT_S, steady_fraction=1.0)
    assert result["samples_per_cycle"] == 20
    assert result["cycle_count"] == 4
    assert result["sample_count"] == 80
    assert result["axes"]["vx"]["cycle_mean"] == pytest.approx(0.5)
    assert result["axes"]["wz"]["harmonic_amplitude"] == pytest.approx(0.12, abs=1.0e-7)
    assert result["torque_saturation_rate_max"] == 0.0
    assert result["raw_action_clip_rate"] == 0.0


def test_export_rsl_checkpoint_creates_manifest_bound_jit_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adaptation = exporter._make_mlp(wtw.OBSERVATION_HISTORY_DIM, (256, 128), wtw.LATENT_DIM)
    body = exporter._make_mlp(
        wtw.OBSERVATION_HISTORY_DIM + wtw.LATENT_DIM,
        (512, 256, 128),
        wtw.ACTION_DIM,
    )
    model_state = {
        **{f"adaptation_module.{key}": value for key, value in adaptation.state_dict().items()},
        **{f"actor_body.{key}": value for key, value in body.state_dict().items()},
        "std": torch.full((wtw.ACTION_DIM,), 0.2),
    }
    checkpoint_path = tmp_path / "model_7.pt"
    parent_path = tmp_path / "ac_weights_last.pt"
    output_dir = tmp_path / "export"
    torch.save({"model_state_dict": model_state, "iter": 7}, checkpoint_path)
    parent_path.write_bytes(b"trusted-parent-checkpoint")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_wtw_checkpoint.py",
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--stage",
            "r0_actor_continuation",
            "--parent-checkpoint",
            str(parent_path),
            "--expected-parent-sha256",
            _file_sha256(parent_path),
            "--git-commit",
            "b" * 40,
        ],
    )

    exporter.main()

    manifest_path = output_dir / "manifest.json"
    adapter = wtw.WTWPolicyAdapter.from_jit_paths(
        body_path=output_dir / "body.jit",
        adaptation_path=output_dir / "adaptation_module.jit",
        manifest_path=manifest_path,
    )
    assert adapter.manifest["rsl_checkpoint"]["iteration"] == 7
    assert adapter.manifest["controller"]["gripper_target_m"] == [0.044, 0.044]
    history = torch.randn(3, wtw.OBSERVATION_HISTORY_DIM)
    with torch.inference_mode():
        expected_latent = adaptation(history)
        expected_action = body(torch.cat((history, expected_latent), dim=-1))
        actual_latent, actual_action = adapter._run_modules(history)
    torch.testing.assert_close(actual_latent, expected_latent)
    torch.testing.assert_close(actual_action, expected_action)


def test_walking_settle_primes_history_with_target_command_instead_of_zero() -> None:
    """The non-scored takeover must not feed a contradictory zero command to the actor."""

    tree = ast.parse(EVALUATOR_FILE.read_text(encoding="utf-8"))
    settle_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_segment"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "settle_segment"
    ]
    assert len(settle_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in settle_calls[0].keywords}
    for keyword in ("policy_command", "next_policy_command"):
        value = keywords[keyword]
        assert isinstance(value, ast.Name)
        assert value.id == "command_policy_command"


def test_walking_acceptance_uses_cycle_means_and_ignores_stop_gate() -> None:
    samples = []
    for segment_index, kind, command in ((1, "command", 0.5), (2, "stop", 0.0)):
        for _ in range(40):
            samples.append(
                {
                    "segment_index": segment_index,
                    "cmd_vx": command,
                    "cmd_vy": 0.0,
                    "cmd_wz": 0.0,
                    "measured_vx": command,
                    "measured_vy": 0.0,
                    "measured_wz": 0.0,
                    "leg_torque_saturated": [False] * wtw.ACTION_DIM,
                    "raw_action_clipped": [False] * wtw.ACTION_DIM,
                    "arm_joint_error": [0.0] * 6,
                }
            )
    summary = {
        "passed": False,
        "command_segments": 1,
        "criteria": {
            "gain_min": 0.85,
            "gain_max": 1.15,
            "relative_rmse_limit": 0.15,
            "linear_absolute_floor": 0.05,
            "yaw_absolute_floor": 0.08,
            "zero_linear_rmse_limit": 0.05,
            "zero_yaw_rmse_limit": 0.08,
        },
        "segments": [
            {"segment_index": 1, "segment_kind": "command", "stability_pass": True},
            {"segment_index": 2, "segment_kind": "stop", "stability_pass": False},
        ],
    }
    result = metrics.augment_summary_with_wtw_metrics(
        samples,
        copy.deepcopy(summary),
        sample_dt_s=wtw.POLICY_DT_S,
        steady_fraction=1.0,
        max_wz_harmonic_amplitude=0.15,
        max_torque_saturation_rate=0.01,
        max_action_clip_rate=0.001,
        expected_command_segments=1,
    )
    assert result["walking_only"]["passed"] is True
    assert result["passed"] is True

    missing_command = metrics.augment_summary_with_wtw_metrics(
        samples,
        copy.deepcopy(summary),
        sample_dt_s=wtw.POLICY_DT_S,
        steady_fraction=1.0,
        max_wz_harmonic_amplitude=0.15,
        max_torque_saturation_rate=0.01,
        max_action_clip_rate=0.001,
        expected_command_segments=2,
    )
    assert missing_command["walking_only"]["passed"] is False


def test_stationary_metrics_distinguish_standing_from_stepping() -> None:
    feet = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    rows = []
    for index in range(100):
        arm_command = [0.3 * index / 99.0, 0.3, 0.5, 0.0, 0.0, 0.0]
        rows.append(
            {
                "segment_time_s": (index + 1) * wtw.POLICY_DT_S,
                "state_is_post_auto_reset": False,
                "done": False,
                "measured_vx": 0.01,
                "measured_vy": 0.0,
                "measured_wz": 0.01,
                "base_x": 0.0001 * index,
                "base_y": 0.0,
                "base_z": 0.30,
                "base_roll": 0.01,
                "base_pitch": 0.01,
                "foot_contact_force_n": {name: 50.0 for name in feet},
                "foot_contact_slip_mps": {name: 0.01 for name in feet},
                "leg_joint_vel": [0.1] * wtw.ACTION_DIM,
                "nonfoot_contact_bodies": [],
                "arm_command": arm_command,
                "arm_joint_pos": [arm_command[0] - 0.01, *arm_command[1:]],
                "arm_tracking_max_abs_rad": 0.01,
            }
        )

    standing = metrics.stationary_segment_metrics(rows, arm_motion_required=True)
    assert standing["passed"] is True
    assert standing["base_stable"] is True
    assert standing["feet_static"] is True

    stepping_rows = copy.deepcopy(rows)
    for index, row in enumerate(stepping_rows):
        row["base_x"] = 0.002 * index
        row["leg_joint_vel"] = [0.8] * wtw.ACTION_DIM
        if index % 2:
            row["foot_contact_force_n"]["FL_foot"] = 0.0
    stepping = metrics.stationary_segment_metrics(stepping_rows)
    assert stepping["passed"] is False
    assert stepping["base_stable"] is False
    assert stepping["feet_static"] is False
    assert "foot_liftoff" in stepping["feet_failed_checks"]
