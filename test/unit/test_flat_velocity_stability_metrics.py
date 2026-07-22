"""Unit tests for the Isaac-independent flat velocity benchmark metrics."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_SCRIPTS = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPTS))

from flat_velocity_stability_metrics import MetricThresholds, analyze_samples, build_schedule


def _samples(
    *,
    name: str = "vx_test",
    kind: str = "command",
    command: tuple[float, float, float] = (0.25, 0.0, 0.0),
    measured: tuple[float, float, float] = (0.24, 0.0, 0.0),
    done: bool = False,
    tilt_rad: float = 0.03,
) -> list[dict[str, object]]:
    rows = []
    for index in range(20):
        rows.append(
            {
                "segment_index": 1,
                "segment_name": name,
                "segment_kind": kind,
                "segment_time_s": 0.02 * (index + 1),
                "evaluate": True,
                "cmd_vx": command[0],
                "cmd_vy": command[1],
                "cmd_wz": command[2],
                "measured_vx": measured[0],
                "measured_vy": measured[1],
                "measured_wz": measured[2],
                "base_z": 0.30 + 0.001 * (index % 2),
                "base_roll": tilt_rad,
                "base_pitch": -tilt_rad,
                "action_abs_max": 0.5,
                "done": done and index == 19,
                "termination_terms": ["bad_orientation"] if done and index == 19 else [],
            }
        )
    return rows


def test_quick_and_full_schedules_include_evaluated_stops() -> None:
    quick = build_schedule("quick", settle_s=2.0, hold_s=3.0, stop_s=1.5, repeats=1)
    full = build_schedule("full", settle_s=2.0, hold_s=3.0, stop_s=1.5, repeats=1)

    assert len(quick) == 15
    assert sum(segment.evaluate for segment in quick) == 14
    assert all(segment.kind == "stop" and segment.evaluate for segment in quick[2::2])
    assert len(full) == 41
    assert sum(segment.evaluate for segment in full) == 40


def test_planar_schedule_covers_requested_envelope_without_yaw() -> None:
    schedule = build_schedule("planar", settle_s=2.0, hold_s=3.0, stop_s=1.5, repeats=1)
    commands = [segment for segment in schedule if segment.kind == "command"]

    assert min(segment.vx for segment in commands) == -0.7
    assert max(segment.vx for segment in commands) == 0.7
    assert min(segment.vy for segment in commands) == -0.2
    assert max(segment.vy for segment in commands) == 0.2
    assert all(segment.wz == 0.0 for segment in commands)
    assert len(schedule) == 41
    assert all(segment.kind == "stop" and segment.evaluate for segment in schedule[2::2])


def test_good_tracking_passes_and_reports_transients() -> None:
    summary = analyze_samples(
        _samples(),
        thresholds=MetricThresholds(),
        expected_evaluated_segments=1,
    )
    segment = summary["segments"][0]

    assert summary["passed"] is True
    assert segment["tracking_pass"] is True
    assert segment["stability_pass"] is True
    assert segment["rise_time_vx_s"] == 0.02
    assert segment["settling_time_vx_s"] == 0.02


def test_low_speed_dead_zone_and_cross_axis_drift_fail() -> None:
    dead_zone = analyze_samples(
        _samples(command=(0.10, 0.0, 0.0), measured=(0.01, 0.0, 0.0)),
        thresholds=MetricThresholds(),
        expected_evaluated_segments=1,
    )["segments"][0]
    drift = analyze_samples(
        _samples(measured=(0.24, 0.12, 0.0)),
        thresholds=MetricThresholds(),
        expected_evaluated_segments=1,
    )["segments"][0]

    assert dead_zone["tracking_pass_vx"] is False
    assert "vx_tracking" in dead_zone["failed_checks"]
    assert drift["tracking_pass_vx"] is True
    assert drift["tracking_pass_vy"] is False
    assert "vy_tracking" in drift["failed_checks"]


def test_stop_residual_tilt_reset_and_incomplete_schedule_fail() -> None:
    stop = analyze_samples(
        _samples(name="stop", kind="stop", command=(0.0, 0.0, 0.0), measured=(0.12, 0.0, 0.0)),
        thresholds=MetricThresholds(),
        expected_evaluated_segments=1,
    )["segments"][0]
    unstable_summary = analyze_samples(
        _samples(done=True, tilt_rad=0.40),
        thresholds=MetricThresholds(),
        expected_evaluated_segments=2,
    )
    unstable = unstable_summary["segments"][0]

    assert stop["passed"] is False
    assert "vx_tracking" in stop["failed_checks"]
    assert unstable["tilt_pass"] is False
    assert unstable["no_reset_pass"] is False
    assert unstable["termination_terms"] == ["bad_orientation"]
    assert unstable_summary["complete_schedule"] is False
    assert unstable_summary["missing_segment_count"] == 1
    assert unstable_summary["passed"] is False
