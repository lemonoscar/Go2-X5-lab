"""Isaac-independent walking-only schedules and gait-cycle metrics for WTW."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean
from typing import Any, Sequence


GAIT_FREQUENCY_HZ = 2.5
WALKING_COMMANDS = (
    ("vx_neg_075", -0.75, 0.0, 0.0),
    ("vx_neg_050", -0.50, 0.0, 0.0),
    ("vx_neg_025", -0.25, 0.0, 0.0),
    ("vx_pos_025", 0.25, 0.0, 0.0),
    ("vx_pos_050", 0.50, 0.0, 0.0),
    ("vx_pos_075", 0.75, 0.0, 0.0),
    ("vy_neg_040", 0.0, -0.40, 0.0),
    ("vy_neg_025", 0.0, -0.25, 0.0),
    ("vy_pos_025", 0.0, 0.25, 0.0),
    ("vy_pos_040", 0.0, 0.40, 0.0),
    ("wz_neg_050", 0.0, 0.0, -0.50),
    ("wz_neg_030", 0.0, 0.0, -0.30),
    ("wz_pos_030", 0.0, 0.0, 0.30),
    ("wz_pos_050", 0.0, 0.0, 0.50),
    ("mix_forward_left", 0.50, 0.25, 0.30),
    ("mix_forward_right", 0.50, -0.25, -0.30),
    ("mix_reverse_left", -0.50, 0.25, -0.30),
    ("mix_reverse_right", -0.50, -0.25, 0.30),
)


def source_planar_deadzone(vx: float, vy: float, *, threshold: float = 0.20) -> tuple[float, float]:
    """Return the source-training planar command after its inclusive norm deadzone."""

    if threshold < 0.0:
        raise ValueError("deadzone threshold must be non-negative")
    return (0.0, 0.0) if math.hypot(vx, vy) <= threshold else (vx, vy)


def integer_cycle_window(
    values: Sequence[Any],
    *,
    sample_dt_s: float,
    gait_frequency_hz: float = GAIT_FREQUENCY_HZ,
    steady_fraction: float = 0.50,
) -> tuple[Sequence[Any], int, int]:
    """Select the latest steady-state samples spanning only complete gait cycles."""

    if sample_dt_s <= 0.0 or gait_frequency_hz <= 0.0:
        raise ValueError("sample_dt_s and gait_frequency_hz must be positive")
    if not 0.0 < steady_fraction <= 1.0:
        raise ValueError("steady_fraction must be in (0, 1]")
    samples_per_cycle_float = 1.0 / (sample_dt_s * gait_frequency_hz)
    samples_per_cycle = round(samples_per_cycle_float)
    if samples_per_cycle <= 0 or not math.isclose(
        samples_per_cycle_float, samples_per_cycle, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("gait period must contain an integer number of control samples")
    steady_count = max(1, math.floor(len(values) * steady_fraction)) if values else 0
    complete_count = (steady_count // samples_per_cycle) * samples_per_cycle
    if complete_count == 0:
        return values[0:0], samples_per_cycle, 0
    return values[-complete_count:], samples_per_cycle, complete_count // samples_per_cycle


def harmonic_amplitude(
    values: Sequence[float],
    *,
    sample_dt_s: float,
    frequency_hz: float = GAIT_FREQUENCY_HZ,
) -> float:
    """Return the single-sided sinusoidal amplitude at one frequency."""

    if not values:
        raise ValueError("harmonic amplitude requires at least one sample")
    if sample_dt_s <= 0.0 or frequency_hz <= 0.0:
        raise ValueError("sample_dt_s and frequency_hz must be positive")
    mean = fmean(float(value) for value in values)
    cosine = 0.0
    sine = 0.0
    for index, value in enumerate(values):
        phase = 2.0 * math.pi * frequency_hz * sample_dt_s * index
        centered = float(value) - mean
        cosine += centered * math.cos(phase)
        sine += centered * math.sin(phase)
    return 2.0 * math.hypot(cosine, sine) / len(values)


def gait_cycle_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    sample_dt_s: float,
    steady_fraction: float = 0.50,
    gait_frequency_hz: float = GAIT_FREQUENCY_HZ,
) -> dict[str, Any]:
    """Aggregate velocity, torque, contact, and arm metrics on complete gait cycles."""

    window, samples_per_cycle, cycle_count = integer_cycle_window(
        rows,
        sample_dt_s=sample_dt_s,
        gait_frequency_hz=gait_frequency_hz,
        steady_fraction=steady_fraction,
    )
    result: dict[str, Any] = {
        "frequency_hz": gait_frequency_hz,
        "samples_per_cycle": samples_per_cycle,
        "cycle_count": cycle_count,
        "sample_count": len(window),
        "duration_s": len(window) * sample_dt_s,
        "axes": {},
    }
    if not window:
        result["status"] = "insufficient_complete_cycles"
        return result

    for axis in ("vx", "vy", "wz"):
        command = float(window[0][f"cmd_{axis}"])
        values = [float(row[f"measured_{axis}"]) for row in window]
        per_cycle_means = [
            fmean(values[start : start + samples_per_cycle])
            for start in range(0, len(values), samples_per_cycle)
        ]
        phase_bin_mean = [
            fmean(values[index] for index in range(phase, len(values), samples_per_cycle))
            for phase in range(samples_per_cycle)
        ]
        result["axes"][axis] = {
            "command": command,
            "cycle_mean": fmean(per_cycle_means),
            "cycle_mean_rmse": math.sqrt(fmean((value - command) ** 2 for value in per_cycle_means)),
            "sample_rmse": math.sqrt(fmean((value - command) ** 2 for value in values)),
            "per_cycle_mean": per_cycle_means,
            "phase_bin_mean": phase_bin_mean,
            "harmonic_amplitude": harmonic_amplitude(
                values,
                sample_dt_s=sample_dt_s,
                frequency_hz=gait_frequency_hz,
            ),
        }

    saturation_rows = [row.get("leg_torque_saturated") for row in window]
    if saturation_rows and all(isinstance(value, list) for value in saturation_rows):
        joint_count = len(saturation_rows[0])
        if all(len(value) == joint_count for value in saturation_rows):
            per_joint = [
                sum(bool(row[joint_index]) for row in saturation_rows) / len(saturation_rows)
                for joint_index in range(joint_count)
            ]
            result["torque_saturation_rate_per_joint"] = per_joint
            result["torque_saturation_rate_max"] = max(per_joint, default=0.0)

    clip_rows = [row.get("raw_action_clipped") for row in window]
    if clip_rows and all(isinstance(value, list) for value in clip_rows):
        action_count = len(clip_rows[0])
        clipped = sum(bool(value) for row in clip_rows for value in row)
        result["raw_action_clip_rate"] = clipped / (len(clip_rows) * action_count)

    arm_errors = [row.get("arm_joint_error") for row in window]
    if arm_errors and all(isinstance(value, list) for value in arm_errors):
        joint_count = len(arm_errors[0])
        if all(len(value) == joint_count for value in arm_errors):
            result["arm_joint_rmse_rad"] = [
                math.sqrt(fmean(float(row[joint_index]) ** 2 for row in arm_errors))
                for joint_index in range(joint_count)
            ]
            result["arm_joint_max_abs_rad"] = [
                max(abs(float(row[joint_index])) for row in arm_errors)
                for joint_index in range(joint_count)
            ]

    foot_force: dict[str, list[float]] = defaultdict(list)
    foot_slip: dict[str, list[float]] = defaultdict(list)
    foot_impulse: dict[str, float] = defaultdict(float)
    nonfoot_bodies: set[str] = set()
    for row in window:
        for name, value in row.get("foot_contact_force_n", {}).items():
            foot_force[str(name)].append(float(value))
        for name, value in row.get("foot_contact_slip_mps", {}).items():
            foot_slip[str(name)].append(float(value))
        for name, value in row.get("foot_contact_impulse_n_s", {}).items():
            foot_impulse[str(name)] += float(value)
        nonfoot_bodies.update(str(name) for name in row.get("nonfoot_contact_bodies", []))
    if foot_force:
        result["foot_contact"] = {
            name: {
                "peak_force_n": max(values),
                "mean_force_n": fmean(values),
                "peak_slip_mps": max(foot_slip.get(name, [0.0])),
                "integrated_impulse_n_s": foot_impulse.get(name, 0.0),
            }
            for name, values in foot_force.items()
        }
    result["nonfoot_contact_bodies"] = sorted(nonfoot_bodies)
    result["status"] = "ok"
    return result


def augment_summary_with_wtw_metrics(
    samples: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    *,
    sample_dt_s: float,
    steady_fraction: float,
    max_wz_harmonic_amplitude: float,
    max_torque_saturation_rate: float,
    max_action_clip_rate: float,
    expected_command_segments: int | None = None,
) -> dict[str, Any]:
    """Attach complete-cycle diagnostics and a walking-only acceptance result."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample["segment_index"])].append(sample)

    command_segments = []
    criteria = summary.get("criteria", {})
    for metric in summary.get("segments", []):
        cycle = gait_cycle_metrics(
            grouped.get(int(metric["segment_index"]), []),
            sample_dt_s=sample_dt_s,
            steady_fraction=steady_fraction,
        )
        metric["gait_cycle"] = cycle
        if metric.get("segment_kind") != "command":
            continue
        command_segments.append(metric)
        walking_failed: list[str] = []
        cycle_tracking_pass = True
        cycle_tracking: dict[str, bool] = {}
        for axis in ("vx", "vy", "wz"):
            axis_cycle = cycle.get("axes", {}).get(axis, {})
            command = axis_cycle.get("command")
            measured = axis_cycle.get("cycle_mean")
            if command is None or measured is None:
                axis_pass = False
            elif abs(float(command)) > 1.0e-6:
                gain = float(measured) / float(command)
                floor_key = "yaw_absolute_floor" if axis == "wz" else "linear_absolute_floor"
                tolerance = max(
                    float(criteria.get(floor_key, 0.0)),
                    float(criteria.get("relative_rmse_limit", 0.0)) * abs(float(command)),
                )
                axis_pass = bool(
                    float(criteria.get("gain_min", float("inf")))
                    <= gain
                    <= float(criteria.get("gain_max", float("-inf")))
                    and abs(float(measured) - float(command)) <= tolerance
                )
            else:
                limit_key = "zero_yaw_rmse_limit" if axis == "wz" else "zero_linear_rmse_limit"
                axis_pass = abs(float(measured)) <= float(criteria.get(limit_key, 0.0))
            cycle_tracking[axis] = axis_pass
            cycle_tracking_pass = cycle_tracking_pass and axis_pass
            if not axis_pass:
                walking_failed.append(f"{axis}_cycle_mean_tracking")
        metric["cycle_mean_tracking"] = cycle_tracking
        metric["cycle_mean_tracking_pass"] = cycle_tracking_pass
        wz_harmonic = cycle.get("axes", {}).get("wz", {}).get("harmonic_amplitude")
        torque_rate = cycle.get("torque_saturation_rate_max")
        clip_rate = cycle.get("raw_action_clip_rate")
        metric["wz_harmonic_pass"] = wz_harmonic is not None and wz_harmonic <= max_wz_harmonic_amplitude
        metric["torque_saturation_pass"] = torque_rate is not None and torque_rate <= max_torque_saturation_rate
        metric["action_clip_pass"] = clip_rate is not None and clip_rate <= max_action_clip_rate
        if not metric["wz_harmonic_pass"]:
            walking_failed.append("wz_2p5hz_harmonic")
        if not metric["torque_saturation_pass"]:
            walking_failed.append("torque_saturation")
        if not metric["action_clip_pass"]:
            walking_failed.append("raw_action_clip")
        if not metric.get("stability_pass"):
            walking_failed.append("stability")
        metric["walking_failed_checks"] = walking_failed
        metric["walking_passed"] = bool(
            cycle_tracking_pass
            and metric.get("stability_pass")
            and metric["wz_harmonic_pass"]
            and metric["torque_saturation_pass"]
            and metric["action_clip_pass"]
        )

    expected_commands = (
        int(summary.get("command_segments", 0))
        if expected_command_segments is None
        else expected_command_segments
    )
    passed_commands = sum(bool(metric.get("walking_passed")) for metric in command_segments)
    survival_passed = sum(not bool(metric.get("fell_or_reset")) for metric in command_segments)
    required_passed = math.ceil(0.90 * expected_commands) if expected_commands else 0
    walking = {
        "expected_command_segments": expected_commands,
        "evaluated_command_segments": len(command_segments),
        "survived_command_segments": survival_passed,
        "passed_command_segments": passed_commands,
        "required_passed_command_segments": required_passed,
        "stop_segments_are_diagnostic_only": True,
        "passed": bool(
            expected_commands > 0
            and len(command_segments) == expected_commands
            and survival_passed == expected_commands
            and passed_commands >= required_passed
        ),
    }
    summary["all_segment_metrics_passed"] = bool(summary.get("passed"))
    summary["walking_only"] = walking
    summary["passed"] = walking["passed"]
    return summary
