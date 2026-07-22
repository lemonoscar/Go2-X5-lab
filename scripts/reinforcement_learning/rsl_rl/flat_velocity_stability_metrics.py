"""Pure-Python schedules, metrics, and reports for flat velocity tracking."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


AXES = (
    ("vx", "cmd_vx", "measured_vx"),
    ("vy", "cmd_vy", "measured_vy"),
    ("wz", "cmd_wz", "measured_wz"),
)


@dataclass(frozen=True)
class CommandSegment:
    """One constant body-frame velocity command in the benchmark schedule."""

    name: str
    duration_s: float
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    kind: str = "command"
    evaluate: bool = True

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class MetricThresholds:
    """Explicit pass thresholds; linear and yaw units remain separate."""

    gain_min: float = 0.70
    gain_max: float = 1.30
    relative_rmse_limit: float = 0.30
    linear_absolute_floor: float = 0.04
    yaw_absolute_floor: float = 0.08
    zero_linear_rmse_limit: float = 0.08
    zero_yaw_rmse_limit: float = 0.10
    max_tilt_rad: float = 0.35
    max_base_height_std_m: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.gain_min <= self.gain_max:
            raise ValueError("gain bounds must satisfy 0 <= gain_min <= gain_max")
        positive_values = (
            self.relative_rmse_limit,
            self.linear_absolute_floor,
            self.yaw_absolute_floor,
            self.zero_linear_rmse_limit,
            self.zero_yaw_rmse_limit,
            self.max_tilt_rad,
            self.max_base_height_std_m,
        )
        if any(value <= 0.0 for value in positive_values):
            raise ValueError("metric tolerances must be positive")

    def absolute_floor(self, axis: str) -> float:
        return self.yaw_absolute_floor if axis == "wz" else self.linear_absolute_floor

    def zero_rmse_limit(self, axis: str) -> float:
        return self.zero_yaw_rmse_limit if axis == "wz" else self.zero_linear_rmse_limit

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


QUICK_COMMANDS = (
    ("vx_pos_010", 0.10, 0.0, 0.0),
    ("vx_pos_025", 0.25, 0.0, 0.0),
    ("vy_pos_010", 0.0, 0.10, 0.0),
    ("vy_neg_010", 0.0, -0.10, 0.0),
    ("wz_pos_020", 0.0, 0.0, 0.20),
    ("wz_neg_040", 0.0, 0.0, -0.40),
    ("terminal_mix", 0.16, 0.08, 0.30),
)


FULL_COMMANDS = (
    ("vx_pos_005", 0.05, 0.0, 0.0),
    ("vx_pos_010", 0.10, 0.0, 0.0),
    ("vx_pos_016", 0.16, 0.0, 0.0),
    ("vx_pos_025", 0.25, 0.0, 0.0),
    ("vx_pos_040", 0.40, 0.0, 0.0),
    ("vx_neg_010", -0.10, 0.0, 0.0),
    ("vx_neg_025", -0.25, 0.0, 0.0),
    ("vy_pos_005", 0.0, 0.05, 0.0),
    ("vy_pos_010", 0.0, 0.10, 0.0),
    ("vy_pos_020", 0.0, 0.20, 0.0),
    ("vy_neg_010", 0.0, -0.10, 0.0),
    ("vy_neg_020", 0.0, -0.20, 0.0),
    ("wz_pos_010", 0.0, 0.0, 0.10),
    ("wz_pos_020", 0.0, 0.0, 0.20),
    ("wz_pos_040", 0.0, 0.0, 0.40),
    ("wz_pos_060", 0.0, 0.0, 0.60),
    ("wz_neg_020", 0.0, 0.0, -0.20),
    ("wz_neg_040", 0.0, 0.0, -0.40),
    ("terminal_mix", 0.16, 0.08, 0.30),
    ("dwa_arc", 0.25, 0.0, 0.30),
)


PLANAR_COMMANDS = (
    ("vx_neg_070", -0.70, 0.0, 0.0),
    ("vx_neg_050", -0.50, 0.0, 0.0),
    ("vx_neg_030", -0.30, 0.0, 0.0),
    ("vx_neg_015", -0.15, 0.0, 0.0),
    ("vx_pos_015", 0.15, 0.0, 0.0),
    ("vx_pos_030", 0.30, 0.0, 0.0),
    ("vx_pos_050", 0.50, 0.0, 0.0),
    ("vx_pos_070", 0.70, 0.0, 0.0),
    ("vy_neg_020", 0.0, -0.20, 0.0),
    ("vy_neg_015", 0.0, -0.15, 0.0),
    ("vy_neg_010", 0.0, -0.10, 0.0),
    ("vy_neg_005", 0.0, -0.05, 0.0),
    ("vy_pos_005", 0.0, 0.05, 0.0),
    ("vy_pos_010", 0.0, 0.10, 0.0),
    ("vy_pos_015", 0.0, 0.15, 0.0),
    ("vy_pos_020", 0.0, 0.20, 0.0),
    ("mix_forward_left", 0.50, 0.15, 0.0),
    ("mix_forward_right", 0.50, -0.15, 0.0),
    ("mix_reverse_left", -0.50, 0.15, 0.0),
    ("mix_reverse_right", -0.50, -0.15, 0.0),
)


def build_schedule(
    profile: str,
    *,
    settle_s: float,
    hold_s: float,
    stop_s: float,
    repeats: int,
) -> list[CommandSegment]:
    """Build a command/stop schedule with unique names for every repeat."""

    if profile not in {"quick", "full", "planar"}:
        raise ValueError(f"unsupported profile: {profile}")
    if min(settle_s, hold_s, stop_s) <= 0.0 or repeats <= 0:
        raise ValueError("durations and repeats must be positive")

    commands = {
        "quick": QUICK_COMMANDS,
        "full": FULL_COMMANDS,
        "planar": PLANAR_COMMANDS,
    }[profile]
    schedule = [CommandSegment("initial_settle", settle_s, kind="settle", evaluate=False)]
    for repeat in range(1, repeats + 1):
        for name, vx, vy, wz in commands:
            suffix = f"r{repeat}"
            schedule.append(CommandSegment(f"{name}_{suffix}", hold_s, vx, vy, wz))
            schedule.append(CommandSegment(f"stop_after_{name}_{suffix}", stop_s, kind="stop"))
    return schedule


def _rmse(errors: Iterable[float]) -> float:
    values = list(errors)
    return math.sqrt(fmean(value * value for value in values))


def _rise_time(rows: list[dict[str, Any]], command: float, measured_key: str) -> float | None:
    if abs(command) <= 1.0e-6:
        return None
    direction = 1.0 if command > 0.0 else -1.0
    threshold = 0.90 * abs(command)
    for row in rows:
        if direction * float(row[measured_key]) >= threshold:
            return float(row["segment_time_s"])
    return None


def _settling_time(
    rows: list[dict[str, Any]],
    command: float,
    measured_key: str,
    tolerance: float,
) -> float | None:
    for index, row in enumerate(rows):
        if all(abs(float(item[measured_key]) - command) <= tolerance for item in rows[index:]):
            return float(row["segment_time_s"])
    return None


def analyze_samples(
    samples: list[dict[str, Any]],
    *,
    thresholds: MetricThresholds,
    steady_fraction: float = 0.50,
    expected_evaluated_segments: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-segment tracking and base-stability metrics."""

    if not 0.0 < steady_fraction <= 1.0:
        raise ValueError("steady_fraction must be in (0, 1]")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample["segment_index"])].append(sample)

    segment_metrics: list[dict[str, Any]] = []
    for segment_index in sorted(grouped):
        rows = grouped[segment_index]
        if not rows or not bool(rows[0].get("evaluate", False)):
            continue

        steady_start = max(0, min(len(rows) - 1, int(len(rows) * (1.0 - steady_fraction))))
        steady = rows[steady_start:]
        metric: dict[str, Any] = {
            "segment_index": segment_index,
            "segment_name": str(rows[0]["segment_name"]),
            "segment_kind": str(rows[0].get("segment_kind", "command")),
            "samples": len(rows),
            "duration_s": float(rows[-1]["segment_time_s"]),
            "fell_or_reset": any(bool(row.get("done", False)) for row in rows),
            "termination_terms": sorted(
                {
                    str(term)
                    for row in rows
                    for term in row.get("termination_terms", [])
                }
            ),
        }
        commanded_axes: list[str] = []
        zero_axes: list[str] = []
        failed_checks: list[str] = []

        for axis, command_key, measured_key in AXES:
            command = float(rows[0][command_key])
            values = [float(row[measured_key]) for row in steady]
            errors = [value - command for value in values]
            mean_value = fmean(values)
            rmse = _rmse(errors)
            tolerance = max(thresholds.absolute_floor(axis), thresholds.relative_rmse_limit * abs(command))
            settling_tolerance = tolerance if abs(command) > 1.0e-6 else thresholds.zero_rmse_limit(axis)

            metric[f"cmd_{axis}"] = command
            metric[f"mean_{axis}"] = mean_value
            metric[f"std_{axis}"] = pstdev(values) if len(values) > 1 else 0.0
            metric[f"mae_{axis}"] = fmean(abs(error) for error in errors)
            metric[f"rmse_{axis}"] = rmse
            metric[f"gain_{axis}"] = mean_value / command if abs(command) > 1.0e-6 else None
            metric[f"rise_time_{axis}_s"] = _rise_time(rows, command, measured_key)
            metric[f"settling_time_{axis}_s"] = _settling_time(
                rows, command, measured_key, settling_tolerance
            )
            metric[f"rmse_limit_{axis}"] = (
                tolerance if abs(command) > 1.0e-6 else thresholds.zero_rmse_limit(axis)
            )

            if abs(command) > 1.0e-6:
                commanded_axes.append(axis)
                gain = float(metric[f"gain_{axis}"])
                axis_pass = thresholds.gain_min <= gain <= thresholds.gain_max and rmse <= tolerance
            else:
                zero_axes.append(axis)
                axis_pass = rmse <= thresholds.zero_rmse_limit(axis)
            metric[f"tracking_pass_{axis}"] = axis_pass
            if not axis_pass:
                failed_checks.append(f"{axis}_tracking")

        base_z_values = [float(row["base_z"]) for row in steady]
        roll_values = [abs(float(row["base_roll"])) for row in rows]
        pitch_values = [abs(float(row["base_pitch"])) for row in rows]
        metric["commanded_axes"] = commanded_axes
        metric["zero_axes"] = zero_axes
        metric["mean_base_z_m"] = fmean(base_z_values)
        metric["std_base_z_m"] = pstdev(base_z_values) if len(base_z_values) > 1 else 0.0
        metric["max_abs_roll_rad"] = max(roll_values)
        metric["max_abs_pitch_rad"] = max(pitch_values)
        metric["max_abs_tilt_rad"] = max(metric["max_abs_roll_rad"], metric["max_abs_pitch_rad"])
        action_values = [
            float(row["action_abs_max"])
            for row in rows
            if row.get("action_abs_max") is not None
        ]
        metric["action_abs_max"] = max(action_values) if action_values else None
        metric["tracking_pass"] = all(bool(metric[f"tracking_pass_{axis}"]) for axis, _, _ in AXES)
        metric["tilt_pass"] = metric["max_abs_tilt_rad"] <= thresholds.max_tilt_rad
        metric["base_height_pass"] = metric["std_base_z_m"] <= thresholds.max_base_height_std_m
        metric["no_reset_pass"] = not metric["fell_or_reset"]
        if not metric["tilt_pass"]:
            failed_checks.append("tilt")
        if not metric["base_height_pass"]:
            failed_checks.append("base_height_std")
        if not metric["no_reset_pass"]:
            failed_checks.append("reset")
        metric["stability_pass"] = bool(
            metric["tilt_pass"] and metric["base_height_pass"] and metric["no_reset_pass"]
        )
        metric["failed_checks"] = failed_checks
        metric["passed"] = bool(metric["tracking_pass"] and metric["stability_pass"])
        segment_metrics.append(metric)

    command_segments = [item for item in segment_metrics if item["segment_kind"] == "command"]
    stop_segments = [item for item in segment_metrics if item["segment_kind"] == "stop"]
    evaluated = len(segment_metrics)
    expected = evaluated if expected_evaluated_segments is None else expected_evaluated_segments
    complete = evaluated == expected
    passed_segments = sum(bool(item["passed"]) for item in segment_metrics)
    return {
        "criteria": {
            "steady_window_fraction": steady_fraction,
            **thresholds.to_dict(),
            "scope_note": (
                "Deterministic flat-ground tracking isolates the policy/controller response; "
                "it does not prove domain-randomized or real-robot robustness."
            ),
        },
        "expected_evaluated_segments": expected,
        "evaluated_segments": evaluated,
        "complete_schedule": complete,
        "passed_segments": passed_segments,
        "command_segments": len(command_segments),
        "passed_command_segments": sum(bool(item["passed"]) for item in command_segments),
        "stop_segments": len(stop_segments),
        "passed_stop_segments": sum(bool(item["passed"]) for item in stop_segments),
        "pass_rate": passed_segments / expected if expected else 0.0,
        "passed": bool(complete and segment_metrics and passed_segments == evaluated),
        "failed_segments": [str(item["segment_name"]) for item in segment_metrics if not item["passed"]],
        "missing_segment_count": max(0, expected - evaluated),
        "segments": segment_metrics,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, samples: list[dict[str, Any]]) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib_not_installed"

    times = [float(row["time_s"]) for row in samples]
    fig, plot_axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for plot_axis, (axis, command_key, measured_key) in zip(plot_axes, AXES):
        plot_axis.plot(times, [float(row[command_key]) for row in samples], "k--", linewidth=1.0, label="command")
        plot_axis.plot(times, [float(row[measured_key]) for row in samples], linewidth=1.0, label="measured")
        plot_axis.set_ylabel(f"{axis} (m/s)" if axis != "wz" else "wz (rad/s)")
        plot_axis.grid(True, alpha=0.3)
        plot_axis.legend(loc="upper right")
    plot_axes[-1].set_xlabel("benchmark time (s)")
    fig.suptitle("Go2-X5 DogOnly flat velocity tracking")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return None


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _write_markdown(path: Path, summary: dict[str, Any], metadata: dict[str, Any]) -> None:
    criteria = summary["criteria"]
    lines = [
        "# Go2-X5 DogOnly Flat 速度追踪稳定性报告",
        "",
        f"- Checkpoint：`{metadata['checkpoint']}`",
        f"- Task：`{metadata['task']}`",
        f"- Profile/repeats：`{metadata['profile']}` / `{metadata['repeats']}`",
        f"- Seed / control dt：`{metadata['seed']}` / `{metadata['control_dt_s']:.4f} s`",
        "- 地形与随机性：deterministic plane；reset 固定；corruption/domain randomization/delay 已关闭",
        f"- 完整日程：**{summary['complete_schedule']}**；总通过：**{summary['passed']}**",
        f"- 已评估/预期段数：{summary['evaluated_segments']}/{summary['expected_evaluated_segments']}；"
        f"缺失：{summary['missing_segment_count']}",
        f"- 命令段：{summary['passed_command_segments']}/{summary['command_segments']}；"
        f"停稳段：{summary['passed_stop_segments']}/{summary['stop_segments']}",
        "",
        "## 判定标准",
        "",
        f"- 每段后 `{criteria['steady_window_fraction']:.0%}` 为稳态窗口。",
        f"- 非零命令轴：gain 位于 `[{criteria['gain_min']:.2f}, {criteria['gain_max']:.2f}]`，"
        f"RMSE 不超过 `max(axis floor, {criteria['relative_rmse_limit']:.0%} * |command|)`。",
        f"- axis floor：linear `{criteria['linear_absolute_floor']:.3f} m/s`，"
        f"yaw `{criteria['yaw_absolute_floor']:.3f} rad/s`。",
        f"- 零命令轴 RMSE：linear `<= {criteria['zero_linear_rmse_limit']:.3f} m/s`，"
        f"yaw `<= {criteria['zero_yaw_rmse_limit']:.3f} rad/s`。",
        f"- 姿态/高度：max |roll/pitch| `<= {criteria['max_tilt_rad']:.3f} rad`，"
        f"base-z std `<= {criteria['max_base_height_std_m']:.3f} m`，且无 reset。",
        "",
        "## 分段结果",
        "",
        "| segment | kind | cmd vx/vy/wz | mean vx/vy/wz | RMSE vx/vy/wz | "
        "settle vx/vy/wz | max tilt | z std | failure | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|:---:|",
    ]
    if metadata.get("terminated_early"):
        event = metadata.get("termination_event") or {}
        lines.insert(
            8,
            f"- 提前终止：`{event.get('segment_name', 'unknown')}`，terms=`{event.get('terms', [])}`",
        )
    for item in summary["segments"]:
        settling = "/".join(
            _format_optional(item[f"settling_time_{axis}_s"]) for axis in ("vx", "vy", "wz")
        )
        lines.append(
            f"| {item['segment_name']} | {item['segment_kind']} | "
            f"{item['cmd_vx']:.2f}/{item['cmd_vy']:.2f}/{item['cmd_wz']:.2f} | "
            f"{item['mean_vx']:.2f}/{item['mean_vy']:.2f}/{item['mean_wz']:.2f} | "
            f"{item['rmse_vx']:.3f}/{item['rmse_vy']:.3f}/{item['rmse_wz']:.3f} | {settling} | "
            f"{item['max_abs_tilt_rad']:.3f} | {item['std_base_z_m']:.3f} | "
            f"{','.join(item['failed_checks']) or '-'} | "
            f"{'通过' if item['passed'] else '失败'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该测试用于隔离 flat plane 上低层 policy 与控制器的速度阶跃响应。"
            "它不包含训练时 domain randomization、外部推力或复杂地形，"
            "因此不能单独证明真实机鲁棒性。命令段与停稳段分开统计，"
            "可区分持续跟踪不足、跨轴漂移和制动残留。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_artifacts(
    output_dir: Path,
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write machine-readable metrics, a report, and an optional velocity plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_warning = _write_plot(output_dir / "velocity_tracking.png", samples)
    if plot_warning:
        summary["plot_warning"] = plot_warning
    payload = {"metadata": metadata, **summary}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "segment_metrics.csv", summary["segments"])
    _write_markdown(output_dir / "report.md", summary, metadata)
