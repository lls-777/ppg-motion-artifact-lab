#!/usr/bin/env python3
"""设计可解释的 ACC-aware SQI V1。

本脚本只读取 S1-S10 的 ACC 特征表，不读取 S11-S15。

研究目标：
1. 比较运动强度、频率污染和二者组合的门控能力；
2. 使用按受试者平衡的目标选择透明门限；
3. 通过 leave-one-subject-out 检查门限的跨人稳定性；
4. 在全部 S1-S10 上冻结一组设计参数，供后续 S11-S15
   一次性验证。

本阶段默认只在终端输出结果；只有显式传入 --write-outputs
才会保存冻结设计参数、CSV 和图片。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    DATA_ROOT
    / "02_结果"
    / "acc_motion_features_v0"
    / "S1_S10_acc_motion_features_v0.csv"
)
OUTPUT_DIR = DATA_ROOT / "02_结果" / "acc_aware_sqi_v1_design"

DESIGN_SUBJECTS = tuple(range(1, 11))
EXPECTED_DESIGN_WINDOWS = 43_215
EXPECTED_BASELINE_USABLE_WINDOWS = 12_906

ERROR_THRESHOLD_BPM = 10.0

# 候选阈值由训练受试者中 PPG-only usable 窗口的分位数产生。
# 最终写入 C 时使用的是冻结后的数值阈值，不需要实现分位数计算。
MOTION_PERCENTILES = (50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0)
POWER_PERCENTILES = (50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0)
GAP_PERCENTILES = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)

# 选择门限时，优先保护原本误差不超过 10 bpm 的窗口。
MIN_MACRO_CORRECT_RETENTION_PERCENT = 95.0
MIN_SUBJECT_CORRECT_RETENTION_PERCENT = 85.0

RULE_FAMILIES = (
    "motion_only",
    "collision_only",
    "motion_and_collision",
)

REQUIRED_COLUMNS = (
    "subject_id",
    "split",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "ppg_hr_bpm",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "sqi_score",
    "state",
    "reason",
    "motion_rms_g",
    "jerk_rms_g_per_s",
    "acc_power_near_ppg_ratio",
    "ppg_acc_harmonic_gap_hz",
)

STRING_COLUMNS = (
    "split",
    "activity_name",
    "state",
    "reason",
)

WINDOW_OUTPUT_FIELDS = (
    "subject_id",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "ppg_hr_bpm",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "sqi_score",
    "ppg_sqi_state",
    "ppg_sqi_reason",
    "motion_rms_g",
    "jerk_rms_g_per_s",
    "acc_power_near_ppg_ratio",
    "ppg_acc_harmonic_gap_hz",
    "motion_flag",
    "collision_flag",
    "acc_aware_state",
    "acc_aware_reason",
)

METRIC_FIELDS = (
    "count",
    "coverage_percent",
    "mae_bpm",
    "rmse_bpm",
    "p90_bpm",
    "within_3_bpm_percent",
    "failure_gt_10_bpm_percent",
)

DESIGN_METRIC_FIELDS = METRIC_FIELDS + (
    "coverage_of_all_design_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="设计并冻结 ACC-aware SQI V1 参数"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="显式保存设计参数、CSV 和 PNG 证据",
    )
    return parser.parse_args()


def load_design_data(
    csv_path: Path,
) -> dict[str, np.ndarray]:
    """读取并验证 S1-S10 ACC 特征表。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 ACC 特征表：{csv_path}")

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in columns
        ]
        if missing:
            raise ValueError(
                "ACC 特征表缺少字段：" + ", ".join(missing)
            )
        rows = list(reader)

    if len(rows) != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            f"窗口数异常：期望 {EXPECTED_DESIGN_WINDOWS}，"
            f"实际 {len(rows)}"
        )

    data: dict[str, np.ndarray] = {}

    for column in STRING_COLUMNS:
        data[column] = np.asarray(
            [row[column] for row in rows],
            dtype=object,
        )

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in STRING_COLUMNS
    ]

    for column in numeric_columns:
        try:
            data[column] = np.asarray(
                [float(row[column]) for row in rows],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ValueError(
                f"字段 {column} 中存在无法转换的数值"
            ) from error

        if not np.all(np.isfinite(data[column])):
            raise ValueError(
                f"字段 {column} 中存在 NaN 或无穷值"
            )

    data["subject_id"] = data["subject_id"].astype(np.int64)
    data["activity_id"] = data["activity_id"].astype(np.int64)

    observed_subjects = set(data["subject_id"].tolist())
    if observed_subjects != set(DESIGN_SUBJECTS):
        raise ValueError(
            f"受试者异常：期望 {DESIGN_SUBJECTS}，"
            f"实际 {sorted(observed_subjects)}"
        )

    observed_splits = set(data["split"].tolist())
    if observed_splits != {"design"}:
        raise ValueError(
            "输入中出现非设计集数据："
            f"{sorted(observed_splits)}"
        )

    baseline_usable_count = int(
        np.sum(data["state"] == "usable")
    )
    if baseline_usable_count != EXPECTED_BASELINE_USABLE_WINDOWS:
        raise ValueError(
            "PPG-only usable 窗口数异常："
            f"期望 {EXPECTED_BASELINE_USABLE_WINDOWS}，"
            f"实际 {baseline_usable_count}"
        )

    return data


def candidate_parameters(
    family: str,
    data: dict[str, np.ndarray],
    training_mask: np.ndarray,
) -> list[dict[str, float]]:
    """从训练受试者生成透明、有限的候选阈值。"""
    if not np.any(training_mask):
        raise ValueError("训练范围中没有 PPG-only usable 窗口")

    motion_thresholds = {
        percentile: (
            float(
                np.percentile(
                    data["motion_rms_g"][training_mask],
                    percentile,
                )
            ),
            float(
                np.percentile(
                    data["jerk_rms_g_per_s"][training_mask],
                    percentile,
                )
            ),
        )
        for percentile in MOTION_PERCENTILES
    }
    power_thresholds = {
        percentile: float(
            np.percentile(
                data["acc_power_near_ppg_ratio"][training_mask],
                percentile,
            )
        )
        for percentile in POWER_PERCENTILES
    }
    gap_thresholds = {
        percentile: float(
            np.percentile(
                data["ppg_acc_harmonic_gap_hz"][training_mask],
                percentile,
            )
        )
        for percentile in GAP_PERCENTILES
    }

    parameters: list[dict[str, float]] = []

    if family == "motion_only":
        for motion_percentile in MOTION_PERCENTILES:
            motion_rms, jerk_rms = motion_thresholds[
                motion_percentile
            ]
            parameters.append(
                {
                    "motion_percentile": motion_percentile,
                    "motion_rms_threshold_g": motion_rms,
                    "jerk_rms_threshold_g_per_s": jerk_rms,
                }
            )
        return parameters

    if family == "collision_only":
        for power_percentile, gap_percentile in product(
            POWER_PERCENTILES,
            GAP_PERCENTILES,
        ):
            parameters.append(
                {
                    "power_percentile": power_percentile,
                    "gap_percentile": gap_percentile,
                    "acc_power_near_ppg_threshold": (
                        power_thresholds[power_percentile]
                    ),
                    "harmonic_gap_threshold_hz": (
                        gap_thresholds[gap_percentile]
                    ),
                }
            )
        return parameters

    if family == "motion_and_collision":
        for (
            motion_percentile,
            power_percentile,
            gap_percentile,
        ) in product(
            MOTION_PERCENTILES,
            POWER_PERCENTILES,
            GAP_PERCENTILES,
        ):
            motion_rms, jerk_rms = motion_thresholds[
                motion_percentile
            ]
            parameters.append(
                {
                    "motion_percentile": motion_percentile,
                    "power_percentile": power_percentile,
                    "gap_percentile": gap_percentile,
                    "motion_rms_threshold_g": motion_rms,
                    "jerk_rms_threshold_g_per_s": jerk_rms,
                    "acc_power_near_ppg_threshold": (
                        power_thresholds[power_percentile]
                    ),
                    "harmonic_gap_threshold_hz": (
                        gap_thresholds[gap_percentile]
                    ),
                }
            )
        return parameters

    raise ValueError(f"未知规则族：{family}")


def apply_rule(
    family: str,
    data: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回运动标志、频率污染标志和拒绝标志。"""
    count = len(data["subject_id"])
    motion = np.zeros(count, dtype=bool)
    collision = np.zeros(count, dtype=bool)

    if family in ("motion_only", "motion_and_collision"):
        motion = (
            (
                data["motion_rms_g"]
                >= parameters["motion_rms_threshold_g"]
            )
            | (
                data["jerk_rms_g_per_s"]
                >= parameters["jerk_rms_threshold_g_per_s"]
            )
        )

    if family in ("collision_only", "motion_and_collision"):
        collision = (
            (
                data["acc_power_near_ppg_ratio"]
                >= parameters["acc_power_near_ppg_threshold"]
            )
            | (
                data["ppg_acc_harmonic_gap_hz"]
                <= parameters["harmonic_gap_threshold_hz"]
            )
        )

    if family == "motion_only":
        reject = motion
    elif family == "collision_only":
        reject = collision
    else:
        reject = motion & collision

    return motion, collision, reject


def safe_percent(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def evaluate_rejection(
    data: dict[str, np.ndarray],
    universe_mask: np.ndarray,
    reject_mask: np.ndarray,
) -> dict[str, float]:
    """按受试者平衡地评价拒绝规则。"""
    errors = data["absolute_error_bpm"]
    subjects = data["subject_id"]

    correct_retentions: list[float] = []
    error_captures: list[float] = []
    remaining_failures: list[float] = []

    for subject_id in sorted(set(subjects[universe_mask].tolist())):
        subject_mask = universe_mask & (subjects == subject_id)
        correct = (
            subject_mask
            & (errors <= ERROR_THRESHOLD_BPM)
        )
        failure = (
            subject_mask
            & (errors > ERROR_THRESHOLD_BPM)
        )
        retained = subject_mask & ~reject_mask

        if np.any(correct):
            correct_retentions.append(
                float(100.0 * np.mean(~reject_mask[correct]))
            )
        if np.any(failure):
            error_captures.append(
                float(100.0 * np.mean(reject_mask[failure]))
            )
        if np.any(retained):
            remaining_failures.append(
                float(
                    100.0
                    * np.mean(
                        errors[retained] > ERROR_THRESHOLD_BPM
                    )
                )
            )

    correct = (
        universe_mask
        & (errors <= ERROR_THRESHOLD_BPM)
    )
    failure = (
        universe_mask
        & (errors > ERROR_THRESHOLD_BPM)
    )
    retained = universe_mask & ~reject_mask

    return {
        "macro_correct_retention_percent": float(
            np.mean(correct_retentions)
        ),
        "minimum_subject_correct_retention_percent": float(
            np.min(correct_retentions)
        ),
        "macro_error_capture_percent": float(
            np.mean(error_captures)
        ),
        "macro_remaining_failure_percent": float(
            np.mean(remaining_failures)
        ),
        "overall_correct_retention_percent": safe_percent(
            int(np.sum(correct & ~reject_mask)),
            int(np.sum(correct)),
        ),
        "overall_error_capture_percent": safe_percent(
            int(np.sum(failure & reject_mask)),
            int(np.sum(failure)),
        ),
        "retained_coverage_percent": safe_percent(
            int(np.sum(retained)),
            int(np.sum(universe_mask)),
        ),
        "remaining_failure_percent": (
            float(
                100.0
                * np.mean(
                    errors[retained] > ERROR_THRESHOLD_BPM
                )
            )
            if np.any(retained)
            else 100.0
        ),
    }


def select_rule(
    family: str,
    data: dict[str, np.ndarray],
    training_mask: np.ndarray,
) -> dict[str, Any]:
    """选择满足准确窗口保护条件且捕获错误最多的规则。"""
    feasible: list[tuple[dict[str, float], dict[str, float]]] = []
    fallback: list[tuple[dict[str, float], dict[str, float]]] = []

    for parameters in candidate_parameters(
        family,
        data,
        training_mask,
    ):
        _, _, reject = apply_rule(family, data, parameters)
        metrics = evaluate_rejection(
            data,
            training_mask,
            reject,
        )
        fallback.append((parameters, metrics))

        if (
            metrics["macro_correct_retention_percent"]
            >= MIN_MACRO_CORRECT_RETENTION_PERCENT
            and metrics[
                "minimum_subject_correct_retention_percent"
            ]
            >= MIN_SUBJECT_CORRECT_RETENTION_PERCENT
        ):
            feasible.append((parameters, metrics))

    if feasible:
        parameters, metrics = max(
            feasible,
            key=lambda item: (
                item[1]["macro_error_capture_percent"],
                item[1]["macro_correct_retention_percent"],
                item[1]["overall_error_capture_percent"],
                item[1]["retained_coverage_percent"],
            ),
        )
        target_met = True
    else:
        # 如果保护条件不可达，明确保留失败状态，不隐藏问题。
        parameters, metrics = max(
            fallback,
            key=lambda item: (
                (
                    item[1]["macro_error_capture_percent"]
                    + item[1]["macro_correct_retention_percent"]
                )
                / 2.0,
                item[1]["macro_correct_retention_percent"],
            ),
        )
        target_met = False

    return {
        "family": family,
        "target_met": target_met,
        "parameters": parameters,
        "metrics": metrics,
    }


def summarize_mask(
    errors: np.ndarray,
    universe_mask: np.ndarray,
    selected_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """使用与 PPG-only SQI 相同的误差和覆盖率口径。"""
    selected = universe_mask & selected_mask
    count = int(np.sum(selected))
    total = int(np.sum(universe_mask))

    if count == 0:
        return {
            "count": 0,
            "coverage_percent": 0.0,
            "mae_bpm": None,
            "rmse_bpm": None,
            "p90_bpm": None,
            "within_3_bpm_percent": None,
            "failure_gt_10_bpm_percent": None,
        }

    selected_errors = errors[selected]
    return {
        "count": count,
        "coverage_percent": safe_percent(count, total),
        "mae_bpm": float(np.mean(selected_errors)),
        "rmse_bpm": float(
            np.sqrt(np.mean(selected_errors**2))
        ),
        "p90_bpm": float(
            np.percentile(selected_errors, 90)
        ),
        "within_3_bpm_percent": float(
            100.0 * np.mean(selected_errors <= 3.0)
        ),
        "failure_gt_10_bpm_percent": float(
            100.0
            * np.mean(
                selected_errors > ERROR_THRESHOLD_BPM
            )
        ),
    }


def assign_acc_states(
    data: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在原 PPG-only 状态上增加 ACC-aware 状态和原因。"""
    states = data["state"].copy()
    reasons = data["reason"].copy()
    baseline_usable = data["state"] == "usable"

    motion, collision, reject = apply_rule(
        "motion_and_collision",
        data,
        parameters,
    )

    acc_reject = baseline_usable & reject
    acc_low = (
        baseline_usable
        & ~reject
        & (motion | collision)
    )
    acc_usable = (
        baseline_usable
        & ~motion
        & ~collision
    )

    states[acc_usable] = "usable"
    reasons[acc_usable] = "none"

    states[acc_low] = "low_confidence"
    reasons[acc_low & motion & ~collision] = "strong_motion"
    reasons[acc_low & collision & ~motion] = (
        "possible_motion_frequency_collision"
    )

    states[acc_reject] = "reject"
    reasons[acc_reject] = "motion_frequency_collision"

    return states, reasons, motion, collision


def run_loso(
    data: dict[str, np.ndarray],
    baseline_usable: np.ndarray,
) -> list[dict[str, Any]]:
    """按受试者留一，检查组合门控的跨人稳定性。"""
    rows: list[dict[str, Any]] = []

    for held_subject in DESIGN_SUBJECTS:
        training_mask = (
            baseline_usable
            & (data["subject_id"] != held_subject)
        )
        held_mask = (
            baseline_usable
            & (data["subject_id"] == held_subject)
        )

        selection = select_rule(
            "motion_and_collision",
            data,
            training_mask,
        )
        _, _, reject = apply_rule(
            "motion_and_collision",
            data,
            selection["parameters"],
        )
        metrics = evaluate_rejection(
            data,
            held_mask,
            reject,
        )

        rows.append(
            {
                "held_subject": held_subject,
                "target_met_on_training": selection["target_met"],
                "parameters": selection["parameters"],
                **metrics,
            }
        )

    return rows


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}"


def print_selection(selection: dict[str, Any]) -> None:
    metrics = selection["metrics"]
    print(
        f"\n{selection['family']}: "
        f"target_met={selection['target_met']}"
    )
    for name, value in selection["parameters"].items():
        print(f"  {name}={value:.6f}")
    print(
        "  macro correct retention="
        f"{metrics['macro_correct_retention_percent']:.2f}%"
    )
    print(
        "  minimum subject correct retention="
        f"{metrics['minimum_subject_correct_retention_percent']:.2f}%"
    )
    print(
        "  macro error capture="
        f"{metrics['macro_error_capture_percent']:.2f}%"
    )
    print(
        "  retained coverage="
        f"{metrics['retained_coverage_percent']:.2f}%"
    )
    print(
        "  remaining >10 bpm="
        f"{metrics['remaining_failure_percent']:.2f}%"
    )


def print_output_metrics(
    name: str,
    metrics: dict[str, float | int | None],
) -> None:
    all_design_coverage = metrics.get(
        "coverage_of_all_design_percent"
    )
    all_design_text = (
        ""
        if all_design_coverage is None
        else (
            ", all-design coverage="
            f"{format_metric(all_design_coverage)}%"
        )
    )
    print(
        f"  {name}: "
        f"n={metrics['count']}, "
        "relative coverage="
        f"{format_metric(metrics['coverage_percent'])}%"
        f"{all_design_text}, "
        f"MAE={format_metric(metrics['mae_bpm'])} bpm, "
        f"RMSE={format_metric(metrics['rmse_bpm'])} bpm, "
        f"P90={format_metric(metrics['p90_bpm'])} bpm, "
        ">10 bpm="
        f"{format_metric(metrics['failure_gt_10_bpm_percent'])}%"
    )


def flatten_selection(
    selection: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "family": selection["family"],
        "target_met": selection["target_met"],
    }
    row.update(selection["parameters"])
    row.update(selection["metrics"])
    return row


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_window_table(
    output_path: Path,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    reasons: np.ndarray,
    motion: np.ndarray,
    collision: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=WINDOW_OUTPUT_FIELDS,
        )
        writer.writeheader()

        for index in range(len(states)):
            writer.writerow(
                {
                    "subject_id": int(data["subject_id"][index]),
                    "window_start_s": data["window_start_s"][index],
                    "window_center_s": data["window_center_s"][index],
                    "window_end_s": data["window_end_s"][index],
                    "activity_id": int(data["activity_id"][index]),
                    "activity_name": data["activity_name"][index],
                    "ppg_hr_bpm": data["ppg_hr_bpm"][index],
                    "ecg_hr_bpm": data["ecg_hr_bpm"][index],
                    "absolute_error_bpm": (
                        data["absolute_error_bpm"][index]
                    ),
                    "sqi_score": data["sqi_score"][index],
                    "ppg_sqi_state": data["state"][index],
                    "ppg_sqi_reason": data["reason"][index],
                    "motion_rms_g": data["motion_rms_g"][index],
                    "jerk_rms_g_per_s": (
                        data["jerk_rms_g_per_s"][index]
                    ),
                    "acc_power_near_ppg_ratio": (
                        data["acc_power_near_ppg_ratio"][index]
                    ),
                    "ppg_acc_harmonic_gap_hz": (
                        data["ppg_acc_harmonic_gap_hz"][index]
                    ),
                    "motion_flag": bool(motion[index]),
                    "collision_flag": bool(collision[index]),
                    "acc_aware_state": states[index],
                    "acc_aware_reason": reasons[index],
                }
            )


def build_partition_rows(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    group_column: str,
) -> list[dict[str, Any]]:
    """生成分受试者或分活动的误差—覆盖率结果。"""
    baseline_usable = data["state"] == "usable"
    errors = data["absolute_error_bpm"]
    rows: list[dict[str, Any]] = []

    for group_value in sorted(set(data[group_column].tolist())):
        group_mask = data[group_column] == group_value
        universe = baseline_usable & group_mask

        if not np.any(universe):
            continue

        if group_column == "subject_id":
            group_name = f"S{int(group_value)}"
        else:
            first_index = int(np.flatnonzero(group_mask)[0])
            group_name = str(data["activity_name"][first_index])

        output_masks = (
            ("ppg_only_usable", baseline_usable),
            ("acc_aware_usable", states == "usable"),
            (
                "acc_aware_usable_plus_low",
                np.isin(
                    states,
                    ("usable", "low_confidence"),
                ),
            ),
            ("acc_aware_reject", states == "reject"),
        )

        for output_range, selected_mask in output_masks:
            metrics = summarize_mask(
                errors,
                universe,
                selected_mask,
            )
            rows.append(
                {
                    "group_id": int(group_value),
                    "group_name": group_name,
                    "output_range": output_range,
                    **metrics,
                }
            )

    return rows


def plot_family_comparison(
    selections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [
        "Motion only",
        "Collision only",
        "Motion + collision",
    ]
    correct_retention = [
        row["metrics"]["overall_correct_retention_percent"]
        for row in selections
    ]
    error_capture = [
        row["metrics"]["overall_error_capture_percent"]
        for row in selections
    ]
    remaining_failure = [
        row["metrics"]["remaining_failure_percent"]
        for row in selections
    ]

    x_positions = np.arange(len(labels))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        x_positions - width / 2,
        correct_retention,
        width,
        label="Correct retention",
    )
    axes[0].bar(
        x_positions + width / 2,
        error_capture,
        width,
        label="Large-error capture",
    )
    axes[0].set_xticks(x_positions, labels, rotation=20)
    axes[0].set_ylim(0.0, 105.0)
    axes[0].set_ylabel("Percent (%)")
    axes[0].set_title("Rule-family comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, remaining_failure, color="#C55A11")
    axes[1].set_ylabel("Retained windows with error >10 bpm (%)")
    axes[1].set_title("Residual failure after rejection")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_loso_results(
    loso_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [f"S{row['held_subject']}" for row in loso_rows]
    correct_retention = [
        row["overall_correct_retention_percent"]
        for row in loso_rows
    ]
    error_capture = [
        row["overall_error_capture_percent"]
        for row in loso_rows
    ]

    x_positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.5))

    axis.bar(
        x_positions - width / 2,
        correct_retention,
        width,
        label="Correct retention",
    )
    axis.bar(
        x_positions + width / 2,
        error_capture,
        width,
        label="Large-error capture",
    )
    axis.axhline(
        MIN_MACRO_CORRECT_RETENTION_PERCENT,
        color="#C00000",
        linestyle="--",
        linewidth=1.2,
        label="95% training-selection target",
    )
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Percent (%)")
    axis.set_title(
        "Leave-one-subject-out stability: combined ACC gate"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_output_tradeoff(
    metrics: dict[str, dict[str, float | int | None]],
    output_path: Path,
) -> None:
    keys = (
        "ppg_only_usable",
        "acc_aware_usable",
        "acc_aware_usable_plus_low",
    )
    labels = (
        "PPG-only usable",
        "ACC usable",
        "ACC usable + low",
    )
    coverage = [
        float(metrics[key]["coverage_percent"] or 0.0)
        for key in keys
    ]
    failure = [
        float(
            metrics[key]["failure_gt_10_bpm_percent"]
            or 0.0
        )
        for key in keys
    ]

    x_positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))

    axis.bar(
        x_positions - width / 2,
        coverage,
        width,
        label="Coverage vs PPG-only usable",
    )
    axis.bar(
        x_positions + width / 2,
        failure,
        width,
        label="Error >10 bpm",
    )
    axis.set_xticks(x_positions, labels, rotation=15)
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Percent (%)")
    axis.set_title("Accuracy-coverage trade-off on S1-S10")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    data: dict[str, np.ndarray],
    family_selections: list[dict[str, Any]],
    loso_rows: list[dict[str, Any]],
    final_selection: dict[str, Any],
    states: np.ndarray,
    reasons: np.ndarray,
    motion: np.ndarray,
    collision: np.ndarray,
    metrics: dict[str, dict[str, float | int | None]],
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_path = (
        OUTPUT_DIR / "S1_S10_acc_aware_sqi_v1_windows.csv"
    )
    family_path = OUTPUT_DIR / "family_comparison.csv"
    loso_path = OUTPUT_DIR / "loso_subject_results.csv"
    design_metrics_path = OUTPUT_DIR / "design_metrics.csv"
    subject_path = OUTPUT_DIR / "subject_metrics.csv"
    activity_path = OUTPUT_DIR / "activity_metrics.csv"
    parameter_path = (
        OUTPUT_DIR / "acc_aware_sqi_v1_frozen_design.json"
    )
    family_plot_path = OUTPUT_DIR / "family_comparison.png"
    loso_plot_path = OUTPUT_DIR / "loso_subject_stability.png"
    tradeoff_plot_path = OUTPUT_DIR / "design_output_tradeoff.png"

    write_window_table(
        window_path,
        data,
        states,
        reasons,
        motion,
        collision,
    )

    family_rows = [
        flatten_selection(selection)
        for selection in family_selections
    ]
    selection_fields = (
        "family",
        "target_met",
        "motion_percentile",
        "power_percentile",
        "gap_percentile",
        "motion_rms_threshold_g",
        "jerk_rms_threshold_g_per_s",
        "acc_power_near_ppg_threshold",
        "harmonic_gap_threshold_hz",
        "macro_correct_retention_percent",
        "minimum_subject_correct_retention_percent",
        "macro_error_capture_percent",
        "macro_remaining_failure_percent",
        "overall_correct_retention_percent",
        "overall_error_capture_percent",
        "retained_coverage_percent",
        "remaining_failure_percent",
    )
    write_csv(family_path, family_rows, selection_fields)

    flattened_loso: list[dict[str, Any]] = []
    for row in loso_rows:
        flattened = {
            key: value
            for key, value in row.items()
            if key != "parameters"
        }
        flattened.update(row["parameters"])
        flattened_loso.append(flattened)

    write_csv(
        loso_path,
        flattened_loso,
        (
            "held_subject",
            "target_met_on_training",
            "motion_percentile",
            "power_percentile",
            "gap_percentile",
            "motion_rms_threshold_g",
            "jerk_rms_threshold_g_per_s",
            "acc_power_near_ppg_threshold",
            "harmonic_gap_threshold_hz",
            "macro_correct_retention_percent",
            "minimum_subject_correct_retention_percent",
            "macro_error_capture_percent",
            "macro_remaining_failure_percent",
            "overall_correct_retention_percent",
            "overall_error_capture_percent",
            "retained_coverage_percent",
            "remaining_failure_percent",
        ),
    )

    design_rows = [
        {"output_range": name, **row}
        for name, row in metrics.items()
    ]
    write_csv(
        design_metrics_path,
        design_rows,
        ("output_range",) + DESIGN_METRIC_FIELDS,
    )

    partition_fields = (
        "group_id",
        "group_name",
        "output_range",
    ) + METRIC_FIELDS
    write_csv(
        subject_path,
        build_partition_rows(data, states, "subject_id"),
        partition_fields,
    )
    write_csv(
        activity_path,
        build_partition_rows(data, states, "activity_id"),
        partition_fields,
    )

    frozen_design = {
        "status": "design_frozen_not_validated",
        "dataset": "PPG-DaLiA",
        "input_subjects": "S1-S10",
        "validation_subjects": "S11-S15",
        "validation_data_read": False,
        "window_seconds": 8.0,
        "step_seconds": 2.0,
        "selection_constraints": {
            "minimum_macro_correct_retention_percent": (
                MIN_MACRO_CORRECT_RETENTION_PERCENT
            ),
            "minimum_subject_correct_retention_percent": (
                MIN_SUBJECT_CORRECT_RETENTION_PERCENT
            ),
        },
        "logic": {
            "motion_flag": (
                "motion_rms >= threshold OR "
                "jerk_rms >= threshold"
            ),
            "collision_flag": (
                "acc_power_near_ppg >= threshold OR "
                "harmonic_gap <= threshold"
            ),
            "reject": "motion_flag AND collision_flag",
            "low_confidence": (
                "motion_flag XOR collision_flag"
            ),
            "usable": (
                "NOT motion_flag AND NOT collision_flag"
            ),
        },
        "final_selection": final_selection,
        "family_comparison": family_selections,
        "loso_results": loso_rows,
        "design_metrics": metrics,
    }

    with parameter_path.open("w", encoding="utf-8") as file:
        json.dump(
            frozen_design,
            file,
            ensure_ascii=False,
            indent=2,
        )

    plot_family_comparison(
        family_selections,
        family_plot_path,
    )
    plot_loso_results(loso_rows, loso_plot_path)
    plot_output_tradeoff(metrics, tradeoff_plot_path)

    return [
        window_path,
        family_path,
        loso_path,
        design_metrics_path,
        subject_path,
        activity_path,
        parameter_path,
        family_plot_path,
        loso_plot_path,
        tradeoff_plot_path,
    ]


def main(should_write_outputs: bool = False) -> int:
    print("ACC-aware SQI V1：设计阶段")
    print(f"输入：{INPUT_CSV}")
    print("数据边界：仅 S1-S10；不读取 S11-S15")

    data = load_design_data(INPUT_CSV)
    baseline_usable = data["state"] == "usable"

    print(
        f"设计窗口={len(data['subject_id'])}，"
        f"PPG-only usable={int(np.sum(baseline_usable))}"
    )

    print("\n三类规则的 S1-S10 设计集比较")
    family_selections = [
        select_rule(family, data, baseline_usable)
        for family in RULE_FAMILIES
    ]
    for selection in family_selections:
        print_selection(selection)

    print("\n组合规则 leave-one-subject-out")
    loso_rows = run_loso(data, baseline_usable)
    for row in loso_rows:
        print(
            f"  S{row['held_subject']}: "
            "correct retention="
            f"{row['overall_correct_retention_percent']:.2f}%, "
            "error capture="
            f"{row['overall_error_capture_percent']:.2f}%, "
            "coverage="
            f"{row['retained_coverage_percent']:.2f}%, "
            "remaining >10 bpm="
            f"{row['remaining_failure_percent']:.2f}%"
        )

    print(
        "  LOSO 平均 correct retention="
        f"{np.mean([row['overall_correct_retention_percent'] for row in loso_rows]):.2f}%"
    )
    print(
        "  LOSO 平均 error capture="
        f"{np.mean([row['overall_error_capture_percent'] for row in loso_rows]):.2f}%"
    )

    final_selection = select_rule(
        "motion_and_collision",
        data,
        baseline_usable,
    )
    states, reasons, motion, collision = assign_acc_states(
        data,
        final_selection["parameters"],
    )

    errors = data["absolute_error_bpm"]
    metrics = {
        "ppg_only_usable": summarize_mask(
            errors,
            baseline_usable,
            baseline_usable,
        ),
        "acc_aware_usable": summarize_mask(
            errors,
            baseline_usable,
            states == "usable",
        ),
        "acc_aware_usable_plus_low": summarize_mask(
            errors,
            baseline_usable,
            np.isin(
                states,
                ("usable", "low_confidence"),
            ),
        ),
        "acc_aware_reject": summarize_mask(
            errors,
            baseline_usable,
            states == "reject",
        ),
    }

    # coverage_percent 的分母是 PPG-only usable 窗口；
    # 另存相对于全部 S1-S10 设计窗口的覆盖率，避免汇报歧义。
    for metric_row in metrics.values():
        metric_row["coverage_of_all_design_percent"] = safe_percent(
            int(metric_row["count"]),
            EXPECTED_DESIGN_WINDOWS,
        )

    print("\n冻结前的 S1-S10 设计结果")
    for name, row in metrics.items():
        print_output_metrics(name, row)

    print("\n拟冻结的组合规则参数")
    print_selection(final_selection)

    print(
        "\n注意：以上是设计集和 LOSO 稳定性结果，"
        "不是 S11-S15 验证结果。"
    )
    print(
        "本脚本没有读取验证集，也没有写出参数文件；"
        "需先检查结果是否合理，再冻结并保存。"
    )

    if should_write_outputs:
        output_paths = write_outputs(
            data,
            family_selections,
            loso_rows,
            final_selection,
            states,
            reasons,
            motion,
            collision,
            metrics,
        )
        print("\n已保存冻结设计结果：")
        for output_path in output_paths:
            print(f"  {output_path}")
    else:
        print(
            "本次未写入结果；如需保存冻结设计，"
            "请使用 --write-outputs。"
        )

    return 0


if __name__ == "__main__":
    try:
        arguments = parse_args()
        raise SystemExit(main(arguments.write_outputs))
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
