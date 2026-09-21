#!/usr/bin/env python3
"""建立 PPG-DaLiA 的 ACC 运动特征证据 V0。

研究目标：
1. 检查 PPG-only SQI 已判为 usable、但心率误差仍大于 10 bpm 的窗口；
2. 判断这些剩余错误是否能被 ACC 运动特征解释；
3. 为后续 ACC-aware SQI V1 提供特征依据，但本脚本不设置门限。

数据隔离原则：
- S1-S10：设计集，可以查看 ACC 特征与心率误差的关系；
- S11-S15：验证集，只检查数据长度和时间窗对齐，不计算、不展示
  ACC 特征与心率误差的关系。

本脚本默认只在终端输出统计结果；只有显式传入 --write-outputs
才会保存 CSV、JSON 和图片。
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_s1_motion_artifacts import (
    ACC_FS_HZ,
    ACC_STEP_SAMPLES,
    ACC_WINDOW_SAMPLES,
    MOTION_MAX_HZ,
    MOTION_MIN_HZ,
    PEAK_HALF_WIDTH_HZ,
    dominant_motion_frequency,
    harmonic_frequency_gap,
)
from analyze_s1_ppg_sqi import safe_spearman


DATA_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    DATA_ROOT.parent / "90_共享原始数据" / "ppg_dalia" / "PPG_FieldStudy"
)
INPUT_SQI_CSV = (
    DATA_ROOT
    / "02_结果"
    / "ppg_only_sqi_v0"
    / "S1_S15_ppg_only_sqi_v0_windows.csv"
)
OUTPUT_DIR = DATA_ROOT / "02_结果" / "acc_motion_features_v0"

SUBJECT_IDS = tuple(range(1, 16))
DESIGN_SUBJECTS = frozenset(range(1, 11))
VALIDATION_SUBJECTS = frozenset(range(11, 16))

EXPECTED_TOTAL_WINDOWS = 64_697
EXPECTED_DESIGN_WINDOWS = 43_215
EXPECTED_VALIDATION_WINDOWS = 21_482

WINDOW_SECONDS = 8.0
STEP_SECONDS = 2.0
ERROR_THRESHOLD_BPM = 10.0

REQUIRED_COLUMNS = {
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
}

# 特征本身不使用 ECG。ECG 误差只在设计集用于判断特征是否有区分能力。
FEATURES = (
    ("motion_rms_g", "运动强度 RMS"),
    ("acc_peak_to_peak_g", "ACC 合加速度峰峰值"),
    ("jerk_rms_g_per_s", "Jerk RMS"),
    ("acc_dominant_frequency_hz", "ACC 主频"),
    ("acc_peak_power_ratio", "ACC 主频能量占比"),
    ("acc_power_near_ppg_ratio", "PPG 候选频率附近 ACC 能量占比"),
    ("ppg_acc_fundamental_gap_hz", "PPG 与 ACC 基频距离"),
    ("ppg_acc_second_harmonic_gap_hz", "PPG 与 ACC 二次谐波距离"),
    ("ppg_acc_harmonic_gap_hz", "PPG 与 ACC 1-3 次谐波最小距离"),
)

WINDOW_OUTPUT_FIELDS = (
    "subject_id",
    "split",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "ppg_hr_bpm",
    "ppg_frequency_hz",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "sqi_score",
    "state",
    "reason",
) + tuple(field for field, _ in FEATURES)

ACTIVITY_ENGLISH_NAMES = {
    0: "Transition/other",
    1: "Sitting",
    2: "Stairs",
    3: "Table soccer",
    4: "Cycling",
    5: "Driving",
    6: "Lunch",
    7: "Walking",
    8: "Working",
}

PLOT_FEATURES = (
    ("motion_rms_g", "Motion RMS (g)"),
    ("acc_peak_to_peak_g", "Magnitude peak-to-peak (g)"),
    ("jerk_rms_g_per_s", "Jerk RMS (g/s)"),
    (
        "acc_power_near_ppg_ratio",
        "ACC power near PPG candidate",
    ),
    (
        "ppg_acc_fundamental_gap_hz",
        "PPG-ACC fundamental gap (Hz)",
    ),
    (
        "ppg_acc_harmonic_gap_hz",
        "PPG-ACC harmonic gap (Hz)",
    ),
)


def read_sqi_rows() -> list[dict[str, Any]]:
    """读取已经冻结的 PPG-only SQI 窗口结果。"""
    if not INPUT_SQI_CSV.exists():
        raise FileNotFoundError(f"找不到 SQI V0 窗口文件：{INPUT_SQI_CSV}")

    rows: list[dict[str, Any]] = []

    with INPUT_SQI_CSV.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"SQI CSV 缺少字段：{sorted(missing)}")

        for raw_row in reader:
            rows.append(
                {
                    "subject_id": int(
                        raw_row["subject_id"].removeprefix("S")
                    ),
                    "split": raw_row["split"],
                    "window_start_s": float(raw_row["window_start_s"]),
                    "window_center_s": float(raw_row["window_center_s"]),
                    "window_end_s": float(raw_row["window_end_s"]),
                    "activity_id": int(float(raw_row["activity_id"])),
                    "activity_name": raw_row["activity_name"],
                    "ppg_hr_bpm": float(raw_row["ppg_hr_bpm"]),
                    "ecg_hr_bpm": float(raw_row["ecg_hr_bpm"]),
                    "absolute_error_bpm": float(
                        raw_row["absolute_error_bpm"]
                    ),
                    "sqi_score": float(raw_row["sqi_score"]),
                    "state": raw_row["state"],
                    "reason": raw_row["reason"],
                }
            )

    if len(rows) != EXPECTED_TOTAL_WINDOWS:
        raise ValueError(
            f"窗口总数异常：期望 {EXPECTED_TOTAL_WINDOWS}，实际 {len(rows)}"
        )

    design_count = sum(row["subject_id"] in DESIGN_SUBJECTS for row in rows)
    validation_count = sum(
        row["subject_id"] in VALIDATION_SUBJECTS for row in rows
    )

    if design_count != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            f"设计集窗口数异常：期望 {EXPECTED_DESIGN_WINDOWS}，"
            f"实际 {design_count}"
        )

    if validation_count != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError(
            f"验证集窗口数异常：期望 {EXPECTED_VALIDATION_WINDOWS}，"
            f"实际 {validation_count}"
        )

    return rows


def group_rows_by_subject(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["subject_id"]].append(row)

    for subject_id in SUBJECT_IDS:
        if subject_id not in grouped:
            raise ValueError(f"SQI CSV 中缺少 S{subject_id}")

        grouped[subject_id].sort(key=lambda row: row["window_start_s"])

    return grouped


def load_subject_acc(subject_id: int) -> np.ndarray:
    """从原始 pkl 文件读取腕部三轴 ACC。"""
    data_path = (
        DATASET_DIR
        / f"S{subject_id}"
        / f"S{subject_id}.pkl"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"找不到 S{subject_id} 数据：{data_path}")

    with data_path.open("rb") as file:
        subject = pickle.load(file, encoding="latin1")

    try:
        acc = np.asarray(subject["signal"]["wrist"]["ACC"], dtype=float)
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"S{subject_id} 缺少 signal -> wrist -> ACC"
        ) from error

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(
            f"S{subject_id} ACC 形状异常：期望 (N, 3)，实际 {acc.shape}"
        )

    if not np.all(np.isfinite(acc)):
        raise ValueError(f"S{subject_id} ACC 中存在 NaN 或无穷值")

    return acc


def validate_subject_alignment(
    subject_id: int,
    rows: list[dict[str, Any]],
    acc: np.ndarray,
) -> None:
    """检查 SQI CSV 的时间窗能否与原始 ACC 一一对应。"""
    if len(acc) < ACC_WINDOW_SAMPLES:
        raise ValueError(f"S{subject_id} ACC 长度小于一个 8 秒窗口")

    expected_window_count = (
        1
        + (len(acc) - ACC_WINDOW_SAMPLES)
        // ACC_STEP_SAMPLES
    )

    if len(rows) != expected_window_count:
        raise ValueError(
            f"S{subject_id} 窗口数不一致："
            f"SQI={len(rows)}，ACC 可切分={expected_window_count}"
        )

    expected_start_s = np.arange(len(rows), dtype=float) * STEP_SECONDS
    actual_start_s = np.asarray(
        [row["window_start_s"] for row in rows],
        dtype=float,
    )
    actual_center_s = np.asarray(
        [row["window_center_s"] for row in rows],
        dtype=float,
    )
    actual_end_s = np.asarray(
        [row["window_end_s"] for row in rows],
        dtype=float,
    )

    if not np.allclose(actual_start_s, expected_start_s, atol=1e-6):
        raise ValueError(f"S{subject_id} 窗口起点与 ACC 不对齐")

    if not np.allclose(
        actual_center_s,
        expected_start_s + WINDOW_SECONDS / 2.0,
        atol=1e-6,
    ):
        raise ValueError(f"S{subject_id} 窗口中心与 ACC 不对齐")

    if not np.allclose(
        actual_end_s,
        expected_start_s + WINDOW_SECONDS,
        atol=1e-6,
    ):
        raise ValueError(f"S{subject_id} 窗口终点与 ACC 不对齐")


def calculate_acc_features(
    acc_window: np.ndarray,
    ppg_frequency_hz: float,
) -> dict[str, float]:
    """计算一个 8 秒窗口内的运动特征。"""
    magnitude = np.linalg.norm(acc_window, axis=1)

    # E4 ACC 的单位为 g；静止时模长约为 1 g。
    motion_proxy = magnitude - 1.0
    motion_rms_g = float(np.sqrt(np.mean(motion_proxy**2)))
    acc_peak_to_peak_g = float(np.ptp(magnitude))

    # Jerk 是加速度随时间的变化率，对快速动作和冲击更敏感。
    jerk = np.diff(acc_window, axis=0) * ACC_FS_HZ
    jerk_rms_g_per_s = float(
        np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
    )

    (
        acc_dominant_frequency_hz,
        acc_peak_power_ratio,
        frequencies_hz,
        combined_power,
    ) = dominant_motion_frequency(acc_window)

    motion_band = (
        (frequencies_hz >= MOTION_MIN_HZ)
        & (frequencies_hz <= MOTION_MAX_HZ)
    )
    near_ppg_band = (
        motion_band
        & (
            np.abs(frequencies_hz - ppg_frequency_hz)
            <= PEAK_HALF_WIDTH_HZ
        )
    )

    total_motion_power = float(np.sum(combined_power[motion_band]))
    power_near_ppg = float(np.sum(combined_power[near_ppg_band]))

    if total_motion_power > 0.0:
        acc_power_near_ppg_ratio = power_near_ppg / total_motion_power
    else:
        acc_power_near_ppg_ratio = 0.0

    return {
        "motion_rms_g": motion_rms_g,
        "acc_peak_to_peak_g": acc_peak_to_peak_g,
        "jerk_rms_g_per_s": jerk_rms_g_per_s,
        "acc_dominant_frequency_hz": acc_dominant_frequency_hz,
        "acc_peak_power_ratio": acc_peak_power_ratio,
        "acc_power_near_ppg_ratio": acc_power_near_ppg_ratio,
        "ppg_acc_fundamental_gap_hz": abs(
            ppg_frequency_hz - acc_dominant_frequency_hz
        ),
        "ppg_acc_second_harmonic_gap_hz": abs(
            ppg_frequency_hz - 2.0 * acc_dominant_frequency_hz
        ),
        "ppg_acc_harmonic_gap_hz": harmonic_frequency_gap(
            ppg_frequency_hz,
            acc_dominant_frequency_hz,
        ),
    }


def build_design_records(
    grouped_rows: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """只为 S1-S10 计算特征，S11-S15 只进行对齐检查。"""
    design_records: list[dict[str, Any]] = []

    for subject_id in SUBJECT_IDS:
        rows = grouped_rows[subject_id]
        acc = load_subject_acc(subject_id)
        validate_subject_alignment(subject_id, rows, acc)

        expected_split = (
            "design"
            if subject_id in DESIGN_SUBJECTS
            else "validation"
        )
        actual_splits = {row["split"] for row in rows}

        if actual_splits != {expected_split}:
            raise ValueError(
                f"S{subject_id} split 异常：{sorted(actual_splits)}"
            )

        print(
            f"S{subject_id}: 对齐通过，"
            f"ACC={acc.shape}，窗口数={len(rows)}"
        )

        if subject_id in VALIDATION_SUBJECTS:
            continue

        for window_index, row in enumerate(rows):
            start = window_index * ACC_STEP_SAMPLES
            end = start + ACC_WINDOW_SAMPLES
            acc_window = acc[start:end]

            ppg_frequency_hz = row["ppg_hr_bpm"] / 60.0
            features = calculate_acc_features(
                acc_window,
                ppg_frequency_hz,
            )

            record = dict(row)
            record["ppg_frequency_hz"] = ppg_frequency_hz
            record.update(features)
            design_records.append(record)

    if len(design_records) != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            f"设计集特征数异常：期望 {EXPECTED_DESIGN_WINDOWS}，"
            f"实际 {len(design_records)}"
        )

    return design_records


def values(
    records: list[dict[str, Any]],
    field: str,
) -> np.ndarray:
    result = np.asarray(
        [float(record[field]) for record in records],
        dtype=float,
    )

    if not np.all(np.isfinite(result)):
        raise ValueError(f"字段 {field} 中存在 NaN 或无穷值")

    return result


def format_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def print_correlations(
    title: str,
    records: list[dict[str, Any]],
) -> None:
    errors = values(records, "absolute_error_bpm")

    print(f"\n{title}（n={len(records)}）")
    print("Spearman ρ：正值表示特征越大时，心率误差倾向越大。")

    for field, label in FEATURES:
        correlation = safe_spearman(values(records, field), errors)
        print(f"  {label}: ρ={format_number(correlation)}")


def percentile_summary(
    records: list[dict[str, Any]],
    field: str,
) -> str:
    feature_values = values(records, field)
    p25, median, p75 = np.percentile(feature_values, [25, 50, 75])
    return f"{median:.4f}（P25={p25:.4f}, P75={p75:.4f}）"


def split_usable_records(
    usable_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    good = [
        record
        for record in usable_records
        if record["absolute_error_bpm"] <= ERROR_THRESHOLD_BPM
    ]
    residual_failures = [
        record
        for record in usable_records
        if record["absolute_error_bpm"] > ERROR_THRESHOLD_BPM
    ]
    return good, residual_failures


def print_usable_group_comparison(
    usable_records: list[dict[str, Any]],
) -> None:
    good, residual_failures = split_usable_records(usable_records)

    print("\nPPG-only SQI 已接受窗口的 ACC 特征对照")
    print(
        f"  误差 <= {ERROR_THRESHOLD_BPM:.0f} bpm：{len(good)} 窗"
    )
    print(
        f"  误差 > {ERROR_THRESHOLD_BPM:.0f} bpm："
        f"{len(residual_failures)} 窗"
    )

    for field, label in FEATURES:
        good_summary = percentile_summary(good, field)
        failure_summary = percentile_summary(residual_failures, field)
        print(f"  {label}")
        print(f"    正确组：{good_summary}")
        print(f"    剩余错误组：{failure_summary}")


def print_activity_summary(
    usable_records: list[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in usable_records:
        grouped[record["activity_name"]].append(record)

    print("\nPPG-only SQI 已接受窗口：分活动剩余失败率")

    for activity_name, activity_records in sorted(grouped.items()):
        error = values(activity_records, "absolute_error_bpm")
        failure_rate = float(
            np.mean(error > ERROR_THRESHOLD_BPM) * 100.0
        )

        print(
            f"  {activity_name}: n={len(activity_records)}, "
            f">10 bpm={failure_rate:.2f}%, "
            f"MAE={np.mean(error):.2f} bpm"
        )


def build_correlation_rows(
    design_records: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scope, records in (
        ("all_design_windows", design_records),
        ("ppg_sqi_usable", usable_records),
    ):
        errors = values(records, "absolute_error_bpm")

        for field, label in FEATURES:
            correlation = safe_spearman(values(records, field), errors)
            rows.append(
                {
                    "scope": scope,
                    "window_count": len(records),
                    "feature": field,
                    "feature_label": label,
                    "spearman_rho": correlation,
                }
            )

    return rows


def build_group_comparison_rows(
    usable_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    good, residual_failures = split_usable_records(usable_records)
    rows: list[dict[str, Any]] = []

    for group_name, records in (
        ("error_le_10_bpm", good),
        ("error_gt_10_bpm", residual_failures),
    ):
        for field, label in FEATURES:
            feature_values = values(records, field)
            p25, median, p75 = np.percentile(
                feature_values,
                [25, 50, 75],
            )
            rows.append(
                {
                    "group": group_name,
                    "window_count": len(records),
                    "feature": field,
                    "feature_label": label,
                    "p25": float(p25),
                    "median": float(median),
                    "p75": float(p75),
                }
            )

    return rows


def build_activity_rows(
    usable_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for record in usable_records:
        grouped[record["activity_id"]].append(record)

    rows: list[dict[str, Any]] = []

    for activity_id, records in sorted(grouped.items()):
        errors = values(records, "absolute_error_bpm")
        failure_count = int(np.sum(errors > ERROR_THRESHOLD_BPM))

        rows.append(
            {
                "activity_id": activity_id,
                "activity_name": records[0]["activity_name"],
                "activity_name_en": ACTIVITY_ENGLISH_NAMES.get(
                    activity_id,
                    f"Activity {activity_id}",
                ),
                "window_count": len(records),
                "failure_count": failure_count,
                "failure_rate_percent": float(
                    failure_count / len(records) * 100.0
                ),
                "mae_bpm": float(np.mean(errors)),
            }
        )

    return rows


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_group_comparison(
    usable_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    good, residual_failures = split_usable_records(usable_records)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))

    for axis, (field, title) in zip(axes.flat, PLOT_FEATURES):
        axis.boxplot(
            [
                values(good, field),
                values(residual_failures, field),
            ],
            tick_labels=["Error <=10", "Error >10"],
            showfliers=False,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "ACC features in PPG-only SQI accepted windows (S1-S10)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_activity_failures(
    activity_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    names = [row["activity_name_en"] for row in activity_rows]
    failure_rates = [
        row["failure_rate_percent"]
        for row in activity_rows
    ]

    figure, axis = plt.subplots(figsize=(11, 5.5))
    bars = axis.bar(names, failure_rates, color="#4472C4")

    axis.set_ylabel("Accepted windows with error >10 bpm (%)")
    axis.set_xlabel("Activity")
    axis.set_title(
        "Residual failures after PPG-only SQI (S1-S10)"
    )
    axis.set_ylim(0.0, max(failure_rates) * 1.18)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)

    for bar, rate in zip(bars, failure_rates):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    design_records: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    correlation_rows = build_correlation_rows(
        design_records,
        usable_records,
    )
    group_rows = build_group_comparison_rows(usable_records)
    activity_rows = build_activity_rows(usable_records)
    good, residual_failures = split_usable_records(usable_records)

    window_path = OUTPUT_DIR / "S1_S10_acc_motion_features_v0.csv"
    correlation_path = (
        OUTPUT_DIR / "S1_S10_acc_feature_correlations_v0.csv"
    )
    group_path = (
        OUTPUT_DIR / "S1_S10_usable_group_comparison_v0.csv"
    )
    activity_path = (
        OUTPUT_DIR / "S1_S10_activity_residual_failure_v0.csv"
    )
    summary_path = (
        OUTPUT_DIR / "S1_S10_acc_motion_summary_v0.json"
    )
    group_plot_path = (
        OUTPUT_DIR / "acc_feature_group_comparison.png"
    )
    activity_plot_path = (
        OUTPUT_DIR / "usable_activity_residual_failure.png"
    )

    write_csv(window_path, design_records, WINDOW_OUTPUT_FIELDS)
    write_csv(
        correlation_path,
        correlation_rows,
        (
            "scope",
            "window_count",
            "feature",
            "feature_label",
            "spearman_rho",
        ),
    )
    write_csv(
        group_path,
        group_rows,
        (
            "group",
            "window_count",
            "feature",
            "feature_label",
            "p25",
            "median",
            "p75",
        ),
    )
    write_csv(
        activity_path,
        activity_rows,
        (
            "activity_id",
            "activity_name",
            "activity_name_en",
            "window_count",
            "failure_count",
            "failure_rate_percent",
            "mae_bpm",
        ),
    )

    summary = {
        "dataset": "PPG-DaLiA",
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "design_subjects": "S1-S10",
        "validation_subjects": "S11-S15",
        "validation_usage": "alignment_check_only",
        "design_window_count": len(design_records),
        "ppg_sqi_usable_window_count": len(usable_records),
        "usable_error_le_10_window_count": len(good),
        "usable_error_gt_10_window_count": len(residual_failures),
        "usable_error_gt_10_percent": (
            len(residual_failures) / len(usable_records) * 100.0
        ),
        "correlations": correlation_rows,
        "group_comparison": group_rows,
        "activity_summary": activity_rows,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    plot_group_comparison(usable_records, group_plot_path)
    plot_activity_failures(activity_rows, activity_plot_path)

    return [
        window_path,
        correlation_path,
        group_path,
        activity_path,
        summary_path,
        group_plot_path,
        activity_plot_path,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="建立 S1-S10 ACC 运动特征证据 V0"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="显式保存 S1-S10 的 CSV、JSON 和证据图",
    )
    return parser.parse_args()


def main(should_write_outputs: bool = False) -> int:
    print("ACC 运动特征证据 V0")
    print(f"输入：{INPUT_SQI_CSV}")
    print("窗口：8 秒，步长：2 秒")
    print("S1-S10：设计分析；S11-S15：仅检查对齐")

    rows = read_sqi_rows()
    grouped_rows = group_rows_by_subject(rows)
    design_records = build_design_records(grouped_rows)

    usable_records = [
        record
        for record in design_records
        if record["state"] == "usable"
    ]

    print_correlations(
        "S1-S10 全部窗口：ACC 特征与绝对误差的关系",
        design_records,
    )
    print_correlations(
        "S1-S10 PPG-only SQI 已接受窗口",
        usable_records,
    )
    print_usable_group_comparison(usable_records)
    print_activity_summary(usable_records)

    print(
        "\n验证集保护：S11-S15 只完成长度与时间窗对齐检查，"
        "没有计算或展示 ACC 特征和心率误差的关系。"
    )
    print(
        "本阶段不设置 ACC 门限；需要先根据以上设计集结果，"
        "确定哪些特征真正有区分能力。"
    )

    if should_write_outputs:
        output_paths = write_outputs(
            design_records,
            usable_records,
        )
        print("\n已生成可追溯结果：")
        for output_path in output_paths:
            print(f"  {output_path}")
    else:
        print(
            "\n本次没有写入结果文件；如需保存，请使用 "
            "--write-outputs。"
        )

    return 0


if __name__ == "__main__":
    try:
        arguments = parse_args()
        raise SystemExit(main(arguments.write_outputs))
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
