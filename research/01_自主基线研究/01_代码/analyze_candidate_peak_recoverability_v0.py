#!/usr/bin/env python3
"""分析错误心率窗口的频谱候选峰是否具有理论恢复空间。

本脚本只使用 S1-S10 设计集，不读取 S11-S15。

研究问题：
1. 当前 PPG-only HR 选择错误主峰时，前 5 个频率候选中是否仍有
   接近 ECG 参考心率的候选；
2. ACC-aware SQI 保留和拒绝的错误窗口分别有多少理论恢复空间；
3. 正确候选是否容易与 ACC 基频或谐波发生碰撞。

ECG 只用于分析完成后的 oracle 判断，不参与候选峰生成、排序或
ACC 特征计算。因此本脚本得到的是算法上限，不是可部署算法。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_all_subjects_ppg_sqi import load_subject
from analyze_s1_motion_artifacts import (
    ACC_STEP_SAMPLES,
    ACC_WINDOW_SAMPLES,
    dominant_motion_frequency,
    harmonic_frequency_gap,
)
from analyze_s1_ppg_sqi import (
    SECOND_PEAK_MIN_SEPARATION_HZ,
    ppg_power_spectrum,
    validate_window_counts,
)
from build_acc_motion_features_v0 import (
    load_subject_acc,
    validate_subject_alignment,
)
from ppg_hr_baseline import (
    MAX_HR_HZ,
    MIN_HR_HZ,
    N_FFT,
    PEAK_BAND_HALF_WIDTH_HZ,
    PPG_FS_HZ,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    interpolate_spectral_peak,
)
from validate_acc_aware_sqi_v1 import load_frozen_design


DATA_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    DATA_ROOT
    / "02_结果"
    / "acc_aware_sqi_v1_design"
    / "S1_S10_acc_aware_sqi_v1_windows.csv"
)
OUTPUT_DIR = (
    DATA_ROOT
    / "02_结果"
    / "candidate_peak_recoverability_v0"
)

DESIGN_SUBJECTS = tuple(range(1, 11))
EXPECTED_DESIGN_WINDOWS = 43_215
EXPECTED_PPG_USABLE_FAILURES = 1_284
EXPECTED_RETAINED_FAILURES = 193
EXPECTED_REJECTED_FAILURES = 1_091

WINDOW_SECONDS = 8.0
STEP_SECONDS = 2.0
FAILURE_THRESHOLD_BPM = 10.0
RECOVERY_STRICT_BPM = 3.0
RECOVERY_RELAXED_BPM = 5.0
TOP_K = 5

REQUIRED_COLUMNS = {
    "subject_id",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "ppg_hr_bpm",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "ppg_sqi_state",
    "ppg_sqi_reason",
    "acc_aware_state",
    "acc_aware_reason",
}

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
    "acc_aware_state",
    "acc_aware_reason",
    "acc_dominant_frequency_hz",
    "candidate_count",
    "oracle_best_rank",
    "oracle_best_hr_bpm",
    "oracle_best_absolute_error_bpm",
    "oracle_best_power_ratio",
    "oracle_best_acc_harmonic_gap_hz",
    "oracle_best_acc_collision",
    "top_5_recoverable_le_3_bpm",
    "top_5_recoverable_le_5_bpm",
)

CANDIDATE_OUTPUT_FIELDS = (
    "subject_id",
    "window_start_s",
    "activity_id",
    "activity_name",
    "acc_aware_state",
    "candidate_rank",
    "candidate_frequency_hz",
    "candidate_hr_bpm",
    "candidate_power_ratio",
    "candidate_absolute_error_bpm",
    "candidate_acc_harmonic_gap_hz",
    "candidate_acc_collision",
    "is_oracle_best",
    "within_3_bpm",
    "within_5_bpm",
)

SUMMARY_FIELDS = (
    "scope",
    "window_count",
    "mean_candidate_count",
    "top_5_recoverable_le_3_bpm_count",
    "top_5_recoverable_le_3_bpm_percent",
    "top_5_recoverable_le_5_bpm_count",
    "top_5_recoverable_le_5_bpm_percent",
    "oracle_best_mae_bpm",
    "oracle_best_p90_bpm",
    "median_oracle_best_rank",
    "oracle_best_acc_collision_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 PPG 错误窗口的候选峰可恢复上限"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="显式保存窗口、候选峰和汇总结果",
    )
    return parser.parse_args()


def safe_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def load_design_rows() -> list[dict[str, Any]]:
    """读取并验证 S1-S10 冻结设计窗口。"""
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV}")

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"输入 CSV 缺少字段：{sorted(missing)}"
            )

        rows = [
            {
                "subject_id": int(row["subject_id"]),
                "window_start_s": float(row["window_start_s"]),
                "window_center_s": float(row["window_center_s"]),
                "window_end_s": float(row["window_end_s"]),
                "activity_id": int(row["activity_id"]),
                "activity_name": row["activity_name"],
                "ppg_hr_bpm": float(row["ppg_hr_bpm"]),
                "ecg_hr_bpm": float(row["ecg_hr_bpm"]),
                "absolute_error_bpm": float(
                    row["absolute_error_bpm"]
                ),
                "ppg_sqi_state": row["ppg_sqi_state"],
                "ppg_sqi_reason": row["ppg_sqi_reason"],
                "acc_aware_state": row["acc_aware_state"],
                "acc_aware_reason": row["acc_aware_reason"],
            }
            for row in reader
        ]

    if len(rows) != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            f"设计窗口数异常：期望 {EXPECTED_DESIGN_WINDOWS}，"
            f"实际 {len(rows)}"
        )

    observed_subjects = {
        row["subject_id"]
        for row in rows
    }
    if observed_subjects != set(DESIGN_SUBJECTS):
        raise ValueError(
            f"设计受试者异常：{sorted(observed_subjects)}"
        )

    failures = [
        row
        for row in rows
        if (
            row["ppg_sqi_state"] == "usable"
            and row["absolute_error_bpm"]
            > FAILURE_THRESHOLD_BPM
        )
    ]
    retained = [
        row
        for row in failures
        if row["acc_aware_state"]
        in {"usable", "low_confidence"}
    ]
    rejected = [
        row
        for row in failures
        if row["acc_aware_state"] == "reject"
    ]

    expected_counts = (
        (
            "PPG-only usable 大误差",
            len(failures),
            EXPECTED_PPG_USABLE_FAILURES,
        ),
        (
            "ACC-aware 保留大误差",
            len(retained),
            EXPECTED_RETAINED_FAILURES,
        ),
        (
            "ACC-aware 拒绝大误差",
            len(rejected),
            EXPECTED_REJECTED_FAILURES,
        ),
    )
    for name, actual, expected in expected_counts:
        if actual != expected:
            raise ValueError(
                f"{name}窗口数异常："
                f"期望 {expected}，实际 {actual}"
            )

    rows.sort(
        key=lambda row: (
            row["subject_id"],
            row["window_start_s"],
        )
    )
    return rows


def extract_frequency_candidates(
    ppg_window: np.ndarray,
) -> list[dict[str, float | int]]:
    """按功率排序提取相互至少间隔 0.20 Hz 的候选峰。"""
    frequency, power = ppg_power_spectrum(ppg_window)
    candidate_mask = (
        (frequency >= MIN_HR_HZ)
        & (frequency <= MAX_HR_HZ)
    )
    candidate_indices = np.flatnonzero(candidate_mask)

    if len(candidate_indices) == 0:
        raise ValueError("生理心率频段中没有频点")

    total_power = float(np.sum(power[candidate_mask]))
    if total_power <= 0.0:
        raise ValueError("生理心率频段总功率不大于零")

    order = np.argsort(
        -power[candidate_indices],
        kind="mergesort",
    )
    ranked_indices = candidate_indices[order]
    selected: list[dict[str, float | int]] = []

    for peak_index in ranked_indices:
        interpolated_index = interpolate_spectral_peak(
            power,
            int(peak_index),
        )
        peak_frequency_hz = float(
            np.clip(
                interpolated_index * PPG_FS_HZ / N_FFT,
                MIN_HR_HZ,
                MAX_HR_HZ,
            )
        )

        if any(
            abs(
                peak_frequency_hz
                - float(candidate["candidate_frequency_hz"])
            )
            < SECOND_PEAK_MIN_SEPARATION_HZ
            for candidate in selected
        ):
            continue

        peak_band = candidate_mask & (
            np.abs(frequency - peak_frequency_hz)
            <= PEAK_BAND_HALF_WIDTH_HZ
        )
        selected.append(
            {
                "candidate_rank": len(selected) + 1,
                "candidate_frequency_hz": peak_frequency_hz,
                "candidate_hr_bpm": peak_frequency_hz * 60.0,
                "candidate_power_ratio": (
                    float(np.sum(power[peak_band]))
                    / total_power
                ),
            }
        )

        if len(selected) == TOP_K:
            break

    if not selected:
        raise ValueError("未能生成频率候选")

    return selected


def analyze_target_window(
    row: dict[str, Any],
    ppg: np.ndarray,
    acc: np.ndarray,
    harmonic_gap_threshold_hz: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """分析一个 PPG-only usable 但误差大于 10 bpm 的窗口。"""
    window_index = int(
        round(row["window_start_s"] / STEP_SECONDS)
    )
    expected_start_s = window_index * STEP_SECONDS
    if not np.isclose(
        row["window_start_s"],
        expected_start_s,
        atol=1e-9,
    ):
        raise ValueError("窗口起点不能映射到 2 秒步长")

    ppg_start = window_index * STEP_SAMPLES
    acc_start = window_index * ACC_STEP_SAMPLES
    ppg_window = ppg[
        ppg_start : ppg_start + WINDOW_SAMPLES
    ]
    acc_window = acc[
        acc_start : acc_start + ACC_WINDOW_SAMPLES
    ]

    candidates = extract_frequency_candidates(ppg_window)
    rank_one_hr = float(candidates[0]["candidate_hr_bpm"])
    if not np.isclose(
        rank_one_hr,
        row["ppg_hr_bpm"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "候选峰第 1 名未复现原 PPG-only HR："
            f"当前={rank_one_hr}，原结果={row['ppg_hr_bpm']}"
        )

    (
        acc_dominant_frequency_hz,
        _,
        _,
        _,
    ) = dominant_motion_frequency(acc_window)

    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_hr = float(candidate["candidate_hr_bpm"])
        candidate_frequency = float(
            candidate["candidate_frequency_hz"]
        )
        candidate_error = abs(
            candidate_hr - row["ecg_hr_bpm"]
        )
        acc_gap = harmonic_frequency_gap(
            candidate_frequency,
            acc_dominant_frequency_hz,
        )

        candidate_rows.append(
            {
                "subject_id": row["subject_id"],
                "window_start_s": row["window_start_s"],
                "activity_id": row["activity_id"],
                "activity_name": row["activity_name"],
                "acc_aware_state": row["acc_aware_state"],
                **candidate,
                "candidate_absolute_error_bpm": candidate_error,
                "candidate_acc_harmonic_gap_hz": acc_gap,
                "candidate_acc_collision": (
                    acc_gap <= harmonic_gap_threshold_hz
                ),
                "within_3_bpm": (
                    candidate_error <= RECOVERY_STRICT_BPM
                ),
                "within_5_bpm": (
                    candidate_error <= RECOVERY_RELAXED_BPM
                ),
            }
        )

    oracle_best = min(
        candidate_rows,
        key=lambda candidate: (
            candidate["candidate_absolute_error_bpm"],
            candidate["candidate_rank"],
        ),
    )
    for candidate in candidate_rows:
        candidate["is_oracle_best"] = (
            candidate["candidate_rank"]
            == oracle_best["candidate_rank"]
        )

    window_result = {
        **row,
        "acc_dominant_frequency_hz": (
            acc_dominant_frequency_hz
        ),
        "candidate_count": len(candidate_rows),
        "oracle_best_rank": oracle_best["candidate_rank"],
        "oracle_best_hr_bpm": oracle_best["candidate_hr_bpm"],
        "oracle_best_absolute_error_bpm": (
            oracle_best["candidate_absolute_error_bpm"]
        ),
        "oracle_best_power_ratio": (
            oracle_best["candidate_power_ratio"]
        ),
        "oracle_best_acc_harmonic_gap_hz": (
            oracle_best["candidate_acc_harmonic_gap_hz"]
        ),
        "oracle_best_acc_collision": (
            oracle_best["candidate_acc_collision"]
        ),
        "top_5_recoverable_le_3_bpm": any(
            candidate["within_3_bpm"]
            for candidate in candidate_rows
        ),
        "top_5_recoverable_le_5_bpm": any(
            candidate["within_5_bpm"]
            for candidate in candidate_rows
        ),
    }
    return window_result, candidate_rows


def build_results(
    rows: list[dict[str, Any]],
    harmonic_gap_threshold_hz: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """只处理 S1-S10 中 PPG-only usable 的大误差窗口。"""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)

    window_results: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []

    for subject_id in DESIGN_SUBJECTS:
        subject_rows = grouped[subject_id]
        subject_rows.sort(
            key=lambda row: row["window_start_s"]
        )

        ppg, reference_hr, activity = load_subject(
            f"S{subject_id}"
        )
        acc = load_subject_acc(subject_id)
        window_count = validate_window_counts(
            ppg,
            reference_hr,
            activity,
        )
        validate_subject_alignment(
            subject_id,
            subject_rows,
            acc,
        )

        if window_count != len(subject_rows):
            raise ValueError(
                f"S{subject_id} 原始数据与设计表窗口数不一致"
            )

        csv_reference = np.asarray(
            [row["ecg_hr_bpm"] for row in subject_rows],
            dtype=np.float64,
        )
        if not np.allclose(
            reference_hr,
            csv_reference,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"S{subject_id} ECG 标签与设计表不一致"
            )

        target_rows = [
            row
            for row in subject_rows
            if (
                row["ppg_sqi_state"] == "usable"
                and row["absolute_error_bpm"]
                > FAILURE_THRESHOLD_BPM
            )
        ]

        for row in target_rows:
            window_result, candidate_rows = (
                analyze_target_window(
                    row,
                    ppg,
                    acc,
                    harmonic_gap_threshold_hz,
                )
            )
            window_results.append(window_result)
            candidate_results.extend(candidate_rows)

        print(
            f"  S{subject_id}: 对齐通过，"
            f"分析大误差窗口={len(target_rows)}"
        )

    if len(window_results) != EXPECTED_PPG_USABLE_FAILURES:
        raise ValueError(
            "候选峰分析窗口数与回归基准不一致"
        )

    return window_results, candidate_results


def select_scope(
    records: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    if scope == "all_ppg_usable_failures":
        return records
    if scope == "acc_aware_retained_failures":
        return [
            row
            for row in records
            if row["acc_aware_state"]
            in {"usable", "low_confidence"}
        ]
    if scope == "acc_aware_rejected_failures":
        return [
            row
            for row in records
            if row["acc_aware_state"] == "reject"
        ]
    raise ValueError(f"未知分析范围：{scope}")


def summarize_scope(
    records: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    selected = select_scope(records, scope)
    count = len(selected)
    if count == 0:
        return {
            "scope": scope,
            "window_count": 0,
            "mean_candidate_count": None,
            "top_5_recoverable_le_3_bpm_count": 0,
            "top_5_recoverable_le_3_bpm_percent": 0.0,
            "top_5_recoverable_le_5_bpm_count": 0,
            "top_5_recoverable_le_5_bpm_percent": 0.0,
            "oracle_best_mae_bpm": None,
            "oracle_best_p90_bpm": None,
            "median_oracle_best_rank": None,
            "oracle_best_acc_collision_percent": None,
        }

    strict_count = sum(
        row["top_5_recoverable_le_3_bpm"]
        for row in selected
    )
    relaxed_count = sum(
        row["top_5_recoverable_le_5_bpm"]
        for row in selected
    )
    oracle_errors = np.asarray(
        [
            row["oracle_best_absolute_error_bpm"]
            for row in selected
        ],
        dtype=np.float64,
    )
    oracle_ranks = np.asarray(
        [row["oracle_best_rank"] for row in selected],
        dtype=np.float64,
    )

    return {
        "scope": scope,
        "window_count": count,
        "mean_candidate_count": float(
            np.mean(
                [
                    row["candidate_count"]
                    for row in selected
                ]
            )
        ),
        "top_5_recoverable_le_3_bpm_count": strict_count,
        "top_5_recoverable_le_3_bpm_percent": safe_percent(
            strict_count,
            count,
        ),
        "top_5_recoverable_le_5_bpm_count": relaxed_count,
        "top_5_recoverable_le_5_bpm_percent": safe_percent(
            relaxed_count,
            count,
        ),
        "oracle_best_mae_bpm": float(np.mean(oracle_errors)),
        "oracle_best_p90_bpm": float(
            np.percentile(oracle_errors, 90)
        ),
        "median_oracle_best_rank": float(
            np.median(oracle_ranks)
        ),
        "oracle_best_acc_collision_percent": float(
            100.0
            * np.mean(
                [
                    row["oracle_best_acc_collision"]
                    for row in selected
                ]
            )
        ),
    }


def build_group_summary(
    records: list[dict[str, Any]],
    group_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = (
        "all_ppg_usable_failures",
        "acc_aware_retained_failures",
        "acc_aware_rejected_failures",
    )

    for scope in scopes:
        scoped = select_scope(records, scope)
        group_values = sorted(
            {
                row[group_field]
                for row in scoped
            }
        )

        for group_value in group_values:
            group_records = [
                row
                for row in scoped
                if row[group_field] == group_value
            ]
            summary = summarize_scope(
                group_records,
                "all_ppg_usable_failures",
            )
            summary["scope"] = scope
            summary["group_id"] = group_value

            if group_field == "subject_id":
                summary["group_name"] = f"S{group_value}"
            else:
                summary["group_name"] = (
                    group_records[0]["activity_name"]
                )

            rows.append(summary)

    return rows


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    window_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    scope_summary: list[dict[str, Any]],
    subject_summary: list[dict[str, Any]],
    activity_summary: list[dict[str, Any]],
    frozen_hash: str,
    harmonic_gap_threshold_hz: float,
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_path = OUTPUT_DIR / "failure_window_results.csv"
    candidate_path = OUTPUT_DIR / "candidate_peak_results.csv"
    scope_path = OUTPUT_DIR / "scope_summary.csv"
    subject_path = OUTPUT_DIR / "subject_summary.csv"
    activity_path = OUTPUT_DIR / "activity_summary.csv"
    summary_path = (
        OUTPUT_DIR
        / "candidate_peak_recoverability_v0.json"
    )

    write_csv(
        window_path,
        window_results,
        WINDOW_OUTPUT_FIELDS,
    )
    write_csv(
        candidate_path,
        candidate_results,
        CANDIDATE_OUTPUT_FIELDS,
    )
    write_csv(
        scope_path,
        scope_summary,
        SUMMARY_FIELDS,
    )

    group_fields = (
        "group_id",
        "group_name",
        *SUMMARY_FIELDS,
    )
    write_csv(subject_path, subject_summary, group_fields)
    write_csv(activity_path, activity_summary, group_fields)

    summary_document = {
        "status": (
            "oracle_analysis_complete_not_selection_algorithm"
        ),
        "dataset": "PPG-DaLiA",
        "subjects": "S1-S10",
        "validation_subjects_read": False,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "ppg_sampling_rate_hz": PPG_FS_HZ,
        "heart_rate_search_bpm": [
            MIN_HR_HZ * 60.0,
            MAX_HR_HZ * 60.0,
        ],
        "n_fft": N_FFT,
        "top_k": TOP_K,
        "minimum_candidate_separation_hz": (
            SECOND_PEAK_MIN_SEPARATION_HZ
        ),
        "recovery_tolerance_bpm": {
            "strict": RECOVERY_STRICT_BPM,
            "relaxed": RECOVERY_RELAXED_BPM,
        },
        "frozen_acc_aware_design_sha256": frozen_hash,
        "frozen_harmonic_gap_threshold_hz": (
            harmonic_gap_threshold_hz
        ),
        "candidate_generation_uses_ecg": False,
        "oracle_evaluation_uses_ecg": True,
        "scope_summary": scope_summary,
        "limitations": [
            (
                "Oracle 最优候选使用 ECG 选择，"
                "不能作为在线算法输入"
            ),
            (
                "候选中存在正确峰只代表理论恢复上限，"
                "不代表已有可部署选择规则"
            ),
            (
                "数据为 E4 导出的单通道 BVP，"
                "不等同于未来芯片原始 ADC"
            ),
        ],
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary_document,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return [
        window_path,
        candidate_path,
        scope_path,
        subject_path,
        activity_path,
        summary_path,
    ]


def print_summary(
    scope_summary: list[dict[str, Any]],
) -> None:
    print("\n候选峰 oracle 可恢复性")
    for row in scope_summary:
        print(
            f"  {row['scope']}: "
            f"n={row['window_count']}，"
            "Top-5 ≤3 bpm="
            f"{row['top_5_recoverable_le_3_bpm_percent']:.2f}% ，"
            "Top-5 ≤5 bpm="
            f"{row['top_5_recoverable_le_5_bpm_percent']:.2f}% ，"
            "oracle MAE="
            f"{row['oracle_best_mae_bpm']:.2f} bpm，"
            "oracle P90="
            f"{row['oracle_best_p90_bpm']:.2f} bpm"
        )


def main(should_write_outputs: bool = False) -> int:
    print("候选峰可恢复性 V0")
    print("数据边界：只读取 S1-S10，不读取 S11-S15")
    print(
        "ECG 只用于 oracle 评价，"
        "不参与候选峰生成与排序"
    )

    frozen_design, parameters, frozen_hash = (
        load_frozen_design()
    )
    del frozen_design
    harmonic_gap_threshold_hz = parameters[
        "harmonic_gap_threshold_hz"
    ]

    rows = load_design_rows()
    window_results, candidate_results = build_results(
        rows,
        harmonic_gap_threshold_hz,
    )

    scopes = (
        "all_ppg_usable_failures",
        "acc_aware_retained_failures",
        "acc_aware_rejected_failures",
    )
    scope_summary = [
        summarize_scope(window_results, scope)
        for scope in scopes
    ]
    subject_summary = build_group_summary(
        window_results,
        "subject_id",
    )
    activity_summary = build_group_summary(
        window_results,
        "activity_id",
    )
    print_summary(scope_summary)

    if should_write_outputs:
        output_paths = write_outputs(
            window_results,
            candidate_results,
            scope_summary,
            subject_summary,
            activity_summary,
            frozen_hash,
            harmonic_gap_threshold_hz,
        )
        print("\n已保存候选峰分析结果：")
        for output_path in output_paths:
            print(f"  {output_path}")
    else:
        print(
            "\n本次未写入结果；确认分析逻辑后，"
            "使用 --write-outputs 保存。"
        )

    return 0


if __name__ == "__main__":
    try:
        arguments = parse_args()
        raise SystemExit(main(arguments.write_outputs))
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
