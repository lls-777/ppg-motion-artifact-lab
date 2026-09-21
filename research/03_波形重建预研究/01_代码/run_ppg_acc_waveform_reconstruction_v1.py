#!/usr/bin/env python3
"""PPG-DaLiA 上的连续门控 PPG 波形增强预研究 V1。

V1 在 V0 的基础上验证三个改动：
1. NLMS 在完整 30 秒片段上连续运行，不再每个重叠 8 秒窗口重新初始化；
2. 同时使用 SMoLK 伪影分数和 ACC 动态幅度控制自适应与伪影扣除；
3. 只使用 S1 选择固定参数，然后在 S14、S15 上进行探索性评估。

如果 S1 上所有主动增强候选都不如处理后 PPG，程序会选择
``no_reconstruction``，避免为了展示波形而强行修改信号。

Empatica E4 BVP 不是 LED/PD 原始 ADC，PPG-DaLiA 也没有运动状态下的
同步干净 PPG。因此，本实验评估的是“心率导向的波形增强候选”，不能证明
已经恢复出真实、无伪影的生理波形。

默认只计算并打印摘要。只有显式传入 ``--write-outputs`` 才保存结果。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

import run_ppg_acc_waveform_reconstruction_v0 as v0


SEGMENT_MAPPING_PATH = (
    v0.SMOLK_ROOT / "outputs/dalia_mapping/dalia_segment_mapping.csv"
)
OUTPUT_DIR = (
    v0.DATA_ROOT / "02_结果/V1"
)

DESIGN_SUBJECT = 1
EVALUATION_SUBJECTS = (14, 15)
ALL_SUBJECTS = (DESIGN_SUBJECT,) + EVALUATION_SUBJECTS

EXPECTED_SEGMENT_COUNTS = {1: 307, 14: 298, 15: 264}
EXPECTED_WINDOW_COUNTS = {1: 3684, 14: 3576, 15: 3168}

SEGMENT_SECONDS = 30
SEGMENT_SAMPLES = SEGMENT_SECONDS * v0.PPG_FS_HZ
ACC_SEGMENT_SAMPLES = SEGMENT_SECONDS * v0.ACC_FS_HZ

ACC_GATE_LOW_G = 0.02
ACC_GATE_HIGH_G = 0.12
SMOLK_GATE_START = 0.5
NLMS_EPSILON = 1e-6

METHOD_ORDER = (
    "processed_ppg",
    "v0_window_reset_guided",
    "v1_continuous_gated",
)
METHOD_LABELS = {
    "processed_ppg": "Published processed PPG",
    "v0_window_reset_guided": "V0 window-reset guided NLMS",
    "v1_continuous_gated": "V1 continuous dual-gated NLMS",
}

# 参数候选在读取 S14、S15 结果前固定。S1 只用于选择，不进入最终汇总。
CANDIDATES = (
    {
        "name": "no_reconstruction",
        "taps": 0,
        "step_size": 0.0,
    },
    {
        "name": "gated_t8_mu0.005",
        "taps": 8,
        "step_size": 0.005,
    },
    {
        "name": "gated_t8_mu0.020",
        "taps": 8,
        "step_size": 0.020,
    },
    {
        "name": "gated_t16_mu0.005",
        "taps": 16,
        "step_size": 0.005,
    },
    {
        "name": "gated_t16_mu0.020",
        "taps": 16,
        "step_size": 0.020,
    },
)

WINDOW_FIELDS = (
    "subject_id",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "method",
    "reference_hr_bpm",
    "peak_hr_bpm",
    "peak_absolute_error_bpm",
    "peak_output_valid",
    "spectral_hr_bpm",
    "spectral_absolute_error_bpm",
    "detected_peak_count",
    "valid_ibi_count",
    "smolk_probability_mean",
    "acc_gate_mean",
    "dual_gate_mean",
    "dual_gate_active_fraction",
    "all_region_mean_absolute_change_std",
    "clean_region_mean_absolute_change_std",
)

METRIC_FIELDS = (
    "scope",
    "method",
    "estimator",
    "window_count",
    "valid_count",
    "coverage_percent",
    "mae_bpm",
    "rmse_bpm",
    "median_absolute_error_bpm",
    "p90_absolute_error_bpm",
    "failure_gt_10_bpm_percent",
)

WAVEFORM_FIELDS = (
    "scope",
    "method",
    "window_count",
    "mean_all_region_change_std",
    "mean_clean_region_change_std",
    "p90_all_region_change_std",
    "mean_dual_gate_active_fraction",
)

CANDIDATE_FIELDS = (
    "candidate",
    "taps",
    "step_size",
    "window_count",
    "spectral_mae_bpm",
    "spectral_rmse_bpm",
    "failure_gt_10_bpm_percent",
    "mean_clean_region_change_std",
    "selected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 PPG+ACC 连续门控波形增强预研究 V1。"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="保存 CSV、JSON 和 PNG；默认只打印结果。",
    )
    return parser.parse_args()


def load_segment_mapping() -> list[dict[str, int | float]]:
    if not SEGMENT_MAPPING_PATH.is_file():
        raise FileNotFoundError(
            f"缺少 30 秒片段映射：{SEGMENT_MAPPING_PATH}"
        )

    rows: list[dict[str, int | float]] = []
    with SEGMENT_MAPPING_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        for raw in csv.DictReader(stream):
            subject_id = int(raw["subject_id"])
            if raw["split"] != "test" or subject_id not in ALL_SUBJECTS:
                continue
            rows.append(
                {
                    "processed_row": int(raw["processed_row"]),
                    "subject_id": subject_id,
                    "chunk_index": int(raw["chunk_index"]),
                    "start_sample": int(raw["start_sample"]),
                    "end_sample": int(raw["end_sample"]),
                    "start_s": float(raw["start_s"]),
                    "end_s": float(raw["end_s"]),
                }
            )

    rows.sort(key=lambda row: int(row["processed_row"]))
    counts = {
        subject_id: sum(
            int(row["subject_id"]) == subject_id for row in rows
        )
        for subject_id in ALL_SUBJECTS
    }
    if counts != EXPECTED_SEGMENT_COUNTS:
        raise ValueError(f"30 秒测试片段数量异常：{counts}")

    processed_rows = [int(row["processed_row"]) for row in rows]
    if processed_rows != list(range(sum(EXPECTED_SEGMENT_COUNTS.values()))):
        raise ValueError("测试片段行号不连续或存在重复")
    return rows


def load_hr_mapping() -> list[dict[str, int | float]]:
    if not v0.MAPPING_PATH.is_file():
        raise FileNotFoundError(f"缺少 HR 窗口映射：{v0.MAPPING_PATH}")

    rows: list[dict[str, int | float]] = []
    with v0.MAPPING_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        for raw in csv.DictReader(stream):
            subject_id = int(raw["subject_id"])
            if (
                subject_id not in ALL_SUBJECTS
                or raw["mapping_status"] != "contained"
            ):
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "hr_window_index": int(raw["hr_window_index"]),
                    "window_start_sample": int(raw["window_start_sample"]),
                    "window_end_sample": int(raw["window_end_sample"]),
                    "window_start_s": float(raw["window_start_s"]),
                    "window_center_s": float(raw["window_center_s"]),
                    "window_end_s": float(raw["window_end_s"]),
                    "processed_test_row": int(raw["processed_test_row"]),
                    "segment_offset_start_sample": int(
                        raw["segment_offset_start_sample"]
                    ),
                }
            )

    rows.sort(key=lambda row: (
        int(row["subject_id"]),
        int(row["hr_window_index"]),
    ))
    counts = {
        subject_id: sum(
            int(row["subject_id"]) == subject_id for row in rows
        )
        for subject_id in ALL_SUBJECTS
    }
    if counts != EXPECTED_WINDOW_COUNTS:
        raise ValueError(f"8 秒窗口数量异常：{counts}")
    return rows


def build_segment_references(acc_64hz: np.ndarray) -> np.ndarray:
    if acc_64hz.shape != (SEGMENT_SAMPLES, 3):
        raise ValueError(f"30 秒 ACC 形状异常：{acc_64hz.shape}")

    magnitude = np.linalg.norm(acc_64hz, axis=1, keepdims=True)
    references = np.column_stack((acc_64hz, magnitude))
    references = signal.detrend(references, axis=0, type="linear")
    scales = np.std(references, axis=0)
    usable = scales >= 1e-10
    references[:, usable] /= scales[usable]
    references[:, ~usable] = 0.0
    return references


def calculate_gates(
    acc_64hz: np.ndarray,
    smolk_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据物理运动幅度和伪影分数构造两个门及其乘积。"""
    dynamic_acc = signal.detrend(acc_64hz, axis=0, type="linear")
    motion_amplitude_g = np.linalg.norm(dynamic_acc, axis=1)
    acc_gate = np.clip(
        (motion_amplitude_g - ACC_GATE_LOW_G)
        / (ACC_GATE_HIGH_G - ACC_GATE_LOW_G),
        0.0,
        1.0,
    )
    smolk_gate = np.clip(
        (smolk_score - SMOLK_GATE_START)
        / (1.0 - SMOLK_GATE_START),
        0.0,
        1.0,
    )
    dual_gate = acc_gate * smolk_gate
    return acc_gate, smolk_gate, dual_gate


def load_segment_contexts(
    subject_id: int,
    subject: dict[str, np.ndarray],
    processed_test: np.ndarray,
    smooth_probability: np.ndarray,
    segment_rows: list[dict[str, int | float]],
) -> dict[int, dict[str, np.ndarray]]:
    contexts: dict[int, dict[str, np.ndarray]] = {}
    for row in segment_rows:
        if int(row["subject_id"]) != subject_id:
            continue

        processed_row = int(row["processed_row"])
        processed = v0.standardize(processed_test[processed_row])
        smolk_score = smooth_probability[processed_row]

        acc_start = int(round(float(row["start_s"]) * v0.ACC_FS_HZ))
        acc_end = acc_start + ACC_SEGMENT_SAMPLES
        acc_window = subject["acc"][acc_start:acc_end]
        if acc_window.shape != (ACC_SEGMENT_SAMPLES, 3):
            raise ValueError(
                f"S{subject_id} 第 {processed_row} 行 ACC 越界"
            )
        acc_64hz = signal.resample_poly(acc_window, 2, 1, axis=0)
        if acc_64hz.shape != (SEGMENT_SAMPLES, 3):
            raise ValueError(
                f"S{subject_id} 第 {processed_row} 行 ACC 重采样异常"
            )

        references = build_segment_references(acc_64hz)
        acc_gate, smolk_gate, dual_gate = calculate_gates(
            acc_64hz,
            smolk_score,
        )
        contexts[processed_row] = {
            "processed": processed,
            "smolk_score": smolk_score,
            "acc_64hz": acc_64hz,
            "references": references,
            "acc_gate": acc_gate,
            "smolk_gate": smolk_gate,
            "dual_gate": dual_gate,
        }
    return contexts


def gated_nlms_cancel(
    ppg: np.ndarray,
    references: np.ndarray,
    gate: np.ndarray,
    taps: int,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """连续估计运动分量，只在双门打开的位置学习和扣除。"""
    if ppg.shape != (SEGMENT_SAMPLES,):
        raise ValueError(f"PPG 片段形状异常：{ppg.shape}")
    if references.shape[0] != SEGMENT_SAMPLES:
        raise ValueError("PPG 与 ACC 参考信号长度不一致")
    if gate.shape != (SEGMENT_SAMPLES,):
        raise ValueError(f"门控信号形状异常：{gate.shape}")

    weights = np.zeros(references.shape[1] * taps, dtype=np.float64)
    history = np.zeros((taps, references.shape[1]), dtype=np.float64)
    reconstructed = np.empty(SEGMENT_SAMPLES, dtype=np.float64)
    removed_motion = np.empty(SEGMENT_SAMPLES, dtype=np.float64)

    for index in range(SEGMENT_SAMPLES):
        history[1:] = history[:-1]
        history[0] = references[index]
        vector = history.reshape(-1)
        prediction = float(np.dot(weights, vector))
        error = float(ppg[index] - prediction)
        power = float(np.dot(vector, vector))

        weights += (
            gate[index]
            * step_size
            * error
            * vector
            / (NLMS_EPSILON + power)
        )
        removed_motion[index] = gate[index] * prediction
        reconstructed[index] = ppg[index] - removed_motion[index]

    if not np.all(np.isfinite(reconstructed)):
        raise FloatingPointError("连续门控 NLMS 输出出现非有限值")
    return reconstructed, removed_motion


def apply_candidate(
    context: dict[str, np.ndarray],
    candidate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if candidate["name"] == "no_reconstruction":
        return (
            context["processed"].copy(),
            np.zeros(SEGMENT_SAMPLES, dtype=np.float64),
        )
    return gated_nlms_cancel(
        context["processed"],
        context["references"],
        context["dual_gate"],
        int(candidate["taps"]),
        float(candidate["step_size"]),
    )


def score_candidate(
    candidate: dict[str, Any],
    contexts: dict[int, dict[str, np.ndarray]],
    subject: dict[str, np.ndarray],
    hr_rows: list[dict[str, int | float]],
) -> dict[str, Any]:
    reconstructed = {
        processed_row: apply_candidate(context, candidate)[0]
        for processed_row, context in contexts.items()
    }
    errors: list[float] = []
    clean_changes: list[float] = []

    for row in hr_rows:
        if int(row["subject_id"]) != DESIGN_SUBJECT:
            continue
        processed_row = int(row["processed_test_row"])
        offset = int(row["segment_offset_start_sample"])
        window_slice = slice(offset, offset + v0.WINDOW_SAMPLES)
        waveform = reconstructed[processed_row][window_slice]
        reference_hr = float(
            subject["reference_hr"][int(row["hr_window_index"])]
        )
        spectral_hr, _, _ = v0.estimate_window_hr(waveform)
        errors.append(abs(float(spectral_hr) - reference_hr))

        context = contexts[processed_row]
        difference = np.abs(
            waveform - context["processed"][window_slice]
        )
        clean = (
            context["smolk_score"][window_slice] <= SMOLK_GATE_START
        )
        if np.any(clean):
            clean_changes.append(float(np.mean(difference[clean])))

    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "candidate": candidate["name"],
        "taps": int(candidate["taps"]),
        "step_size": float(candidate["step_size"]),
        "window_count": len(errors),
        "spectral_mae_bpm": float(np.mean(error_array)),
        "spectral_rmse_bpm": float(
            np.sqrt(np.mean(error_array**2))
        ),
        "failure_gt_10_bpm_percent": float(
            100.0 * np.mean(error_array > v0.ERROR_THRESHOLD_BPM)
        ),
        "mean_clean_region_change_std": (
            float(np.mean(clean_changes)) if clean_changes else 0.0
        ),
        "selected": False,
    }


def select_candidate(
    candidate_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = next(
        row
        for row in candidate_scores
        if row["candidate"] == "no_reconstruction"
    )
    active = [
        row
        for row in candidate_scores
        if row["candidate"] != "no_reconstruction"
    ]
    best_active = min(
        active,
        key=lambda row: (
            row["spectral_mae_bpm"],
            row["mean_clean_region_change_std"],
            row["candidate"],
        ),
    )
    selected_score = (
        best_active
        if best_active["spectral_mae_bpm"]
        < baseline["spectral_mae_bpm"]
        else baseline
    )
    selected_score["selected"] = True
    return next(
        candidate
        for candidate in CANDIDATES
        if candidate["name"] == selected_score["candidate"]
    )


def calculate_window_record(
    row: dict[str, int | float],
    subject: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    method: str,
    waveform: np.ndarray,
    baseline: np.ndarray,
    smolk_score: np.ndarray,
    acc_gate: np.ndarray,
    dual_gate: np.ndarray,
    activity_id: int,
) -> tuple[dict[str, Any], np.ndarray]:
    reference_hr = float(
        subject["reference_hr"][int(row["hr_window_index"])]
    )
    peak_hr, peaks, valid_ibi_count = v0.estimate_peak_hr(waveform)
    spectral_hr, _, _ = v0.estimate_window_hr(waveform)
    difference = np.abs(waveform - baseline)
    clean = smolk_score <= SMOLK_GATE_START

    record = {
        "subject_id": int(row["subject_id"]),
        "window_start_s": float(row["window_start_s"]),
        "window_center_s": float(row["window_center_s"]),
        "window_end_s": float(row["window_end_s"]),
        "activity_id": activity_id,
        "activity_name": v0.ACTIVITY_NAMES[activity_id],
        "method": method,
        "reference_hr_bpm": reference_hr,
        "peak_hr_bpm": peak_hr,
        "peak_absolute_error_bpm": (
            abs(peak_hr - reference_hr)
            if np.isfinite(peak_hr)
            else float("nan")
        ),
        "peak_output_valid": bool(np.isfinite(peak_hr)),
        "spectral_hr_bpm": float(spectral_hr),
        "spectral_absolute_error_bpm": float(
            abs(spectral_hr - reference_hr)
        ),
        "detected_peak_count": int(len(peaks)),
        "valid_ibi_count": valid_ibi_count,
        "smolk_probability_mean": float(np.mean(smolk_score)),
        "acc_gate_mean": float(np.mean(acc_gate)),
        "dual_gate_mean": float(np.mean(dual_gate)),
        "dual_gate_active_fraction": float(np.mean(dual_gate > 0.0)),
        "all_region_mean_absolute_change_std": float(
            np.mean(difference)
        ),
        "clean_region_mean_absolute_change_std": (
            float(np.mean(difference[clean]))
            if np.any(clean)
            else float("nan")
        ),
    }
    return record, peaks


def evaluate_subjects(
    subjects: dict[int, dict[str, np.ndarray]],
    contexts: dict[int, dict[str, np.ndarray]],
    selected_candidate: dict[str, Any],
    hr_rows: list[dict[str, int | float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v1_outputs = {
        processed_row: apply_candidate(
            context,
            selected_candidate,
        )[0]
        for processed_row, context in contexts.items()
    }
    records: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None

    for row in hr_rows:
        subject_id = int(row["subject_id"])
        if subject_id not in EVALUATION_SUBJECTS:
            continue

        subject = subjects[subject_id]
        processed_row = int(row["processed_test_row"])
        offset = int(row["segment_offset_start_sample"])
        window_slice = slice(offset, offset + v0.WINDOW_SAMPLES)
        context = contexts[processed_row]

        baseline = context["processed"][window_slice]
        smolk_score = context["smolk_score"][window_slice]
        acc_gate = context["acc_gate"][window_slice]
        dual_gate = context["dual_gate"][window_slice]

        acc_start = int(
            round(float(row["window_start_s"]) * v0.ACC_FS_HZ)
        )
        acc_end = acc_start + v0.ACC_WINDOW_SAMPLES
        acc_window = subject["acc"][acc_start:acc_end]
        acc_64hz = v0.resample_acc(acc_window)
        v0_references = v0.build_acc_references(acc_64hz)
        v0_nlms, _ = v0.nlms_cancel(baseline, v0_references)
        v0_guided = (
            (1.0 - smolk_score) * baseline
            + smolk_score * v0_nlms
        )
        v1_waveform = v1_outputs[processed_row][window_slice]

        activity_id = v0.window_activity(
            subject["activity"],
            float(row["window_start_s"]),
        )
        waveforms = {
            "processed_ppg": baseline,
            "v0_window_reset_guided": v0_guided,
            "v1_continuous_gated": v1_waveform,
        }
        method_details: dict[str, Any] = {}
        for method, waveform in waveforms.items():
            record, peaks = calculate_window_record(
                row,
                subject,
                context,
                method,
                waveform,
                baseline,
                smolk_score,
                acc_gate,
                dual_gate,
                activity_id,
            )
            records.append(record)
            method_details[method] = {
                "waveform": waveform.copy(),
                "peaks": peaks.copy(),
                "peak_hr_bpm": record["peak_hr_bpm"],
                "spectral_hr_bpm": record["spectral_hr_bpm"],
            }

        # 固定选择 S14 的首个完整上下楼窗口，不按效果挑选示例。
        if (
            example is None
            and subject_id == 14
            and activity_id == 2
        ):
            example = {
                "subject_id": subject_id,
                "window_start_s": float(row["window_start_s"]),
                "reference_hr_bpm": float(
                    subject["reference_hr"][
                        int(row["hr_window_index"])
                    ]
                ),
                "acc_64hz": acc_64hz.copy(),
                "smolk_score": smolk_score.copy(),
                "acc_gate": acc_gate.copy(),
                "dual_gate": dual_gate.copy(),
                "baseline": baseline.copy(),
                "methods": method_details,
            }

    if example is None:
        raise ValueError("没有找到固定的 S14 上下楼示例")
    expected = (
        sum(EXPECTED_WINDOW_COUNTS[s] for s in EVALUATION_SUBJECTS)
        * len(METHOD_ORDER)
    )
    if len(records) != expected:
        raise ValueError(f"评估窗口结果数异常：{len(records)}")
    return records, example


def summarize_rows(
    rows: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        for estimator, estimate_field in (
            ("peak_interval", "peak_hr_bpm"),
            ("spectral", "spectral_hr_bpm"),
        ):
            estimated = np.asarray(
                [row[estimate_field] for row in selected],
                dtype=np.float64,
            )
            reference = np.asarray(
                [row["reference_hr_bpm"] for row in selected],
                dtype=np.float64,
            )
            valid = np.isfinite(estimated)
            errors = np.abs(estimated[valid] - reference[valid])
            if len(errors) == 0:
                raise ValueError(
                    f"{scope}/{method}/{estimator} 没有有效结果"
                )
            summaries.append(
                {
                    "scope": scope,
                    "method": method,
                    "estimator": estimator,
                    "window_count": len(selected),
                    "valid_count": int(np.sum(valid)),
                    "coverage_percent": float(100.0 * np.mean(valid)),
                    "mae_bpm": float(np.mean(errors)),
                    "rmse_bpm": float(np.sqrt(np.mean(errors**2))),
                    "median_absolute_error_bpm": float(
                        np.median(errors)
                    ),
                    "p90_absolute_error_bpm": float(
                        np.percentile(errors, 90)
                    ),
                    "failure_gt_10_bpm_percent": float(
                        100.0
                        * np.mean(errors > v0.ERROR_THRESHOLD_BPM)
                    ),
                }
            )
    return summaries


def build_activity_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["activity_id"])].append(row)

    result: list[dict[str, Any]] = []
    for activity_id in sorted(grouped):
        result.extend(
            summarize_rows(
                grouped[activity_id],
                f"activity_{activity_id}_"
                f"{v0.ACTIVITY_NAMES[activity_id]}",
            )
        )
    return result


def summarize_waveform_changes(
    rows: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        all_change = np.asarray(
            [
                row["all_region_mean_absolute_change_std"]
                for row in selected
            ],
            dtype=np.float64,
        )
        clean_change = np.asarray(
            [
                row["clean_region_mean_absolute_change_std"]
                for row in selected
            ],
            dtype=np.float64,
        )
        active_fraction = np.asarray(
            [row["dual_gate_active_fraction"] for row in selected],
            dtype=np.float64,
        )
        result.append(
            {
                "scope": scope,
                "method": method,
                "window_count": len(selected),
                "mean_all_region_change_std": float(
                    np.mean(all_change)
                ),
                "mean_clean_region_change_std": float(
                    np.nanmean(clean_change)
                ),
                "p90_all_region_change_std": float(
                    np.percentile(all_change, 90)
                ),
                "mean_dual_gate_active_fraction": float(
                    np.mean(active_fraction)
                ),
            }
        )
    return result


def save_metric_figure(
    metrics: list[dict[str, Any]],
    waveform_metrics: list[dict[str, Any]],
) -> None:
    metric_index = {
        (row["method"], row["estimator"]): row for row in metrics
    }
    waveform_index = {
        row["method"]: row for row in waveform_metrics
    }
    positions = np.arange(len(METHOD_ORDER))
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    colors = ("#176b87", "#e69f00", "#c44900")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 9),
        constrained_layout=True,
    )
    specifications = (
        (
            axes[0, 0],
            [
                metric_index[(method, "spectral")]["mae_bpm"]
                for method in METHOD_ORDER
            ],
            "Spectral HR MAE",
            "bpm",
        ),
        (
            axes[0, 1],
            [
                metric_index[(method, "spectral")][
                    "failure_gt_10_bpm_percent"
                ]
                for method in METHOD_ORDER
            ],
            "Spectral HR failures >10 bpm",
            "%",
        ),
        (
            axes[1, 0],
            [
                metric_index[(method, "peak_interval")]["mae_bpm"]
                for method in METHOD_ORDER
            ],
            "Peak-interval HR MAE",
            "bpm",
        ),
        (
            axes[1, 1],
            [
                waveform_index[method][
                    "mean_clean_region_change_std"
                ]
                for method in METHOD_ORDER
            ],
            "Mean change in SMoLK-clean regions",
            "standardized amplitude",
        ),
    )
    for axis, values, title, unit in specifications:
        bars = axis.bar(positions, values, color=colors)
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(
        "PPG-DaLiA S14-S15 | V0 versus continuous gated V1"
    )
    figure.savefig(
        OUTPUT_DIR / "v1_hr_and_waveform_comparison.png",
        dpi=180,
    )
    plt.close(figure)


def save_example_figure(example: dict[str, Any]) -> None:
    time_s = np.arange(v0.WINDOW_SAMPLES) / v0.PPG_FS_HZ
    figure, axes = plt.subplots(
        6,
        1,
        figsize=(13, 14),
        constrained_layout=True,
    )

    for method, color in zip(
        METHOD_ORDER,
        ("#176b87", "#e69f00", "#c44900"),
    ):
        axes[0].plot(
            time_s,
            example["methods"][method]["waveform"],
            label=METHOD_LABELS[method],
            color=color,
            alpha=0.8,
        )
    axes[0].set_ylabel("Standardized\nPPG")
    axes[0].set_title(
        f"Fixed example: S{example['subject_id']} stairs, "
        f"{example['window_start_s']:.1f}-"
        f"{example['window_start_s'] + v0.WINDOW_SECONDS:.1f} s"
    )
    axes[0].legend(ncol=3, fontsize=8)

    for axis_index, label, color in (
        (0, "ACC x", "#0072b2"),
        (1, "ACC y", "#d55e00"),
        (2, "ACC z", "#009e73"),
    ):
        axes[1].plot(
            time_s,
            example["acc_64hz"][:, axis_index],
            label=label,
            color=color,
            alpha=0.8,
        )
    axes[1].set_ylabel("ACC (g)")
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(
        time_s,
        example["smolk_score"],
        label="SMoLK score",
        color="#c44900",
    )
    axes[2].plot(
        time_s,
        example["acc_gate"],
        label="ACC gate",
        color="#0072b2",
    )
    axes[2].plot(
        time_s,
        example["dual_gate"],
        label="Dual gate",
        color="#000000",
        linewidth=1.4,
    )
    axes[2].axhline(
        SMOLK_GATE_START,
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Gate / score")
    axes[2].legend(ncol=3, fontsize=8)

    for method, color in (
        ("v0_window_reset_guided", "#e69f00"),
        ("v1_continuous_gated", "#c44900"),
    ):
        removed = (
            example["baseline"]
            - example["methods"][method]["waveform"]
        )
        axes[3].plot(
            time_s,
            removed,
            label=f"Removed by {METHOD_LABELS[method]}",
            color=color,
        )
    axes[3].set_ylabel("Removed\ncomponent")
    axes[3].legend(fontsize=8)

    v1_detail = example["methods"]["v1_continuous_gated"]
    axes[4].plot(
        time_s,
        v1_detail["waveform"],
        color="#c44900",
    )
    axes[4].scatter(
        time_s[v1_detail["peaks"]],
        v1_detail["waveform"][v1_detail["peaks"]],
        color="black",
        s=18,
        label="Detected peaks",
        zorder=3,
    )
    axes[4].set_ylabel("V1 enhanced\nPPG")
    axes[4].legend(fontsize=8)

    positions = np.arange(len(METHOD_ORDER))
    width = 0.34
    peak_values = [
        example["methods"][method]["peak_hr_bpm"]
        for method in METHOD_ORDER
    ]
    spectral_values = [
        example["methods"][method]["spectral_hr_bpm"]
        for method in METHOD_ORDER
    ]
    axes[5].bar(
        positions - width / 2,
        peak_values,
        width,
        label="Peak-interval HR",
        color="#56b4e9",
    )
    axes[5].bar(
        positions + width / 2,
        spectral_values,
        width,
        label="Spectral HR",
        color="#d55e00",
    )
    axes[5].axhline(
        example["reference_hr_bpm"],
        color="black",
        linestyle="--",
        label=f"Reference {example['reference_hr_bpm']:.1f} bpm",
    )
    axes[5].set_xticks(
        positions,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=15,
        ha="right",
    )
    axes[5].set_ylabel("HR (bpm)")
    axes[5].legend(fontsize=8)

    for axis in axes[:5]:
        axis.set_xlim(0, v0.WINDOW_SECONDS)
        axis.grid(alpha=0.2)
    axes[4].set_xlabel("Time within 8 s window (s)")
    figure.savefig(
        OUTPUT_DIR / "fixed_s14_stairs_v1_reconstruction.png",
        dpi=180,
    )
    plt.close(figure)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields}
            for row in rows
        )
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    v0.configure_plot_style()

    if not v0.TEST_X_PATH.is_file():
        raise FileNotFoundError(
            f"缺少处理后测试 PPG：{v0.TEST_X_PATH}"
        )
    processed_test = np.load(v0.TEST_X_PATH, allow_pickle=False)
    if processed_test.shape != (869, SEGMENT_SAMPLES):
        raise ValueError(
            f"处理后测试 PPG 形状异常：{processed_test.shape}"
        )

    _, smooth_probability = v0.load_smolk_probabilities(
        processed_test
    )
    segment_rows = load_segment_mapping()
    hr_rows = load_hr_mapping()
    subjects = {
        subject_id: v0.load_subject(subject_id)
        for subject_id in ALL_SUBJECTS
    }

    design_contexts = load_segment_contexts(
        DESIGN_SUBJECT,
        subjects[DESIGN_SUBJECT],
        processed_test,
        smooth_probability,
        segment_rows,
    )
    candidate_scores = [
        score_candidate(
            candidate,
            design_contexts,
            subjects[DESIGN_SUBJECT],
            hr_rows,
        )
        for candidate in CANDIDATES
    ]
    selected_candidate = select_candidate(candidate_scores)

    evaluation_contexts: dict[int, dict[str, np.ndarray]] = {}
    for subject_id in EVALUATION_SUBJECTS:
        evaluation_contexts.update(
            load_segment_contexts(
                subject_id,
                subjects[subject_id],
                processed_test,
                smooth_probability,
                segment_rows,
            )
        )

    records, example = evaluate_subjects(
        subjects,
        evaluation_contexts,
        selected_candidate,
        hr_rows,
    )
    method_metrics = summarize_rows(
        records,
        "exploratory_S14_S15",
    )
    activity_metrics = build_activity_metrics(records)
    waveform_metrics = summarize_waveform_changes(
        records,
        "exploratory_S14_S15",
    )

    config = {
        "status": "public_dataset_pre_research_not_product_validation",
        "design_subject": DESIGN_SUBJECT,
        "evaluation_subjects": list(EVALUATION_SUBJECTS),
        "selection_rule": (
            "Choose the active candidate with the lowest S1 spectral HR "
            "MAE only if it beats no reconstruction; otherwise select "
            "no reconstruction."
        ),
        "candidates": list(CANDIDATES),
        "selected_candidate": selected_candidate,
        "ppg_sampling_rate_hz": v0.PPG_FS_HZ,
        "acc_original_sampling_rate_hz": v0.ACC_FS_HZ,
        "continuous_processing_seconds": SEGMENT_SECONDS,
        "hr_evaluation_window_seconds": v0.WINDOW_SECONDS,
        "acc_gate_low_g": ACC_GATE_LOW_G,
        "acc_gate_high_g": ACC_GATE_HIGH_G,
        "smolk_gate_start": SMOLK_GATE_START,
        "nlms_epsilon": NLMS_EPSILON,
    }
    summary = {
        "status": "completed_public_dataset_pre_research_v1",
        "research_question": (
            "Does continuous 30 s NLMS with joint SMoLK and ACC gating "
            "improve HR estimates while reducing changes in regions that "
            "SMoLK considers clean?"
        ),
        "design": {
            "parameter_selection_subject": DESIGN_SUBJECT,
            "candidate_scores": candidate_scores,
            "selected_candidate": selected_candidate,
        },
        "evaluation": {
            "subjects": list(EVALUATION_SUBJECTS),
            "contained_window_count": sum(
                EXPECTED_WINDOW_COUNTS[subject_id]
                for subject_id in EVALUATION_SUBJECTS
            ),
            "method_metrics": method_metrics,
            "waveform_change_metrics": waveform_metrics,
            "interpretation": (
                "S14-S15 are retained as a fixed exploratory evaluation "
                "set, but they were already inspected in V0 and therefore "
                "are not a fresh blinded test set."
            ),
        },
        "limitations": [
            "Empatica E4 BVP is a device-processed signal, not raw LED/PD ADC.",
            "No clean motion-corrupted PPG target is available for direct waveform-fidelity evaluation.",
            "The ACC gate uses fixed physical-amplitude thresholds that require recalibration on future hardware.",
            "NLMS models only motion components that are linearly related to the ACC references.",
            "SMoLK uses centered processing and is not currently an online causal detector.",
            "Improved HR error does not by itself prove that the reconstructed pulse morphology is physiologically correct.",
            "S1 parameter selection and S14-S15 evaluation remain public-dataset pre-research, not product validation.",
        ],
    }

    printed = {
        "status": summary["status"],
        "selected_candidate": selected_candidate,
        "s1_candidate_scores": candidate_scores,
        "s14_s15_method_metrics": method_metrics,
        "waveform_change_metrics": waveform_metrics,
        "write_outputs": bool(args.write_outputs),
    }
    if not args.write_outputs:
        print(json.dumps(printed, ensure_ascii=False, indent=2))
        return

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"输出目录已存在，请先保存旧结果：{OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True)
    write_csv(
        OUTPUT_DIR / "candidate_selection_s1.csv",
        candidate_scores,
        CANDIDATE_FIELDS,
    )
    write_csv(
        OUTPUT_DIR / "window_metrics.csv",
        records,
        WINDOW_FIELDS,
    )
    write_csv(
        OUTPUT_DIR / "method_metrics.csv",
        method_metrics,
        METRIC_FIELDS,
    )
    write_csv(
        OUTPUT_DIR / "activity_metrics.csv",
        activity_metrics,
        METRIC_FIELDS,
    )
    write_csv(
        OUTPUT_DIR / "waveform_change_metrics.csv",
        waveform_metrics,
        WAVEFORM_FIELDS,
    )
    atomic_json(OUTPUT_DIR / "config.json", config)
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    save_metric_figure(method_metrics, waveform_metrics)
    save_example_figure(example)

    printed["output"] = str(OUTPUT_DIR)
    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
