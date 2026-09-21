#!/usr/bin/env python3
"""跨 S1–S15 验证 PPG-only HR 与候选 SQI 特征。"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analyze_s1_ppg_sqi import (
    ACTIVITY_FS_HZ,
    ACTIVITY_NAMES,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    STEP_SECONDS,
    WINDOW_SECONDS,
    build_feature_results,
    safe_spearman,
)
from ppg_hr_baseline import (
    MAX_HR_HZ,
    MIN_HR_HZ,
    N_FFT,
    PPG_FS_HZ,
    configure_plot_style,
    require_key,
    require_mapping,
)


DATA_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = DATA_ROOT.parent / "90_共享原始数据/ppg_dalia/PPG_FieldStudy"
OUTPUT_DIR = DATA_ROOT / "02_结果/sqi_cross_subject"
SUBJECT_IDS = tuple(f"S{index}" for index in range(1, 16))

# 使用已经保存的 S1 结果作为回归基准，防止批量脚本改变实验定义。
EXPECTED_S1_WINDOW_COUNT = 4603
EXPECTED_TOTAL_WINDOW_COUNT = 64697
EXPECTED_S1_CORRELATIONS = {
    "finite_ratio": None,
    "flatline_ratio": -0.053862165680604444,
    "extreme_repeat_ratio": -0.04005801946373836,
    "robust_amplitude": 0.4589069909264167,
    "derivative_outlier_ratio": -0.09551695643848519,
    "peak_power_ratio": -0.5596739702747378,
    "peak_to_second_power_ratio": -0.5378143663246013,
    "spectral_entropy": 0.5033375854665123,
    "autocorrelation_peak": -0.4022269887756117,
}

PRIMARY_ACTIVITY_FEATURES = (
    "robust_amplitude",
    "peak_power_ratio",
    "peak_to_second_power_ratio",
    "spectral_entropy",
    "autocorrelation_peak",
)


def load_subject(
    subject_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只读加载一个受试者的腕部 BVP、ECG HR 和活动标签。"""
    data_path = (
        DATASET_DIR
        / subject_id
        / f"{subject_id}.pkl"
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"未找到数据文件：{data_path}")

    with data_path.open("rb") as data_file:
        subject = require_mapping(
            pickle.load(data_file, encoding="latin1"),
            f"{subject_id}.pkl 顶层对象",
        )

    signal = require_mapping(
        require_key(
            subject,
            "signal",
            f"{subject_id}.pkl 顶层",
        ),
        "signal",
    )
    wrist = require_mapping(
        require_key(signal, "wrist", "signal"),
        "signal/wrist",
    )

    ppg = np.asarray(
        require_key(wrist, "BVP", "signal/wrist"),
        dtype=np.float64,
    ).reshape(-1)
    reference_hr = np.asarray(
        require_key(
            subject,
            "label",
            f"{subject_id}.pkl 顶层",
        ),
        dtype=np.float64,
    ).reshape(-1)
    activity = np.asarray(
        require_key(
            subject,
            "activity",
            f"{subject_id}.pkl 顶层",
        ),
        dtype=np.float64,
    ).reshape(-1)

    for name, values in (
        ("PPG", ppg),
        ("ECG 参考心率", reference_hr),
        ("活动标签", activity),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{subject_id} 的{name}中存在非有限数值。"
            )

    return ppg, reference_hr, activity


def hr_metrics(
    absolute_error: np.ndarray,
) -> dict[str, float]:
    """计算每个受试者一致的窗口级 HR 评价指标。"""
    return {
        "mae_bpm": float(np.mean(absolute_error)),
        "rmse_bpm": float(
            np.sqrt(np.mean(absolute_error**2))
        ),
        "p90_bpm": float(
            np.percentile(absolute_error, 90)
        ),
        "max_error_bpm": float(np.max(absolute_error)),
        "within_3_bpm_percent": float(
            100.0 * np.mean(absolute_error <= 3.0)
        ),
        "failure_gt_10_bpm_percent": float(
            100.0 * np.mean(absolute_error > 10.0)
        ),
    }


def summarize_subject(
    subject_id: str,
    results: dict[str, np.ndarray],
) -> dict[str, Any]:
    """汇总一个受试者的误差和特征相关性。"""
    absolute_error = results["absolute_error_bpm"]
    return {
        "subject": subject_id,
        "window_count": int(len(absolute_error)),
        **hr_metrics(absolute_error),
        "spearman": {
            feature: safe_spearman(
                results[feature],
                absolute_error,
            )
            for feature in FEATURE_COLUMNS
        },
    }


def validate_s1_regression(
    summary: dict[str, Any],
) -> None:
    """确认批量计算没有改变已验证的 S1 实验定义。"""
    if summary["window_count"] != EXPECTED_S1_WINDOW_COUNT:
        raise ValueError(
            "S1 窗口数回归检查失败："
            f"期望 {EXPECTED_S1_WINDOW_COUNT}，"
            f"实际 {summary['window_count']}。"
        )

    for feature, expected in (
        EXPECTED_S1_CORRELATIONS.items()
    ):
        actual = summary["spearman"][feature]

        if expected is None:
            if actual is not None:
                raise ValueError(
                    f"S1 {feature} 回归检查失败："
                    f"期望 NA，实际 {actual}。"
                )
            continue

        if actual is None or not np.isclose(
            actual,
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"S1 {feature} 回归检查失败："
                f"期望 {expected}，实际 {actual}。"
            )


def combine_results(
    results_by_subject: list[
        tuple[str, dict[str, np.ndarray]]
    ],
) -> dict[str, np.ndarray]:
    """连接各受试者窗口，用于分活动汇总。"""
    combined: dict[str, np.ndarray] = {}
    columns = (
        "absolute_error_bpm",
        "activity_id",
        *FEATURE_COLUMNS,
    )

    for column in columns:
        combined[column] = np.concatenate(
            [
                results[column]
                for _, results in results_by_subject
            ]
        )

    combined["subject_id"] = np.concatenate(
        [
            np.full(
                len(results["absolute_error_bpm"]),
                subject_id,
                dtype=object,
            )
            for subject_id, results in results_by_subject
        ]
    )
    return combined


def build_activity_summary(
    combined: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """跨受试者汇总每种活动的误差和特征关系。"""
    summaries: list[dict[str, Any]] = []

    for activity_id, activity_name in (
        ACTIVITY_NAMES.items()
    ):
        mask = combined["activity_id"] == activity_id
        if not np.any(mask):
            continue

        absolute_error = combined[
            "absolute_error_bpm"
        ][mask]
        summaries.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "window_count": int(np.sum(mask)),
                "subject_count": int(
                    len(
                        np.unique(
                            combined["subject_id"][mask]
                        )
                    )
                ),
                **hr_metrics(absolute_error),
                "spearman": {
                    feature: safe_spearman(
                        combined[feature][mask],
                        absolute_error,
                    )
                    for feature in FEATURE_COLUMNS
                },
            }
        )

    return summaries


def build_feature_stability(
    subject_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """统计每项特征在不同受试者中的相关方向一致性。"""
    stability: list[dict[str, Any]] = []

    for feature in FEATURE_COLUMNS:
        correlations = [
            summary["spearman"][feature]
            for summary in subject_summaries
            if summary["spearman"][feature] is not None
        ]

        if not correlations:
            stability.append(
                {
                    "feature": feature,
                    "valid_subject_count": 0,
                    "median_spearman": None,
                    "positive_subject_count": 0,
                    "negative_subject_count": 0,
                    "dominant_direction": "NA",
                    "direction_consistency_percent": None,
                }
            )
            continue

        values = np.asarray(
            correlations,
            dtype=np.float64,
        )
        positive_count = int(np.sum(values > 0))
        negative_count = int(np.sum(values < 0))
        dominant_count = max(
            positive_count,
            negative_count,
        )

        if positive_count > negative_count:
            dominant_direction = "positive"
        elif negative_count > positive_count:
            dominant_direction = "negative"
        else:
            dominant_direction = "mixed"

        stability.append(
            {
                "feature": feature,
                "valid_subject_count": int(len(values)),
                "median_spearman": float(
                    np.median(values)
                ),
                "positive_subject_count": positive_count,
                "negative_subject_count": negative_count,
                "dominant_direction": dominant_direction,
                "direction_consistency_percent": float(
                    100.0 * dominant_count / len(values)
                ),
            }
        )

    return stability


def format_correlation(value: float | None) -> str:
    """统一打印相关系数。"""
    return "NA" if value is None else f"{value:.3f}"


def validate_aggregate_counts(
    subject_summaries: list[dict[str, Any]],
    activity_summaries: list[dict[str, Any]],
    combined: dict[str, np.ndarray],
) -> int:
    """确认受试者、活动和合并窗口统计相互一致。"""
    subject_total = sum(
        summary["window_count"]
        for summary in subject_summaries
    )
    combined_total = len(combined["absolute_error_bpm"])
    activity_total = sum(
        summary["window_count"]
        for summary in activity_summaries
    )

    counts = {
        "subject_total": subject_total,
        "combined_total": combined_total,
        "activity_total": activity_total,
        "expected_total": EXPECTED_TOTAL_WINDOW_COUNT,
    }
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"跨受试者窗口总数不一致：{counts}"
        )

    return subject_total


def csv_float(value: float | None) -> str:
    """将浮点结果稳定地写入 CSV，None 使用空字符串。"""
    return "" if value is None else f"{value:.9f}"


def write_window_csv(
    results_by_subject: list[
        tuple[str, dict[str, np.ndarray]]
    ],
) -> Path:
    """保存所有受试者的逐窗口特征和离线误差。"""
    output_path = OUTPUT_DIR / (
        "S1_S15_window_features.csv"
    )
    fieldnames = (
        "subject_id",
        "window_start_s",
        "window_center_s",
        "window_end_s",
        "activity_id",
        "activity_name",
        "ppg_hr_bpm",
        "ecg_hr_bpm",
        "absolute_error_bpm",
        "error_group",
        *FEATURE_COLUMNS,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for subject_id, results in results_by_subject:
            for index in range(
                len(results["window_center_s"])
            ):
                activity_id = int(
                    results["activity_id"][index]
                )
                row: dict[str, str | int] = {
                    "subject_id": subject_id,
                    "window_start_s": csv_float(
                        float(
                            results["window_start_s"][
                                index
                            ]
                        )
                    ),
                    "window_center_s": csv_float(
                        float(
                            results["window_center_s"][
                                index
                            ]
                        )
                    ),
                    "window_end_s": csv_float(
                        float(
                            results["window_end_s"][index]
                        )
                    ),
                    "activity_id": activity_id,
                    "activity_name": (
                        ACTIVITY_NAMES[activity_id]
                    ),
                    "ppg_hr_bpm": csv_float(
                        float(
                            results["estimated_hr_bpm"][
                                index
                            ]
                        )
                    ),
                    "ecg_hr_bpm": csv_float(
                        float(
                            results["reference_hr_bpm"][
                                index
                            ]
                        )
                    ),
                    "absolute_error_bpm": csv_float(
                        float(
                            results["absolute_error_bpm"][
                                index
                            ]
                        )
                    ),
                    "error_group": str(
                        results["error_group"][index]
                    ),
                }

                for feature in FEATURE_COLUMNS:
                    row[feature] = csv_float(
                        float(results[feature][index])
                    )

                writer.writerow(row)

    return output_path


def write_subject_metrics_csv(
    subject_summaries: list[dict[str, Any]],
) -> Path:
    """保存每个受试者的 HR 误差指标。"""
    output_path = OUTPUT_DIR / (
        "S1_S15_subject_hr_metrics.csv"
    )
    fieldnames = (
        "subject",
        "window_count",
        "mae_bpm",
        "rmse_bpm",
        "p90_bpm",
        "max_error_bpm",
        "within_3_bpm_percent",
        "failure_gt_10_bpm_percent",
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in subject_summaries:
            writer.writerow(
                {
                    "subject": summary["subject"],
                    "window_count": (
                        summary["window_count"]
                    ),
                    "mae_bpm": csv_float(
                        summary["mae_bpm"]
                    ),
                    "rmse_bpm": csv_float(
                        summary["rmse_bpm"]
                    ),
                    "p90_bpm": csv_float(
                        summary["p90_bpm"]
                    ),
                    "max_error_bpm": csv_float(
                        summary["max_error_bpm"]
                    ),
                    "within_3_bpm_percent": csv_float(
                        summary[
                            "within_3_bpm_percent"
                        ]
                    ),
                    "failure_gt_10_bpm_percent": (
                        csv_float(
                            summary[
                                "failure_gt_10_bpm_percent"
                            ]
                        )
                    ),
                }
            )

    return output_path


def write_subject_correlations_csv(
    subject_summaries: list[dict[str, Any]],
) -> Path:
    """以长表形式保存各受试者的特征相关性。"""
    output_path = OUTPUT_DIR / (
        "S1_S15_subject_feature_correlations.csv"
    )
    fieldnames = (
        "subject",
        "feature",
        "spearman_with_absolute_hr_error",
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in subject_summaries:
            for feature in FEATURE_COLUMNS:
                writer.writerow(
                    {
                        "subject": summary["subject"],
                        "feature": feature,
                        (
                            "spearman_with_absolute_hr_error"
                        ): csv_float(
                            summary["spearman"][feature]
                        ),
                    }
                )

    return output_path


def write_activity_summary_csv(
    activity_summaries: list[dict[str, Any]],
) -> Path:
    """保存跨受试者的分活动误差和特征相关性。"""
    output_path = OUTPUT_DIR / (
        "S1_S15_activity_summary.csv"
    )
    metric_fields = (
        "mae_bpm",
        "rmse_bpm",
        "p90_bpm",
        "max_error_bpm",
        "within_3_bpm_percent",
        "failure_gt_10_bpm_percent",
    )
    correlation_fields = tuple(
        f"spearman_{feature}"
        for feature in FEATURE_COLUMNS
    )
    fieldnames = (
        "activity_id",
        "activity_name",
        "subject_count",
        "window_count",
        *metric_fields,
        *correlation_fields,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in activity_summaries:
            row: dict[str, str | int] = {
                "activity_id": summary["activity_id"],
                "activity_name": summary["activity_name"],
                "subject_count": summary["subject_count"],
                "window_count": summary["window_count"],
            }
            for metric in metric_fields:
                row[metric] = csv_float(summary[metric])
            for feature in FEATURE_COLUMNS:
                row[f"spearman_{feature}"] = csv_float(
                    summary["spearman"][feature]
                )
            writer.writerow(row)

    return output_path


def write_feature_stability_csv(
    feature_stability: list[dict[str, Any]],
) -> Path:
    """保存特征在不同受试者中的方向稳定性。"""
    output_path = OUTPUT_DIR / (
        "S1_S15_feature_stability.csv"
    )
    fieldnames = (
        "feature",
        "valid_subject_count",
        "median_spearman",
        "positive_subject_count",
        "negative_subject_count",
        "dominant_direction",
        "direction_consistency_percent",
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in feature_stability:
            writer.writerow(
                {
                    "feature": summary["feature"],
                    "valid_subject_count": (
                        summary["valid_subject_count"]
                    ),
                    "median_spearman": csv_float(
                        summary["median_spearman"]
                    ),
                    "positive_subject_count": (
                        summary[
                            "positive_subject_count"
                        ]
                    ),
                    "negative_subject_count": (
                        summary[
                            "negative_subject_count"
                        ]
                    ),
                    "dominant_direction": (
                        summary["dominant_direction"]
                    ),
                    (
                        "direction_consistency_percent"
                    ): csv_float(
                        summary[
                            "direction_consistency_percent"
                        ]
                    ),
                }
            )

    return output_path


def write_summary_json(
    total_window_count: int,
    subject_summaries: list[dict[str, Any]],
    activity_summaries: list[dict[str, Any]],
    feature_stability: list[dict[str, Any]],
) -> Path:
    """保存完整参数、汇总结果和研究边界。"""
    output_path = OUTPUT_DIR / "S1_S15_summary.json"
    summary = {
        "analysis": (
            "PPG-only HR and candidate SQI "
            "cross-subject validation"
        ),
        "subject_ids": list(SUBJECT_IDS),
        "subject_count": len(SUBJECT_IDS),
        "total_window_count": total_window_count,
        "candidate_sqi_only": True,
        "formal_quality_thresholds_defined": False,
        "feature_uses_reference_hr": False,
        "reference_hr_used_only_for_offline_evaluation": (
            True
        ),
        "feature_uses_acc": False,
        "parameters": {
            "ppg_sampling_rate_hz": PPG_FS_HZ,
            "activity_sampling_rate_hz": ACTIVITY_FS_HZ,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "n_fft": N_FFT,
            "candidate_hr_hz": [
                MIN_HR_HZ,
                MAX_HR_HZ,
            ],
        },
        "subject_summaries": subject_summaries,
        "activity_summaries": activity_summaries,
        "feature_stability": feature_stability,
        "limitations": [
            (
                "HR error is an indirect evaluation target, "
                "not an SQI ground-truth label"
            ),
            (
                "E4 BVP is device-exported data, "
                "not per-channel ADC data"
            ),
            (
                "activity categories have unequal "
                "window counts"
            ),
            (
                "periodic motion can produce strong "
                "spectral and autocorrelation peaks"
            ),
            (
                "no formal quality score, state threshold "
                "or product gate is defined"
            ),
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )
        json_file.write("\n")

    return output_path


def plot_subject_hr_error(
    subject_summaries: list[dict[str, Any]],
) -> Path:
    """比较各受试者的 MAE 和 P90 误差。"""
    subject_names = [
        summary["subject"]
        for summary in subject_summaries
    ]
    mae = [
        summary["mae_bpm"]
        for summary in subject_summaries
    ]
    p90 = [
        summary["p90_bpm"]
        for summary in subject_summaries
    ]
    positions = np.arange(len(subject_names))
    width = 0.38

    figure, axis = plt.subplots(
        figsize=(13, 6),
        constrained_layout=True,
    )
    axis.bar(
        positions - width / 2,
        mae,
        width,
        label="MAE",
        color="#4c78a8",
    )
    axis.bar(
        positions + width / 2,
        p90,
        width,
        label="P90 绝对误差",
        color="#f58518",
    )
    axis.set_xticks(positions, subject_names)
    axis.set_ylabel("HR 绝对误差 (bpm)")
    axis.set_title(
        "PPG-DaLiA：各受试者 PPG-only HR 误差"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    output_path = OUTPUT_DIR / (
        "S1_S15_subject_hr_error.png"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_feature_stability(
    subject_summaries: list[dict[str, Any]],
) -> Path:
    """绘制核心候选特征的跨受试者相关性热力图。"""
    subject_names = [
        summary["subject"]
        for summary in subject_summaries
    ]
    matrix = np.asarray(
        [
            [
                (
                    np.nan
                    if summary["spearman"][feature]
                    is None
                    else summary["spearman"][feature]
                )
                for feature in PRIMARY_ACTIVITY_FEATURES
            ]
            for summary in subject_summaries
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(11, 8),
        constrained_layout=True,
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-0.7,
        vmax=0.7,
    )
    axis.set_xticks(
        np.arange(len(PRIMARY_ACTIVITY_FEATURES)),
        [
            FEATURE_LABELS[feature]
            for feature in PRIMARY_ACTIVITY_FEATURES
        ],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(subject_names)),
        subject_names,
    )
    axis.set_title(
        "候选特征与 HR 绝对误差的跨受试者 "
        "Spearman 相关"
    )
    figure.colorbar(
        image,
        ax=axis,
        label="Spearman ρ",
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if abs(value) >= 0.4
                        else "black"
                    ),
                )

    output_path = OUTPUT_DIR / (
        "S1_S15_feature_stability.png"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_activity_hr_error(
    activity_summaries: list[dict[str, Any]],
) -> Path:
    """比较不同活动的 HR 误差和严重失败比例。"""
    activity_names = [
        summary["activity_name"]
        for summary in activity_summaries
    ]
    mae = [
        summary["mae_bpm"]
        for summary in activity_summaries
    ]
    failure_rate = [
        summary["failure_gt_10_bpm_percent"]
        for summary in activity_summaries
    ]
    positions = np.arange(len(activity_names))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].bar(
        positions,
        mae,
        color="#4c78a8",
    )
    axes[0].set_ylabel("MAE (bpm)")
    axes[0].set_title(
        "不同活动下的 PPG-only HR 误差"
    )
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        positions,
        failure_rate,
        color="#e45756",
    )
    axes[1].set_ylabel("误差 >10 bpm 窗口比例 (%)")
    axes[1].set_xticks(
        positions,
        activity_names,
        rotation=25,
        ha="right",
    )
    axes[1].grid(axis="y", alpha=0.25)

    output_path = OUTPUT_DIR / (
        "S1_S15_activity_hr_error.png"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_outputs(
    total_window_count: int,
    results_by_subject: list[
        tuple[str, dict[str, np.ndarray]]
    ],
    subject_summaries: list[dict[str, Any]],
    activity_summaries: list[dict[str, Any]],
    feature_stability: list[dict[str, Any]],
) -> list[Path]:
    """统一创建经过窗口数校验的结果文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    return [
        write_window_csv(results_by_subject),
        write_subject_metrics_csv(subject_summaries),
        write_subject_correlations_csv(
            subject_summaries
        ),
        write_activity_summary_csv(activity_summaries),
        write_feature_stability_csv(feature_stability),
        write_summary_json(
            total_window_count,
            subject_summaries,
            activity_summaries,
            feature_stability,
        ),
        plot_subject_hr_error(subject_summaries),
        plot_feature_stability(subject_summaries),
        plot_activity_hr_error(activity_summaries),
    ]


def parse_arguments() -> argparse.Namespace:
    """解析显式输出开关。"""
    parser = argparse.ArgumentParser(
        description=(
            "跨 S1–S15 验证 PPG-only HR "
            "与候选 SQI 特征。"
        )
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help=(
            "在窗口数校验通过后写出 CSV、JSON 和图片；"
            "默认只向终端打印。"
        ),
    )
    return parser.parse_args()


def print_summary(
    subject_summaries: list[dict[str, Any]],
    activity_summaries: list[dict[str, Any]],
    feature_stability: list[dict[str, Any]],
) -> None:
    """只向终端打印结果，本阶段不创建输出文件。"""
    print(
        "PPG-DaLiA S1–S15："
        "PPG-only HR 与候选 SQI 跨受试者验证"
    )
    print("S1 回归检查: PASS")

    print("\n各受试者 HR 误差：")
    print(
        "subject windows MAE RMSE P90 max "
        "within_3% failure_gt_10%"
    )
    for summary in subject_summaries:
        print(
            f"{summary['subject']:>3} "
            f"{summary['window_count']:>7d} "
            f"{summary['mae_bpm']:>6.2f} "
            f"{summary['rmse_bpm']:>6.2f} "
            f"{summary['p90_bpm']:>6.2f} "
            f"{summary['max_error_bpm']:>6.2f} "
            f"{summary['within_3_bpm_percent']:>9.1f} "
            f"{summary['failure_gt_10_bpm_percent']:>14.1f}"
        )

    print(
        "\n各受试者特征与 HR 绝对误差的 "
        "Spearman 相关："
    )
    for summary in subject_summaries:
        print(f"\n{summary['subject']}:")
        for feature in FEATURE_COLUMNS:
            value = summary["spearman"][feature]
            print(
                f"  {feature}: "
                f"{format_correlation(value)}"
            )

    print("\n跨受试者特征方向稳定性：")
    print(
        "feature valid median positive negative "
        "direction consistency%"
    )
    for result in feature_stability:
        consistency = result[
            "direction_consistency_percent"
        ]
        consistency_text = (
            "NA"
            if consistency is None
            else f"{consistency:.1f}"
        )
        print(
            f"{result['feature']} "
            f"{result['valid_subject_count']} "
            f"{format_correlation(result['median_spearman'])} "
            f"{result['positive_subject_count']} "
            f"{result['negative_subject_count']} "
            f"{result['dominant_direction']} "
            f"{consistency_text}"
        )

    print("\n跨受试者分活动结果：")
    for summary in activity_summaries:
        print(
            f"\n{summary['activity_id']} "
            f"{summary['activity_name']}: "
            f"subjects={summary['subject_count']}, "
            f"windows={summary['window_count']}, "
            f"MAE={summary['mae_bpm']:.2f} bpm, "
            f"≤3 bpm={summary['within_3_bpm_percent']:.1f}%, "
            f">10 bpm="
            f"{summary['failure_gt_10_bpm_percent']:.1f}%"
        )
        for feature in PRIMARY_ACTIVITY_FEATURES:
            value = summary["spearman"][feature]
            print(
                f"  {feature}: "
                f"{format_correlation(value)}"
            )

    print(
        "\n说明：本次只验证候选特征的跨人稳定性，"
        "不生成正式 SQI、阈值或质量状态。"
    )


def main() -> int:
    """运行 S1–S15 跨受试者候选特征验证。"""
    try:
        arguments = parse_arguments()
        results_by_subject: list[
            tuple[str, dict[str, np.ndarray]]
        ] = []
        subject_summaries: list[dict[str, Any]] = []

        for subject_id in SUBJECT_IDS:
            print(
                f"正在分析 {subject_id}...",
                file=sys.stderr,
            )
            ppg, reference_hr, activity = load_subject(
                subject_id
            )
            results = build_feature_results(
                ppg,
                reference_hr,
                activity,
            )
            summary = summarize_subject(
                subject_id,
                results,
            )

            if subject_id == "S1":
                validate_s1_regression(summary)

            results_by_subject.append(
                (subject_id, results)
            )
            subject_summaries.append(summary)

        combined = combine_results(results_by_subject)
        activity_summaries = build_activity_summary(
            combined
        )
        feature_stability = build_feature_stability(
            subject_summaries
        )
        total_window_count = validate_aggregate_counts(
            subject_summaries,
            activity_summaries,
            combined,
        )
        print_summary(
            subject_summaries,
            activity_summaries,
            feature_stability,
        )

        if arguments.write_outputs:
            output_paths = write_outputs(
                total_window_count,
                results_by_subject,
                subject_summaries,
                activity_summaries,
                feature_stability,
            )
            print("\n已生成可追溯结果：")
            for output_path in output_paths:
                print(f"  {output_path}")
        else:
            print(
                "\n未写入结果文件。需要保存时使用："
                "\n  python "
                "analyze_all_subjects_ppg_sqi.py "
                "--write-outputs"
            )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"跨受试者验证运行失败：{error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
