#!/usr/bin/env python3
"""探索 PPG-DaLiA S1 的 PPG-only 信号质量特征。"""

from __future__ import annotations

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

from ppg_hr_baseline import (
    DATA_PATH,
    MAX_HR_HZ,
    MIN_HR_HZ,
    N_FFT,
    PEAK_BAND_HALF_WIDTH_HZ,
    PPG_FS_HZ,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    build_window_results,
    configure_plot_style,
    remove_linear_trend,
    require_key,
    require_mapping,
)


DATA_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DATA_ROOT / "02_结果/sqi_analysis"

ACTIVITY_FS_HZ = 4
WINDOW_SECONDS = 8
STEP_SECONDS = 2
ACTIVITY_WINDOW_SAMPLES = ACTIVITY_FS_HZ * WINDOW_SECONDS
ACTIVITY_STEP_SAMPLES = ACTIVITY_FS_HZ * STEP_SECONDS

# 第二候选峰必须与主峰保持一定距离，避免把同一宽谱峰的相邻频点
# 误认为两个独立候选。
SECOND_PEAK_MIN_SEPARATION_HZ = 0.20
DERIVATIVE_OUTLIER_MAD_FACTOR = 6.0
MAX_PEAK_TO_SECOND_RATIO = 100.0

# 这两个误差边界只用于探索特征分布，不是产品质量阈值。
ACCURATE_ERROR_MAX_BPM = 3.0
INTERMEDIATE_ERROR_MAX_BPM = 10.0

ACTIVITY_NAMES = {
    0: "过渡/其他",
    1: "静坐",
    2: "上下楼",
    3: "桌上足球",
    4: "骑车",
    5: "开车",
    6: "午餐",
    7: "步行",
    8: "办公",
}

ERROR_GROUPS = (
    "accurate_le_3",
    "intermediate_3_to_10",
    "failure_gt_10",
)

ERROR_GROUP_LABELS = {
    "accurate_le_3": "误差 ≤3 bpm",
    "intermediate_3_to_10": "误差 3–10 bpm",
    "failure_gt_10": "误差 >10 bpm",
}

ERROR_GROUP_COLORS = {
    "accurate_le_3": "#2ca02c",
    "intermediate_3_to_10": "#ff7f0e",
    "failure_gt_10": "#d62728",
}

FEATURE_COLUMNS = (
    "finite_ratio",
    "flatline_ratio",
    "extreme_repeat_ratio",
    "robust_amplitude",
    "derivative_outlier_ratio",
    "peak_power_ratio",
    "peak_to_second_power_ratio",
    "spectral_entropy",
    "autocorrelation_peak",
)

FEATURE_LABELS = {
    "finite_ratio": "有限数据比例",
    "flatline_ratio": "相邻样本不变比例",
    "extreme_repeat_ratio": "极值重复比例",
    "robust_amplitude": "稳健幅度 P95-P5",
    "derivative_outlier_ratio": "异常跳变比例",
    "peak_power_ratio": "主峰带能量占比",
    "peak_to_second_power_ratio": "第一峰/第二峰能量比",
    "spectral_entropy": "归一化频谱熵",
    "autocorrelation_peak": "自相关周期强度",
}


def load_s1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只读加载腕部 BVP、ECG 心率标签和活动标签。"""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")

    with DATA_PATH.open("rb") as data_file:
        subject = require_mapping(
            pickle.load(data_file, encoding="latin1"),
            "S1.pkl 顶层对象",
        )

    signal = require_mapping(
        require_key(subject, "signal", "S1.pkl 顶层"),
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
        require_key(subject, "label", "S1.pkl 顶层"),
        dtype=np.float64,
    ).reshape(-1)
    activity = np.asarray(
        require_key(subject, "activity", "S1.pkl 顶层"),
        dtype=np.float64,
    ).reshape(-1)

    for name, values in (
        ("PPG", ppg),
        ("ECG 参考心率", reference_hr),
        ("活动标签", activity),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 中存在非有限数值。")

    return ppg, reference_hr, activity


def validate_window_counts(
    ppg: np.ndarray,
    reference_hr: np.ndarray,
    activity: np.ndarray,
) -> int:
    """确认 PPG、活动标签和 ECG 标签可以逐窗口对齐。"""
    ppg_window_count = (
        1 + (len(ppg) - WINDOW_SAMPLES) // STEP_SAMPLES
    )
    activity_window_count = (
        1
        + (
            len(activity) - ACTIVITY_WINDOW_SAMPLES
        )
        // ACTIVITY_STEP_SAMPLES
    )

    counts = {
        "PPG": ppg_window_count,
        "activity": activity_window_count,
        "reference_hr": len(reference_hr),
    }

    if len(set(counts.values())) != 1:
        raise ValueError(f"窗口数不一致，无法逐窗对齐：{counts}")

    return ppg_window_count


def activity_mode(activity_window: np.ndarray) -> int:
    """使用窗口内出现次数最多的编号作为窗口活动标签。"""
    labels = np.rint(activity_window).astype(np.int64)
    labels = labels[(labels >= 0) & (labels <= 8)]

    if len(labels) == 0:
        return 0

    return int(np.argmax(np.bincount(labels, minlength=9)))


def error_group(absolute_error_bpm: float) -> str:
    """按照研究用边界标记准确、中间和明显失败窗口。"""
    if absolute_error_bpm <= ACCURATE_ERROR_MAX_BPM:
        return "accurate_le_3"
    if absolute_error_bpm <= INTERMEDIATE_ERROR_MAX_BPM:
        return "intermediate_3_to_10"
    return "failure_gt_10"


def ppg_power_spectrum(
    ppg_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """沿用心率基线的去趋势、Hann 窗和 FFT 设置。"""
    detrended = remove_linear_trend(ppg_window)
    tapered = detrended * np.hanning(len(detrended))
    spectrum = np.fft.rfft(tapered, n=N_FFT)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(
        N_FFT,
        d=1.0 / PPG_FS_HZ,
    )
    return frequency, power


def spectral_quality_features(
    ppg_window: np.ndarray,
    selected_frequency_hz: float,
) -> tuple[float, float]:
    """计算竞争峰能量比和归一化频谱熵。"""
    frequency, power = ppg_power_spectrum(ppg_window)
    candidate_mask = (
        (frequency >= MIN_HR_HZ)
        & (frequency <= MAX_HR_HZ)
    )
    candidate_power = power[candidate_mask]
    total_power = float(np.sum(candidate_power))

    if total_power <= 0 or len(candidate_power) <= 1:
        return 0.0, 1.0

    probability = candidate_power / total_power
    nonzero_probability = probability[probability > 0]
    spectral_entropy = -float(
        np.sum(nonzero_probability * np.log(nonzero_probability))
    ) / float(np.log(len(candidate_power)))

    first_peak_mask = candidate_mask & (
        np.abs(frequency - selected_frequency_hz)
        <= PEAK_BAND_HALF_WIDTH_HZ
    )
    first_peak_power = float(np.sum(power[first_peak_mask]))

    second_candidate_mask = candidate_mask & (
        np.abs(frequency - selected_frequency_hz)
        >= SECOND_PEAK_MIN_SEPARATION_HZ
    )

    if not np.any(second_candidate_mask):
        return MAX_PEAK_TO_SECOND_RATIO, spectral_entropy

    second_candidate_indices = np.flatnonzero(second_candidate_mask)
    second_peak_index = int(
        second_candidate_indices[
            np.argmax(power[second_candidate_mask])
        ]
    )
    second_frequency_hz = float(frequency[second_peak_index])
    second_peak_mask = candidate_mask & (
        np.abs(frequency - second_frequency_hz)
        <= PEAK_BAND_HALF_WIDTH_HZ
    )
    second_peak_power = float(np.sum(power[second_peak_mask]))

    tiny = np.finfo(np.float64).tiny
    peak_to_second_ratio = min(
        first_peak_power / max(second_peak_power, tiny),
        MAX_PEAK_TO_SECOND_RATIO,
    )

    return float(peak_to_second_ratio), spectral_entropy


def autocorrelation_periodicity(ppg_window: np.ndarray) -> float:
    """计算生理心率周期范围内最大的归一化自相关值。"""
    detrended = remove_linear_trend(ppg_window)
    sample_count = len(detrended)
    fft_size = 1 << (2 * sample_count - 1).bit_length()
    spectrum = np.fft.rfft(detrended, n=fft_size)
    autocorrelation = np.fft.irfft(
        np.abs(spectrum) ** 2,
        n=fft_size,
    )[:sample_count]

    overlap_count = np.arange(
        sample_count,
        0,
        -1,
        dtype=np.float64,
    )
    autocorrelation = autocorrelation / overlap_count

    zero_lag = float(autocorrelation[0])
    if zero_lag <= 0:
        return 0.0

    autocorrelation = autocorrelation / zero_lag
    minimum_lag = max(
        1,
        int(np.floor(PPG_FS_HZ / MAX_HR_HZ)),
    )
    maximum_lag = min(
        sample_count - 1,
        int(np.ceil(PPG_FS_HZ / MIN_HR_HZ)),
    )

    if maximum_lag < minimum_lag:
        return 0.0

    return float(
        np.clip(
            np.max(
                autocorrelation[
                    minimum_lag : maximum_lag + 1
                ]
            ),
            -1.0,
            1.0,
        )
    )


def window_quality_features(
    ppg_window: np.ndarray,
    selected_frequency_hz: float,
    peak_power_ratio: float,
) -> dict[str, float]:
    """仅根据当前 PPG 窗口计算候选质量特征。"""
    finite_mask = np.isfinite(ppg_window)
    finite_ratio = float(np.mean(finite_mask))

    if not np.all(finite_mask):
        return {
            "finite_ratio": finite_ratio,
            "flatline_ratio": 1.0,
            "extreme_repeat_ratio": 1.0,
            "robust_amplitude": 0.0,
            "derivative_outlier_ratio": 1.0,
            "peak_power_ratio": 0.0,
            "peak_to_second_power_ratio": 0.0,
            "spectral_entropy": 1.0,
            "autocorrelation_peak": 0.0,
        }

    ppg_window = np.asarray(ppg_window, dtype=np.float64)
    differences = np.diff(ppg_window)
    flatline_ratio = (
        float(np.mean(differences == 0.0))
        if len(differences)
        else 1.0
    )

    minimum_value = float(np.min(ppg_window))
    maximum_value = float(np.max(ppg_window))
    extreme_repeat_ratio = max(
        float(np.mean(ppg_window == minimum_value)),
        float(np.mean(ppg_window == maximum_value)),
    )

    percentile_5, percentile_95 = np.percentile(
        ppg_window,
        [5, 95],
    )
    robust_amplitude = float(percentile_95 - percentile_5)

    if len(differences) == 0:
        derivative_outlier_ratio = 1.0
    else:
        derivative_median = float(np.median(differences))
        derivative_mad = float(
            np.median(
                np.abs(differences - derivative_median)
            )
        )
        robust_scale = 1.4826 * derivative_mad

        if robust_scale <= np.finfo(np.float64).eps:
            derivative_outlier_ratio = 0.0
        else:
            derivative_outlier_ratio = float(
                np.mean(
                    np.abs(differences - derivative_median)
                    > DERIVATIVE_OUTLIER_MAD_FACTOR
                    * robust_scale
                )
            )

    (
        peak_to_second_power_ratio,
        spectral_entropy,
    ) = spectral_quality_features(
        ppg_window,
        selected_frequency_hz,
    )

    return {
        "finite_ratio": finite_ratio,
        "flatline_ratio": flatline_ratio,
        "extreme_repeat_ratio": extreme_repeat_ratio,
        "robust_amplitude": robust_amplitude,
        "derivative_outlier_ratio": derivative_outlier_ratio,
        "peak_power_ratio": float(peak_power_ratio),
        "peak_to_second_power_ratio": (
            peak_to_second_power_ratio
        ),
        "spectral_entropy": spectral_entropy,
        "autocorrelation_peak": (
            autocorrelation_periodicity(ppg_window)
        ),
    }


def build_feature_results(
    ppg: np.ndarray,
    reference_hr: np.ndarray,
    activity: np.ndarray,
) -> dict[str, np.ndarray]:
    """计算 S1 全时段的 HR、活动标签和 PPG 质量特征。"""
    window_count = validate_window_counts(
        ppg,
        reference_hr,
        activity,
    )
    results = build_window_results(ppg, reference_hr)

    feature_values = {
        feature: np.empty(window_count, dtype=np.float64)
        for feature in FEATURE_COLUMNS
    }
    activity_id = np.empty(window_count, dtype=np.int64)

    for index in range(window_count):
        ppg_start = index * STEP_SAMPLES
        ppg_end = ppg_start + WINDOW_SAMPLES
        activity_start = index * ACTIVITY_STEP_SAMPLES
        activity_end = (
            activity_start + ACTIVITY_WINDOW_SAMPLES
        )

        ppg_window = ppg[ppg_start:ppg_end]
        activity_window = activity[
            activity_start:activity_end
        ]
        activity_id[index] = activity_mode(activity_window)

        window_features = window_quality_features(
            ppg_window,
            float(results["dominant_frequency_hz"][index]),
            float(results["peak_power_ratio"][index]),
        )

        for feature in FEATURE_COLUMNS:
            feature_values[feature][index] = (
                window_features[feature]
            )

    groups = np.asarray(
        [
            error_group(float(value))
            for value in results["absolute_error_bpm"]
        ],
        dtype=object,
    )

    return {
        **results,
        **feature_values,
        "activity_id": activity_id,
        "error_group": groups,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    """计算带并列值平均名次的秩，不依赖 SciPy。"""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def safe_spearman(
    feature: np.ndarray,
    absolute_error: np.ndarray,
) -> float | None:
    """计算 Spearman 相关；常量特征返回 None。"""
    finite_mask = np.isfinite(feature) & np.isfinite(
        absolute_error
    )
    feature = feature[finite_mask]
    absolute_error = absolute_error[finite_mask]

    if (
        len(feature) < 2
        or float(np.std(feature)) <= 0
        or float(np.std(absolute_error)) <= 0
    ):
        return None

    correlation = float(
        np.corrcoef(
            average_ranks(feature),
            average_ranks(absolute_error),
        )[0, 1]
    )
    return correlation if np.isfinite(correlation) else None


def descriptive_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    """保存可复核的窗口数和稳健分布统计。"""
    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:
        return {"count": 0}

    return {
        "count": int(len(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "p10": float(np.percentile(finite_values, 10)),
        "p90": float(np.percentile(finite_values, 90)),
    }


def write_feature_csv(
    results: dict[str, np.ndarray],
) -> Path:
    """保存全部窗口的输入、评价结果和候选质量特征。"""
    output_path = OUTPUT_DIR / "S1_ppg_sqi_features.csv"
    fieldnames = (
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

        for index in range(
            len(results["window_center_s"])
        ):
            activity_id = int(results["activity_id"][index])
            row: dict[str, str | int] = {
                "window_start_s": (
                    f"{results['window_start_s'][index]:.3f}"
                ),
                "window_center_s": (
                    f"{results['window_center_s'][index]:.3f}"
                ),
                "window_end_s": (
                    f"{results['window_end_s'][index]:.3f}"
                ),
                "activity_id": activity_id,
                "activity_name": ACTIVITY_NAMES[activity_id],
                "ppg_hr_bpm": (
                    f"{results['estimated_hr_bpm'][index]:.6f}"
                ),
                "ecg_hr_bpm": (
                    f"{results['reference_hr_bpm'][index]:.6f}"
                ),
                "absolute_error_bpm": (
                    f"{results['absolute_error_bpm'][index]:.6f}"
                ),
                "error_group": str(
                    results["error_group"][index]
                ),
            }

            for feature in FEATURE_COLUMNS:
                row[feature] = (
                    f"{results[feature][index]:.9f}"
                )

            writer.writerow(row)

    return output_path


def build_summary(
    results: dict[str, np.ndarray],
) -> dict[str, Any]:
    """构建参数、分组统计和特征—误差相关性摘要。"""
    absolute_error = results["absolute_error_bpm"]
    correlations = {
        feature: safe_spearman(
            results[feature],
            absolute_error,
        )
        for feature in FEATURE_COLUMNS
    }

    group_summary: dict[str, Any] = {}
    for group in ERROR_GROUPS:
        mask = results["error_group"] == group
        group_summary[group] = {
            "label": ERROR_GROUP_LABELS[group],
            "window_count": int(np.sum(mask)),
            "absolute_error_bpm": descriptive_statistics(
                absolute_error[mask]
            ),
            "features": {
                feature: descriptive_statistics(
                    results[feature][mask]
                )
                for feature in FEATURE_COLUMNS
            },
        }

    activity_counts = {
        ACTIVITY_NAMES[int(activity_id)]: int(count)
        for activity_id, count in zip(
            *np.unique(
                results["activity_id"],
                return_counts=True,
            )
        )
    }

    return {
        "analysis": "PPG-only candidate SQI exploration V0",
        "subject": "S1",
        "candidate_sqi_only": True,
        "formal_quality_thresholds_defined": False,
        "feature_uses_reference_hr": False,
        "reference_hr_used_only_for_offline_evaluation": True,
        "feature_uses_acc": False,
        "parameters": {
            "ppg_sampling_rate_hz": PPG_FS_HZ,
            "activity_sampling_rate_hz": ACTIVITY_FS_HZ,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "n_fft": N_FFT,
            "candidate_hr_hz": [MIN_HR_HZ, MAX_HR_HZ],
            "second_peak_min_separation_hz": (
                SECOND_PEAK_MIN_SEPARATION_HZ
            ),
            "derivative_outlier_mad_factor": (
                DERIVATIVE_OUTLIER_MAD_FACTOR
            ),
            "research_error_groups_bpm": {
                "accurate": "<=3",
                "intermediate": ">3 and <=10",
                "failure": ">10",
            },
        },
        "window_count": int(
            len(results["window_center_s"])
        ),
        "activity_window_counts": activity_counts,
        "spearman_with_absolute_hr_error": correlations,
        "error_group_summary": group_summary,
        "limitations": [
            "S1 exploration only; no product threshold is defined",
            "HR error is an indirect evaluation target, not an SQI label",
            "E4 BVP is device-exported data, not per-channel ADC data",
            "extreme_repeat_ratio is only a clipping candidate feature",
            "periodic motion can also produce strong spectral and autocorrelation peaks",
        ],
    }


def write_summary_json(
    summary: dict[str, Any],
) -> Path:
    """保存分析参数和统计结果。"""
    output_path = OUTPUT_DIR / "S1_sqi_summary.json"

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )
        json_file.write("\n")

    return output_path


def plot_feature_vs_error(
    results: dict[str, np.ndarray],
    correlations: dict[str, float | None],
) -> Path:
    """绘制每项 PPG 特征与 HR 绝对误差的关系。"""
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 13),
        constrained_layout=True,
    )
    absolute_error = results["absolute_error_bpm"]

    for axis, feature in zip(
        axes.flat,
        FEATURE_COLUMNS,
    ):
        for group in ERROR_GROUPS:
            mask = results["error_group"] == group
            axis.scatter(
                results[feature][mask],
                absolute_error[mask],
                s=8,
                alpha=0.22,
                color=ERROR_GROUP_COLORS[group],
                label=ERROR_GROUP_LABELS[group],
                edgecolors="none",
            )

        correlation = correlations[feature]
        correlation_text = (
            "NA"
            if correlation is None
            else f"{correlation:.3f}"
        )
        axis.set_xlabel(FEATURE_LABELS[feature])
        axis.set_ylabel("HR 绝对误差 (bpm)")
        axis.set_title(
            f"{FEATURE_LABELS[feature]}\n"
            f"Spearman ρ={correlation_text}"
        )
        axis.grid(alpha=0.25)

    axes.flat[0].legend(
        loc="upper right",
        fontsize=8,
    )
    figure.suptitle(
        "S1 全时段：PPG 候选质量特征与心率误差",
        fontsize=16,
    )

    output_path = OUTPUT_DIR / (
        "S1_sqi_feature_vs_error.png"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_feature_distributions(
    results: dict[str, np.ndarray],
) -> Path:
    """比较三个误差组中的候选质量特征分布。"""
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 13),
        constrained_layout=True,
    )

    for axis, feature in zip(
        axes.flat,
        FEATURE_COLUMNS,
    ):
        distributions = [
            results[feature][
                results["error_group"] == group
            ]
            for group in ERROR_GROUPS
        ]
        boxplot = axis.boxplot(
            distributions,
            tick_labels=("≤3", "3–10", ">10"),
            showfliers=False,
            patch_artist=True,
        )

        for box, group in zip(
            boxplot["boxes"],
            ERROR_GROUPS,
        ):
            box.set_facecolor(ERROR_GROUP_COLORS[group])
            box.set_alpha(0.55)

        axis.set_xlabel("HR 绝对误差分组 (bpm)")
        axis.set_ylabel(FEATURE_LABELS[feature])
        axis.set_title(FEATURE_LABELS[feature])
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(
        "S1 全时段：不同心率误差组的 PPG 特征分布",
        fontsize=16,
    )

    output_path = OUTPUT_DIR / (
        "S1_sqi_feature_distribution.png"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def print_summary(summary: dict[str, Any]) -> None:
    """在终端输出便于核对的研究摘要。"""
    print("PPG-DaLiA S1：PPG 候选 SQI 特征探索")
    print(f"窗口总数: {summary['window_count']}")

    print("\n误差分组：")
    for group in ERROR_GROUPS:
        group_result = summary["error_group_summary"][group]
        print(
            f"  {group_result['label']}: "
            f"{group_result['window_count']} 个窗口"
        )

    print("\n特征与 HR 绝对误差的 Spearman 相关：")
    for feature in FEATURE_COLUMNS:
        correlation = summary[
            "spearman_with_absolute_hr_error"
        ][feature]
        value = (
            "NA"
            if correlation is None
            else f"{correlation:.3f}"
        )
        print(f"  {feature}: {value}")

    print(
        "\n说明：这些是候选质量特征，"
        "尚未形成正式 SQI、阈值或质量状态。"
    )


def main() -> int:
    """运行 S1 全时段 PPG 质量特征探索。"""
    try:
        ppg, reference_hr, activity = load_s1()
        results = build_feature_results(
            ppg,
            reference_hr,
            activity,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        configure_plot_style()

        summary = build_summary(results)
        csv_path = write_feature_csv(results)
        json_path = write_summary_json(summary)
        scatter_path = plot_feature_vs_error(
            results,
            summary["spearman_with_absolute_hr_error"],
        )
        distribution_path = plot_feature_distributions(
            results
        )

        print_summary(summary)
        print("\n已生成可追溯结果：")
        print(f"  窗口特征: {csv_path}")
        print(f"  统计摘要: {json_path}")
        print(f"  特征—误差图: {scatter_path}")
        print(f"  分组分布图: {distribution_path}")
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"PPG SQI 特征探索运行失败：{error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
