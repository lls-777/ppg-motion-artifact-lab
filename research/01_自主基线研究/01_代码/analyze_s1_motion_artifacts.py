#!/usr/bin/env python3
"""分析 PPG-DaLiA S1 的运动信号、PPG 频谱与心率误差关系。"""

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
    build_window_results,
    configure_plot_style,
    remove_linear_trend,
)


DATA_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    DATA_ROOT.parent
    / "90_共享原始数据/ppg_dalia/PPG_FieldStudy/S1/S1.pkl"
)
OUTPUT_DIR = DATA_ROOT / "02_结果/motion_analysis"

PPG_FS_HZ = 64
ACC_FS_HZ = 32
ACTIVITY_FS_HZ = 4
WINDOW_SECONDS = 8
STEP_SECONDS = 2

PPG_WINDOW_SAMPLES = PPG_FS_HZ * WINDOW_SECONDS
ACC_WINDOW_SAMPLES = ACC_FS_HZ * WINDOW_SECONDS
ACTIVITY_WINDOW_SAMPLES = ACTIVITY_FS_HZ * WINDOW_SECONDS

PPG_STEP_SAMPLES = PPG_FS_HZ * STEP_SECONDS
ACC_STEP_SAMPLES = ACC_FS_HZ * STEP_SECONDS
ACTIVITY_STEP_SAMPLES = ACTIVITY_FS_HZ * STEP_SECONDS

PPG_N_FFT = 4096
ACC_N_FFT = 2048
DISPLAY_MAX_HZ = 4.0
MOTION_MIN_HZ = 0.25
MOTION_MAX_HZ = 4.0
PEAK_HALF_WIDTH_HZ = 0.10

BASELINE_START_S = 120.0
BASELINE_END_S = 180.0
STAIRS_START_S = 850.0
STAIRS_END_S = 910.0
ACCURATE_ERROR_BPM = 3.0

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


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """确认数据结构为字典。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} 应为字典，实际为 {type(value).__name__}。")
    return value


def require_key(mapping: dict[str, Any], key: str, location: str) -> Any:
    """读取必要字段，并在字段缺失时列出实际字段。"""
    if key not in mapping:
        available = ", ".join(mapping.keys())
        raise KeyError(
            f"在 {location} 中未找到 {key!r}；可用字段为：{available}"
        )
    return mapping[key]


def load_s1() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """只读加载同步后的腕部 BVP、ACC、ECG HR 和活动标签。"""
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
    acc = np.asarray(
        require_key(wrist, "ACC", "signal/wrist"),
        dtype=np.float64,
    )
    reference_hr = np.asarray(
        require_key(subject, "label", "S1.pkl 顶层"),
        dtype=np.float64,
    ).reshape(-1)
    activity = np.asarray(
        require_key(subject, "activity", "S1.pkl 顶层"),
        dtype=np.float64,
    ).reshape(-1)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"ACC 应为 N×3，实际 shape 为 {acc.shape}。")

    for name, values in (
        ("PPG", ppg),
        ("ACC", acc),
        ("ECG 参考心率", reference_hr),
        ("活动标签", activity),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 中存在非有限数值。")

    return ppg, acc, reference_hr, activity


def validate_window_counts(
    ppg: np.ndarray,
    acc: np.ndarray,
    reference_hr: np.ndarray,
    activity: np.ndarray,
) -> None:
    """确认四类数据可以按同一 8 秒/2 秒窗口逐窗对应。"""
    counts = {
        "PPG": 1
        + (len(ppg) - PPG_WINDOW_SAMPLES) // PPG_STEP_SAMPLES,
        "ACC": 1
        + (len(acc) - ACC_WINDOW_SAMPLES) // ACC_STEP_SAMPLES,
        "activity": 1
        + (
            len(activity) - ACTIVITY_WINDOW_SAMPLES
        )
        // ACTIVITY_STEP_SAMPLES,
        "reference_hr": len(reference_hr),
    }

    if len(set(counts.values())) != 1:
        raise ValueError(f"窗口数不一致，无法逐窗对齐：{counts}")


def power_spectrum(
    values: np.ndarray,
    sampling_rate_hz: float,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对单个窗口去线性趋势并计算加窗功率谱。"""
    detrended = remove_linear_trend(
        np.asarray(values, dtype=np.float64).reshape(-1)
    )
    tapered = detrended * np.hanning(len(detrended))
    spectrum = np.fft.rfft(tapered, n=n_fft)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sampling_rate_hz,
    )
    return frequency, power


def normalize_for_plot(
    frequency: np.ndarray,
    power: np.ndarray,
) -> np.ndarray:
    """将显示频段内的功率归一化，便于比较不同量纲信号。"""
    display_mask = (
        (frequency >= MOTION_MIN_HZ)
        & (frequency <= DISPLAY_MAX_HZ)
    )
    scale = float(np.max(power[display_mask]))
    if scale <= 0:
        return np.zeros_like(power)
    return power / scale


def dominant_motion_frequency(
    acc_window: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """从三轴 ACC 合成功率谱中提取运动主频与峰值集中度。"""
    combined_power: np.ndarray | None = None
    frequency: np.ndarray | None = None

    for axis in range(3):
        axis_frequency, axis_power = power_spectrum(
            acc_window[:, axis],
            ACC_FS_HZ,
            ACC_N_FFT,
        )
        if combined_power is None:
            frequency = axis_frequency
            combined_power = axis_power
        else:
            combined_power += axis_power

    if frequency is None or combined_power is None:
        raise ValueError("无法计算 ACC 合成功率谱。")

    candidate_mask = (
        (frequency >= MOTION_MIN_HZ)
        & (frequency <= MOTION_MAX_HZ)
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        raise ValueError("ACC 运动频段中没有可用频点。")

    peak_index = int(
        candidate_indices[np.argmax(combined_power[candidate_mask])]
    )
    dominant_hz = float(frequency[peak_index])

    peak_mask = candidate_mask & (
        np.abs(frequency - dominant_hz) <= PEAK_HALF_WIDTH_HZ
    )
    band_power = float(np.sum(combined_power[candidate_mask]))
    peak_ratio = (
        float(np.sum(combined_power[peak_mask])) / band_power
        if band_power > 0
        else 0.0
    )

    return dominant_hz, peak_ratio, frequency, combined_power


def activity_mode(activity_window: np.ndarray) -> int:
    """使用窗口内出现次数最多的活动编号作为窗口活动标签。"""
    labels = np.rint(activity_window).astype(np.int64)
    labels = labels[(labels >= 0) & (labels <= 8)]
    if len(labels) == 0:
        return 0
    return int(np.argmax(np.bincount(labels, minlength=9)))


def scene_name(start_s: float, end_s: float) -> str:
    """标记当前报告重点关注的静止或上下楼场景。"""
    if start_s >= BASELINE_START_S and end_s <= BASELINE_END_S:
        return "baseline"
    if start_s >= STAIRS_START_S and end_s <= STAIRS_END_S:
        return "stairs"
    return "other"


def harmonic_frequency_gap(
    ppg_frequency_hz: float,
    acc_frequency_hz: float,
) -> float:
    """计算 PPG 主峰与 ACC 主频前三阶谐波的最小距离。"""
    candidates = (
        acc_frequency_hz,
        2.0 * acc_frequency_hz,
        3.0 * acc_frequency_hz,
    )
    return float(
        min(abs(ppg_frequency_hz - value) for value in candidates)
    )


def build_motion_results(
    acc: np.ndarray,
    activity: np.ndarray,
    baseline_results: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """逐窗口计算可解释的 ACC 特征和 PPG-ACC 频率关系。"""
    window_count = len(baseline_results["window_start_s"])

    motion_rms_g = np.empty(window_count, dtype=np.float64)
    acc_peak_to_peak_g = np.empty(window_count, dtype=np.float64)
    jerk_rms_g_per_s = np.empty(window_count, dtype=np.float64)
    acc_dominant_frequency_hz = np.empty(
        window_count,
        dtype=np.float64,
    )
    acc_peak_power_ratio = np.empty(window_count, dtype=np.float64)
    ppg_acc_frequency_gap_hz = np.empty(
        window_count,
        dtype=np.float64,
    )
    ppg_acc_harmonic_gap_hz = np.empty(
        window_count,
        dtype=np.float64,
    )
    activity_id = np.empty(window_count, dtype=np.int64)

    for index in range(window_count):
        acc_start = index * ACC_STEP_SAMPLES
        acc_end = acc_start + ACC_WINDOW_SAMPLES
        acc_window = acc[acc_start:acc_end]

        activity_start = index * ACTIVITY_STEP_SAMPLES
        activity_end = activity_start + ACTIVITY_WINDOW_SAMPLES
        activity_window = activity[activity_start:activity_end]

        acc_magnitude = np.linalg.norm(acc_window, axis=1)
        motion_proxy = acc_magnitude - 1.0
        jerk = np.diff(acc_window, axis=0) * ACC_FS_HZ

        motion_rms_g[index] = float(
            np.sqrt(np.mean(motion_proxy**2))
        )
        acc_peak_to_peak_g[index] = float(np.ptp(acc_magnitude))
        jerk_rms_g_per_s[index] = float(
            np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        )

        (
            acc_dominant_frequency_hz[index],
            acc_peak_power_ratio[index],
            _,
            _,
        ) = dominant_motion_frequency(acc_window)

        ppg_frequency_hz = float(
            baseline_results["dominant_frequency_hz"][index]
        )
        acc_frequency_hz = acc_dominant_frequency_hz[index]
        ppg_acc_frequency_gap_hz[index] = abs(
            ppg_frequency_hz - acc_frequency_hz
        )
        ppg_acc_harmonic_gap_hz[index] = harmonic_frequency_gap(
            ppg_frequency_hz,
            acc_frequency_hz,
        )
        activity_id[index] = activity_mode(activity_window)

    return {
        **baseline_results,
        "activity_id": activity_id,
        "motion_rms_g": motion_rms_g,
        "acc_peak_to_peak_g": acc_peak_to_peak_g,
        "jerk_rms_g_per_s": jerk_rms_g_per_s,
        "acc_dominant_frequency_hz": acc_dominant_frequency_hz,
        "acc_peak_power_ratio": acc_peak_power_ratio,
        "ppg_acc_frequency_gap_hz": ppg_acc_frequency_gap_hz,
        "ppg_acc_harmonic_gap_hz": ppg_acc_harmonic_gap_hz,
    }


def time_mask(
    results: dict[str, np.ndarray],
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """选择完整落在指定时间段内的窗口。"""
    return (
        (results["window_start_s"] >= start_s)
        & (results["window_end_s"] <= end_s)
    )


def closest_to_median_index(
    indices: np.ndarray,
    values: np.ndarray,
) -> int:
    """选择最接近组内中位数的窗口，避免只展示最好样本。"""
    median = float(np.median(values[indices]))
    relative_index = int(
        np.argmin(np.abs(values[indices] - median))
    )
    return int(indices[relative_index])


def select_representative_windows(
    results: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """选择静止典型、楼梯较准和楼梯严重错误三个窗口。"""
    error = results["absolute_error_bpm"]
    baseline_indices = np.flatnonzero(
        time_mask(results, BASELINE_START_S, BASELINE_END_S)
    )
    stairs_indices = np.flatnonzero(
        time_mask(results, STAIRS_START_S, STAIRS_END_S)
    )

    if len(baseline_indices) == 0 or len(stairs_indices) == 0:
        raise ValueError("静止或上下楼场景中没有完整窗口。")

    baseline_accurate = baseline_indices[
        error[baseline_indices] <= ACCURATE_ERROR_BPM
    ]
    if len(baseline_accurate) == 0:
        baseline_accurate = baseline_indices

    stairs_accurate = stairs_indices[
        error[stairs_indices] <= ACCURATE_ERROR_BPM
    ]

    baseline_index = closest_to_median_index(
        baseline_accurate,
        error,
    )
    stairs_accurate_index = int(
        stairs_accurate[np.argmin(error[stairs_accurate])]
        if len(stairs_accurate) > 0
        else stairs_indices[np.argmin(error[stairs_indices])]
    )
    stairs_worst_index = int(
        stairs_indices[np.argmax(error[stairs_indices])]
    )

    return {
        "baseline_representative": {
            "index": baseline_index,
            "title": "静止且心率准确的代表窗口",
            "file_name": "S1_baseline_representative_spectrum.png",
            "selection": "静止准确窗口中误差最接近中位数",
        },
        "stairs_accurate": {
            "index": stairs_accurate_index,
            "title": "上下楼但心率相对准确的窗口",
            "file_name": "S1_stairs_accurate_spectrum.png",
            "selection": "上下楼窗口中绝对误差最小",
        },
        "stairs_severe_error": {
            "index": stairs_worst_index,
            "title": "上下楼且心率严重错误的窗口",
            "file_name": "S1_stairs_severe_error_spectrum.png",
            "selection": "上下楼窗口中绝对误差最大",
        },
    }


def plot_representative_window(
    ppg: np.ndarray,
    acc: np.ndarray,
    results: dict[str, np.ndarray],
    case: dict[str, Any],
) -> Path:
    """绘制一个代表窗口的时域波形和PPG/ACC频谱证据。"""
    index = int(case["index"])
    start_s = float(results["window_start_s"][index])
    end_s = float(results["window_end_s"][index])

    ppg_start = index * PPG_STEP_SAMPLES
    ppg_window = ppg[
        ppg_start : ppg_start + PPG_WINDOW_SAMPLES
    ]
    acc_start = index * ACC_STEP_SAMPLES
    acc_window = acc[
        acc_start : acc_start + ACC_WINDOW_SAMPLES
    ]

    ppg_time = np.arange(len(ppg_window)) / PPG_FS_HZ
    acc_time = np.arange(len(acc_window)) / ACC_FS_HZ
    motion_proxy = np.abs(np.linalg.norm(acc_window, axis=1) - 1.0)

    ppg_frequency, ppg_power = power_spectrum(
        ppg_window,
        PPG_FS_HZ,
        PPG_N_FFT,
    )
    (
        acc_dominant_hz,
        _,
        acc_frequency,
        acc_power,
    ) = dominant_motion_frequency(acc_window)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        constrained_layout=True,
    )

    axes[0].plot(ppg_time, ppg_window, linewidth=0.9)
    axes[0].set_ylabel("BVP")
    axes[0].set_title("设备导出的单路腕部 BVP（本脚本未额外滤波）")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        acc_time,
        motion_proxy,
        color="#ff7f0e",
        linewidth=0.9,
    )
    axes[1].set_ylabel("|‖ACC‖ - 1 g|")
    axes[1].set_title("三轴 ACC 运动强度代理")
    axes[1].grid(alpha=0.25)

    ppg_display = normalize_for_plot(ppg_frequency, ppg_power)
    acc_display = normalize_for_plot(acc_frequency, acc_power)
    axes[2].plot(
        ppg_frequency,
        ppg_display,
        label="PPG归一化功率谱",
        color="#1f77b4",
    )
    axes[2].plot(
        acc_frequency,
        acc_display,
        label="三轴ACC合成功率谱",
        color="#ff7f0e",
        alpha=0.85,
    )

    ppg_hr_hz = float(
        results["dominant_frequency_hz"][index]
    )
    ecg_hr_hz = float(results["reference_hr_bpm"][index]) / 60.0
    axes[2].axvline(
        ppg_hr_hz,
        color="#d62728",
        linestyle="--",
        label=f"PPG选中频率 {ppg_hr_hz:.2f} Hz",
    )
    axes[2].axvline(
        ecg_hr_hz,
        color="#2ca02c",
        linestyle="--",
        label=f"ECG参考频率 {ecg_hr_hz:.2f} Hz",
    )
    axes[2].axvline(
        acc_dominant_hz,
        color="#9467bd",
        linestyle=":",
        label=f"ACC主频 {acc_dominant_hz:.2f} Hz",
    )
    axes[2].set_xlim(0, DISPLAY_MAX_HZ)
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlabel("频率（Hz）")
    axes[2].set_ylabel("归一化功率")
    axes[2].set_title("PPG与ACC频谱对照")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(alpha=0.25)

    figure.suptitle(
        f"{case['title']}：{start_s:.0f}–{end_s:.0f}秒\n"
        f"PPG HR={results['estimated_hr_bpm'][index]:.2f} bpm，"
        f"ECG HR={results['reference_hr_bpm'][index]:.2f} bpm，"
        f"绝对误差={results['absolute_error_bpm'][index]:.2f} bpm",
        fontsize=14,
    )

    output_path = OUTPUT_DIR / str(case["file_name"])
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """计算探索性Pearson相关；常量数组返回NaN。"""
    if len(first) < 2:
        return float("nan")
    if float(np.std(first)) == 0 or float(np.std(second)) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def plot_motion_error_scatter(
    results: dict[str, np.ndarray],
) -> tuple[Path, float]:
    """绘制S1全部窗口的运动强度与心率误差散点图。"""
    motion = results["motion_rms_g"]
    error = results["absolute_error_bpm"]
    activity_id = results["activity_id"]
    correlation = safe_correlation(motion, error)

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(13, 8),
        constrained_layout=True,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(ACTIVITY_NAMES)))
    for activity_value, color in zip(ACTIVITY_NAMES, colors):
        mask = activity_id == activity_value
        if not np.any(mask):
            continue
        axis.scatter(
            motion[mask],
            error[mask],
            s=18,
            alpha=0.48,
            color=color,
            label=(
                f"{activity_value} "
                f"{ACTIVITY_NAMES[activity_value]}"
            ),
        )

    axis.set_xlabel("ACC运动强度 Motion RMS（g）")
    axis.set_ylabel("PPG-only心率绝对误差（bpm）")
    axis.set_title(
        "S1全部窗口：ACC运动强度与PPG-only心率误差\n"
        f"探索性Pearson相关 r={correlation:.3f}；"
        "相关不等于因果"
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)

    output_path = OUTPUT_DIR / "S1_motion_vs_hr_error.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path, correlation


def write_window_csv(results: dict[str, np.ndarray]) -> Path:
    """保存全部窗口的心率、活动和运动特征。"""
    output_path = OUTPUT_DIR / "S1_motion_window_features.csv"
    fieldnames = (
        "window_start_s",
        "window_center_s",
        "window_end_s",
        "scene",
        "activity_id",
        "activity_name",
        "ppg_hr_bpm",
        "ecg_hr_bpm",
        "absolute_error_bpm",
        "ppg_dominant_frequency_hz",
        "ppg_peak_power_ratio",
        "motion_rms_g",
        "acc_peak_to_peak_g",
        "jerk_rms_g_per_s",
        "acc_dominant_frequency_hz",
        "acc_peak_power_ratio",
        "ppg_acc_frequency_gap_hz",
        "ppg_acc_harmonic_gap_hz",
    )

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(len(results["window_start_s"])):
            start_s = float(results["window_start_s"][index])
            end_s = float(results["window_end_s"][index])
            activity_id = int(results["activity_id"][index])
            writer.writerow(
                {
                    "window_start_s": f"{start_s:.3f}",
                    "window_center_s": (
                        f"{results['window_center_s'][index]:.3f}"
                    ),
                    "window_end_s": f"{end_s:.3f}",
                    "scene": scene_name(start_s, end_s),
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
                    "ppg_dominant_frequency_hz": (
                        f"{results['dominant_frequency_hz'][index]:.6f}"
                    ),
                    "ppg_peak_power_ratio": (
                        f"{results['peak_power_ratio'][index]:.6f}"
                    ),
                    "motion_rms_g": (
                        f"{results['motion_rms_g'][index]:.6f}"
                    ),
                    "acc_peak_to_peak_g": (
                        f"{results['acc_peak_to_peak_g'][index]:.6f}"
                    ),
                    "jerk_rms_g_per_s": (
                        f"{results['jerk_rms_g_per_s'][index]:.6f}"
                    ),
                    "acc_dominant_frequency_hz": (
                        f"{results['acc_dominant_frequency_hz'][index]:.6f}"
                    ),
                    "acc_peak_power_ratio": (
                        f"{results['acc_peak_power_ratio'][index]:.6f}"
                    ),
                    "ppg_acc_frequency_gap_hz": (
                        f"{results['ppg_acc_frequency_gap_hz'][index]:.6f}"
                    ),
                    "ppg_acc_harmonic_gap_hz": (
                        f"{results['ppg_acc_harmonic_gap_hz'][index]:.6f}"
                    ),
                }
            )

    return output_path


def scene_summary(
    results: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float | int]:
    """汇总一个场景的运动特征和心率误差。"""
    return {
        "window_count": int(np.sum(mask)),
        "mean_motion_rms_g": float(
            np.mean(results["motion_rms_g"][mask])
        ),
        "median_motion_rms_g": float(
            np.median(results["motion_rms_g"][mask])
        ),
        "mean_jerk_rms_g_per_s": float(
            np.mean(results["jerk_rms_g_per_s"][mask])
        ),
        "mean_hr_absolute_error_bpm": float(
            np.mean(results["absolute_error_bpm"][mask])
        ),
        "motion_error_correlation": safe_correlation(
            results["motion_rms_g"][mask],
            results["absolute_error_bpm"][mask],
        ),
    }


def representative_summary(
    results: dict[str, np.ndarray],
    case: dict[str, Any],
) -> dict[str, Any]:
    """保存一个代表窗口的选择依据和关键数值。"""
    index = int(case["index"])
    activity_id = int(results["activity_id"][index])
    return {
        "window_index": index,
        "selection": str(case["selection"]),
        "window_start_s": float(results["window_start_s"][index]),
        "window_center_s": float(results["window_center_s"][index]),
        "window_end_s": float(results["window_end_s"][index]),
        "activity_id": activity_id,
        "activity_name": ACTIVITY_NAMES[activity_id],
        "ppg_hr_bpm": float(results["estimated_hr_bpm"][index]),
        "ecg_hr_bpm": float(results["reference_hr_bpm"][index]),
        "absolute_error_bpm": float(
            results["absolute_error_bpm"][index]
        ),
        "motion_rms_g": float(results["motion_rms_g"][index]),
        "acc_dominant_frequency_hz": float(
            results["acc_dominant_frequency_hz"][index]
        ),
        "ppg_dominant_frequency_hz": float(
            results["dominant_frequency_hz"][index]
        ),
        "ppg_acc_frequency_gap_hz": float(
            results["ppg_acc_frequency_gap_hz"][index]
        ),
        "ppg_acc_harmonic_gap_hz": float(
            results["ppg_acc_harmonic_gap_hz"][index]
        ),
    }


def write_summary_json(
    results: dict[str, np.ndarray],
    representatives: dict[str, dict[str, Any]],
    overall_correlation: float,
) -> Path:
    """保存参数、公式定义、场景结果和代表窗口。"""
    baseline_mask = time_mask(
        results,
        BASELINE_START_S,
        BASELINE_END_S,
    )
    stairs_mask = time_mask(
        results,
        STAIRS_START_S,
        STAIRS_END_S,
    )

    activity_summaries: dict[str, dict[str, float | int]] = {}
    for activity_id, activity_name in ACTIVITY_NAMES.items():
        mask = results["activity_id"] == activity_id
        if np.any(mask):
            activity_summaries[
                f"{activity_id}_{activity_name}"
            ] = scene_summary(results, mask)

    payload = {
        "analysis": "PPG-DaLiA S1 motion-artifact evidence V0",
        "subject": "S1",
        "data_provenance": {
            "ppg": (
                "Empatica E4 device-exported single-channel BVP; "
                "not per-LED/photodiode ADC data"
            ),
            "acc": "Empatica E4 wrist three-axis ACC",
            "reference_hr": "ECG-derived 8-second mean HR label",
            "activity": "protocol activity annotation",
        },
        "parameters": {
            "ppg_sampling_rate_hz": PPG_FS_HZ,
            "acc_sampling_rate_hz": ACC_FS_HZ,
            "activity_sampling_rate_hz": ACTIVITY_FS_HZ,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "motion_frequency_band_hz": [
                MOTION_MIN_HZ,
                MOTION_MAX_HZ,
            ],
        },
        "feature_definitions": {
            "motion_rms_g": (
                "sqrt(mean((norm(ACC_xyz) - 1 g)^2))"
            ),
            "acc_peak_to_peak_g": (
                "max(norm(ACC_xyz)) - min(norm(ACC_xyz))"
            ),
            "jerk_rms_g_per_s": (
                "RMS of the three-axis first difference multiplied "
                "by ACC sampling rate"
            ),
            "acc_dominant_frequency_hz": (
                "maximum of summed X/Y/Z ACC power spectra in "
                "0.25-4.0 Hz"
            ),
            "ppg_acc_frequency_gap_hz": (
                "absolute difference between PPG-selected frequency "
                "and ACC dominant frequency"
            ),
            "ppg_acc_harmonic_gap_hz": (
                "minimum difference between PPG-selected frequency "
                "and the first three ACC harmonics"
            ),
        },
        "overall_motion_error_correlation": overall_correlation,
        "scenes": {
            "baseline_120_180_s": scene_summary(
                results,
                baseline_mask,
            ),
            "stairs_850_910_s": scene_summary(
                results,
                stairs_mask,
            ),
        },
        "activities": activity_summaries,
        "representative_windows": {
            key: representative_summary(results, case)
            for key, case in representatives.items()
        },
        "limitations": [
            "Correlation and spectral overlap do not prove causality.",
            "The E4 BVP is a device-exported combined channel, not raw ADC.",
            "Feature thresholds and a formal SQI score are not defined yet.",
            "Results from S1 cannot be treated as cross-subject performance.",
        ],
    }

    output_path = OUTPUT_DIR / "S1_motion_analysis_summary.json"
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")
    return output_path


def print_case(
    key: str,
    case: dict[str, Any],
    results: dict[str, np.ndarray],
) -> None:
    """在终端打印一个代表窗口的核心数据。"""
    summary = representative_summary(results, case)
    print(f"\n{key}：{case['title']}")
    print(
        "  窗口: "
        f"{summary['window_start_s']:.0f}–"
        f"{summary['window_end_s']:.0f} 秒"
    )
    print(f"  选择依据: {summary['selection']}")
    print(f"  活动: {summary['activity_name']}")
    print(f"  PPG HR: {summary['ppg_hr_bpm']:.2f} bpm")
    print(f"  ECG HR: {summary['ecg_hr_bpm']:.2f} bpm")
    print(f"  绝对误差: {summary['absolute_error_bpm']:.2f} bpm")
    print(f"  Motion RMS: {summary['motion_rms_g']:.4f} g")
    print(
        "  PPG/ACC 主频: "
        f"{summary['ppg_dominant_frequency_hz']:.3f}/"
        f"{summary['acc_dominant_frequency_hz']:.3f} Hz"
    )


def main() -> int:
    """生成S1运动伪影的可追溯证据。"""
    try:
        ppg, acc, reference_hr, activity = load_s1()
        validate_window_counts(ppg, acc, reference_hr, activity)

        baseline_results = build_window_results(ppg, reference_hr)
        results = build_motion_results(
            acc,
            activity,
            baseline_results,
        )
        representatives = select_representative_windows(results)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        configure_plot_style()

        figure_paths = [
            plot_representative_window(
                ppg,
                acc,
                results,
                case,
            )
            for case in representatives.values()
        ]
        scatter_path, overall_correlation = (
            plot_motion_error_scatter(results)
        )
        figure_paths.append(scatter_path)

        csv_path = write_window_csv(results)
        json_path = write_summary_json(
            results,
            representatives,
            overall_correlation,
        )

        print("PPG-DaLiA S1 运动伪影分析完成")
        print(
            "  全S1 Motion RMS与HR绝对误差Pearson相关: "
            f"{overall_correlation:.3f}"
        )

        for key, case in representatives.items():
            print_case(key, case, results)

        print("\n已生成：")
        for figure_path in figure_paths:
            print(f"  图片: {figure_path}")
        print(f"  窗口特征: {csv_path}")
        print(f"  分析摘要: {json_path}")
        print(
            "\n边界：当前结果用于寻找候选关系，"
            "不能仅凭频谱接近或相关性宣称因果。"
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as error:
        print(f"运动伪影分析失败：{error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
