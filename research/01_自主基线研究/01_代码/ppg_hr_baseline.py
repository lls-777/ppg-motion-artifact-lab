#!/usr/bin/env python3
"""PPG-DaLiA S1 的可解释 PPG-only 窗口级心率基线。"""

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


DATA_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    DATA_ROOT.parent
    / "90_共享原始数据/ppg_dalia/PPG_FieldStudy/S1/S1.pkl"
)
OUTPUT_DIR = DATA_ROOT / "02_结果/hr_baseline"

PPG_FS_HZ = 64
WINDOW_SECONDS = 8
STEP_SECONDS = 2
WINDOW_SAMPLES = PPG_FS_HZ * WINDOW_SECONDS
STEP_SAMPLES = PPG_FS_HZ * STEP_SECONDS

# V0 仅在宽泛的生理心率范围内寻找 PPG 频谱主峰。
MIN_HR_BPM = 30.0
MAX_HR_BPM = 220.0
MIN_HR_HZ = MIN_HR_BPM / 60.0
MAX_HR_HZ = MAX_HR_BPM / 60.0

# 零填充用于更细致地插值谱峰位置，不代表增加了真实频率分辨率。
N_FFT = 4096
PEAK_BAND_HALF_WIDTH_HZ = 0.10

SCENES = (
    {
        "key": "baseline",
        "title": "S1 静止基线",
        "start_s": 120.0,
        "end_s": 180.0,
        "output_name": "S1_baseline_hr_comparison.png",
    },
    {
        "key": "stairs",
        "title": "S1 上下楼",
        "start_s": 850.0,
        "end_s": 910.0,
        "output_name": "S1_stairs_hr_comparison.png",
    },
)


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """确认数据结构为字典，避免产生难理解的索引错误。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} 应为字典，实际为 {type(value).__name__}。")
    return value


def require_key(mapping: dict[str, Any], key: str, location: str) -> Any:
    """读取必要字段，并在数据版本不符时输出实际可用字段。"""
    if key not in mapping:
        available = ", ".join(mapping.keys())
        raise KeyError(
            f"在 {location} 中未找到 {key!r}；可用字段为：{available}"
        )
    return mapping[key]


def load_s1() -> tuple[np.ndarray, np.ndarray]:
    """只读加载腕部 PPG 和 ECG 导出的窗口级心率标签。"""
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

    if not np.all(np.isfinite(ppg)):
        raise ValueError("PPG 中存在非有限数值。")
    if not np.all(np.isfinite(reference_hr)):
        raise ValueError("ECG 参考心率中存在非有限数值。")

    return ppg, reference_hr


def remove_linear_trend(values: np.ndarray) -> np.ndarray:
    """使用闭式线性拟合去除单个窗口的均值与线性漂移。"""
    sample_index = np.arange(len(values), dtype=np.float64)
    centered_index = sample_index - np.mean(sample_index)
    centered_values = values - np.mean(values)
    denominator = float(np.dot(centered_index, centered_index))

    if denominator <= 0:
        return centered_values

    slope = float(np.dot(centered_index, centered_values) / denominator)
    return centered_values - slope * centered_index


def interpolate_spectral_peak(
    power: np.ndarray,
    peak_index: int,
) -> float:
    """用相邻三个对数功率点对谱峰位置进行抛物线插值。"""
    if peak_index <= 0 or peak_index >= len(power) - 1:
        return float(peak_index)

    tiny = np.finfo(np.float64).tiny
    left, center, right = np.log(
        power[peak_index - 1 : peak_index + 2] + tiny
    )
    denominator = left - 2.0 * center + right

    if abs(denominator) < 1e-12:
        return float(peak_index)

    offset = 0.5 * (left - right) / denominator
    return float(peak_index + np.clip(offset, -1.0, 1.0))


def estimate_window_hr(
    ppg_window: np.ndarray,
) -> tuple[float, float, float]:
    """仅根据一个 PPG 窗口估计 HR，不读取 ECG 标签或 ACC。"""
    detrended = remove_linear_trend(ppg_window)

    if float(np.std(detrended)) < 1e-8:
        return float("nan"), float("nan"), 0.0

    tapered = detrended * np.hanning(len(detrended))
    spectrum = np.fft.rfft(tapered, n=N_FFT)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / PPG_FS_HZ)

    candidate_mask = (
        (frequencies >= MIN_HR_HZ)
        & (frequencies <= MAX_HR_HZ)
    )
    candidate_indices = np.flatnonzero(candidate_mask)

    if len(candidate_indices) == 0:
        return float("nan"), float("nan"), 0.0

    peak_index = int(
        candidate_indices[np.argmax(power[candidate_mask])]
    )
    interpolated_index = interpolate_spectral_peak(power, peak_index)
    peak_frequency_hz = interpolated_index * PPG_FS_HZ / N_FFT
    peak_frequency_hz = float(
        np.clip(peak_frequency_hz, MIN_HR_HZ, MAX_HR_HZ)
    )
    estimated_hr_bpm = peak_frequency_hz * 60.0

    # 记录主峰邻域能量占候选心率频段总能量的比例。
    peak_band_mask = candidate_mask & (
        np.abs(frequencies - peak_frequency_hz)
        <= PEAK_BAND_HALF_WIDTH_HZ
    )
    candidate_power = float(np.sum(power[candidate_mask]))
    peak_band_power = float(np.sum(power[peak_band_mask]))
    peak_power_ratio = (
        peak_band_power / candidate_power
        if candidate_power > 0
        else 0.0
    )

    return estimated_hr_bpm, peak_frequency_hz, peak_power_ratio


def build_window_results(
    ppg: np.ndarray,
    reference_hr: np.ndarray,
) -> dict[str, np.ndarray]:
    """按照 8 秒窗口、2 秒步长计算全部 PPG-only HR。"""
    window_count = 1 + (len(ppg) - WINDOW_SAMPLES) // STEP_SAMPLES

    if window_count != len(reference_hr):
        raise ValueError(
            "PPG 窗口数与 ECG 标签数不一致："
            f"PPG 窗口={window_count}，ECG 标签={len(reference_hr)}。"
        )

    window_start_samples = (
        np.arange(window_count, dtype=np.int64) * STEP_SAMPLES
    )
    window_start_s = window_start_samples / PPG_FS_HZ
    window_end_s = window_start_s + WINDOW_SECONDS
    window_center_s = window_start_s + WINDOW_SECONDS / 2

    estimated_hr = np.empty(window_count, dtype=np.float64)
    dominant_frequency_hz = np.empty(window_count, dtype=np.float64)
    peak_power_ratio = np.empty(window_count, dtype=np.float64)

    for index, start_sample in enumerate(window_start_samples):
        end_sample = start_sample + WINDOW_SAMPLES
        ppg_window = ppg[start_sample:end_sample]
        (
            estimated_hr[index],
            dominant_frequency_hz[index],
            peak_power_ratio[index],
        ) = estimate_window_hr(ppg_window)

    if not np.all(np.isfinite(estimated_hr)):
        invalid_count = int(np.sum(~np.isfinite(estimated_hr)))
        raise ValueError(f"{invalid_count} 个窗口未能产生有限 HR。")

    if np.any(estimated_hr < MIN_HR_BPM) or np.any(
        estimated_hr > MAX_HR_BPM
    ):
        raise ValueError("存在超出设定搜索范围的 PPG HR。")

    absolute_error = np.abs(estimated_hr - reference_hr)

    return {
        "window_start_s": window_start_s,
        "window_center_s": window_center_s,
        "window_end_s": window_end_s,
        "estimated_hr_bpm": estimated_hr,
        "reference_hr_bpm": reference_hr,
        "absolute_error_bpm": absolute_error,
        "dominant_frequency_hz": dominant_frequency_hz,
        "peak_power_ratio": peak_power_ratio,
    }


def scene_mask(
    results: dict[str, np.ndarray],
    scene: dict[str, Any],
) -> np.ndarray:
    """选择完整落在指定场景时间段内的 8 秒窗口。"""
    return (
        (results["window_start_s"] >= scene["start_s"])
        & (results["window_end_s"] <= scene["end_s"])
    )


def scene_name_for_window(start_s: float, end_s: float) -> str:
    """为 CSV 中的窗口标记当前关注场景。"""
    for scene in SCENES:
        if start_s >= scene["start_s"] and end_s <= scene["end_s"]:
            return str(scene["key"])
    return "other"


def calculate_metrics(
    results: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float | int]:
    """计算一个场景的窗口级 HR 评价指标。"""
    estimated = results["estimated_hr_bpm"][mask]
    reference = results["reference_hr_bpm"][mask]
    absolute_error = results["absolute_error_bpm"][mask]
    peak_ratio = results["peak_power_ratio"][mask]

    if len(estimated) == 0:
        raise ValueError("指定场景中没有完整的 8 秒窗口。")

    return {
        "window_count": int(len(estimated)),
        "mae_bpm": float(np.mean(absolute_error)),
        "median_absolute_error_bpm": float(np.median(absolute_error)),
        "rmse_bpm": float(
            np.sqrt(np.mean((estimated - reference) ** 2))
        ),
        "p90_absolute_error_bpm": float(
            np.percentile(absolute_error, 90)
        ),
        "max_absolute_error_bpm": float(np.max(absolute_error)),
        "estimated_hr_min_bpm": float(np.min(estimated)),
        "estimated_hr_max_bpm": float(np.max(estimated)),
        "reference_hr_min_bpm": float(np.min(reference)),
        "reference_hr_max_bpm": float(np.max(reference)),
        "mean_peak_power_ratio": float(np.mean(peak_ratio)),
    }


def write_window_csv(results: dict[str, np.ndarray]) -> Path:
    """保存全部窗口结果，便于后续 SQI 和运动分析复用。"""
    output_path = OUTPUT_DIR / "S1_hr_baseline_windows.csv"
    fieldnames = (
        "window_start_s",
        "window_center_s",
        "window_end_s",
        "scene",
        "ppg_hr_bpm",
        "ecg_hr_bpm",
        "absolute_error_bpm",
        "dominant_frequency_hz",
        "peak_power_ratio",
    )

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(len(results["window_start_s"])):
            start_s = float(results["window_start_s"][index])
            end_s = float(results["window_end_s"][index])
            writer.writerow(
                {
                    "window_start_s": f"{start_s:.3f}",
                    "window_center_s": (
                        f"{results['window_center_s'][index]:.3f}"
                    ),
                    "window_end_s": f"{end_s:.3f}",
                    "scene": scene_name_for_window(start_s, end_s),
                    "ppg_hr_bpm": (
                        f"{results['estimated_hr_bpm'][index]:.6f}"
                    ),
                    "ecg_hr_bpm": (
                        f"{results['reference_hr_bpm'][index]:.6f}"
                    ),
                    "absolute_error_bpm": (
                        f"{results['absolute_error_bpm'][index]:.6f}"
                    ),
                    "dominant_frequency_hz": (
                        f"{results['dominant_frequency_hz'][index]:.6f}"
                    ),
                    "peak_power_ratio": (
                        f"{results['peak_power_ratio'][index]:.6f}"
                    ),
                }
            )

    return output_path


def write_metrics_json(
    scene_metrics: dict[str, dict[str, float | int]],
) -> Path:
    """保存算法参数和真实指标，保证报告数字可追溯。"""
    output_path = OUTPUT_DIR / "S1_hr_baseline_metrics.json"
    payload = {
        "algorithm": "PPG-only spectral baseline V0",
        "subject": "S1",
        "estimator_uses_reference_label": False,
        "estimator_uses_acc": False,
        "parameters": {
            "ppg_sampling_rate_hz": PPG_FS_HZ,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "min_hr_bpm": MIN_HR_BPM,
            "max_hr_bpm": MAX_HR_BPM,
            "n_fft": N_FFT,
            "preprocessing": [
                "mean removal",
                "linear detrending",
                "Hann window",
            ],
            "estimator": "maximum spectral peak with parabolic interpolation",
            "peak_power_ratio_definition": (
                "energy within ±0.10 Hz of the selected peak divided by "
                "total energy in the candidate HR band"
            ),
        },
        "scenes": scene_metrics,
        "limitations": [
            "PPG-only baseline; ACC is not used for estimation",
            "zero-padding interpolates the spectrum but does not add information",
            "results cover S1 and cannot be treated as product performance",
            "peak_power_ratio is a candidate quality feature, not confidence",
        ],
    }

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")

    return output_path


def configure_plot_style() -> None:
    """配置适用于 macOS 和报告图片的中文字体。"""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_scene(
    ppg: np.ndarray,
    results: dict[str, np.ndarray],
    scene: dict[str, Any],
    metrics: dict[str, float | int],
) -> Path:
    """绘制原始 PPG、谱峰质量、HR 对比和绝对误差。"""
    mask = scene_mask(results, scene)
    start_sample = int(scene["start_s"] * PPG_FS_HZ)
    end_sample = int(scene["end_s"] * PPG_FS_HZ)
    ppg_time = np.arange(start_sample, end_sample) / PPG_FS_HZ
    ppg_segment = ppg[start_sample:end_sample]

    window_time = results["window_center_s"][mask]
    estimated_hr = results["estimated_hr_bpm"][mask]
    reference_hr = results["reference_hr_bpm"][mask]
    absolute_error = results["absolute_error_bpm"][mask]
    peak_power_ratio = results["peak_power_ratio"][mask]

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        ppg_time,
        ppg_segment,
        color="#1f77b4",
        linewidth=0.7,
    )
    axes[0].set_ylabel("PPG / BVP")
    axes[0].set_title(
        f"{scene['title']}：PPG-only HR 基线"
        f"（MAE={metrics['mae_bpm']:.2f} bpm，"
        f"窗口数={metrics['window_count']}）"
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        window_time,
        peak_power_ratio,
        color="#9467bd",
        marker="o",
        markersize=3,
        linewidth=1,
    )
    axes[1].set_ylabel("主峰带能量占比")
    axes[1].set_title("PPG 频谱主峰突出程度（候选质量特征，非可信度）")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        window_time,
        reference_hr,
        color="#2ca02c",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="ECG 参考 HR",
    )
    axes[2].plot(
        window_time,
        estimated_hr,
        color="#d62728",
        marker="x",
        markersize=4,
        linewidth=1.2,
        label="PPG 估计 HR",
    )
    axes[2].set_ylabel("心率 (bpm)")
    axes[2].set_title("ECG 参考心率与 PPG-only 估计心率")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    axes[3].bar(
        window_time,
        absolute_error,
        width=1.2,
        color="#ff7f0e",
        alpha=0.85,
    )
    axes[3].set_ylabel("绝对误差 (bpm)")
    axes[3].set_xlabel("时间（秒）")
    axes[3].set_title(
        f"窗口级绝对误差：中位数="
        f"{metrics['median_absolute_error_bpm']:.2f} bpm，"
        f"P90={metrics['p90_absolute_error_bpm']:.2f} bpm"
    )
    axes[3].grid(alpha=0.25)

    for axis in axes:
        axis.set_xlim(scene["start_s"], scene["end_s"])

    output_path = OUTPUT_DIR / str(scene["output_name"])
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def print_metrics(
    scene: dict[str, Any],
    metrics: dict[str, float | int],
) -> None:
    """在终端输出可直接核对的场景指标。"""
    print(f"\n{scene['title']}（{scene['start_s']:.0f}–{scene['end_s']:.0f} 秒）")
    print(f"  窗口数: {metrics['window_count']}")
    print(f"  MAE: {metrics['mae_bpm']:.3f} bpm")
    print(
        "  中位绝对误差: "
        f"{metrics['median_absolute_error_bpm']:.3f} bpm"
    )
    print(f"  RMSE: {metrics['rmse_bpm']:.3f} bpm")
    print(
        f"  P90 绝对误差: "
        f"{metrics['p90_absolute_error_bpm']:.3f} bpm"
    )
    print(f"  最大绝对误差: {metrics['max_absolute_error_bpm']:.3f} bpm")
    print(
        "  PPG 估计 HR 范围: "
        f"{metrics['estimated_hr_min_bpm']:.2f}–"
        f"{metrics['estimated_hr_max_bpm']:.2f} bpm"
    )
    print(
        "  ECG 参考 HR 范围: "
        f"{metrics['reference_hr_min_bpm']:.2f}–"
        f"{metrics['reference_hr_max_bpm']:.2f} bpm"
    )


def main() -> int:
    """运行基线、保存证据并输出真实指标。"""
    try:
        ppg, reference_hr = load_s1()
        results = build_window_results(ppg, reference_hr)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        configure_plot_style()

        scene_metrics: dict[str, dict[str, float | int]] = {}
        figure_paths: list[Path] = []

        for scene in SCENES:
            mask = scene_mask(results, scene)
            metrics = calculate_metrics(results, mask)
            scene_metrics[str(scene["key"])] = metrics
            figure_paths.append(
                plot_scene(ppg, results, scene, metrics)
            )
            print_metrics(scene, metrics)

        csv_path = write_window_csv(results)
        json_path = write_metrics_json(scene_metrics)

        print("\n已生成可追溯结果：")
        for figure_path in figure_paths:
            print(f"  图片: {figure_path}")
        print(f"  窗口明细: {csv_path}")
        print(f"  参数与指标: {json_path}")
        print(
            "\n说明：ECG 标签仅用于误差评价，"
            "未参与 PPG HR 的频率选择。"
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as error:
        print(f"PPG-only HR 基线运行失败：{error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
