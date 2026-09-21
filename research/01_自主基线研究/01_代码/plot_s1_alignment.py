#!/usr/bin/env python3
"""绘制 PPG-DaLiA S1 的 PPG、运动代理与参考心率对齐图。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    DATA_ROOT.parent
    / "90_共享原始数据/ppg_dalia/PPG_FieldStudy/S1/S1.pkl"
)
OUTPUT_DIR = DATA_ROOT / "02_结果"

PPG_FS_HZ = 64
ACC_FS_HZ = 32
HR_WINDOW_SECONDS = 8
HR_STEP_SECONDS = 2

# 两段均来自 S1_activity.csv：
# 120–180 秒处于静止基线；850–910 秒处于上下楼活动。
PLOT_WINDOWS = (
    ("baseline", 120.0, 180.0, "S1 静止基线"),
    ("stairs", 850.0, 910.0, "S1 上下楼"),
)


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """确认读取到的是预期的字典结构。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} 应为字典，实际为 {type(value).__name__}。")
    return value


def require_key(mapping: dict[str, Any], key: str, location: str) -> Any:
    """读取必需字段，并在数据结构不符时给出可读错误。"""
    if key not in mapping:
        available = ", ".join(mapping.keys())
        raise KeyError(
            f"在 {location} 中未找到 {key!r}；可用字段为：{available}"
        )
    return mapping[key]


def read_s1() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只读加载 S1 的同步 PPG、ACC 与 ECG 心率标签。"""
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

    ppg = np.asarray(require_key(wrist, "BVP", "signal/wrist")).reshape(-1)
    acc = np.asarray(require_key(wrist, "ACC", "signal/wrist"))
    hr_label = np.asarray(
        require_key(subject, "label", "S1.pkl 顶层")
    ).reshape(-1)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"ACC 应为 N×3 三轴数组，实际 shape 为 {acc.shape}。")

    return ppg, acc, hr_label


def time_slice(
    values: np.ndarray,
    sampling_rate_hz: float,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按秒切出连续信号，并返回对应的绝对时间轴。"""
    start_index = int(start_seconds * sampling_rate_hz)
    end_index = int(end_seconds * sampling_rate_hz)

    if start_index < 0 or end_index > len(values) or start_index >= end_index:
        raise ValueError(
            f"时间窗 {start_seconds:g}–{end_seconds:g} 秒超出信号范围。"
        )

    time_axis = np.arange(start_index, end_index) / sampling_rate_hz
    return time_axis, values[start_index:end_index]


def plot_window(
    ppg: np.ndarray,
    acc: np.ndarray,
    hr_label: np.ndarray,
    file_stem: str,
    start_seconds: float,
    end_seconds: float,
    title: str,
) -> Path:
    """绘制一个时间窗的三行对齐图，并返回图片路径。"""
    ppg_time, ppg_segment = time_slice(
        ppg, PPG_FS_HZ, start_seconds, end_seconds
    )
    acc_time, acc_segment = time_slice(
        acc, ACC_FS_HZ, start_seconds, end_seconds
    )

    # ACC 单位为 g。向量模长包含重力；偏离 1 g 的绝对值只作为
    # 初步运动强度代理，不应当被解释为最终运动伪影算法。
    acc_magnitude = np.linalg.norm(acc_segment, axis=1)
    motion_proxy = np.abs(acc_magnitude - 1.0)

    # 每个 HR 标签代表一个 8 秒窗口，以下将它放在该窗口中心。
    hr_center_time = (
        np.arange(len(hr_label)) * HR_STEP_SECONDS
        + HR_WINDOW_SECONDS / 2
    )
    complete_window = (
        (hr_center_time >= start_seconds + HR_WINDOW_SECONDS / 2)
        & (hr_center_time <= end_seconds - HR_WINDOW_SECONDS / 2)
    )

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(ppg_time, ppg_segment, color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("PPG / BVP")
    axes[0].set_title(
        f"{title}（{start_seconds:g}–{end_seconds:g} 秒）"
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(acc_time, motion_proxy, color="#ff7f0e", linewidth=0.7)
    axes[1].set_ylabel("|‖ACC‖ - 1 g|")
    axes[1].set_title("三轴 ACC 运动强度代理")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        hr_center_time[complete_window],
        hr_label[complete_window],
        color="#2ca02c",
        marker="o",
        markersize=3,
        linewidth=1,
    )
    axes[2].set_ylabel("参考心率 (bpm)")
    axes[2].set_title("ECG 导出参考心率（8 秒窗口，2 秒步长）")
    axes[2].set_xlabel("时间（秒）")
    axes[2].grid(alpha=0.25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{file_stem}.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> int:
    """生成静止和上下楼两张对齐图。"""
    try:
        ppg, acc, hr_label = read_s1()
        for file_stem, start, end, title in PLOT_WINDOWS:
            output_path = plot_window(
                ppg,
                acc,
                hr_label,
                file_stem=file_stem,
                start_seconds=start,
                end_seconds=end,
                title=title,
            )
            print(f"已生成：{output_path}")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"绘图失败：{error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
