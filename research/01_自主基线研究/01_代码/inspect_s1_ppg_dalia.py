#!/usr/bin/env python3
"""只读检查 PPG-DaLiA 的 S1 同步数据文件。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any


DATA_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    DATA_ROOT.parent
    / "90_共享原始数据/ppg_dalia/PPG_FieldStudy/S1/S1.pkl"
)

# 采样率来自 PPG-DaLiA 的说明文件；用于把样本数换算成秒。
SAMPLING_RATES_HZ = {
    "PPG (wrist/BVP)": 64,
    "ACC (wrist/ACC)": 32,
    "activity": 4,
    "HR label": 0.5,  # 每 2 秒一个 ECG 导出的心率标签
}


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """确认一个字段是字典，避免后续出现难理解的索引错误。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} 应为字典，实际为 {type(value).__name__}。")
    return value


def require_key(mapping: dict[str, Any], key: str, location: str) -> Any:
    """读取必需字段，并在数据版本不一致时说明缺少的位置。"""
    if key not in mapping:
        available = ", ".join(mapping.keys())
        raise KeyError(
            f"在 {location} 中未找到 {key!r}；可用字段为：{available}"
        )
    return mapping[key]


def print_signal_summary(name: str, values: Any, sampling_rate_hz: float) -> None:
    """打印数组的形状、时长和少量样本，不改动任何原始数据。"""
    shape = getattr(values, "shape", None)
    if not shape:
        raise TypeError(f"{name} 不是预期的数组，无法读取 shape。")

    sample_count = int(shape[0])
    duration_seconds = sample_count / sampling_rate_hz
    preview = values[:5].tolist()

    print(f"\n{name}")
    print(f"  shape: {shape}")
    print(f"  采样率: {sampling_rate_hz:g} Hz")
    print(f"  估算时长: {duration_seconds:.1f} 秒")
    print(f"  前 5 个样本: {preview}")


def main() -> int:
    """加载 S1.pkl，并输出周三画图前需要核对的基础事实。"""
    if not DATA_PATH.is_file():
        print(f"未找到数据文件：{DATA_PATH}", file=sys.stderr)
        print("请检查 PPG-DaLiA 是否已解压，以及目录名是否被修改。", file=sys.stderr)
        return 1

    try:
        with DATA_PATH.open("rb") as data_file:
            subject = pickle.load(data_file, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError) as error:
        print(f"无法读取 {DATA_PATH.name}：{error}", file=sys.stderr)
        return 1

    try:
        subject = require_mapping(subject, "S1.pkl 顶层对象")
        signal = require_mapping(
            require_key(subject, "signal", "S1.pkl 顶层"),
            "signal",
        )
        wrist = require_mapping(
            require_key(signal, "wrist", "signal"),
            "signal/wrist",
        )

        ppg = require_key(wrist, "BVP", "signal/wrist")
        acc = require_key(wrist, "ACC", "signal/wrist")
        hr_label = require_key(subject, "label", "S1.pkl 顶层")
        activity = require_key(subject, "activity", "S1.pkl 顶层")
    except (KeyError, TypeError) as error:
        print(f"数据结构与预期不一致：{error}", file=sys.stderr)
        return 1

    print("PPG-DaLiA S1：只读检查结果")
    print(f"数据文件: {DATA_PATH}")
    print(f"顶层字段: {', '.join(subject.keys())}")
    print("\n读取路径：")
    print("  PPG       -> subject['signal']['wrist']['BVP']")
    print("  三轴 ACC  -> subject['signal']['wrist']['ACC']")
    print("  参考心率  -> subject['label']")
    print("  活动标签  -> subject['activity']")

    print_signal_summary(
        "PPG（腕部 BVP）",
        ppg,
        SAMPLING_RATES_HZ["PPG (wrist/BVP)"],
    )
    print_signal_summary(
        "ACC（三轴加速度）",
        acc,
        SAMPLING_RATES_HZ["ACC (wrist/ACC)"],
    )
    print_signal_summary(
        "参考心率标签（ECG 导出）",
        hr_label,
        SAMPLING_RATES_HZ["HR label"],
    )
    print_signal_summary(
        "活动标签",
        activity,
        SAMPLING_RATES_HZ["activity"],
    )

    print(
        "\n下一步：确认上述输出无误后，使用同一读取路径截取 "
        "120–180 秒静止段与 850–910 秒上下楼段，绘制对齐图。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
