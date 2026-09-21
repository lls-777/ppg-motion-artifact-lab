#!/usr/bin/env python3
"""PPG-DaLiA 上的 ACC 辅助 PPG 波形重建预研究 V0。

目标：
1. 在同一批 S14-S15 8 秒窗口上比较原始 BVP、作者处理后 PPG、
   ACC-NLMS 和 SMoLK 引导 ACC-NLMS；
2. 同时报告峰间期 HR 与现有频谱 HR，避免只凭波形观感判断效果；
3. 生成可追溯的窗口结果和固定选择的上下楼示例。

这是公开数据上的可行性研究。E4 BVP 不是 LED/PD 原始 ADC，本脚本也没有
真实干净 PPG 作为重建标签，因此输出只能称为心率导向的波形增强候选。
默认只计算并打印摘要；显式传入 --write-outputs 才保存证据文件。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.signal import savgol_filter

DATA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DATA_ROOT.parent
# 重建沿用第一阶段的心率估计与活动标签处理。
sys.path.insert(0, str(PROJECT_ROOT / "01_自主基线研究/01_代码"))

from analyze_s1_ppg_sqi import ACTIVITY_NAMES, activity_mode
from ppg_hr_baseline import (
    MAX_HR_BPM,
    MIN_HR_BPM,
    PPG_FS_HZ,
    configure_plot_style,
    estimate_window_hr,
)


SMOLK_ROOT = PROJECT_ROOT / "02_论文复现与验证/01_SMoLK论文复现"
sys.path.insert(0, str(SMOLK_ROOT))

from run_smoke import COMMIT, LearnedFilters, normalize  # noqa: E402


DATASET_ROOT = (
    PROJECT_ROOT / "90_共享原始数据" / "ppg_dalia" / "PPG_FieldStudy"
)
TEST_X_PATH = (
    SMOLK_ROOT
    / "data/new_PPG_DaLiA_test/processed_dataset/scaled_ppgs.npy"
)
MODEL_PATH = SMOLK_ROOT / "outputs/full_seed0/model_state.pt"
MAPPING_PATH = (
    SMOLK_ROOT / "outputs/dalia_mapping/dalia_hr_window_mapping.csv"
)
OUTPUT_DIR = (
    DATA_ROOT / "02_结果/V0"
)

PRIMARY_SUBJECTS = (14, 15)
EXPECTED_CONTAINED_WINDOWS = {14: 3576, 15: 3168}
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * PPG_FS_HZ
ACC_FS_HZ = 32
ACTIVITY_FS_HZ = 4
ACC_WINDOW_SAMPLES = WINDOW_SECONDS * ACC_FS_HZ
ACTIVITY_WINDOW_SAMPLES = WINDOW_SECONDS * ACTIVITY_FS_HZ

NLMS_TAPS = 16
NLMS_STEP_SIZE = 0.05
NLMS_EPSILON = 1e-6
SMOLK_SAVGOL_SAMPLES = 151
SMOLK_THRESHOLD = 0.5
PEAK_PROMINENCE_STD = 0.20
ERROR_THRESHOLD_BPM = 10.0

METHOD_ORDER = (
    "raw_bvp",
    "processed_ppg",
    "acc_nlms",
    "smolk_guided_nlms",
)
METHOD_LABELS = {
    "raw_bvp": "Raw E4 BVP",
    "processed_ppg": "Published processed PPG",
    "acc_nlms": "ACC-NLMS",
    "smolk_guided_nlms": "SMoLK-guided ACC-NLMS",
}
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
    "smolk_artifact_fraction",
    "acc_motion_rms_g",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 PPG+ACC 波形重建预研究 V0。"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="保存 CSV、JSON 和 PNG；默认只打印结果。",
    )
    return parser.parse_args()


def standardize(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values)
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-10:
        raise ValueError("无法标准化常量或非有限波形")
    return centered / scale


def load_subject(subject_id: int) -> dict[str, np.ndarray]:
    path = DATASET_ROOT / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 S{subject_id}：{path}")
    with path.open("rb") as stream:
        subject = pickle.load(stream, encoding="latin1")

    try:
        wrist = subject["signal"]["wrist"]
        result = {
            "ppg": np.asarray(wrist["BVP"], dtype=np.float64).reshape(-1),
            "acc": np.asarray(wrist["ACC"], dtype=np.float64),
            "reference_hr": np.asarray(
                subject["label"], dtype=np.float64
            ).reshape(-1),
            "activity": np.asarray(
                subject["activity"], dtype=np.float64
            ).reshape(-1),
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"S{subject_id} 数据结构不符合 PPG-DaLiA") from error

    if result["acc"].ndim != 2 or result["acc"].shape[1] != 3:
        raise ValueError(f"S{subject_id} ACC 形状异常：{result['acc'].shape}")
    for name, values in result.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"S{subject_id} {name} 含非有限值")
    return result


def load_mapping_rows() -> list[dict[str, int | float]]:
    if not MAPPING_PATH.is_file():
        raise FileNotFoundError(f"缺少 SMoLK 时间映射：{MAPPING_PATH}")

    rows: list[dict[str, int | float]] = []
    with MAPPING_PATH.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            subject_id = int(raw["subject_id"])
            if (
                subject_id not in PRIMARY_SUBJECTS
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

    rows.sort(key=lambda row: (row["subject_id"], row["hr_window_index"]))
    counts = {
        subject_id: sum(
            int(row["subject_id"]) == subject_id for row in rows
        )
        for subject_id in PRIMARY_SUBJECTS
    }
    if counts != EXPECTED_CONTAINED_WINDOWS:
        raise ValueError(f"SMoLK 对齐窗口数异常：{counts}")
    keys = {
        (int(row["subject_id"]), int(row["hr_window_index"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("SMoLK 映射中存在重复窗口")
    return rows


def load_smolk_probabilities(
    processed_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"缺少 SMoLK 模型：{MODEL_PATH}")
    bundle = torch.load(MODEL_PATH, map_location="cpu")
    expected = ("smolk_full_seed0_state_v1", COMMIT, 0)
    actual = (
        bundle.get("format"),
        bundle.get("upstream_commit"),
        bundle.get("seed"),
    )
    if actual != expected:
        raise ValueError(f"SMoLK 模型元数据异常：{actual}")

    model = LearnedFilters()
    model.load_state_dict(bundle["model"])
    model.eval()
    normalized = normalize(processed_test)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized), 128):
            batch = torch.from_numpy(normalized[start : start + 128, None, :])
            outputs.append(model(batch).numpy())
    probability = np.concatenate(outputs)
    smoothed = savgol_filter(
        probability,
        SMOLK_SAVGOL_SAMPLES,
        3,
        axis=-1,
    )
    return probability, np.clip(smoothed, 0.0, 1.0)


def resample_acc(acc_window: np.ndarray) -> np.ndarray:
    if acc_window.shape != (ACC_WINDOW_SAMPLES, 3):
        raise ValueError(f"ACC 窗口形状异常：{acc_window.shape}")
    aligned = signal.resample_poly(acc_window, 2, 1, axis=0)
    if aligned.shape != (WINDOW_SAMPLES, 3):
        raise ValueError(f"ACC 重采样形状异常：{aligned.shape}")
    return aligned


def build_acc_references(acc_64hz: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(acc_64hz, axis=1, keepdims=True)
    references = np.column_stack((acc_64hz, magnitude))
    references = signal.detrend(references, axis=0, type="linear")
    scales = np.std(references, axis=0)
    usable = scales >= 1e-10
    references[:, usable] /= scales[usable]
    references[:, ~usable] = 0.0
    return references


def nlms_cancel(
    ppg: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用与运动相关的参考信号估计 PPG 中的线性运动成分。"""
    if ppg.shape != (WINDOW_SAMPLES,):
        raise ValueError(f"PPG 窗口形状异常：{ppg.shape}")
    if references.shape[0] != WINDOW_SAMPLES:
        raise ValueError("PPG 与 ACC 参考信号长度不一致")

    weights = np.zeros(references.shape[1] * NLMS_TAPS, dtype=np.float64)
    history = np.zeros((NLMS_TAPS, references.shape[1]), dtype=np.float64)
    residual = np.empty(WINDOW_SAMPLES, dtype=np.float64)
    estimated_motion = np.empty(WINDOW_SAMPLES, dtype=np.float64)

    for index in range(WINDOW_SAMPLES):
        history[1:] = history[:-1]
        history[0] = references[index]
        vector = history.reshape(-1)
        prediction = float(np.dot(weights, vector))
        error = float(ppg[index] - prediction)
        power = float(np.dot(vector, vector))
        weights += (
            NLMS_STEP_SIZE
            * error
            * vector
            / (NLMS_EPSILON + power)
        )
        estimated_motion[index] = prediction
        residual[index] = error

    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("NLMS 输出出现非有限值")
    return residual, estimated_motion


def estimate_peak_hr(
    waveform: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    values = standardize(waveform)
    minimum_distance = max(
        1,
        int(np.floor(PPG_FS_HZ * 60.0 / MAX_HR_BPM)),
    )
    prominence = PEAK_PROMINENCE_STD * float(np.std(values))
    peaks, _ = signal.find_peaks(
        values,
        distance=minimum_distance,
        prominence=prominence,
    )
    if len(peaks) < 3:
        return float("nan"), peaks, 0

    intervals_s = np.diff(peaks) / PPG_FS_HZ
    valid = (
        (intervals_s >= 60.0 / MAX_HR_BPM)
        & (intervals_s <= 60.0 / MIN_HR_BPM)
    )
    valid_intervals = intervals_s[valid]
    if len(valid_intervals) < 2:
        return float("nan"), peaks, int(len(valid_intervals))
    hr_bpm = 60.0 / float(np.median(valid_intervals))
    return hr_bpm, peaks, int(len(valid_intervals))


def window_activity(
    activity: np.ndarray,
    window_start_s: float,
) -> int:
    start = int(round(window_start_s * ACTIVITY_FS_HZ))
    end = start + ACTIVITY_WINDOW_SAMPLES
    if end > len(activity):
        raise ValueError("活动标签窗口越界")
    return activity_mode(activity[start:end])


def process_windows() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    if not TEST_X_PATH.is_file():
        raise FileNotFoundError(f"缺少处理后测试 PPG：{TEST_X_PATH}")
    processed_test = np.load(TEST_X_PATH, allow_pickle=False)
    if processed_test.shape != (869, 1920):
        raise ValueError(f"处理后测试 PPG 形状异常：{processed_test.shape}")

    _, smooth_probability = load_smolk_probabilities(processed_test)
    mapping_rows = load_mapping_rows()
    subjects = {
        subject_id: load_subject(subject_id)
        for subject_id in PRIMARY_SUBJECTS
    }

    records: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    for mapping in mapping_rows:
        subject_id = int(mapping["subject_id"])
        subject = subjects[subject_id]
        start = int(mapping["window_start_sample"])
        end = int(mapping["window_end_sample"])
        hr_index = int(mapping["hr_window_index"])
        row_index = int(mapping["processed_test_row"])
        offset = int(mapping["segment_offset_start_sample"])
        processed_slice = slice(offset, offset + WINDOW_SAMPLES)

        raw_bvp = subject["ppg"][start:end]
        processed_ppg = standardize(processed_test[row_index, processed_slice])
        if raw_bvp.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"S{subject_id} PPG 窗口越界")

        acc_start = int(round(float(mapping["window_start_s"]) * ACC_FS_HZ))
        acc_end = acc_start + ACC_WINDOW_SAMPLES
        acc_window = subject["acc"][acc_start:acc_end]
        acc_64hz = resample_acc(acc_window)
        references = build_acc_references(acc_64hz)
        nlms_ppg, estimated_motion = nlms_cancel(
            processed_ppg,
            references,
        )

        smolk_score = smooth_probability[row_index, processed_slice]
        reconstructed = (
            (1.0 - smolk_score) * processed_ppg
            + smolk_score * nlms_ppg
        )
        waveforms = {
            "raw_bvp": raw_bvp,
            "processed_ppg": processed_ppg,
            "acc_nlms": nlms_ppg,
            "smolk_guided_nlms": reconstructed,
        }

        reference_hr = float(subject["reference_hr"][hr_index])
        activity_id = window_activity(
            subject["activity"],
            float(mapping["window_start_s"]),
        )
        motion_rms = float(
            np.sqrt(
                np.mean(
                    (
                        np.linalg.norm(acc_window, axis=1)
                        - 1.0
                    )
                    ** 2
                )
            )
        )
        example_methods: dict[str, Any] = {}
        for method, waveform in waveforms.items():
            peak_hr, peaks, valid_ibi_count = estimate_peak_hr(waveform)
            spectral_hr, _, _ = estimate_window_hr(waveform)
            peak_error = (
                abs(peak_hr - reference_hr)
                if np.isfinite(peak_hr)
                else float("nan")
            )
            record = {
                "subject_id": subject_id,
                "window_start_s": float(mapping["window_start_s"]),
                "window_center_s": float(mapping["window_center_s"]),
                "window_end_s": float(mapping["window_end_s"]),
                "activity_id": activity_id,
                "activity_name": ACTIVITY_NAMES[activity_id],
                "method": method,
                "reference_hr_bpm": reference_hr,
                "peak_hr_bpm": peak_hr,
                "peak_absolute_error_bpm": peak_error,
                "peak_output_valid": bool(np.isfinite(peak_hr)),
                "spectral_hr_bpm": float(spectral_hr),
                "spectral_absolute_error_bpm": float(
                    abs(spectral_hr - reference_hr)
                ),
                "detected_peak_count": int(len(peaks)),
                "valid_ibi_count": valid_ibi_count,
                "smolk_probability_mean": float(np.mean(smolk_score)),
                "smolk_artifact_fraction": float(
                    np.mean(smolk_score > SMOLK_THRESHOLD)
                ),
                "acc_motion_rms_g": motion_rms,
            }
            records.append(record)
            example_methods[method] = {
                "waveform": waveform.copy(),
                "peak_hr_bpm": peak_hr,
                "peaks": peaks.copy(),
            }

        # 固定选择 S14 的首个完整上下楼窗口，不根据算法效果挑图。
        if (
            example is None
            and subject_id == 14
            and activity_id == 2
        ):
            example = {
                "subject_id": subject_id,
                "window_start_s": float(mapping["window_start_s"]),
                "reference_hr_bpm": reference_hr,
                "raw_bvp": raw_bvp.copy(),
                "acc_64hz": acc_64hz.copy(),
                "smolk_score": smolk_score.copy(),
                "processed_ppg": processed_ppg.copy(),
                "nlms_ppg": nlms_ppg.copy(),
                "estimated_motion": estimated_motion.copy(),
                "reconstructed": reconstructed.copy(),
                "methods": example_methods,
            }

    if example is None:
        raise ValueError("没有找到固定的 S14 上下楼示例")
    expected_records = sum(EXPECTED_CONTAINED_WINDOWS.values()) * len(
        METHOD_ORDER
    )
    if len(records) != expected_records:
        raise ValueError(f"窗口结果数异常：{len(records)}")
    config = {
        "status": "public_dataset_pre_research_not_product_validation",
        "subjects": list(PRIMARY_SUBJECTS),
        "contained_windows": EXPECTED_CONTAINED_WINDOWS,
        "ppg_sampling_rate_hz": PPG_FS_HZ,
        "acc_original_sampling_rate_hz": ACC_FS_HZ,
        "window_seconds": WINDOW_SECONDS,
        "smolk_model": "full_seed0_12_kernel",
        "smolk_savgol_samples": SMOLK_SAVGOL_SAMPLES,
        "smolk_threshold": SMOLK_THRESHOLD,
        "nlms_reference_channels": "ACC x/y/z plus magnitude",
        "nlms_taps": NLMS_TAPS,
        "nlms_step_size": NLMS_STEP_SIZE,
        "nlms_epsilon": NLMS_EPSILON,
        "peak_prominence_std": PEAK_PROMINENCE_STD,
        "peak_hr_range_bpm": [MIN_HR_BPM, MAX_HR_BPM],
        "error_threshold_bpm": ERROR_THRESHOLD_BPM,
    }
    return records, example, config


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
                raise ValueError(f"{scope}/{method}/{estimator} 没有有效结果")
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
                    "median_absolute_error_bpm": float(np.median(errors)),
                    "p90_absolute_error_bpm": float(
                        np.percentile(errors, 90)
                    ),
                    "failure_gt_10_bpm_percent": float(
                        100.0 * np.mean(errors > ERROR_THRESHOLD_BPM)
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
                f"activity_{activity_id}_{ACTIVITY_NAMES[activity_id]}",
            )
        )
    return result


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


def save_metric_figure(metrics: list[dict[str, Any]]) -> None:
    selected = {
        row["method"]: row
        for row in metrics
        if row["scope"] == "primary_S14_S15"
        and row["estimator"] == "peak_interval"
    }
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    colors = ("#6c757d", "#176b87", "#e69f00", "#c44900")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.5),
        constrained_layout=True,
    )
    specifications = (
        ("mae_bpm", "Peak-interval HR MAE", "bpm"),
        (
            "failure_gt_10_bpm_percent",
            "Peak-interval HR failures >10 bpm",
            "%",
        ),
        ("coverage_percent", "Peak-interval HR coverage", "%"),
    )
    positions = np.arange(len(METHOD_ORDER))
    for axis, (field, title, unit) in zip(axes, specifications):
        values = [selected[method][field] for method in METHOD_ORDER]
        bars = axis.bar(positions, values, color=colors)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(
        "PPG-DaLiA S14-S15 | identical contained 8 s windows"
    )
    figure.savefig(OUTPUT_DIR / "hr_method_comparison.png", dpi=180)
    plt.close(figure)


def save_example_figure(example: dict[str, Any]) -> None:
    time_s = np.arange(WINDOW_SAMPLES) / PPG_FS_HZ
    figure, axes = plt.subplots(
        6,
        1,
        figsize=(13, 13),
        constrained_layout=True,
    )
    axes[0].plot(time_s, example["raw_bvp"], color="#176b87")
    axes[0].set_ylabel("E4 BVP\n(device output)")
    axes[0].set_title(
        f"Fixed example: S{example['subject_id']} stairs, "
        f"{example['window_start_s']:.1f}-"
        f"{example['window_start_s'] + WINDOW_SECONDS:.1f} s"
    )

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
        color="#c44900",
        label="SMoLK SG-151 score",
    )
    axes[2].axhline(
        SMOLK_THRESHOLD,
        color="gray",
        linestyle="--",
        label="0.5 threshold",
    )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Artifact score")
    axes[2].legend(fontsize=8)

    axes[3].plot(
        time_s,
        example["processed_ppg"],
        label="Published processed PPG",
        color="#176b87",
        alpha=0.7,
    )
    axes[3].plot(
        time_s,
        example["nlms_ppg"],
        label="ACC-NLMS",
        color="#e69f00",
        alpha=0.75,
    )
    axes[3].plot(
        time_s,
        example["reconstructed"],
        label="SMoLK-guided reconstruction",
        color="#c44900",
        linewidth=1.4,
    )
    axes[3].set_ylabel("Standardized\namplitude")
    axes[3].legend(ncol=3, fontsize=8)

    reconstruction = example["reconstructed"]
    peaks = example["methods"]["smolk_guided_nlms"]["peaks"]
    axes[4].plot(time_s, reconstruction, color="#c44900")
    axes[4].scatter(
        time_s[peaks],
        reconstruction[peaks],
        color="black",
        s=18,
        label="Detected peaks",
        zorder=3,
    )
    axes[4].set_ylabel("Reconstructed\nPPG")
    axes[4].legend(fontsize=8)

    peak_values = [
        example["methods"][method]["peak_hr_bpm"]
        for method in METHOD_ORDER
    ]
    axes[5].bar(
        np.arange(len(METHOD_ORDER)),
        peak_values,
        color=("#6c757d", "#176b87", "#e69f00", "#c44900"),
    )
    axes[5].axhline(
        example["reference_hr_bpm"],
        color="black",
        linestyle="--",
        label=f"Reference {example['reference_hr_bpm']:.1f} bpm",
    )
    axes[5].set_xticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=15,
        ha="right",
    )
    axes[5].set_ylabel("Peak HR (bpm)")
    axes[5].legend(fontsize=8)

    for axis in axes[:5]:
        axis.set_xlim(0, WINDOW_SECONDS)
        axis.grid(alpha=0.2)
    axes[4].set_xlabel("Time within 8 s window (s)")
    figure.savefig(
        OUTPUT_DIR / "fixed_s14_stairs_reconstruction.png",
        dpi=180,
    )
    plt.close(figure)


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
    configure_plot_style()

    records, example, config = process_windows()
    metrics = summarize_rows(records, "primary_S14_S15")
    activity_metrics = build_activity_metrics(records)
    summary = {
        "status": "completed_public_dataset_pre_research_v0",
        "research_question": (
            "Can an interpretable ACC-NLMS baseline, optionally controlled by "
            "SMoLK artifact scores, produce a PPG waveform that improves "
            "heart-rate estimation on the existing public-data pipeline?"
        ),
        "primary_scope": {
            "subjects": list(PRIMARY_SUBJECTS),
            "contained_window_count": sum(
                EXPECTED_CONTAINED_WINDOWS.values()
            ),
            "interpretation": (
                "Retrospective pre-research on previously analysed S14-S15; "
                "not a fresh product holdout."
            ),
        },
        "config": config,
        "method_metrics": metrics,
        "limitations": [
            "Empatica E4 BVP is a device-processed single channel, not LED/PD raw ADC.",
            "No clean real-motion PPG target exists, so waveform fidelity is not directly measurable.",
            "NLMS assumes a linear relation between ACC references and PPG motion artifact.",
            "SMoLK uses 30 s normalization and centered noncausal operations.",
            "The fixed peak detector is a V0 research baseline, not a validated product HR algorithm.",
            "The current model weights and thresholds must be retrained or recalibrated on future in-house sensor data.",
        ],
        "output_interpretation": (
            "A negative HR result is still a valid feasibility finding; "
            "visual smoothness alone is not evidence of successful reconstruction."
        ),
    }

    printed = {
        "status": summary["status"],
        "window_count": summary["primary_scope"]["contained_window_count"],
        "peak_interval_metrics": [
            row
            for row in metrics
            if row["estimator"] == "peak_interval"
        ],
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
        OUTPUT_DIR / "window_metrics.csv",
        records,
        WINDOW_FIELDS,
    )
    write_csv(
        OUTPUT_DIR / "method_metrics.csv",
        metrics,
        METRIC_FIELDS,
    )
    write_csv(
        OUTPUT_DIR / "activity_metrics.csv",
        activity_metrics,
        METRIC_FIELDS,
    )
    atomic_json(OUTPUT_DIR / "config.json", config)
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    save_metric_figure(metrics)
    save_example_figure(example)
    printed["output"] = str(OUTPUT_DIR)
    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
