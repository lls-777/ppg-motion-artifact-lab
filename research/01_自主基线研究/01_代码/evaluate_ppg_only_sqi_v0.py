#!/usr/bin/env python3
"""设计并验证可解释的 PPG-only SQI V0。

本脚本不重新计算 PPG 特征，而是读取已经生成的 S1–S15
窗口级特征表。S1–S10 只用于拟合归一化参数和选择阈值，
S11–S15 只用于内部独立验证。

这里的 SQI 表示“当前 PPG 窗口是否适合现有 PPG-only
心率基线”，不是对传感器信号质量的医学金标准标注。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ppg_hr_baseline import configure_plot_style


DATA_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    DATA_ROOT
    / "02_结果/sqi_cross_subject/S1_S15_window_features.csv"
)
OUTPUT_DIR = DATA_ROOT / "02_结果/ppg_only_sqi_v0"

DESIGN_SUBJECTS = tuple(
    f"S{index}" for index in range(1, 11)
)
VALIDATION_SUBJECTS = tuple(
    f"S{index}" for index in range(11, 16)
)

EXPECTED_TOTAL_WINDOWS = 64697
EXPECTED_DESIGN_WINDOWS = 43215
EXPECTED_VALIDATION_WINDOWS = 21482

REQUIRED_COLUMNS = (
    "subject_id",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "ppg_hr_bpm",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "finite_ratio",
    "flatline_ratio",
    "extreme_repeat_ratio",
    "robust_amplitude",
    "peak_power_ratio",
    "autocorrelation_peak",
)

# 三个特征分别描述频域集中程度、时域周期性和振幅异常。
# 权重暂时采用透明的工程初值，后续根据验证结果再决定是否调整。
SQI_WEIGHTS = {
    "spectral_quality": 0.40,
    "periodicity_quality": 0.40,
    "amplitude_quality": 0.20,
}

# 归一化参数只允许从 S1–S10 计算。
LOW_PERCENTILE = 10.0
HIGH_PERCENTILE = 90.0

# 以下是数据完整性保护阈值，不是从 PPG-DaLiA 标签学习出的
# 产品阈值，因此验证报告必须将其标记为工程保护条件。
MIN_FINITE_RATIO = 1.0
MAX_FLATLINE_RATIO = 0.10
MAX_EXTREME_REPEAT_RATIO = 0.05

# 阈值目标描述“保留下来的窗口中，现有 PPG-only HR
# 发生大误差的比例”。它们是本轮研究目标，不是产品指标。
USABLE_MAX_FAILURE_PERCENT = 10.0
USABLE_MIN_COVERAGE_PERCENT = 5.0
ACCEPTED_MAX_FAILURE_PERCENT = 25.0
ACCEPTED_MIN_COVERAGE_PERCENT = 10.0

THRESHOLD_GRID = np.linspace(0.0, 100.0, 201)


def parse_args() -> argparse.Namespace:
    """读取命令行参数，默认保持零文件写入。"""
    parser = argparse.ArgumentParser(
        description="验证 PPG-only SQI V0。"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="显式生成 CSV、JSON 和 PNG 证据文件。",
    )
    return parser.parse_args()


def load_feature_table(
    csv_path: Path,
) -> dict[str, np.ndarray]:
    """读取并验证跨受试者窗口级特征表。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"未找到窗口特征表：{csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(
                "窗口特征表缺少必要字段："
                + ", ".join(missing)
            )
        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL_WINDOWS:
        raise ValueError(
            "窗口总数检查失败："
            f"期望 {EXPECTED_TOTAL_WINDOWS}，"
            f"实际 {len(rows)}。"
        )

    string_columns = (
        "subject_id",
        "activity_name",
    )
    numeric_columns = tuple(
        column
        for column in REQUIRED_COLUMNS
        if column not in string_columns
    )

    data: dict[str, np.ndarray] = {}
    for column in string_columns:
        data[column] = np.asarray(
            [row[column] for row in rows],
            dtype=object,
        )

    for column in numeric_columns:
        try:
            data[column] = np.asarray(
                [float(row[column]) for row in rows],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ValueError(
                f"字段 {column} 中存在无法转换的数值。"
            ) from error

    data["activity_id"] = data[
        "activity_id"
    ].astype(np.int64)
    return data


def build_subject_masks(
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """生成设计集与验证集掩码，并检查受试者边界。"""
    observed = set(subject_ids.tolist())
    expected = set(
        DESIGN_SUBJECTS + VALIDATION_SUBJECTS
    )
    if observed != expected:
        raise ValueError(
            "受试者集合与预期不一致："
            f"期望 {sorted(expected)}，"
            f"实际 {sorted(observed)}。"
        )

    design_mask = np.isin(
        subject_ids,
        DESIGN_SUBJECTS,
    )
    validation_mask = np.isin(
        subject_ids,
        VALIDATION_SUBJECTS,
    )

    if np.any(design_mask & validation_mask):
        raise ValueError(
            "设计集和验证集存在受试者重叠。"
        )
    if np.any(~(design_mask | validation_mask)):
        raise ValueError(
            "存在未被分配的数据窗口。"
        )
    if int(np.sum(design_mask)) != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            "设计集窗口数检查失败："
            f"期望 {EXPECTED_DESIGN_WINDOWS}，"
            f"实际 {int(np.sum(design_mask))}。"
        )
    if (
        int(np.sum(validation_mask))
        != EXPECTED_VALIDATION_WINDOWS
    ):
        raise ValueError(
            "验证集窗口数检查失败："
            f"期望 {EXPECTED_VALIDATION_WINDOWS}，"
            f"实际 {int(np.sum(validation_mask))}。"
        )

    return design_mask, validation_mask


def build_hard_reject_masks(
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """识别数据无效、平线和疑似削顶窗口。"""
    feature_names = (
        "finite_ratio",
        "flatline_ratio",
        "extreme_repeat_ratio",
        "robust_amplitude",
        "peak_power_ratio",
        "autocorrelation_peak",
    )
    all_numeric_finite = np.ones(
        len(data["subject_id"]),
        dtype=bool,
    )
    for feature in feature_names:
        all_numeric_finite &= np.isfinite(data[feature])

    invalid_data = (
        ~all_numeric_finite
        | (data["finite_ratio"] < MIN_FINITE_RATIO)
    )
    flatline = (
        ~invalid_data
        & (
            data["flatline_ratio"]
            >= MAX_FLATLINE_RATIO
        )
    )
    possible_clipping = (
        ~invalid_data
        & ~flatline
        & (
            data["extreme_repeat_ratio"]
            >= MAX_EXTREME_REPEAT_RATIO
        )
    )

    return {
        "invalid_data": invalid_data,
        "flatline": flatline,
        "possible_clipping": possible_clipping,
    }


def fit_percentile_bounds(
    values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    log_transform: bool = False,
) -> dict[str, float | bool]:
    """在设计集上拟合稳健的 P10/P90 归一化边界。"""
    transformed = np.asarray(
        values,
        dtype=np.float64,
    )
    if log_transform:
        transformed = np.log1p(
            np.maximum(transformed, 0.0)
        )

    clean = transformed[
        fit_mask & np.isfinite(transformed)
    ]
    if clean.size == 0:
        raise ValueError(
            "没有可用于拟合归一化参数的窗口。"
        )

    lower = float(
        np.percentile(clean, LOW_PERCENTILE)
    )
    upper = float(
        np.percentile(clean, HIGH_PERCENTILE)
    )
    if not upper > lower:
        raise ValueError(
            "特征 P90 必须大于 P10："
            f"P10={lower}，P90={upper}。"
        )

    return {
        "p10": lower,
        "p90": upper,
        "log_transform": log_transform,
    }


def apply_quality_mapping(
    values: np.ndarray,
    parameters: dict[str, float | bool],
    *,
    higher_is_better: bool,
) -> np.ndarray:
    """将一个特征映射到 0–1 的可解释质量分量。"""
    transformed = np.asarray(
        values,
        dtype=np.float64,
    )
    if bool(parameters["log_transform"]):
        transformed = np.log1p(
            np.maximum(transformed, 0.0)
        )

    lower = float(parameters["p10"])
    upper = float(parameters["p90"])
    quality = np.clip(
        (transformed - lower) / (upper - lower),
        0.0,
        1.0,
    )
    if not higher_is_better:
        quality = 1.0 - quality
    return quality


def build_sqi(
    data: dict[str, np.ndarray],
    fit_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, dict[str, float | bool]],
]:
    """拟合设计集参数并计算所有窗口的 SQI。"""
    parameters = {
        "peak_power_ratio": fit_percentile_bounds(
            data["peak_power_ratio"],
            fit_mask,
        ),
        "autocorrelation_peak": fit_percentile_bounds(
            data["autocorrelation_peak"],
            fit_mask,
        ),
        "robust_amplitude": fit_percentile_bounds(
            data["robust_amplitude"],
            fit_mask,
            log_transform=True,
        ),
    }

    components = {
        "spectral_quality": apply_quality_mapping(
            data["peak_power_ratio"],
            parameters["peak_power_ratio"],
            higher_is_better=True,
        ),
        "periodicity_quality": apply_quality_mapping(
            data["autocorrelation_peak"],
            parameters["autocorrelation_peak"],
            higher_is_better=True,
        ),
        "amplitude_quality": apply_quality_mapping(
            data["robust_amplitude"],
            parameters["robust_amplitude"],
            higher_is_better=False,
        ),
    }

    score = 100.0 * sum(
        SQI_WEIGHTS[name] * components[name]
        for name in SQI_WEIGHTS
    )
    return score, components, parameters


def threshold_statistics(
    scores: np.ndarray,
    absolute_error: np.ndarray,
    universe_mask: np.ndarray,
    eligible_mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """计算某个候选阈值的覆盖率与大误差率。"""
    accepted = (
        universe_mask
        & eligible_mask
        & (scores >= threshold)
    )
    count = int(np.sum(accepted))
    universe_count = int(np.sum(universe_mask))
    if count == 0:
        return {
            "threshold": float(threshold),
            "count": 0,
            "coverage_percent": 0.0,
            "failure_gt_10_bpm_percent": 100.0,
        }

    return {
        "threshold": float(threshold),
        "count": count,
        "coverage_percent": float(
            100.0 * count / universe_count
        ),
        "failure_gt_10_bpm_percent": float(
            100.0
            * np.mean(absolute_error[accepted] > 10.0)
        ),
    }


def select_threshold(
    scores: np.ndarray,
    absolute_error: np.ndarray,
    design_mask: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    maximum_failure_percent: float,
    minimum_coverage_percent: float,
    maximum_threshold: float = 100.0,
) -> dict[str, float | bool]:
    """只用设计集选择满足误差目标且覆盖率最大的阈值。"""
    candidates = [
        threshold_statistics(
            scores,
            absolute_error,
            design_mask,
            eligible_mask,
            threshold,
        )
        for threshold in THRESHOLD_GRID
        if threshold <= maximum_threshold
    ]
    candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["coverage_percent"]
            >= minimum_coverage_percent
        )
    ]
    if not candidates:
        raise ValueError(
            "没有达到最低覆盖率的候选阈值。"
        )

    satisfying = [
        candidate
        for candidate in candidates
        if (
            candidate[
                "failure_gt_10_bpm_percent"
            ]
            <= maximum_failure_percent
        )
    ]

    if satisfying:
        selected = max(
            satisfying,
            key=lambda candidate: (
                candidate["coverage_percent"],
                -candidate["threshold"],
            ),
        )
        target_met = True
    else:
        # 如果目标不可达，不隐藏失败；选择大误差率最低的
        # 可用候选，并在输出中明确标记 target_met=False。
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate[
                    "failure_gt_10_bpm_percent"
                ],
                -candidate["coverage_percent"],
                candidate["threshold"],
            ),
        )
        target_met = False

    return {
        **selected,
        "target_met": target_met,
        "target_max_failure_percent": (
            maximum_failure_percent
        ),
        "required_minimum_coverage_percent": (
            minimum_coverage_percent
        ),
    }


def dominant_soft_reason(
    components: dict[str, np.ndarray],
) -> np.ndarray:
    """把最低质量分量转换为可解释的拒绝原因。"""
    ordered_names = (
        "spectral_quality",
        "periodicity_quality",
        "amplitude_quality",
    )
    reason_names = np.asarray(
        (
            "spectral_ambiguity",
            "weak_periodicity",
            "amplitude_anomaly",
        ),
        dtype=object,
    )
    matrix = np.column_stack(
        [components[name] for name in ordered_names]
    )
    return reason_names[np.argmin(matrix, axis=1)]


def assign_states(
    scores: np.ndarray,
    components: dict[str, np.ndarray],
    hard_masks: dict[str, np.ndarray],
    usable_threshold: float,
    accepted_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """输出 usable、low_confidence、reject 和对应原因。"""
    count = len(scores)
    states = np.full(count, "reject", dtype=object)
    reasons = np.full(count, "none", dtype=object)

    hard_reject = np.zeros(count, dtype=bool)
    for reason_name in (
        "invalid_data",
        "flatline",
        "possible_clipping",
    ):
        reason_mask = (
            hard_masks[reason_name] & ~hard_reject
        )
        reasons[reason_mask] = reason_name
        hard_reject |= hard_masks[reason_name]

    soft_reasons = dominant_soft_reason(components)
    eligible = ~hard_reject & np.isfinite(scores)

    usable = eligible & (scores >= usable_threshold)
    low_confidence = (
        eligible
        & (scores >= accepted_threshold)
        & ~usable
    )
    soft_reject = (
        eligible
        & (scores < accepted_threshold)
    )

    states[usable] = "usable"
    states[low_confidence] = "low_confidence"
    reasons[low_confidence] = soft_reasons[
        low_confidence
    ]
    reasons[soft_reject] = soft_reasons[soft_reject]

    return states, reasons


def summarize_selection(
    absolute_error: np.ndarray,
    universe_mask: np.ndarray,
    selected_mask: np.ndarray,
) -> dict[str, float | int | None]:
    """汇总指定数据范围中的覆盖率和 HR 误差。"""
    selected = universe_mask & selected_mask
    count = int(np.sum(selected))
    total = int(np.sum(universe_mask))

    if count == 0:
        return {
            "count": 0,
            "coverage_percent": 0.0,
            "mae_bpm": None,
            "rmse_bpm": None,
            "p90_bpm": None,
            "within_3_bpm_percent": None,
            "failure_gt_10_bpm_percent": None,
        }

    errors = absolute_error[selected]
    return {
        "count": count,
        "coverage_percent": float(
            100.0 * count / total
        ),
        "mae_bpm": float(np.mean(errors)),
        "rmse_bpm": float(
            np.sqrt(np.mean(errors**2))
        ),
        "p90_bpm": float(
            np.percentile(errors, 90)
        ),
        "within_3_bpm_percent": float(
            100.0 * np.mean(errors <= 3.0)
        ),
        "failure_gt_10_bpm_percent": float(
            100.0 * np.mean(errors > 10.0)
        ),
    }


def build_partition_metrics(
    absolute_error: np.ndarray,
    universe_mask: np.ndarray,
    states: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """生成一个数据分区的完整门控指标。"""
    return {
        "all_windows": summarize_selection(
            absolute_error,
            universe_mask,
            np.ones(len(states), dtype=bool),
        ),
        "usable": summarize_selection(
            absolute_error,
            universe_mask,
            states == "usable",
        ),
        "usable_plus_low_confidence": (
            summarize_selection(
                absolute_error,
                universe_mask,
                np.isin(
                    states,
                    ("usable", "low_confidence"),
                ),
            )
        ),
        "reject": summarize_selection(
            absolute_error,
            universe_mask,
            states == "reject",
        ),
    }


def build_activity_metrics(
    data: dict[str, np.ndarray],
    validation_mask: np.ndarray,
    states: np.ndarray,
) -> list[dict[str, Any]]:
    """生成验证集的分活动门控指标。"""
    activity_pairs = sorted(
        {
            (int(activity_id), str(activity_name))
            for activity_id, activity_name in zip(
                data["activity_id"],
                data["activity_name"],
            )
        }
    )
    accepted = np.isin(
        states,
        ("usable", "low_confidence"),
    )
    results: list[dict[str, Any]] = []

    for activity_id, activity_name in activity_pairs:
        activity_mask = (
            validation_mask
            & (data["activity_id"] == activity_id)
        )
        if not np.any(activity_mask):
            continue

        results.append(
            {
                "activity_id": activity_id,
                "activity_name": activity_name,
                "baseline": summarize_selection(
                    data["absolute_error_bpm"],
                    activity_mask,
                    np.ones(len(states), dtype=bool),
                ),
                "usable": summarize_selection(
                    data["absolute_error_bpm"],
                    activity_mask,
                    states == "usable",
                ),
                "usable_plus_low_confidence": (
                    summarize_selection(
                        data["absolute_error_bpm"],
                        activity_mask,
                        accepted,
                    )
                ),
            }
        )

    return results


def build_coverage_curve(
    scores: np.ndarray,
    absolute_error: np.ndarray,
    design_mask: np.ndarray,
    validation_mask: np.ndarray,
    eligible_mask: np.ndarray,
) -> list[dict[str, float]]:
    """计算固定阈值在设计集和验证集上的覆盖率—误差曲线。"""
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLD_GRID:
        design = threshold_statistics(
            scores,
            absolute_error,
            design_mask,
            eligible_mask,
            float(threshold),
        )
        validation = threshold_statistics(
            scores,
            absolute_error,
            validation_mask,
            eligible_mask,
            float(threshold),
        )
        rows.append(
            {
                "threshold": float(threshold),
                "design_count": int(design["count"]),
                "design_coverage_percent": float(
                    design["coverage_percent"]
                ),
                "design_failure_gt_10_bpm_percent": (
                    float(
                        design[
                            "failure_gt_10_bpm_percent"
                        ]
                    )
                ),
                "validation_count": int(
                    validation["count"]
                ),
                "validation_coverage_percent": float(
                    validation["coverage_percent"]
                ),
                "validation_failure_gt_10_bpm_percent": (
                    float(
                        validation[
                            "failure_gt_10_bpm_percent"
                        ]
                    )
                ),
            }
        )
    return rows


def write_json(
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    """以稳定格式写入 UTF-8 JSON。"""
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")


def write_window_results(
    output_path: Path,
    data: dict[str, np.ndarray],
    design_mask: np.ndarray,
    scores: np.ndarray,
    components: dict[str, np.ndarray],
    states: np.ndarray,
    reasons: np.ndarray,
) -> None:
    """保存每个窗口的分数、状态和可解释原因。"""
    fieldnames = (
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
        "spectral_quality",
        "periodicity_quality",
        "amplitude_quality",
        "state",
        "reason",
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index in range(len(states)):
            writer.writerow(
                {
                    "subject_id": data[
                        "subject_id"
                    ][index],
                    "split": (
                        "design"
                        if design_mask[index]
                        else "validation"
                    ),
                    "window_start_s": float(
                        data["window_start_s"][index]
                    ),
                    "window_center_s": float(
                        data["window_center_s"][index]
                    ),
                    "window_end_s": float(
                        data["window_end_s"][index]
                    ),
                    "activity_id": int(
                        data["activity_id"][index]
                    ),
                    "activity_name": data[
                        "activity_name"
                    ][index],
                    "ppg_hr_bpm": float(
                        data["ppg_hr_bpm"][index]
                    ),
                    "ecg_hr_bpm": float(
                        data["ecg_hr_bpm"][index]
                    ),
                    "absolute_error_bpm": float(
                        data[
                            "absolute_error_bpm"
                        ][index]
                    ),
                    "sqi_score": float(scores[index]),
                    "spectral_quality": float(
                        components[
                            "spectral_quality"
                        ][index]
                    ),
                    "periodicity_quality": float(
                        components[
                            "periodicity_quality"
                        ][index]
                    ),
                    "amplitude_quality": float(
                        components[
                            "amplitude_quality"
                        ][index]
                    ),
                    "state": states[index],
                    "reason": reasons[index],
                }
            )


def write_coverage_curve(
    output_path: Path,
    rows: list[dict[str, float]],
) -> None:
    """保存完整阈值扫描结果。"""
    fieldnames = tuple(rows[0].keys())
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_validation_tradeoff(
    output_path: Path,
    coverage_curve: list[dict[str, float]],
    usable_threshold: float,
    accepted_threshold: float,
) -> None:
    """绘制设计集与验证集的覆盖率—大误差率曲线。"""
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(10, 6))

    for prefix, label, color in (
        ("design", "S1–S10 设计集", "#4472C4"),
        (
            "validation",
            "S11–S15 内部验证集",
            "#ED7D31",
        ),
    ):
        coverage = [
            row[f"{prefix}_coverage_percent"]
            for row in coverage_curve
        ]
        failure = [
            row[
                f"{prefix}_failure_gt_10_bpm_percent"
            ]
            for row in coverage_curve
        ]
        axis.plot(
            coverage,
            failure,
            color=color,
            linewidth=2,
            label=label,
        )

    for threshold, label, marker in (
        (usable_threshold, "usable 阈值", "o"),
        (
            accepted_threshold,
            "usable + low 阈值",
            "s",
        ),
    ):
        row = min(
            coverage_curve,
            key=lambda item: abs(
                item["threshold"] - threshold
            ),
        )
        axis.scatter(
            row["validation_coverage_percent"],
            row[
                "validation_failure_gt_10_bpm_percent"
            ],
            s=70,
            marker=marker,
            zorder=3,
            label=(
                f"{label}：{threshold:.1f}"
            ),
        )

    axis.axhline(
        USABLE_MAX_FAILURE_PERCENT,
        color="#70AD47",
        linestyle="--",
        linewidth=1,
        label="usable 研究目标：大误差率 10%",
    )
    axis.axhline(
        ACCEPTED_MAX_FAILURE_PERCENT,
        color="#A5A5A5",
        linestyle="--",
        linewidth=1,
        label="宽松研究目标：大误差率 25%",
    )
    axis.set_xlabel("保留窗口覆盖率（%）")
    axis.set_ylabel("保留窗口中误差 >10 bpm 的比例（%）")
    axis.set_title(
        "PPG-only SQI V0：覆盖率与心率大误差率权衡"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_activity_validation(
    output_path: Path,
    activity_metrics: list[dict[str, Any]],
) -> None:
    """绘制验证集各活动门控前后的误差与覆盖率。"""
    configure_plot_style()
    labels = [
        item["activity_name"]
        for item in activity_metrics
    ]
    baseline_failure = [
        item["baseline"][
            "failure_gt_10_bpm_percent"
        ]
        for item in activity_metrics
    ]
    accepted_failure = [
        item["usable_plus_low_confidence"][
            "failure_gt_10_bpm_percent"
        ]
        for item in activity_metrics
    ]
    accepted_coverage = [
        item["usable_plus_low_confidence"][
            "coverage_percent"
        ]
        for item in activity_metrics
    ]

    x_positions = np.arange(len(labels))
    width = 0.34
    figure, axis_error = plt.subplots(
        figsize=(13, 7)
    )
    axis_error.bar(
        x_positions - width / 2,
        baseline_failure,
        width,
        color="#A5A5A5",
        label="门控前大误差率",
    )
    axis_error.bar(
        x_positions + width / 2,
        accepted_failure,
        width,
        color="#4472C4",
        label="门控后大误差率",
    )
    axis_error.set_ylabel("误差 >10 bpm 的窗口比例（%）")
    axis_error.set_ylim(0.0, 100.0)
    axis_error.set_xticks(x_positions)
    axis_error.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )
    axis_error.grid(axis="y", alpha=0.25)

    axis_coverage = axis_error.twinx()
    axis_coverage.plot(
        x_positions,
        accepted_coverage,
        color="#ED7D31",
        marker="o",
        linewidth=2,
        label="门控后覆盖率",
    )
    axis_coverage.set_ylabel("门控后覆盖率（%）")
    axis_coverage.set_ylim(0.0, 100.0)

    handles_error, labels_error = (
        axis_error.get_legend_handles_labels()
    )
    handles_coverage, labels_coverage = (
        axis_coverage.get_legend_handles_labels()
    )
    axis_error.legend(
        handles_error + handles_coverage,
        labels_error + labels_coverage,
        loc="upper left",
    )
    axis_error.set_title(
        "S11–S15：PPG-only SQI V0 分活动验证"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def count_csv_rows(csv_path: Path) -> int:
    """统计 CSV 数据行，用于输出回归检查。"""
    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))


def validate_output_files(
    paths: dict[str, Path],
) -> None:
    """验证输出数量、JSON 可读性和图片非空。"""
    window_count = count_csv_rows(paths["windows"])
    if window_count != EXPECTED_TOTAL_WINDOWS:
        raise ValueError(
            "SQI 窗口输出数量检查失败："
            f"期望 {EXPECTED_TOTAL_WINDOWS}，"
            f"实际 {window_count}。"
        )

    curve_count = count_csv_rows(
        paths["coverage_curve"]
    )
    if curve_count != len(THRESHOLD_GRID):
        raise ValueError(
            "阈值扫描输出数量检查失败："
            f"期望 {len(THRESHOLD_GRID)}，"
            f"实际 {curve_count}。"
        )

    for json_name in ("parameters", "metrics"):
        with paths[json_name].open(
            "r",
            encoding="utf-8",
        ) as json_file:
            json.load(json_file)

    for image_name in (
        "validation_tradeoff",
        "activity_validation",
    ):
        if paths[image_name].stat().st_size == 0:
            raise ValueError(
                f"输出图片为空：{paths[image_name]}"
            )


def write_outputs(
    data: dict[str, np.ndarray],
    design_mask: np.ndarray,
    validation_mask: np.ndarray,
    eligible_mask: np.ndarray,
    scores: np.ndarray,
    components: dict[str, np.ndarray],
    parameters: dict[
        str,
        dict[str, float | bool],
    ],
    usable_selection: dict[str, float | bool],
    accepted_selection: dict[str, float | bool],
    states: np.ndarray,
    reasons: np.ndarray,
) -> dict[str, Path]:
    """显式生成并验证 SQI V0 的全部证据文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "windows": (
            OUTPUT_DIR
            / "S1_S15_ppg_only_sqi_v0_windows.csv"
        ),
        "parameters": (
            OUTPUT_DIR / "sqi_v0_parameters.json"
        ),
        "metrics": OUTPUT_DIR / "sqi_v0_metrics.json",
        "coverage_curve": (
            OUTPUT_DIR / "sqi_v0_coverage_curve.csv"
        ),
        "validation_tradeoff": (
            OUTPUT_DIR
            / "sqi_v0_validation_tradeoff.png"
        ),
        "activity_validation": (
            OUTPUT_DIR
            / "sqi_v0_activity_validation.png"
        ),
    }

    activity_metrics = build_activity_metrics(
        data,
        validation_mask,
        states,
    )
    coverage_curve = build_coverage_curve(
        scores,
        data["absolute_error_bpm"],
        design_mask,
        validation_mask,
        eligible_mask,
    )

    write_window_results(
        paths["windows"],
        data,
        design_mask,
        scores,
        components,
        states,
        reasons,
    )
    write_coverage_curve(
        paths["coverage_curve"],
        coverage_curve,
    )

    parameters_payload = {
        "schema_version": 1,
        "input_file": str(INPUT_CSV),
        "split": {
            "design_subjects": list(DESIGN_SUBJECTS),
            "validation_subjects": list(
                VALIDATION_SUBJECTS
            ),
            "design_window_count": int(
                np.sum(design_mask)
            ),
            "validation_window_count": int(
                np.sum(validation_mask)
            ),
        },
        "normalization": {
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
            "feature_parameters": parameters,
        },
        "weights": SQI_WEIGHTS,
        "hard_reject_rules": {
            "minimum_finite_ratio": (
                MIN_FINITE_RATIO
            ),
            "maximum_flatline_ratio": (
                MAX_FLATLINE_RATIO
            ),
            "maximum_extreme_repeat_ratio": (
                MAX_EXTREME_REPEAT_RATIO
            ),
            "status": (
                "engineering safeguards; "
                "not product thresholds"
            ),
        },
        "thresholds": {
            "usable": usable_selection,
            "usable_plus_low_confidence": (
                accepted_selection
            ),
            "status": (
                "research targets fitted on S1-S10"
            ),
        },
    }
    write_json(
        paths["parameters"],
        parameters_payload,
    )

    validation_non_usable = (
        validation_mask & (states != "usable")
    )
    metrics_payload = {
        "schema_version": 1,
        "design": build_partition_metrics(
            data["absolute_error_bpm"],
            design_mask,
            states,
        ),
        "validation": build_partition_metrics(
            data["absolute_error_bpm"],
            validation_mask,
            states,
        ),
        "validation_by_activity": activity_metrics,
        "validation_state_counts": dict(
            sorted(
                Counter(
                    states[validation_mask].tolist()
                ).items()
            )
        ),
        "validation_non_usable_reason_counts": dict(
            sorted(
                Counter(
                    reasons[
                        validation_non_usable
                    ].tolist()
                ).items()
            )
        ),
        "interpretation_boundary": (
            "This SQI estimates whether a window is "
            "suitable for the current PPG-only HR "
            "baseline. It is not an independently "
            "labelled physiological signal-quality "
            "ground truth."
        ),
    }
    write_json(paths["metrics"], metrics_payload)

    plot_validation_tradeoff(
        paths["validation_tradeoff"],
        coverage_curve,
        float(usable_selection["threshold"]),
        float(accepted_selection["threshold"]),
    )
    plot_activity_validation(
        paths["activity_validation"],
        activity_metrics,
    )
    validate_output_files(paths)
    return paths


def format_value(
    value: float | int | None,
    suffix: str = "",
) -> str:
    """统一终端指标格式。"""
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}{suffix}"


def print_metric_row(
    name: str,
    metrics: dict[str, float | int | None],
) -> None:
    """打印一行覆盖率与误差结果。"""
    print(
        f"  {name:<18}"
        f" n={format_value(metrics['count']):>6}"
        "  coverage="
        f"{format_value(metrics['coverage_percent'], '%'):>7}"
        "  MAE="
        f"{format_value(metrics['mae_bpm'], ' bpm'):>10}"
        "  P90="
        f"{format_value(metrics['p90_bpm'], ' bpm'):>10}"
        "  >10 bpm="
        f"{format_value(metrics['failure_gt_10_bpm_percent'], '%'):>7}"
    )


def print_evaluation(
    title: str,
    absolute_error: np.ndarray,
    universe_mask: np.ndarray,
    states: np.ndarray,
) -> None:
    """打印一个数据分区上的基线和门控结果。"""
    print(f"\n{title}")
    print_metric_row(
        "全部窗口",
        summarize_selection(
            absolute_error,
            universe_mask,
            np.ones(len(states), dtype=bool),
        ),
    )
    print_metric_row(
        "usable",
        summarize_selection(
            absolute_error,
            universe_mask,
            states == "usable",
        ),
    )
    print_metric_row(
        "usable + low",
        summarize_selection(
            absolute_error,
            universe_mask,
            np.isin(
                states,
                ("usable", "low_confidence"),
            ),
        ),
    )
    print_metric_row(
        "reject",
        summarize_selection(
            absolute_error,
            universe_mask,
            states == "reject",
        ),
    )


def print_activity_evaluation(
    data: dict[str, np.ndarray],
    validation_mask: np.ndarray,
    states: np.ndarray,
) -> None:
    """按活动显示验证集上的覆盖率和大误差率。"""
    print("\nS11–S15 分活动内部验证")
    activity_pairs = sorted(
        {
            (int(activity_id), str(activity_name))
            for activity_id, activity_name in zip(
                data["activity_id"],
                data["activity_name"],
            )
        }
    )

    accepted = np.isin(
        states,
        ("usable", "low_confidence"),
    )
    for activity_id, activity_name in activity_pairs:
        activity_mask = (
            validation_mask
            & (data["activity_id"] == activity_id)
        )
        if not np.any(activity_mask):
            continue

        baseline = summarize_selection(
            data["absolute_error_bpm"],
            activity_mask,
            np.ones(len(states), dtype=bool),
        )
        gated = summarize_selection(
            data["absolute_error_bpm"],
            activity_mask,
            accepted,
        )
        print(
            f"  {activity_id} {activity_name:<12}"
            f" baseline_n={baseline['count']:>5}"
            "  baseline_>10="
            f"{format_value(baseline['failure_gt_10_bpm_percent'], '%'):>7}"
            "  accepted_coverage="
            f"{format_value(gated['coverage_percent'], '%'):>7}"
            "  accepted_>10="
            f"{format_value(gated['failure_gt_10_bpm_percent'], '%'):>7}"
        )


def main() -> int:
    """执行 SQI V0；仅在显式请求时写入证据文件。"""
    args = parse_args()
    data = load_feature_table(INPUT_CSV)
    design_mask, validation_mask = build_subject_masks(
        data["subject_id"]
    )
    hard_masks = build_hard_reject_masks(data)
    hard_reject = np.logical_or.reduce(
        tuple(hard_masks.values())
    )
    eligible = ~hard_reject

    score, components, parameters = build_sqi(
        data,
        design_mask & eligible,
    )

    usable_selection = select_threshold(
        score,
        data["absolute_error_bpm"],
        design_mask,
        eligible,
        maximum_failure_percent=(
            USABLE_MAX_FAILURE_PERCENT
        ),
        minimum_coverage_percent=(
            USABLE_MIN_COVERAGE_PERCENT
        ),
    )
    accepted_selection = select_threshold(
        score,
        data["absolute_error_bpm"],
        design_mask,
        eligible,
        maximum_failure_percent=(
            ACCEPTED_MAX_FAILURE_PERCENT
        ),
        minimum_coverage_percent=(
            ACCEPTED_MIN_COVERAGE_PERCENT
        ),
        maximum_threshold=float(
            usable_selection["threshold"]
        ),
    )

    states, reasons = assign_states(
        score,
        components,
        hard_masks,
        usable_threshold=float(
            usable_selection["threshold"]
        ),
        accepted_threshold=float(
            accepted_selection["threshold"]
        ),
    )

    print("PPG-only SQI V0")
    print(f"输入文件：{INPUT_CSV}")
    print(
        "数据划分："
        f"S1–S10={int(np.sum(design_mask))}，"
        f"S11–S15={int(np.sum(validation_mask))}"
    )

    print("\n设计集归一化参数")
    for feature_name, feature_parameters in (
        parameters.items()
    ):
        transform = (
            "log1p"
            if feature_parameters["log_transform"]
            else "none"
        )
        print(
            f"  {feature_name}: "
            f"P10={feature_parameters['p10']:.6f}, "
            f"P90={feature_parameters['p90']:.6f}, "
            f"transform={transform}"
        )

    print("\n固定权重")
    for component_name, weight in SQI_WEIGHTS.items():
        print(f"  {component_name}: {weight:.2f}")

    print("\n设计集阈值选择")
    for name, selection in (
        ("usable", usable_selection),
        ("usable + low", accepted_selection),
    ):
        print(
            f"  {name}: "
            f"threshold={selection['threshold']:.1f}, "
            f"coverage={selection['coverage_percent']:.2f}%, "
            ">10 bpm="
            f"{selection['failure_gt_10_bpm_percent']:.2f}%, "
            f"target_met={selection['target_met']}"
        )

    print_evaluation(
        "S1–S10 设计集",
        data["absolute_error_bpm"],
        design_mask,
        states,
    )
    print_evaluation(
        "S11–S15 内部独立验证集",
        data["absolute_error_bpm"],
        validation_mask,
        states,
    )
    print_activity_evaluation(
        data,
        validation_mask,
        states,
    )

    print("\nS11–S15 状态数量")
    validation_states = Counter(
        states[validation_mask].tolist()
    )
    for state_name in (
        "usable",
        "low_confidence",
        "reject",
    ):
        print(
            f"  {state_name}: "
            f"{validation_states[state_name]}"
        )

    print("\nS11–S15 非 usable 原因")
    validation_reasons = Counter(
        reasons[
            validation_mask
            & (states != "usable")
        ].tolist()
    )
    for reason_name, count in sorted(
        validation_reasons.items()
    ):
        print(f"  {reason_name}: {count}")

    print(
        "\n说明：阈值和归一化参数仅由 S1–S10 生成；"
        "S11–S15 的误差没有参与参数选择。"
    )
    print(
        "本结果衡量 SQI 对当前 PPG-only HR 基线的门控能力，"
        "不能替代人工信号质量标签或真实设备验证。"
    )

    if args.write_outputs:
        output_paths = write_outputs(
            data,
            design_mask,
            validation_mask,
            eligible,
            score,
            components,
            parameters,
            usable_selection,
            accepted_selection,
            states,
            reasons,
        )
        print("\n已生成并验证证据文件")
        for output_name, output_path in (
            output_paths.items()
        ):
            print(f"  {output_name}: {output_path}")
    else:
        print(
            "\n当前为只读运行；如需生成证据文件，"
            "请显式添加 --write-outputs。"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
