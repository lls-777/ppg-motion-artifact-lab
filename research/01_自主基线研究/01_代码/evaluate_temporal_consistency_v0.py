#!/usr/bin/env python3
"""评价 ACC-aware SQI V1 之后的因果时间连续性特征。

本脚本只读取 S1-S10 已冻结的窗口级设计结果，不读取 S11-S15。

研究目标：
1. 只使用当前和过去窗口，计算心率跳变、历史中位数偏差和
   连续运动风险时长；
2. 比较时间特征对 ACC-aware V1 剩余错误的增量识别能力；
3. 通过条件式 leave-one-subject-out 检查增量规则的跨人稳定性；
4. 根据透明的研究准入条件，决定是否建议把时间模块加入
   后续独立验证。

注意：这里的 LOSO 以已经冻结的静态 ACC-aware V1 状态为起点，
只评价时间模块的增量稳定性，不是完整管线的端到端 LOSO。

默认只在终端输出；只有显式传入 --write-outputs 才保存 CSV 和 JSON。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


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
    / "temporal_consistency_v0_design"
)

DESIGN_SUBJECTS = tuple(range(1, 11))
EXPECTED_DESIGN_WINDOWS = 43_215
EXPECTED_TEMPORAL_UNIVERSE_WINDOWS = 11_171

WINDOW_SECONDS = 8.0
STEP_SECONDS = 2.0
HISTORY_LOOKBACK_SECONDS = 20.0
MIN_HISTORY_POINTS = 3
ERROR_THRESHOLD_BPM = 10.0

# 心率阈值从每轮训练受试者的时间特征分位数产生。
TEMPORAL_PERCENTILES = (90.0, 92.5, 95.0, 97.5, 99.0)

# 短时间持续运动在只读探索中会大量误拒正确窗口，因此仍保留
# 一组从中等到极长的候选值，用实验明确展示其能力边界。
DURATION_THRESHOLDS_SECONDS = (
    80.0,
    112.0,
    160.0,
    224.0,
    320.0,
    400.0,
    480.0,
)

# 训练选择约束：时间模块不能通过大量丢弃正确窗口换取低误差。
MIN_MACRO_CORRECT_RETENTION_PERCENT = 95.0
MIN_SUBJECT_CORRECT_RETENTION_PERCENT = 85.0

# 以下是是否值得进入独立验证的研究准入条件，不是产品标准。
MIN_LOSO_MEAN_ERROR_CAPTURE_PERCENT = 20.0
MIN_LOSO_SUBJECTS_WITH_CAPTURE = 8

FAMILIES = (
    "jump_only",
    "history_deviation_only",
    "risk_duration_only",
    "jump_or_history_deviation",
)

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
    "ppg_sqi_state",
    "ppg_sqi_reason",
    "motion_flag",
    "collision_flag",
    "acc_aware_state",
    "acc_aware_reason",
)

PARAMETER_FIELDS = (
    "jump_threshold_bpm",
    "history_deviation_threshold_bpm",
    "duration_threshold_s",
)

METRIC_FIELDS = (
    "macro_correct_retention_percent",
    "minimum_subject_correct_retention_percent",
    "macro_error_capture_percent",
    "retained_coverage_percent",
    "remaining_failure_percent",
)

TEMPORAL_OUTPUT_FIELDS = (
    "history_usable_count",
    "jump_from_last_usable_bpm",
    "history_median_deviation_bpm",
    "motion_run_seconds",
    "collision_run_seconds",
    "candidate_temporal_reject",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评价因果时间连续性特征 V0"
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="显式保存时间特征、消融结果和研究决策",
    )
    return parser.parse_args()


def parse_bool(value: str, column: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{column} 出现非法布尔值：{value!r}")


def load_design_data(
    csv_path: Path,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    """读取并验证 S1-S10 的 ACC-aware 窗口表。"""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"找不到 ACC-aware 窗口表：{csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in columns
        ]
        if missing:
            raise ValueError(
                "ACC-aware 窗口表缺少字段："
                + ", ".join(missing)
            )
        rows = list(reader)

    if len(rows) != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            f"窗口数异常：期望 {EXPECTED_DESIGN_WINDOWS}，"
            f"实际 {len(rows)}"
        )

    data: dict[str, np.ndarray] = {
        "subject_id": np.asarray(
            [int(row["subject_id"]) for row in rows],
            dtype=int,
        ),
        "window_center_s": np.asarray(
            [float(row["window_center_s"]) for row in rows],
            dtype=float,
        ),
        "ppg_hr_bpm": np.asarray(
            [float(row["ppg_hr_bpm"]) for row in rows],
            dtype=float,
        ),
        "absolute_error_bpm": np.asarray(
            [float(row["absolute_error_bpm"]) for row in rows],
            dtype=float,
        ),
        "ppg_sqi_state": np.asarray(
            [row["ppg_sqi_state"] for row in rows],
            dtype=object,
        ),
        "acc_aware_state": np.asarray(
            [row["acc_aware_state"] for row in rows],
            dtype=object,
        ),
        "motion_flag": np.asarray(
            [
                parse_bool(row["motion_flag"], "motion_flag")
                for row in rows
            ],
            dtype=bool,
        ),
        "collision_flag": np.asarray(
            [
                parse_bool(
                    row["collision_flag"],
                    "collision_flag",
                )
                for row in rows
            ],
            dtype=bool,
        ),
    }

    actual_subjects = tuple(
        sorted(set(data["subject_id"].tolist()))
    )
    if actual_subjects != DESIGN_SUBJECTS:
        raise ValueError(
            f"受试者范围异常：期望 {DESIGN_SUBJECTS}，"
            f"实际 {actual_subjects}"
        )

    for subject_id in DESIGN_SUBJECTS:
        indices = np.flatnonzero(
            data["subject_id"] == subject_id
        )
        time_steps = np.diff(
            data["window_center_s"][indices]
        )
        if not np.allclose(time_steps, STEP_SECONDS):
            raise ValueError(
                f"S{subject_id} 的窗口不是连续 "
                f"{STEP_SECONDS:g} 秒步长"
            )

    return rows, data


def build_causal_features(
    data: dict[str, np.ndarray],
) -> None:
    """按受试者计算只依赖过去高可信窗口的时间特征。"""
    row_count = len(data["subject_id"])
    history_count = np.zeros(row_count, dtype=int)
    jump = np.full(row_count, np.nan, dtype=float)
    deviation = np.full(row_count, np.nan, dtype=float)
    motion_run = np.zeros(row_count, dtype=float)
    collision_run = np.zeros(row_count, dtype=float)

    for subject_id in DESIGN_SUBJECTS:
        history: deque[tuple[float, float]] = deque()
        motion_run_count = 0
        collision_run_count = 0

        for index in np.flatnonzero(
            data["subject_id"] == subject_id
        ):
            current_time = data["window_center_s"][index]

            while (
                history
                and current_time - history[0][0]
                > HISTORY_LOOKBACK_SECONDS
            ):
                history.popleft()

            history_count[index] = len(history)
            if history:
                jump[index] = abs(
                    data["ppg_hr_bpm"][index]
                    - history[-1][1]
                )
            if len(history) >= MIN_HISTORY_POINTS:
                history_median = float(
                    np.median(
                        [item[1] for item in history]
                    )
                )
                deviation[index] = abs(
                    data["ppg_hr_bpm"][index]
                    - history_median
                )

            if data["motion_flag"][index]:
                motion_run_count += 1
                motion_run[index] = (
                    WINDOW_SECONDS
                    + STEP_SECONDS
                    * (motion_run_count - 1)
                )
            else:
                motion_run_count = 0

            if data["collision_flag"][index]:
                collision_run_count += 1
                collision_run[index] = (
                    WINDOW_SECONDS
                    + STEP_SECONDS
                    * (collision_run_count - 1)
                )
            else:
                collision_run_count = 0

            # 当前窗口先完成特征计算，再进入历史，避免使用自身。
            if (
                data["ppg_sqi_state"][index] == "usable"
                and data["acc_aware_state"][index] == "usable"
            ):
                history.append(
                    (
                        current_time,
                        data["ppg_hr_bpm"][index],
                    )
                )

    data["history_usable_count"] = history_count
    data["jump_from_last_usable_bpm"] = jump
    data["history_median_deviation_bpm"] = deviation
    data["motion_run_seconds"] = motion_run
    data["collision_run_seconds"] = collision_run


def temporal_universe(
    data: dict[str, np.ndarray],
) -> np.ndarray:
    """返回静态 ACC-aware V1 当前仍保留的 PPG-only 窗口。"""
    universe = (
        (data["ppg_sqi_state"] == "usable")
        & np.isin(
            data["acc_aware_state"],
            ("usable", "low_confidence"),
        )
    )
    actual_count = int(np.sum(universe))
    if actual_count != EXPECTED_TEMPORAL_UNIVERSE_WINDOWS:
        raise ValueError(
            "时间模块输入窗口数异常："
            f"期望 {EXPECTED_TEMPORAL_UNIVERSE_WINDOWS}，"
            f"实际 {actual_count}"
        )
    return universe


def safe_percent(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def evaluate_rejection(
    data: dict[str, np.ndarray],
    evaluation_mask: np.ndarray,
    reject_mask: np.ndarray,
) -> dict[str, float]:
    """按受试者平衡地评价时间规则的增量拒绝效果。"""
    errors = data["absolute_error_bpm"]
    subjects = data["subject_id"]

    correct_retentions: list[float] = []
    error_captures: list[float] = []

    for subject_id in sorted(
        set(subjects[evaluation_mask].tolist())
    ):
        subject_mask = (
            evaluation_mask
            & (subjects == subject_id)
        )
        correct = (
            subject_mask
            & (errors <= ERROR_THRESHOLD_BPM)
        )
        failure = (
            subject_mask
            & (errors > ERROR_THRESHOLD_BPM)
        )

        if np.any(correct):
            correct_retentions.append(
                float(
                    100.0
                    * np.mean(~reject_mask[correct])
                )
            )
        if np.any(failure):
            error_captures.append(
                float(
                    100.0
                    * np.mean(reject_mask[failure])
                )
            )

    retained = evaluation_mask & ~reject_mask
    return {
        "macro_correct_retention_percent": float(
            np.mean(correct_retentions)
        ),
        "minimum_subject_correct_retention_percent": float(
            np.min(correct_retentions)
        ),
        "macro_error_capture_percent": float(
            np.mean(error_captures)
        ),
        "retained_coverage_percent": safe_percent(
            int(np.sum(retained)),
            int(np.sum(evaluation_mask)),
        ),
        "remaining_failure_percent": float(
            100.0
            * np.mean(
                errors[retained] > ERROR_THRESHOLD_BPM
            )
        ),
    }


def candidate_parameters(
    family: str,
    data: dict[str, np.ndarray],
    training_mask: np.ndarray,
) -> list[dict[str, float]]:
    """只根据训练受试者产生候选阈值。"""
    jump = data["jump_from_last_usable_bpm"]
    deviation = data["history_median_deviation_bpm"]

    jump_valid = training_mask & np.isfinite(jump)
    deviation_valid = (
        training_mask & np.isfinite(deviation)
    )

    jump_thresholds = [
        float(np.percentile(jump[jump_valid], percentile))
        for percentile in TEMPORAL_PERCENTILES
    ]
    deviation_thresholds = [
        float(
            np.percentile(
                deviation[deviation_valid],
                percentile,
            )
        )
        for percentile in TEMPORAL_PERCENTILES
    ]

    if family == "jump_only":
        return [
            {"jump_threshold_bpm": threshold}
            for threshold in jump_thresholds
        ]
    if family == "history_deviation_only":
        return [
            {
                "history_deviation_threshold_bpm":
                threshold
            }
            for threshold in deviation_thresholds
        ]
    if family == "risk_duration_only":
        return [
            {"duration_threshold_s": threshold}
            for threshold in DURATION_THRESHOLDS_SECONDS
        ]
    if family == "jump_or_history_deviation":
        return [
            {
                "jump_threshold_bpm": jump_threshold,
                "history_deviation_threshold_bpm":
                deviation_threshold,
            }
            for jump_threshold, deviation_threshold
            in product(
                jump_thresholds,
                deviation_thresholds,
            )
        ]
    raise ValueError(f"未知时间规则：{family}")


def apply_rule(
    family: str,
    data: dict[str, np.ndarray],
    parameters: dict[str, float],
) -> np.ndarray:
    """应用一个候选时间规则，不使用 ECG 或活动标签。"""
    jump = data["jump_from_last_usable_bpm"]
    deviation = data["history_median_deviation_bpm"]

    if family == "jump_only":
        return (
            np.isfinite(jump)
            & (
                jump
                >= parameters["jump_threshold_bpm"]
            )
        )

    if family == "history_deviation_only":
        return (
            np.isfinite(deviation)
            & (
                deviation
                >= parameters[
                    "history_deviation_threshold_bpm"
                ]
            )
        )

    if family == "risk_duration_only":
        duration_threshold = parameters[
            "duration_threshold_s"
        ]
        single_risk = (
            data["acc_aware_state"] == "low_confidence"
        )
        return single_risk & (
            (
                data["motion_flag"]
                & (
                    data["motion_run_seconds"]
                    >= duration_threshold
                )
            )
            | (
                data["collision_flag"]
                & (
                    data["collision_run_seconds"]
                    >= duration_threshold
                )
            )
        )

    if family == "jump_or_history_deviation":
        jump_risk = (
            np.isfinite(jump)
            & (
                jump
                >= parameters["jump_threshold_bpm"]
            )
        )
        deviation_risk = (
            np.isfinite(deviation)
            & (
                deviation
                >= parameters[
                    "history_deviation_threshold_bpm"
                ]
            )
        )
        return jump_risk | deviation_risk

    raise ValueError(f"未知时间规则：{family}")


def select_rule(
    family: str,
    data: dict[str, np.ndarray],
    training_mask: np.ndarray,
) -> dict[str, Any]:
    """选择满足正确保留约束且捕获错误最多的规则。"""
    evaluated: list[
        tuple[dict[str, float], dict[str, float]]
    ] = []
    feasible: list[
        tuple[dict[str, float], dict[str, float]]
    ] = []

    for parameters in candidate_parameters(
        family,
        data,
        training_mask,
    ):
        reject = apply_rule(family, data, parameters)
        metrics = evaluate_rejection(
            data,
            training_mask,
            reject,
        )
        item = (parameters, metrics)
        evaluated.append(item)

        if (
            metrics["macro_correct_retention_percent"]
            >= MIN_MACRO_CORRECT_RETENTION_PERCENT
            and metrics[
                "minimum_subject_correct_retention_percent"
            ]
            >= MIN_SUBJECT_CORRECT_RETENTION_PERCENT
        ):
            feasible.append(item)

    if feasible:
        parameters, metrics = max(
            feasible,
            key=lambda item: (
                item[1]["macro_error_capture_percent"],
                item[1][
                    "macro_correct_retention_percent"
                ],
                item[1]["retained_coverage_percent"],
            ),
        )
        target_met = True
    else:
        # 约束不可达时优先选择误拒最少的候选，
        # 并明确保留 target_met=False。
        parameters, metrics = max(
            evaluated,
            key=lambda item: (
                item[1][
                    "minimum_subject_correct_retention_percent"
                ],
                item[1][
                    "macro_correct_retention_percent"
                ],
                item[1]["macro_error_capture_percent"],
            ),
        )
        target_met = False

    return {
        "family": family,
        "target_met": target_met,
        "parameters": parameters,
        "metrics": metrics,
    }


def run_conditional_loso(
    data: dict[str, np.ndarray],
    universe: np.ndarray,
) -> list[dict[str, Any]]:
    """在冻结静态门控之后检查时间规则的增量稳定性。"""
    rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        for held_subject in DESIGN_SUBJECTS:
            training_mask = (
                universe
                & (data["subject_id"] != held_subject)
            )
            held_mask = (
                universe
                & (data["subject_id"] == held_subject)
            )
            selection = select_rule(
                family,
                data,
                training_mask,
            )
            reject = apply_rule(
                family,
                data,
                selection["parameters"],
            )
            metrics = evaluate_rejection(
                data,
                held_mask,
                reject,
            )
            rows.append(
                {
                    "family": family,
                    "held_subject": held_subject,
                    "target_met_on_training":
                    selection["target_met"],
                    **{
                        field: selection["parameters"].get(
                            field
                        )
                        for field in PARAMETER_FIELDS
                    },
                    **metrics,
                }
            )

    return rows


def summarize_loso(
    loso_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for family in FAMILIES:
        family_rows = [
            row
            for row in loso_rows
            if row["family"] == family
        ]
        correct = [
            float(
                row[
                    "macro_correct_retention_percent"
                ]
            )
            for row in family_rows
        ]
        capture = [
            float(row["macro_error_capture_percent"])
            for row in family_rows
        ]
        summaries.append(
            {
                "family": family,
                "mean_correct_retention_percent":
                float(np.mean(correct)),
                "minimum_held_correct_retention_percent":
                float(np.min(correct)),
                "mean_error_capture_percent":
                float(np.mean(capture)),
                "subjects_with_nonzero_capture": int(
                    np.sum(np.asarray(capture) > 0.0)
                ),
                "mean_retained_coverage_percent": float(
                    np.mean(
                        [
                            row[
                                "retained_coverage_percent"
                            ]
                            for row in family_rows
                        ]
                    )
                ),
                "mean_remaining_failure_percent": float(
                    np.mean(
                        [
                            row[
                                "remaining_failure_percent"
                            ]
                            for row in family_rows
                        ]
                    )
                ),
            }
        )

    return summaries


def build_feature_statistics(
    data: dict[str, np.ndarray],
    universe: np.ndarray,
) -> list[dict[str, Any]]:
    errors = data["absolute_error_bpm"]
    rows: list[dict[str, Any]] = []

    features = (
        "jump_from_last_usable_bpm",
        "history_median_deviation_bpm",
        "motion_run_seconds",
        "collision_run_seconds",
    )
    groups = (
        ("error_le_10_bpm", errors <= ERROR_THRESHOLD_BPM),
        ("error_gt_10_bpm", errors > ERROR_THRESHOLD_BPM),
    )

    for feature in features:
        values = data[feature]
        for group_name, group_mask in groups:
            selected = (
                universe
                & group_mask
                & np.isfinite(values)
            )
            rows.append(
                {
                    "feature": feature,
                    "group": group_name,
                    "valid_count": int(np.sum(selected)),
                    "median": float(
                        np.median(values[selected])
                    ),
                    "p90": float(
                        np.percentile(
                            values[selected],
                            90.0,
                        )
                    ),
                }
            )

    return rows


def selection_to_row(
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": selection["family"],
        "target_met": selection["target_met"],
        **{
            field: selection["parameters"].get(field)
            for field in PARAMETER_FIELDS
        },
        **selection["metrics"],
    }


def choose_research_decision(
    loso_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = max(
        loso_summaries,
        key=lambda row: (
            row["mean_error_capture_percent"],
            row["mean_correct_retention_percent"],
        ),
    )

    criteria = {
        "mean_correct_retention": (
            candidate["mean_correct_retention_percent"]
            >= MIN_MACRO_CORRECT_RETENTION_PERCENT
        ),
        "minimum_held_correct_retention": (
            candidate[
                "minimum_held_correct_retention_percent"
            ]
            >= MIN_SUBJECT_CORRECT_RETENTION_PERCENT
        ),
        "mean_error_capture": (
            candidate["mean_error_capture_percent"]
            >= MIN_LOSO_MEAN_ERROR_CAPTURE_PERCENT
        ),
        "subjects_with_nonzero_capture": (
            candidate["subjects_with_nonzero_capture"]
            >= MIN_LOSO_SUBJECTS_WITH_CAPTURE
        ),
    }
    recommended = all(criteria.values())

    return {
        "status": (
            "recommend_freeze_for_independent_validation"
            if recommended
            else "do_not_freeze_temporal_v0"
        ),
        "recommended": recommended,
        "candidate_family": candidate["family"],
        "criteria": criteria,
        "candidate_loso_summary": candidate,
        "interpretation": (
            "时间特征满足当前研究准入条件，可进入冻结验证。"
            if recommended
            else (
                "时间特征存在局部增量，但跨人错误捕获不足；"
                "暂不加入冻结算法。"
            )
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def csv_value(value: float | int | bool) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def write_outputs(
    original_rows: list[dict[str, str]],
    data: dict[str, np.ndarray],
    candidate_reject: np.ndarray,
    feature_statistics: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    loso_rows: list[dict[str, Any]],
    loso_summaries: list[dict[str, Any]],
    decision: dict[str, Any],
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_path = (
        OUTPUT_DIR
        / "S1_S10_temporal_features_v0.csv"
    )
    feature_path = OUTPUT_DIR / "feature_statistics.csv"
    family_path = OUTPUT_DIR / "family_comparison.csv"
    loso_path = (
        OUTPUT_DIR
        / "conditional_loso_subject_results.csv"
    )
    loso_summary_path = (
        OUTPUT_DIR
        / "conditional_loso_summary.csv"
    )
    decision_path = (
        OUTPUT_DIR
        / "temporal_consistency_v0_decision.json"
    )

    window_rows: list[dict[str, Any]] = []
    for index, original_row in enumerate(original_rows):
        row: dict[str, Any] = dict(original_row)
        for field in TEMPORAL_OUTPUT_FIELDS[:-1]:
            row[field] = csv_value(data[field][index])
        row["candidate_temporal_reject"] = bool(
            candidate_reject[index]
        )
        window_rows.append(row)

    write_csv(
        window_path,
        window_rows,
        tuple(original_rows[0].keys())
        + TEMPORAL_OUTPUT_FIELDS,
    )
    write_csv(
        feature_path,
        feature_statistics,
        ("feature", "group", "valid_count", "median", "p90"),
    )
    write_csv(
        family_path,
        [selection_to_row(item) for item in selections],
        (
            "family",
            "target_met",
        )
        + PARAMETER_FIELDS
        + METRIC_FIELDS,
    )
    write_csv(
        loso_path,
        loso_rows,
        (
            "family",
            "held_subject",
            "target_met_on_training",
        )
        + PARAMETER_FIELDS
        + METRIC_FIELDS,
    )
    write_csv(
        loso_summary_path,
        loso_summaries,
        (
            "family",
            "mean_correct_retention_percent",
            "minimum_held_correct_retention_percent",
            "mean_error_capture_percent",
            "subjects_with_nonzero_capture",
            "mean_retained_coverage_percent",
            "mean_remaining_failure_percent",
        ),
    )

    payload = {
        "status": decision["status"],
        "dataset": "PPG-DaLiA",
        "input_subjects": "S1-S10",
        "validation_subjects": "S11-S15",
        "validation_data_read": False,
        "conditional_on": "frozen static ACC-aware SQI V1",
        "causal_feature_definition": {
            "history_source": (
                "past ACC-aware usable windows only"
            ),
            "history_lookback_seconds":
            HISTORY_LOOKBACK_SECONDS,
            "minimum_history_points":
            MIN_HISTORY_POINTS,
            "future_windows_used": False,
            "ecg_used_as_feature": False,
            "activity_label_used_as_feature": False,
        },
        "training_selection_constraints": {
            "minimum_macro_correct_retention_percent":
            MIN_MACRO_CORRECT_RETENTION_PERCENT,
            "minimum_subject_correct_retention_percent":
            MIN_SUBJECT_CORRECT_RETENTION_PERCENT,
        },
        "research_admission_criteria": {
            "minimum_loso_mean_error_capture_percent":
            MIN_LOSO_MEAN_ERROR_CAPTURE_PERCENT,
            "minimum_loso_subjects_with_capture":
            MIN_LOSO_SUBJECTS_WITH_CAPTURE,
        },
        "full_design_selections": [
            selection_to_row(item)
            for item in selections
        ],
        "conditional_loso_summary": loso_summaries,
        "decision": decision,
    }
    with decision_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return [
        window_path,
        feature_path,
        family_path,
        loso_path,
        loso_summary_path,
        decision_path,
    ]


def format_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    args = parse_args()
    original_rows, data = load_design_data(INPUT_CSV)
    build_causal_features(data)
    universe = temporal_universe(data)

    feature_statistics = build_feature_statistics(
        data,
        universe,
    )
    selections = [
        select_rule(family, data, universe)
        for family in FAMILIES
    ]
    loso_rows = run_conditional_loso(data, universe)
    loso_summaries = summarize_loso(loso_rows)
    decision = choose_research_decision(loso_summaries)

    candidate_selection = next(
        item
        for item in selections
        if item["family"] == decision["candidate_family"]
    )
    candidate_reject = (
        universe
        & apply_rule(
            candidate_selection["family"],
            data,
            candidate_selection["parameters"],
        )
    )

    error_count = int(
        np.sum(
            universe
            & (
                data["absolute_error_bpm"]
                > ERROR_THRESHOLD_BPM
            )
        )
    )
    print("时间连续性 V0：S1-S10 设计阶段")
    print(f"输入：{INPUT_CSV}")
    print("数据边界：只读取 S1-S10，不读取 S11-S15")
    print(
        f"静态门控保留窗口={int(np.sum(universe))}，"
        f"其中误差 >10 bpm={error_count}"
    )

    print("\n时间特征正确组/错误组统计")
    for row in feature_statistics:
        print(
            f"  {row['feature']} / {row['group']}: "
            f"n={row['valid_count']}, "
            f"median={row['median']:.3f}, "
            f"P90={row['p90']:.3f}"
        )

    print("\nS1-S10 完整设计集消融")
    for selection in selections:
        parameters = ", ".join(
            f"{field}={format_optional(selection['parameters'].get(field))}"
            for field in PARAMETER_FIELDS
            if selection["parameters"].get(field) is not None
        )
        metrics = selection["metrics"]
        print(
            f"  {selection['family']}: "
            f"target_met={selection['target_met']}, "
            f"{parameters}"
        )
        print(
            "    correct retention="
            f"{metrics['macro_correct_retention_percent']:.2f}%, "
            "error capture="
            f"{metrics['macro_error_capture_percent']:.2f}%, "
            "coverage="
            f"{metrics['retained_coverage_percent']:.2f}%, "
            "remaining >10 bpm="
            f"{metrics['remaining_failure_percent']:.2f}%"
        )

    print("\n条件式 LOSO 汇总")
    for row in loso_summaries:
        print(
            f"  {row['family']}: "
            "mean correct retention="
            f"{row['mean_correct_retention_percent']:.2f}%, "
            "minimum held correct retention="
            f"{row['minimum_held_correct_retention_percent']:.2f}%, "
            "mean error capture="
            f"{row['mean_error_capture_percent']:.2f}%, "
            "subjects with capture="
            f"{row['subjects_with_nonzero_capture']}/10"
        )

    print("\n研究决策")
    print(f"  status={decision['status']}")
    print(
        f"  candidate={decision['candidate_family']}"
    )
    print(f"  {decision['interpretation']}")
    print(
        "  注意：这是静态 ACC-aware V1 之后的"
        "条件式时间增量检查，不是独立验证。"
    )

    if args.write_outputs:
        paths = write_outputs(
            original_rows,
            data,
            candidate_reject,
            feature_statistics,
            selections,
            loso_rows,
            loso_summaries,
            decision,
        )
        print("\n已保存时间连续性 V0 设计证据：")
        for path in paths:
            print(f"  {path}")
    else:
        print(
            "\n本次未写入结果；检查合理后使用 "
            "--write-outputs 保存。"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
