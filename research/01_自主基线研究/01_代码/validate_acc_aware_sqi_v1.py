#!/usr/bin/env python3
"""一次性验证冻结的 ACC-aware SQI V1。

默认模式只检查冻结参数及 S1-S10 设计结果能否复现，不读取
S11-S15 原始 ACC。

只有显式传入 --run-validation 时，才使用冻结门限处理 S11-S15。
验证过程中不重新选择特征、不计算分位数，也不调整任何门限。

范围说明：
S11-S15 没有参与 ACC 特征和门限设计，但曾参与 PPG-only SQI V0
验证。因此本结果属于“ACC-aware 增量留出验证”，不是整个
PPG 心率系统从头到尾的完全独立测试。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_acc_motion_features_v0 import (
    ACC_STEP_SAMPLES,
    ACC_WINDOW_SAMPLES,
    calculate_acc_features,
    load_subject_acc,
    validate_subject_alignment,
)
from evaluate_acc_aware_sqi_v1 import (
    METRIC_FIELDS,
    assign_acc_states,
    build_partition_rows,
    safe_percent,
    summarize_mask,
    write_csv,
    write_window_table,
)


DATA_ROOT = Path(__file__).resolve().parents[1]
PPG_SQI_CSV = (
    DATA_ROOT
    / "02_结果"
    / "ppg_only_sqi_v0"
    / "S1_S15_ppg_only_sqi_v0_windows.csv"
)
DESIGN_WINDOWS_CSV = (
    DATA_ROOT
    / "02_结果"
    / "acc_aware_sqi_v1_design"
    / "S1_S10_acc_aware_sqi_v1_windows.csv"
)
FROZEN_DESIGN_JSON = (
    DATA_ROOT
    / "02_结果"
    / "acc_aware_sqi_v1_design"
    / "acc_aware_sqi_v1_frozen_design.json"
)
OUTPUT_DIR = (
    DATA_ROOT
    / "02_结果"
    / "acc_aware_sqi_v1_validation"
)

DESIGN_SUBJECTS = tuple(range(1, 11))
VALIDATION_SUBJECTS = tuple(range(11, 16))
EXPECTED_DESIGN_WINDOWS = 43_215
EXPECTED_VALIDATION_WINDOWS = 21_482
ERROR_THRESHOLD_BPM = 10.0

# 冻结参数文件必须保持逐字节不变。
EXPECTED_FROZEN_SHA256 = (
    "ff3b3aba79e0de9c8931fdd476ac3b6628ebc3e144dc82f572277bd08ed9477e"
)

# 在读取 S11-S15 前预先定义通过标准，避免看到结果后再改口径。
MIN_OVERALL_CORRECT_RETENTION_PERCENT = 90.0
MIN_OVERALL_ERROR_CAPTURE_PERCENT = 70.0
MAX_RETAINED_FAILURE_PERCENT = 5.0
MIN_SUBJECT_CORRECT_RETENTION_PERCENT = 85.0
MIN_PASSING_SUBJECTS = 4

FEATURE_FIELDS = (
    "motion_rms_g",
    "jerk_rms_g_per_s",
    "acc_power_near_ppg_ratio",
    "ppg_acc_harmonic_gap_hz",
)

NUMERIC_FIELDS = (
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "ppg_hr_bpm",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "sqi_score",
) + FEATURE_FIELDS

DESIGN_REQUIRED_COLUMNS = {
    "subject_id",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "activity_id",
    "activity_name",
    "ppg_hr_bpm",
    "ecg_hr_bpm",
    "absolute_error_bpm",
    "sqi_score",
    "ppg_sqi_state",
    "ppg_sqi_reason",
    *FEATURE_FIELDS,
    "motion_flag",
    "collision_flag",
    "acc_aware_state",
    "acc_aware_reason",
}

VALIDATION_REQUIRED_COLUMNS = {
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
    "state",
    "reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证冻结的 ACC-aware SQI V1"
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="显式读取 S11-S15 并执行一次性增量留出验证",
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="将验证结果写入独立输出目录",
    )
    return parser.parse_args()


def parse_subject_id(raw_value: Any) -> int:
    return int(str(raw_value).removeprefix("S"))


def parse_bool(raw_value: Any) -> bool:
    normalized = str(raw_value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"无法解析布尔值：{raw_value}")


def load_frozen_design(
) -> tuple[dict[str, Any], dict[str, float], str]:
    """校验并读取不可变的冻结参数。"""
    if not FROZEN_DESIGN_JSON.is_file():
        raise FileNotFoundError(
            f"找不到冻结参数：{FROZEN_DESIGN_JSON}"
        )

    raw_bytes = FROZEN_DESIGN_JSON.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != EXPECTED_FROZEN_SHA256:
        raise ValueError(
            "冻结参数文件 SHA-256 已变化；"
            "为避免验证污染，本次停止。"
        )

    document = json.loads(raw_bytes.decode("utf-8"))
    if document.get("status") != "design_frozen_not_validated":
        raise ValueError("冻结参数状态异常")
    if document.get("input_subjects") != "S1-S10":
        raise ValueError("冻结参数的设计受试者范围异常")
    if document.get("validation_subjects") != "S11-S15":
        raise ValueError("冻结参数的验证受试者范围异常")
    if document.get("validation_data_read") is not False:
        raise ValueError("冻结参数记录显示验证数据已被读取")

    final_selection = document.get("final_selection", {})
    if final_selection.get("family") != "motion_and_collision":
        raise ValueError("冻结规则族不是 motion_and_collision")

    raw_parameters = final_selection.get("parameters", {})
    parameter_names = (
        "motion_rms_threshold_g",
        "jerk_rms_threshold_g_per_s",
        "acc_power_near_ppg_threshold",
        "harmonic_gap_threshold_hz",
    )

    try:
        parameters = {
            name: float(raw_parameters[name])
            for name in parameter_names
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("冻结参数缺失或格式错误") from error

    if not all(np.isfinite(value) for value in parameters.values()):
        raise ValueError("冻结参数中存在 NaN 或无穷值")

    return document, parameters, digest


def read_csv_rows(
    csv_path: Path,
    required_columns: set[str],
) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{csv_path}")

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = required_columns - columns
        if missing:
            raise ValueError(
                f"{csv_path.name} 缺少字段：{sorted(missing)}"
            )
        return list(reader)


def rows_to_data(
    rows: list[dict[str, Any]],
    state_field: str,
    reason_field: str,
) -> dict[str, np.ndarray]:
    """转换为现有 ACC-aware 规则所使用的数组结构。"""
    data: dict[str, np.ndarray] = {
        "subject_id": np.asarray(
            [parse_subject_id(row["subject_id"]) for row in rows],
            dtype=np.int64,
        ),
        "activity_id": np.asarray(
            [int(float(row["activity_id"])) for row in rows],
            dtype=np.int64,
        ),
        "activity_name": np.asarray(
            [row["activity_name"] for row in rows],
            dtype=object,
        ),
        "state": np.asarray(
            [row[state_field] for row in rows],
            dtype=object,
        ),
        "reason": np.asarray(
            [row[reason_field] for row in rows],
            dtype=object,
        ),
    }

    for field in NUMERIC_FIELDS:
        data[field] = np.asarray(
            [float(row[field]) for row in rows],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(data[field])):
            raise ValueError(
                f"字段 {field} 中存在 NaN 或无穷值"
            )

    return data


def summarize_outputs(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    total_windows: int,
    total_coverage_field: str,
) -> dict[str, dict[str, float | int | None]]:
    baseline_usable = data["state"] == "usable"
    errors = data["absolute_error_bpm"]

    output_masks = {
        "ppg_only_usable": baseline_usable,
        "acc_aware_usable": states == "usable",
        "acc_aware_usable_plus_low": np.isin(
            states,
            ("usable", "low_confidence"),
        ),
        "acc_aware_reject": states == "reject",
    }

    metrics = {
        name: summarize_mask(
            errors,
            baseline_usable,
            mask,
        )
        for name, mask in output_masks.items()
    }

    for row in metrics.values():
        row[total_coverage_field] = safe_percent(
            int(row["count"]),
            total_windows,
        )

    return metrics


def compare_design_metrics(
    actual: dict[str, dict[str, float | int | None]],
    expected: dict[str, Any],
) -> None:
    for output_range, expected_row in expected.items():
        if output_range not in actual:
            raise ValueError(
                f"设计复现缺少输出范围：{output_range}"
            )

        for field, expected_value in expected_row.items():
            actual_value = actual[output_range].get(field)
            if expected_value is None:
                if actual_value is not None:
                    raise ValueError(
                        f"{output_range}.{field} 复现不一致"
                    )
                continue

            if actual_value is None or not np.isclose(
                float(actual_value),
                float(expected_value),
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{output_range}.{field} 复现不一致："
                    f"当前={actual_value}，冻结={expected_value}"
                )


def verify_design_reproduction(
    frozen_design: dict[str, Any],
    parameters: dict[str, float],
) -> None:
    """在不读取验证集的情况下复现冻结规则。"""
    rows = read_csv_rows(
        DESIGN_WINDOWS_CSV,
        DESIGN_REQUIRED_COLUMNS,
    )
    if len(rows) != EXPECTED_DESIGN_WINDOWS:
        raise ValueError(
            f"设计窗口数异常：{len(rows)}"
        )

    subjects = {
        parse_subject_id(row["subject_id"])
        for row in rows
    }
    if subjects != set(DESIGN_SUBJECTS):
        raise ValueError(
            f"设计受试者异常：{sorted(subjects)}"
        )

    data = rows_to_data(
        rows,
        state_field="ppg_sqi_state",
        reason_field="ppg_sqi_reason",
    )
    states, reasons, motion, collision = assign_acc_states(
        data,
        parameters,
    )

    expected_motion = np.asarray(
        [parse_bool(row["motion_flag"]) for row in rows],
        dtype=bool,
    )
    expected_collision = np.asarray(
        [parse_bool(row["collision_flag"]) for row in rows],
        dtype=bool,
    )
    expected_states = np.asarray(
        [row["acc_aware_state"] for row in rows],
        dtype=object,
    )
    expected_reasons = np.asarray(
        [row["acc_aware_reason"] for row in rows],
        dtype=object,
    )

    comparisons = (
        ("motion_flag", motion, expected_motion),
        ("collision_flag", collision, expected_collision),
        ("acc_aware_state", states, expected_states),
        ("acc_aware_reason", reasons, expected_reasons),
    )
    for name, actual, expected in comparisons:
        if not np.array_equal(actual, expected):
            mismatch_count = int(np.sum(actual != expected))
            raise ValueError(
                f"{name} 设计复现失败："
                f"{mismatch_count} 个窗口不一致"
            )

    metrics = summarize_outputs(
        data,
        states,
        EXPECTED_DESIGN_WINDOWS,
        "coverage_of_all_design_percent",
    )
    compare_design_metrics(
        metrics,
        frozen_design["design_metrics"],
    )


def load_validation_rows() -> list[dict[str, Any]]:
    """仅在显式验证模式下提取 S11-S15 窗口。"""
    raw_rows = read_csv_rows(
        PPG_SQI_CSV,
        VALIDATION_REQUIRED_COLUMNS,
    )
    rows: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        subject_id = parse_subject_id(raw_row["subject_id"])
        if subject_id not in VALIDATION_SUBJECTS:
            continue
        if raw_row["split"] != "validation":
            raise ValueError(
                f"S{subject_id} 的 split 不是 validation"
            )

        rows.append(
            {
                "subject_id": subject_id,
                "split": raw_row["split"],
                "window_start_s": float(
                    raw_row["window_start_s"]
                ),
                "window_center_s": float(
                    raw_row["window_center_s"]
                ),
                "window_end_s": float(raw_row["window_end_s"]),
                "activity_id": int(
                    float(raw_row["activity_id"])
                ),
                "activity_name": raw_row["activity_name"],
                "ppg_hr_bpm": float(raw_row["ppg_hr_bpm"]),
                "ecg_hr_bpm": float(raw_row["ecg_hr_bpm"]),
                "absolute_error_bpm": float(
                    raw_row["absolute_error_bpm"]
                ),
                "sqi_score": float(raw_row["sqi_score"]),
                "state": raw_row["state"],
                "reason": raw_row["reason"],
            }
        )

    if len(rows) != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError(
            "验证窗口数异常："
            f"期望 {EXPECTED_VALIDATION_WINDOWS}，"
            f"实际 {len(rows)}"
        )

    rows.sort(
        key=lambda row: (
            row["subject_id"],
            row["window_start_s"],
        )
    )
    return rows


def build_validation_data(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """用既有公式计算 S11-S15 ACC 特征，不拟合门限。"""
    records: list[dict[str, Any]] = []

    for subject_id in VALIDATION_SUBJECTS:
        subject_rows = [
            row
            for row in rows
            if row["subject_id"] == subject_id
        ]
        acc = load_subject_acc(subject_id)
        validate_subject_alignment(
            subject_id,
            subject_rows,
            acc,
        )

        for window_index, row in enumerate(subject_rows):
            start = window_index * ACC_STEP_SAMPLES
            end = start + ACC_WINDOW_SAMPLES
            ppg_frequency_hz = row["ppg_hr_bpm"] / 60.0

            features = calculate_acc_features(
                acc[start:end],
                ppg_frequency_hz,
            )
            record = dict(row)
            record.update(features)
            records.append(record)

        print(
            f"  S{subject_id}: "
            f"ACC 对齐与特征计算通过，"
            f"窗口数={len(subject_rows)}"
        )

    if len(records) != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError("验证特征窗口数异常")

    return rows_to_data(
        records,
        state_field="state",
        reason_field="reason",
    )


def evaluate_acceptance(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    metrics: dict[str, dict[str, float | int | None]],
) -> tuple[str, dict[str, Any], dict[str, bool]]:
    """按验证前声明的标准生成通过或失败结论。"""
    baseline_usable = data["state"] == "usable"
    errors = data["absolute_error_bpm"]
    rejected = states == "reject"

    correct = baseline_usable & (errors <= ERROR_THRESHOLD_BPM)
    failures = baseline_usable & (errors > ERROR_THRESHOLD_BPM)

    overall_correct_retention = safe_percent(
        int(np.sum(correct & ~rejected)),
        int(np.sum(correct)),
    )
    overall_error_capture = safe_percent(
        int(np.sum(failures & rejected)),
        int(np.sum(failures)),
    )

    subject_rows: list[dict[str, Any]] = []
    for subject_id in VALIDATION_SUBJECTS:
        subject_correct = (
            correct
            & (data["subject_id"] == subject_id)
        )
        retention = safe_percent(
            int(np.sum(subject_correct & ~rejected)),
            int(np.sum(subject_correct)),
        )
        subject_rows.append(
            {
                "subject_id": subject_id,
                "correct_window_count": int(
                    np.sum(subject_correct)
                ),
                "correct_retention_percent": retention,
                "passes_85_percent": (
                    retention
                    >= MIN_SUBJECT_CORRECT_RETENTION_PERCENT
                ),
            }
        )

    passing_subjects = sum(
        row["passes_85_percent"]
        for row in subject_rows
    )
    retained_failure = metrics[
        "acc_aware_usable_plus_low"
    ]["failure_gt_10_bpm_percent"]
    if retained_failure is None:
        raise ValueError("保留窗口的失败率无法计算")

    checks = {
        "overall_correct_retention": (
            overall_correct_retention
            >= MIN_OVERALL_CORRECT_RETENTION_PERCENT
        ),
        "overall_error_capture": (
            overall_error_capture
            >= MIN_OVERALL_ERROR_CAPTURE_PERCENT
        ),
        "retained_failure_rate": (
            retained_failure
            <= MAX_RETAINED_FAILURE_PERCENT
        ),
        "cross_subject_stability": (
            passing_subjects >= MIN_PASSING_SUBJECTS
        ),
    }

    diagnostics = {
        "overall_correct_retention_percent": (
            overall_correct_retention
        ),
        "overall_error_capture_percent": overall_error_capture,
        "retained_failure_gt_10_bpm_percent": retained_failure,
        "subjects_passing_retention_requirement": (
            passing_subjects
        ),
        "subject_correct_retention": subject_rows,
    }
    status = (
        "incremental_holdout_pass"
        if all(checks.values())
        else "incremental_holdout_fail"
    )
    return status, diagnostics, checks


def build_reason_rows(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    reasons: np.ndarray,
) -> list[dict[str, Any]]:
    baseline_usable = data["state"] == "usable"
    pairs = Counter(
        zip(
            states[baseline_usable].tolist(),
            reasons[baseline_usable].tolist(),
        )
    )
    total = int(np.sum(baseline_usable))

    return [
        {
            "acc_aware_state": state,
            "acc_aware_reason": reason,
            "count": count,
            "percent_of_ppg_only_usable": safe_percent(
                count,
                total,
            ),
        }
        for (state, reason), count in sorted(pairs.items())
    ]


def write_validation_outputs(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    reasons: np.ndarray,
    motion: np.ndarray,
    collision: np.ndarray,
    metrics: dict[str, dict[str, float | int | None]],
    result_document: dict[str, Any],
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_path = (
        OUTPUT_DIR
        / "S11_S15_acc_aware_sqi_v1_validation_windows.csv"
    )
    metrics_path = OUTPUT_DIR / "validation_metrics.csv"
    subject_path = OUTPUT_DIR / "subject_metrics.csv"
    activity_path = OUTPUT_DIR / "activity_metrics.csv"
    reason_path = OUTPUT_DIR / "state_reason_counts.csv"
    result_path = (
        OUTPUT_DIR
        / "acc_aware_sqi_v1_validation_result.json"
    )

    write_window_table(
        window_path,
        data,
        states,
        reasons,
        motion,
        collision,
    )
    write_csv(
        metrics_path,
        [
            {"output_range": name, **row}
            for name, row in metrics.items()
        ],
        (
            "output_range",
            *METRIC_FIELDS,
            "coverage_of_all_validation_percent",
        ),
    )

    partition_fields = (
        "group_id",
        "group_name",
        "output_range",
        *METRIC_FIELDS,
    )
    write_csv(
        subject_path,
        build_partition_rows(data, states, "subject_id"),
        partition_fields,
    )
    write_csv(
        activity_path,
        build_partition_rows(data, states, "activity_id"),
        partition_fields,
    )
    write_csv(
        reason_path,
        build_reason_rows(data, states, reasons),
        (
            "acc_aware_state",
            "acc_aware_reason",
            "count",
            "percent_of_ppg_only_usable",
        ),
    )

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            result_document,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return [
        window_path,
        metrics_path,
        subject_path,
        activity_path,
        reason_path,
        result_path,
    ]


def print_metrics(
    metrics: dict[str, dict[str, float | int | None]],
) -> None:
    for name, row in metrics.items():
        print(
            f"  {name}: "
            f"n={row['count']}，"
            f"相对覆盖率={row['coverage_percent']:.2f}% ，"
            f"全部验证窗覆盖率="
            f"{row['coverage_of_all_validation_percent']:.2f}% ，"
            f"MAE={row['mae_bpm']:.2f} bpm，"
            f"P90={row['p90_bpm']:.2f} bpm，"
            f">10 bpm="
            f"{row['failure_gt_10_bpm_percent']:.2f}%"
        )


def run_validation(
    frozen_design: dict[str, Any],
    parameters: dict[str, float],
    frozen_hash: str,
    should_write_outputs: bool,
) -> None:
    print("\n开始读取 S11-S15：门限保持冻结，不重新拟合")
    rows = load_validation_rows()
    data = build_validation_data(rows)
    states, reasons, motion, collision = assign_acc_states(
        data,
        parameters,
    )

    metrics = summarize_outputs(
        data,
        states,
        EXPECTED_VALIDATION_WINDOWS,
        "coverage_of_all_validation_percent",
    )
    status, diagnostics, checks = evaluate_acceptance(
        data,
        states,
        metrics,
    )

    criteria = {
        "minimum_overall_correct_retention_percent": (
            MIN_OVERALL_CORRECT_RETENTION_PERCENT
        ),
        "minimum_overall_error_capture_percent": (
            MIN_OVERALL_ERROR_CAPTURE_PERCENT
        ),
        "maximum_retained_failure_gt_10_bpm_percent": (
            MAX_RETAINED_FAILURE_PERCENT
        ),
        "minimum_subject_correct_retention_percent": (
            MIN_SUBJECT_CORRECT_RETENTION_PERCENT
        ),
        "minimum_passing_subjects": MIN_PASSING_SUBJECTS,
    }
    result_document = {
        "status": status,
        "dataset": "PPG-DaLiA",
        "validation_scope": "ACC-aware incremental holdout",
        "validation_subjects": "S11-S15",
        "validation_data_read": True,
        "threshold_refit": False,
        "frozen_design_sha256": frozen_hash,
        "frozen_parameters": parameters,
        "acceptance_criteria": criteria,
        "acceptance_checks": checks,
        "diagnostics": diagnostics,
        "validation_metrics": metrics,
        "scope_limit": (
            "S11-S15 未参与 ACC 特征和门限设计，"
            "但曾用于 PPG-only SQI V0 验证；"
            "因此不是端到端完全独立测试。"
        ),
        "source_design_status": frozen_design["status"],
    }

    print("\nS11-S15 验证结果")
    print_metrics(metrics)
    print("\n预声明验收标准")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n总判定：{status}")

    if should_write_outputs:
        output_paths = write_validation_outputs(
            data,
            states,
            reasons,
            motion,
            collision,
            metrics,
            result_document,
        )
        print("\n已保存验证结果：")
        for output_path in output_paths:
            print(f"  {output_path}")
    else:
        print("\n本次没有写入验证结果。")


def main(
    should_run_validation: bool = False,
    should_write_outputs: bool = False,
) -> int:
    if should_write_outputs and not should_run_validation:
        raise ValueError(
            "--write-outputs 必须与 --run-validation 同时使用"
        )

    frozen_design, parameters, frozen_hash = (
        load_frozen_design()
    )
    verify_design_reproduction(
        frozen_design,
        parameters,
    )

    print("ACC-aware SQI V1 冻结设计复现：PASS")
    print(f"冻结参数 SHA-256：{frozen_hash}")

    if not should_run_validation:
        print(
            "S11-S15 原始 ACC 未读取。"
            "确认后使用 --run-validation --write-outputs "
            "执行一次性验证。"
        )
        return 0

    run_validation(
        frozen_design,
        parameters,
        frozen_hash,
        should_write_outputs,
    )
    return 0


if __name__ == "__main__":
    try:
        arguments = parse_args()
        raise SystemExit(
            main(
                arguments.run_validation,
                arguments.write_outputs,
            )
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
