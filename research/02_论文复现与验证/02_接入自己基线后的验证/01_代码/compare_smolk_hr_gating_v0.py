#!/usr/bin/env python3
"""在相同 8 秒窗口和保留数量下比较 SMoLK 与冻结 SQI+ACC。"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

DATA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DATA_ROOT.parents[1]
# 复用第一阶段的冻结基线，避免复制算法实现。
sys.path.insert(0, str(PROJECT_ROOT / "01_自主基线研究/01_代码"))

from evaluate_acc_aware_sqi_v1 import assign_acc_states
from validate_acc_aware_sqi_v1 import (
    DESIGN_REQUIRED_COLUMNS,
    DESIGN_WINDOWS_CSV,
    build_validation_data,
    load_frozen_design,
    load_validation_rows,
    read_csv_rows,
    rows_to_data,
    verify_design_reproduction,
)


SMOLK_MAPPING_CSV = (
    PROJECT_ROOT
    / "02_论文复现与验证/01_SMoLK论文复现/outputs/dalia_mapping/dalia_hr_window_mapping.csv"
)
OUTPUT_DIR = DATA_ROOT / "02_结果/smolk_hr_comparison_v1"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
WINDOWS_CSV = OUTPUT_DIR / "aligned_windows.csv"
METHOD_CSV = OUTPUT_DIR / "method_metrics.csv"
SUBJECT_CSV = OUTPUT_DIR / "subject_metrics.csv"
FIGURE_PATH = OUTPUT_DIR / "comparison.png"

PRIMARY_SUBJECTS = (14, 15)
CONTEXT_SUBJECTS = (1, 14, 15)
ERROR_THRESHOLD_BPM = 10.0
EXPECTED_ALIGNED_COUNTS = {1: 3684, 14: 3576, 15: 3168}

WINDOW_FIELDS = (
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
    "acc_aware_state",
    "smolk_probability_mean",
    "smolk_artifact_fraction_savgol_151",
    "primary_analysis",
    "acc_aware_keep",
    "smolk_matched_keep",
)
METRIC_FIELDS = (
    "scope",
    "method",
    "count",
    "coverage_percent",
    "mae_bpm",
    "rmse_bpm",
    "median_absolute_error_bpm",
    "p90_absolute_error_bpm",
    "within_3_bpm_percent",
    "failure_gt_10_bpm_percent",
)
SUBJECT_FIELDS = (
    "subject_id",
    "method",
    "count",
    "coverage_percent",
    "mae_bpm",
    "failure_gt_10_bpm_percent",
)


def records_from_data(
    data: dict[str, np.ndarray],
    acc_states: np.ndarray,
    subjects: tuple[int, ...],
) -> list[dict[str, Any]]:
    selected = np.isin(data["subject_id"], subjects)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(selected):
        rows.append(
            {
                "subject_id": int(data["subject_id"][index]),
                "window_start_s": float(data["window_start_s"][index]),
                "window_center_s": float(data["window_center_s"][index]),
                "window_end_s": float(data["window_end_s"][index]),
                "activity_id": int(data["activity_id"][index]),
                "activity_name": str(data["activity_name"][index]),
                "ppg_hr_bpm": float(data["ppg_hr_bpm"][index]),
                "ecg_hr_bpm": float(data["ecg_hr_bpm"][index]),
                "absolute_error_bpm": float(data["absolute_error_bpm"][index]),
                "sqi_score": float(data["sqi_score"][index]),
                "ppg_sqi_state": str(data["state"][index]),
                "acc_aware_state": str(acc_states[index]),
            }
        )
    return rows


def load_frozen_baseline() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_design, parameters, frozen_hash = load_frozen_design()
    verify_design_reproduction(frozen_design, parameters)

    design_rows = read_csv_rows(DESIGN_WINDOWS_CSV, DESIGN_REQUIRED_COLUMNS)
    design_data = rows_to_data(
        design_rows,
        state_field="ppg_sqi_state",
        reason_field="ppg_sqi_reason",
    )
    design_states, _, _, _ = assign_acc_states(design_data, parameters)

    validation_rows = load_validation_rows()
    validation_data = build_validation_data(validation_rows)
    validation_states, _, _, _ = assign_acc_states(validation_data, parameters)

    rows = records_from_data(design_data, design_states, (1,))
    rows.extend(
        records_from_data(validation_data, validation_states, (14, 15))
    )
    return rows, {
        "frozen_design_sha256": frozen_hash,
        "frozen_parameters": parameters,
        "s1_role": "ACC-aware design subject; context only",
        "s14_s15_role": "ACC-aware frozen validation subjects; primary comparison",
    }


def load_aligned_rows(
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_key: dict[tuple[int, float], dict[str, Any]] = {}
    for row in baseline_rows:
        key = (int(row["subject_id"]), float(row["window_start_s"]))
        if key in baseline_by_key:
            raise ValueError(f"Duplicate baseline key: {key}")
        baseline_by_key[key] = row

    with SMOLK_MAPPING_CSV.open("r", encoding="utf-8", newline="") as stream:
        mapping_rows = list(csv.DictReader(stream))
    aligned: list[dict[str, Any]] = []
    for mapping in mapping_rows:
        subject_id = int(mapping["subject_id"])
        if subject_id not in CONTEXT_SUBJECTS or mapping["mapping_status"] != "contained":
            continue
        key = (subject_id, float(mapping["window_start_s"]))
        if key not in baseline_by_key:
            raise ValueError(f"Missing frozen baseline window: {key}")
        row = dict(baseline_by_key[key])
        row.update(
            {
                "smolk_probability_mean": float(mapping["smolk_probability_mean"]),
                "smolk_artifact_fraction_savgol_151": float(
                    mapping["smolk_artifact_fraction_savgol_151"]
                ),
            }
        )
        aligned.append(row)

    counts = {
        subject_id: sum(row["subject_id"] == subject_id for row in aligned)
        for subject_id in CONTEXT_SUBJECTS
    }
    if counts != EXPECTED_ALIGNED_COUNTS:
        raise ValueError(f"Unexpected aligned counts: {counts}")
    if len(aligned) != len({
        (row["subject_id"], row["window_start_s"]) for row in aligned
    }):
        raise ValueError("Aligned window keys are not unique")
    return aligned


def summarize(
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    denominator: int,
    scope: str,
    method: str,
) -> dict[str, Any]:
    errors = np.asarray(
        [row["absolute_error_bpm"] for row in rows],
        dtype=np.float64,
    )[mask]
    if len(errors) == 0:
        raise ValueError(f"Empty selection: {scope}/{method}")
    return {
        "scope": scope,
        "method": method,
        "count": int(len(errors)),
        "coverage_percent": float(100.0 * len(errors) / denominator),
        "mae_bpm": float(np.mean(errors)),
        "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))),
        "median_absolute_error_bpm": float(np.median(errors)),
        "p90_absolute_error_bpm": float(np.percentile(errors, 90)),
        "within_3_bpm_percent": float(100.0 * np.mean(errors <= 3.0)),
        "failure_gt_10_bpm_percent": float(
            100.0 * np.mean(errors > ERROR_THRESHOLD_BPM)
        ),
    }


def select_lowest_scores(
    rows: list[dict[str, Any]],
    eligible: np.ndarray,
    count: int,
) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    if count <= 0 or count > len(indices):
        raise ValueError(f"Invalid matched count {count}/{len(indices)}")
    ordered = sorted(
        indices,
        key=lambda index: (
            rows[index]["smolk_probability_mean"],
            rows[index]["subject_id"],
            rows[index]["window_start_s"],
        ),
    )
    selected = np.zeros(len(rows), dtype=bool)
    selected[ordered[:count]] = True
    return selected


def state_risk(states: np.ndarray) -> np.ndarray:
    order = {"usable": 0.0, "low_confidence": 1.0, "reject": 2.0}
    try:
        return np.asarray([order[str(state)] for state in states])
    except KeyError as error:
        raise ValueError(f"Unexpected SQI state: {error}") from error


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


def save_figure(method_rows: list[dict[str, Any]]) -> None:
    selected = {
        row["method"]: row
        for row in method_rows
        if row["scope"] == "primary_S14_S15"
        and row["method"] in {
            "acc_aware_usable_plus_low",
            "smolk_matched_acc_count",
        }
    }
    names = ["SQI+ACC", "SMoLK matched"]
    colors = ["#176b87", "#c86428"]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), constrained_layout=True)
    for axis, field, title, unit in (
        (axes[0], "mae_bpm", "Retained-window MAE", "bpm"),
        (
            axes[1],
            "failure_gt_10_bpm_percent",
            "Retained >10 bpm failures",
            "%",
        ),
    ):
        values = [
            selected["acc_aware_usable_plus_low"][field],
            selected["smolk_matched_acc_count"][field],
        ]
        bars = axis.bar(names, values, color=colors)
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
            )
    coverage = selected["acc_aware_usable_plus_low"]["coverage_percent"]
    figure.suptitle(
        f"S14-S15, identical aligned windows and retained count "
        f"(coverage {coverage:.2f}%)"
    )
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    outputs = (SUMMARY_JSON, WINDOWS_CSV, METHOD_CSV, SUBJECT_CSV, FIGURE_PATH)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Comparison output exists; preserve it before rerunning")
    if not SMOLK_MAPPING_CSV.is_file():
        raise FileNotFoundError(f"Missing SMoLK mapping: {SMOLK_MAPPING_CSV}")

    baseline_rows, baseline_provenance = load_frozen_baseline()
    rows = load_aligned_rows(baseline_rows)
    count = len(rows)
    subject_ids = np.asarray([row["subject_id"] for row in rows])
    primary = np.isin(subject_ids, PRIMARY_SUBJECTS)
    primary_count = int(np.sum(primary))
    ppg_states = np.asarray([row["ppg_sqi_state"] for row in rows])
    acc_states = np.asarray([row["acc_aware_state"] for row in rows])
    errors = np.asarray([row["absolute_error_bpm"] for row in rows])
    smolk_scores = np.asarray([row["smolk_probability_mean"] for row in rows])

    baseline_usable = ppg_states == "usable"
    ppg_keep = primary & baseline_usable
    acc_keep = (
        primary
        & baseline_usable
        & np.isin(acc_states, ("usable", "low_confidence"))
    )
    smolk_keep = select_lowest_scores(rows, primary, int(np.sum(acc_keep)))
    native_clean = primary & (
        np.asarray([
            row["smolk_artifact_fraction_savgol_151"] for row in rows
        ]) == 0.0
    )

    masks = {
        "all_aligned": primary,
        "ppg_only_usable": ppg_keep,
        "acc_aware_usable_plus_low": acc_keep,
        "smolk_matched_acc_count": smolk_keep,
        "smolk_151_no_predicted_artifact": native_clean,
    }
    method_rows = [
        summarize(
            rows,
            mask,
            primary_count,
            "primary_S14_S15",
            method,
        )
        for method, mask in masks.items()
    ]

    subject_rows: list[dict[str, Any]] = []
    for subject_id in PRIMARY_SUBJECTS:
        subject_mask = subject_ids == subject_id
        subject_acc = (
            subject_mask
            & baseline_usable
            & np.isin(
                acc_states, ("usable", "low_confidence")
            )
        )
        subject_smolk = select_lowest_scores(
            rows, subject_mask, int(np.sum(subject_acc))
        )
        for method, mask in (
            ("acc_aware_usable_plus_low", subject_acc),
            ("smolk_matched_within_subject", subject_smolk),
        ):
            result = summarize(
                rows,
                mask,
                int(np.sum(subject_mask)),
                f"S{subject_id}",
                method,
            )
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "method": method,
                    **{
                        field: result[field]
                        for field in SUBJECT_FIELDS[2:]
                    },
                }
            )

    failure = errors[primary] > ERROR_THRESHOLD_BPM
    if len(np.unique(failure)) != 2:
        raise ValueError("Primary scope must contain successes and >10 bpm failures")
    auc = {
        "smolk_probability_mean": float(
            roc_auc_score(failure, smolk_scores[primary])
        ),
        "negative_ppg_sqi_score": float(
            roc_auc_score(
                failure,
                -np.asarray([row["sqi_score"] for row in rows])[primary],
            )
        ),
        "ppg_sqi_state_risk": float(
            roc_auc_score(failure, state_risk(ppg_states[primary]))
        ),
        "acc_aware_state_risk": float(
            roc_auc_score(failure, state_risk(acc_states[primary]))
        ),
    }

    method_by_name = {row["method"]: row for row in method_rows}
    acc_result = method_by_name["acc_aware_usable_plus_low"]
    smolk_result = method_by_name["smolk_matched_acc_count"]
    failure_delta = (
        smolk_result["failure_gt_10_bpm_percent"]
        - acc_result["failure_gt_10_bpm_percent"]
    )
    mae_delta = smolk_result["mae_bpm"] - acc_result["mae_bpm"]
    summary = {
        "status": "completed_retrospective_common_subset_comparison",
        "research_question": (
            "At the same retained count on exactly aligned S14-S15 8 s windows, "
            "does SMoLK artifact probability identify >10 bpm HR failures more "
            "effectively than the frozen SQI+ACC gate?"
        ),
        "primary_endpoint": "retained failure_gt_10_bpm_percent",
        "secondary_endpoint": "retained mae_bpm",
        "scope": {
            "primary_subjects": list(PRIMARY_SUBJECTS),
            "context_subjects_not_in_primary_result": [1],
            "aligned_primary_windows": primary_count,
            "excluded_cross_boundary_and_tail_windows": True,
            "interpretation": (
                "Retrospective common-subset comparison; S14-S15 are frozen ACC "
                "validation subjects and SMoLK test subjects, but PPG-only SQI "
                "had previously been evaluated on them."
            ),
        },
        "baseline_provenance": baseline_provenance,
        "smolk_score": (
            "Mean per-sample artifact probability inside the aligned 8 s window; "
            "lower is retained first."
        ),
        "methods": method_rows,
        "subject_metrics": subject_rows,
        "failure_detection_auc": auc,
        "matched_comparison": {
            "retained_count_each": int(acc_result["count"]),
            "coverage_percent_each": acc_result["coverage_percent"],
            "smolk_minus_acc_failure_percentage_points": failure_delta,
            "smolk_minus_acc_mae_bpm": mae_delta,
            "primary_hypothesis_outcome": (
                "supported"
                if failure_delta < 0
                else "not_supported"
                if failure_delta > 0
                else "tied"
            ),
        },
        "limitations": [
            "This evaluates gating of an unchanged PPG HR estimate, not motion-artifact removal or HR recovery.",
            "The SMoLK score uses a 30 s context and noncausal centered operations.",
            "S14-S15 were previously used to evaluate PPG-only SQI, so this is not a new end-to-end holdout.",
            "Matched count controls rejection strength but does not calibrate either score as a probability.",
            "Only one SMoLK random initialization is used.",
        ],
    }

    for index, row in enumerate(rows):
        row["primary_analysis"] = bool(primary[index])
        row["acc_aware_keep"] = bool(acc_keep[index])
        row["smolk_matched_keep"] = bool(smolk_keep[index])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(WINDOWS_CSV, rows, WINDOW_FIELDS)
    write_csv(METHOD_CSV, method_rows, METRIC_FIELDS)
    write_csv(SUBJECT_CSV, subject_rows, SUBJECT_FIELDS)
    save_figure(method_rows)
    temporary = SUMMARY_JSON.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, SUMMARY_JSON)
    print(json.dumps(
        {
            "primary_windows": primary_count,
            "retained_count_each": int(acc_result["count"]),
            "acc_failure_gt_10_percent": acc_result[
                "failure_gt_10_bpm_percent"
            ],
            "smolk_failure_gt_10_percent": smolk_result[
                "failure_gt_10_bpm_percent"
            ],
            "failure_delta_percentage_points": failure_delta,
            "acc_mae_bpm": acc_result["mae_bpm"],
            "smolk_mae_bpm": smolk_result["mae_bpm"],
            "mae_delta_bpm": mae_delta,
            "outcome": summary["matched_comparison"][
                "primary_hypothesis_outcome"
            ],
            "output": str(OUTPUT_DIR),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
