#!/usr/bin/env python3
"""探索 SMoLK 能否在 PPG-only usable 内替代 ACC 二阶段筛选。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


DATA_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = DATA_ROOT / "02_结果/smolk_hr_comparison_v1"
INPUT_WINDOWS = INPUT_DIR / "aligned_windows.csv"
INPUT_SUMMARY = INPUT_DIR / "summary.json"
OUTPUT_DIR = DATA_ROOT / "02_结果/smolk_second_stage_followup_v0"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
METRICS_CSV = OUTPUT_DIR / "method_metrics.csv"
DISAGREEMENT_CSV = OUTPUT_DIR / "disagreement_metrics.csv"
SUBJECT_CSV = OUTPUT_DIR / "subject_metrics.csv"

PRIMARY_SUBJECTS = (14, 15)
ERROR_THRESHOLD_BPM = 10.0
METRIC_FIELDS = (
    "scope",
    "method",
    "count",
    "coverage_of_ppg_usable_percent",
    "mae_bpm",
    "rmse_bpm",
    "p90_bpm",
    "within_3_bpm_percent",
    "failure_gt_10_bpm_percent",
)
SUBJECT_FIELDS = (
    "subject_id",
    "method",
    "count",
    "coverage_of_subject_ppg_usable_percent",
    "mae_bpm",
    "failure_gt_10_bpm_percent",
)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Unexpected boolean: {value}")


def load_rows() -> list[dict[str, Any]]:
    summary = json.loads(INPUT_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_retrospective_common_subset_comparison":
        raise ValueError("Input comparison is not a valid completed result")
    with INPUT_WINDOWS.open("r", encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        subject_id = int(raw["subject_id"])
        if subject_id not in PRIMARY_SUBJECTS:
            continue
        rows.append(
            {
                "subject_id": subject_id,
                "window_start_s": float(raw["window_start_s"]),
                "absolute_error_bpm": float(raw["absolute_error_bpm"]),
                "ppg_sqi_state": raw["ppg_sqi_state"],
                "acc_aware_state": raw["acc_aware_state"],
                "smolk_probability_mean": float(raw["smolk_probability_mean"]),
                "acc_aware_keep": parse_bool(raw["acc_aware_keep"]),
            }
        )
    if len(rows) != 6744:
        raise ValueError(f"Unexpected primary aligned rows: {len(rows)}")
    return rows


def select_lowest(
    rows: list[dict[str, Any]],
    eligible: np.ndarray,
    count: int,
) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    if count <= 0 or count > len(indices):
        raise ValueError(f"Invalid selection count: {count}/{len(indices)}")
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


def summarize(
    errors: np.ndarray,
    mask: np.ndarray,
    denominator: int,
    scope: str,
    method: str,
) -> dict[str, Any]:
    selected = errors[mask]
    if len(selected) == 0:
        raise ValueError(f"Empty selection: {scope}/{method}")
    return {
        "scope": scope,
        "method": method,
        "count": int(len(selected)),
        "coverage_of_ppg_usable_percent": float(
            100.0 * len(selected) / denominator
        ),
        "mae_bpm": float(np.mean(selected)),
        "rmse_bpm": float(np.sqrt(np.mean(selected ** 2))),
        "p90_bpm": float(np.percentile(selected, 90)),
        "within_3_bpm_percent": float(100.0 * np.mean(selected <= 3.0)),
        "failure_gt_10_bpm_percent": float(
            100.0 * np.mean(selected > ERROR_THRESHOLD_BPM)
        ),
    }


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


def main() -> None:
    outputs = (SUMMARY_JSON, METRICS_CSV, DISAGREEMENT_CSV, SUBJECT_CSV)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Follow-up output exists; preserve it before rerunning")
    if not INPUT_WINDOWS.is_file() or not INPUT_SUMMARY.is_file():
        raise FileNotFoundError("Valid v1 comparison inputs are missing")

    rows = load_rows()
    errors = np.asarray([row["absolute_error_bpm"] for row in rows])
    subjects = np.asarray([row["subject_id"] for row in rows])
    ppg_usable = np.asarray([
        row["ppg_sqi_state"] == "usable" for row in rows
    ])
    acc_keep = np.asarray([row["acc_aware_keep"] for row in rows])
    if np.any(acc_keep & ~ppg_usable):
        raise ValueError("ACC keep escaped the PPG-only usable universe")

    ppg_count = int(np.sum(ppg_usable))
    acc_count = int(np.sum(acc_keep))
    smolk_keep = select_lowest(rows, ppg_usable, acc_count)
    method_masks = {
        "ppg_only_usable_universe": ppg_usable,
        "acc_second_stage_keep": acc_keep,
        "smolk_second_stage_matched_count": smolk_keep,
    }
    method_rows = [
        summarize(
            errors,
            mask,
            ppg_count,
            "S14_S15_ppg_usable",
            method,
        )
        for method, mask in method_masks.items()
    ]

    disagreement_masks = {
        "both_keep": acc_keep & smolk_keep,
        "acc_only_keep": acc_keep & ~smolk_keep,
        "smolk_only_keep": ~acc_keep & smolk_keep & ppg_usable,
        "both_reject": ~acc_keep & ~smolk_keep & ppg_usable,
    }
    disagreement_rows = [
        summarize(
            errors,
            mask,
            ppg_count,
            "S14_S15_ppg_usable",
            group,
        )
        for group, mask in disagreement_masks.items()
    ]

    subject_rows: list[dict[str, Any]] = []
    for subject_id in PRIMARY_SUBJECTS:
        subject_universe = (subjects == subject_id) & ppg_usable
        subject_acc = (subjects == subject_id) & acc_keep
        subject_smolk = select_lowest(
            rows, subject_universe, int(np.sum(subject_acc))
        )
        denominator = int(np.sum(subject_universe))
        for method, mask in (
            ("acc_second_stage_keep", subject_acc),
            ("smolk_second_stage_matched_within_subject", subject_smolk),
        ):
            result = summarize(
                errors,
                mask,
                denominator,
                f"S{subject_id}_ppg_usable",
                method,
            )
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "method": method,
                    "count": result["count"],
                    "coverage_of_subject_ppg_usable_percent": result[
                        "coverage_of_ppg_usable_percent"
                    ],
                    "mae_bpm": result["mae_bpm"],
                    "failure_gt_10_bpm_percent": result[
                        "failure_gt_10_bpm_percent"
                    ],
                }
            )

    failure = errors[ppg_usable] > ERROR_THRESHOLD_BPM
    if len(np.unique(failure)) != 2:
        raise ValueError("PPG usable universe lacks both failure classes")
    acc_risk_order = {"usable": 0.0, "low_confidence": 1.0, "reject": 2.0}
    acc_risk = np.asarray([
        acc_risk_order[row["acc_aware_state"]] for row in rows
    ])
    auc = {
        "smolk_probability_mean": float(
            roc_auc_score(
                failure,
                np.asarray([
                    row["smolk_probability_mean"] for row in rows
                ])[ppg_usable],
            )
        ),
        "acc_aware_state_risk": float(
            roc_auc_score(failure, acc_risk[ppg_usable])
        ),
    }

    by_method = {row["method"]: row for row in method_rows}
    acc_result = by_method["acc_second_stage_keep"]
    smolk_result = by_method["smolk_second_stage_matched_count"]
    failure_delta = (
        smolk_result["failure_gt_10_bpm_percent"]
        - acc_result["failure_gt_10_bpm_percent"]
    )
    mae_delta = smolk_result["mae_bpm"] - acc_result["mae_bpm"]
    summary = {
        "status": "completed_posthoc_second_stage_followup",
        "timing": (
            "Exploratory follow-up defined after the preregistered full-gate "
            "replacement comparison was observed."
        ),
        "research_question": (
            "Within exactly the same PPG-only usable S14-S15 windows, can "
            "SMoLK replace ACC as the second-stage rejection signal at the "
            "same retained count?"
        ),
        "ppg_usable_windows": ppg_count,
        "retained_count_each": acc_count,
        "coverage_of_ppg_usable_percent_each": acc_result[
            "coverage_of_ppg_usable_percent"
        ],
        "methods": method_rows,
        "disagreement_groups": disagreement_rows,
        "subject_metrics": subject_rows,
        "failure_detection_auc_within_ppg_usable": auc,
        "smolk_minus_acc_failure_percentage_points": failure_delta,
        "smolk_minus_acc_mae_bpm": mae_delta,
        "followup_outcome": (
            "smolk_better"
            if failure_delta < 0
            else "acc_better"
            if failure_delta > 0
            else "tied"
        ),
        "limitations": [
            "This follow-up is exploratory and was defined after seeing the primary replacement result.",
            "It compares ranking/rejection only and does not retrain or combine SMoLK with ACC.",
            "S14-S15 are not a new end-to-end PPG-only SQI holdout.",
            "SMoLK still uses 30 s context and noncausal centered operations.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(METRICS_CSV, method_rows, METRIC_FIELDS)
    write_csv(DISAGREEMENT_CSV, disagreement_rows, METRIC_FIELDS)
    write_csv(SUBJECT_CSV, subject_rows, SUBJECT_FIELDS)
    temporary = SUMMARY_JSON.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, SUMMARY_JSON)
    print(json.dumps(
        {
            "ppg_usable_windows": ppg_count,
            "retained_count_each": acc_count,
            "acc_failure_gt_10_percent": acc_result[
                "failure_gt_10_bpm_percent"
            ],
            "smolk_failure_gt_10_percent": smolk_result[
                "failure_gt_10_bpm_percent"
            ],
            "acc_mae_bpm": acc_result["mae_bpm"],
            "smolk_mae_bpm": smolk_result["mae_bpm"],
            "failure_delta_percentage_points": failure_delta,
            "mae_delta_bpm": mae_delta,
            "outcome": summary["followup_outcome"],
            "output": str(OUTPUT_DIR),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
