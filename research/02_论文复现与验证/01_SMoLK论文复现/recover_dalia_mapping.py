#!/usr/bin/env python3
"""恢复 SMoLK PPG-DaLiA 数组行到原始受试者时间的精确映射。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal
from scipy.signal import savgol_filter

from run_smoke import COMMIT, LearnedFilters, normalize


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RAW_ROOT = (
    PROJECT_ROOT
    / "90_共享原始数据/ppg_dalia/PPG_FieldStudy"
)
TRAIN_X = ROOT / "data/new_PPG_DaLiA_train/processed_dataset/scaled_ppgs.npy"
TRAIN_Y = ROOT / "data/new_PPG_DaLiA_train/processed_dataset/seg_labels.npy"
TEST_X = ROOT / "data/new_PPG_DaLiA_test/processed_dataset/scaled_ppgs.npy"
TEST_Y = ROOT / "data/new_PPG_DaLiA_test/processed_dataset/seg_labels.npy"
MODEL_PATH = ROOT / "outputs/full_seed0/model_state.pt"
OUT_DIR = ROOT / "outputs/dalia_mapping"
SEGMENT_CSV = OUT_DIR / "dalia_segment_mapping.csv"
HR_WINDOW_CSV = OUT_DIR / "dalia_hr_window_mapping.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"

PPG_FS_HZ = 64
SEGMENT_SECONDS = 30
SEGMENT_SAMPLES = PPG_FS_HZ * SEGMENT_SECONDS
HR_WINDOW_SECONDS = 8
HR_STEP_SECONDS = 2
HR_WINDOW_SAMPLES = PPG_FS_HZ * HR_WINDOW_SECONDS
HR_STEP_SAMPLES = PPG_FS_HZ * HR_STEP_SECONDS
TRAIN_SUBJECTS = tuple(range(2, 14))
TEST_SUBJECTS = (1, 14, 15)
EXPECTED_ROWS = {"train": 3436, "test": 869}

SEGMENT_FIELDS = (
    "split",
    "processed_row",
    "subject_id",
    "chunk_index",
    "start_sample",
    "end_sample",
    "start_s",
    "end_s",
    "artifact_label_fraction",
    "processed_row_sha256",
)
HR_FIELDS = (
    "subject_id",
    "hr_window_index",
    "window_start_sample",
    "window_end_sample",
    "window_start_s",
    "window_center_s",
    "window_end_s",
    "fully_contained_30s_segment",
    "mapping_status",
    "processed_test_row",
    "segment_chunk_index",
    "segment_offset_start_sample",
    "label_artifact_fraction",
    "smolk_probability_mean",
    "smolk_artifact_fraction_no_smoothing",
    "smolk_artifact_fraction_savgol_51",
    "smolk_artifact_fraction_savgol_151",
)


def load_subject(subject_id: int) -> tuple[np.ndarray, int]:
    path = RAW_ROOT / f"S{subject_id}/S{subject_id}.pkl"
    with path.open("rb") as stream:
        subject = pickle.load(stream, encoding="latin1")
    ppg = np.asarray(subject["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1)
    reference_hr = np.asarray(subject["label"], dtype=np.float64).reshape(-1)
    expected_hr_rows = 1 + (len(ppg) - HR_WINDOW_SAMPLES) // HR_STEP_SAMPLES
    if len(reference_hr) != expected_hr_rows or not np.isfinite(ppg).all():
        raise ValueError(f"Unexpected S{subject_id} PPG/HR alignment")
    return ppg, len(reference_hr)


def sha256_row(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def scale_segment(raw_segment: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """复现 Segade db2npy.py 中的二阶因果带通和段内 Min-Max。"""
    filtered = signal.sosfilt(sos, raw_segment)
    span = float(filtered.max() - filtered.min())
    if span <= 0:
        raise ValueError("Constant filtered PPG segment")
    return (filtered - filtered.min()) / span


def build_candidates() -> tuple[
    dict[str, dict[str, int | float]],
    dict[int, int],
]:
    sos = signal.butter(
        2,
        [0.9 / (PPG_FS_HZ / 2), 5.0 / (PPG_FS_HZ / 2)],
        btype="bandpass",
        output="sos",
    )
    candidates: dict[str, dict[str, int | float]] = {}
    subject_chunk_counts: dict[int, int] = {}
    for subject_id in range(1, 16):
        ppg, _ = load_subject(subject_id)
        chunk_count = len(ppg) // SEGMENT_SAMPLES
        subject_chunk_counts[subject_id] = chunk_count
        for chunk_index in range(chunk_count):
            start = chunk_index * SEGMENT_SAMPLES
            scaled = scale_segment(ppg[start:start + SEGMENT_SAMPLES], sos)
            digest = sha256_row(scaled)
            if digest in candidates:
                raise ValueError(f"Duplicate processed segment hash: {digest}")
            candidates[digest] = {
                "subject_id": subject_id,
                "chunk_index": chunk_index,
                "start_sample": start,
                "end_sample": start + SEGMENT_SAMPLES,
                "start_s": start / PPG_FS_HZ,
                "end_s": (start + SEGMENT_SAMPLES) / PPG_FS_HZ,
            }
    return candidates, subject_chunk_counts


def map_processed_split(
    split: str,
    x_path: Path,
    y_path: Path,
    allowed_subjects: tuple[int, ...],
    candidates: dict[str, dict[str, int | float]],
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    y = np.load(y_path, mmap_mode="r", allow_pickle=False)
    if x.shape != y.shape or x.shape != (EXPECTED_ROWS[split], SEGMENT_SAMPLES):
        raise ValueError(f"Unexpected {split} arrays: {x.shape}, {y.shape}")
    rows: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for processed_row, (ppg_row, label_row) in enumerate(zip(x, y)):
        digest = sha256_row(ppg_row)
        if digest not in candidates:
            raise ValueError(f"Unmatched {split} row {processed_row}")
        metadata = candidates[digest]
        subject_id = int(metadata["subject_id"])
        chunk_index = int(metadata["chunk_index"])
        if subject_id not in allowed_subjects:
            raise ValueError(f"{split} row {processed_row} crossed subject split")
        key = (subject_id, chunk_index)
        if key in used:
            raise ValueError(f"Repeated {split} segment: {key}")
        used.add(key)
        rows.append(
            {
                "split": split,
                "processed_row": processed_row,
                **metadata,
                "artifact_label_fraction": float(np.mean(label_row)),
                "processed_row_sha256": digest,
            }
        )
    return rows, used


def load_test_probabilities(test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = torch.load(MODEL_PATH, map_location="cpu")
    expected = ("smolk_full_seed0_state_v1", COMMIT, 0)
    actual = (bundle.get("format"), bundle.get("upstream_commit"), bundle.get("seed"))
    if actual != expected:
        raise ValueError(f"Unexpected model metadata: {actual}")
    model = LearnedFilters()
    model.load_state_dict(bundle["model"])
    model.eval()
    outputs = []
    normalized = normalize(test_x)
    with torch.no_grad():
        for start in range(0, len(normalized), 128):
            batch = torch.from_numpy(normalized[start:start + 128, None, :])
            outputs.append(model(batch).numpy())
    probability = np.concatenate(outputs)
    return (
        probability,
        savgol_filter(probability, 51, 3, axis=-1),
        savgol_filter(probability, 151, 3, axis=-1),
    )


def build_hr_window_rows(
    segment_rows: list[dict[str, Any]],
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, int]]]:
    inverse = {
        (int(row["subject_id"]), int(row["chunk_index"])): int(row["processed_row"])
        for row in segment_rows
    }
    probability, smooth_51, smooth_151 = load_test_probabilities(test_x)
    hr_rows: list[dict[str, Any]] = []
    counts: dict[int, dict[str, int]] = {}
    for subject_id in TEST_SUBJECTS:
        ppg, hr_count = load_subject(subject_id)
        contained_count = 0
        cross_boundary_count = 0
        unpublished_tail_count = 0
        for index in range(hr_count):
            start = index * HR_STEP_SAMPLES
            end = start + HR_WINDOW_SAMPLES
            chunk_index = start // SEGMENT_SAMPLES
            within_single_chunk = chunk_index == (end - 1) // SEGMENT_SAMPLES
            published = (subject_id, chunk_index) in inverse
            contained = within_single_chunk and published
            mapping_status = (
                "contained"
                if contained
                else "unpublished_tail"
                if within_single_chunk
                else "cross_boundary"
            )
            if mapping_status == "cross_boundary":
                cross_boundary_count += 1
            elif mapping_status == "unpublished_tail":
                unpublished_tail_count += 1
            row: dict[str, Any] = {
                "subject_id": subject_id,
                "hr_window_index": index,
                "window_start_sample": start,
                "window_end_sample": end,
                "window_start_s": start / PPG_FS_HZ,
                "window_center_s": (start + HR_WINDOW_SAMPLES / 2) / PPG_FS_HZ,
                "window_end_s": end / PPG_FS_HZ,
                "fully_contained_30s_segment": contained,
                "mapping_status": mapping_status,
                "processed_test_row": "",
                "segment_chunk_index": "",
                "segment_offset_start_sample": "",
                "label_artifact_fraction": "",
                "smolk_probability_mean": "",
                "smolk_artifact_fraction_no_smoothing": "",
                "smolk_artifact_fraction_savgol_51": "",
                "smolk_artifact_fraction_savgol_151": "",
            }
            if contained:
                processed_row = inverse[(subject_id, chunk_index)]
                offset = start - chunk_index * SEGMENT_SAMPLES
                sample_slice = slice(offset, offset + HR_WINDOW_SAMPLES)
                row.update(
                    {
                        "processed_test_row": processed_row,
                        "segment_chunk_index": chunk_index,
                        "segment_offset_start_sample": offset,
                        "label_artifact_fraction": float(np.mean(test_y[processed_row, sample_slice])),
                        "smolk_probability_mean": float(np.mean(probability[processed_row, sample_slice])),
                        "smolk_artifact_fraction_no_smoothing": float(
                            np.mean(probability[processed_row, sample_slice] > 0.5)
                        ),
                        "smolk_artifact_fraction_savgol_51": float(
                            np.mean(smooth_51[processed_row, sample_slice] > 0.5)
                        ),
                        "smolk_artifact_fraction_savgol_151": float(
                            np.mean(smooth_151[processed_row, sample_slice] > 0.5)
                        ),
                    }
                )
                contained_count += 1
            hr_rows.append(row)
        counts[subject_id] = {
            "ppg_samples": len(ppg),
            "hr_windows": hr_count,
            "fully_contained_windows": contained_count,
            "cross_boundary_windows": cross_boundary_count,
            "unpublished_tail_windows": unpublished_tail_count,
        }
    return hr_rows, counts


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    outputs = (SEGMENT_CSV, HR_WINDOW_CSV, SUMMARY_JSON)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Mapping output exists; preserve it before rerunning")
    required = (TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, MODEL_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    candidates, subject_chunk_counts = build_candidates()
    train_rows, train_used = map_processed_split(
        "train", TRAIN_X, TRAIN_Y, TRAIN_SUBJECTS, candidates
    )
    test_rows, test_used = map_processed_split(
        "test", TEST_X, TEST_Y, TEST_SUBJECTS, candidates
    )
    expected_test = {
        (subject_id, chunk_index)
        for subject_id in TEST_SUBJECTS
        for chunk_index in range(subject_chunk_counts[subject_id])
    }
    if test_used != expected_test:
        raise ValueError("Test mapping is not a complete permutation of raw 30 s chunks")

    expected_train = {
        (subject_id, chunk_index)
        for subject_id in TRAIN_SUBJECTS
        for chunk_index in range(subject_chunk_counts[subject_id])
    }
    excluded_train = sorted(expected_train - train_used)
    if len(excluded_train) != 5:
        raise ValueError(f"Expected five excluded train chunks, found {excluded_train}")

    test_x = np.load(TEST_X, allow_pickle=False)
    test_y = np.load(TEST_Y, allow_pickle=False)
    hr_rows, hr_counts = build_hr_window_rows(test_rows, test_x, test_y)
    by_split_subject = {
        split: dict(sorted(Counter(int(row["subject_id"]) for row in rows).items()))
        for split, rows in (("train", train_rows), ("test", test_rows))
    }
    summary = {
        "status": "exact_dalia_time_mapping_recovered",
        "upstream_commit": COMMIT,
        "source_script": (
            "https://github.com/chengstark/Segade/blob/"
            "f5b2ad6e180a1474bc47d6c5ec5d482edaed6797/db2npy.py"
        ),
        "mapping_method": {
            "segment_seconds": SEGMENT_SECONDS,
            "ppg_sampling_rate_hz": PPG_FS_HZ,
            "filter": "Butterworth bandpass, order 2, 0.9-5 Hz, scipy.signal.sosfilt",
            "scaling": "per-segment min-max to [0, 1]",
            "match": "exact SHA-256 of all 1920 float64 samples",
        },
        "processed_rows": {"train": len(train_rows), "test": len(test_rows)},
        "mapped_rows": {"train": len(train_used), "test": len(test_used)},
        "subject_chunk_counts": subject_chunk_counts,
        "mapped_rows_by_split_subject": by_split_subject,
        "excluded_raw_train_chunks": [
            {
                "subject_id": subject_id,
                "chunk_index": chunk_index,
                "start_s": chunk_index * SEGMENT_SECONDS,
                "end_s": (chunk_index + 1) * SEGMENT_SECONDS,
                "reason": "not recorded in public processed arrays; annotation-stage cause unknown",
            }
            for subject_id, chunk_index in excluded_train
        ],
        "hr_window_mapping": {
            "window_seconds": HR_WINDOW_SECONDS,
            "step_seconds": HR_STEP_SECONDS,
            "by_subject": hr_counts,
            "fully_contained_total": sum(
                value["fully_contained_windows"] for value in hr_counts.values()
            ),
            "cross_boundary_total": sum(
                value["cross_boundary_windows"] for value in hr_counts.values()
            ),
            "unpublished_tail_total": sum(
                value["unpublished_tail_windows"] for value in hr_counts.values()
            ),
        },
        "limitations": [
            "The unavailable annotation database prevents identifying why five training chunks were excluded.",
            "Cross-boundary 8 s windows are left unmapped rather than combining independently normalized 30 s segments.",
            "SMoLK probabilities use full 30 s normalization, centered same convolution, and centered SG smoothing.",
            "This mapping does not establish heart-rate improvement or real-time deployability.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(SEGMENT_CSV, train_rows + test_rows, SEGMENT_FIELDS)
    write_csv(HR_WINDOW_CSV, hr_rows, HR_FIELDS)
    temporary = SUMMARY_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, SUMMARY_JSON)
    print(json.dumps(
        {
            "segment_rows": len(train_rows) + len(test_rows),
            "hr_rows": len(hr_rows),
            "fully_contained": summary["hr_window_mapping"]["fully_contained_total"],
            "cross_boundary": summary["hr_window_mapping"]["cross_boundary_total"],
            "unpublished_tail": summary["hr_window_mapping"]["unpublished_tail_total"],
            "excluded_train_chunks": summary["excluded_raw_train_chunks"],
            "output": str(OUT_DIR),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
