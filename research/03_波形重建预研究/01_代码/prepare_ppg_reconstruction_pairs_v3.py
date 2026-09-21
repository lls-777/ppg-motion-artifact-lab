#!/usr/bin/env python3
"""V3：用标签筛选的 E4 带限参考构造配对数据；不是无噪声形态真值。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from prepare_ppg_reconstruction_pairs_v2 import (
    FS, N, SEED, DALIA, acc_features, bandpass, donor_bank, load_dalia,
    make_noise, peak_hr, peak_times, sha, write_csv, write_json, zscore,
)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "02_论文复现与验证/01_SMoLK论文复现"
MAPPING = PAPER / "outputs/dalia_mapping/dalia_segment_mapping.csv"
ARRAYS = PAPER / "data/new_PPG_DaLiA_train/processed_dataset"
PAIRS = ROOT / "03_重建训练数据/ppg_reconstruction_pairs_v3"
OUT = ROOT / "02_结果/V3"
V2_PAIRS = ROOT / "03_重建训练数据/ppg_reconstruction_pairs_v2"
SPLITS = {"train": list(range(2, 10)), "validation": [10, 11], "test": [12, 13]}
EXPECTED = {2: 97, 3: 110, 4: 70, 5: 55, 6: 72, 7: 230,
            8: 42, 9: 11, 10: 170, 11: 156, 12: 87, 13: 173}
CONFIG = {
    "version": "e4_reference_v3", "seed": SEED, "fs_hz": FS, "samples": N,
    "reference_and_acc_splits": SPLITS, "band_hz": [0.5, 8.0],
    "offset_seconds": [2, 10, 18], "label_guard_seconds": 1,
    "quality_rule": "published mask=0 over 10 s; finite/nonflat 8 s; "
                    "peak HR 40..200 and official ECG-HR agreement <=5 bpm",
    "edge_rule": "start>=8 s; window end at least 8 s before recording end",
    "snr_db": [10.0, 0.0, -5.0], "identity_examples": "one per reference",
    "expected_reference_counts": EXPECTED,
    "qualification": "screened band-limited reference, not expert noise-free morphology",
    "limits": [
        "Zero artifact mask does not prove complete annotation or noise-free wide-band PPG.",
        "ACC donors and reference PPG are not simultaneous paired artifact measurements.",
        "Reference source, population, HR distribution and number of updates differ from V2.",
        "Same seed does not imply the same V2 noise or ACC-donor realization.",
        "Validation has no reference below 50 bpm; claims in that range remain limited.",
        "S14/S15 are historical regression data and never enter reference/donor splits.",
    ],
}


def check_roles(rows):
    """同一数据集承担参考和干扰两种角色，必须检查角色并集的受试者隔离。"""
    groups = {}
    for split, allowed in SPLITS.items():
        selected = [r for r in rows if r["split"] == split]
        roles = {int(r["reference_subject"]) for r in selected}
        roles |= {int(r["acc_subject"]) for r in selected}
        if not roles or not roles <= set(allowed):
            raise ValueError(f"Unexpected subject roles: {split}")
        groups[split] = roles
        keys = [(int(r["reference_subject"]), int(r["reference_start_sample"]),
                 r["condition"]) for r in selected]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate reference/condition: {split}")
    names = list(groups)
    if any(groups[a] & groups[b] for i, a in enumerate(names) for b in names[:i]):
        raise ValueError("Cross-role subject leakage")


def collect_references():
    """只读筛选；将标签时间坐标映射到连续带通后的原始 BVP。"""
    with MAPPING.open() as stream:
        mapping = list(csv.DictReader(stream))
    x = np.load(ARRAYS / "scaled_ppgs.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(ARRAYS / "seg_labels.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (3436, 1920) or y.shape != x.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected published arrays")
    old = json.loads((V2_PAIRS / "config.json").read_text())
    sources = {}
    accepted, screening, counts = [], [], []
    # 该窄带处理只用于复核映射，不是模型的输入预处理。
    sos = signal.butter(2, [0.9 / (FS / 2), 5.0 / (FS / 2)],
                        btype="bandpass", output="sos")
    for split, subjects in SPLITS.items():
        for sid in subjects:
            source = DALIA / f"S{sid}/S{sid}.pkl"
            digest = sha(source)
            if digest != old["acc_source_sha256"][str(sid)]:
                raise ValueError(f"Changed raw source: S{sid}")
            sources[str(sid)] = digest
            data = load_dalia(sid)
            raw = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1)
            hr = np.asarray(data["label"], dtype=float).reshape(-1)
            if not np.isfinite(raw).all() or len(hr) != 1 + (len(raw) - N) // 128:
                raise ValueError(f"Unexpected HR time axis: S{sid}")
            ppg = bandpass(raw)
            subject_rows = sorted((r for r in mapping if int(r["subject_id"]) == sid),
                                  key=lambda r: int(r["start_sample"]))
            subject_refs = []
            for row in subject_rows:
                index, start = int(row["processed_row"]), int(row["start_sample"])
                if row["split"] != "train" or start != int(row["chunk_index"]) * 1920:
                    raise ValueError("Unexpected segment coordinate")
                filtered = signal.sosfilt(sos, raw[start:start + 1920])
                scaled = (filtered - filtered.min()) / (filtered.max() - filtered.min())
                # 数组与原始重建须同时匹配记录的字节哈希。
                for values in (x[index], scaled):
                    if hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() != row["processed_row_sha256"]:
                        raise ValueError(f"Changed mapping: S{sid}, row {index}")
                labels = y[index]
                if not np.isclose(labels.mean(), float(row["artifact_label_fraction"])):
                    raise ValueError("Label/mapping mismatch")
                for offset in CONFIG["offset_seconds"]:
                    first = start + offset * FS
                    label_ok = bool(np.all(labels[(offset - 1) * FS:(offset + 9) * FS] == 0))
                    edge_ok = first >= N and first + 2 * N <= len(ppg)
                    observed, reference, keep = None, None, False
                    if label_ok and edge_ok:
                        target = ppg[first:first + N]
                        reference = float(hr[first // 128])
                        observed = peak_hr(target) if np.isfinite(target).all() and target.std() >= 1e-8 else np.nan
                        keep = bool(np.isfinite(observed) and np.isfinite(reference)
                                    and 40 <= observed <= 200 and abs(observed - reference) <= 5)
                        if keep:
                            item = {"split": split, "reference_subject": sid,
                                    "reference_start_sample": first, "reference_start_s": first / FS,
                                    "reference_hr": reference, "reference_peak_hr": float(observed),
                                    "processed_row": index}
                            accepted.append((item, zscore(target)))
                            subject_refs.append(item)
                    screening.append({"split": split, "subject": sid, "start_s": first / FS,
                                      "label_ok": label_ok, "edge_ok": edge_ok, "accepted": keep,
                                      "reference_hr": reference,
                                      "peak_hr": observed if observed is not None and np.isfinite(observed) else None})
            if len(subject_refs) != EXPECTED[sid]:
                raise ValueError(f"Reference inventory changed: S{sid}: {len(subject_refs)}")
            values = [a["reference_hr"] for a in subject_refs]
            counts.append({"split": split, "subject": sid, "references": len(values),
                           "hr_40_50": sum(40 <= h < 50 for h in values),
                           "hr_50_60": sum(50 <= h < 60 for h in values)})
            print(f"{split} S{sid}: {len(subject_refs)} screened references", flush=True)
    return accepted, screening, counts, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if not args.write_outputs:
        print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
        return
    if PAIRS.exists():
        raise FileExistsError(PAIRS)
    accepted, screening, counts, sources = collect_references()
    rng = np.random.default_rng(SEED)
    manifest, bundles = [], {}
    for split, subjects in SPLITS.items():
        bank, xs, ys = donor_bank(subjects), [], []
        for meta, target in accepted:
            if meta["split"] != split:
                continue
            for level in (None, *CONFIG["snr_db"]):
                sid, first, refs = bank[int(rng.integers(len(bank)))]
                noise = np.zeros(N) if level is None else make_noise(refs, rng) * 10 ** (-level / 20)
                observed = target + noise
                mu, sd = float(observed.mean()), float(observed.std())
                xs.append(np.column_stack(((observed - mu) / sd, refs)).T)
                ys.append(((target - mu) / sd)[None])
                manifest.append({**meta, "index": len(xs) - 1, "acc_subject": sid,
                                 "acc_start_s": first / FS,
                                 "condition": "identity" if level is None else f"snr_{level:g}",
                                 "input_mean": mu, "input_std": sd})
        bundles[split] = (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32))
        if not all(np.isfinite(a).all() for a in bundles[split]):
            raise ValueError("Nonfinite pair")
    check_roles(manifest)
    PAIRS.mkdir(parents=True)
    write_json(PAIRS / "state.json", {"status": "preparing"})
    saved = {}
    for split, (x, y) in bundles.items():
        path = PAIRS / f"{split}.npz"
        np.savez_compressed(path, x=x, y=y)
        saved[split] = {"pairs": len(x), "sha256": sha(path)}
    write_csv(PAIRS / "manifest.csv", manifest)
    write_csv(PAIRS / "quality_screen.csv", screening)
    write_csv(PAIRS / "reference_inventory.csv", counts)
    write_json(PAIRS / "config.json", {
        **CONFIG, "files": saved, "reference_inventory": counts, "source_sha256": sources,
        "manifest_sha256": sha(PAIRS / "manifest.csv"),
        "mapping_sha256": sha(MAPPING),
        "published_array_sha256": {name: sha(ARRAYS / name)
                                   for name in ("scaled_ppgs.npy", "seg_labels.npy")},
        "script_sha256": sha(Path(__file__)),
        "v2_helper_sha256": sha(Path(__file__).with_name("prepare_ppg_reconstruction_pairs_v2.py")),
    })
    write_json(PAIRS / "state.json", {"status": "complete"})
    print(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
