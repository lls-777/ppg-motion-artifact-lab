#!/usr/bin/env python3
"""固定 V3 的 PPG、目标与污染；仅更新 ACC 输入特征。默认不写文件。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from acc_preprocessing_v1 import CORRECTED_MODE, resample_acc
from prepare_ppg_reconstruction_pairs_v2 import (
    ROOT, DALIA, FS, N, acc_features, load_dalia, sha, write_csv, write_json,
)
from prepare_ppg_reconstruction_pairs_v3 import SPLITS, EXPECTED, check_roles

BASE = ROOT / "03_重建训练数据/ppg_reconstruction_pairs_v3"
PAIRS = ROOT / "03_重建训练数据/ppg_reconstruction_pairs_acc_consistency_v1"
OUT = ROOT / "02_结果/ACC训练推理一致性V1"
CODE = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def baseline():
    if json.loads((BASE / "state.json").read_text())["status"] != "complete":
        raise ValueError("Incomplete baseline dataset")
    config = json.loads((BASE / "config.json").read_text())
    trained = json.loads((ROOT / "02_结果/V3/training/config.json").read_text())
    if sha(BASE / "config.json") != trained["pairs_config_sha256"]:
        raise ValueError("Changed baseline config")
    if sha(BASE / "manifest.csv") != config["manifest_sha256"]:
        raise ValueError("Changed baseline manifest")
    if sha(CODE / "prepare_ppg_reconstruction_pairs_v2.py") != config["v2_helper_sha256"]:
        raise ValueError("Changed feature/noise helper")
    rows = read_rows(BASE / "manifest.csv")
    check_roles(rows)
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        count = 4 * sum(EXPECTED[s] for s in SPLITS[split])
        if [int(r["index"]) for r in part] != list(range(count)):
            raise ValueError("Changed row order/count")
        if {r["condition"] for r in part} != {"identity", "snr_10", "snr_0", "snr_-5"}:
            raise ValueError("Changed noise conditions")
    return config, rows


def read_arrays(directory, config, split):
    if split not in SPLITS or sha(directory / f"{split}.npz") != config["files"][split]["sha256"]:
        raise ValueError("Unknown split or changed arrays")
    with np.load(directory / f"{split}.npz", allow_pickle=False) as bundle:
        x, y = bundle["x"].copy(), bundle["y"].copy()
    count = 4 * sum(EXPECTED[s] for s in SPLITS[split])
    if x.shape != (count, 5, N) or y.shape != (count, 1, N):
        raise ValueError("Unexpected dimensions")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (x, y)):
        raise ValueError("Unexpected dtype or nonfinite arrays")
    return x, y


def load_split(split):
    config = json.loads((PAIRS / "config.json").read_text())
    if json.loads((PAIRS / "state.json").read_text())["status"] != "complete":
        raise ValueError("Incomplete corrected dataset")
    old, _ = baseline()
    expected = {
        "source_config_sha256": sha(BASE / "config.json"),
        "manifest_sha256": sha(BASE / "manifest.csv"),
        "acc_code_sha256": sha(CODE / "acc_preprocessing_v1.py"),
        "feature_code_sha256": sha(CODE / "prepare_ppg_reconstruction_pairs_v2.py"),
        "mode": CORRECTED_MODE, "script_sha256": sha(Path(__file__)),
        "numpy": np.__version__, "scipy": scipy.__version__,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Changed data provenance")
    x, y = read_arrays(PAIRS, config, split)
    old_x, old_y = read_arrays(BASE, old, split)
    if array_sha(x[:, 0]) != array_sha(old_x[:, 0]) or array_sha(y) != array_sha(old_y):
        raise ValueError("PPG or target changed")
    return x, y


def check_dataset(splits=("train", "validation")):
    if (PAIRS / "manifest.csv").read_bytes() != (BASE / "manifest.csv").read_bytes():
        raise ValueError("Changed manifest")
    baseline()
    for split in splits:
        load_split(split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    if not parser.parse_args().write_outputs:
        print(f"ACC-only replacement; fixed V3 PPG/targets; output={PAIRS}")
        return
    if PAIRS.exists():
        raise FileExistsError(PAIRS)
    config, rows = baseline()
    bundles, audit = {}, []
    for split in SPLITS:
        x, y = read_arrays(BASE, config, split)
        ppg_sha, target_sha = array_sha(x[:, 0]), array_sha(y)
        part = [r for r in rows if r["split"] == split]
        for sid in sorted({int(r["acc_subject"]) for r in part}):
            if sha(DALIA / f"S{sid}/S{sid}.pkl") != config["source_sha256"][str(sid)]:
                raise ValueError("Changed raw ACC source")
            raw = load_dalia(sid)["signal"]["wrist"]["ACC"]
            old = resample_acc(raw, dc_normalize=False)
            corrected = resample_acc(raw, dc_normalize=True)
            for meta in part:
                if int(meta["acc_subject"]) != sid:
                    continue
                sample = float(meta["acc_start_s"]) * FS
                first, index = int(sample), int(meta["index"])
                if sample != first or first < N or first + 2 * N > len(old):
                    raise ValueError("Invalid donor coordinate")
                historical = acc_features(old[first:first + N]).T.astype(np.float32)
                np.testing.assert_array_equal(historical, x[index, 1:])
                updated = acc_features(corrected[first:first + N]).T.astype(np.float32)
                delta = updated.astype(float) - historical.astype(float)
                x[index, 1:] = updated
                audit.append({"split": split, "index": index, "acc_subject": sid,
                              "acc_start_s": first / FS, "legacy_exact": True,
                              "feature_change_rms": float(np.sqrt(np.mean(delta**2)))})
        if array_sha(x[:, 0]) != ppg_sha or array_sha(y) != target_sha or not np.isfinite(x).all():
            raise ValueError("Replacement damaged frozen data")
        bundles[split] = (x, y)
    PAIRS.mkdir(parents=True)
    write_json(PAIRS / "state.json", {"status": "preparing"})
    files = {}
    for split, (x, y) in bundles.items():
        path = PAIRS / f"{split}.npz"
        np.savez_compressed(path, x=x, y=y)
        files[split] = {"pairs": len(x), "sha256": sha(path),
                        "ppg_sha256": array_sha(x[:, 0]), "target_sha256": array_sha(y)}
    (PAIRS / "manifest.csv").write_bytes((BASE / "manifest.csv").read_bytes())
    write_csv(PAIRS / "feature_audit.csv", audit)
    write_json(PAIRS / "config.json", {
        "version": "acc_consistency_v1", "mode": CORRECTED_MODE, "files": files,
        "source_config_sha256": sha(BASE / "config.json"),
        "manifest_sha256": sha(PAIRS / "manifest.csv"),
        "acc_code_sha256": sha(CODE / "acc_preprocessing_v1.py"),
        "feature_code_sha256": sha(CODE / "prepare_ppg_reconstruction_pairs_v2.py"),
        "script_sha256": sha(Path(__file__)), "numpy": np.__version__, "scipy": scipy.__version__,
        "roles": {"train": "S2-S9", "validation": "S10-S11",
                  "test": "S12-S13 historical paired evaluation; not blind"},
        "limit": "Original synthetic contamination is frozen; only model ACC inputs change.",
    })
    write_json(PAIRS / "state.json", {"status": "complete"})
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
