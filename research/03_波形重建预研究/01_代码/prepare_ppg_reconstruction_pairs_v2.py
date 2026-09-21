#!/usr/bin/env python3
"""建立受控配对数据；--write-outputs 才下载和写入，参考不是无噪声真值。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pickle
import re
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "90_共享原始数据/bidmc_v1"
PAIRS = ROOT / "03_重建训练数据/ppg_reconstruction_pairs_v2"
OUT = ROOT / "02_结果/V2"
DALIA = ROOT.parent / "90_共享原始数据/ppg_dalia/PPG_FieldStudy"
URL = "https://www.physionet.org/files/bidmc/1.0.0/"
FS, N, SEED = 64, 512, 20260917
DONORS = {"train": list(range(2, 10)), "validation": [10, 11], "test": [12, 13]}
CONFIG = {
    "seed": SEED, "fs_hz": FS, "samples": N, "band_hz": [0.5, 8.0],
    "bidmc_record_numbers": list(range(1, 54)),
    "split": "sorted patient IDs shuffled once; 70% train, 15% validation, remainder test",
    "acc_donors": DONORS, "snr_db": [10.0, 0.0, -5.0],
    "identity_examples": "one per accepted reference segment, with nonzero ACC",
    "reference_sampling": "8 s segments starting at 8, 24, ..., 456 s",
    "quality_rule": "finite, nonflat; peak HR 40..200; ECG monitor HR agreement <=5 bpm",
    "qualification": "heuristic quality screen, not expert clean-waveform annotation",
    "noise": "random ACC FIR plus squared ACC, drift and independent contact-like transient",
    "source": "https://physionet.org/content/bidmc/1.0.0/",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bandpass(values, band=(0.5, 8.0)):
    sos = signal.butter(2, band, btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def zscore(values):
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    if not np.all(np.isfinite(values)) or scale < 1e-8:
        raise ValueError("Nonfinite or constant signal")
    return (values - np.mean(values)) / scale


def acc_features(acc):
    """四列：三轴与模长；逐窗口去线性趋势后缩放，非原始 g 单位。"""
    refs = np.column_stack((acc, np.linalg.norm(acc, axis=1)))
    refs = signal.detrend(refs, axis=0)
    scale = np.std(refs, axis=0)
    return refs / np.maximum(scale, 1e-6)


def peak_times(wave):
    if np.std(wave) < 1e-8 or not np.all(np.isfinite(wave)):
        return np.array([], dtype=float)
    peaks, _ = signal.find_peaks(zscore(wave), distance=18, prominence=0.2)
    return peaks / FS


def peak_hr(wave):
    intervals = np.diff(peak_times(wave))
    intervals = intervals[(intervals >= 60 / 220) & (intervals <= 2)]
    return float(60 / np.median(intervals)) if len(intervals) >= 2 else float("nan")


def load_dalia(subject):
    path = DALIA / f"S{subject}/S{subject}.pkl"
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    return data


def fetch(relative, checksums=None):
    """固定官方地址；已有文件通过官方 SHA256 校验后才复用。"""
    path = RAW / relative
    data = path.read_bytes() if path.exists() else None
    if data is None:
        for attempt in range(3):
            try:
                with urlopen(URL + relative, timeout=60) as response:
                    data = response.read()
                break
            except (HTTPError, URLError, TimeoutError) as error:
                retryable = not isinstance(error, HTTPError) or error.code in (
                    408, 429, 500, 502, 503, 504)
                if not retryable or attempt == 2:
                    raise
                print(f"Retry {attempt + 1}/2: {relative}: {error}", flush=True)
                sleep(2 ** attempt)
    if checksums is not None:
        expected = checksums.get(relative)
        if expected is None or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Official checksum mismatch: {relative}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return data.decode("utf-8-sig")


def csv_columns(text, names):
    rows = list(csv.DictReader(io.StringIO(text), skipinitialspace=True))
    return [np.asarray([float(row[name]) for row in rows]) for name in names]


def donor_bank(subjects):
    bank = []
    for sid in subjects:
        data = load_dalia(sid)
        acc = signal.resample_poly(data["signal"]["wrist"]["ACC"], 2, 1, axis=0)
        starts = np.arange(N, len(acc) - 2 * N, N)
        starts = starts[np.linspace(0, len(starts) - 1, min(300, len(starts))).astype(int)]
        for start in starts:
            window = acc[start:start + N]
            if np.all(np.isfinite(window)) and np.max(np.std(window, axis=0)) > 0.01:
                bank.append((sid, int(start), acc_features(window)))
    if not bank:
        raise ValueError("No usable ACC donors")
    return bank


def make_noise(refs, rng):
    """这是可审计的合成污染假设，不是经过辨识的真实光学传递模型。"""
    noise = np.zeros(N)
    for column in range(4):
        kernel = rng.normal(size=17) * np.exp(-np.arange(17) / 6)
        noise += signal.lfilter(kernel, [1], refs[:, column])
    noise += 0.3 * np.sum(refs**2 - np.mean(refs**2, axis=0), axis=1)
    time = np.arange(N) / FS
    noise += rng.normal() * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * time)
    center = rng.integers(32, N - 32)
    noise += rng.normal(0, 3) * np.exp(-0.5 * ((np.arange(N) - center) / 8)**2)
    return zscore(bandpass(noise))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if not args.write_outputs:
        print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
        return
    if PAIRS.exists():
        raise FileExistsError(PAIRS)
    for sid in sum(DONORS.values(), []):
        if not (DALIA / f"S{sid}/S{sid}.pkl").is_file():
            raise FileNotFoundError(f"Missing ACC donor S{sid}")
    RAW.mkdir(parents=True, exist_ok=True)
    checksum_text = fetch("SHA256SUMS.txt")
    checksums = {}
    for line in checksum_text.splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.lstrip("*").removeprefix("./")] = digest
    fetch("LICENSE", checksums)
    fetch("README", checksums)
    patients = {}
    for rec in CONFIG["bidmc_record_numbers"]:
        text = fetch(f"bidmc_csv/bidmc_{rec:02d}_Fix.txt", checksums)
        match = re.search(r"MIMIC II matched wdb ID:\s*(s\d+)", text)
        if not match:
            raise ValueError(f"Unknown patient ID in record {rec}")
        patients[rec] = match.group(1)
        fetch(f"bidmc_csv/bidmc_{rec:02d}_Signals.csv", checksums)
        fetch(f"bidmc_csv/bidmc_{rec:02d}_Numerics.csv", checksums)
        print(f"BIDMC source {rec:02d}/53 verified", flush=True)
    groups = sorted(set(patients.values()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    a, b = int(0.7 * len(groups)), int(0.85 * len(groups))
    splits = {"train": groups[:a], "validation": groups[a:b], "test": groups[b:]}
    PAIRS.mkdir()
    write_json(PAIRS / "state.json", {"status": "preparing"})
    manifest, screening, saved = [], [], {}
    for split, patient_ids in splits.items():
        bank = donor_bank(DONORS[split])
        xs, ys = [], []
        for rec, patient in patients.items():
            if patient not in patient_ids:
                continue
            prefix = f"bidmc_csv/bidmc_{rec:02d}_"
            time, ppg = csv_columns(fetch(prefix + "Signals.csv", checksums),
                                    ["Time [s]", "PLETH"])
            nt, hr = csv_columns(fetch(prefix + "Numerics.csv", checksums),
                                 ["Time [s]", "HR"])
            # 官方 CSV 的时间列在100秒后舍入到0.01秒；采样仍为125 Hz。
            expected_time = np.arange(len(time)) / 125
            if not np.allclose(time, expected_time, rtol=0, atol=0.0051):
                raise ValueError(f"Unexpected BIDMC time axis: {rec}")
            if not np.all(np.isfinite(ppg)):
                raise ValueError(f"Nonfinite BIDMC recording: {rec}")
            ppg = bandpass(signal.resample_poly(ppg, 64, 125))
            for start_s in range(8, 464, 16):
                target = ppg[start_s * FS:(start_s + 8) * FS]
                numeric = hr[(nt >= start_s) & (nt < start_s + 8)]
                numeric = numeric[np.isfinite(numeric) & (numeric > 0)]
                reference_hr = float(np.median(numeric)) if len(numeric) else np.nan
                observed_hr = peak_hr(target) if len(target) == N else np.nan
                keep = bool(np.isfinite(observed_hr) and np.isfinite(reference_hr)
                            and 40 <= observed_hr <= 200
                            and abs(observed_hr - reference_hr) <= 5)
                screening.append({"record": rec, "patient": patient, "split": split,
                                  "start_s": start_s, "accepted": keep})
                if not keep:
                    continue
                target = zscore(target)
                for level in (None, *CONFIG["snr_db"]):
                    sid, acc_start, refs = bank[int(rng.integers(len(bank)))]
                    noise = np.zeros(N) if level is None else make_noise(refs, rng) * 10**(-level / 20)
                    observed = target + noise
                    mu, sd = float(observed.mean()), float(observed.std())
                    # 目标使用输入的同一个变换，禁止用干净目标统计量归一化输入。
                    xs.append(np.column_stack(((observed - mu) / sd, refs)).T)
                    ys.append(((target - mu) / sd)[None])
                    manifest.append({"split": split, "index": len(xs) - 1,
                                     "record": rec, "patient": patient, "start_s": start_s,
                                     "acc_subject": sid, "acc_start_s": acc_start / FS,
                                     "condition": "identity" if level is None else f"snr_{level:g}",
                                     "input_mean": mu, "input_std": sd})
            print(f"{split}: BIDMC {rec:02d}, pairs={len(xs)}", flush=True)
        if len(xs) < 32:
            raise ValueError(f"Too few accepted pairs in {split}: {len(xs)}")
        path = PAIRS / f"{split}.npz"
        np.savez_compressed(path, x=np.asarray(xs, dtype=np.float32),
                            y=np.asarray(ys, dtype=np.float32))
        saved[split] = {"pairs": len(xs), "sha256": sha(path)}
    write_csv(PAIRS / "manifest.csv", manifest)
    write_csv(PAIRS / "quality_screen.csv", screening)
    write_json(PAIRS / "config.json", {**CONFIG, "patient_splits": splits, "files": saved,
               "script_sha256": sha(Path(__file__)),
               "manifest_sha256": sha(PAIRS / "manifest.csv"),
               "source_sha256": {str(p.relative_to(RAW)): sha(p)
                                 for p in sorted(RAW.rglob("*")) if p.is_file()},
               "acc_source_sha256": {str(s): sha(DALIA / f"S{s}/S{s}.pkl")
                                    for s in sum(DONORS.values(), [])}})
    write_json(PAIRS / "state.json", {"status": "complete"})
    print(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
