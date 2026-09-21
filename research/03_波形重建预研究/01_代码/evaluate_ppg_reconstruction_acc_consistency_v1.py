#!/usr/bin/env python3
"""冻结规则的一次开发回归；只有 --write-outputs 才执行和写入新目录。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy
import torch

from acc_preprocessing_v1 import resample_acc
from prepare_ppg_reconstruction_pairs_v2 import (
    FS, N, DALIA, acc_features, bandpass, load_dalia,
)
from prepare_ppg_reconstruction_acc_consistency_v1 import (
    ROOT, PAIRS, OUT, BASE, load_split, check_dataset, read_rows,
    sha, write_csv, write_json,
)
from train_ppg_reconstruction_acc_consistency_v1 import TRAIN
from train_ppg_reconstruction_v2 import ResidualPPG, predict
from evaluate_ppg_reconstruction_v3 import FIELDS, THRESHOLDS, metric_row, center_peaks

RESULT = OUT / "evaluation"
HISTORY = ROOT / "02_结果/ACC预处理修正V1"
OLD_MODEL = ROOT / "02_结果/V3/training/model.pt"
METHODS = ("input", "v3_corrected", "retrained")
FIXED = {(14, 262.0), (15, 126.0)}
TOLERANCE = 1e-7
EXTRA_FIELDS = ("beat_timing_mae_ms", "beat_tp", "beat_fp", "beat_fn",
                "matched_ibi_count", "matched_ibi_mae_ms")
LIMITS = [
    "Historical development regression; S12-S15 are previously observed subjects.",
    "Original synthetic contamination is frozen; only model ACC inputs change.",
    "Overlapping real windows are not independent subjects or a blinded test.",
    "Paired references are heuristically screened, not expert clean-waveform truth.",
    "Real PPG has no synchronized clean morphology truth.",
    "Input-relative peak changes are not ECG beat accuracy or HRV validation.",
    "Validation has no 40-50 bpm reference window.",
    "One fixed seed and comparison; no parameter adjustment after evaluation.",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(past):
    if read_json(TRAIN / "state.json")["status"] != "complete":
        raise ValueError("New training is incomplete")
    if sha(OLD_MODEL) != past["model_sha256"]:
        raise ValueError("Historical model changed")
    expected = {**past["historical_source_sha256"], **past["code_sha256"]}
    for name, digest in expected.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Historical helper changed: {name}")
    if sha(ROOT.parent / "01_自主基线研究/01_代码/ppg_hr_baseline.py") != past["hr_baseline_code_sha256"]:
        raise ValueError("HR evaluation helper changed")
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)}
    if versions != past["versions"]:
        raise ValueError("Historical evaluation environment changed")
    models, hashes = {}, {}
    for name, path, expected_format, dataset in (
        ("v3_corrected", OLD_MODEL, "ppg_reconstruction_v3", BASE),
        ("retrained", TRAIN / "model.pt", "ppg_reconstruction_acc_consistency_v1", PAIRS),
    ):
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if (bundle["format"] != expected_format
                or bundle["config"]["pairs_config_sha256"] != sha(dataset / "config.json")):
            raise ValueError(f"Checkpoint/dataset mismatch: {name}")
        if name == "retrained" and bundle["config"] != read_json(TRAIN / "config.json"):
            raise ValueError("New checkpoint and training config differ")
        model = ResidualPPG()
        model.load_state_dict(bundle["model"], strict=True)
        model.eval()
        models[name], hashes[name] = model, sha(path)
    return models, hashes


def predictions(models, x):
    outputs = {"input": x[:, 0]}
    outputs.update({name: predict(model, x)[:, 0] for name, model in models.items()})
    if any(value.shape != (len(x), N) or not np.isfinite(value).all()
           for value in outputs.values()):
        raise ValueError("Invalid model output")
    return outputs


def save_example(name, inputs, waves, subject, start, hr, target=None):
    import matplotlib.pyplot as plt

    arrays = dict(inputs=inputs, subject=int(subject), start_s=float(start),
                  reference_hr=float(hr), fs_hz=FS)
    arrays.update({f"wave_{m}": waves[m] for m in METHODS})
    if target is not None:
        arrays["target"] = target
    np.savez_compressed(RESULT / f"{name}.npz", **arrays)
    time = np.arange(N) / FS
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for method in METHODS:
        wave = waves[method]
        axes[0].plot(time, wave, label=method, alpha=0.8)
        if target is not None:
            axes[1].plot(time, wave - target, label=method)
        else:
            peaks = center_peaks(wave)
            axes[1].plot(time, wave, label=method, alpha=0.7)
            axes[1].scatter(peaks, wave[np.rint(peaks * FS).astype(int)], s=12)
    if target is not None:
        axes[0].plot(time, target, "k--", label="Screened reference")
    for i, label in enumerate(("ACC x", "ACC y", "ACC z", "ACC magnitude")):
        axes[2].plot(time, inputs[i + 1], label=label)
    axes[0].set_ylabel("PPG / input std")
    axes[1].set_ylabel("Output minus reference" if target is not None else "PPG and detected peaks")
    axes[2].set_ylabel("Detrended ACC / column std")
    for axis in axes:
        axis.set(xlabel="Seconds within window", xlim=(0, 8))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(f"S{subject} | {float(start):g} s | ECG-derived HR {float(hr):.2f} bpm")
    fig.savefig(RESULT / f"{name}.png", dpi=160)
    plt.close(fig)


def paired_evaluation(models):
    check_dataset(splits=("test",))
    x, target = load_split("test")
    metadata = sorted(
        (r for r in read_rows(PAIRS / "manifest.csv") if r["split"] == "test"),
        key=lambda r: int(r["index"]),
    )
    if (len(x) != 1040 or len(metadata) != len(x)
            or [int(r["index"]) for r in metadata] != list(range(len(x)))
            or {r["reference_subject"] for r in metadata} != {"12", "13"}):
        raise ValueError("Historical paired evaluation set changed")
    waves = predictions(models, x)
    rows, saved = [], False
    for i, meta in enumerate(metadata):
        for method in METHODS:
            rows.append(metric_row(
                "paired", meta["reference_subject"], meta["reference_start_s"],
                meta["condition"], "", float(meta["reference_hr"]),
                method, waves[method][i], x[i], target[i, 0],
            ))
        if not saved and meta["condition"] == "snr_0":
            save_example("paired_first_snr_0", x[i], {m: waves[m][i] for m in METHODS},
                         meta["reference_subject"], meta["reference_start_s"],
                         meta["reference_hr"], target[i, 0])
            saved = True
    if not saved:
        raise ValueError("Missing fixed first 0 dB example")
    return rows


def assert_replay(row, prior):
    categories = {"scope", "subject", "reference_subject", "condition",
                  "activity", "hr_stratum", "method"}
    for field, value in prior.items():
        if field not in categories and not np.isclose(
                float(row[field]), float(value), atol=TOLERANCE, rtol=0, equal_nan=True):
            raise ValueError(f"Replay mismatch: S{row['subject']} "
                             f"{row['reference_start_s']} {row['method']} {field}")


def real_evaluation(models, past):
    saved = read_rows(HISTORY / "windows.csv")
    previous = {(r["subject"], float(r["reference_start_s"]), r["method"]): r for r in saved}
    contexts = [r for r in saved if r["method"] == "input"]
    if (len(contexts) != 6682 or len(previous) != 4 * 6682
            or {r["subject"] for r in contexts} != {"14", "15"}):
        raise ValueError("Historical real windows changed or duplicated")
    historical_path = ROOT / "02_结果/V3/evaluation/real_windows.csv"
    if sha(historical_path) != past["historical_windows_sha256"]:
        raise ValueError("Historical window set changed")
    historical = [r for r in read_rows(historical_path) if r["method"] == "input"]
    if contexts != [{**r, "scope": "acc_dc_diagnostic"} for r in historical]:
        raise ValueError("Input contexts differ from the frozen V3 evaluation")
    rows, hashes, found = [], {}, set()
    for sid in (14, 15):
        hashes[str(sid)] = sha(DALIA / f"S{sid}/S{sid}.pkl")
        if hashes[str(sid)] != past["raw_source_sha256"][str(sid)]:
            raise ValueError(f"Raw source changed: S{sid}")
        data = load_dalia(sid)
        wrist = data["signal"]["wrist"]
        ppg = bandpass(np.asarray(wrist["BVP"], dtype=float).reshape(-1))
        acc = resample_acc(np.asarray(wrist["ACC"], dtype=float), dc_normalize=True)
        selected = [r for r in contexts if r["subject"] == str(sid)]
        batch = []
        for meta in selected:
            sample = float(meta["reference_start_s"]) * FS
            first = int(sample)
            if sample != first or first < N or first + 2 * N > len(ppg) or first % 128:
                raise ValueError("Historical window coordinate changed")
            if float(meta["reference_hr"]) != float(data["label"][first // 128]):
                raise ValueError("Historical HR label changed")
            raw = ppg[first:first + N]
            if not np.isfinite(raw).all() or np.std(raw) < 1e-8:
                raise ValueError("Invalid real PPG window")
            refs = acc_features(acc[first:first + N])
            batch.append(np.column_stack(((raw - raw.mean()) / raw.std(), refs)).T)
        x = np.asarray(batch, dtype=np.float32)
        if x.shape != (len(selected), 5, N) or not np.isfinite(x).all():
            raise ValueError("Invalid corrected real input")
        waves = predictions(models, x)
        for i, meta in enumerate(selected):
            start, activity = float(meta["reference_start_s"]), int(meta["activity"])
            for method in METHODS:
                row = metric_row("historical_regression", sid, start, f"activity_{activity}",
                                 activity, float(meta["reference_hr"]), method,
                                 waves[method][i], x[i])
                if method != "retrained":
                    old = "input" if method == "input" else "dc_corrected"
                    assert_replay(row, previous[(str(sid), start, old)])
                rows.append(row)
            if (sid, start) in FIXED:
                if activity != 1:
                    raise ValueError("Fixed sitting example changed")
                save_example(f"real_s{sid}_{start:g}s", x[i],
                             {m: waves[m][i] for m in METHODS},
                             sid, start, meta["reference_hr"])
                found.add((sid, start))
        print(f"S{sid}: {len(selected)} historical windows replayed", flush=True)
    if len(rows) != 3 * 6682 or found != FIXED:
        raise ValueError("Incomplete real evaluation or fixed examples")
    return rows, hashes


def aggregate(rows):
    output = []
    groups = sorted({(r["scope"], r["subject"], r["condition"]) for r in rows})
    for scope, subject, condition in groups:
        for method in METHODS:
            selected = [r for r in rows if
                        (r["scope"], r["subject"], r["condition"], r["method"])
                        == (scope, subject, condition, method)]
            if not selected:
                raise ValueError("Missing paired method")
            for field in (*FIELDS, *EXTRA_FIELDS):
                values = np.asarray([r[field] for r in selected], dtype=float)
                valid = values[np.isfinite(values)]
                output.append({
                    "scope": scope, "subject": subject, "condition": condition,
                    "method": method, "metric": field, "windows": len(values),
                    "valid": len(valid), "coverage": len(valid) / len(values),
                    "mean": float(valid.mean()) if len(valid) else "",
                    "p90": float(np.percentile(valid, 90)) if len(valid) else "",
                })
    return output


def acceptance(metrics):
    table = {(r["scope"], r["subject"], r["condition"], r["method"], r["metric"]): r
             for r in metrics}
    checks = []

    def value(scope, sid, condition, method, field, *, p90=False, complete=False):
        row = table.get((scope, sid, condition, method, field))
        if not row or not row["valid"] or complete and row["valid"] != row["windows"]:
            return None
        return row["p90" if p90 else "mean"]

    def record(name, observed, limit, direction="max"):
        passed = observed is not None and limit is not None
        if passed:
            passed = observed <= limit + TOLERANCE if direction == "max" else observed >= limit - TOLERANCE
        checks.append({"name": name, "observed": observed, "limit": limit,
                       "direction": direction, "passed": bool(passed)})

    for sid in ("12", "13"):
        for field, threshold, direction, p90 in (
            ("nrmse", "identity_nrmse_mean_max", "max", False),
            ("nrmse", "identity_nrmse_p90_max", "max", True),
            ("correlation", "identity_correlation_mean_min", "min", False),
            ("beat_f1", "identity_beat_f1_mean_min", "min", False),
        ):
            record(f"S{sid}/identity/{threshold}",
                   value("paired", sid, "identity", "retrained", field, p90=p90, complete=True),
                   THRESHOLDS[threshold], direction)
        for condition in ("snr_10", "snr_0", "snr_-5"):
            for field, direction in (("nrmse", "max"), ("beat_f1", "min")):
                new = value("paired", sid, condition, "retrained", field, complete=True)
                for baseline in ("input", "v3_corrected"):
                    record(f"S{sid}/{condition}/{field}/vs_{baseline}", new,
                           value("paired", sid, condition, baseline, field, complete=True),
                           direction)
            record(f"S{sid}/{condition}/snr_gain",
                   value("paired", sid, condition, "retrained", "snr_gain_db", complete=True),
                   THRESHOLDS["synthetic_snr_gain_mean_min_db"], "min")
    for sid in ("14", "15"):
        for activity in range(1, 9):
            condition = f"activity_{activity}"
            for prefix in ("peak", "spectral"):
                for suffix, margin in (("error", 1.0), ("failure", 0.02)):
                    field, complete = f"{prefix}_{suffix}", suffix == "failure"
                    new = value("historical_regression", sid, condition, "retrained",
                                field, complete=complete)
                    for baseline in ("input", "v3_corrected"):
                        old = value("historical_regression", sid, condition, baseline,
                                    field, complete=complete)
                        record(f"S{sid}/{condition}/{field}/vs_{baseline}",
                               new, None if old is None else old + margin)
        for kind in ("extra", "missing"):
            field = f"input_relative_{kind}_peak_fraction"
            record(f"S{sid}/sitting/{field}",
                   value("historical_regression", sid, "activity_1", "retrained",
                         field, complete=True),
                   THRESHOLDS[f"sitting_{kind}_peak_fraction_max"])
    return {"status": "development_checks_met" if all(r["passed"] for r in checks)
            else "development_checks_not_met",
            "checks": checks, "numeric_tolerance": TOLERANCE, "limits": LIMITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    if not parser.parse_args().write_outputs:
        print(f"One frozen development evaluation; --write-outputs creates {RESULT}")
        return
    if RESULT.exists():
        raise FileExistsError(RESULT)
    RESULT.mkdir(parents=True)
    write_json(RESULT / "state.json", {"status": "evaluating"})
    # 历史指标函数间接导入 Matplotlib；把缓存和图片都限制在本轮结果目录。
    os.environ["MPLCONFIGDIR"], os.environ["MPLBACKEND"] = str(RESULT / "_mplconfig"), "Agg"
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        past = read_json(HISTORY / "summary.json")
        models, model_hashes = load_models(past)
        paired = paired_evaluation(models)
        real, source_hashes = real_evaluation(models, past)
        metrics = aggregate(paired + real)
        verdict = acceptance(metrics)
        write_csv(RESULT / "paired_windows.csv", paired)
        write_csv(RESULT / "real_windows.csv", real)
        write_csv(RESULT / "metrics.csv", metrics)
        write_json(RESULT / "acceptance.json", verdict)
        artifacts = sorted(p for p in RESULT.iterdir() if p.is_file() and p.name != "state.json")
        write_json(RESULT / "summary.json", {
            "status": verdict["status"], "methods": METHODS,
            "paired_windows": len(paired) // 3, "real_windows": len(real) // 3,
            "real_input_and_old_corrected_replay_passed": True,
            "replay_absolute_tolerance": TOLERANCE, "model_sha256": model_hashes,
            "raw_source_sha256": source_hashes,
            "pairs_config_sha256": sha(PAIRS / "config.json"),
            "pairs_manifest_sha256": sha(PAIRS / "manifest.csv"),
            "historical_summary_sha256": sha(HISTORY / "summary.json"),
            "historical_windows_sha256": sha(HISTORY / "windows.csv"),
            "script_sha256": sha(Path(__file__)),
            "artifacts_sha256": {p.name: sha(p) for p in artifacts},
            "real_hr_mae_margin_bpm": 1.0, "real_failure_margin": 0.02,
            "example_selection": "First test snr_0 row; S14 262 s; S15 126 s.",
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "torch": str(torch.__version__)},
            "limits": LIMITS,
        })
        write_json(RESULT / "state.json", {"status": "complete"})
        print(f"{verdict['status']}: {RESULT}")
    except Exception as error:
        write_json(RESULT / "state.json", {"status": "failed",
                   "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
