#!/usr/bin/env python3
"""冻结 V2/V3 后比较同一输入；仅 --write-outputs 才生成评价产物。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from prepare_ppg_reconstruction_pairs_v3 import (
    FS, N, ROOT, PAIRS, OUT, DALIA, sha, write_csv, write_json,
    bandpass, acc_features, load_dalia, peak_times,
)
from train_ppg_reconstruction_v3 import (
    TRAIN, ResidualPPG, predict, load_split, check_split_isolation,
)

RESULT = OUT / "evaluation"
V2 = ROOT / "02_结果/V2"
METHODS = ("input", "frozen_v2", "frozen_v3")
FIELDS = (
    "nrmse", "correlation", "snr_gain_db", "spectral_error", "peak_error",
    "spectral_failure", "peak_failure", "beat_f1", "change_rmse",
    "input_relative_extra_peak_fraction", "input_relative_missing_peak_fraction",
)
FIXED = {(14, 262.0): 1, (15, 126.0): 1, (14, 816.0): 2}
THRESHOLDS = {
    "identity_nrmse_mean_max": 0.15,
    "identity_nrmse_p90_max": 0.30,
    "identity_correlation_mean_min": 0.98,
    "identity_beat_f1_mean_min": 0.97,
    "synthetic_snr_gain_mean_min_db": 1.0,
    "sitting_peak_mae_margin_bpm": 2.0,
    "sitting_peak_failure_margin": 0.05,
    "sitting_spectral_mae_margin_bpm": 1.0,
    "sitting_extra_peak_fraction_max": 0.05,
    "sitting_missing_peak_fraction_max": 0.05,
    "sitting_extra_peak_ratio_to_v2_max": 0.50,
    "low_hr_test_references_per_subject_min": 10,
    "low_hr_test_subjects_min": 2,
    "motion_hr_mae_margin_bpm": 1.0,
    "motion_failure_margin": 0.02,
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def hr_stratum(hr):
    if not np.isfinite(hr):
        raise ValueError("Nonfinite reference HR")
    return "40_50" if hr < 50 else "50_60" if hr < 60 else "ge60"


def center_peaks(wave):
    peaks = peak_times(wave)
    return peaks[(peaks >= 0.5) & (peaks < 7.5)]


def variants(models, x):
    predictions = {name: predict(model, x)[:, 0] for name, model in models.items()}
    for i in range(len(x)):
        yield {"input": x[i, 0], **{name: value[i] for name, value in predictions.items()}}


def metric_row(scope, subject, start, condition, activity, hr, method, wave, inputs,
               target=None):
    from evaluate_ppg_reconstruction_v2 import evaluate_wave, match_beats

    row = {
        "scope": scope, "subject": str(subject), "reference_subject": str(subject),
        "reference_start_s": float(start), "condition": condition,
        "activity": activity, "hr_stratum": hr_stratum(hr), "method": method,
        **{field: np.nan for field in FIELDS},
        "input_relative_extra_peak_count": np.nan,
        "input_relative_missing_peak_count": np.nan,
        "input_peak_count": np.nan,
    }
    row.update(evaluate_wave(wave, hr, None if target is None else peak_times(target)))
    row["change_rmse"] = float(np.sqrt(np.mean((wave - inputs[0]) ** 2)))
    for prefix in ("spectral", "peak"):
        error = row[f"{prefix}_error"]
        row[f"{prefix}_failure"] = float(not np.isfinite(error) or error > 10)
    if target is not None:
        mse = float(np.mean((wave - target) ** 2))
        input_mse = float(np.mean((inputs[0] - target) ** 2))
        row["nrmse"] = float(np.sqrt(mse) / max(float(np.std(target)), 1e-8))
        row["correlation"] = (float(np.corrcoef(wave, target)[0, 1])
                              if np.std(wave) > 1e-8 else np.nan)
        row["snr_gain_db"] = (float(10 * np.log10(input_mse / max(mse, 1e-12)))
                              if input_mse > 1e-12 else np.nan)
    else:
        reference = center_peaks(inputs[0])
        changes = match_beats(reference, center_peaks(wave), tolerance=0.10)
        row["input_peak_count"] = len(reference)
        row["input_relative_extra_peak_count"] = changes["beat_fp"]
        row["input_relative_missing_peak_count"] = changes["beat_fn"]
        if len(reference):
            row["input_relative_extra_peak_fraction"] = changes["beat_fp"] / len(reference)
            row["input_relative_missing_peak_fraction"] = changes["beat_fn"] / len(reference)
    return row


def synthetic_evaluation(models, manifest):
    x, y = load_split("test")
    metadata = sorted((r for r in manifest if r["split"] == "test"),
                      key=lambda r: int(r["index"]))
    if [int(r["index"]) for r in metadata] != list(range(len(x))):
        raise ValueError("Test manifest indices do not match test arrays")
    rows, examples = [], []
    for i, waves in enumerate(variants(models, x)):
        meta, target = metadata[i], y[i, 0]
        hr = float(meta["reference_hr"])
        for method, wave in waves.items():
            rows.append(metric_row(
                "paired", meta["reference_subject"], meta["reference_start_s"],
                meta["condition"], "", hr, method, wave, x[i], target))
        if not examples and meta["condition"] == "snr_0":
            examples.append({
                "name": "paired_first_snr_0", "waves": waves, "acc": x[i, 1:].T,
                "target": target, "reference_hr": hr,
                "title": f"E4 S{meta['reference_subject']} | {meta['reference_start_s']} s",
            })
    if not examples:
        raise ValueError("Missing fixed first 0 dB example")
    return rows, examples


def real_evaluation(models, legacy_summary):
    from evaluate_ppg_reconstruction_v2 import calibrate_delay
    from run_ppg_acc_waveform_reconstruction_v0 import load_mapping_rows, window_activity

    mappings = load_mapping_rows()
    legacy_rows = read_csv(V2 / "evaluation/real_windows.csv")
    expected = {(int(r["subject"]), float(r["start_s"]))
                for r in legacy_rows if r["method"] == "input"}
    if len(expected) != 6682:
        raise ValueError("Historical V2 window set differs from the agreed 6682 windows")
    rows, examples, calibrations, source_hashes, seen = [], [], {}, {}, set()
    found_fixed = set()
    for sid in (14, 15):
        source = DALIA / f"S{sid}/S{sid}.pkl"
        source_hashes[str(sid)] = sha(source)
        if source_hashes[str(sid)] != legacy_summary["real_source_sha256"][str(sid)]:
            raise ValueError(f"Historical real source changed: S{sid}")
        data = load_dalia(sid)
        wrist = data["signal"]["wrist"]
        ppg = bandpass(np.asarray(wrist["BVP"], dtype=float).reshape(-1))
        acc = signal.resample_poly(np.asarray(wrist["ACC"], dtype=float), 2, 1, axis=0)
        rpeaks = np.asarray(data["rpeaks"], dtype=float) / 700
        if (len(rpeaks) < 30 or not np.isfinite(rpeaks).all()
                or np.any(np.diff(rpeaks) < 0) or rpeaks[0] < 0
                or rpeaks[-1] > len(ppg) / FS + 1):
            raise ValueError(f"Invalid ECG R-peak time base: S{sid}")
        keep = np.r_[True, np.diff(rpeaks) > 0]
        calibration = calibrate_delay(ppg, data["activity"], rpeaks[keep])
        calibration["duplicate_rpeaks_removed"] = int(np.count_nonzero(~keep))
        calibrations[str(sid)] = calibration
        batch, contexts = [], []
        for mapping in mappings:
            if int(mapping["subject_id"]) != sid:
                continue
            start = int(mapping["window_start_sample"])
            start_s = start / FS
            if start < N or start + 2 * N > len(ppg):
                continue
            if calibration["start_s"] is not None:
                if (start_s < calibration["end_s"] + 0.5
                        and start_s + 8 > calibration["start_s"] - 0.5):
                    continue
            key = (sid, start_s)
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate historical window: {key}")
            raw, refs = ppg[start:start + N], acc_features(acc[start:start + N])
            if not np.isfinite(raw).all() or not np.isfinite(refs).all() or np.std(raw) < 1e-8:
                raise ValueError(f"Invalid historical window: {key}")
            batch.append(np.column_stack(((raw - raw.mean()) / raw.std(), refs)).T)
            contexts.append(mapping)
            seen.add(key)
        x = np.asarray(batch, dtype=np.float32)
        for mapping, inputs, waves in zip(contexts, x, variants(models, x)):
            start_s = float(mapping["window_start_s"])
            activity = window_activity(np.asarray(data["activity"]).reshape(-1), start_s)
            hr = float(data["label"][int(mapping["hr_window_index"])])
            for method, wave in waves.items():
                rows.append(metric_row("historical_regression", sid, start_s,
                                       f"activity_{activity}", activity, hr,
                                       method, wave, inputs))
            key = (sid, start_s)
            if key in FIXED:
                if activity != FIXED[key]:
                    raise ValueError(f"Fixed example activity changed: {key}")
                found_fixed.add(key)
                examples.append({
                    "name": f"real_s{sid}_{start_s:g}s", "waves": waves,
                    "acc": inputs[1:].T, "reference_hr": hr,
                    "title": f"Historical S{sid} | {start_s:g} s | activity {activity}",
                })
    if seen != expected or found_fixed != set(FIXED):
        raise ValueError("Historical window set or fixed examples are incomplete")
    return rows, examples, calibrations, source_hashes


def aggregate(rows):
    results = []
    for scope, condition in sorted({(r["scope"], r["condition"]) for r in rows}):
        group = [r for r in rows if (r["scope"], r["condition"]) == (scope, condition)]
        subjects = sorted({r["subject"] for r in group})
        for band in ("all", "lt60", "40_50", "50_60", "ge60"):
            band_rows = [r for r in group if band == "all"
                         or (band == "lt60" and r["reference_hr"] < 60)
                         or r["hr_stratum"] == band]
            for subject in ("all", *subjects):
                for method in METHODS:
                    selected = [r for r in band_rows if r["method"] == method
                                and (subject == "all" or r["subject"] == subject)]
                    if not selected:
                        continue
                    for field in FIELDS:
                        values = np.asarray([r[field] for r in selected], dtype=float)
                        valid = values[np.isfinite(values)]
                        results.append({
                            "scope": scope, "condition": condition, "hr_stratum": band,
                            "subject": subject, "method": method, "metric": field,
                            "windows": len(selected), "valid": len(valid),
                            "coverage": len(valid) / len(selected),
                            "mean": float(valid.mean()) if len(valid) else "",
                            "p90": float(np.percentile(valid, 90)) if len(valid) else "",
                        })
    return results


def coverage(manifest):
    output = {}
    for split in ("train", "validation", "test"):
        unique = {(r["reference_subject"], r["reference_start_s"]): float(r["reference_hr"])
                  for r in manifest if r["split"] == split}
        low = [(key, hr) for key, hr in unique.items() if hr < 60]
        output[split] = {
            "references": len(unique), "subjects": len({key[0] for key in unique}),
            "lt60_references": len(low), "lt60_subjects": len({key[0] for key, _ in low}),
            "40_50_references": sum(40 <= hr < 50 for hr in unique.values()),
            "lt60_by_subject": {
                subject: sum(key[0] == subject and hr < 60 for key, hr in unique.items())
                for subject in sorted({key[0] for key in unique})
            },
        }
    return output


def acceptance(rows, counts):
    checks = []

    def check(name, selected, field, limit, direction="max", percentile=None):
        values = np.asarray([r[field] for r in selected], dtype=float)
        complete = bool(len(values) and np.isfinite(values).all() and np.isfinite(limit))
        value = (float(np.percentile(values, percentile)) if percentile is not None
                 else float(values.mean())) if complete else None
        passed = complete and (value <= limit if direction == "max" else value >= limit)
        checks.append({"name": name, "windows": len(values), "metric": field,
                       "value": value, "limit": float(limit) if np.isfinite(limit) else None,
                       "direction": direction, "passed": bool(passed)})

    def select(scope, subject, condition, method, band="all"):
        return [r for r in rows if r["scope"] == scope and r["subject"] == subject
                and r["condition"] == condition and r["method"] == method
                and (band == "all" or r["reference_hr"] < 60)]

    def mean(selected, field):
        values = np.asarray([r[field] for r in selected], dtype=float)
        return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan

    for subject in ("12", "13"):
        for band in ("all", "lt60"):
            tag = f"paired/S{subject}/{band}"
            clean = select("paired", subject, "identity", "frozen_v3", band)
            check(tag + "/identity_nrmse", clean, "nrmse",
                  THRESHOLDS["identity_nrmse_mean_max"])
            check(tag + "/identity_nrmse_p90", clean, "nrmse",
                  THRESHOLDS["identity_nrmse_p90_max"], percentile=90)
            check(tag + "/identity_correlation", clean, "correlation",
                  THRESHOLDS["identity_correlation_mean_min"], "min")
            check(tag + "/identity_f1", clean, "beat_f1",
                  THRESHOLDS["identity_beat_f1_mean_min"], "min")
            for condition in ("snr_10", "snr_0", "snr_-5"):
                selected = select("paired", subject, condition, "frozen_v3", band)
                baseline = select("paired", subject, condition, "input", band)
                check(tag + "/" + condition + "/gain", selected, "snr_gain_db",
                      THRESHOLDS["synthetic_snr_gain_mean_min_db"], "min")
                check(tag + "/" + condition + "/nrmse", selected, "nrmse",
                      mean(baseline, "nrmse"))
                check(tag + "/" + condition + "/f1", selected, "beat_f1",
                      mean(baseline, "beat_f1"), "min")
    for subject in ("14", "15"):
        selected = select("historical_regression", subject, "activity_1", "frozen_v3")
        baseline = select("historical_regression", subject, "activity_1", "input")
        previous = select("historical_regression", subject, "activity_1", "frozen_v2")
        tag = f"historical/S{subject}/sitting"
        for field, threshold in (
            ("peak_error", "sitting_peak_mae_margin_bpm"),
            ("peak_failure", "sitting_peak_failure_margin"),
            ("spectral_error", "sitting_spectral_mae_margin_bpm"),
        ):
            check(tag + "/" + field, selected, field,
                  mean(baseline, field) + THRESHOLDS[threshold])
        extra = "input_relative_extra_peak_fraction"
        check(tag + "/extra", selected, extra, THRESHOLDS["sitting_extra_peak_fraction_max"])
        check(tag + "/missing", selected, "input_relative_missing_peak_fraction",
              THRESHOLDS["sitting_missing_peak_fraction_max"])
        check(tag + "/extra_vs_v2", selected, extra,
              THRESHOLDS["sitting_extra_peak_ratio_to_v2_max"] * mean(previous, extra))
    # 每人每活动都比较，避免静息收益或其他活动的收益掩盖运动退步。
    for subject in ("14", "15"):
        for activity in range(2, 9):
            condition = f"activity_{activity}"
            selected = select("historical_regression", subject, condition, "frozen_v3")
            for comparator in ("input", "frozen_v2"):
                baseline = select("historical_regression", subject, condition, comparator)
                keys = lambda part: [r["reference_start_s"] for r in part]
                if keys(selected) != keys(baseline):
                    raise ValueError("Motion comparator window mismatch")
                for field in ("spectral_error", "peak_error", "spectral_failure", "peak_failure"):
                    threshold = ("motion_failure_margin" if field.endswith("failure")
                                 else "motion_hr_mae_margin_bpm")
                    tag = f"motion/S{subject}/{condition}/{comparator}/{field}"
                    check(tag, selected, field, mean(baseline, field) + THRESHOLDS[threshold])
    low = counts["test"]["lt60_by_subject"]
    qualified_subjects = sum(
        value >= THRESHOLDS["low_hr_test_references_per_subject_min"] for value in low.values())
    coverage_ok = qualified_subjects >= THRESHOLDS["low_hr_test_subjects_min"]
    checks.append({"name": "low_hr_test_coverage", "passed": bool(coverage_ok),
                   "minimum_subjects": THRESHOLDS["low_hr_test_subjects_min"],
                   "minimum_references_per_subject": THRESHOLDS["low_hr_test_references_per_subject_min"],
                   "observed": low})
    limited = counts["validation"]["40_50_references"] == 0
    passed = all(item["passed"] for item in checks)
    return {
        "status": ("criteria_met_coverage_limited" if limited else "criteria_met")
                  if passed else "criteria_not_met",
        "passed": passed, "coverage_limited": limited, "checks": checks,
        "motion_noninferiority_met": all(c["passed"] for c in checks
                                        if c["name"].startswith("motion/")),
        "coverage_limit": "Validation has no 40–50 bpm reference window." if limited else "",
    }


def plot_example(example):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(N) / FS
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for method, wave in example["waves"].items():
        axes[0].plot(time, wave, label=method, alpha=0.8)
    target = example.get("target")
    if target is not None:
        axes[0].plot(time, target, "k--", label="Screened E4 reference")
        for method, wave in example["waves"].items():
            axes[1].plot(time, wave - target, label=method)
        axes[1].set_ylabel("Output minus reference")
    else:
        for method in METHODS:
            wave = example["waves"][method]
            peaks = center_peaks(wave)
            axes[1].plot(time, wave, label=method, alpha=0.7)
            axes[1].scatter(peaks, wave[np.rint(peaks * FS).astype(int)], s=12)
        axes[1].set_ylabel("PPG and detected peaks")
    for column, label in enumerate(("ACC x", "ACC y", "ACC z", "ACC magnitude")):
        axes[2].plot(time, example["acc"][:, column], label=label)
    axes[0].set_ylabel("PPG / input std")
    axes[2].set(xlabel="Seconds within window", ylabel="Detrended ACC / axis std")
    for axis in axes:
        axis.legend(fontsize=8, ncol=2)
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 8)
    fig.suptitle(f"{example['title']} | ECG-derived HR {example['reference_hr']:.2f} bpm")
    fig.savefig(RESULT / f"{example['name']}.png", dpi=160)
    plt.close(fig)
    arrays = {f"wave_{name}": wave for name, wave in example["waves"].items()}
    arrays.update(acc=example["acc"], reference_hr=example["reference_hr"])
    if target is not None:
        arrays["target"] = target
    np.savez_compressed(RESULT / f"{example['name']}.npz", **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if not args.write_outputs:
        print(f"Frozen V2/V3 on shared V3 test inputs; output: {RESULT}")
        return
    if RESULT.exists():
        raise FileExistsError(RESULT)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    check_split_isolation()
    if json.loads((TRAIN / "state.json").read_text())["status"] != "complete":
        raise ValueError("V3 training is incomplete")
    legacy_summary = json.loads((V2 / "evaluation/summary.json").read_text())
    models, model_hashes = {}, {}
    for name, path, format_name in (
        ("frozen_v2", V2 / "training/model.pt", "ppg_reconstruction_v2"),
        ("frozen_v3", TRAIN / "model.pt", "ppg_reconstruction_v3"),
    ):
        model_hashes[name] = sha(path)
        bundle = torch.load(path, map_location="cpu")
        if bundle["format"] != format_name:
            raise ValueError(f"Unexpected checkpoint format: {name}")
        if name == "frozen_v2" and model_hashes[name] != legacy_summary["model_sha256"]:
            raise ValueError("Historical V2 checkpoint changed")
        if name == "frozen_v3" and bundle["config"]["pairs_config_sha256"] != sha(PAIRS / "config.json"):
            raise ValueError("V3 checkpoint/dataset mismatch")
        model = ResidualPPG()
        model.load_state_dict(bundle["model"])
        models[name] = model
    manifest = read_csv(PAIRS / "manifest.csv")
    paired, paired_examples = synthetic_evaluation(models, manifest)
    real, real_examples, calibrations, sources = real_evaluation(models, legacy_summary)
    counts = coverage(manifest)
    decision = acceptance(paired + real, counts)
    RESULT.mkdir(parents=True)
    write_json(RESULT / "state.json", {"status": "writing"})
    write_csv(RESULT / "paired_windows.csv", paired)
    write_csv(RESULT / "real_windows.csv", real)
    write_csv(RESULT / "metrics.csv", aggregate(paired + real))
    for example in paired_examples + real_examples:
        plot_example(example)
    import matplotlib.pyplot as plt
    history = read_csv(TRAIN / "history.csv")
    arrays = {field: np.asarray([float(r[field]) for r in history])
              for field in ("epoch", "train_l1", "validation_l1")}
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for field in ("train_l1", "validation_l1"):
        axis.plot(arrays["epoch"], arrays[field], label=field)
    axis.set(xlabel="Epoch (0 = identity)", ylabel="Mean absolute waveform error")
    axis.legend()
    fig.savefig(RESULT / "training_history.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(RESULT / "training_history.npz", **arrays)
    write_json(RESULT / "acceptance.json", {"thresholds": THRESHOLDS, **decision})
    write_json(RESULT / "summary.json", {
        "status": "completed_pre_research_v3", "acceptance_status": decision["status"],
        "paired_examples": len(paired) // len(METHODS), "real_windows": len(real) // len(METHODS),
        "coverage": counts, "coverage_limited": decision["coverage_limited"],
        "delay_calibration_for_historical_exclusions": calibrations,
        "model_sha256": model_hashes, "real_source_sha256": sources,
        "pairs_config_sha256": sha(PAIRS / "config.json"), "script_sha256": sha(Path(__file__)),
        "limits": [
            "E4 references are heuristically screened, not expert clean-waveform truth.",
            "Synthetic noise is unchanged in formulation, not identical noise realizations.",
            "687 training references and 860 updates differ from V2's 847 and 1060.",
            "A reference-source replacement does not isolate a causal sensor-domain effect.",
            "S14/S15 are previously inspected historical regressions, not an untouched test.",
            "Real relative extra/missing peaks compare output with input, not ECG beat truth.",
            "Real ECG beat timing is not scored; delay calibration only reproduces exclusions.",
            "Window summaries are weighted by window count; read every subject separately.",
            "Overlapping real HR windows are not independent observations; no CI is reported.",
            "Validation has no 40–50 bpm references; success cannot establish that subgroup.",
            "No real clean PPG target exists; HR improvement does not prove morphology recovery.",
            "No test-time alignment, amplitude fitting, output normalization or example selection.",
        ],
    })
    write_json(RESULT / "state.json", {"status": "complete"})
    print(f"Saved {RESULT}; acceptance={decision['status']}")


if __name__ == "__main__":
    main()
