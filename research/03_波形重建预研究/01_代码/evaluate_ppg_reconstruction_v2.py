#!/usr/bin/env python3
"""冻结模型后评价配对波形恢复与真实运动 HR；ECG 只用于评价。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

from prepare_ppg_reconstruction_pairs_v2 import (
    FS, N, OUT, PAIRS, ROOT, acc_features, bandpass, load_dalia,
    peak_hr, peak_times, sha, write_csv, write_json,
)
from train_ppg_reconstruction_v2 import TRAIN, ResidualPPG, load_split, predict
# 与前两版保持相同的基线心率计算实现。
sys.path.insert(0, str(ROOT.parent / "01_自主基线研究/01_代码"))
from ppg_hr_baseline import estimate_window_hr
from run_ppg_acc_waveform_reconstruction_v0 import (
    load_mapping_rows, nlms_cancel, window_activity,
)

RESULT = OUT / "evaluation"
METHODS = ("input", "bandpass_0.9_5", "acc_nlms", "residual_model")


def variants(model, x):
    predicted = predict(model, x)[:, 0]
    for i, inputs in enumerate(x):
        waveform, refs = inputs[0], inputs[1:].T
        yield {
            "input": waveform,
            "bandpass_0.9_5": bandpass(waveform, (0.9, 5.0)),
            "acc_nlms": nlms_cancel(waveform, refs)[0],
            "residual_model": predicted[i],
        }


def match_beats(reference, detected, tolerance=0.10):
    """有序一对一匹配；仅连续匹配的相邻搏动用于间期误差。"""
    i = j = 0
    matched = []
    while i < len(reference) and j < len(detected):
        delta = detected[j] - reference[i]
        if abs(delta) <= tolerance:
            matched.append((i, j))
            i, j = i + 1, j + 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    count = len(matched)
    timing = [abs(detected[j] - reference[i]) * 1000 for i, j in matched]
    ibi = [abs((detected[j1] - detected[j0]) - (reference[i1] - reference[i0])) * 1000
           for (i0, j0), (i1, j1) in zip(matched, matched[1:])
           if i1 == i0 + 1 and j1 == j0 + 1]
    return {
        "beat_tp": count, "beat_fp": len(detected) - count,
        "beat_fn": len(reference) - count,
        "beat_precision": count / len(detected) if len(detected) else 0.0,
        "beat_recall": count / len(reference) if len(reference) else np.nan,
        "beat_f1": 2 * count / (len(reference) + len(detected))
                   if len(reference) + len(detected) else np.nan,
        "beat_timing_mae_ms": float(np.mean(timing)) if timing else np.nan,
        "matched_ibi_count": len(ibi),
        "matched_ibi_mae_ms": float(np.mean(ibi)) if ibi else np.nan,
    }


def evaluate_wave(wave, reference_hr, reference_beats):
    hr = float(estimate_window_hr(wave)[0])
    phr = peak_hr(wave)
    result = {"spectral_hr": hr, "peak_hr": phr, "reference_hr": reference_hr,
              "spectral_error": abs(hr - reference_hr), "peak_error": abs(phr - reference_hr)}
    detected = peak_times(wave)
    detected = detected[(detected >= 0.5) & (detected < 7.5)]
    if reference_beats is None:
        result.update({k: np.nan for k in match_beats([], [])})
    else:
        reference_beats = reference_beats[(reference_beats >= 0.5) & (reference_beats < 7.5)]
        result.update(match_beats(reference_beats, detected))
    return result


def calibrate_delay(ppg, activity, rpeaks):
    """最早足够长的静坐段校准一次；所有方法共享并排除重叠评价窗口。"""
    sitting = np.asarray(activity).reshape(-1) == 1
    edges = np.diff(np.r_[False, sitting, False].astype(int))
    for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (last - first) / 4 < 80:
            continue
        start, end = first / 4 + 8, first / 4 + 68
        detected = peak_times(ppg[int(start * FS):int(end * FS)]) + start
        reference = rpeaks[(rpeaks >= start + 0.5) & (rpeaks < end - 1.0)]
        detected = detected[(detected >= start + 0.5) & (detected < end - 0.5)]
        if len(reference) < 30:
            return {"start_s": start, "end_s": end, "delay_s": None}
        scores = []
        for delay in np.arange(0.10, 0.501, 0.01):
            score = match_beats(reference + delay, detected, tolerance=0.06)
            scores.append((score["beat_f1"],
                           -score["beat_timing_mae_ms"] if score["beat_tp"] else -np.inf,
                           float(delay)))
        best = max(scores)
        detail = match_beats(reference + best[2], detected, tolerance=0.06)
        return {"start_s": start, "end_s": end, "reference_beats": len(reference),
                "matched_beats": detail["beat_tp"],
                "delay_s": best[2] if best[0] >= 0.8 else None, "calibration_f1": best[0]}
    return {"start_s": None, "end_s": None, "delay_s": None}


def synthetic_evaluation(model):
    x, y = load_split("test")
    with (PAIRS / "manifest.csv").open() as stream:
        metadata = [r for r in csv.DictReader(stream) if r["split"] == "test"]
    if len(metadata) != len(x):
        raise ValueError("Test manifest mismatch")
    rows, example = [], None
    for i, waves in enumerate(variants(model, x)):
        meta, target = metadata[i], y[i, 0]
        beats, hr = peak_times(target), peak_hr(target)
        input_mse = float(np.mean((x[i, 0] - target)**2))
        for method, wave in waves.items():
            mse = float(np.mean((wave - target)**2))
            row = {"scope": "synthetic", "group": meta["condition"], "record": meta["record"],
                   "start_s": meta["start_s"], "method": method, "beat_evaluation_window": True,
                   "nrmse": np.sqrt(mse) / max(float(np.std(target)), 1e-8),
                   "correlation": float(np.corrcoef(wave, target)[0, 1]) if np.std(wave) > 1e-8 else np.nan,
                   "snr_gain_db": 10 * np.log10(input_mse / max(mse, 1e-12))
                                  if input_mse > 1e-12 else np.nan}
            row.update(evaluate_wave(wave, hr, beats))
            rows.append(row)
        # 第一条 0 dB 测试样本固定展示，不按重建效果挑图。
        if example is None and meta["condition"] == "snr_0":
            example = {"target": target, "waves": waves, "acc": x[i, 1:].T,
                       "record": meta["record"], "start_s": meta["start_s"]}
    if example is None:
        raise ValueError("Missing fixed synthetic example")
    return rows, example


def real_evaluation(model):
    mappings = load_mapping_rows()
    rows, calibrations, sources = [], {}, {}
    example = None
    for sid in (14, 15):
        data = load_dalia(sid)
        wrist = data["signal"]["wrist"]
        ppg = bandpass(np.asarray(wrist["BVP"], dtype=float).reshape(-1))
        acc = signal.resample_poly(np.asarray(wrist["ACC"], dtype=float), 2, 1, axis=0)
        rpeaks = np.asarray(data["rpeaks"], dtype=float) / 700
        if (len(rpeaks) < 30 or not np.isfinite(rpeaks).all()
                or np.any(np.diff(rpeaks) < 0) or rpeaks[0] < 0
                or rpeaks[-1] > len(ppg) / FS + 1):
            raise ValueError(f"Invalid ECG R-peak time base: S{sid}")
        # S14 原始标注有一个重复 R 峰；仅评价时去重，保留原始数据并记录数量。
        keep = np.r_[True, np.diff(rpeaks) > 0]
        duplicate_rpeaks = int(np.count_nonzero(~keep))
        rpeaks = rpeaks[keep]
        calibration = calibrate_delay(ppg, data["activity"], rpeaks)
        calibration["duplicate_rpeaks_removed"] = duplicate_rpeaks
        calibrations[str(sid)] = calibration
        path = ROOT.parent / f"90_共享原始数据/ppg_dalia/PPG_FieldStudy/S{sid}/S{sid}.pkl"
        sources[str(sid)] = sha(path)
        batch, contexts = [], []
        for mapping in mappings:
            if int(mapping["subject_id"]) != sid:
                continue
            start = int(mapping["window_start_sample"])
            start_s = start / FS
            if start < N or start + 2 * N > len(ppg):
                continue
            if calibration["start_s"] is not None:
                if start_s < calibration["end_s"] + 0.5 and start_s + 8 > calibration["start_s"] - 0.5:
                    continue
            raw = ppg[start:start + N]
            refs = acc_features(acc[start:start + N])
            if not np.isfinite(raw).all() or not np.isfinite(refs).all() or np.std(raw) < 1e-8:
                raise ValueError(f"Invalid evaluation input S{sid} at {start_s}")
            batch.append(np.column_stack(((raw - raw.mean()) / raw.std(), refs)).T)
            contexts.append(mapping)
        for mapping, waves, inputs in zip(contexts, variants(model, np.asarray(batch, dtype=np.float32)), batch):
            start_s = float(mapping["window_start_s"])
            activity = window_activity(np.asarray(data["activity"]).reshape(-1), start_s)
            hr = float(data["label"][int(mapping["hr_window_index"])])
            delay = calibration["delay_s"]
            beat_eligible = int(mapping["hr_window_index"]) % 4 == 0
            reference = None if delay is None or not beat_eligible else rpeaks + delay - start_s
            reference = None if reference is None else reference[(reference >= 0) & (reference < 8)]
            for method, wave in waves.items():
                row = {"scope": "real_motion", "group": f"S{sid}_activity_{activity}",
                       "subject": sid, "start_s": start_s, "method": method,
                       "beat_evaluation_window": beat_eligible,
                       "change_rmse": float(np.sqrt(np.mean((wave - inputs[0])**2)))}
                row.update(evaluate_wave(wave, hr, reference))
                rows.append(row)
            if example is None and sid == 14 and activity == 2 and beat_eligible:
                example = {"waves": waves, "acc": inputs[1:].T, "reference": reference,
                           "reference_hr": hr, "start_s": start_s}
        print(f"real S{sid}: {len(contexts)} windows, delay={calibration}", flush=True)
    if example is None:
        raise ValueError("Missing fixed stairs example")
    return rows, example, calibrations, sources


def aggregate(rows):
    """所有均值附有效数；无输出不能靠排除失败窗口变成高分。"""
    results = []
    for scope in sorted({r["scope"] for r in rows}):
        scoped = [r for r in rows if r["scope"] == scope]
        for group in ["all"] + sorted({r["group"] for r in scoped}):
            for method in METHODS:
                selected = [r for r in scoped if r["method"] == method
                            and (group == "all" or r["group"] == group)]
                fields = ("nrmse", "correlation", "snr_gain_db", "change_rmse",
                          "spectral_error", "peak_error", "beat_precision", "beat_recall", "beat_f1",
                          "beat_timing_mae_ms", "matched_ibi_mae_ms")
                for field in fields:
                    evaluated = [r for r in selected if r["beat_evaluation_window"]] if (
                        field.startswith("beat_") or field.startswith("matched_")) else selected
                    values = np.array([r.get(field, np.nan) for r in evaluated])
                    valid = values[np.isfinite(values)]
                    results.append({"scope": scope, "group": group, "method": method,
                                    "metric": field, "windows": len(evaluated),
                                    "valid": len(valid),
                                    "coverage": len(valid) / len(evaluated) if evaluated else 0,
                                    "mean": float(valid.mean()) if len(valid) else "",
                                    "p90": float(np.percentile(valid, 90)) if len(valid) else "",
                                    "failure_or_missing_gt10_fraction": (
                                        float(np.mean(~np.isfinite(values) | (values > 10)))
                                        if field in ("spectral_error", "peak_error") and len(values) else "")})
    return results


def plot_example(example, name, paired):
    t = np.arange(N) / FS
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for method, wave in example["waves"].items():
        axes[0].plot(t, wave, label=method, alpha=0.8)
    if paired:
        axes[0].plot(t, example["target"], "k--", label="Band-limited reference")
        for method, wave in example["waves"].items():
            axes[1].plot(t, wave - example["target"], label=method, alpha=0.8)
        axes[1].set_ylabel("Output minus reference")
    else:
        enhanced = example["waves"]["residual_model"]
        axes[1].plot(t, enhanced, label="Residual model")
        peaks = peak_times(enhanced)
        axes[1].scatter(peaks, enhanced[np.rint(peaks * FS).astype(int)], c="black", s=14,
                        label="Detected PPG peaks")
        if example["reference"] is not None:
            for k, peak in enumerate(example["reference"]):
                axes[1].axvline(peak, color="green", alpha=0.35,
                               label="ECG R + fixed delay" if k == 0 else None)
        axes[1].set_ylabel("Enhanced PPG (input units)")
    for column, label in enumerate(("ACC x", "ACC y", "ACC z", "ACC magnitude")):
        axes[2].plot(t, example["acc"][:, column], label=label, alpha=0.8)
    axes[0].set_ylabel("PPG in input std units")
    axes[2].set_ylabel("Detrended ACC / axis std")
    axes[2].set_xlabel("Seconds within 8 s window")
    for ax in axes:
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 8)
    label = f"BIDMC {example['record']}" if paired else "PPG-DaLiA S14"
    fig.suptitle(f"{label} | {name} | source start {example['start_s']} s")
    fig.savefig(RESULT / f"{name}.png", dpi=160)
    plt.close(fig)
    arrays = {f"wave_{k}": v for k, v in example["waves"].items()}
    arrays.update(acc=example["acc"])
    if paired:
        arrays["target"] = example["target"]
    elif example["reference"] is not None:
        arrays["ecg_reference_times_s"] = example["reference"]
    np.savez_compressed(RESULT / f"{name}.npz", **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if not args.write_outputs:
        print(f"Frozen model: {TRAIN / 'model.pt'}; evaluation: {RESULT}")
        return
    if RESULT.exists():
        raise FileExistsError(RESULT)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if json.loads((TRAIN / "state.json").read_text())["status"] != "complete":
        raise ValueError("Training incomplete")
    bundle = torch.load(TRAIN / "model.pt", map_location="cpu")
    if bundle["format"] != "ppg_reconstruction_v2":
        raise ValueError("Unknown checkpoint")
    if bundle["config"]["pairs_config_sha256"] != sha(PAIRS / "config.json"):
        raise ValueError("Checkpoint/dataset mismatch")
    model = ResidualPPG()
    model.load_state_dict(bundle["model"])
    synthetic, synthetic_example = synthetic_evaluation(model)
    real, real_example, delays, sources = real_evaluation(model)
    RESULT.mkdir(parents=True)
    write_json(RESULT / "state.json", {"status": "writing"})
    write_csv(RESULT / "synthetic_windows.csv", synthetic)
    write_csv(RESULT / "real_windows.csv", real)
    write_csv(RESULT / "metrics.csv", aggregate(synthetic + real))
    plot_example(synthetic_example, "synthetic_fixed_example", paired=True)
    plot_example(real_example, "real_s14_stairs", paired=False)
    with (TRAIN / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for field in ("train_l1", "validation_l1"):
        ax.plot([int(r["epoch"]) for r in history],
                [float(r[field]) for r in history], label=field)
    ax.set(xlabel="Epoch (0 = identity)", ylabel="Mean absolute waveform error")
    ax.legend()
    fig.savefig(RESULT / "training_history.png", dpi=160)
    plt.close(fig)
    write_json(RESULT / "summary.json", {
        "status": "completed_pre_research_v2", "selected_epoch": bundle["config"]["best_epoch"],
        "synthetic_examples": len(synthetic) // len(METHODS),
        "fixed_examples": {"bidmc_record": synthetic_example["record"],
                           "bidmc_start_s": synthetic_example["start_s"],
                           "dalia_subject": 14, "dalia_start_s": real_example["start_s"]},
        "real_windows": len(real) // len(METHODS), "delay_calibration": delays,
        "model_sha256": sha(TRAIN / "model.pt"), "real_source_sha256": sources,
        "script_sha256": sha(Path(__file__)),
        "limits": [
            "Synthetic ACC corruption is a simulation, not measured wrist artifact.",
            "BIDMC reference is screened and band-limited, not an expert clean target.",
            "Pair beat targets are peaks detected on reference PPG, not ECG annotations.",
            "S14/S15 were inspected in previous experiments; this is exploratory.",
            "Real ECG delay uses excluded resting calibration, identical for every method.",
            "No real-motion clean PPG target: HR/beat improvement does not prove morphology recovery.",
            "Matched-IBI error is conditional; read together with beat F1 and matched counts.",
            "Rest-to-motion pulse-arrival delay variation contributes to real timing errors.",
            "Real beat metrics use HR index % 4 == 0; HR uses all eligible mapped windows.",
            "Common 0.5-8 Hz input differs from historical SMoLK narrow-band experiments.",
            "Offline filters/window normalization/convolutions are noncausal.",
        ]})
    (RESULT / "README.md").write_text(
        "# V2 图与指标说明\n\n"
        "合成例图固定取测试集首个 0 dB 样本；真实例图固定取 S14 首个可评价上下楼窗口。\n"
        "三层横轴均为采样点序号 / 64，即窗口内秒数。\n"
        "第一层：所有 PPG 共用输入的均值和标准差，u=(x-mu_input)/sd_input；"
        "参考目标也用同一个变换，重建结果没有另行标准化。单位不是伏特。\n"
        "合成图第二层：每个输出减参考波形；0 才表示逐点吻合。\n"
        "真实图第二层：模型波形和检出峰；绿线是 ECG R 峰加固定校准延迟，"
        "不是干净 PPG 波形，也不是训练输入。校准失败时不画绿线、不报相应搏动指标。\n"
        "第三层：ACC 三轴及模长，各列去线性趋势再除自身标准差，无 g 单位。\n"
        "训练图：横轴训练轮数，纵轴平均绝对波形误差；0 是恒等输出。"
        "训练值为各批更新前损失的加权均值，验证值为该轮结束后的损失。\n\n"
        "NRMSE=RMSE(output,target)/std(target)；相关系数是去均值后的归一化内积。\n"
        "SNR增益=10log10(MSE(input,target)/MSE(output,target))，未污染样本不定义增益。\n"
        "HR 沿用固定频谱估计，并独立报告峰间期估计。真实参考为 ECG 导出的8秒标签。\n"
        "搏动匹配容差100ms，一对一；只评价窗口中间0.5–7.5秒。"
        "F1=2TP/(2TP+FP+FN)；时间误差和间期误差单位为ms。\n"
        "真实静坐校准只用未增强PPG：最早至少80秒静坐段中的第8–68秒，"
        "搜索100–500ms固定延迟，校准F1至少0.8；校准段及两侧0.5秒重叠窗口排除。\n"
        "真实搏动评价只用HR索引%4==0的不重叠子集；HR用全部合格窗口。\n"
        "搏动指标是逐窗口均值；固定延迟校准后误差仍含运动时生理延迟变化。\n"
        "metrics.csv 每个均值附有效数和覆盖率；超过10bpm的失败率把无输出计为失败。\n"
        "没有对测试输出做平移、幅度拟合或选择最好看的图。\n",
        encoding="utf-8")
    write_json(RESULT / "state.json", {"status": "complete"})
    print(f"Saved {RESULT}")


if __name__ == "__main__":
    main()
