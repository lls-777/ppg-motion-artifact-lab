#!/usr/bin/env python3
"""ACC 修正的固定回归：冻结 V3，不训练，不覆盖历史实验。"""
from __future__ import annotations

import argparse
import io
import json
import unittest
from pathlib import Path

import numpy as np
import scipy
import torch

from acc_preprocessing_v1 import CORRECTED_MODE, resample_acc
from prepare_ppg_reconstruction_pairs_v2 import (
    FS, N, ROOT, DALIA, acc_features, bandpass, load_dalia, sha, write_csv, write_json,
)
from train_ppg_reconstruction_v2 import ResidualPPG, predict

CODE = Path(__file__).resolve().parent
V3 = ROOT / "02_结果/V3"
OUT = ROOT / "02_结果/ACC预处理修正V1"
METHODS = ("input", "legacy", "dc_corrected", "zero_acc")
FIXED = {(14, 262.0), (15, 126.0)}
FIELDS = ("peak_error", "spectral_error", "peak_failure", "spectral_failure",
          "input_relative_extra_peak_fraction", "input_relative_missing_peak_fraction",
          "change_rmse")


def numeric_checks():
    import test_acc_preprocessing_v1 as tests

    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    return {"passed": True, "tests": result.testsRun, "log": stream.getvalue()}


def collect(model, historical):
    from evaluate_ppg_reconstruction_v3 import metric_row, read_csv

    saved = read_csv(V3 / "evaluation/real_windows.csv")
    previous = {(r["subject"], float(r["reference_start_s"]), r["method"]): r
                for r in saved}
    contexts = [r for r in saved if r["method"] == "input"]
    if len(previous) != 3 * 6682 or len(contexts) != 6682:
        raise ValueError("Historical rows changed or contain duplicates")
    rows, examples, source_hashes = [], {}, {}
    for sid in (14, 15):
        source = DALIA / f"S{sid}/S{sid}.pkl"
        source_hashes[str(sid)] = sha(source)
        if source_hashes[str(sid)] != historical["real_source_sha256"][str(sid)]:
            raise ValueError("Raw source changed")
        data = load_dalia(sid)
        wrist = data["signal"]["wrist"]
        raw_acc = np.asarray(wrist["ACC"], dtype=float)
        ppg = bandpass(np.asarray(wrist["BVP"], dtype=float).reshape(-1))
        acc = {name: resample_acc(raw_acc, dc_normalize=mode)
               for name, mode in (("legacy", False), ("dc_corrected", True))}
        selected = [r for r in contexts if r["subject"] == str(sid)]
        batches = {name: [] for name in METHODS[1:]}
        for meta in selected:
            start_s = float(meta["reference_start_s"])
            first = int(round(start_s * FS))
            if first < N or first + 2 * N > len(ppg) or first % 128:
                raise ValueError("Historical window coordinate changed")
            if float(meta["reference_hr"]) != float(data["label"][first // 128]):
                raise ValueError("Historical ECG HR label changed")
            raw = ppg[first:first + N]
            u = (raw - raw.mean()) / raw.std()
            features = {name: acc_features(a[first:first + N]) for name, a in acc.items()}
            features["zero_acc"] = np.zeros_like(features["legacy"])
            for name in batches:
                batches[name].append(np.column_stack((u, features[name])).T)
            if (sid, start_s) in FIXED:
                examples[(sid, start_s)] = {
                    "raw_acc_32hz": raw_acc[first // 2:(first + N) // 2],
                    "legacy_acc_64hz": acc["legacy"][first:first + N],
                    "corrected_acc_64hz": acc["dc_corrected"][first:first + N],
                    "legacy_features": features["legacy"].astype(np.float32),
                    "corrected_features": features["dc_corrected"].astype(np.float32),
                    "zero_features": features["zero_acc"].astype(np.float32),
                    "reference_hr": float(meta["reference_hr"]),
                    "subject": sid, "start_s": start_s,
                }
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in batches.items()}
        for values in arrays.values():
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite model inputs")
            np.testing.assert_array_equal(values[:, 0], arrays["legacy"][:, 0])
        predicted = {name: predict(model, values)[:, 0] for name, values in arrays.items()}
        for i, meta in enumerate(selected):
            start_s = float(meta["reference_start_s"])
            activity, hr = int(meta["activity"]), float(meta["reference_hr"])
            waves = {"input": arrays["legacy"][i, 0],
                     **{name: value[i] for name, value in predicted.items()}}
            for name, wave in waves.items():
                row = metric_row("acc_dc_diagnostic", sid, start_s, f"activity_{activity}",
                                 activity, hr, name, wave, arrays["legacy"][i])
                if name in ("input", "legacy"):
                    old_name = "input" if name == "input" else "frozen_v3"
                    prior = previous[(str(sid), start_s, old_name)]
                    for field in FIELDS:
                        if not np.isclose(row[field], float(prior[field]),
                                          atol=1e-7, rtol=0, equal_nan=True):
                            raise ValueError(f"Historical reproduction mismatch: {field}")
                rows.append(row)
            if (sid, start_s) in FIXED:
                example = examples[(sid, start_s)]
                example.update({f"wave_{name}": wave for name, wave in waves.items()})
                with np.load(V3 / f"evaluation/real_s{sid}_{start_s:g}s.npz") as old:
                    np.testing.assert_array_equal(
                        example["legacy_features"].astype(np.float32), old["acc"])
                    np.testing.assert_array_equal(waves["input"], old["wave_input"])
                    np.testing.assert_allclose(
                        waves["legacy"], old["wave_frozen_v3"], rtol=0, atol=1e-7)
        print(f"S{sid}: {len(selected)} identical windows verified", flush=True)
    if set(examples) != FIXED or len(rows) != 4 * 6682:
        raise ValueError("Incomplete fixed evaluation")
    return rows, examples, source_hashes


def aggregate(rows):
    results = []
    groups = ("all", "motion_2_8", *(f"activity_{a}" for a in range(9)))
    for subject in ("all", "14", "15"):
        for group in groups:
            for method in METHODS:
                selected = [r for r in rows if r["method"] == method
                            and (subject == "all" or r["subject"] == subject)
                            and (group == "all" or group == r["condition"]
                                 or group == "motion_2_8" and 2 <= r["activity"] <= 8)]
                if not selected:
                    continue
                for field in FIELDS:
                    values = np.asarray([r[field] for r in selected])
                    valid = values[np.isfinite(values)]
                    results.append({"subject": subject, "group": group, "method": method,
                                    "metric": field, "windows": len(values), "valid": len(valid),
                                    "coverage": len(valid) / len(values),
                                    "mean": float(valid.mean()) if len(valid) else "",
                                    "p90": float(np.percentile(valid, 90)) if len(valid) else ""})
    return results


def plot_example(example):
    import matplotlib.pyplot as plt
    from evaluate_ppg_reconstruction_v3 import center_peaks

    sid, start = example["subject"], example["start_s"]
    name = f"s{sid}_{start:g}s"
    np.savez_compressed(OUT / f"{name}.npz", **example)
    t, raw_t = np.arange(N) / FS, np.arange(N // 2) / 32
    mean_y = float(example["raw_acc_32hz"][:, 1].mean())
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].plot(raw_t, example["raw_acc_32hz"][:, 1] - mean_y, "k.", label="Raw 32 Hz")
    for key, label in (("legacy_acc_64hz", "Legacy"), ("corrected_acc_64hz", "DC corrected")):
        axes[0].plot(t, example[key][:, 1] - mean_y, label=label, alpha=0.8)
    axes[0].set_ylabel("ACC y minus raw mean (g)")
    for key, label in (("legacy_features", "Legacy"), ("corrected_features", "DC corrected")):
        axes[1].plot(t, example[key][:, 1], label=label)
    axes[1].set_ylabel("ACC y feature (unitless)")
    for method in ("input", "legacy", "dc_corrected"):
        wave = example[f"wave_{method}"]
        peaks = center_peaks(wave)
        axes[2].plot(t, wave, label=method, alpha=0.8)
        axes[2].scatter(peaks, wave[np.rint(peaks * FS).astype(int)], s=10)
    axes[2].set_ylabel("PPG / input std")
    for method in ("legacy", "dc_corrected"):
        axes[3].plot(t, example[f"wave_{method}"] - example["wave_input"], label=method)
    axes[3].set(xlabel="Seconds within 8 s window", ylabel="Output minus input")
    for axis in axes:
        axis.set_xlim(0, 8)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=3)
    fig.suptitle(f"S{sid} | {start:g} s | raw ACC y mean {mean_y:.8f} g")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


def write_readme(metrics):
    def value(subject, method, field):
        row = next(r for r in metrics if r["subject"] == subject and r["group"] == "activity_1"
                   and r["method"] == method and r["metric"] == field)
        return f"{row['mean']:.3f}" if row["mean"] != "" else "不可用"

    lines = [
        "# ACC 预处理修正 V1：冻结模型回归",
        "",
        "这是旧 V3 权重上的推理输入修正诊断；V3 原判定保留，不是新训练模型或波形重建验收。",
        "",
        "## 静息峰间心率 MAE（bpm，越低越好）",
        "",
        "| 受试者 | 输入 PPG | 原 V3 | DC 校正 | 全零 ACC 对照 |",
        "|---|---:|---:|---:|---:|",
    ]
    for sid in ("14", "15"):
        lines.append("| S" + sid + " | " + " | ".join(
            value(sid, m, "peak_error") for m in METHODS) + " |")
    lines.extend([
        "", "全部 6682 个窗口、每人每活动及 4416 个活动 2–8 窗口见 metrics.csv；",
        "两种 HR 的 >10 bpm 或缺失均计入失败率，不能只看有效窗口的 MAE。",
        "",
        "## 修正与边界",
        "",
        "L 为默认 32→64 Hz 多相重采样，g=L(全1数组)。修正版计算 L(ACC)/g，随后沿用原四路特征与标准化。",
        "这是单位常量响应校正，不改变原始 ACC 文件。完整记录先重采样，再切原定 8 秒窗，继续排除记录边缘。",
        "原 V2/V3 源码、数据与模型保持原样；新函数须由后续训练数据生成和推理代码显式共同采用，当前训练数据未切换。",
        "近恒定信号的真实微小噪声仍可能被标准化放大；本次没有增设静息门控或修改标准差下限。",
        "冻结模型原先采用旧预处理训练，因此不能直接代表用修正版重新训练的结果。",
        "全零 ACC 只用于检查输入依赖，不是部署选择，也不能据此推断 ACC 无用。",
        "真实 PPG 无干净形态真值，ECG 逐搏延迟校准此前失败。相对 input 的多检/漏检比例不是 ECG 真值误检率。",
        "S14/S15 已被多次观察，重叠窗口不独立；心率改善不等于逐搏位置或 HRV 恢复。",
        "",
        "## 图的来源与坐标",
        "",
        "两张图固定为 S14 的 262–270 秒、S15 的 126–134 秒，不按新结果挑选。",
        "四层横轴都是窗口内秒数，32 Hz 点为 n/32，64 Hz 点为 n/64。",
        "第一层：y轴原始/重采样数值减去原始32Hz窗口的y轴均值，单位g；减均值只便于显示微小波纹，均值写在图标题。",
        "第二层：重采样ACC y轴去线性趋势，再除以 max(本列标准差,1e-6)，无g单位；这里显示模型真正接收的y轴特征。",
        "第三层：输入PPG按本窗均值和标准差标准化，模型输出使用同一尺度；圆点为算法检出的峰，不是ECG真值。",
        "第四层：模型输出减输入PPG，同一标准差尺度；0表示该点没有被改变，不是与真实干净波形的误差。",
        "NPZ包含原始32Hz三轴ACC、两版64Hz三轴ACC、四路特征和所有对照波形，可核对图中每一个数值。",
        "",
    ])
    for name in ("s14_262s", "s15_126s"):
        lines.extend([f"![{name}](<{OUT / (name + '.png')}>)", ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    if not parser.parse_args().write_outputs:
        print(f"Frozen V3; no training; use --write-outputs to save to {OUT}")
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    checks = numeric_checks()
    historical = json.loads((V3 / "evaluation/summary.json").read_text())
    pairs_config = ROOT / "03_重建训练数据/ppg_reconstruction_pairs_v3/config.json"
    pairs = json.loads(pairs_config.read_text())
    model_path = V3 / "training/model.pt"
    if sha(model_path) != historical["model_sha256"]["frozen_v3"]:
        raise ValueError("Frozen V3 model changed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    bundle = torch.load(model_path, map_location="cpu")
    if bundle["format"] != "ppg_reconstruction_v3" or bundle["config"]["pairs_config_sha256"] != sha(pairs_config):
        raise ValueError("Frozen V3 checkpoint/dataset mismatch")
    expected = {
        "prepare_ppg_reconstruction_pairs_v2.py": pairs["v2_helper_sha256"],
        "prepare_ppg_reconstruction_pairs_v3.py": pairs["script_sha256"],
        "train_ppg_reconstruction_v2.py": bundle["config"]["v2_model_code_sha256"],
        "train_ppg_reconstruction_v3.py": bundle["config"]["script_sha256"],
        "evaluate_ppg_reconstruction_v3.py": historical["script_sha256"],
    }
    if any(sha(CODE / name) != digest for name, digest in expected.items()):
        raise ValueError("Historical source code changed")
    model = ResidualPPG()
    model.load_state_dict(bundle["model"])
    rows, examples, sources = collect(model, historical)
    metrics = aggregate(rows)
    OUT.mkdir(parents=True)
    write_json(OUT / "state.json", {"status": "writing"})
    write_json(OUT / "numeric_checks.json", checks)
    write_csv(OUT / "windows.csv", rows)
    write_csv(OUT / "metrics.csv", metrics)
    for example in examples.values():
        plot_example(example)
    write_readme(metrics)
    names = ("acc_preprocessing_v1.py", "test_acc_preprocessing_v1.py",
             "run_acc_preprocessing_regression_v1.py", "evaluate_ppg_reconstruction_v2.py")
    write_json(OUT / "summary.json", {
        "status": "diagnostic_complete_not_v3_acceptance", "mode": CORRECTED_MODE,
        "windows": 6682, "methods": METHODS, "training_performed": False,
        "legacy_reproduction_tolerance": 1e-7, "numeric_checks_passed": True,
        "model_sha256": sha(model_path), "raw_source_sha256": sources,
        "historical_source_sha256": expected,
        "code_sha256": {name: sha(CODE / name) for name in names},
        "historical_windows_sha256": sha(V3 / "evaluation/real_windows.csv"),
        "hr_baseline_code_sha256": sha(ROOT.parent / "01_自主基线研究/01_代码/ppg_hr_baseline.py"),
        "pairs_config_sha256": sha(pairs_config),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "resampling": "32->64 Hz; scipy default Kaiser 5.0, padtype constant, cval 0; L(acc)/L(ones)",
        "features": "xyz+magnitude; per-window linear detrend/std; floor 1e-6; unchanged",
        "limits": [
            "Inference intervention only; old training pairs and checkpoint remain unchanged.",
            "Relative input-peak agreement is not ECG timing or clean morphology truth.",
            "Previously observed subjects and overlapping windows; no population claim.",
            "Constant preservation does not prove arbitrary dynamic-signal fidelity.",
            "Current experiment does not rerun synthetic training/test acceptance.",
        ],
    })
    write_json(OUT / "state.json", {"status": "complete"})
    print(f"Saved {OUT}; diagnostic only, not a new V3 acceptance.")


if __name__ == "__main__":
    main()
