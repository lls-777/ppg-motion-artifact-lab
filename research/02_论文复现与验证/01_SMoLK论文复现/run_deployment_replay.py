#!/usr/bin/env python3
"""导出 SMoLK 紧凑模型，并用脱离 PyTorch 的 NumPy 前向做固定回放。"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from deployment_numpy_runtime import numpy_probability, postprocess
from run_full import DATASETS, load_dataset
from run_pruning import CompactLearnedFilters
from run_smoke import COMMIT, LENGTHS, SAMPLES

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "pruning_seed0" / "recommended_compact_state.pt"
PRUNING_RESULT = ROOT / "outputs" / "pruning_seed0" / "result.json"
OUT_DIR = ROOT / "outputs" / "deployment_replay_v0"
THREADS = 1
WARMUP = 50
BENCHMARK_BLOCKS = 20
RUNS_PER_BLOCK = 100


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_compact_model():
    saved = torch.load(SOURCE, map_location="cpu")
    expected = ("smolk_compact_pruned_state_v1", COMMIT, 0)
    actual = (
        saved.get("format"),
        saved.get("upstream_commit"),
        saved.get("source_seed"),
    )
    if actual != expected or saved.get("configuration") != "code_cosine_moderate_1":
        raise ValueError(f"Unexpected compact checkpoint: {actual}")
    counts = [
        int(saved["model"][f"conv{group}.weight"].shape[0])
        for group in (1, 2, 3)
    ]
    model = CompactLearnedFilters(counts)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved, counts


def absorbed_arrays(model):
    arrays = {}
    for group in (1, 2, 3):
        conv = getattr(model, f"conv{group}")
        weights = getattr(model, f"w{group}")
        magnitude = weights.detach().abs().numpy()
        arrays[f"k{group}"] = (
            conv.weight.detach().numpy()[:, 0] * magnitude[:, None]
        ).astype(np.float32)
        arrays[f"b{group}"] = (
            conv.bias.detach().numpy() * magnitude
        ).astype(np.float32)
        arrays[f"sign{group}"] = (
            torch.sign(weights).detach().numpy().astype(np.int8)
        )
    return arrays


def benchmark(function):
    for _ in range(WARMUP):
        function()
    block_ms = []
    for _ in range(BENCHMARK_BLOCKS):
        started = time.perf_counter_ns()
        for _ in range(RUNS_PER_BLOCK):
            function()
        block_ms.append(
            (time.perf_counter_ns() - started) / RUNS_PER_BLOCK / 1e6
        )
    return {
        "median_ms": float(np.median(block_ms)),
        "p10_ms": float(np.percentile(block_ms, 10)),
        "p90_ms": float(np.percentile(block_ms, 90)),
        "blocks": BENCHMARK_BLOCKS,
        "runs_per_block": RUNS_PER_BLOCK,
    }


def main():
    final_path = OUT_DIR / "result.json"
    if final_path.exists():
        raise FileExistsError(f"Completed result already exists: {final_path}")
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    model, checkpoint, counts = load_compact_model()
    pruning_result = json.loads(PRUNING_RESULT.read_text(encoding="utf-8"))
    if pruning_result.get("recommended_configuration") != checkpoint["configuration"]:
        raise ValueError("Pruning result and compact checkpoint disagree")

    dalia_x, _ = load_dataset("DaLiA")
    input_array = np.ascontiguousarray(dalia_x[0], dtype=np.float32)
    if not np.isfinite(input_array).all():
        raise ValueError("Non-finite replay input")
    with torch.inference_mode():
        torch_probability = model(
            torch.from_numpy(input_array[None, None, :])
        ).numpy()[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUT_DIR / "input_normalized.npy"
    bundle_path = OUT_DIR / "absorbed_model.npz"
    probability_path = OUT_DIR / "expected_probability.npy"
    smooth_path = OUT_DIR / "expected_savgol151_probability.npy"
    mask_path = OUT_DIR / "expected_mask.npy"
    np.save(input_path, input_array, allow_pickle=False)
    np.savez(bundle_path, **absorbed_arrays(model))

    with np.load(bundle_path, allow_pickle=False) as loaded:
        bundle = {name: loaded[name] for name in loaded.files}
    numpy_output = numpy_probability(input_array, bundle)
    smooth, mask = postprocess(numpy_output)
    torch_smooth, torch_mask = postprocess(torch_probability.astype(np.float32))
    torch_numpy_max_abs = float(np.max(np.abs(torch_probability - numpy_output)))
    smooth_max_abs = float(np.max(np.abs(torch_smooth - smooth)))
    mask_mismatch = int(np.count_nonzero(torch_mask != mask))
    if torch_numpy_max_abs > 3e-6 or smooth_max_abs > 3e-6 or mask_mismatch:
        raise AssertionError(
            f"Reference mismatch: {torch_numpy_max_abs}, "
            f"{smooth_max_abs}, {mask_mismatch}"
        )
    np.save(probability_path, numpy_output, allow_pickle=False)
    np.save(smooth_path, smooth, allow_pickle=False)
    np.save(mask_path, mask, allow_pickle=False)

    reloaded_input = np.load(input_path, allow_pickle=False)
    reloaded_probability = np.load(probability_path, allow_pickle=False)
    replay_probability = numpy_probability(reloaded_input, bundle)
    replay_max_abs = float(
        np.max(np.abs(replay_probability - reloaded_probability))
    )
    if replay_max_abs != 0:
        raise AssertionError(f"Saved replay is not exact: {replay_max_abs}")

    files = [
        input_path,
        bundle_path,
        probability_path,
        smooth_path,
        mask_path,
    ]
    resources = pruning_result["configurations"][checkpoint["configuration"]][
        "resources"
    ]
    longest_right_padding = max(length - 1 - (length - 1) // 2 for length in LENGTHS)
    sg_future = 151 // 2
    result = {
        "status": "window_core_deployment_replay_v0",
        "source": {
            "upstream_commit": COMMIT,
            "checkpoint": str(SOURCE),
            "configuration": checkpoint["configuration"],
            "source_seed": checkpoint["source_seed"],
            "kept_indices": checkpoint["kept_indices"],
        },
        "input_contract": {
            "signal": "single-channel PPG after the upstream 0.9-5 Hz bandpass",
            "sampling_rate_hz": 64,
            "window_seconds": 30,
            "samples": SAMPLES,
            "dtype": "float32",
            "normalization": "per-window mean/std, NumPy ddof=0",
            "units_after_normalization": "standard deviations",
            "fixed_case": {
                "dataset": "PPG-DaLiA test",
                "processed_array_row": 0,
                "processed_array_git_blob_sha1":
                    DATASETS["DaLiA"]["files"]["x"][2],
            },
        },
        "model_contract": {
            "kernel_counts_long_moderate_short": counts,
            "kernel_lengths_samples": list(LENGTHS),
            "operation": "zero-padded cross-correlation -> bias -> ReLU -> "
                         "signed sum -> sigmoid",
            "padding_left_samples": [(length - 1) // 2 for length in LENGTHS],
            "padding_right_samples": [
                length - 1 - (length - 1) // 2 for length in LENGTHS
            ],
            "postprocess": "SciPy Savitzky-Golay 151 samples, order 3, "
                           "mode=interp; artifact if >0.5",
        },
        "equivalence": {
            "torch_vs_absorbed_numpy_max_abs_probability": torch_numpy_max_abs,
            "torch_vs_numpy_max_abs_savgol_probability": smooth_max_abs,
            "torch_vs_numpy_mask_mismatch_samples": mask_mismatch,
            "saved_bundle_replay_max_abs_probability": replay_max_abs,
        },
        "output_summary": {
            "probability_min": float(numpy_output.min()),
            "probability_max": float(numpy_output.max()),
            "artifact_samples": int(mask.sum()),
            "artifact_fraction": float(mask.mean()),
        },
        "resources_from_pruning_result": resources,
        "streaming_constraints": {
            "longest_convolution_right_context_samples": longest_right_padding,
            "savgol_future_probability_samples": sg_future,
            "combined_future_input_samples_for_exact_centered_output":
                longest_right_padding + sg_future,
            "combined_future_seconds":
                (longest_right_padding + sg_future) / 64,
            "window_normalization": "requires the complete 30 s window or an "
                                    "equivalent buffered/two-pass design",
        },
        "computer_replay_benchmark": {
            "scope": "single-thread NumPy; input already normalized; batch size 1",
            "probability_only": benchmark(
                lambda: numpy_probability(input_array, bundle)
            ),
            "probability_savgol_mask": benchmark(
                lambda: postprocess(numpy_probability(input_array, bundle))
            ),
            "machine": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "excluded_from_this_replay": [
            "sensor acquisition and device-specific raw units",
            "0.9-5 Hz bandpass implementation and filter-state boundary policy",
            "strict streaming without future samples",
            "C conversion and Python/C numerical tolerance",
            "MCU toolchain, Flash/RAM peak, latency, energy, and power",
        ],
    }
    final_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": str(final_path),
        "torch_numpy_max_abs": torch_numpy_max_abs,
        "mask_mismatch": mask_mismatch,
        "replay_max_abs": replay_max_abs,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
