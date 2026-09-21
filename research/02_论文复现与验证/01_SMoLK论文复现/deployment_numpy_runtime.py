#!/usr/bin/env python3
"""SMoLK 窗口核心的 NumPy/SciPy 参考运行时；不依赖 PyTorch。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

SAMPLES = 1920
LENGTHS = (192, 96, 64)


def numpy_probability(x, bundle):
    """执行 same-padding 互相关、偏置、ReLU、带符号求和和 sigmoid。"""
    if x.shape != (SAMPLES,) or x.dtype != np.float32:
        raise ValueError(f"Expected float32 ({SAMPLES},), got {x.shape} {x.dtype}")
    logits = np.zeros(SAMPLES, dtype=np.float32)
    for group, length in enumerate(LENGTHS, 1):
        kernels = np.asarray(bundle[f"k{group}"], dtype=np.float32)
        biases = np.asarray(bundle[f"b{group}"], dtype=np.float32)
        signs = np.asarray(bundle[f"sign{group}"], dtype=np.int8)
        if kernels.ndim != 2 or kernels.shape[1] != length:
            raise ValueError(f"Unexpected group {group} shape: {kernels.shape}")
        if biases.shape != (len(kernels),) or signs.shape != (len(kernels),):
            raise ValueError(f"Unexpected group {group} bias/sign shape")
        left = (length - 1) // 2
        right = length - 1 - left
        padded = np.pad(x, (left, right))
        for kernel, bias, sign in zip(kernels, biases, signs):
            activation = np.maximum(
                np.float32(0),
                np.correlate(padded, kernel, mode="valid") + bias,
            )
            logits += activation * np.float32(sign)
    return (np.float32(1) / (
        np.float32(1) + np.exp(-logits)
    )).astype(np.float32)


def postprocess(probability):
    if probability.shape != (SAMPLES,):
        raise ValueError(f"Expected ({SAMPLES},), got {probability.shape}")
    smooth = savgol_filter(
        probability,
        window_length=151,
        polyorder=3,
        mode="interp",
    ).astype(np.float32)
    return smooth, (smooth > np.float32(0.5)).astype(np.uint8)


def verify_directory(root):
    x = np.load(root / "input_normalized.npy", allow_pickle=False)
    expected_probability = np.load(
        root / "expected_probability.npy", allow_pickle=False
    )
    expected_smooth = np.load(
        root / "expected_savgol151_probability.npy", allow_pickle=False
    )
    expected_mask = np.load(root / "expected_mask.npy", allow_pickle=False)
    with np.load(root / "absorbed_model.npz", allow_pickle=False) as loaded:
        bundle = {name: loaded[name] for name in loaded.files}
    probability = numpy_probability(x, bundle)
    smooth, mask = postprocess(probability)
    result = {
        "runtime": "numpy_scipy_without_torch",
        "directory": str(root.resolve()),
        "probability_max_abs_difference": float(
            np.max(np.abs(probability - expected_probability))
        ),
        "savgol_max_abs_difference": float(
            np.max(np.abs(smooth - expected_smooth))
        ),
        "mask_mismatch_samples": int(np.count_nonzero(mask != expected_mask)),
    }
    if (
        result["probability_max_abs_difference"] != 0
        or result["savgol_max_abs_difference"] != 0
        or result["mask_mismatch_samples"] != 0
    ):
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs" / "deployment_replay_v0",
    )
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.directory), indent=2))


if __name__ == "__main__":
    main()
