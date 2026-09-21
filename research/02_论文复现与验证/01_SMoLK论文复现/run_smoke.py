#!/usr/bin/env python3
"""SMoLK 12 核学习验证；不接入现有 HR 基线，不声明论文性能复现。

模型依据 Sully Chen 等人的 PPG Segmentation.ipynb，许可见 LICENSE-SMoLK。
固定读取作者训练集前 64 行、测试集前 8 行，CPU 上训练 32 次。
输出：outputs/smoke_run.json、outputs/segment_learning.png。
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import platform
import random
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import torch
from scipy.signal import savgol_filter
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

COMMIT = "4aa63d72d5fedb80bd5ba6a905dcc75f1cafd12a"
BASE_URL = f"https://raw.githubusercontent.com/SullyChen/SMoLK/{COMMIT}/PPG%20data"
SAMPLES, FS, STEPS, SEED = 1920, 64, 32, 0
LENGTHS = (192, 96, 64)
OUT = Path(__file__).resolve().parent / "outputs"


def load_prefix(split: str, name: str, rows: int) -> tuple[np.ndarray, dict]:
    """只读获取固定提交的 NPY 前缀；校验头与长度，不解序列化 pickle。"""
    url = f"{BASE_URL}/new_PPG_DaLiA_{split}/processed_dataset/{name}.npy"
    limit = 512 + rows * SAMPLES * 8
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{limit-1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 206):
            raise ValueError(f"Unexpected HTTP status: {response.status}")
        if response.status == 206 and not response.headers.get("Content-Range", "").startswith("bytes 0-"):
            raise ValueError("Response does not start at byte zero")
        content = response.read(limit)
    stream = io.BytesIO(content)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("Expected the pinned dataset's NPY v1 header")
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    expected_rows = {"train": 3436, "test": 869}[split]
    if shape != (expected_rows, SAMPLES) or fortran_order or dtype != np.dtype("<f8"):
        raise ValueError(f"Unexpected array metadata: {shape}, {dtype}, {fortran_order}")
    end = stream.tell() + rows * SAMPLES * dtype.itemsize
    if len(content) < end:
        raise ValueError("Incomplete NPY prefix")
    array = np.frombuffer(content[stream.tell():end], dtype=dtype).reshape(rows, SAMPLES).copy()
    if not np.isfinite(array).all():
        raise ValueError("Non-finite input")
    return array, {"url": url, "full_shape": list(shape), "rows_used": [0, rows],
                   "prefix_bytes_hashed": end, "prefix_sha256": hashlib.sha256(content[:end]).hexdigest()}


def normalize(x: np.ndarray) -> np.ndarray:
    """对应作者逐片段 np.std 默认 ddof=0；常量片段明确报错。"""
    std = x.std(axis=1, keepdims=True)
    if (std == 0).any():
        raise ValueError("Cannot standardize a constant segment")
    return ((x - x.mean(axis=1, keepdims=True)) / std).astype(np.float32)


class LearnedFilters(nn.Module):
    """与作者前向结构一致；每种核长度 4 个，共 12 个。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 192, padding="same")
        self.conv2 = nn.Conv1d(1, 4, 96, padding="same")
        self.conv3 = nn.Conv1d(1, 4, 64, padding="same")
        self.w1 = nn.Parameter(torch.zeros(4))
        self.w2 = nn.Parameter(torch.zeros(4))
        self.w3 = nn.Parameter(torch.zeros(4))

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        groups = [F.relu(getattr(self, f"conv{i}")(x)) * getattr(self, f"w{i}")[None, :, None]
                  for i in (1, 2, 3)]
        return torch.cat(groups, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.contributions(x).sum(dim=1))


def numpy_forward(model: LearnedFilters, x: np.ndarray) -> np.ndarray:
    """独立实现互相关、偶数核 same padding 和前向求和，用来核对公式。"""
    terms = []
    for i, length in enumerate(LENGTHS, 1):
        conv = getattr(model, f"conv{i}")
        kernels = conv.weight.detach().numpy()[:, 0].astype(np.float64)
        biases = conv.bias.detach().numpy()
        weights = getattr(model, f"w{i}").detach().numpy()
        left = (length - 1) // 2
        padded = np.pad(x.astype(np.float64), (left, length - 1 - left))
        for kernel, bias, weight in zip(kernels, biases, weights):
            terms.append(np.maximum(0, np.correlate(padded, kernel, mode="valid") + bias) * weight)
    logits = np.stack(terms).sum(axis=0)
    return 1 / (1 + np.exp(-logits))


def absorb_weights(model: LearnedFilters) -> LearnedFilters:
    """利用 ReLU 正齐次性吸收绝对值；保留符号，0 权重仍为 0。"""
    result = copy.deepcopy(model)
    with torch.no_grad():
        for i in (1, 2, 3):
            conv, weight = getattr(result, f"conv{i}"), getattr(result, f"w{i}")
            magnitude = weight.abs()
            conv.weight.mul_(magnitude[:, None, None])
            conv.bias.mul_(magnitude)
            weight.copy_(weight.sign())
    return result


def scores(y: np.ndarray, probability: np.ndarray) -> dict:
    """保持作者 >0.5、151 点三阶 SG、跨片段拼接后计算标签 1 的 F1。"""
    smooth = savgol_filter(probability, 151, 3, axis=-1)
    prediction = smooth > 0.5
    truth = y.astype(bool)
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    denom = 2 * tp + fp + fn
    dice = 2 * tp / denom if denom else 0.0
    library_f1 = f1_score(truth.ravel(), prediction.ravel(), zero_division=0)
    if not np.isclose(dice, library_f1, atol=1e-12):
        raise AssertionError("Independent Dice and sklearn F1 disagree")
    return {"dice_label1": dice, "tp": tp, "fp": fp, "fn": fn,
            "label1_fraction": float(y.mean()), "predicted1_fraction": float(prediction.mean())}


def main() -> None:
    if any((OUT / name).exists() for name in ("smoke_run.json", "segment_learning.png")):
        raise FileExistsError("Output already exists; preserve the previous run before rerunning")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arrays, provenance = {}, {}
    for split, rows in (("train", 64), ("test", 8)):
        for kind in ("scaled_ppgs", "seg_labels"):
            key = f"{split}_{kind}"
            arrays[key], provenance[key] = load_prefix(split, kind, rows)
    train_y, test_y = arrays["train_seg_labels"], arrays["test_seg_labels"]
    for labels in (train_y, test_y):
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Expected binary labels")
    if np.unique(train_y).size != 2:
        raise ValueError("Smoke training subset must include both labels")
    train_x = normalize(arrays["train_scaled_ppgs"])
    test_x = normalize(arrays["test_scaled_ppgs"])
    x, y = torch.from_numpy(train_x[:, None, :]), torch.from_numpy(train_y.astype(np.float32))
    tx = torch.from_numpy(test_x[:, None, :])
    # 在训练前固定选图：训练前缀里第一个同时含 0、1 的片段，不按预测效果挑选。
    mixed = (train_y.mean(axis=1) > 0) & (train_y.mean(axis=1) < 1)
    example = int(np.flatnonzero(mixed)[0])
    model = LearnedFilters()
    assert sum(p.numel() for p in model.parameters()) == 1432
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # 保留作者 512 次更新的衰减轨迹，本次仅运行前 32 次。
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=512)
    with torch.no_grad():
        before = model(x).numpy()
    assert np.all(before == 0.5)
    losses, first_grad = [], {}
    start = time.perf_counter()
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        if step == 0:
            first_grad = {"conv1_max_abs": float(model.conv1.weight.grad.abs().max()),
                          "w1_max_abs": float(model.w1.grad.abs().max())}
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
    training_seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        after = model(x).numpy()
        test_probability = model(tx).numpy()
        contributions = model.contributions(x[example:example+1]).numpy()[0]
        final_bce = float(F.binary_cross_entropy(torch.from_numpy(after), y))
        absorbed_error = float((absorb_weights(model)(tx) - model(tx)).abs().max())
        # 补查负、零、正权重，避免训练恰好只出现单一符号。
        signed_case = copy.deepcopy(model)
        signed_case.w1.copy_(torch.tensor([-0.7, 0.0, 0.3, -1.1]))
        signed_error = float((absorb_weights(signed_case)(tx) - signed_case(tx)).abs().max())
    manual = np.stack([numpy_forward(model, row) for row in test_x])
    forward_error = float(np.max(np.abs(manual - test_probability)))
    p64 = after.astype(np.float64)
    manual_bce = float(-(train_y * np.log(p64) + (1-train_y) * np.log1p(-p64)).mean())
    checks = {"numpy_forward_max_abs": forward_error, "absorbed_max_abs": absorbed_error,
              "signed_zero_absorbed_max_abs": signed_error,
              "bce_absolute_difference": abs(manual_bce-final_bce)}
    if max(checks.values()) > 2e-5 or final_bce >= losses[0]:
        raise AssertionError(f"Smoke verification failed: {checks}, BCE={final_bce}")
    report = {
        "status": "smoke_only_not_paper_reproduction", "upstream_commit": COMMIT,
        "device": "cpu", "seed": SEED, "torch_threads": 2, "steps": STEPS,
        "train_rows": 64, "test_rows": 8, "parameters": 1432,
        "float32_parameter_bytes": 1432*4, "single_feature_tensor_bytes": 12*SAMPLES*4,
        "learning_rate_initial": 0.01, "learning_rate_after_run": scheduler.get_last_lr()[0],
        "weight_decay": 1e-4, "scheduler_total_iters": 512, "savgol_window_samples": 151,
        "savgol_polyorder": 3, "threshold_rule": ">0.5", "label_semantics": {"0": "clean", "1": "artifact"},
        "split_protocol_from_segade": {"train": list(range(2, 14)), "test": [1, 14, 15]},
        "label_and_split_source": "https://doi.org/10.1088/1361-6579/ac3b3d",
        "train_bce_initial": losses[0], "train_bce_final": final_bce,
        "train_before": scores(train_y, before), "train_after": scores(train_y, after),
        "test_prefix_only": scores(test_y, test_probability), "checks": checks,
        "first_step_gradients": first_grad, "loss_before_each_update": losses,
        "training_seconds": training_seconds, "example": {"split": "train", "row_zero_based": example,
            "subject": "unknown", "original_start_time": "unknown",
            "label1_fraction": float(train_y[example].mean())},
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "provenance": provenance,
        "limitations": ["Only dataset prefixes and one seed; no generalization or HR claim",
            "Displayed example was used for training; subject/time mapping unavailable",
            "No channel-removal pruning, latency benchmark, or hardware deployment",
            "Exact row-to-subject/time mapping and upstream preprocessing history remain unresolved"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    t = np.arange(SAMPLES) / FS
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.2, 1.3, 1.4, 1]})
    fig.suptitle(f"SMoLK learning run | train row {example} | 64 Hz, 30 s | NOT a paper benchmark", fontsize=14)
    axes[0].plot(t, train_x[example], color="#176b87", lw=0.8)
    axes[0].set(ylabel="Standardized PPG\n(unitless)", xlim=(0, 30))
    smooth = savgol_filter(after[example], 151, 3)
    axes[1].step(t, train_y[example], where="post", color="#202938", lw=1.1, label="Human label (1=artifact)")
    axes[1].plot(t, after[example], alpha=0.45, color="#c86428", label="Model artifact score")
    axes[1].plot(t, smooth, color="#c86428", lw=1.5, label="SG 151 samples, order 3")
    axes[1].axhline(0.5, color="#687787", ls="--", lw=0.8)
    axes[1].set(ylabel="Label / model score", ylim=(-0.15, 1.15), xlim=(0, 30))
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    scale = float(np.max(np.abs(contributions))) or 1.0
    heatmap = axes[2].imshow(contributions, aspect="auto", origin="upper", interpolation="nearest",
                            extent=(0, 30, 12.5, 0.5), cmap="RdBu_r", vmin=-scale, vmax=scale)
    axes[2].set(ylabel="Kernel index\n1-4:192; 5-8:96; 9-12:64", xlabel="Time within segment (s)")
    fig.colorbar(heatmap, ax=axes[2], label="Signed contribution to logit")
    axes[3].plot(np.arange(1, STEPS+1), losses, color="#176b87")
    axes[3].scatter([STEPS+1], [final_bce], color="#c86428", label="After final update")
    axes[3].set(xlabel="Loss evaluation (1-32: before update; 33: after last update)",
                ylabel="Training mean BCE")
    axes[3].legend(fontsize=8)
    for ax in (axes[0], axes[1], axes[3]):
        ax.grid(alpha=0.15)
    fig.savefig(OUT / "segment_learning.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
