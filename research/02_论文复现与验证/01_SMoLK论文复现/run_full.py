#!/usr/bin/env python3
"""SMoLK 12 核完整单次复现：完整训练集、固定种子、三种后处理。

使用作者发布的处理后数组，按 Git blob SHA-1 校验。通过梯度累积降低
CPU 内存占用；每遍历完整训练集后才执行一次 Adam 更新。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import torch
from scipy.signal import savgol_filter

from run_smoke import COMMIT, SAMPLES, LearnedFilters, normalize

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "full_seed0"
STEPS = 512
SEED = 0
CHUNK_ROWS = 128
THREADS = 4
CHECKPOINT_EVERY = 16

DATASETS = {
    "train": {
        "folder": "new_PPG_DaLiA_train",
        "rows": 3436,
        "files": {
            "x": ("scaled_ppgs.npy", 52777088, "5a301782390a9f8516174369c3dbcb7942166625"),
            "y": ("seg_labels.npy", 52777088, "b6057c6acb9d516d08281bf8e31ece4fff57cd42"),
        },
    },
    "DaLiA": {
        "folder": "new_PPG_DaLiA_test",
        "rows": 869,
        "files": {
            "x": ("scaled_ppgs.npy", 13347968, "ccc618834a250d6fe4ca448cfad14323891ca122"),
            "y": ("seg_labels.npy", 13347968, "0509d4ed2159ceb62d0914f9c99e0b4b15f58655"),
        },
    },
    "WESAD": {
        "folder": "WESAD_all",
        "rows": 2888,
        "files": {
            "x": ("scaled_ppgs.npy", 44359808, "0e0e2f4b916c32d00059e827f85d2ce04644dd50"),
            "y": ("seg_labels.npy", 44359808, "a448a551a2bea6faf6e2dd3948195911374cbcd7"),
        },
    },
    "TROIKA": {
        "folder": "TROIKA_channel_1",
        "rows": 113,
        "files": {
            "x": ("scaled_ppgs.npy", 1735808, "48b7a91ed19f2e7fe863370eaa2173a77cb5c6c5"),
            "y": ("seg_labels.npy", 1735808, "80ebc0ee8c0f747c9bca1f6618740790ef1adfae"),
        },
    },
}

PAPER_RESULTS = {
    "table1_small": {"DaLiA": 0.8191, "WESAD": 0.8639, "TROIKA": 0.6891},
    "appendix_savgol_small": {"DaLiA": 0.8185, "WESAD": 0.8632, "TROIKA": 0.6889},
}


def git_blob_sha1(path: Path) -> str:
    """计算 Git blob 对象散列，校验固定提交中的二进制数据。"""
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, rows: int, size: int, blob_sha1: str) -> None:
    if path.stat().st_size != size or git_blob_sha1(path) != blob_sha1:
        raise ValueError(f"Data integrity check failed: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (rows, SAMPLES) or array.dtype != np.dtype("<f8"):
        raise ValueError(f"Unexpected NPY metadata: {path}, {array.shape}, {array.dtype}")


def ensure_file(dataset: str, kind: str) -> Path:
    """下载并缓存；网络失败时重试，正式文件不匹配时停止。"""
    spec = DATASETS[dataset]
    filename, size, blob_sha1 = spec["files"][kind]
    path = DATA_DIR / spec["folder"] / "processed_dataset" / filename
    if path.exists():
        validate_file(path, spec["rows"], size, blob_sha1)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://raw.githubusercontent.com/SullyChen/SMoLK/{COMMIT}/"
        f"PPG%20data/{spec['folder']}/processed_dataset/{filename}"
    )
    part = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {dataset}/{filename} ({size / 1024**2:.1f} MiB)", flush=True)
    for attempt in range(1, 6):
        request = urllib.request.Request(url, headers={"User-Agent": "SMoLK-reproduction"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response, part.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            break
        except (TimeoutError, urllib.error.URLError) as error:
            part.unlink(missing_ok=True)
            if attempt == 5:
                raise
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"Download attempt {attempt}/5 failed: {error}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    validate_file(part, spec["rows"], size, blob_sha1)
    os.replace(part, path)
    return path


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(ensure_file(name, "x"), allow_pickle=False)
    y = np.load(ensure_file(name, "y"), allow_pickle=False)
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Invalid values in {name}")
    return normalize(x), y


def binary_stats(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = y.astype(bool)
    predicted = prediction.astype(bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    denominator = 2 * tp + fp + fn
    return {
        "dice_artifact": 2 * tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "label_artifact_fraction": float(truth.mean()),
        "predicted_artifact_fraction": float(predicted.mean()),
    }


def postprocess_scores(y: np.ndarray, probability: np.ndarray) -> dict[str, dict]:
    configurations = {
        "no_smoothing": probability,
        "savgol_51_samples_0.797s": savgol_filter(probability, 51, 3, axis=-1),
        "savgol_151_samples_2.359s_code": savgol_filter(probability, 151, 3, axis=-1),
    }
    return {name: binary_stats(y, values > 0.5) for name, values in configurations.items()}


def predict(model: LearnedFilters, x: np.ndarray) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), CHUNK_ROWS):
            batch = torch.from_numpy(x[start:start + CHUNK_ROWS, None, :])
            outputs.append(model(batch).numpy())
    return np.concatenate(outputs)


def mean_bce(model: LearnedFilters, x: np.ndarray, y: np.ndarray) -> float:
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), CHUNK_ROWS):
            xb = torch.from_numpy(x[start:start + CHUNK_ROWS, None, :])
            yb = torch.from_numpy(y[start:start + CHUNK_ROWS].astype(np.float32))
            total += float(torch.nn.functional.binary_cross_entropy(
                model(xb), yb, reduction="sum"
            ))
    return total / y.size


def checkpoint_payload(
    model: LearnedFilters,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LinearLR,
    next_step: int,
    losses: list[float],
    elapsed_seconds: float,
) -> dict:
    return {
        "format": "smolk_full_seed0_checkpoint_v1",
        "upstream_commit": COMMIT,
        "seed": SEED,
        "next_step": next_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss_before_update": losses,
        "elapsed_training_seconds": elapsed_seconds,
    }


def train() -> tuple[LearnedFilters, list[float], float]:
    train_x, train_y = load_dataset("train")
    model = LearnedFilters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.2, total_iters=STEPS
    )
    checkpoint = OUT_DIR / "checkpoint.pt"
    start_step, losses, elapsed_before = 0, [], 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu")
        expected = ("smolk_full_seed0_checkpoint_v1", COMMIT, SEED)
        actual = (saved.get("format"), saved.get("upstream_commit"), saved.get("seed"))
        if actual != expected:
            raise ValueError(f"Unexpected checkpoint metadata: {actual}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_step = int(saved["next_step"])
        losses = list(saved["loss_before_update"])
        elapsed_before = float(saved["elapsed_training_seconds"])
        print(f"Resuming at update {start_step}/{STEPS}", flush=True)
    started = time.perf_counter()
    model.train()
    for step in range(start_step, STEPS):
        optimizer.zero_grad()
        loss_value = 0.0
        for start in range(0, len(train_x), CHUNK_ROWS):
            xb = torch.from_numpy(train_x[start:start + CHUNK_ROWS, None, :])
            yb = torch.from_numpy(train_y[start:start + CHUNK_ROWS].astype(np.float32))
            loss = torch.nn.functional.binary_cross_entropy(
                model(xb), yb, reduction="sum"
            ) / train_y.size
            loss.backward()
            loss_value += float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        optimizer.step()
        scheduler.step()
        losses.append(loss_value)
        completed = step + 1
        elapsed = elapsed_before + time.perf_counter() - started
        if completed % CHECKPOINT_EVERY == 0 or completed == STEPS:
            atomic_torch_save(
                checkpoint_payload(model, optimizer, scheduler, completed, losses, elapsed),
                checkpoint,
            )
            print(
                f"update={completed}/{STEPS} loss={loss_value:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return model, losses, elapsed_before + time.perf_counter() - started


def save_summary(result: dict) -> None:
    dataset_names = ["DaLiA", "WESAD", "TROIKA"]
    configurations = [
        "no_smoothing",
        "savgol_51_samples_0.797s",
        "savgol_151_samples_2.359s_code",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(np.arange(1, len(result["loss_before_update"]) + 1),
                 result["loss_before_update"], color="#176b87")
    axes[0].scatter([STEPS + 1], [result["train_bce_after_final_update"]],
                    color="#c86428", label="after update 512")
    axes[0].set(xlabel="Full-dataset parameter update", ylabel="Mean training BCE",
                title="Training loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    width = 0.2
    positions = np.arange(len(dataset_names))
    for index, configuration in enumerate(configurations):
        values = [result["evaluation"][dataset][configuration]["dice_artifact"]
                  for dataset in dataset_names]
        axes[1].bar(positions + (index - 1) * width, values, width, label=configuration)
    paper = [PAPER_RESULTS["table1_small"][dataset] for dataset in dataset_names]
    axes[1].scatter(positions, paper, marker="x", s=70, color="black",
                    label="paper Table 1 small")
    axes[1].set_xticks(positions, dataset_names)
    axes[1].set_ylim(0, 1)
    axes[1].set(ylabel="Artifact Dice", title="Complete test datasets")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=7)
    figure.suptitle("SMoLK 12-kernel full single-seed reproduction")
    figure.savefig(OUT_DIR / "summary.png", dpi=160)
    plt.close(figure)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_result = OUT_DIR / "result.json"
    if final_result.exists():
        raise FileExistsError(f"Completed result already exists: {final_result}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    model, losses, training_seconds = train()
    train_x, train_y = load_dataset("train")
    final_train_bce = mean_bce(model, train_x, train_y)
    evaluation, evaluation_seconds = {}, {}
    for dataset in ("DaLiA", "WESAD", "TROIKA"):
        x, y = load_dataset(dataset)
        started = time.perf_counter()
        probability = predict(model, x)
        evaluation_seconds[dataset] = time.perf_counter() - started
        evaluation[dataset] = postprocess_scores(y, probability)
        print(
            dataset,
            {name: round(values["dice_artifact"], 6)
             for name, values in evaluation[dataset].items()},
            flush=True,
        )
    result = {
        "status": "full_single_seed_reproduction",
        "upstream_commit": COMMIT,
        "seed": SEED,
        "device": "cpu",
        "threads": THREADS,
        "model": {"kernels_per_length": 4, "total_kernels": 12,
                  "kernel_lengths_samples": [192, 96, 64],
                  "parameters": sum(parameter.numel() for parameter in model.parameters())},
        "training": {
            "rows": DATASETS["train"]["rows"],
            "samples_per_row": SAMPLES,
            "updates": STEPS,
            "gradient_accumulation_chunk_rows": CHUNK_ROWS,
            "optimizer": "torch.optim.Adam",
            "initial_lr": 0.01,
            "final_next_lr": 0.002,
            "weight_decay": 1e-4,
            "training_seconds": training_seconds,
            "implementation_difference": (
                "Each update accumulates sum-reduced BCE over row chunks, divides by "
                "all 3436x1920 labels, then performs one optimizer step. The released "
                "notebook processes all rows in one tensor for the 12-kernel model."
            ),
        },
        "train_bce_initial": losses[0],
        "train_bce_after_final_update": final_train_bce,
        "loss_before_update": losses,
        "evaluation": evaluation,
        "evaluation_forward_seconds": evaluation_seconds,
        "paper_reference": PAPER_RESULTS,
        "paper_gap_code_151_minus_table1": {
            dataset: (
                evaluation[dataset]["savgol_151_samples_2.359s_code"]["dice_artifact"]
                - PAPER_RESULTS["table1_small"][dataset]
            )
            for dataset in ("DaLiA", "WESAD", "TROIKA")
        },
        "label_semantics": {"0": "clean", "1": "artifact"},
        "split_protocol": {"train_subjects": list(range(2, 14)),
                           "test_subjects": [1, 14, 15]},
        "data": {
            dataset: {
                "rows": spec["rows"],
                "folder": spec["folder"],
                "files": {
                    kind: {"name": values[0], "bytes": values[1], "git_blob_sha1": values[2]}
                    for kind, values in spec["files"].items()
                },
            }
            for dataset, spec in DATASETS.items()
        },
        "versions": {
            "python": ".".join(map(str, os.sys.version_info[:3])),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "limitations": [
            "One initialization only; the paper reports ten repeated initializations.",
            "Uses the authors' processed arrays; raw preprocessing was not reconstructed.",
            "Gradient accumulation changes floating-point summation order.",
            "Segmentation Dice does not establish heart-rate improvement.",
            "No physical kernel removal, streaming conversion, C port, or MCU measurement.",
        ],
    }
    atomic_torch_save(
        {"format": "smolk_full_seed0_state_v1", "upstream_commit": COMMIT,
         "seed": SEED, "model": model.state_dict()},
        OUT_DIR / "model_state.pt",
    )
    final_result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    save_summary(result)
    print(json.dumps({
        "result": str(final_result),
        "summary": str(OUT_DIR / "summary.png"),
        "training_seconds": training_seconds,
        "paper_gap_code_151_minus_table1": result["paper_gap_code_151_minus_table1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
