#!/usr/bin/env python3
"""固定种子、固定轮数的残差重建；仅验证集选权重，测试集不参与训练。"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_ppg_reconstruction_pairs_v2 import (
    N, OUT, PAIRS, SEED, sha, write_csv, write_json,
)

TRAIN = OUT / "training"
EPOCHS, BATCH_SIZE, LR = 20, 64, 0.001


class ResidualPPG(nn.Module):
    """输入五通道；估计污染分量后从 PPG 扣除，输出一通道波形。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(5, 16, 9, padding=4), nn.ReLU(),
            nn.Conv1d(16, 16, 9, padding=16, dilation=4), nn.ReLU(),
            nn.Conv1d(16, 16, 9, padding=32, dilation=8), nn.ReLU(),
            nn.Conv1d(16, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        return inputs[:, :1] - self.net(inputs)


def load_split(split):
    state = json.loads((PAIRS / "state.json").read_text())
    config = json.loads((PAIRS / "config.json").read_text())
    path = PAIRS / f"{split}.npz"
    if state["status"] != "complete" or sha(path) != config["files"][split]["sha256"]:
        raise ValueError("Incomplete or changed pair dataset")
    with np.load(path, allow_pickle=False) as bundle:
        x, y = bundle["x"].copy(), bundle["y"].copy()
    if x.shape != (len(x), 5, N) or y.shape != (len(x), 1, N):
        raise ValueError("Unexpected pair dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite pair data")
    return x, y


def check_split_isolation():
    config = json.loads((PAIRS / "config.json").read_text())
    if sha(PAIRS / "manifest.csv") != config["manifest_sha256"]:
        raise ValueError("Changed manifest")
    with (PAIRS / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for field in ("patient", "acc_subject"):
        sets = [{r[field] for r in rows if r["split"] == s}
                for s in ("train", "validation", "test")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError(f"Split leakage: {field}")


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + BATCH_SIZE])).numpy()
                               for i in range(0, len(x), BATCH_SIZE)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if not args.write_outputs:
        print(f"train+validation from {PAIRS}; epochs={EPOCHS}; outputs={TRAIN}")
        return
    if TRAIN.exists():
        raise FileExistsError(TRAIN)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    check_split_isolation()
    x, y = load_split("train")
    vx, vy = load_split("validation")
    model = ResidualPPG()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
    best_loss = float(np.mean(np.abs(vx[:, :1] - vy)))
    best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "train_l1": float(np.mean(np.abs(x[:, :1] - y))),
                "validation_l1": best_loss}]
    TRAIN.mkdir(parents=True)
    write_json(TRAIN / "state.json", {"status": "training"})
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for inputs, target in loader:
            optimizer.zero_grad()
            loss = torch.mean(torch.abs(model(inputs) - target))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
        val_loss = float(np.mean(np.abs(predict(model, vx) - vy)))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        history.append({"epoch": epoch, "train_l1": total / len(x),
                        "validation_l1": val_loss})
        if val_loss < best_loss:
            best_state, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epoch, val_loss
        print(f"epoch={epoch} train_L1={total / len(x):.6f} val_L1={val_loss:.6f}", flush=True)
    config = {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
              "parameters": sum(p.numel() for p in model.parameters()),
              "loss": "mean absolute error; identity pairs included with equal example weight",
              "selection": "lowest validation L1; epoch 0 is exact identity fallback",
              "best_epoch": best_epoch, "best_validation_l1": best_loss,
              "pairs_config_sha256": sha(PAIRS / "config.json"),
              "script_sha256": sha(Path(__file__)), "torch_version": str(torch.__version__)}
    torch.save({"format": "ppg_reconstruction_v2", "model": best_state,
                "config": config}, TRAIN / "model.pt")
    write_csv(TRAIN / "history.csv", history)
    write_json(TRAIN / "config.json", config)
    write_json(TRAIN / "state.json", {"status": "complete"})
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
