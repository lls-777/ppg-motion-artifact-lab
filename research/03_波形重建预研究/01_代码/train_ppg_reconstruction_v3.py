#!/usr/bin/env python3
"""V3单轮训练：沿用V2模型和优化设置；不重复扩样，不使用测试选权重。"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from prepare_ppg_reconstruction_pairs_v3 import (
    N, OUT, PAIRS, SEED, SPLITS, EXPECTED, check_roles, sha, write_csv, write_json,
)
from train_ppg_reconstruction_v2 import ResidualPPG, predict

TRAIN = OUT / "training"
EPOCHS, BATCH_SIZE, LR = 20, 64, 0.001


def load_split(split):
    config = json.loads((PAIRS / "config.json").read_text())
    path = PAIRS / f"{split}.npz"
    if json.loads((PAIRS / "state.json").read_text())["status"] != "complete":
        raise ValueError("Pair dataset incomplete")
    if sha(path) != config["files"][split]["sha256"]:
        raise ValueError("Changed pair data")
    with np.load(path, allow_pickle=False) as arrays:
        x, y = arrays["x"].copy(), arrays["y"].copy()
    count = 4 * sum(EXPECTED[s] for s in SPLITS[split])
    if x.shape != (count, 5, N) or y.shape != (count, 1, N):
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
    check_roles(rows)
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        expected = 4 * sum(EXPECTED[s] for s in SPLITS[split])
        if [int(r["index"]) for r in part] != list(range(expected)):
            raise ValueError("Manifest order/count mismatch")
        if {r["condition"] for r in part} != {"identity", "snr_10", "snr_0", "snr_-5"}:
            raise ValueError("Unexpected noise conditions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    if not args.write_outputs:
        print(f"One run: train=2748 pairs; validation=1304; epochs=20; updates=860; {TRAIN}")
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
        history.append({"epoch": epoch, "train_l1": total / len(x), "validation_l1": val_loss})
        if val_loss < best_loss:
            best_state, best_epoch, best_loss = copy.deepcopy(model.state_dict()), epoch, val_loss
        print(f"epoch={epoch} train_L1={total / len(x):.6f} val_L1={val_loss:.6f}", flush=True)
    config = {
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "parameters": sum(p.numel() for p in model.parameters()), "updates": len(loader) * EPOCHS,
        "train_pairs": len(x), "validation_pairs": len(vx),
        "loss": "mean absolute error; unchanged V2 model, identity weight and optimizer",
        "selection": "lowest validation L1, including exact identity epoch 0",
        "best_epoch": best_epoch, "best_validation_l1": best_loss,
        "pairs_config_sha256": sha(PAIRS / "config.json"), "script_sha256": sha(Path(__file__)),
        "v2_model_code_sha256": sha(Path(__file__).with_name("train_ppg_reconstruction_v2.py")),
        "torch_version": str(torch.__version__),
        "comparison_limit": "860 updates versus V2 1060; reference distributions differ",
    }
    torch.save({"format": "ppg_reconstruction_v3", "model": best_state, "config": config},
               TRAIN / "model.pt")
    write_csv(TRAIN / "history.csv", history)
    write_json(TRAIN / "config.json", config)
    write_json(TRAIN / "state.json", {"status": "complete"})
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
