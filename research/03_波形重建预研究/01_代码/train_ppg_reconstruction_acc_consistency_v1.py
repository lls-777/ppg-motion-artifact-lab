#!/usr/bin/env python3
"""ACC 特征一致性单轮对照；保持 V3 设置，仅训练/验证集参与选权重。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from prepare_ppg_reconstruction_acc_consistency_v1 import (
    PAIRS, OUT, load_split, check_dataset, sha, write_csv, write_json,
)
from train_ppg_reconstruction_v2 import ResidualPPG, predict

TRAIN = OUT / "training"
BASELINE = Path(__file__).resolve().parents[1] / "02_结果/V3/training/config.json"
MODEL_CODE = Path(__file__).with_name("train_ppg_reconstruction_v2.py")
SEED, EPOCHS, BATCH_SIZE, LR = 20260917, 20, 64, 0.001
LOSS = "mean absolute error; unchanged V2 model, identity weight and optimizer"
SELECTION = "lowest validation L1, including exact identity epoch 0"


def check_baseline():
    baseline = json.loads(BASELINE.read_text())
    expected = {
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "parameters": 5393, "updates": 860, "train_pairs": 2748,
        "validation_pairs": 1304, "loss": LOSS, "selection": SELECTION,
        "v2_model_code_sha256": sha(MODEL_CODE), "torch_version": str(torch.__version__),
    }
    expected["script_sha256"] = sha(MODEL_CODE.with_name("train_ppg_reconstruction_v3.py"))
    if any(baseline.get(k) != v for k, v in expected.items()):
        raise ValueError("Changed V3 training configuration or environment")
    return sha(BASELINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-outputs", action="store_true")
    if not parser.parse_args().write_outputs:
        print(f"One run: train=2748; validation=1304; epochs=20; updates=860; {TRAIN}")
        return
    if TRAIN.exists():
        raise FileExistsError(TRAIN)
    baseline_sha = check_baseline()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    # 历史评价数组不参与训练或权重选择；模型与采样器初始化顺序沿用 V3。
    check_dataset(splits=("train", "validation"))
    x, y = load_split("train")
    vx, vy = load_split("validation")
    model = ResidualPPG()
    if sum(p.numel() for p in model.parameters()) != 5393:
        raise ValueError("Changed model parameter count")
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
        "train_pairs": len(x), "validation_pairs": len(vx), "loss": LOSS, "selection": SELECTION,
        "best_epoch": best_epoch, "best_validation_l1": best_loss,
        "pairs_config_sha256": sha(PAIRS / "config.json"), "script_sha256": sha(Path(__file__)),
        "v2_model_code_sha256": sha(MODEL_CODE), "baseline_config_sha256": baseline_sha,
        "torch_version": str(torch.__version__),
        "comparison_limit": "Fixed V3 PPG/targets; corrected ACC inputs. Previously observed "
                            "subjects: development comparison, not an independent blind test.",
    }
    torch.save({"format": "ppg_reconstruction_acc_consistency_v1",
                "model": best_state, "config": config}, TRAIN / "model.pt")
    write_csv(TRAIN / "history.csv", history)
    write_json(TRAIN / "config.json", config)
    write_json(TRAIN / "state.json", {"status": "complete"})
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
