#!/usr/bin/env python3
"""SMoLK 12 核剪枝：复现作者置零逻辑，并构造真实删核模型。

固定读取正式 seed-0 权重和三套测试数组。论文写欧氏距离，固定提交的
notebook 实际用余弦相似度；本脚本同时记录两者的最近核对，但按发布代码
执行剪枝。候选只用于评估，不微调，也不覆盖正式复现结果。
"""
from __future__ import annotations

import copy
import csv
import json
import math
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_full import atomic_torch_save, load_dataset, postprocess_scores, predict
from run_smoke import COMMIT, LENGTHS, SAMPLES, LearnedFilters, absorb_weights

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "full_seed0" / "model_state.pt"
OUT_DIR = ROOT / "outputs" / "pruning_seed0"
THREADS = 1
WARMUP = 100
BENCHMARK_BLOCKS = 20
RUNS_PER_BLOCK = 100
GROUP_NAMES = {1: "long_192", 2: "moderate_96", 3: "short_64"}
CONFIGS = {
    "baseline": (),
    "code_cosine_long_1": (1,),
    "code_cosine_moderate_1": (2,),
    "code_cosine_short_1": (3,),
    "code_cosine_one_each": (1, 2, 3),
}


class CompactLearnedFilters(nn.Module):
    """与原前向相同，但每组只分配实际保留的输出通道。"""

    def __init__(self, counts):
        super().__init__()
        for group, (count, length) in enumerate(zip(counts, LENGTHS), 1):
            setattr(self, f"conv{group}", nn.Conv1d(1, count, length, padding="same"))
            setattr(self, f"w{group}", nn.Parameter(torch.empty(count)))

    def contributions(self, x):
        groups = [
            F.relu(getattr(self, f"conv{group}")(x))
            * getattr(self, f"w{group}")[None, :, None]
            for group in (1, 2, 3)
        ]
        return torch.cat(groups, dim=1)

    def forward(self, x):
        return torch.sigmoid(self.contributions(x).sum(dim=1))


def load_base_model():
    saved = torch.load(MODEL_PATH, map_location="cpu")
    expected = ("smolk_full_seed0_state_v1", COMMIT, 0)
    actual = (saved.get("format"), saved.get("upstream_commit"), saved.get("seed"))
    if actual != expected:
        raise ValueError(f"Unexpected model metadata: {actual}")
    model = LearnedFilters()
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def pair_table(model, group, metric):
    conv = getattr(model, f"conv{group}")
    rows = []
    for first in range(conv.out_channels):
        for second in range(first + 1, conv.out_channels):
            left, right = conv.weight[first], conv.weight[second]
            if metric == "cosine":
                score = float(((left / left.norm()) * (right / right.norm())).sum())
            elif metric == "euclidean":
                score = float(torch.linalg.vector_norm(left - right))
            else:
                raise ValueError(metric)
            rows.append({"pair": [second, first], "score": score})
    rows.sort(key=lambda row: row["score"], reverse=metric == "cosine")
    return rows


def pair_rule_comparison(model):
    comparison = {}
    for group in (1, 2, 3):
        cosine = pair_table(model, group, "cosine")[0]
        euclidean = pair_table(model, group, "euclidean")[0]
        comparison[GROUP_NAMES[group]] = {
            "code_cosine": cosine,
            "paper_euclidean": euclidean,
            "same_selected_pair": cosine["pair"] == euclidean["pair"],
        }
    return comparison


def prune_by_code_rule(base, groups):
    """复现 notebook 的单对合并公式；每组只选最相似的一对。"""
    model = copy.deepcopy(base)
    removed = {1: set(), 2: set(), 3: set()}
    operations = []
    with torch.no_grad():
        for group in groups:
            conv = getattr(model, f"conv{group}")
            weights = getattr(model, f"w{group}")
            selected = pair_table(model, group, "cosine")[0]
            first, second = selected["pair"]
            first_mean = conv.weight[first].abs().mean()
            second_mean = conv.weight[second].abs().mean()
            first_effective = weights[first] * first_mean
            second_effective = weights[second] * second_mean
            if first_effective > second_effective:
                keep, remove = first, second
                keep_effective, remove_effective = first_effective, second_effective
            else:
                keep, remove = second, first
                keep_effective, remove_effective = second_effective, first_effective
            before = {
                "first_effective": float(first_effective),
                "second_effective": float(second_effective),
                "keep_weight": float(weights[keep]),
                "keep_bias": float(conv.bias[keep]),
                "remove_weight": float(weights[remove]),
                "remove_bias": float(conv.bias[remove]),
            }
            weights[keep] = (
                (keep_effective + remove_effective)
                / conv.weight[keep].abs().mean()
            )
            conv.bias[keep] += conv.bias[remove]
            weights[remove] = 0.0
            conv.bias[remove] = 0.0
            conv.weight[remove] = 1e-5
            removed[group].add(remove)
            operations.append({
                "group": group,
                "group_name": GROUP_NAMES[group],
                "kernel_length_samples": LENGTHS[group - 1],
                "selected_pair": selected["pair"],
                "cosine_similarity": selected["score"],
                "keep": keep,
                "remove": remove,
                "before": before,
                "after_keep_weight": float(weights[keep]),
                "after_keep_bias": float(conv.bias[keep]),
            })
    model.eval()
    return model, removed, operations


def compact_model(masked, removed):
    kept = {
        group: [
            index for index in range(getattr(masked, f"conv{group}").out_channels)
            if index not in removed[group]
        ]
        for group in (1, 2, 3)
    }
    model = CompactLearnedFilters([len(kept[group]) for group in (1, 2, 3)])
    with torch.no_grad():
        for group in (1, 2, 3):
            source = getattr(masked, f"conv{group}")
            target = getattr(model, f"conv{group}")
            target.weight.copy_(source.weight[kept[group]])
            target.bias.copy_(source.bias[kept[group]])
            getattr(model, f"w{group}").copy_(
                getattr(masked, f"w{group}")[kept[group]]
            )
    model.eval()
    return model, kept


def model_resources(counts):
    kernel_floats = sum(count * length for count, length in zip(counts, LENGTHS))
    total_kernels = sum(counts)
    original_style_floats = kernel_floats + 2 * total_kernels
    absorbed_floats = kernel_floats + total_kernels
    sign_bytes = math.ceil(total_kernels / 8)
    return {
        "kernels_by_group": counts,
        "total_kernels": total_kernels,
        "masked_model_float_parameters": 1432,
        "compact_original_style_float_parameters": original_style_floats,
        "compact_original_style_float32_bytes": original_style_floats * 4,
        "absorbed_float_parameters": absorbed_floats,
        "absorbed_float32_bytes": absorbed_floats * 4,
        "absorbed_sign_bits": total_kernels,
        "absorbed_sign_bytes_packed": sign_bytes,
        "absorbed_total_packed_bytes": absorbed_floats * 4 + sign_bytes,
        "convolution_coefficient_products_per_30s_segment":
            SAMPLES * kernel_floats,
        "materialized_feature_tensor_float32_bytes":
            SAMPLES * total_kernels * 4,
    }


def benchmark(model, example):
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(example)
        block_ms = []
        for _ in range(BENCHMARK_BLOCKS):
            started = time.perf_counter_ns()
            for _ in range(RUNS_PER_BLOCK):
                model(example)
            elapsed = time.perf_counter_ns() - started
            block_ms.append(elapsed / RUNS_PER_BLOCK / 1e6)
    return {
        "median_ms": float(np.median(block_ms)),
        "p10_ms": float(np.percentile(block_ms, 10)),
        "p90_ms": float(np.percentile(block_ms, 90)),
        "blocks": BENCHMARK_BLOCKS,
        "runs_per_block": RUNS_PER_BLOCK,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(result):
    names = list(result["configurations"])
    short_names = [name.replace("code_cosine_", "").replace("_1", "") for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = {"DaLiA": "#176b87", "WESAD": "#4b8f4b", "TROIKA": "#c86428"}
    for dataset, color in colors.items():
        x = [
            100 * result["configurations"][name]["resources"]
            ["original_style_parameter_reduction_fraction"]
            for name in names
        ]
        y = [
            100 * result["configurations"][name]["metrics"][dataset]
            ["savgol_151_samples_2.359s_code"]["dice_retained_fraction"]
            for name in names
        ]
        axes[0].plot(x, y, "o-", color=color, label=dataset)
    axes[0].axhline(95, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Physical parameter reduction (%)",
        ylabel="151-sample SG Dice retained (%)",
        title="Pruning cost on complete test sets",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    positions = np.arange(len(names))
    masked = [
        result["configurations"][name]["latency"]["masked"]["median_ms"]
        for name in names
    ]
    compact = [
        result["configurations"][name]["latency"]["compact"]["median_ms"]
        for name in names
    ]
    axes[1].bar(positions - 0.18, masked, 0.36, label="masked/zeroed")
    axes[1].bar(positions + 0.18, compact, 0.36, label="physical compact")
    axes[1].set_xticks(positions, short_names, rotation=20, ha="right")
    axes[1].set(
        ylabel="Median forward time per 30 s segment (ms)",
        title="Single-thread PyTorch CPU, batch size 1",
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    figure.savefig(OUT_DIR / "summary.png", dpi=160)
    plt.close(figure)


def main():
    final_path = OUT_DIR / "result.json"
    if final_path.exists():
        raise FileExistsError(f"Completed result already exists: {final_path}")
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    base = load_base_model()
    datasets = {
        name: load_dataset(name)
        for name in ("DaLiA", "WESAD", "TROIKA")
    }
    example = torch.from_numpy(datasets["DaLiA"][0][0:1, None, :])
    configurations = {}
    metric_rows = []
    resource_rows = []

    for name, groups in CONFIGS.items():
        masked, removed, operations = prune_by_code_rule(base, groups)
        compact, kept = compact_model(masked, removed)
        absorbed = absorb_weights(compact)
        metrics = {}
        max_masked_compact_difference = 0.0
        for dataset, (x, y) in datasets.items():
            masked_probability = predict(masked, x)
            compact_probability = predict(compact, x)
            max_masked_compact_difference = max(
                max_masked_compact_difference,
                float(np.max(np.abs(masked_probability - compact_probability))),
            )
            metrics[dataset] = postprocess_scores(y, compact_probability)
        with torch.inference_mode():
            compact_example = compact(example).numpy()
            absorbed_example = absorbed(example).numpy()
        absorption_difference = float(
            np.max(np.abs(compact_example - absorbed_example))
        )
        if max_masked_compact_difference > 2e-6 or absorption_difference > 2e-6:
            raise AssertionError(
                f"Equivalence check failed for {name}: "
                f"{max_masked_compact_difference}, {absorption_difference}"
            )
        counts = [len(kept[group]) for group in (1, 2, 3)]
        resources = model_resources(counts)
        resources["original_style_parameter_reduction_fraction"] = (
            1 - resources["compact_original_style_float_parameters"] / 1432
        )
        resources["feature_tensor_reduction_fraction"] = (
            1 - resources["materialized_feature_tensor_float32_bytes"]
            / (12 * SAMPLES * 4)
        )
        resources["convolution_product_reduction_fraction"] = (
            1 - resources["convolution_coefficient_products_per_30s_segment"]
            / (SAMPLES * sum(4 * length for length in LENGTHS))
        )
        configurations[name] = {
            "groups_pruned": list(groups),
            "operations": operations,
            "kept_indices": {str(group): kept[group] for group in (1, 2, 3)},
            "metrics": metrics,
            "resources": resources,
            "equivalence": {
                "masked_vs_physical_max_abs_probability":
                    max_masked_compact_difference,
                "physical_vs_weight_absorbed_max_abs_probability":
                    absorption_difference,
            },
            "latency": {
                "masked": benchmark(masked, example),
                "compact": benchmark(compact, example),
            },
        }

    baseline = configurations["baseline"]["metrics"]
    for name, entry in configurations.items():
        passes = True
        for dataset, methods in entry["metrics"].items():
            for method, values in methods.items():
                baseline_dice = baseline[dataset][method]["dice_artifact"]
                values["dice_absolute_change"] = (
                    values["dice_artifact"] - baseline_dice
                )
                values["dice_retained_fraction"] = (
                    values["dice_artifact"] / baseline_dice
                )
                metric_rows.append({
                    "configuration": name,
                    "dataset": dataset,
                    "postprocess": method,
                    **values,
                })
            passes &= (
                methods["savgol_151_samples_2.359s_code"]
                ["dice_retained_fraction"] >= 0.95
            )
        entry["passes_95_percent_retained_on_all_datasets_151_sg"] = bool(passes)
        resource_rows.append({
            "configuration": name,
            **entry["resources"],
            "masked_median_ms": entry["latency"]["masked"]["median_ms"],
            "compact_median_ms": entry["latency"]["compact"]["median_ms"],
        })

    acceptable = [
        name for name in configurations
        if name != "baseline"
        and configurations[name]
        ["passes_95_percent_retained_on_all_datasets_151_sg"]
    ]
    recommended = max(
        acceptable,
        key=lambda name: configurations[name]["resources"]
        ["original_style_parameter_reduction_fraction"],
        default=None,
    )
    result = {
        "status": "post_training_pruning_single_seed",
        "upstream_commit": COMMIT,
        "source_model": str(MODEL_PATH),
        "seed": 0,
        "selection_rule": {
            "released_code": "maximum cosine similarity within one length group",
            "paper_section_3_4": "minimum Euclidean distance within one length group",
            "effective_contribution": "e_m = w_m * mean(abs(k_m))",
            "merge": "w_keep'=(e_keep+e_remove)/mean(abs(k_keep)); "
                     "b_keep'=b_keep+b_remove",
            "acceptance": "151-sample SG Dice retains >=95% of baseline "
                          "on DaLiA, WESAD, and TROIKA",
        },
        "pair_rule_comparison": pair_rule_comparison(base),
        "configurations": configurations,
        "recommended_configuration": recommended,
        "benchmark_scope": {
            "device": "CPU",
            "threads": THREADS,
            "batch_size": 1,
            "input": "one normalized 30 s x 64 Hz segment",
            "included": "PyTorch model forward only",
            "excluded": "data loading, normalization, Savitzky-Golay, thresholding",
            "machine": platform.platform(),
            "torch": torch.__version__,
        },
        "limitations": [
            "One trained seed only; pruning stability across initializations is unknown.",
            "Post-training merge only; no fine-tuning after pruning.",
            "The released large-model ratio is transferred as one discrete pair; "
            "the paper does not publish a unique 12-kernel pruning ratio.",
            "PyTorch computer latency is not MCU latency, RAM, Flash, energy, or power.",
            "Parameter and feature-tensor sizes are theoretical float32 storage, "
            "not framework peak memory.",
            "Dice retention does not establish heart-rate improvement.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "metrics.csv", metric_rows)
    write_csv(OUT_DIR / "resources.csv", resource_rows)
    if recommended is not None:
        masked, removed, _ = prune_by_code_rule(base, CONFIGS[recommended])
        compact, kept = compact_model(masked, removed)
        atomic_torch_save({
            "format": "smolk_compact_pruned_state_v1",
            "upstream_commit": COMMIT,
            "source_seed": 0,
            "configuration": recommended,
            "kernel_lengths_samples": list(LENGTHS),
            "kept_indices": {str(group): kept[group] for group in (1, 2, 3)},
            "model": compact.state_dict(),
        }, OUT_DIR / "recommended_compact_state.pt")
    final_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_plot(result)
    print(json.dumps({
        "result": str(final_path),
        "recommended_configuration": recommended,
        "summary": str(OUT_DIR / "summary.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
