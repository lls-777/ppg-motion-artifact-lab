#!/usr/bin/env python3
"""分析哪些可部署特征能够区分正确心率候选与错误频谱主峰。

本脚本只读取 PPG-DaLiA 的 S1-S10，不读取 S11-S15。

ECG 只在候选峰生成完毕后用于离线标注和效果评价，不参与：
1. PPG 候选峰生成；
2. ACC 特征计算；
3. 因果历史心率的生成。

本阶段只进行特征可分性分析，不输出最终候选评分公式，也不写文件。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

import numpy as np

from analyze_candidate_peak_recoverability_v0 import (
    DESIGN_SUBJECTS,
    build_results,
    load_design_rows,
)
from validate_acc_aware_sqi_v1 import load_frozen_design


RECOVERABLE_THRESHOLD_BPM = 5.0
MAX_PRIOR_AGE_SECONDS = 30.0

EXPECTED_TARGET_WINDOWS = 1_284
EXPECTED_RECOVERABLE_WINDOWS = 1_010
EXPECTED_CANDIDATES_PER_WINDOW = 5


def window_key(row: dict[str, Any]) -> tuple[int, float]:
    """构造能唯一定位一个受试者窗口的键。"""
    return (
        int(row["subject_id"]),
        float(row["window_start_s"]),
    )


def build_candidate_groups(
    candidate_results: list[dict[str, Any]],
) -> dict[tuple[int, float], list[dict[str, Any]]]:
    """按受试者和窗口起点组织候选峰，并验证候选排名。"""
    grouped: dict[
        tuple[int, float],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in candidate_results:
        grouped[window_key(candidate)].append(candidate)

    for key, candidates in grouped.items():
        candidates.sort(
            key=lambda row: row["candidate_rank"]
        )
        ranks = [
            int(candidate["candidate_rank"])
            for candidate in candidates
        ]
        expected_ranks = list(
            range(1, EXPECTED_CANDIDATES_PER_WINDOW + 1)
        )
        if ranks != expected_ranks:
            raise ValueError(
                f"窗口 {key} 的候选排名异常：{ranks}"
            )

    return dict(grouped)


def build_causal_prior_map(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, float], dict[str, float] | None]:
    """为每个窗口生成只依赖过去信息的历史心率。

    历史值取最近一个 ACC-aware 状态为 usable 的窗口所输出的
    PPG 心率。当前窗口只能读取此前窗口，且历史超过 30 秒失效。

    这里不检查历史心率是否接近 ECG，因为在线设备无法获得该信息。
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["subject_id"])].append(row)

    prior_map: dict[
        tuple[int, float],
        dict[str, float] | None,
    ] = {}

    for subject_id in DESIGN_SUBJECTS:
        subject_rows = sorted(
            grouped[subject_id],
            key=lambda row: row["window_center_s"],
        )
        previous_hr: float | None = None
        previous_center_s: float | None = None
        last_seen_center_s: float | None = None

        for row in subject_rows:
            center_s = float(row["window_center_s"])
            if (
                last_seen_center_s is not None
                and center_s <= last_seen_center_s
            ):
                raise ValueError(
                    f"S{subject_id} 窗口时间不是严格递增"
                )

            prior: dict[str, float] | None = None
            if (
                previous_hr is not None
                and previous_center_s is not None
            ):
                age_s = center_s - previous_center_s
                if 0.0 < age_s <= MAX_PRIOR_AGE_SECONDS:
                    prior = {
                        "prior_hr_bpm": previous_hr,
                        "prior_age_s": age_s,
                    }

            prior_map[window_key(row)] = prior

            # 必须在保存当前窗口的 prior 之后更新，防止读取未来信息。
            if row["acc_aware_state"] == "usable":
                previous_hr = float(row["ppg_hr_bpm"])
                previous_center_s = center_s

            last_seen_center_s = center_s

    return prior_map


def find_unique_candidate(
    candidates: list[dict[str, Any]],
    field: str,
    expected_value: Any,
) -> dict[str, Any]:
    matched = [
        candidate
        for candidate in candidates
        if candidate[field] == expected_value
    ]
    if len(matched) != 1:
        raise ValueError(
            f"候选字段 {field}={expected_value!r} "
            f"应唯一，实际数量={len(matched)}"
        )
    return matched[0]


def build_paired_records(
    window_results: list[dict[str, Any]],
    candidate_groups: dict[
        tuple[int, float],
        list[dict[str, Any]],
    ],
    prior_map: dict[
        tuple[int, float],
        dict[str, float] | None,
    ],
) -> list[dict[str, Any]]:
    """比较每个可恢复窗口的错误第一峰与 oracle 正确候选。"""
    records: list[dict[str, Any]] = []

    for window in window_results:
        if not window["top_5_recoverable_le_5_bpm"]:
            continue

        key = window_key(window)
        candidates = candidate_groups.get(key)
        if candidates is None:
            raise ValueError(f"窗口 {key} 缺少候选峰")

        rank_one = find_unique_candidate(
            candidates,
            "candidate_rank",
            1,
        )
        oracle = find_unique_candidate(
            candidates,
            "is_oracle_best",
            True,
        )

        if (
            float(oracle["candidate_absolute_error_bpm"])
            > RECOVERABLE_THRESHOLD_BPM
        ):
            raise ValueError(
                f"窗口 {key} 被标记为可恢复，"
                "但 oracle 误差大于 5 bpm"
            )
        if int(oracle["candidate_rank"]) == 1:
            raise ValueError(
                f"窗口 {key} 的错误第一峰不应同时是正确候选"
            )

        prior = prior_map.get(key)
        rank_one_continuity_gap: float | None = None
        oracle_continuity_gap: float | None = None
        prior_hr: float | None = None
        prior_age_s: float | None = None

        if prior is not None:
            prior_hr = float(prior["prior_hr_bpm"])
            prior_age_s = float(prior["prior_age_s"])
            rank_one_continuity_gap = abs(
                float(rank_one["candidate_hr_bpm"])
                - prior_hr
            )
            oracle_continuity_gap = abs(
                float(oracle["candidate_hr_bpm"])
                - prior_hr
            )

        records.append(
            {
                "subject_id": int(window["subject_id"]),
                "window_start_s": float(
                    window["window_start_s"]
                ),
                "activity_id": int(window["activity_id"]),
                "activity_name": window["activity_name"],
                "acc_aware_state": window["acc_aware_state"],
                "oracle_rank": int(
                    oracle["candidate_rank"]
                ),
                "rank_one_power_ratio": float(
                    rank_one["candidate_power_ratio"]
                ),
                "oracle_power_ratio": float(
                    oracle["candidate_power_ratio"]
                ),
                "rank_one_acc_gap_hz": float(
                    rank_one[
                        "candidate_acc_harmonic_gap_hz"
                    ]
                ),
                "oracle_acc_gap_hz": float(
                    oracle[
                        "candidate_acc_harmonic_gap_hz"
                    ]
                ),
                "rank_one_acc_collision": bool(
                    rank_one["candidate_acc_collision"]
                ),
                "oracle_acc_collision": bool(
                    oracle["candidate_acc_collision"]
                ),
                "prior_hr_bpm": prior_hr,
                "prior_age_s": prior_age_s,
                "rank_one_continuity_gap_bpm": (
                    rank_one_continuity_gap
                ),
                "oracle_continuity_gap_bpm": (
                    oracle_continuity_gap
                ),
            }
        )

    return records


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def percent(flags: list[bool]) -> float:
    if not flags:
        return float("nan")
    return 100.0 * sum(flags) / len(flags)


def print_feature_summary(
    paired_records: list[dict[str, Any]],
) -> None:
    """输出后续评分公式需要的总体特征证据。"""
    rank_one_power = [
        row["rank_one_power_ratio"]
        for row in paired_records
    ]
    oracle_power = [
        row["oracle_power_ratio"]
        for row in paired_records
    ]
    rank_one_acc_gap = [
        row["rank_one_acc_gap_hz"]
        for row in paired_records
    ]
    oracle_acc_gap = [
        row["oracle_acc_gap_hz"]
        for row in paired_records
    ]

    with_prior = [
        row
        for row in paired_records
        if row["prior_hr_bpm"] is not None
    ]

    print("\n总体特征可分性")
    print(
        "  错误第一峰功率占比中位数="
        f"{median(rank_one_power):.4f}"
    )
    print(
        "  正确候选功率占比中位数="
        f"{median(oracle_power):.4f}"
    )
    print(
        "  正确候选功率低于错误第一峰="
        f"{percent([o < r for o, r in zip(oracle_power, rank_one_power)]):.2f}%"
    )
    print(
        "  错误第一峰 ACC 谐波距离中位数="
        f"{median(rank_one_acc_gap):.4f} Hz"
    )
    print(
        "  正确候选 ACC 谐波距离中位数="
        f"{median(oracle_acc_gap):.4f} Hz"
    )
    print(
        "  错误第一峰发生 ACC 碰撞="
        f"{percent([row['rank_one_acc_collision'] for row in paired_records]):.2f}%"
    )
    print(
        "  正确候选发生 ACC 碰撞="
        f"{percent([row['oracle_acc_collision'] for row in paired_records]):.2f}%"
    )
    print(
        "  ACC 谐波距离更支持正确候选="
        f"{percent([o > r for o, r in zip(oracle_acc_gap, rank_one_acc_gap)]):.2f}%"
    )

    print(
        "  存在 30 秒内因果历史心率="
        f"{len(with_prior)}/{len(paired_records)}"
        f"（{100.0 * len(with_prior) / len(paired_records):.2f}%）"
    )

    if with_prior:
        rank_one_continuity = [
            row["rank_one_continuity_gap_bpm"]
            for row in with_prior
        ]
        oracle_continuity = [
            row["oracle_continuity_gap_bpm"]
            for row in with_prior
        ]
        continuity_favors_oracle = [
            oracle_gap < rank_one_gap
            for oracle_gap, rank_one_gap in zip(
                oracle_continuity,
                rank_one_continuity,
            )
        ]
        acc_favors_oracle = [
            row["oracle_acc_gap_hz"]
            > row["rank_one_acc_gap_hz"]
            for row in with_prior
        ]

        print(
            "  错误第一峰与历史心率差值中位数="
            f"{median(rank_one_continuity):.2f} bpm"
        )
        print(
            "  正确候选与历史心率差值中位数="
            f"{median(oracle_continuity):.2f} bpm"
        )
        print(
            "  时序连续性更支持正确候选="
            f"{percent(continuity_favors_oracle):.2f}%"
        )
        print(
            "  ACC 或时序至少一项支持正确候选="
            f"{percent([a or t for a, t in zip(acc_favors_oracle, continuity_favors_oracle)]):.2f}%"
        )
        print(
            "  ACC 与时序同时支持正确候选="
            f"{percent([a and t for a, t in zip(acc_favors_oracle, continuity_favors_oracle)]):.2f}%"
        )


def print_subject_diagnostics(
    subject_id: int,
    window_results: list[dict[str, Any]],
    paired_records: list[dict[str, Any]],
) -> None:
    """重点说明 S4、S8 是候选缺失还是候选排序问题。"""
    subject_windows = [
        row
        for row in window_results
        if row["subject_id"] == subject_id
    ]
    recoverable = [
        row
        for row in subject_windows
        if row["top_5_recoverable_le_5_bpm"]
    ]
    unrecoverable = [
        row
        for row in subject_windows
        if not row["top_5_recoverable_le_5_bpm"]
    ]
    subject_pairs = [
        row
        for row in paired_records
        if row["subject_id"] == subject_id
    ]
    with_prior = [
        row
        for row in subject_pairs
        if row["prior_hr_bpm"] is not None
    ]

    print(f"\nS{subject_id} 重点诊断")
    print(f"  大误差窗口={len(subject_windows)}")
    print(
        f"  Top-5 内有 ≤5 bpm 候选={len(recoverable)}"
    )
    print(
        "  Top-5 内没有 ≤5 bpm 候选="
        f"{len(unrecoverable)}"
    )
    print(
        "  可恢复窗口的正确候选中位排名="
        f"{median([row['oracle_rank'] for row in subject_pairs]):.1f}"
    )
    print(
        "  错误第一峰 ACC 碰撞率="
        f"{percent([row['rank_one_acc_collision'] for row in subject_pairs]):.2f}%"
    )
    print(
        "  正确候选 ACC 碰撞率="
        f"{percent([row['oracle_acc_collision'] for row in subject_pairs]):.2f}%"
    )

    if with_prior:
        print(
            "  有历史心率时，时序更支持正确候选="
            f"{percent([row['oracle_continuity_gap_bpm'] < row['rank_one_continuity_gap_bpm'] for row in with_prior]):.2f}%"
        )


def main() -> int:
    print("候选峰特征可分性分析 V0")
    print("数据边界：只读取 S1-S10，不读取 S11-S15")
    print(
        "ECG 仅用于离线标注正确候选，"
        "不参与候选生成、ACC 特征和因果历史心率"
    )

    frozen_design, parameters, frozen_hash = (
        load_frozen_design()
    )
    del frozen_design

    rows = load_design_rows()
    window_results, candidate_results = build_results(
        rows,
        parameters["harmonic_gap_threshold_hz"],
    )
    candidate_groups = build_candidate_groups(
        candidate_results
    )
    prior_map = build_causal_prior_map(rows)
    paired_records = build_paired_records(
        window_results,
        candidate_groups,
        prior_map,
    )

    if len(window_results) != EXPECTED_TARGET_WINDOWS:
        raise ValueError(
            "目标窗口数回归失败："
            f"期望 {EXPECTED_TARGET_WINDOWS}，"
            f"实际 {len(window_results)}"
        )
    if len(paired_records) != EXPECTED_RECOVERABLE_WINDOWS:
        raise ValueError(
            "可恢复窗口数回归失败："
            f"期望 {EXPECTED_RECOVERABLE_WINDOWS}，"
            f"实际 {len(paired_records)}"
        )
    if set(candidate_groups) != {
        window_key(row)
        for row in window_results
    }:
        raise ValueError("候选窗口与目标窗口集合不一致")

    print(f"\n冻结参数哈希：{frozen_hash}")
    print(
        f"大误差窗口={len(window_results)}，"
        f"Top-5 可恢复窗口={len(paired_records)}"
    )

    print_feature_summary(paired_records)
    print_subject_diagnostics(
        4,
        window_results,
        paired_records,
    )
    print_subject_diagnostics(
        8,
        window_results,
        paired_records,
    )

    print(
        "\n说明：本结果用于决定候选评分特征，"
        "不是实际候选重排序效果。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
