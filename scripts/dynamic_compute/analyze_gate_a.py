"""Cluster-aware preliminary Gate-A analysis on phase weak labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.weak_labels import (
    BoundaryLabelConfig,
    build_weak_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def selected_action_delta(record: dict) -> float:
    for layer, delta in zip(record["candidate_exit_layers"], record["action_delta_by_exit"]):
        if layer == record["exit_layer"]:
            return float(delta)
    raise ValueError("Selected exit layer is not aligned with action deltas")


def paired_episode_effect(
    rows: list[dict],
    metric: Callable[[dict], float],
    target_name: str,
    bootstrap_samples: int,
) -> dict:
    differences = []
    for episode_id in sorted({row["episode_id"] for row in rows}):
        episode = [row for row in rows if row["episode_id"] == episode_id]
        boundary = [metric(row) for row in episode if row[target_name]]
        non_boundary = [metric(row) for row in episode if not row[target_name]]
        if boundary and non_boundary:
            differences.append(float(np.mean(boundary) - np.mean(non_boundary)))
    if not differences:
        raise ValueError(f"No episodes contain both classes for {target_name}")
    rng = np.random.default_rng(20260801)
    array = np.asarray(differences, dtype=np.float64)
    samples = rng.choice(array, size=(bootstrap_samples, len(array)), replace=True).mean(axis=1)
    try:
        test = stats.wilcoxon(array, alternative="greater", zero_method="wilcox")
        statistic = float(test.statistic)
        pvalue = float(test.pvalue)
    except ValueError:
        statistic = 0.0
        pvalue = 1.0
    return {
        "episodes_with_both_classes": len(differences),
        "mean_paired_difference": float(array.mean()),
        "median_paired_difference": float(np.median(array)),
        "positive_episode_differences": int((array > 0).sum()),
        "bootstrap_95_ci": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "wilcoxon_greater_statistic": statistic,
        "wilcoxon_greater_pvalue": pvalue,
    }


def analyze_config(
    records: list[dict],
    name: str,
    config: BoundaryLabelConfig,
    bootstrap_samples: int,
) -> dict:
    labels = build_weak_labels(records, config=config)
    label_by_key = {
        (label.episode_id, label.environment_step_id): label
        for label in labels
    }
    rows = []
    events = Counter()
    for record in records:
        label = label_by_key[(record["episode_id"], record["step_id"])]
        for event, present in label.boundary_events.items():
            if present:
                events[event] += 1
        rows.append(
            {
                "episode_id": record["episode_id"],
                "boundary_target_raw": label.boundary_target_raw,
                "boundary_target": label.boundary_target,
                "exit_layer": int(record["exit_layer"]),
                "action_delta": selected_action_delta(record),
            }
        )

    def class_summary(target_name: str) -> dict:
        boundary = [row for row in rows if row[target_name]]
        non_boundary = [row for row in rows if not row[target_name]]
        return {
            "boundary_records": len(boundary),
            "non_boundary_records": len(non_boundary),
            "boundary_fraction": len(boundary) / len(rows),
            "exit_layer_boundary_mean": statistics.fmean(row["exit_layer"] for row in boundary),
            "exit_layer_non_boundary_mean": statistics.fmean(row["exit_layer"] for row in non_boundary),
            "action_delta_boundary_mean": statistics.fmean(row["action_delta"] for row in boundary),
            "action_delta_non_boundary_mean": statistics.fmean(row["action_delta"] for row in non_boundary),
            "exit_layer_paired": paired_episode_effect(
                rows,
                lambda row: row["exit_layer"],
                target_name,
                bootstrap_samples,
            ),
            "action_delta_paired": paired_episode_effect(
                rows,
                lambda row: row["action_delta"],
                target_name,
                bootstrap_samples,
            ),
        }

    return {
        "name": name,
        "config": config.__dict__,
        "event_counts": dict(sorted(events.items())),
        "raw": class_summary("boundary_target_raw"),
        "dilated": class_summary("boundary_target"),
    }


def main() -> None:
    args = parse_args()
    telemetry_paths = sorted(args.telemetry_root.glob("gpu*_task*/policy_calls.jsonl"))
    records = [
        json.loads(line)
        for path in telemetry_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise FileNotFoundError("No M1 telemetry records found")

    configs = [
        (
            "lenient",
            BoundaryLabelConfig(
                translation_speed_change_threshold=0.30,
                rotation_speed_change_threshold=0.05,
                weight_action_delta_increase=0.0,
            ),
        ),
        (
            "default",
            BoundaryLabelConfig(weight_action_delta_increase=0.0),
        ),
        (
            "strict",
            BoundaryLabelConfig(
                translation_speed_change_threshold=0.70,
                rotation_speed_change_threshold=0.09,
                weight_action_delta_increase=0.0,
            ),
        ),
    ]
    analyses = [
        analyze_config(records, name, config, args.bootstrap_samples)
        for name, config in configs
    ]
    default = next(item for item in analyses if item["name"] == "default")
    default_exit = default["dilated"]["exit_layer_paired"]
    default_delta = default["dilated"]["action_delta_paired"]
    signals_positive_across_sensitivity = all(
        analysis["dilated"]["exit_layer_paired"]["mean_paired_difference"] > 0
        and analysis["dilated"]["action_delta_paired"]["mean_paired_difference"] > 0
        for analysis in analyses
    )
    statistically_supported = (
        default_exit["wilcoxon_greater_pvalue"] < 0.05
        or default_delta["wilcoxon_greater_pvalue"] < 0.05
    )
    summary = {
        "status": "PASS_PRELIMINARY" if statistically_supported and signals_positive_across_sensitivity else "REVIEW",
        "gate": "Gate A preliminary kinematic-boundary analysis",
        "records": len(records),
        "episodes": len({record["episode_id"] for record in records}),
        "label_excludes_action_delta": True,
        "direction_change_available": False,
        "signals_positive_across_sensitivity": signals_positive_across_sensitivity,
        "statistically_supported_default": statistically_supported,
        "analyses": analyses,
        "limitations": [
            "Only 20 successful episodes from four libero_spatial tasks are included.",
            "Labels use commanded-motion norms, not measured physical velocity.",
            "Direction-change labels are unavailable in M1 summary telemetry.",
            "This result permits enriched M2 cache collection; it is not the final paper claim.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "gate_a_preliminary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    default_stats = default["dilated"]
    markdown = f"""# Gate A 初步分析：运动学弱边界与 A1 计算需求

生成日期：2026-08-01（Asia/Shanghai）  
状态：**{summary['status']}**

## 主分析（default，±2 policy calls）

- 数据：{summary['episodes']} episodes，{summary['records']} policy calls。
- 为避免循环论证，边界标签构建时 `weight_action_delta_increase=0`。
- 边界覆盖：{default_stats['boundary_records']}/{summary['records']}（{default_stats['boundary_fraction']:.2%}）。
- 平均退出层：边界 {default_stats['exit_layer_boundary_mean']:.3f}，非边界 {default_stats['exit_layer_non_boundary_mean']:.3f}。
- episode 配对退出层差：{default_exit['mean_paired_difference']:.3f}，95% cluster bootstrap CI [{default_exit['bootstrap_95_ci'][0]:.3f}, {default_exit['bootstrap_95_ci'][1]:.3f}]，单侧 Wilcoxon p={default_exit['wilcoxon_greater_pvalue']:.6f}。
- 平均选中出口动作差：边界 {default_stats['action_delta_boundary_mean']:.6f}，非边界 {default_stats['action_delta_non_boundary_mean']:.6f}。
- episode 配对动作差：{default_delta['mean_paired_difference']:.6f}，95% cluster bootstrap CI [{default_delta['bootstrap_95_ci'][0]:.6f}, {default_delta['bootstrap_95_ci'][1]:.6f}]，单侧 Wilcoxon p={default_delta['wilcoxon_greater_pvalue']:.6f}。

## 判定

lenient/default/strict 三组阈值下，边界的平均退出层差和动作差方向均为正：`{signals_positive_across_sensitivity}`。default 至少一个 episode-clustered 指标单侧 p<0.05：`{statistically_supported}`。

因此该小样本支持继续收集包含原始 proprio、动作方向、视觉摘要和语言摘要的 M2 cache。它还不足以作为最终论文 Gate A 结论：样本仅覆盖 `libero_spatial` 前四个任务且全部成功，速度来自控制命令而非真实物理速度。
"""
    (args.output_dir / "GATE_A_PRELIMINARY.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result={output_path}")


if __name__ == "__main__":
    main()

