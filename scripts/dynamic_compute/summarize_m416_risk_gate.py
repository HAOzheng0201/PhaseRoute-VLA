"""Apply explicit go/no-go gates to M4.16 nested-LOTO risk analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--feature-name", default="phase_stage")
    parser.add_argument("--min-mae-relative-improvement", type=float, default=0.10)
    parser.add_argument("--min-pearson", type=float, default=0.40)
    parser.add_argument("--min-r2", type=float, default=0.20)
    parser.add_argument("--min-nonnegative-task-r2-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_gate_summary(
    analysis: dict[str, Any],
    *,
    feature_name: str,
    min_mae_relative_improvement: float,
    min_pearson: float,
    min_r2: float,
    min_nonnegative_task_r2_fraction: float,
) -> dict[str, Any]:
    feature = analysis["feature_analyses"][feature_name]
    metrics = feature["nested_loto_metrics"]
    constant = analysis["constant_nested_loto_metrics"]
    relative_improvement = 1.0 - float(metrics["mae"]) / float(constant["mae"])
    task_r2 = {
        str(int(fold["held_task"])): float(fold["metrics"]["r2"])
        for fold in feature["folds"]
    }
    nonnegative_fraction = sum(value >= 0.0 for value in task_r2.values()) / len(
        task_r2
    )
    gates = {
        "mae_relative_improvement": (
            relative_improvement >= min_mae_relative_improvement
        ),
        "pearson": float(metrics["pearson"]) >= min_pearson,
        "r2": float(metrics["r2"]) >= min_r2,
        "nonnegative_task_r2_fraction": (
            nonnegative_fraction >= min_nonnegative_task_r2_fraction
        ),
    }
    finite = all(
        math.isfinite(value)
        for value in (
            relative_improvement,
            float(metrics["pearson"]),
            float(metrics["r2"]),
            nonnegative_fraction,
            *task_r2.values(),
        )
    )
    return {
        "status": "PASS" if analysis.get("status") == "PASS" and finite else "FAIL",
        "scope": "m416_phase_drift_risk_go_no_go",
        "feature_name": feature_name,
        "records": int(analysis["records"]),
        "tasks": list(analysis["tasks"]),
        "constant_nested_loto_metrics": constant,
        "candidate_nested_loto_metrics": metrics,
        "mae_relative_improvement": relative_improvement,
        "task_r2": task_r2,
        "nonnegative_task_r2_fraction": nonnegative_fraction,
        "thresholds": {
            "min_mae_relative_improvement": min_mae_relative_improvement,
            "min_pearson": min_pearson,
            "min_r2": min_r2,
            "min_nonnegative_task_r2_fraction": (
                min_nonnegative_task_r2_fraction
            ),
        },
        "gates": gates,
        "online_rollout_recommended": finite and all(gates.values()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    for value in (
        args.min_mae_relative_improvement,
        args.min_nonnegative_task_r2_fraction,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("fractional gate thresholds must be within [0, 1]")
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    result = build_gate_summary(
        analysis,
        feature_name=args.feature_name,
        min_mae_relative_improvement=args.min_mae_relative_improvement,
        min_pearson=args.min_pearson,
        min_r2=args.min_r2,
        min_nonnegative_task_r2_fraction=(
            args.min_nonnegative_task_r2_fraction
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
