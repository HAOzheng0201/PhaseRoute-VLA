"""Summarize M4.18 paired closed-loop early-exit/full-depth episodes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


OUTCOMES = (
    "both_succeed",
    "both_fail",
    "early_exit_failure_suspected",
    "full_depth_regression_or_trajectory_difference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-result", type=Path, action="append", required=True)
    parser.add_argument("--full-result", type=Path, action="append", required=True)
    parser.add_argument("--min-risk-positive-episodes", type=int, default=5)
    parser.add_argument("--min-risk-negative-episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_name(early_success: bool, full_success: bool) -> str:
    if early_success and full_success:
        return "both_succeed"
    if not early_success and not full_success:
        return "both_fail"
    if not early_success and full_success:
        return "early_exit_failure_suspected"
    return "full_depth_regression_or_trajectory_difference"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "estimate": proportion,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "confidence": 0.95,
    }


def exact_mcnemar_p_value(early_fail_full_success: int, early_success_full_fail: int) -> float:
    if early_fail_full_success < 0 or early_success_full_fail < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = early_fail_full_success + early_success_full_fail
    if discordant == 0:
        return 1.0
    tail = min(early_fail_full_success, early_success_full_fail)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _load_policy_rows(
    items: list[tuple[str, dict[str, Any]]], expected_policy: str
) -> dict[tuple[int, int], tuple[str, dict[str, Any], dict[str, Any]]]:
    rows = {}
    for result_path, result in items:
        if result.get("status") != "PASS":
            raise ValueError(f"non-PASS shard: {result_path}")
        if result.get("scope") != "m418_persistent_closed_loop_counterfactual_shard":
            raise ValueError(f"unexpected shard scope: {result_path}")
        if result.get("policy") != expected_policy:
            raise ValueError(f"unexpected policy in {result_path}")
        for row in result["episode_records"]:
            key = (int(row["task_id"]), int(row["episode_idx"]))
            if key in rows:
                raise ValueError(f"duplicate {expected_policy} episode {key}")
            rows[key] = (result_path, result, row)
    return rows


def build_summary(
    early_items: list[tuple[str, dict[str, Any]]],
    full_items: list[tuple[str, dict[str, Any]]],
    *,
    min_risk_positive_episodes: int,
    min_risk_negative_episodes: int,
) -> dict[str, Any]:
    if min_risk_positive_episodes <= 0 or min_risk_negative_episodes <= 0:
        raise ValueError("risk readiness thresholds must be positive")
    early = _load_policy_rows(early_items, "early_exit")
    full = _load_policy_rows(full_items, "full_depth")
    if early.keys() != full.keys():
        raise ValueError("early/full episode grids differ")

    rows = []
    for key in sorted(early):
        early_path, early_result, early_row = early[key]
        full_path, full_result, full_row = full[key]
        paired_result_fields = (
            "checkpoint_sha256",
            "task_suite",
            "seed",
            "episodes_per_task",
            "episode_start_index",
            "episode_indices",
            "fm_steps",
        )
        if any(
            early_result.get(field) != full_result.get(field)
            for field in paired_result_fields
        ):
            raise ValueError(f"paired shard metadata differs for {key}")
        paired_row_fields = ("task_id", "episode_idx", "episode_seed", "initial_state_sha256")
        if any(early_row.get(field) != full_row.get(field) for field in paired_row_fields):
            raise ValueError(f"paired episode metadata differs for {key}")
        if early_row.get("status") != "PASS" or full_row.get("status") != "PASS":
            raise ValueError(f"episode engineering failure for {key}")

        early_success = bool(early_row["success"])
        full_success = bool(full_row["success"])
        rows.append(
            {
                "task_id": key[0],
                "episode_idx": key[1],
                "episode_seed": int(early_row["episode_seed"]),
                "initial_state_sha256": early_row["initial_state_sha256"],
                "outcome": outcome_name(early_success, full_success),
                "early_exit": {
                    "result_path": early_path,
                    "success": early_success,
                    "policy_calls": int(early_row["policy_calls"]),
                    "latency_ms_total": float(early_row["latency_ms_total"]),
                    "exit_mean_ratio": early_row.get("exit_mean_ratio"),
                    "exit_layer_counts": early_row.get("exit_layer_counts"),
                    "fm_calls_total": early_row.get("fm_calls_total"),
                },
                "full_depth": {
                    "result_path": full_path,
                    "success": full_success,
                    "policy_calls": int(full_row["policy_calls"]),
                    "latency_ms_total": float(full_row["latency_ms_total"]),
                },
                "policy_calls_delta_full_minus_early": (
                    int(full_row["policy_calls"]) - int(early_row["policy_calls"])
                ),
            }
        )

    counts = Counter(row["outcome"] for row in rows)
    outcome_counts = {name: int(counts[name]) for name in OUTCOMES}
    early_failures = outcome_counts["both_fail"] + outcome_counts["early_exit_failure_suspected"]
    suspected = outcome_counts["early_exit_failure_suspected"]
    both_succeed = outcome_counts["both_succeed"]
    task_summaries = {}
    for task_id in sorted({row["task_id"] for row in rows}):
        selected = [row for row in rows if row["task_id"] == task_id]
        task_summaries[str(task_id)] = {
            "episodes": len(selected),
            "outcome_counts": {
                name: sum(row["outcome"] == name for row in selected)
                for name in OUTCOMES
            },
            "early_successes": sum(row["early_exit"]["success"] for row in selected),
            "full_successes": sum(row["full_depth"]["success"] for row in selected),
            "early_policy_calls": sum(row["early_exit"]["policy_calls"] for row in selected),
            "full_policy_calls": sum(row["full_depth"]["policy_calls"] for row in selected),
        }

    readiness_gates = {
        "enough_causal_positive_episodes": suspected >= min_risk_positive_episodes,
        "enough_stable_negative_episodes": both_succeed >= min_risk_negative_episodes,
    }
    return {
        "status": "PASS",
        "scope": "m418_paired_closed_loop_counterfactual_summary",
        "paired_episodes": len(rows),
        "tasks": sorted({row["task_id"] for row in rows}),
        "outcome_counts": outcome_counts,
        "early_successes": sum(row["early_exit"]["success"] for row in rows),
        "full_successes": sum(row["full_depth"]["success"] for row in rows),
        "success_rate_delta_full_minus_early": (
            sum(row["full_depth"]["success"] for row in rows)
            - sum(row["early_exit"]["success"] for row in rows)
        ) / len(rows),
        "observed_early_exit_failures": early_failures,
        "failures_fixed_by_full_depth": suspected,
        "attributable_fraction_among_early_failures": wilson_interval(
            suspected, early_failures
        ),
        "suspected_failure_rate_among_all_episodes": wilson_interval(
            suspected, len(rows)
        ),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(
            suspected,
            outcome_counts["full_depth_regression_or_trajectory_difference"],
        ),
        "policy_calls": {
            "early_exit": sum(row["early_exit"]["policy_calls"] for row in rows),
            "full_depth": sum(row["full_depth"]["policy_calls"] for row in rows),
        },
        "paired_policy_calls_delta_mean": statistics.fmean(
            row["policy_calls_delta_full_minus_early"] for row in rows
        ),
        "task_summaries": task_summaries,
        "risk_training_thresholds": {
            "min_risk_positive_episodes": min_risk_positive_episodes,
            "min_risk_negative_episodes": min_risk_negative_episodes,
        },
        "risk_training_readiness_gates": readiness_gates,
        "risk_training_ready": all(readiness_gates.values()),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    early_items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.early_result
    ]
    full_items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.full_result
    ]
    result = build_summary(
        early_items,
        full_items,
        min_risk_positive_episodes=args.min_risk_positive_episodes,
        min_risk_negative_episodes=args.min_risk_negative_episodes,
    )
    result["input_sha256"] = {
        str(path.resolve()): sha256_file(path)
        for path in (*args.early_result, *args.full_result)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
