"""Summarize M4.19 cross-seed replays of the three discordant states."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.summarize_m418_paired_counterfactual import (
    OUTCOMES,
    exact_mcnemar_p_value,
    outcome_name,
    sha256_file,
)


EXPECTED_OUTCOMES = {
    2: "early_exit_failure_suspected",
    14: "full_depth_regression_or_trajectory_difference",
    22: "full_depth_regression_or_trajectory_difference",
}
OPPOSITE_DISCORDANCE = {
    "early_exit_failure_suspected": (
        "full_depth_regression_or_trajectory_difference"
    ),
    "full_depth_regression_or_trajectory_difference": (
        "early_exit_failure_suspected"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-result", type=Path, action="append", required=True)
    parser.add_argument("--full-result", type=Path, action="append", required=True)
    parser.add_argument("--min-repeat-seeds", type=int, default=3)
    parser.add_argument("--min-expected-matches", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_policy_rows(
    items: list[tuple[str, dict[str, Any]]], expected_policy: str
) -> dict[tuple[int, int, int], tuple[str, dict[str, Any], dict[str, Any]]]:
    rows = {}
    expected_indices = list(EXPECTED_OUTCOMES)
    for result_path, result in items:
        if result.get("status") != "PASS":
            raise ValueError(f"non-PASS shard: {result_path}")
        if result.get("scope") != "m418_persistent_closed_loop_counterfactual_shard":
            raise ValueError(f"unexpected shard scope: {result_path}")
        if result.get("policy") != expected_policy:
            raise ValueError(f"unexpected policy in {result_path}")
        if result.get("task_ids") != [5]:
            raise ValueError(f"M4.19 result must contain only task5: {result_path}")
        if result.get("episode_indices") != expected_indices:
            raise ValueError(f"unexpected episode selection in {result_path}")
        for row in result["episode_records"]:
            key = (int(result["seed"]), int(row["task_id"]), int(row["episode_idx"]))
            if key in rows:
                raise ValueError(f"duplicate {expected_policy} replay {key}")
            rows[key] = (result_path, result, row)
    return rows


def build_summary(
    early_items: list[tuple[str, dict[str, Any]]],
    full_items: list[tuple[str, dict[str, Any]]],
    *,
    min_repeat_seeds: int,
    min_expected_matches: int,
) -> dict[str, Any]:
    if min_repeat_seeds <= 0 or not 1 <= min_expected_matches <= min_repeat_seeds:
        raise ValueError("invalid replay replication thresholds")
    early = _load_policy_rows(early_items, "early_exit")
    full = _load_policy_rows(full_items, "full_depth")
    if early.keys() != full.keys():
        raise ValueError("early/full replay grids differ")
    seeds = sorted({key[0] for key in early})
    if len(seeds) < min_repeat_seeds:
        raise ValueError("not enough distinct replay seeds")
    expected_grid = {
        (seed, 5, episode_idx)
        for seed in seeds
        for episode_idx in EXPECTED_OUTCOMES
    }
    if early.keys() != expected_grid:
        raise ValueError("replay grid is incomplete")

    rows = []
    for key in sorted(early):
        early_path, early_result, early_row = early[key]
        full_path, full_result, full_row = full[key]
        for field in (
            "checkpoint_sha256",
            "task_suite",
            "seed",
            "episodes_per_task",
            "episode_indices",
            "fm_steps",
        ):
            if early_result.get(field) != full_result.get(field):
                raise ValueError(f"paired replay metadata differs for {key}")
        for field in (
            "task_id",
            "episode_idx",
            "episode_seed",
            "initial_state_sha256",
        ):
            if early_row.get(field) != full_row.get(field):
                raise ValueError(f"paired replay episode differs for {key}")
        if early_row.get("status") != "PASS" or full_row.get("status") != "PASS":
            raise ValueError(f"replay engineering failure for {key}")
        early_success = bool(early_row["success"])
        full_success = bool(full_row["success"])
        rows.append(
            {
                "base_seed": key[0],
                "task_id": key[1],
                "episode_idx": key[2],
                "episode_seed": int(early_row["episode_seed"]),
                "initial_state_sha256": early_row["initial_state_sha256"],
                "outcome": outcome_name(early_success, full_success),
                "early_exit": {
                    "result_path": early_path,
                    "success": early_success,
                    "policy_calls": int(early_row["policy_calls"]),
                    "exit_mean_ratio": early_row.get("exit_mean_ratio"),
                    "exit_layer_counts": early_row.get("exit_layer_counts"),
                    "fm_calls_total": early_row.get("fm_calls_total"),
                },
                "full_depth": {
                    "result_path": full_path,
                    "success": full_success,
                    "policy_calls": int(full_row["policy_calls"]),
                },
            }
        )

    state_summaries = {}
    for episode_idx, expected_outcome in EXPECTED_OUTCOMES.items():
        selected = [row for row in rows if row["episode_idx"] == episode_idx]
        counts = Counter(row["outcome"] for row in selected)
        opposite = OPPOSITE_DISCORDANCE[expected_outcome]
        expected_matches = int(counts[expected_outcome])
        opposite_matches = int(counts[opposite])
        state_summaries[str(episode_idx)] = {
            "expected_outcome_from_m418_m418b": expected_outcome,
            "repeat_seeds": len(selected),
            "outcome_counts": {name: int(counts[name]) for name in OUTCOMES},
            "expected_matches": expected_matches,
            "opposite_discordance_matches": opposite_matches,
            "replication_threshold": min_expected_matches,
            "replicated": (
                expected_matches >= min_expected_matches and opposite_matches == 0
            ),
        }

    aggregate = Counter(row["outcome"] for row in rows)
    positive_replicated = state_summaries["2"]["replicated"]
    return {
        "status": "PASS",
        "scope": "m419_discordant_cross_seed_replay_summary",
        "base_seeds": seeds,
        "repeat_seed_count": len(seeds),
        "paired_replays": len(rows),
        "episode_indices": list(EXPECTED_OUTCOMES),
        "outcome_counts": {name: int(aggregate[name]) for name in OUTCOMES},
        "early_successes": sum(row["early_exit"]["success"] for row in rows),
        "full_successes": sum(row["full_depth"]["success"] for row in rows),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(
            aggregate["early_exit_failure_suspected"],
            aggregate["full_depth_regression_or_trajectory_difference"],
        ),
        "replication_rule": {
            "min_repeat_seeds": min_repeat_seeds,
            "min_expected_matches": min_expected_matches,
            "max_opposite_discordance_matches": 0,
        },
        "state_summaries": state_summaries,
        "causal_positive_state_replicated": positive_replicated,
        "all_original_discordances_replicated": all(
            summary["replicated"] for summary in state_summaries.values()
        ),
        "risk_head_training_recommended": False,
        "decision": (
            "causal_positive_replicated_but_repeats_are_not_independent_states"
            if positive_replicated
            else "causal_positive_not_replicated_stop_risk_head_route"
        ),
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
        min_repeat_seeds=args.min_repeat_seeds,
        min_expected_matches=args.min_expected_matches,
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
