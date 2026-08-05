"""Summarize preregistered cross-seed replay of an M4.22 discordant state."""

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
from scripts.dynamic_compute.summarize_m422_three_arm import POLICY_EXPECTATIONS


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
    parser.add_argument("--expected-base-seed", type=int, action="append", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument(
        "--expected-outcome",
        choices=tuple(OPPOSITE_DISCORDANCE),
        required=True,
    )
    parser.add_argument("--min-expected-matches", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_policy_rows(items, expected_policy, *, task_id, episode_idx):
    expected = POLICY_EXPECTATIONS[expected_policy]
    rows = {}
    for result_path, result in items:
        if result.get("status") != "PASS" or result.get("policy") != expected_policy:
            raise ValueError(f"invalid {expected_policy} result: {result_path}")
        for field, value in expected.items():
            if result.get(field) != value:
                raise ValueError(f"semantic mismatch in {result_path}: {field}")
        if result.get("task_ids") != [task_id]:
            raise ValueError(f"unexpected task selection in {result_path}")
        if result.get("episode_indices") != [episode_idx]:
            raise ValueError(f"unexpected episode selection in {result_path}")
        if int(result.get("telemetry_errors", -1)) != 0:
            raise ValueError(f"telemetry error in {result_path}")
        for row in result["episode_records"]:
            key = int(result["seed"])
            if key in rows:
                raise ValueError(f"duplicate {expected_policy} seed {key}")
            if row.get("status") != "PASS":
                raise ValueError(f"episode engineering failure in {result_path}")
            rows[key] = (result_path, result, row)
    return rows


def build_summary(
    early_items,
    full_items,
    *,
    expected_base_seeds,
    task_id,
    episode_idx,
    expected_outcome,
    min_expected_matches,
) -> dict[str, Any]:
    seeds = tuple(int(value) for value in expected_base_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("expected base seeds must be non-empty and unique")
    if not 1 <= min_expected_matches <= len(seeds):
        raise ValueError("invalid replication threshold")
    early = _load_policy_rows(
        early_items, "early_exit", task_id=task_id, episode_idx=episode_idx
    )
    full = _load_policy_rows(
        full_items, "full_depth", task_id=task_id, episode_idx=episode_idx
    )
    if early.keys() != full.keys() or set(early) != set(seeds):
        raise ValueError("cross-seed early/full grid differs from preregistration")

    rows = []
    for seed in sorted(early):
        early_path, early_result, early_row = early[seed]
        full_path, full_result, full_row = full[seed]
        for field in (
            "checkpoint_sha256",
            "task_suite",
            "seed",
            "episodes_per_task",
            "episode_indices",
            "fm_steps",
        ):
            if early_result.get(field) != full_result.get(field):
                raise ValueError(f"paired replay metadata differs for seed {seed}: {field}")
        for field in ("task_id", "episode_idx", "episode_seed", "initial_state_sha256"):
            if early_row.get(field) != full_row.get(field):
                raise ValueError(f"paired episode differs for seed {seed}: {field}")
        early_success = bool(early_row["success"])
        full_success = bool(full_row["success"])
        rows.append(
            {
                "base_seed": seed,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "episode_seed": int(early_row["episode_seed"]),
                "initial_state_sha256": early_row["initial_state_sha256"],
                "outcome": outcome_name(early_success, full_success),
                "early_exit": {
                    "result_path": early_path,
                    "success": early_success,
                    "policy_calls": int(early_row["policy_calls"]),
                    "exit_layer_sequence": early_row.get("exit_layer_sequence"),
                },
                "full_depth": {
                    "result_path": full_path,
                    "success": full_success,
                    "policy_calls": int(full_row["policy_calls"]),
                },
            }
        )

    counts = Counter(row["outcome"] for row in rows)
    opposite_outcome = OPPOSITE_DISCORDANCE[expected_outcome]
    expected_matches = int(counts[expected_outcome])
    opposite_matches = int(counts[opposite_outcome])
    replicated = expected_matches >= min_expected_matches and opposite_matches == 0
    return {
        "status": "PASS",
        "scope": "m422_discordant_state_cross_seed_summary",
        "task_id": task_id,
        "episode_idx": episode_idx,
        "base_seeds": sorted(seeds),
        "paired_replays": len(rows),
        "expected_outcome_from_main_run": expected_outcome,
        "outcome_counts": {name: int(counts[name]) for name in OUTCOMES},
        "early_successes": sum(row["early_exit"]["success"] for row in rows),
        "full_successes": sum(row["full_depth"]["success"] for row in rows),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(
            counts["early_exit_failure_suspected"],
            counts["full_depth_regression_or_trajectory_difference"],
        ),
        "replication_rule": {
            "repeat_seeds": len(seeds),
            "min_expected_matches": min_expected_matches,
            "max_opposite_discordance_matches": 0,
        },
        "expected_matches": expected_matches,
        "opposite_discordance_matches": opposite_matches,
        "replicated": replicated,
        "decision": (
            "discordant_state_replicated"
            if replicated
            else "discordant_state_not_replicated"
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
        expected_base_seeds=args.expected_base_seed,
        task_id=args.task_id,
        episode_idx=args.episode_index,
        expected_outcome=args.expected_outcome,
        min_expected_matches=args.min_expected_matches,
    )
    result["input_sha256"] = {
        str(path.resolve()): sha256_file(path)
        for path in (*args.early_result, *args.full_result)
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
