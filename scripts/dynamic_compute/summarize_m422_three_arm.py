"""Audit A1 early-exit, RP-PEP, and full-depth closed-loop outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.replay_m420b_rp_pep import summarize_latencies
from scripts.dynamic_compute.summarize_m418_paired_counterfactual import (
    OUTCOMES,
    exact_mcnemar_p_value,
    outcome_name,
    wilson_interval,
)


POLICY_EXPECTATIONS = {
    "early_exit": {
        "scope": "m418_persistent_closed_loop_counterfactual_shard",
        "model_class": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
        "early_exit_enabled": True,
        "productive_exit_enabled": False,
    },
    "rp_pep": {
        "scope": "m420b_rp_pep_closed_loop_shard",
        "model_class": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
        "early_exit_enabled": True,
        "productive_exit_enabled": True,
    },
    "full_depth": {
        "scope": "m418_persistent_closed_loop_counterfactual_shard",
        "model_class": "a1.vla.affordvla.AffordVLA",
        "early_exit_enabled": False,
        "productive_exit_enabled": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-result", type=Path, action="append", required=True)
    parser.add_argument("--rp-pep-result", type=Path, action="append", required=True)
    parser.add_argument("--full-result", type=Path, action="append", required=True)
    parser.add_argument("--expected-episode-index", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(items, expected_policy):
    expected = POLICY_EXPECTATIONS[expected_policy]
    rows = {}
    for result_path, result in items:
        if result.get("status") != "PASS":
            raise ValueError(f"non-PASS shard: {result_path}")
        if result.get("policy") != expected_policy:
            raise ValueError(f"unexpected policy in {result_path}")
        for field, value in expected.items():
            if result.get(field) != value:
                raise ValueError(
                    f"{expected_policy} semantic mismatch in {result_path}: {field}"
                )
        if result.get("vision_aggregation_enabled") is not False:
            raise ValueError(f"vision aggregation enabled in {result_path}")
        if int(result.get("telemetry_errors", -1)) != 0:
            raise ValueError(f"telemetry errors in {result_path}")
        for row in result["episode_records"]:
            key = (int(row["task_id"]), int(row["episode_idx"]))
            if key in rows:
                raise ValueError(f"duplicate {expected_policy} episode {key}")
            if row.get("status") != "PASS":
                raise ValueError(f"episode engineering failure for {expected_policy} {key}")
            if int(row.get("policy_calls", 0)) <= 0:
                raise ValueError(f"missing policy calls for {expected_policy} {key}")
            if len(row.get("action_chunk_sha256", [])) != int(row["policy_calls"]):
                raise ValueError(f"missing action hashes for {expected_policy} {key}")
            rows[key] = (result_path, result, row)
    return rows


def _same_metadata(reference_result, reference_row, other_result, other_row, key):
    for field in (
        "checkpoint_sha256",
        "task_suite",
        "seed",
        "episodes_per_task",
        "episode_indices",
        "fm_steps",
    ):
        if reference_result.get(field) != other_result.get(field):
            raise ValueError(f"paired shard metadata differs for {key}: {field}")
    for field in ("task_id", "episode_idx", "episode_seed", "initial_state_sha256"):
        if reference_row.get(field) != other_row.get(field):
            raise ValueError(f"paired episode metadata differs for {key}: {field}")


def build_summary(
    early_items,
    rp_pep_items,
    full_items,
    *,
    expected_task_ids=range(10),
    expected_episode_indices=(27, 28, 29, 30, 31),
) -> dict[str, Any]:
    early = _load_rows(early_items, "early_exit")
    rp_pep = _load_rows(rp_pep_items, "rp_pep")
    full = _load_rows(full_items, "full_depth")
    if early.keys() != rp_pep.keys() or early.keys() != full.keys():
        raise ValueError("early/RP-PEP/full-depth episode grids differ")

    rows = []
    equivalence = Counter()
    latencies = {"early_exit": [], "rp_pep": [], "full_depth": []}
    for key in sorted(early):
        early_path, early_result, early_row = early[key]
        rp_path, rp_result, rp_row = rp_pep[key]
        full_path, full_result, full_row = full[key]
        _same_metadata(early_result, early_row, rp_result, rp_row, key)
        _same_metadata(early_result, early_row, full_result, full_row, key)

        success_match = bool(early_row["success"]) == bool(rp_row["success"])
        action_match = (
            list(early_row["action_chunk_sha256"])
            == list(rp_row["action_chunk_sha256"])
            and bool(early_row["action_chunk_sha256"])
        )
        exit_match = (
            list(early_row.get("exit_layer_sequence", []))
            == list(rp_row.get("exit_layer_sequence", []))
            and bool(early_row.get("exit_layer_sequence", []))
        )
        call_match = (
            int(early_row["policy_calls"])
            == int(rp_row["policy_calls"])
            == len(early_row["action_chunk_sha256"])
            == len(early_row.get("exit_layer_sequence", []))
        )
        for name, matched in (
            ("success_mismatches", success_match),
            ("action_chunk_sha256_mismatches", action_match),
            ("exit_layer_sequence_mismatches", exit_match),
            ("policy_call_count_mismatches", call_match),
        ):
            equivalence[name] += int(not matched)

        early_success = bool(early_row["success"])
        full_success = bool(full_row["success"])
        outcome = outcome_name(early_success, full_success)
        rows.append(
            {
                "task_id": key[0],
                "episode_idx": key[1],
                "episode_seed": int(early_row["episode_seed"]),
                "initial_state_sha256": early_row["initial_state_sha256"],
                "early_rp_equivalent": success_match
                and action_match
                and exit_match
                and call_match,
                "outcome_early_vs_full": outcome,
                "early_exit": {
                    "result_path": early_path,
                    "success": early_success,
                    "policy_calls": int(early_row["policy_calls"]),
                    "fm_calls_total": int(early_row["fm_calls_total"]),
                },
                "rp_pep": {
                    "result_path": rp_path,
                    "success": bool(rp_row["success"]),
                    "policy_calls": int(rp_row["policy_calls"]),
                    "fm_calls_total": int(rp_row["fm_calls_total"]),
                },
                "full_depth": {
                    "result_path": full_path,
                    "success": full_success,
                    "policy_calls": int(full_row["policy_calls"]),
                },
            }
        )
        for policy, row in (
            ("early_exit", early_row),
            ("rp_pep", rp_row),
            ("full_depth", full_row),
        ):
            latencies[policy].extend(
                float(value) for value in row.get("latency_ms_by_call", [])
            )

    expected_task_ids = tuple(int(value) for value in expected_task_ids)
    expected_episode_indices = tuple(int(value) for value in expected_episode_indices)
    if not expected_task_ids or len(expected_task_ids) != len(set(expected_task_ids)):
        raise ValueError("expected task ids must be non-empty and unique")
    if (
        not expected_episode_indices
        or len(expected_episode_indices) != len(set(expected_episode_indices))
    ):
        raise ValueError("expected episode indices must be non-empty and unique")
    expected_grid = {
        (task_id, episode_idx)
        for task_id in expected_task_ids
        for episode_idx in expected_episode_indices
    }
    outcome_counter = Counter(row["outcome_early_vs_full"] for row in rows)
    outcome_counts = {name: int(outcome_counter[name]) for name in OUTCOMES}
    suspected = outcome_counts["early_exit_failure_suspected"]
    reverse = outcome_counts["full_depth_regression_or_trajectory_difference"]
    early_failures = outcome_counts["both_fail"] + suspected
    equivalence_counts = {
        name: int(equivalence[name])
        for name in (
            "success_mismatches",
            "action_chunk_sha256_mismatches",
            "exit_layer_sequence_mismatches",
            "policy_call_count_mismatches",
        )
    }
    gates = {
        "complete_expected_three_arm_grid": set(early) == expected_grid,
        "early_rp_trajectory_equivalence": all(
            value == 0 for value in equivalence_counts.values()
        ),
        "full_depth_never_exit_semantics": True,
    }
    discordant_rows = [
        row
        for row in rows
        if row["outcome_early_vs_full"]
        in {
            "early_exit_failure_suspected",
            "full_depth_regression_or_trajectory_difference",
        }
    ]
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "m422_three_arm_full_depth_attribution_summary",
        "paired_states": len(rows),
        "analyzed_rollouts": len(rows) * 3,
        "new_full_depth_rollouts": len(rows),
        "tasks": sorted({row["task_id"] for row in rows}),
        "episode_indices": sorted({row["episode_idx"] for row in rows}),
        "expected_tasks": list(expected_task_ids),
        "expected_episode_indices": list(expected_episode_indices),
        "successes": {
            "early_exit": sum(row["early_exit"]["success"] for row in rows),
            "rp_pep": sum(row["rp_pep"]["success"] for row in rows),
            "full_depth": sum(row["full_depth"]["success"] for row in rows),
        },
        "early_rp_equivalence": equivalence_counts,
        "outcome_counts_early_vs_full": outcome_counts,
        "observed_early_exit_failures": early_failures,
        "failures_fixed_by_full_depth": suspected,
        "attributable_fraction_among_early_failures": wilson_interval(
            suspected, early_failures
        ),
        "suspected_failure_rate_among_all_states": wilson_interval(
            suspected, len(rows)
        ),
        "success_rate_delta_full_minus_early": (
            sum(row["full_depth"]["success"] for row in rows)
            - sum(row["early_exit"]["success"] for row in rows)
        ) / len(rows),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(suspected, reverse),
        "policy_calls": {
            policy: sum(row[policy]["policy_calls"] for row in rows)
            for policy in ("early_exit", "rp_pep", "full_depth")
        },
        "policy_latency_descriptive": {
            policy: summarize_latencies(values) for policy, values in latencies.items()
        },
        "discordant_states": discordant_rows,
        "cross_seed_followup_required": bool(discordant_rows),
        "preregistered_followup_base_seeds": [20266804, 20267804, 20268804],
        "gates": gates,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    item_groups = []
    for paths in (args.early_result, args.rp_pep_result, args.full_result):
        item_groups.append(
            [
                (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
                for path in paths
            ]
        )
    result = build_summary(
        *item_groups,
        expected_episode_indices=args.expected_episode_index,
    )
    result["input_sha256"] = {
        str(path.resolve()): sha256_file(path)
        for path in (
            *args.early_result,
            *args.rp_pep_result,
            *args.full_result,
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
