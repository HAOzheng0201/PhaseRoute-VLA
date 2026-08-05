"""Summarize the preregistered M4.20b baseline/RP-PEP closed-loop pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.replay_m420b_rp_pep import summarize_latencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", type=Path, action="append", required=True)
    parser.add_argument("--rp-pep-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episode-index", type=int, action="append")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(items, expected_policy):
    rows = {}
    for result_path, result in items:
        if result.get("status") != "PASS":
            raise ValueError(f"non-PASS shard: {result_path}")
        if result.get("policy") != expected_policy:
            raise ValueError(f"unexpected policy in {result_path}")
        if int(result.get("telemetry_errors", -1)) != 0:
            raise ValueError(f"telemetry errors in {result_path}")
        for row in result["episode_records"]:
            key = (int(row["task_id"]), int(row["episode_idx"]))
            if key in rows:
                raise ValueError(f"duplicate {expected_policy} episode {key}")
            rows[key] = (result_path, result, row)
    return rows


def build_summary(
    baseline_items,
    rp_pep_items,
    *,
    expected_task_ids=range(10),
    expected_episode_indices=(27, 28),
) -> dict[str, Any]:
    baseline = _load_rows(baseline_items, "early_exit")
    rp_pep = _load_rows(rp_pep_items, "rp_pep")
    if baseline.keys() != rp_pep.keys():
        raise ValueError("baseline/RP-PEP episode grids differ")

    rows = []
    baseline_latencies = []
    sparse_latencies = []
    for key in sorted(baseline):
        baseline_path, baseline_result, baseline_row = baseline[key]
        sparse_path, sparse_result, sparse_row = rp_pep[key]
        for field in (
            "checkpoint_sha256",
            "task_suite",
            "seed",
            "episodes_per_task",
            "episode_indices",
            "fm_steps",
        ):
            if baseline_result.get(field) != sparse_result.get(field):
                raise ValueError(f"paired shard metadata differs for {key}: {field}")
        for field in (
            "task_id",
            "episode_idx",
            "episode_seed",
            "initial_state_sha256",
        ):
            if baseline_row.get(field) != sparse_row.get(field):
                raise ValueError(f"paired episode metadata differs for {key}: {field}")
        baseline_hashes = list(baseline_row.get("action_chunk_sha256", []))
        sparse_hashes = list(sparse_row.get("action_chunk_sha256", []))
        baseline_exits = list(baseline_row.get("exit_layer_sequence", []))
        sparse_exits = list(sparse_row.get("exit_layer_sequence", []))
        baseline_call_latencies = [
            float(value) for value in baseline_row.get("latency_ms_by_call", [])
        ]
        sparse_call_latencies = [
            float(value) for value in sparse_row.get("latency_ms_by_call", [])
        ]
        baseline_latencies.extend(baseline_call_latencies)
        sparse_latencies.extend(sparse_call_latencies)
        action_match = baseline_hashes == sparse_hashes and bool(baseline_hashes)
        exit_match = baseline_exits == sparse_exits and bool(baseline_exits)
        call_match = (
            int(baseline_row["policy_calls"]) == int(sparse_row["policy_calls"])
            == len(baseline_hashes)
            == len(baseline_exits)
        )
        rows.append(
            {
                "task_id": key[0],
                "episode_idx": key[1],
                "episode_seed": int(baseline_row["episode_seed"]),
                "initial_state_sha256": baseline_row["initial_state_sha256"],
                "success_match": bool(baseline_row["success"])
                == bool(sparse_row["success"]),
                "action_chunk_sha256_match": action_match,
                "exit_layer_sequence_match": exit_match,
                "policy_calls_match": call_match,
                "baseline": {
                    "result_path": baseline_path,
                    "success": bool(baseline_row["success"]),
                    "policy_calls": int(baseline_row["policy_calls"]),
                    "fm_calls_total": int(baseline_row["fm_calls_total"]),
                    "latency_ms_total": float(baseline_row["latency_ms_total"]),
                    "wall_seconds": float(baseline_row["wall_seconds"]),
                },
                "rp_pep": {
                    "result_path": sparse_path,
                    "success": bool(sparse_row["success"]),
                    "policy_calls": int(sparse_row["policy_calls"]),
                    "fm_calls_total": int(sparse_row["fm_calls_total"]),
                    "latency_ms_total": float(sparse_row["latency_ms_total"]),
                    "wall_seconds": float(sparse_row["wall_seconds"]),
                },
            }
        )

    baseline_fm = sum(row["baseline"]["fm_calls_total"] for row in rows)
    sparse_fm = sum(row["rp_pep"]["fm_calls_total"] for row in rows)
    baseline_latency = summarize_latencies(baseline_latencies)
    sparse_latency = summarize_latencies(sparse_latencies)
    fm_reduction = 1.0 - sparse_fm / baseline_fm
    mean_reduction = 1.0 - sparse_latency["mean_ms"] / baseline_latency["mean_ms"]
    median_reduction = 1.0 - (
        sparse_latency["median_ms"] / baseline_latency["median_ms"]
    )
    equivalence = {
        "success_mismatches": sum(not row["success_match"] for row in rows),
        "action_chunk_sha256_mismatches": sum(
            not row["action_chunk_sha256_match"] for row in rows
        ),
        "exit_layer_sequence_mismatches": sum(
            not row["exit_layer_sequence_match"] for row in rows
        ),
        "policy_call_count_mismatches": sum(
            not row["policy_calls_match"] for row in rows
        ),
    }
    expected_task_ids = tuple(int(value) for value in expected_task_ids)
    expected_episode_indices = tuple(
        int(value) for value in expected_episode_indices
    )
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
    gates = {
        "complete_20_pair_grid": set(baseline) == expected_grid,
        "trajectory_equivalence": all(value == 0 for value in equivalence.values()),
        "fm_reduction_at_least_35_percent": fm_reduction >= 0.35,
        "weighted_mean_latency_reduction_at_least_15_percent": mean_reduction >= 0.15,
        "median_latency_reduction_at_least_15_percent": median_reduction >= 0.15,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "m420b_rp_pep_paired_closed_loop_summary",
        "paired_episodes": len(rows),
        "total_rollouts": len(rows) * 2,
        "tasks": sorted({row["task_id"] for row in rows}),
        "episode_indices": sorted({row["episode_idx"] for row in rows}),
        "expected_tasks": list(expected_task_ids),
        "expected_episode_indices": list(expected_episode_indices),
        "baseline_successes": sum(row["baseline"]["success"] for row in rows),
        "rp_pep_successes": sum(row["rp_pep"]["success"] for row in rows),
        "equivalence": equivalence,
        "fm_solver_calls": {
            "baseline": baseline_fm,
            "rp_pep": sparse_fm,
            "reduction_fraction": fm_reduction,
        },
        "policy_latency": {
            "baseline": baseline_latency,
            "rp_pep": sparse_latency,
            "weighted_mean_reduction_fraction": mean_reduction,
            "median_reduction_fraction": median_reduction,
        },
        "wall_seconds": {
            "baseline": sum(row["baseline"]["wall_seconds"] for row in rows),
            "rp_pep": sum(row["rp_pep"]["wall_seconds"] for row in rows),
        },
        "gates": gates,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    baseline_items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.baseline_result
    ]
    rp_pep_items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.rp_pep_result
    ]
    result = build_summary(
        baseline_items,
        rp_pep_items,
        expected_episode_indices=(
            args.expected_episode_index
            if args.expected_episode_index is not None
            else (27, 28)
        ),
    )
    result["input_sha256"] = {
        str(path.resolve()): sha256_file(path)
        for path in (*args.baseline_result, *args.rp_pep_result)
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
