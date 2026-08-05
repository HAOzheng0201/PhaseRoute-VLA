#!/usr/bin/env python3
"""Summarize a four-GPU, ten-task RP-PEP release smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.release import CHECKPOINT_SHA256, sha256_file  # noqa: E402


EXPECTED_TASKS_BY_GPU = {
    0: (0, 4, 8),
    1: (1, 5, 9),
    2: (2, 6),
    3: (3, 7),
}
EXPECTED_FM_CALLS = {3: 2, 11: 4, 13: 5, 27: 7}


def normalize_uuid(value: str) -> str:
    result = str(value).strip().lower()
    return result[4:] if result.startswith("gpu-") else result


def validate_shard(
    result: Mapping[str, Any],
    *,
    gpu_index: int,
    expected_gpu_uuid: str,
    episode_index: int,
    seed: int,
) -> dict[str, bool]:
    records = result.get("episode_records", [])
    formula_ok = True
    for record in records:
        layers = [int(value) for value in record.get("exit_layer_sequence", [])]
        calls = [int(value) for value in record.get("fm_calls_by_policy_call", [])]
        formula_ok &= len(layers) == len(calls) == int(record.get("policy_calls", -1))
        formula_ok &= all(
            layer in EXPECTED_FM_CALLS and call == EXPECTED_FM_CALLS[layer]
            for layer, call in zip(layers, calls)
        )
    expected_tasks = EXPECTED_TASKS_BY_GPU[gpu_index]
    return {
        "status": result.get("status") == "PASS",
        "scope": result.get("scope") == "m420b_rp_pep_closed_loop_shard",
        "policy": result.get("policy") == "rp_pep",
        "model_class": result.get("model_class")
        == "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
        "productive_exit_enabled": result.get("productive_exit_enabled") is True,
        "vision_aggregation_disabled": result.get("vision_aggregation_enabled")
        is False,
        "checkpoint_sha256": result.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "task_suite": result.get("task_suite") == "libero_spatial",
        "task_grid": tuple(int(value) for value in result.get("task_ids", []))
        == expected_tasks,
        "episode_grid": result.get("episode_indices") == [episode_index],
        "seed": int(result.get("seed", -1)) == seed,
        "fm_steps": int(result.get("fm_steps", -1)) == 10,
        "completed_episodes": int(result.get("completed_episodes", -1))
        == len(expected_tasks),
        "episode_record_count": len(records) == len(expected_tasks),
        "telemetry_errors_zero": int(result.get("telemetry_errors", -1)) == 0,
        "policy_calls_positive": int(result.get("policy_calls", 0)) > 0,
        "gpu_uuid": normalize_uuid(result.get("physical_gpu_uuid_visible", ""))
        == normalize_uuid(expected_gpu_uuid)
        == normalize_uuid(result.get("physical_gpu_uuid_nvidia_smi", "")),
        "rp_pep_fm_formula": formula_ok,
    }


def summarize(
    shard_results: Sequence[Mapping[str, Any]],
    *,
    expected_gpu_uuids: Mapping[int, str],
    episode_index: int,
    seed: int,
) -> dict[str, Any]:
    if len(shard_results) != 4:
        raise ValueError("release smoke requires exactly four shards")
    if set(expected_gpu_uuids) != set(range(4)) or any(
        not normalize_uuid(expected_gpu_uuids[gpu]) for gpu in range(4)
    ):
        raise ValueError("expected_gpu_uuids must define non-empty UUIDs for GPUs 0-3")
    checks_by_gpu = {
        str(gpu): validate_shard(
            shard_results[gpu],
            gpu_index=gpu,
            expected_gpu_uuid=expected_gpu_uuids[gpu],
            episode_index=episode_index,
            seed=seed,
        )
        for gpu in range(4)
    }
    all_records = [
        record for result in shard_results for record in result["episode_records"]
    ]
    task_ids = [int(record["task_id"]) for record in all_records]
    global_checks = {
        "four_shards": len(shard_results) == 4,
        "ten_tasks_exactly_once": sorted(task_ids) == list(range(10)),
        "ten_episode_records": len(all_records) == 10,
        "four_unique_front_gpu_uuids": len(
            {
                normalize_uuid(result["physical_gpu_uuid_visible"])
                for result in shard_results
            }
        )
        == 4,
        "all_shard_checks": all(
            all(checks.values()) for checks in checks_by_gpu.values()
        ),
    }
    status = "PASS" if all(global_checks.values()) else "FAIL"
    exit_counts: dict[str, int] = {}
    for record in all_records:
        for layer, count in record["exit_layer_counts"].items():
            exit_counts[str(layer)] = exit_counts.get(str(layer), 0) + int(count)
    return {
        "status": status,
        "scope": "phase_route_vla_rp_pep_release_smoke_front4",
        "policy": "rp_pep",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "physical_gpu_indices": [0, 1, 2, 3],
        "physical_gpu_uuids": {
            str(gpu): expected_gpu_uuids[gpu] for gpu in range(4)
        },
        "episode_index": episode_index,
        "seed": seed,
        "completed_episodes": len(all_records),
        "successes": int(sum(bool(record["success"]) for record in all_records)),
        "success_rate": float(
            sum(bool(record["success"]) for record in all_records) / len(all_records)
        ),
        "policy_calls": int(sum(int(record["policy_calls"]) for record in all_records)),
        "fm_calls_total": int(
            sum(int(record["fm_calls_total"]) for record in all_records)
        ),
        "exit_layer_counts": exit_counts,
        "task_results": [
            {
                "task_id": int(record["task_id"]),
                "success": bool(record["success"]),
                "policy_calls": int(record["policy_calls"]),
                "fm_calls_total": int(record["fm_calls_total"]),
                "initial_state_sha256": str(record["initial_state_sha256"]),
            }
            for record in sorted(all_records, key=lambda item: int(item["task_id"]))
        ],
        "checks_by_gpu": checks_by_gpu,
        "global_checks": global_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261329)
    parser.add_argument(
        "--expected-gpu-uuid",
        action="append",
        required=True,
        metavar="INDEX=UUID",
        help="repeat exactly once for each physical GPU index 0-3",
    )
    return parser.parse_args()


def parse_expected_gpu_uuids(values: Sequence[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        index_text, separator, uuid = value.partition("=")
        if not separator or not index_text.isdigit() or not normalize_uuid(uuid):
            raise ValueError(f"invalid --expected-gpu-uuid value: {value!r}")
        index = int(index_text)
        if index not in range(4) or index in result:
            raise ValueError(f"GPU index must be unique and in 0-3: {index}")
        result[index] = uuid.strip()
    if set(result) != set(range(4)):
        raise ValueError("expected exactly one UUID for each physical GPU index 0-3")
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sources = [args.input_root / f"gpu{gpu}" / "result.json" for gpu in range(4)]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing release-smoke shards: {missing}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    result = summarize(
        results,
        expected_gpu_uuids=parse_expected_gpu_uuids(args.expected_gpu_uuid),
        episode_index=args.episode_index,
        seed=args.seed,
    )
    result["input_results"] = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in sources
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
