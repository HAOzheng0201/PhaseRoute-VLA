#!/usr/bin/env python3
"""Freeze D9C completeness without aggregating success, safety, or efficiency."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    D9_ARMS,
    D9_RECORD_COUNT,
    D9_TASK_IDS,
    load_d9_selection_metadata,
)
from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    D9C_ARM_SCHEMA_VERSION,
    D9C_COLLECTION_SCHEMA_VERSION,
    D9C_COLLECTION_STATUS,
    D9C_INCOMPLETE_STATUS,
    D9C_OUTPUT_RELATIVE_PATH,
    D9C_TASK_SCHEMA_VERSION,
    D9C_TASK_STATUS,
    D9CCollectionError,
    PHASE_ROUTE_ARM,
    read_json_object,
    read_jsonl,
    sha256_file,
    validate_d9b_readiness,
    validate_pair_record,
    validate_runner_readiness,
)


OUTPUT = Path("results/v3/v3_d9c_collection_attestation.json")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _sidecar_digest(path: Path, sidecar: Path) -> str:
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if expected != observed:
        raise D9CCollectionError(f"SHA-256 sidecar differs: {path}")
    return observed


def _validate_arm_payload(
    arm_dir: Path,
    *,
    arm: str,
    canonical_key: str,
    task_id: int,
    episode_index: int,
    seed: int,
    source_commit: str,
) -> dict[str, Any]:
    result_path = arm_dir / "result.json"
    result_sha = _sidecar_digest(result_path, arm_dir / "result.sha256")
    result = read_json_object(result_path)
    if (
        result.get("status") != "COMPLETE_V3_D9C_ARM_ROLLOUT"
        or result.get("schema_version") != D9C_ARM_SCHEMA_VERSION
        or result.get("arm") != arm
        or result.get("canonical_key") != canonical_key
        or result.get("task_id") != task_id
        or result.get("episode_index") != episode_index
        or result.get("seed") != seed
        or result.get("source_git_commit") != source_commit
        or result.get("source_worktree_dirty") is not False
        or result.get("gpu", {}).get("physical_index") != task_id % 4
        or result.get("gpu", {}).get("visible_count") != 1
        or result.get("claim_boundary", {}).get("cross_pair_aggregate_computed")
        is not False
        or result.get("claim_boundary", {}).get("D9_primary_gate_evaluated")
        is not False
    ):
        raise D9CCollectionError(f"D9C arm result differs: {arm_dir}")
    telemetry = result["telemetry"]
    telemetry_path = arm_dir / str(telemetry["path"])
    if (
        sha256_file(telemetry_path) != telemetry["sha256"]
        or len(read_jsonl(telemetry_path)) != telemetry["records"]
        or telemetry["records"] != result["policy_accounting"]["policy_calls"]
    ):
        raise D9CCollectionError("D9C telemetry payload differs")
    cache_bytes = 0
    cache_shards = 0
    if arm == PHASE_ROUTE_ARM:
        runtime = result.get("phase_route_runtime")
        cache = result.get("same_noise_cache")
        if not isinstance(runtime, Mapping) or not isinstance(cache, Mapping):
            raise D9CCollectionError("PhaseRoute runtime/cache evidence is missing")
        runtime_path = arm_dir / str(runtime["path"])
        if (
            sha256_file(runtime_path) != runtime["sha256"]
            or len(read_jsonl(runtime_path)) != runtime["records"]
            or runtime["records"] != telemetry["records"]
        ):
            raise D9CCollectionError("PhaseRoute runtime payload differs")
        manifest_path = arm_dir / str(cache["manifest_path"])
        inventory_path = arm_dir / str(cache["inventory_path"])
        if (
            sha256_file(manifest_path) != cache["manifest_sha256"]
            or sha256_file(inventory_path) != cache["inventory_sha256"]
        ):
            raise D9CCollectionError("PhaseRoute cache manifest SHA-256 differs")
        manifest = read_jsonl(manifest_path)
        inventory = read_jsonl(inventory_path)
        if (
            len(manifest) != telemetry["records"]
            or len(inventory) != telemetry["records"]
        ):
            raise D9CCollectionError("PhaseRoute cache inventory count differs")
        cache_root = manifest_path.parent
        for item in inventory:
            payload = cache_root / str(item["relative_path"])
            if (
                not payload.is_file()
                or payload.stat().st_size != item["bytes"]
                or sha256_file(payload) != item["sha256"]
            ):
                raise D9CCollectionError(f"PhaseRoute cache shard differs: {payload}")
            cache_bytes += int(item["bytes"])
            cache_shards += 1
        if cache_bytes != cache["cache_bytes"]:
            raise D9CCollectionError("PhaseRoute cache byte accounting differs")
    elif result.get("phase_route_runtime") is not None or result.get(
        "same_noise_cache"
    ) is not None:
        raise D9CCollectionError("original A1 arm unexpectedly contains PhaseRoute cache")
    return {
        "result_sha256": result_sha,
        "telemetry_sha256": telemetry["sha256"],
        "policy_call_records": telemetry["records"],
        "cache_shards": cache_shards,
        "cache_bytes": cache_bytes,
        "initial_state_sha256": result["initial_state_sha256"],
        "gpu_uuid": result["gpu"]["uuid"],
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D9C collection freeze is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9C collection freeze requires a clean worktree")
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9C collection freeze refuses overwrite")
    source_commit = git_output("rev-parse", "HEAD")
    d9b = validate_d9b_readiness(REPO_ROOT)
    runner = validate_runner_readiness(REPO_ROOT)
    records = load_d9_selection_metadata(REPO_ROOT)
    raw_root = REPO_ROOT / D9C_OUTPUT_RELATIVE_PATH
    pair_sha256: dict[str, str] = {}
    task_sha256: dict[str, str] = {}
    payload_binding: dict[str, Any] = {}
    abort_ledger: list[dict[str, Any]] = []
    complete_rollouts = 0
    total_policy_records = 0
    total_cache_shards = 0
    total_cache_bytes = 0

    for task_id in D9_TASK_IDS:
        task_dir = raw_root / f"task{task_id}"
        task_path = task_dir / "task_result.json"
        task_sha = _sidecar_digest(task_path, task_dir / "task_result.sha256")
        task_result = read_json_object(task_path)
        if (
            task_result.get("status") != D9C_TASK_STATUS
            or task_result.get("schema_version") != D9C_TASK_SCHEMA_VERSION
            or task_result.get("task_id") != task_id
            or task_result.get("physical_gpu_index") != task_id % 4
            or task_result.get("source_git_commit") != source_commit
            or task_result.get("complete_pair_count") != 10
            or task_result.get("complete_rollout_count") != 20
            or any(task_result.get("interim_aggregate", {}).values())
            or task_result.get("claim_boundary", {}).get(
                "overall_or_per_task_gate_evaluated"
            )
            is not False
        ):
            raise D9CCollectionError(f"D9C task result differs: task{task_id}")
        task_sha256[str(task_id)] = task_sha
        for record in (item for item in records if item.task_id == task_id):
            pair_dir = task_dir / f"pair_episode{record.episode_index}"
            pair_path = pair_dir / "pair_record.json"
            pair_sha = _sidecar_digest(pair_path, pair_dir / "pair_record.sha256")
            pair = read_json_object(pair_path)
            validate_pair_record(pair, record=record)
            if task_result["pair_record_sha256"].get(record.canonical_key) != pair_sha:
                raise D9CCollectionError("task-to-pair SHA-256 binding differs")
            pair_sha256[record.canonical_key] = pair_sha
            arm_binding: dict[str, Any] = {}
            for arm in D9_ARMS:
                binding = _validate_arm_payload(
                    pair_dir / arm,
                    arm=arm,
                    canonical_key=record.canonical_key,
                    task_id=record.task_id,
                    episode_index=record.episode_index,
                    seed=record.seed,
                    source_commit=source_commit,
                )
                if pair["arms"][arm]["result_sha256"] != binding["result_sha256"]:
                    raise D9CCollectionError("pair-to-arm SHA-256 binding differs")
                arm_binding[arm] = binding
                complete_rollouts += 1
                total_policy_records += binding["policy_call_records"]
                total_cache_shards += binding["cache_shards"]
                total_cache_bytes += binding["cache_bytes"]
            if len({arm_binding[arm]["initial_state_sha256"] for arm in D9_ARMS}) != 1:
                raise D9CCollectionError("paired arms do not bind the same state")
            payload_binding[record.canonical_key] = arm_binding
            for abort_path in sorted(pair_dir.glob(".attempts/*/*/abort.json")):
                abort = read_json_object(abort_path)
                if (
                    abort.get("status") != "ABORT_V3_D9C_INFRASTRUCTURE_FAILURE"
                    or abort.get("canonical_key") != record.canonical_key
                    or abort.get("task_id") != record.task_id
                    or abort.get("episode_index") != record.episode_index
                    or abort.get("seed") != record.seed
                    or abort.get("source_git_commit") != source_commit
                    or abort.get("same_tuple_required_for_retry") is not True
                    or abort.get("outcome_based_retry") is not False
                ):
                    raise D9CCollectionError("D9C infrastructure retry ledger differs")
                abort_ledger.append(
                    {
                        "path": abort_path.relative_to(REPO_ROOT).as_posix(),
                        "sha256": sha256_file(abort_path),
                        "failure_type": abort.get("failure_type"),
                        "failure_message": abort.get("failure_message"),
                    }
                )

    checks = {
        "D9B_readiness_exact": bool(d9b),
        "D9C_frozen_runner_exact": bool(runner),
        "source_worktree_clean": not bool(git_output("status", "--porcelain=v1")),
        "all_10_tasks_complete": len(task_sha256) == 10,
        "all_100_pairs_complete": len(pair_sha256) == D9_RECORD_COUNT,
        "all_200_rollouts_complete": complete_rollouts == 2 * D9_RECORD_COUNT,
        "all_raw_policy_telemetry_bound": total_policy_records > 0,
        "all_PhaseRoute_calls_have_same_noise_cache": total_cache_shards > 0,
        "only_physical_GPU_0_to_3_used": True,
        "no_interim_success_safety_or_efficiency_aggregate": True,
        "D9_primary_gate_not_evaluated": True,
    }
    if not all(checks.values()):
        print(D9C_INCOMPLETE_STATUS, file=sys.stderr)
        raise D9CCollectionError(f"D9C completeness checks failed: {checks}")
    result = {
        "status": D9C_COLLECTION_STATUS,
        "schema_version": D9C_COLLECTION_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": source_commit,
        "source_worktree_dirty": False,
        "environment": {
            "python": platform.python_version(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "prerequisites": {"D9B": d9b, "D9C_runner": runner},
        "completeness": {
            "tasks": len(task_sha256),
            "pairs": len(pair_sha256),
            "rollouts": complete_rollouts,
            "raw_policy_call_records": total_policy_records,
            "PhaseRoute_same_noise_cache_shards": total_cache_shards,
            "PhaseRoute_same_noise_cache_bytes": total_cache_bytes,
            "infrastructure_abort_attempts": len(abort_ledger),
        },
        "task_result_sha256": task_sha256,
        "pair_record_sha256": pair_sha256,
        "arm_payload_binding": payload_binding,
        "infrastructure_abort_ledger": abort_ledger,
        "checks": checks,
        "access_ledger": {
            "official_episode_40_49_pairs_opened": D9_RECORD_COUNT,
            "active_rollouts": complete_rollouts,
            "original_A1_rollouts": D9_RECORD_COUNT,
            "PhaseRoute_rollouts": D9_RECORD_COUNT,
            "success_safety_efficiency_aggregate_calls": 0,
            "D9_primary_gate_calls": 0,
        },
        "authorization": {
            "next_stage": "D9D_SAME_NOISE_REPLAY_ONLY",
            "D9E_one_shot_aggregate_authorized": False,
            "additional_test_tuning_or_second_test": False,
        },
        "claim_boundary": {
            "D9C_is_complete_raw_collection": True,
            "D9C_is_D9_pass_or_negative": False,
            "success_rate_reported": False,
            "safety_rate_reported": False,
            "efficiency_rate_reported": False,
            "superiority_or_noninferiority_claim_authorized": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(D9C_COLLECTION_STATUS)


if __name__ == "__main__":
    main()
