#!/usr/bin/env python3
"""Aggregate D8C context/replay into prospective truth without router scoring."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    build_gripper_v2_features,
    build_gripper_v2_targets,
    stream_sha256,
    validate_runtime_context,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CLUSTER_COUNT,
    D8_CLUSTERS_PER_TASK,
    D8_TASK_IDS,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation_collection import (  # noqa: E402
    D8C_CANDIDATE_SCHEMA_VERSION,
    D8C_CONTEXT_SCHEMA_VERSION,
    D8C_DATASET_SCHEMA_VERSION,
    D8C_REPLAY_LAYERS,
    D8C_ROLE,
    D8C_SUITE,
    parse_fresh_cluster_key,
    validate_d8c_prerequisites,
)
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_ACTION_THRESHOLD,
    mean_action_cosine_distance,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d8c-dataset-result.v1"
SHARD_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-result", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8C JSON must be an object: {path}")
    return dict(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_context(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    result = _json_object(path)
    if (
        result.get("status") != "PASS_V3_D8C_CONTEXT"
        or result.get("source_git_commit") != git_output("rev-parse", "HEAD")
        or result.get("source_worktree_dirty") is not False
    ):
        raise PermissionError("D8C context result differs")
    payload_path = path.parent / str(result["payload"])
    records_path = path.parent / str(result["records"])
    if (
        stream_sha256(payload_path) != result["payload_sha256"]
        or stream_sha256(records_path) != result["records_sha256"]
    ):
        raise PermissionError("D8C context evidence SHA differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    records = _jsonl(records_path)
    if (
        payload.get("schema_version") != D8C_CONTEXT_SCHEMA_VERSION
        or payload.get("role") != D8C_ROLE
        or payload.get("suite") != D8C_SUITE
        or payload.get("task_replicate_identity_is_runtime_input") is not False
        or payload.get("layer27_is_runtime_input") is not False
        or len(records) != int(result["rows"])
    ):
        raise PermissionError("D8C context semantics differ")
    rows = int(payload["dataset_index"].numel())
    if not torch.equal(payload["dataset_index"], torch.arange(rows)):
        raise PermissionError("D8C context dataset index is not contiguous")
    validate_runtime_context(payload["runtime_inputs"], rows=rows)
    return result, dict(payload), records


def _load_candidates(
    root: Path, context: Mapping[str, Any], context_records: list[dict[str, Any]]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payloads = []
    candidate_records: dict[int, dict[str, Any]] = {}
    shard_results: dict[str, Any] = {}
    for shard in range(SHARD_COUNT):
        directory = root / f"shard{shard}"
        result_path = directory / "result.json"
        result = _json_object(result_path)
        if (
            result.get("status") != "PASS_V3_D8C_CANDIDATE_SHARD"
            or result.get("shard_index") != shard
            or result.get("source_git_commit") != git_output("rev-parse", "HEAD")
            or result.get("source_worktree_dirty") is not False
        ):
            raise PermissionError(f"D8C candidate shard {shard} result differs")
        payload_path = directory / str(result["payload"])
        records_path = directory / str(result["records"])
        if (
            stream_sha256(payload_path) != result["payload_sha256"]
            or stream_sha256(records_path) != result["records_sha256"]
        ):
            raise PermissionError(f"D8C candidate shard {shard} SHA differs")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != D8C_CANDIDATE_SCHEMA_VERSION
            or payload.get("shard_index") != shard
            or payload.get("shard_count") != SHARD_COUNT
            or payload.get("layer27_is_consistency_teacher_only") is not True
            or payload.get("router_scored") is not False
            or not torch.equal(
                payload["candidate_layers"],
                torch.tensor(D8C_REPLAY_LAYERS, dtype=torch.long),
            )
        ):
            raise PermissionError(f"D8C candidate shard {shard} semantics differ")
        for record in _jsonl(records_path):
            index = int(record["dataset_index"])
            if index in candidate_records:
                raise PermissionError("D8C candidate dataset index repeats")
            candidate_records[index] = record
        payloads.append(payload)
        shard_results[str(shard)] = {
            "path": str(result_path),
            "sha256": stream_sha256(result_path),
            "payload_sha256": str(result["payload_sha256"]),
            "rows": int(result["rows"]),
            "physical_gpu_index": int(result["physical_gpu_index"]),
        }
    tensor_names = (
        "dataset_index",
        "task_id",
        "replicate_id",
        "policy_seed",
        "call_ordinal",
        "shared_fm_input_x",
        "candidate_actions",
    )
    concatenated = {
        name: torch.cat([payload[name] for payload in payloads]).contiguous()
        for name in tensor_names
    }
    keys = sum((list(payload["cluster_keys"]) for payload in payloads), [])
    order = torch.argsort(concatenated["dataset_index"])
    merged = {name: value[order].contiguous() for name, value in concatenated.items()}
    merged_keys = [keys[index] for index in order.tolist()]
    rows = int(context["dataset_index"].numel())
    if set(candidate_records) != set(range(rows)):
        raise PermissionError("D8C candidate record coverage differs")
    for name in ("dataset_index", "task_id", "replicate_id", "policy_seed", "call_ordinal"):
        if not torch.equal(merged[name], context[name]):
            raise PermissionError(f"D8C context/candidate identity differs: {name}")
    if merged_keys != context["cluster_keys"]:
        raise PermissionError("D8C context/candidate cluster identity differs")
    if (
        merged["candidate_actions"].shape != (rows, 3, 8, 7)
        or merged["shared_fm_input_x"].shape != (rows, 8, 7)
        or not bool(torch.isfinite(merged["candidate_actions"]).all())
        or not bool(torch.isfinite(merged["shared_fm_input_x"]).all())
    ):
        raise PermissionError("D8C merged candidate geometry differs")
    context_by_index = {
        int(record["dataset_index"]): record for record in context_records
    }
    for index in range(rows):
        if (
            candidate_records[index]["source_payload_sha256"]
            != context_by_index[index]["source_payload_sha256"]
        ):
            raise PermissionError("D8C candidate/context source hash differs")
    merged["cluster_keys"] = merged_keys
    return merged, shard_results


def _load_l11_telemetry(
    raw_root: Path, context: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    by_key: dict[tuple[str, int], tuple[int, float]] = {}
    telemetry_hashes: dict[str, str] = {}
    behavior_exit_counts: Counter[int] = Counter()
    behavior_fm_calls = 0
    behavior_successes = 0
    for task in D8_TASK_IDS:
        result_path = raw_root / f"task{task}/result.json"
        result = _json_object(result_path)
        telemetry_path = result_path.parent / "policy_calls.jsonl"
        observed_hash = stream_sha256(telemetry_path)
        if (
            result.get("status") != "PASS_V3_D8C_RAW_TASK"
            or result.get("task_id") != task
            or result.get("source_git_commit") != git_output("rev-parse", "HEAD")
            or observed_hash != result.get("telemetry_sha256")
        ):
            raise PermissionError("D8C raw telemetry result differs")
        telemetry_hashes[str(task)] = observed_hash
        behavior_successes += int(result["behavior_successes"])
        counters: dict[str, int] = defaultdict(int)
        for record in _jsonl(telemetry_path):
            cluster_key = str(record["episode_id"])
            parsed_task, _replicate = parse_fresh_cluster_key(cluster_key)
            if parsed_task != task or int(record["task_id"]) != task:
                raise PermissionError("D8C telemetry identity differs")
            ordinal = counters[cluster_key]
            counters[cluster_key] += 1
            events = [
                event
                for event in record["extra"]["exit_events"]
                if event.get("layer_idx") == 11 and event.get("evaluated") is True
            ]
            if len(events) != 1:
                raise PermissionError("D8C telemetry lacks exactly one L11 event")
            delta = events[0].get("action_delta")
            if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                raise PermissionError("D8C L11 action delta is invalid")
            key = (cluster_key, ordinal)
            if key in by_key:
                raise PermissionError("D8C duplicate telemetry key")
            by_key[key] = (int(record["step_id"]), float(delta))
            behavior_exit_counts[int(record["exit_layer"])] += 1
            behavior_fm_calls += int(record["fm_calls"])
    rows = int(context["dataset_index"].numel())
    values = torch.empty(rows, dtype=torch.float64)
    for index in range(rows):
        key = (str(context["cluster_keys"][index]), int(context["call_ordinal"][index]))
        record = by_key.get(key)
        if record is None or record[0] != int(context["step_id"][index]):
            raise PermissionError("D8C telemetry/context identity differs")
        values[index] = record[1]
    if len(by_key) != rows or not bool(torch.isfinite(values).all()):
        raise PermissionError("D8C telemetry coverage differs")
    return values.contiguous(), {
        "telemetry_sha256": telemetry_hashes,
        "behavior_exit_counts": {
            str(layer): behavior_exit_counts[layer]
            for layer in sorted(behavior_exit_counts)
        },
        "behavior_fm_calls": behavior_fm_calls,
        "behavior_successes": behavior_successes,
        "behavior_clusters": D8_CLUSTER_COUNT,
        "behavior_success_rate": behavior_successes / D8_CLUSTER_COUNT,
    }


def _run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D8C dataset aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8C dataset aggregation requires a clean worktree")
    prerequisites = validate_d8c_prerequisites(REPO_ROOT)
    expected_context = (
        REPO_ROOT / "reports" / "v3_d8_fresh_context" / "result.json"
    ).resolve()
    context_path = args.context_result.resolve(strict=True)
    if context_path != expected_context:
        raise PermissionError("D8C context result path differs")
    expected_candidates = (REPO_ROOT / "reports" / "v3_d8_fresh_candidates").resolve()
    candidate_root = args.candidate_root.resolve(strict=True)
    if candidate_root != expected_candidates:
        raise PermissionError("D8C candidate root path differs")
    expected_raw = (REPO_ROOT / "reports" / "v3_d8_fresh_raw").resolve()
    raw_root = args.raw_root.resolve(strict=True)
    if raw_root != expected_raw:
        raise PermissionError("D8C raw root path differs")
    expected_output = (REPO_ROOT / "reports" / "v3_d8_fresh_dataset").resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("D8C dataset output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D8C refuses to overwrite dataset evidence")

    started = time.perf_counter()
    context_result, context, context_records = _load_context(context_path)
    candidates, shard_results = _load_candidates(
        candidate_root, context, context_records
    )
    l11_delta, behavior = _load_l11_telemetry(raw_root, context)
    actions = candidates["candidate_actions"].float().contiguous()
    rows = int(actions.shape[0])
    features = build_gripper_v2_features(context["runtime_inputs"], actions[:, :2])
    targets = build_gripper_v2_targets(actions)
    l13_delta = mean_action_cosine_distance(actions[:, 0], actions[:, 1])
    action_consistency = torch.stack(
        (l11_delta <= D5_ACTION_THRESHOLD, l13_delta <= D5_ACTION_THRESHOLD), dim=1
    )
    full_distance = torch.stack(
        (
            mean_action_cosine_distance(actions[:, 0], actions[:, 2]),
            mean_action_cosine_distance(actions[:, 1], actions[:, 2]),
        ),
        dim=1,
    )
    full_unsafe = full_distance > D5_ACTION_THRESHOLD
    gripper_unsafe = targets.occurrence[:, :, 0]
    expected_gripper = (
        (actions[:, :2, :, 6] >= 0.0) != (actions[:, 2, :, 6] >= 0.0)[:, None]
    ).any(dim=2)
    if not torch.equal(gripper_unsafe, expected_gripper):
        raise PermissionError("D8C gripper target recomputation differs")

    flat_features = features.reshape(rows * 2, 97).contiguous()
    flat_layer = torch.tensor([11, 13], dtype=torch.long).repeat(rows)
    flat_source = torch.arange(rows, dtype=torch.long).repeat_interleave(2)
    flat_task = context["task_id"].repeat_interleave(2)
    flat_replicate = context["replicate_id"].repeat_interleave(2)
    flat_policy_seed = context["policy_seed"].repeat_interleave(2)
    flat_cluster_keys = [
        key for key in context["cluster_keys"] for _candidate in range(2)
    ]
    unsafe_target = torch.stack(
        (full_unsafe.reshape(-1), gripper_unsafe.reshape(-1)), dim=1
    ).contiguous()
    cluster_counts = Counter(context["cluster_keys"])
    task_cluster_sets = {
        task: {
            key
            for key in context["cluster_keys"]
            if parse_fresh_cluster_key(key)[0] == task
        }
        for task in D8_TASK_IDS
    }
    if (
        len(cluster_counts) != D8_CLUSTER_COUNT
        or any(len(values) != D8_CLUSTERS_PER_TASK for values in task_cluster_sets.values())
        or flat_features.shape != (rows * 2, 97)
        or not bool(torch.isfinite(flat_features).all())
    ):
        raise PermissionError("D8C cluster or feature geometry differs")

    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    payload = {
        "schema_version": D8C_DATASET_SCHEMA_VERSION,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "D8_readiness_sha256": prerequisites["D8_readiness_sha256"],
        "feature_dimension": 97,
        "feature_layout": {
            "legacy_causal_context": [0, 82],
            "current_candidate_gripper_sign_sequence": [82, 90],
            "current_candidate_gripper_transition_pattern": [90, 97],
        },
        "features": flat_features,
        "candidate_layer": flat_layer,
        "source_row": flat_source,
        "task_id": flat_task,
        "replicate_id": flat_replicate,
        "policy_seed": flat_policy_seed,
        "cluster_keys": flat_cluster_keys,
        "call_ordinal": context["call_ordinal"].repeat_interleave(2),
        "step_id": context["step_id"].repeat_interleave(2),
        "behavior_exit_layer": context["behavior_exit_layer"].repeat_interleave(2),
        "action_consistency": action_consistency.reshape(-1).contiguous(),
        "unsafe_target": unsafe_target,
        "target_axis_order": ["full_action_unsafe", "gripper_step_unsafe"],
        "full_action_distance": full_distance.reshape(-1).contiguous(),
        "l11_telemetry_action_delta": l11_delta,
        "l13_same_noise_action_delta": l13_delta,
        "candidate_actions": actions,
        "shared_fm_input_x": candidates["shared_fm_input_x"],
        "layer27_runtime_visible": False,
        "layer27_is_consistency_teacher_only": True,
        "task_replicate_identity_is_runtime_input": False,
        "router_scored": False,
        "confirmation_gate_inspected": False,
        "official_episode_40_49_opened": False,
    }
    payload_path = incomplete / "fresh_confirmation_dataset.pt"
    torch.save(payload, payload_path)
    per_layer = {}
    for layer_index, layer in enumerate((11, 13)):
        per_layer[str(layer)] = {
            "rows": rows,
            "action_consistency_safe": int(action_consistency[:, layer_index].sum()),
            "full_action_unsafe": int(full_unsafe[:, layer_index].sum()),
            "gripper_step_unsafe": int(gripper_unsafe[:, layer_index].sum()),
            "joint_unsafe": int(
                (full_unsafe[:, layer_index] | gripper_unsafe[:, layer_index]).sum()
            ),
        }
    checks = {
        "context_and_four_candidate_shards_pass_current_commit": len(shard_results) == 4,
        "all_200_clusters_and_all_policy_calls_accounted_for": len(cluster_counts)
        == D8_CLUSTER_COUNT,
        "same_noise_candidate_context_identity_exact": True,
        "97d_features_use_current_candidate_only": features.shape == (rows, 2, 97),
        "full_action_and_gripper_truth_finite_and_exact": bool(
            torch.isfinite(full_distance).all()
        ),
        "layer27_offline_consistency_teacher_only": True,
        "official_episode_40_49_not_opened": all(
            ":episode" not in key for key in context["cluster_keys"]
        ),
        "router_not_loaded_scored_refit_or_thresholded": True,
        "D7_not_applied_to_environment": True,
    }
    result = {
        "status": "PASS_V3_D8C_DATASET" if all(checks.values()) else "FAIL_V3_D8C_DATASET",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "policy_calls": rows,
        "candidate_rows": rows * 2,
        "clusters": len(cluster_counts),
        "clusters_per_task": [len(task_cluster_sets[task]) for task in D8_TASK_IDS],
        "feature_dimension": 97,
        "target_support": per_layer,
        "behavior_a1": behavior,
        "context_result": {
            "path": str(context_path),
            "sha256": stream_sha256(context_path),
            "payload_sha256": str(context_result["payload_sha256"]),
        },
        "candidate_shard_results": shard_results,
        "payload": payload_path.name,
        "payload_sha256": stream_sha256(payload_path),
        "checks": checks,
        "access_ledger": {
            "fresh_context_payload_opened": True,
            "fresh_candidate_payloads_opened": 4,
            "candidate_truth_computed": True,
            "final_router_loaded": False,
            "confirmation_gate_inspected": False,
            "official_episode_40_49_opened": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "claim_boundary": {
            "D8C_collection_complete": True,
            "D8_confirmation_gate_evaluated": False,
            "generated_states_are_official_benchmark_states": False,
            "behavior_success_is_descriptive_only": True,
            "closed_loop_D7_success": False,
            "superiority_claim_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        raise RuntimeError("D8C dataset failed one or more gates")
    shutil.move(str(incomplete), str(output))
    print("PASS_V3_D8C_DATASET", flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        incomplete = args.output_dir.resolve().with_name(args.output_dir.name + ".incomplete")
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_V3_D8C_DATASET",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
