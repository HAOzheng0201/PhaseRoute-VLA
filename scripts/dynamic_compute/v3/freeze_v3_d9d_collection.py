#!/usr/bin/env python3
"""Freeze complete D9D call truth without aggregating the D9 gate."""

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

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    read_json_object,
    read_jsonl,
    sha256_array,
    sha256_file,
)
from a1.vla.dynamic_compute.v3.same_noise_replay import (  # noqa: E402
    D9C_COLLECTION_SHA256,
    D9D_ACTION_THRESHOLD,
    D9D_COLLECTION_SCHEMA_VERSION,
    D9D_COLLECTION_STATUS,
    D9D_EXPECTED_ROWS,
    D9D_OUTPUT_RELATIVE_PATH,
    D9D_REPLAY_LAYERS,
    D9D_SEVERE_RATIO,
    D9D_SHARD_COUNT,
    D9D_SHARD_RESULT_SCHEMA_VERSION,
    D9D_SHARD_SCHEMA_VERSION,
    D9D_SHARD_STATUS,
    hash_online_action,
    validate_d9c_collection,
    validate_d9d_runner_readiness,
)


OUTPUT = Path("results/v3/v3_d9d_collection_attestation.json")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _validate_tensor(
    payload: Mapping[str, Any], name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    value = payload.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or tuple(value.shape) != shape
        or value.dtype != dtype
    ):
        raise PermissionError(f"D9D payload tensor differs: {name}")
    return value.contiguous()


def _validate_shard(shard: int, *, expected_commit: str) -> dict[str, Any]:
    shard_dir = REPO_ROOT / D9D_OUTPUT_RELATIVE_PATH / f"shard{shard}"
    result_path = shard_dir / "result.json"
    result = read_json_object(result_path)
    if (
        result.get("status") != D9D_SHARD_STATUS
        or result.get("schema_version") != D9D_SHARD_RESULT_SCHEMA_VERSION
        or result.get("source_git_commit") != expected_commit
        or result.get("source_worktree_dirty") is not False
        or result.get("shard_index") != shard
        or result.get("shard_count") != D9D_SHARD_COUNT
        or result.get("rows") != D9D_EXPECTED_ROWS // D9D_SHARD_COUNT
        or result.get("candidate_layers") != list(D9D_REPLAY_LAYERS)
        or result.get("physical_gpu_index") != shard
        or not all(result.get("checks", {}).values())
        or result.get("claim_boundary", {}).get(
            "success_safety_efficiency_aggregate_computed"
        )
        is not False
        or result.get("claim_boundary", {}).get("D9_primary_gate_evaluated")
        is not False
    ):
        raise PermissionError(f"D9D shard result semantics differ: shard{shard}")
    records_path = shard_dir / str(result["records"])
    payload_path = shard_dir / str(result["payload"])
    if (
        sha256_file(records_path) != result["records_sha256"]
        or sha256_file(payload_path) != result["payload_sha256"]
    ):
        raise PermissionError("D9D shard evidence SHA-256 differs")
    records = read_jsonl(records_path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    rows = int(result["rows"])
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != D9D_SHARD_SCHEMA_VERSION
        or payload.get("role") != "D9D_same_noise_truth_only"
        or payload.get("suite") != "libero_10"
        or payload.get("D9C_collection_sha256") != D9C_COLLECTION_SHA256
        or payload.get("shard_index") != shard
        or payload.get("shard_count") != D9D_SHARD_COUNT
        or payload.get("full_action_threshold") != D9D_ACTION_THRESHOLD
        or payload.get("severe_ratio") != D9D_SEVERE_RATIO
        or payload.get("layer27_is_consistency_teacher_only") is not True
        or payload.get("router_scored") is not False
        or payload.get("active_control") is not False
        or payload.get("D9_gate_evaluated") is not False
        or len(records) != rows
    ):
        raise PermissionError("D9D shard payload semantics differ")
    layers = _validate_tensor(payload, "candidate_layers", (3,), torch.int64)
    if not torch.equal(layers, torch.tensor(D9D_REPLAY_LAYERS)):
        raise PermissionError("D9D candidate layer order differs")
    dataset_index = _validate_tensor(payload, "dataset_index", (rows,), torch.int64)
    task_id = _validate_tensor(payload, "task_id", (rows,), torch.int64)
    episode_index = _validate_tensor(payload, "episode_index", (rows,), torch.int64)
    seed = _validate_tensor(payload, "seed", (rows,), torch.int64)
    call_ordinal = _validate_tensor(payload, "call_ordinal", (rows,), torch.int64)
    step_id = _validate_tensor(payload, "step_id", (rows,), torch.int64)
    selected_layer = _validate_tensor(payload, "selected_layer", (rows,), torch.int64)
    candidates = _validate_tensor(
        payload, "candidate_actions", (rows, 3, 8, 7), torch.float32
    )
    shared = _validate_tensor(payload, "shared_fm_input_x", (rows, 8, 7), torch.float32)
    online = _validate_tensor(payload, "online_selected_action", (rows, 8, 7), torch.float32)
    distance = _validate_tensor(payload, "full_action_distance", (rows,), torch.float64)
    full_unsafe = _validate_tensor(payload, "full_action_unsafe", (rows,), torch.bool)
    gripper_unsafe = _validate_tensor(payload, "gripper_unsafe", (rows,), torch.bool)
    severe = _validate_tensor(payload, "severe_full_action", (rows,), torch.bool)
    replay_error = _validate_tensor(
        payload, "selected_replay_max_abs_error", (rows,), torch.float64
    )
    bit_exact = _validate_tensor(payload, "selected_replay_bit_exact", (rows,), torch.bool)
    strings = {
        name: payload.get(name)
        for name in (
            "canonical_keys",
            "source_npz_sha256",
            "shared_fm_input_sha256",
            "candidate_actions_sha256",
            "online_selected_action_sha256",
        )
    }
    if any(not isinstance(value, list) or len(value) != rows for value in strings.values()):
        raise PermissionError("D9D shard string-array geometry differs")
    if (
        not bool(torch.isfinite(candidates).all())
        or not bool(torch.isfinite(shared).all())
        or not bool(torch.isfinite(online).all())
        or not bool(torch.isfinite(distance).all())
        or not bool(torch.isfinite(replay_error).all())
        or not bool((dataset_index.remainder(D9D_SHARD_COUNT) == shard).all())
        or not set(selected_layer.tolist()).issubset(set(D9D_REPLAY_LAYERS))
    ):
        raise PermissionError("D9D shard numeric integrity differs")
    selected_index = torch.empty(rows, dtype=torch.long)
    for index, layer in enumerate(D9D_REPLAY_LAYERS):
        selected_index[selected_layer == layer] = index
    replayed_selected = candidates[torch.arange(rows), selected_index]
    expected_error = (replayed_selected - online).abs().amax(dim=(1, 2)).double()
    similarity = torch.nn.functional.cosine_similarity(
        online.double(), candidates[:, 2].double(), dim=-1, eps=1.0e-8
    )
    expected_distance = (1.0 - similarity).mean(dim=1).clamp_min(0.0)
    expected_full = expected_distance > D9D_ACTION_THRESHOLD
    expected_gripper = (
        (online[:, :, 6] >= 0.0) != (candidates[:, 2, :, 6] >= 0.0)
    ).any(dim=1)
    expected_severe = expected_distance > D9D_SEVERE_RATIO * D9D_ACTION_THRESHOLD
    if (
        not torch.equal(expected_error, replay_error)
        or not torch.allclose(expected_distance, distance, rtol=0.0, atol=1.0e-12)
        or not torch.equal(expected_full, full_unsafe)
        or not torch.equal(expected_gripper, gripper_unsafe)
        or not torch.equal(expected_severe, severe)
        or not torch.equal(
            (replayed_selected == online).all(dim=(1, 2)), bit_exact
        )
    ):
        raise PermissionError("D9D per-call truth recomputation differs")
    for ordinal, record in enumerate(records):
        if (
            record.get("dataset_index") != int(dataset_index[ordinal])
            or record.get("task_id") != int(task_id[ordinal])
            or record.get("episode_index") != int(episode_index[ordinal])
            or record.get("seed") != int(seed[ordinal])
            or record.get("canonical_key") != strings["canonical_keys"][ordinal]
            or record.get("call_ordinal") != int(call_ordinal[ordinal])
            or record.get("step_id") != int(step_id[ordinal])
            or record.get("online_selected_layer") != int(selected_layer[ordinal])
            or record.get("source_npz_sha256")
            != strings["source_npz_sha256"][ordinal]
            or record.get("shared_fm_input_sha256")
            != strings["shared_fm_input_sha256"][ordinal]
            or record.get("candidate_actions_sha256")
            != strings["candidate_actions_sha256"][ordinal]
            or record.get("online_selected_action_sha256")
            != strings["online_selected_action_sha256"][ordinal]
            or record.get("selected_replay_max_abs_error")
            != float(replay_error[ordinal])
            or record.get("selected_replay_bit_exact") != bool(bit_exact[ordinal])
            or record.get("selected_replay_role")
            != "float16_cache_quantization_diagnostic_only"
            or record.get("full_action_distance_selected_vs_L27")
            != float(distance[ordinal])
            or record.get("full_action_unsafe") != bool(full_unsafe[ordinal])
            or record.get("gripper_XOR_selected_vs_L27")
            != bool(gripper_unsafe[ordinal])
            or record.get("severe_full_action") != bool(severe[ordinal])
            or record.get("router_scored") is not False
            or record.get("LIBERO_environment_created") is not False
            or record.get("environment_action_executed") is not False
            or record.get("online_selected_action_modified") is not False
        ):
            raise PermissionError(f"D9D record/payload binding differs at row {ordinal}")
        if (
            sha256_array(candidates[ordinal].numpy())
            != strings["candidate_actions_sha256"][ordinal]
            or sha256_array(shared[ordinal].numpy())
            != strings["shared_fm_input_sha256"][ordinal]
            or hash_online_action(online[ordinal])
            != strings["online_selected_action_sha256"][ordinal]
        ):
            raise PermissionError(f"D9D row hash differs at row {ordinal}")
    return {
        "result_path": result_path.relative_to(REPO_ROOT).as_posix(),
        "result_sha256": sha256_file(result_path),
        "records_path": records_path.relative_to(REPO_ROOT).as_posix(),
        "records_sha256": result["records_sha256"],
        "payload_path": payload_path.relative_to(REPO_ROOT).as_posix(),
        "payload_sha256": result["payload_sha256"],
        "rows": rows,
        "dataset_index": dataset_index,
        "physical_gpu_index": result["physical_gpu_index"],
        "gpu_uuid": result["gpu_uuid"],
        "source_NPZ_hashes_verified_during_replay": result["checks"][
            "all_source_NPZ_inventory_hashes_exact"
        ],
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D9D collection freeze is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9D collection freeze requires a clean worktree")
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9D collection freeze refuses overwrite")
    collection = validate_d9c_collection(REPO_ROOT)
    readiness = validate_d9d_runner_readiness(REPO_ROOT)
    source_commit = git_output("rev-parse", "HEAD")
    shards = {
        str(shard): _validate_shard(shard, expected_commit=source_commit)
        for shard in range(D9D_SHARD_COUNT)
    }
    indices = torch.cat([shards[str(shard)].pop("dataset_index") for shard in range(4)])
    checks = {
        "D9C_collection_binding_exact": collection["sha256"] == D9C_COLLECTION_SHA256,
        "D9D_runner_readiness_exact": bool(readiness),
        "all_four_front_GPU_shards_complete": len(shards) == D9D_SHARD_COUNT,
        "all_3700_policy_calls_have_truth": int(indices.numel()) == D9D_EXPECTED_ROWS,
        "global_dataset_index_coverage_exact": torch.equal(
            torch.sort(indices).values, torch.arange(D9D_EXPECTED_ROWS)
        ),
        "only_physical_GPU_0_to_3_used": {
            item["physical_gpu_index"] for item in shards.values()
        }
        == set(range(4)),
        "source_NPZ_SHA_verified_before_every_open": all(
            item["source_NPZ_hashes_verified_during_replay"] for item in shards.values()
        ),
        "per_call_truth_recomputed_and_hash_bound": True,
        "no_LIBERO_environment_router_action_mutation_or_active_control": True,
        "no_success_safety_efficiency_or_D9_gate_aggregate": True,
        "source_worktree_clean": not bool(git_output("status", "--porcelain=v1")),
    }
    if not all(checks.values()):
        raise PermissionError(f"D9D collection checks failed: {checks}")
    result = {
        "status": D9D_COLLECTION_STATUS,
        "schema_version": D9D_COLLECTION_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": source_commit,
        "source_worktree_dirty": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_initialized": torch.cuda.is_initialized(),
        },
        "D9C_collection": {
            key: value for key, value in collection.items() if key != "arm_payload_binding"
        },
        "D9D_runner_readiness": readiness,
        "completeness": {
            "shards": D9D_SHARD_COUNT,
            "policy_call_truth_rows": D9D_EXPECTED_ROWS,
            "candidate_replays": D9D_EXPECTED_ROWS * len(D9D_REPLAY_LAYERS),
            "candidate_layers": list(D9D_REPLAY_LAYERS),
        },
        "shard_binding": shards,
        "checks": checks,
        "access_ledger": {
            "D9C_PhaseRoute_policy_states_replayed": D9D_EXPECTED_ROWS,
            "source_cache_NPZ_SHA_checks": D9D_EXPECTED_ROWS,
            "LIBERO_environments_created": 0,
            "environment_actions_executed": 0,
            "routers_loaded": 0,
            "success_safety_efficiency_aggregate_calls": 0,
            "D9_primary_gate_calls": 0,
        },
        "authorization": {
            "next_stage": "D9E_ONE_SHOT_SUCCESS_EFFICIENCY_SAFETY_AGGREGATE",
            "D9E_one_shot_aggregate_authorized": True,
            "additional_test_tuning_or_second_independent_test": False,
        },
        "claim_boundary": {
            "D9D_is_complete_per_call_truth": True,
            "D9D_is_D9_pass_or_negative": False,
            "layer27_is_expert_or_success_certificate": False,
            "success_rate_reported": False,
            "safety_rate_reported": False,
            "efficiency_rate_reported": False,
            "D9_primary_gate_evaluated": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(D9D_COLLECTION_STATUS)


if __name__ == "__main__":
    main()
