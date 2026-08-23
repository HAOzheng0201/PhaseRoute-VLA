"""Fail-closed contracts for the V3-D9D same-noise replay.

The helpers in this module never construct a LIBERO environment, execute an
action, load the PhaseRoute router, or aggregate the final D9 gate.  They bind
the complete D9C PhaseRoute cache and construct per-policy-call truth only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .development_collection import D2_CHECKPOINT_SHA256
from .independent_test_protocol import (
    D9_RECORD_COUNT,
    D9_TASK_IDS,
    D9TestRecord,
    load_d9_contract,
    load_d9_selection_metadata,
)
from .paired_active_collection import (
    D9C_ARM_SCHEMA_VERSION,
    D9C_COLLECTION_SCHEMA_VERSION,
    D9C_COLLECTION_STATUS,
    D9C_OUTPUT_RELATIVE_PATH,
    PHASE_ROUTE_ARM,
    PHASE_ROUTE_TEACHER_KIND,
    read_json_object,
    read_jsonl,
    sha256_array,
    sha256_file,
)
from ..vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)


D9C_COLLECTION_RELATIVE_PATH = Path(
    "results/v3/v3_d9c_collection_attestation.json"
)
D9C_COLLECTION_SHA256 = (
    "e4994368622590ec0cce0beb02b870f9a28e4c2f04fd9f1f93f424cb98d9292d"
)
D9C_SOURCE_GIT_COMMIT = "1a0598d67994755b9f8abd88563ea2d03b7ff47c"
D9D_RUNNER_READINESS_RELATIVE_PATH = Path(
    "results/v3/v3_d9d_runner_readiness.json"
)
D9D_RUNNER_READINESS_STATUS = "PASS_V3_D9D_FROZEN_RUNNER_READINESS"
D9D_OUTPUT_RELATIVE_PATH = Path("reports/v3_d9d_same_noise_replay")
D9D_LOG_RELATIVE_PATH = Path("reports/v3_d9d_same_noise_logs")
D9D_REPLAY_LAYERS = (11, 13, 27)
D9D_SHARD_COUNT = 4
D9D_EXPECTED_ROWS = 3700
D9D_ACTION_THRESHOLD = 0.00390625
D9D_SEVERE_RATIO = 4.0
D9D_SELECTED_REPLAY_ATOL = 1.0e-6
D9D_SHARD_SCHEMA_VERSION = "phase-route-vla.v3.d9d-truth-shard.v1"
D9D_SHARD_RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d9d-shard-result.v1"
D9D_COLLECTION_SCHEMA_VERSION = "phase-route-vla.v3.d9d-truth-collection.v1"
D9D_SHARD_STATUS = "PASS_V3_D9D_SAME_NOISE_TRUTH_SHARD"
D9D_COLLECTION_STATUS = "COMPLETE_V3_D9D_SAME_NOISE_TRUTH"


class D9DReplayError(ValueError):
    """Raised whenever source lineage or replay truth violates D9D."""


@dataclass(frozen=True)
class D9DCall:
    dataset_index: int
    task_id: int
    episode_index: int
    seed: int
    canonical_key: str
    call_ordinal: int
    step_id: int
    selected_layer: int
    array_path: Path
    array_relative_path: str
    array_bytes: int
    array_sha256: str
    arm_result_sha256: str
    manifest_sha256: str
    inventory_sha256: str


@dataclass(frozen=True)
class D9DCallTruth:
    selected_candidate_index: int
    selected_replay_max_abs_error: float
    selected_replay_within_atol: bool
    selected_replay_bit_exact: bool
    full_action_distance: float
    full_action_unsafe: bool
    gripper_unsafe: bool
    severe_full_action: bool


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    try:
        expected = sidecar.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError) as error:
        raise D9DReplayError(f"D9D SHA-256 sidecar is missing: {path}") from error
    observed = sha256_file(path)
    if expected != observed:
        raise D9DReplayError(f"D9D SHA-256 sidecar differs: {path}")
    return observed


def validate_d9c_collection(repo_root: str | Path) -> dict[str, Any]:
    """Bind the immutable D9C collection without reading result aggregates."""

    root = Path(repo_root).resolve(strict=True)
    load_d9_contract(root)
    path = root / D9C_COLLECTION_RELATIVE_PATH
    observed_sha = _sidecar_digest(path)
    result = read_json_object(path)
    completeness = result.get("completeness", {})
    authorization = result.get("authorization", {})
    boundary = result.get("claim_boundary", {})
    bindings = result.get("arm_payload_binding")
    if (
        observed_sha != D9C_COLLECTION_SHA256
        or result.get("status") != D9C_COLLECTION_STATUS
        or result.get("schema_version") != D9C_COLLECTION_SCHEMA_VERSION
        or result.get("source_git_commit") != D9C_SOURCE_GIT_COMMIT
        or result.get("source_worktree_dirty") is not False
        or completeness.get("tasks") != len(D9_TASK_IDS)
        or completeness.get("pairs") != D9_RECORD_COUNT
        or completeness.get("rollouts") != 2 * D9_RECORD_COUNT
        or completeness.get("PhaseRoute_same_noise_cache_shards")
        != D9D_EXPECTED_ROWS
        or authorization.get("next_stage") != "D9D_SAME_NOISE_REPLAY_ONLY"
        or authorization.get("D9E_one_shot_aggregate_authorized") is not False
        or boundary.get("D9C_is_complete_raw_collection") is not True
        or boundary.get("D9C_is_D9_pass_or_negative") is not False
        or not isinstance(bindings, Mapping)
        or len(bindings) != D9_RECORD_COUNT
    ):
        raise D9DReplayError("D9C frozen collection semantics differ")
    return {
        "path": D9C_COLLECTION_RELATIVE_PATH.as_posix(),
        "sha256": observed_sha,
        "source_git_commit": D9C_SOURCE_GIT_COMMIT,
        "cache_rows": D9D_EXPECTED_ROWS,
        "arm_payload_binding": bindings,
    }


def validate_d9d_runner_readiness(repo_root: str | Path) -> dict[str, Any]:
    """Verify D9D code digests frozen before the first replay shard."""

    root = Path(repo_root).resolve(strict=True)
    path = root / D9D_RUNNER_READINESS_RELATIVE_PATH
    readiness = read_json_object(path)
    if (
        readiness.get("status") != D9D_RUNNER_READINESS_STATUS
        or not all(readiness.get("checks", {}).values())
        or readiness.get("access_ledger", {}).get("cache_NPZ_payloads_opened") != 0
        or readiness.get("access_ledger", {}).get("model_loaded") is not False
        or readiness.get("access_ledger", {}).get("CUDA_initialized") is not False
        or readiness.get("authorization", {}).get("next_stage")
        != "D9D_EXACT_FRONT4_SAME_NOISE_REPLAY"
    ):
        raise D9DReplayError("D9D runner readiness semantics differ")
    bound_code = readiness.get("bound_code_sha256")
    if not isinstance(bound_code, Mapping) or not bound_code:
        raise D9DReplayError("D9D runner bound-code inventory is missing")
    for relative, expected in bound_code.items():
        if sha256_file(root / str(relative)) != expected:
            raise D9DReplayError(f"D9D runner code changed: {relative}")
    return {
        "path": D9D_RUNNER_READINESS_RELATIVE_PATH.as_posix(),
        "sha256": _sidecar_digest(path),
        "source_git_commit": readiness["source_git_commit"],
        "bound_code_files": len(bound_code),
    }


def _safe_child(root: Path, relative: str, *, context: str) -> Path:
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts or child.as_posix() != relative:
        raise D9DReplayError(f"{context} path is unsafe")
    resolved = (root / child).resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise D9DReplayError(f"{context} is not a regular child file")
    return resolved


def _validate_arm_calls(
    root: Path,
    *,
    record: D9TestRecord,
    arm_binding: Mapping[str, Any],
    dataset_index_start: int,
) -> tuple[D9DCall, ...]:
    arm_dir = (
        root
        / D9C_OUTPUT_RELATIVE_PATH
        / f"task{record.task_id}"
        / f"pair_episode{record.episode_index}"
        / PHASE_ROUTE_ARM
    ).resolve(strict=True)
    result_path = arm_dir / "result.json"
    result_sha = _sidecar_digest(result_path)
    result = read_json_object(result_path)
    cache = result.get("same_noise_cache", {})
    runtime_info = result.get("phase_route_runtime", {})
    telemetry_info = result.get("telemetry", {})
    if (
        result_sha != arm_binding.get("result_sha256")
        or result.get("status") != "COMPLETE_V3_D9C_ARM_ROLLOUT"
        or result.get("schema_version") != D9C_ARM_SCHEMA_VERSION
        or result.get("source_git_commit") != D9C_SOURCE_GIT_COMMIT
        or result.get("source_worktree_dirty") is not False
        or result.get("arm") != PHASE_ROUTE_ARM
        or result.get("canonical_key") != record.canonical_key
        or result.get("task_id") != record.task_id
        or result.get("episode_index") != record.episode_index
        or result.get("seed") != record.seed
        or result.get("claim_boundary", {}).get("D9_primary_gate_evaluated")
        is not False
        or cache.get("all_policy_calls_cached") is not True
    ):
        raise D9DReplayError(f"D9C PhaseRoute arm binding differs: {record.canonical_key}")
    manifest_path = _safe_child(
        arm_dir, str(cache.get("manifest_path")), context="D9D manifest"
    )
    inventory_path = _safe_child(
        arm_dir, str(cache.get("inventory_path")), context="D9D inventory"
    )
    runtime_path = _safe_child(
        arm_dir, str(runtime_info.get("path")), context="D9D runtime telemetry"
    )
    telemetry_path = _safe_child(
        arm_dir, str(telemetry_info.get("path")), context="D9D policy telemetry"
    )
    manifest_sha = sha256_file(manifest_path)
    inventory_sha = sha256_file(inventory_path)
    if (
        manifest_sha != cache.get("manifest_sha256")
        or inventory_sha != cache.get("inventory_sha256")
        or sha256_file(runtime_path) != runtime_info.get("sha256")
        or sha256_file(telemetry_path) != telemetry_info.get("sha256")
    ):
        raise D9DReplayError("D9D source metadata SHA-256 differs")
    manifests = read_jsonl(manifest_path)
    inventory = read_jsonl(inventory_path)
    runtime = read_jsonl(runtime_path)
    telemetry = read_jsonl(telemetry_path)
    expected_calls = int(result.get("policy_accounting", {}).get("policy_calls", -1))
    if (
        expected_calls <= 0
        or len(manifests) != expected_calls
        or len(inventory) != expected_calls
        or len(runtime) != expected_calls
        or len(telemetry) != expected_calls
        or cache.get("cache_records") != expected_calls
        or runtime_info.get("records") != expected_calls
        or telemetry_info.get("records") != expected_calls
    ):
        raise D9DReplayError("D9D per-arm call counts differ")
    cache_root = manifest_path.parent.resolve(strict=True)
    calls: list[D9DCall] = []
    previous_step = -1
    for ordinal, (manifest, item, run, policy) in enumerate(
        zip(manifests, inventory, runtime, telemetry, strict=True)
    ):
        array_relative = str(manifest.get("array_path"))
        step_id = manifest.get("step_id")
        selected_layer = manifest.get("teacher_exit_layer")
        context = run.get("context", {})
        if (
            manifest.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION
            or manifest.get("teacher_kind") != PHASE_ROUTE_TEACHER_KIND
            or manifest.get("checkpoint_sha256") != D2_CHECKPOINT_SHA256
            or manifest.get("episode_id") != record.canonical_key
            or manifest.get("task_id") != record.task_id
            or not has_complete_candidate_fm_traces(manifest)
            or selected_layer not in D9D_REPLAY_LAYERS
            or type(step_id) is not int
            or step_id <= previous_step
            or item.get("relative_path") != array_relative
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or context.get("episode_id") != record.canonical_key
            or context.get("task_id") != record.task_id
            or context.get("call_ordinal") != ordinal
            or context.get("step_id") != step_id
            or run.get("selected_layer") != selected_layer
            or run.get("prepared") is not True
            or run.get("committed") is not True
            or policy.get("episode_id") != record.canonical_key
            or policy.get("task_id") != record.task_id
            or policy.get("step_id") != step_id
            or policy.get("exit_layer") != selected_layer
        ):
            raise D9DReplayError(
                f"D9D call identity differs: {record.canonical_key} call {ordinal}"
            )
        array_path = _safe_child(
            cache_root, array_relative, context="D9D cache array"
        )
        if array_path.stat().st_size != item["bytes"]:
            raise D9DReplayError("D9D cache array byte size differs")
        calls.append(
            D9DCall(
                dataset_index=dataset_index_start + ordinal,
                task_id=record.task_id,
                episode_index=record.episode_index,
                seed=record.seed,
                canonical_key=record.canonical_key,
                call_ordinal=ordinal,
                step_id=step_id,
                selected_layer=int(selected_layer),
                array_path=array_path,
                array_relative_path=array_relative,
                array_bytes=int(item["bytes"]),
                array_sha256=str(item["sha256"]),
                arm_result_sha256=result_sha,
                manifest_sha256=manifest_sha,
                inventory_sha256=inventory_sha,
            )
        )
        previous_step = step_id
    return tuple(calls)


def load_d9d_calls(repo_root: str | Path) -> tuple[D9DCall, ...]:
    """Load the complete deterministic call index without opening NPZ files."""

    root = Path(repo_root).resolve(strict=True)
    collection = validate_d9c_collection(root)
    records = load_d9_selection_metadata(root)
    bindings = collection["arm_payload_binding"]
    calls: list[D9DCall] = []
    for record in records:
        pair_binding = bindings.get(record.canonical_key)
        if not isinstance(pair_binding, Mapping):
            raise D9DReplayError("D9D pair binding is missing")
        arm_binding = pair_binding.get(PHASE_ROUTE_ARM)
        if not isinstance(arm_binding, Mapping):
            raise D9DReplayError("D9D PhaseRoute arm binding is missing")
        calls.extend(
            _validate_arm_calls(
                root,
                record=record,
                arm_binding=arm_binding,
                dataset_index_start=len(calls),
            )
        )
    if (
        len(calls) != D9D_EXPECTED_ROWS
        or [call.dataset_index for call in calls] != list(range(D9D_EXPECTED_ROWS))
        or set(call.task_id for call in calls) != set(D9_TASK_IDS)
    ):
        raise D9DReplayError("D9D global call index coverage differs")
    return tuple(calls)


def validate_gpu_contract(
    *,
    shard_index: int,
    physical_gpu_index: int,
    visible_devices: str | None,
    visible_gpu_count: int,
    expected_gpu_uuid: str,
    observed_gpu_uuid: str,
) -> None:
    if (
        type(shard_index) is not int
        or shard_index not in range(D9D_SHARD_COUNT)
        or physical_gpu_index != shard_index
        or visible_devices != str(physical_gpu_index)
        or visible_gpu_count != 1
        or not expected_gpu_uuid.startswith("GPU-")
        or observed_gpu_uuid.removeprefix("GPU-")
        != expected_gpu_uuid.removeprefix("GPU-")
    ):
        raise D9DReplayError("D9D front-four GPU contract differs")


def build_call_truth(
    candidate_actions: torch.Tensor,
    *,
    selected_layer: int,
    online_selected_action: torch.Tensor,
) -> D9DCallTruth:
    """Construct one per-call same-noise truth record without aggregation."""

    if (
        candidate_actions.device.type != "cpu"
        or candidate_actions.shape != (3, 8, 7)
        or not candidate_actions.is_floating_point()
        or online_selected_action.device.type != "cpu"
        or online_selected_action.shape != (8, 7)
        or not online_selected_action.is_floating_point()
        or not bool(torch.isfinite(candidate_actions).all())
        or not bool(torch.isfinite(online_selected_action).all())
        or selected_layer not in D9D_REPLAY_LAYERS
    ):
        raise D9DReplayError("D9D truth action geometry differs")
    selected_index = D9D_REPLAY_LAYERS.index(selected_layer)
    selected = candidate_actions[selected_index].float().contiguous()
    online = online_selected_action.float().contiguous()
    error = float((selected - online).abs().max().item())
    reference = candidate_actions[2].double()
    similarity = torch.nn.functional.cosine_similarity(
        selected.double(), reference, dim=-1, eps=1.0e-8
    )
    distance = float((1.0 - similarity).mean().item())
    if not math.isfinite(distance) or distance < -1.0e-12:
        raise D9DReplayError("D9D full-action distance is invalid")
    distance = max(0.0, distance)
    gripper_unsafe = bool(
        ((selected[:, 6] >= 0.0) != (reference[:, 6] >= 0.0)).any().item()
    )
    return D9DCallTruth(
        selected_candidate_index=selected_index,
        selected_replay_max_abs_error=error,
        selected_replay_within_atol=error <= D9D_SELECTED_REPLAY_ATOL,
        selected_replay_bit_exact=torch.equal(selected, online),
        full_action_distance=distance,
        full_action_unsafe=distance > D9D_ACTION_THRESHOLD,
        gripper_unsafe=gripper_unsafe,
        severe_full_action=distance > D9D_SEVERE_RATIO * D9D_ACTION_THRESHOLD,
    )


def hash_online_action(value: np.ndarray | torch.Tensor) -> str:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    return sha256_array(np.asarray(array, dtype=np.float32))


__all__ = [
    "D9C_COLLECTION_SHA256",
    "D9C_SOURCE_GIT_COMMIT",
    "D9D_ACTION_THRESHOLD",
    "D9DCall",
    "D9DCallTruth",
    "D9D_COLLECTION_SCHEMA_VERSION",
    "D9D_COLLECTION_STATUS",
    "D9D_EXPECTED_ROWS",
    "D9D_LOG_RELATIVE_PATH",
    "D9D_OUTPUT_RELATIVE_PATH",
    "D9D_REPLAY_LAYERS",
    "D9DReplayError",
    "D9D_RUNNER_READINESS_RELATIVE_PATH",
    "D9D_RUNNER_READINESS_STATUS",
    "D9D_SELECTED_REPLAY_ATOL",
    "D9D_SEVERE_RATIO",
    "D9D_SHARD_COUNT",
    "D9D_SHARD_RESULT_SCHEMA_VERSION",
    "D9D_SHARD_SCHEMA_VERSION",
    "D9D_SHARD_STATUS",
    "build_call_truth",
    "hash_online_action",
    "load_d9d_calls",
    "validate_d9c_collection",
    "validate_d9d_runner_readiness",
    "validate_gpu_contract",
]
