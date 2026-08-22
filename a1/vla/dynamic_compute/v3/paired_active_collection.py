"""Protocol guards and evidence helpers for the frozen V3-D9C collection.

This module deliberately contains no LIBERO environment construction.  It can
therefore be exercised before the independent-test states are opened.  The
GPU runner imports these helpers only after a separately frozen readiness
attestation has bound the runner and all D9B-protected control code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from a1.vla.dynamic_compute.v3.independent_test_protocol import (
    D9_ARMS,
    D9_EPISODE_INDICES,
    D9_GPU_ALLOWLIST,
    D9_RECORDS_PER_TASK,
    D9_TASK_IDS,
    D9TestRecord,
    load_d9_contract,
    load_d9_selection_metadata,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)


D9B_READINESS_RELATIVE_PATH = Path(
    "results/v3/v3_d9b_readiness_attestation.json"
)
D9B_READINESS_SHA256 = (
    "a768d7ee3f123d6858fc850467deb7883afec2e3af2cf40921f9b4e7cfcb03f1"
)
D9B_READINESS_STATUS = (
    "PASS_V3_D9B_READINESS_FOR_ONE_SHOT_PAIRED_ACTIVE_TEST"
)
D9C_RUNNER_READINESS_RELATIVE_PATH = Path(
    "results/v3/v3_d9c_runner_readiness.json"
)
D9C_RUNNER_READINESS_STATUS = "PASS_V3_D9C_FROZEN_RUNNER_READINESS"
D9C_ARM_SCHEMA_VERSION = "phase-route-vla.v3.d9c-arm-result.v1"
D9C_PAIR_SCHEMA_VERSION = "phase-route-vla.v3.d9c-pair-record.v1"
D9C_TASK_SCHEMA_VERSION = "phase-route-vla.v3.d9c-task-collection.v1"
D9C_COLLECTION_SCHEMA_VERSION = "phase-route-vla.v3.d9c-collection.v1"
D9C_TASK_STATUS = "COMPLETE_V3_D9C_TASK_PAIRED_ACTIVE_COLLECTION"
D9C_COLLECTION_STATUS = "COMPLETE_V3_D9C_PAIRED_ACTIVE_COLLECTION"
D9C_INCOMPLETE_STATUS = "INCOMPLETE_V3_D9_INDEPENDENT_TEST_NOT_PASS_OR_NEGATIVE"
D9C_OUTPUT_RELATIVE_PATH = Path("reports/v3_d9c_paired_active")
D9C_LAUNCH_LOG_RELATIVE_PATH = Path("reports/v3_d9c_launch_logs")
PHASE_ROUTE_ARM = "frozen_PhaseRoute_D8"
ORIGINAL_A1_ARM = "frozen_original_A1"
PHASE_ROUTE_TEACHER_KIND = "phase_route_selected_action"


class D9CCollectionError(ValueError):
    """Raised when D9C evidence differs from the frozen protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    """Hash one state with explicit dtype and shape domain separation."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(b"phase-route-vla.d9c.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise D9CCollectionError(f"JSON object required: {path}")
    return dict(value)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise D9CCollectionError(
                f"JSONL record {ordinal} is not an object: {path}"
            )
        records.append(dict(value))
    return records


def task_schedule(repo_root: str | Path, task_id: int) -> tuple[D9TestRecord, ...]:
    if type(task_id) is not int or task_id not in D9_TASK_IDS:
        raise D9CCollectionError("D9C task id must be in 0..9")
    records = tuple(
        record
        for record in load_d9_selection_metadata(repo_root)
        if record.task_id == task_id
    )
    if (
        len(records) != D9_RECORDS_PER_TASK
        or tuple(record.episode_index for record in records)
        != D9_EPISODE_INDICES
        or any(record.physical_gpu_index != task_id % 4 for record in records)
    ):
        raise D9CCollectionError("D9C task schedule differs from freeze")
    return records


def expected_task_output(repo_root: str | Path, task_id: int) -> Path:
    if task_id not in D9_TASK_IDS:
        raise D9CCollectionError("D9C task id must be in 0..9")
    return (Path(repo_root).resolve() / D9C_OUTPUT_RELATIVE_PATH / f"task{task_id}")


def validate_task_output(
    repo_root: str | Path, task_id: int, requested: str | Path
) -> Path:
    expected = expected_task_output(repo_root, task_id)
    observed = Path(requested).resolve()
    if observed != expected:
        raise D9CCollectionError(
            f"D9C task output differs: expected {expected}, got {observed}"
        )
    return observed


def validate_gpu_contract(
    *,
    task_id: int,
    physical_gpu_index: int,
    visible_devices: str | None,
    visible_gpu_count: int,
    expected_gpu_uuid: str,
    observed_gpu_uuid: str,
) -> None:
    assigned = task_id % 4
    if (
        task_id not in D9_TASK_IDS
        or physical_gpu_index != assigned
        or physical_gpu_index not in D9_GPU_ALLOWLIST
        or visible_devices != str(physical_gpu_index)
        or visible_gpu_count != 1
        or not expected_gpu_uuid.startswith("GPU-")
        or observed_gpu_uuid.removeprefix("GPU-")
        != expected_gpu_uuid.removeprefix("GPU-")
    ):
        raise D9CCollectionError("D9C physical-GPU assignment differs")


def validate_d9b_readiness(repo_root: str | Path) -> dict[str, Any]:
    """Verify D9B and every control-code digest it froze."""

    root = Path(repo_root).resolve(strict=True)
    load_d9_contract(root)
    path = root / D9B_READINESS_RELATIVE_PATH
    if sha256_file(path) != D9B_READINESS_SHA256:
        raise D9CCollectionError("D9B readiness SHA-256 differs")
    readiness = read_json_object(path)
    if (
        readiness.get("status") != D9B_READINESS_STATUS
        or not all(readiness.get("readiness_checks", {}).values())
        or readiness.get("authorization", {}).get("authorized")
        != "D9C_ONE_SHOT_PAIRED_ACTIVE_INDEPENDENT_TEST"
        or readiness.get("authorization", {}).get("exact_schedule_only") is not True
        or readiness.get("access_ledger", {}).get(
            "LIBERO_episode_40_49_init_states_opened"
        )
        is not False
        or readiness.get("access_ledger", {}).get("active_control") is not False
    ):
        raise D9CCollectionError("D9B readiness semantics differ")
    bound_code = readiness.get("bound_code_sha256")
    if not isinstance(bound_code, Mapping) or not bound_code:
        raise D9CCollectionError("D9B bound-code inventory is missing")
    mismatches = {
        str(relative): {
            "expected": expected,
            "observed": sha256_file(root / str(relative)),
        }
        for relative, expected in bound_code.items()
        if sha256_file(root / str(relative)) != expected
    }
    if mismatches:
        raise D9CCollectionError(f"D9B-protected code changed: {mismatches}")
    return {
        "path": D9B_READINESS_RELATIVE_PATH.as_posix(),
        "sha256": D9B_READINESS_SHA256,
        "source_git_commit": readiness["source_git_commit"],
        "bound_code_files": len(bound_code),
    }


def validate_runner_readiness(repo_root: str | Path) -> dict[str, Any]:
    """Verify the committed D9C runner attestation without a self-hash cycle."""

    root = Path(repo_root).resolve(strict=True)
    path = root / D9C_RUNNER_READINESS_RELATIVE_PATH
    readiness = read_json_object(path)
    if (
        readiness.get("status") != D9C_RUNNER_READINESS_STATUS
        or not all(readiness.get("checks", {}).values())
        or readiness.get("access_ledger", {}).get("official_test_states_opened")
        is not False
        or readiness.get("access_ledger", {}).get("active_rollouts") != 0
    ):
        raise D9CCollectionError("D9C runner readiness semantics differ")
    bound_code = readiness.get("bound_code_sha256")
    if not isinstance(bound_code, Mapping) or not bound_code:
        raise D9CCollectionError("D9C runner bound-code inventory is missing")
    for relative, expected in bound_code.items():
        if sha256_file(root / str(relative)) != expected:
            raise D9CCollectionError(f"D9C runner code changed: {relative}")
    return {
        "path": D9C_RUNNER_READINESS_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "source_git_commit": readiness["source_git_commit"],
        "bound_code_files": len(bound_code),
    }


def validate_arm_name(arm: str) -> str:
    if arm not in D9_ARMS:
        raise D9CCollectionError(f"unknown D9C arm: {arm}")
    return arm


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise D9CCollectionError(f"{name} must be a non-negative integer")
    return value


def summarize_policy_telemetry(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    expected_episode_id: str,
    expected_task_id: int,
) -> dict[str, Any]:
    """Validate and summarize raw records for exactly one rollout.

    Per-rollout accounting is required evidence, not an interim cross-pair
    performance analysis.  No success value is consumed by this function.
    """

    validate_arm_name(arm)
    if not records:
        raise D9CCollectionError("a completed D9C rollout has no policy calls")
    exit_counts: Counter[int] = Counter()
    fm_calls = 0
    fm_steps = 0
    latency_ms: list[float] = []
    step_ids: list[int] = []
    route_decisions = 0
    for ordinal, raw in enumerate(records):
        record = dict(raw)
        if (
            record.get("episode_id") != expected_episode_id
            or record.get("task_id") != expected_task_id
        ):
            raise D9CCollectionError("policy telemetry identity differs")
        step_id = _strict_nonnegative_int(record.get("step_id"), "step_id")
        if step_ids and step_id <= step_ids[-1]:
            raise D9CCollectionError("policy telemetry steps are not increasing")
        step_ids.append(step_id)
        layer = _strict_nonnegative_int(record.get("exit_layer"), "exit_layer")
        if arm == PHASE_ROUTE_ARM and layer not in (11, 13, 27):
            raise D9CCollectionError("PhaseRoute selected a non-frozen layer")
        if arm == ORIGINAL_A1_ARM and layer not in range(1, 28, 2):
            raise D9CCollectionError("original A1 selected an invalid exit layer")
        exit_counts[layer] += 1
        fm_calls += _strict_nonnegative_int(record.get("fm_calls"), "fm_calls")
        fm_steps += _strict_nonnegative_int(
            record.get("fm_steps_total"), "fm_steps_total"
        )
        latency = float(record.get("latency_ms"))
        if not math.isfinite(latency) or latency < 0:
            raise D9CCollectionError("policy latency must be finite and non-negative")
        latency_ms.append(latency)
        events = record.get("extra", {}).get("exit_events", [])
        if not isinstance(events, Sequence):
            raise D9CCollectionError("exit_events must be a sequence")
        decisions = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event") == "phase_route_decision"
        ]
        if arm == PHASE_ROUTE_ARM:
            if len(decisions) != 1 or decisions[0].get("selected_layer") != layer:
                raise D9CCollectionError(
                    f"PhaseRoute decision mismatch at policy call {ordinal}"
                )
            route_decisions += 1
        elif decisions:
            raise D9CCollectionError("original A1 telemetry contains PhaseRoute decisions")
    latency_array = np.asarray(latency_ms, dtype=np.float64)
    return {
        "policy_calls": len(records),
        "fm_calls": fm_calls,
        "fm_steps": fm_steps,
        "exit_layer_counts": {
            f"L{layer}": int(exit_counts[layer]) for layer in sorted(exit_counts)
        },
        "phase_route_decisions": route_decisions,
        "first_policy_step": step_ids[0],
        "last_policy_step": step_ids[-1],
        "policy_latency_ms": {
            "sum": float(latency_array.sum()),
            "min": float(latency_array.min()),
            "max": float(latency_array.max()),
        },
    }


def validate_phase_route_runtime_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_episode_id: str,
    expected_task_id: int,
    expected_policy_calls: int,
) -> dict[str, Any]:
    if len(records) != expected_policy_calls:
        raise D9CCollectionError("PhaseRoute runtime-record count differs")
    selected: Counter[int] = Counter()
    errors = 0
    for ordinal, raw in enumerate(records):
        record = dict(raw)
        context = record.get("context", {})
        if (
            context.get("episode_id") != expected_episode_id
            or context.get("task_id") != expected_task_id
            or context.get("call_ordinal") != ordinal
            or record.get("prepared") is not True
            or record.get("committed") is not True
            or record.get("selected_layer") not in (11, 13, 27)
        ):
            raise D9CCollectionError("PhaseRoute runtime record differs")
        selected[int(record["selected_layer"])] += 1
        raw_errors = record.get("errors", [])
        if not isinstance(raw_errors, Sequence):
            raise D9CCollectionError("PhaseRoute runtime errors must be a sequence")
        errors += len(raw_errors)
    return {
        "records": len(records),
        "selected_layer_counts": {
            f"L{layer}": int(selected[layer]) for layer in (11, 13, 27)
        },
        "fail_closed_error_events": errors,
    }


def validate_phase_route_cache(
    manifest_records: Sequence[Mapping[str, Any]],
    *,
    expected_episode_id: str,
    expected_task_id: int,
    expected_policy_calls: int,
) -> dict[str, Any]:
    if len(manifest_records) != expected_policy_calls:
        raise D9CCollectionError("PhaseRoute cache count differs from policy calls")
    keys: set[tuple[str, int]] = set()
    early_calls = 0
    for record in manifest_records:
        key = (str(record.get("episode_id")), int(record.get("step_id", -1)))
        if (
            key[0] != expected_episode_id
            or record.get("task_id") != expected_task_id
            or key in keys
            or record.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION
            or record.get("teacher_kind") != PHASE_ROUTE_TEACHER_KIND
            or not has_complete_candidate_fm_traces(record)
        ):
            raise D9CCollectionError("PhaseRoute cache manifest differs")
        keys.add(key)
        if int(record["teacher_exit_layer"]) in (11, 13):
            early_calls += 1
    return {
        "cache_records": len(manifest_records),
        "early_exit_cache_records": early_calls,
        "all_policy_calls_cached": len(manifest_records) == expected_policy_calls,
    }


@dataclass(frozen=True)
class FileInventoryRecord:
    relative_path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def build_file_inventory(
    root: str | Path, relative_paths: Iterable[str | Path]
) -> tuple[FileInventoryRecord, ...]:
    base = Path(root).resolve(strict=True)
    inventory: list[FileInventoryRecord] = []
    seen: set[str] = set()
    for relative in relative_paths:
        normalized = Path(relative)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise D9CCollectionError("inventory path must stay below its root")
        name = normalized.as_posix()
        if name in seen:
            raise D9CCollectionError("inventory contains a duplicate path")
        seen.add(name)
        path = (base / normalized).resolve(strict=True)
        if base not in path.parents or not path.is_file():
            raise D9CCollectionError("inventory entry is not a regular child file")
        inventory.append(
            FileInventoryRecord(name, path.stat().st_size, sha256_file(path))
        )
    return tuple(inventory)


def validate_pair_record(
    value: Mapping[str, Any], *, record: D9TestRecord
) -> None:
    arms = value.get("arms")
    if (
        value.get("schema_version") != D9C_PAIR_SCHEMA_VERSION
        or value.get("status") != "COMPLETE_V3_D9C_PAIRED_ACTIVE_PAIR"
        or value.get("canonical_key") != record.canonical_key
        or value.get("task_id") != record.task_id
        or value.get("episode_index") != record.episode_index
        or value.get("seed") != record.seed
        or value.get("arm_order") != list(record.arm_order)
        or not isinstance(arms, Mapping)
        or set(arms) != set(D9_ARMS)
        or any(
            not isinstance(arms[arm], Mapping)
            or arms[arm].get("status") != "COMPLETE_V3_D9C_ARM_ROLLOUT"
            for arm in D9_ARMS
        )
    ):
        raise D9CCollectionError("D9C pair record differs")
    state_hashes = {arms[arm].get("initial_state_sha256") for arm in D9_ARMS}
    commits = {arms[arm].get("source_git_commit") for arm in D9_ARMS}
    if len(state_hashes) != 1 or None in state_hashes or len(commits) != 1:
        raise D9CCollectionError("D9C paired state or commit differs between arms")


__all__ = [
    "D9B_READINESS_SHA256",
    "D9C_ARM_SCHEMA_VERSION",
    "D9C_COLLECTION_SCHEMA_VERSION",
    "D9C_COLLECTION_STATUS",
    "D9C_INCOMPLETE_STATUS",
    "D9C_LAUNCH_LOG_RELATIVE_PATH",
    "D9C_OUTPUT_RELATIVE_PATH",
    "D9C_PAIR_SCHEMA_VERSION",
    "D9C_RUNNER_READINESS_RELATIVE_PATH",
    "D9C_RUNNER_READINESS_STATUS",
    "D9C_TASK_SCHEMA_VERSION",
    "D9C_TASK_STATUS",
    "D9CCollectionError",
    "FileInventoryRecord",
    "ORIGINAL_A1_ARM",
    "PHASE_ROUTE_ARM",
    "PHASE_ROUTE_TEACHER_KIND",
    "build_file_inventory",
    "expected_task_output",
    "read_json_object",
    "read_jsonl",
    "sha256_array",
    "sha256_file",
    "summarize_policy_telemetry",
    "task_schedule",
    "validate_arm_name",
    "validate_d9b_readiness",
    "validate_gpu_contract",
    "validate_pair_record",
    "validate_phase_route_cache",
    "validate_phase_route_runtime_records",
    "validate_runner_readiness",
    "validate_task_output",
]
