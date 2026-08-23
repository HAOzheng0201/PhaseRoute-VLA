"""Authenticated D8C collection primitives for generated-state confirmation.

D8C is prospective only with respect to the generated MuJoCo states.  The
frozen original A1 early-exit policy remains the behavior policy.  This module
keeps generated-state replicate identity separate from all official LIBERO
episode indices and binds every collection step to the D8 readiness artifact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import numpy as np
import torch

from a1.vla.dynamic_compute.vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)

from .d8_artifacts import D8A_PAYLOAD_SCHEMA_VERSION, canonical_state_bytes
from .development_collection import (
    D2_CHECKPOINT_SHA256,
    D2_REPLAY_LAYERS,
    D2_TASK_IDS,
    stream_sha256,
)
from .fresh_confirmation import (
    D8_CLUSTER_COUNT,
    D8_CLUSTERS_PER_TASK,
    D8_CONTRACT_SHA256,
    D8_POLICY_SEED_BASE,
    D8_REPLICATE_IDS,
    D8_SCHEDULE_SHA256,
    D8_TASK_IDS,
    FreshConfirmationRecord,
    load_d8_contract,
    load_fresh_confirmation_schedule,
)


D8C_ROLE = "prospective_generated_state_shadow_confirmation"
D8C_SUITE = "libero_10"
D8C_FM_STEPS = 10
D8C_REPLAY_LAYERS = D2_REPLAY_LAYERS
D8C_ALLOWED_PHYSICAL_GPUS = (0, 1, 2, 3)

D8C_RAW_TASK_SCHEMA_VERSION = "phase-route-vla.v3.d8c-raw-task-result.v1"
D8C_CONTEXT_SCHEMA_VERSION = "phase-route-vla.v3.d8c-context.v1"
D8C_CANDIDATE_SCHEMA_VERSION = "phase-route-vla.v3.d8c-candidates.v1"
D8C_DATASET_SCHEMA_VERSION = "phase-route-vla.v3.d8c-dataset.v1"
D8C_COLLECTION_RESULT_SCHEMA_VERSION = (
    "phase-route-vla.v3.d8c-formal-collection-result.v1"
)

D8_READINESS_RELATIVE_PATH = Path("results/v3/v3_d8_readiness_attestation.json")
D8_READINESS_SHA256 = (
    "cb13d48898c189814cc3bf02b2cb3171f7df307c3261a2fb7378c8c7a8b34829"
)
D8A_RESULT_RELATIVE_PATH = Path("reports/v3_d8_fresh_states/result.json")
D8A_RESULT_SHA256 = (
    "ff45ff5cc5e4e9f9f61b9ee8d80cbe54b896760e066f11710a063c4b0914d622"
)
D8A_PAYLOAD_RELATIVE_PATH = Path("reports/v3_d8_fresh_states/fresh_states.pt")
D8A_PAYLOAD_SHA256 = (
    "203e34b0049148b9954c42b6d44ceeb9408edaf0fd073080b95e4d2958c6d56f"
)
D8B_RESULT_RELATIVE_PATH = Path("reports/v3_d8_final_router/result.json")
D8B_RESULT_SHA256 = (
    "76d209ef3e92dcf2a4edb329337a0481d8976ee2382d634de172904724cda70d"
)
D8B_PAYLOAD_RELATIVE_PATH = Path("reports/v3_d8_final_router/final_router.pt")
D8B_PAYLOAD_SHA256 = (
    "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830"
)

_FRESH_ID = re.compile(
    r"^libero_10:task([0-9]+):fresh_confirm_v1:replicate([0-9]+)$"
)


class D8CCollectionError(ValueError):
    """Raised when D8C evidence or identity violates the frozen protocol."""


@dataclass(frozen=True)
class FreshConfirmationCall:
    dataset_index: int
    task_id: int
    replicate_id: int
    policy_seed: int
    cluster_key: str
    call_ordinal: int
    step_id: int
    behavior_exit_layer: int
    cache_directory: Path
    array_path: str
    source_manifest_line: int

    @property
    def episode_index(self) -> int:
        """Compatibility key for past-only history; never an official episode."""

        return self.replicate_id

    @property
    def group_key(self) -> str:
        return self.cluster_key


class FreshStateTaskSuite:
    """Proxy a LIBERO suite while exposing only the 20 frozen D8A states/task."""

    def __init__(
        self,
        base_suite: Any,
        states_by_task: Mapping[int, tuple[np.ndarray, ...]],
    ) -> None:
        if set(states_by_task) != set(D8_TASK_IDS):
            raise D8CCollectionError("D8C fresh-state task coverage differs")
        copied: dict[int, tuple[np.ndarray, ...]] = {}
        for task_id in D8_TASK_IDS:
            states = tuple(states_by_task[task_id])
            if len(states) != D8_CLUSTERS_PER_TASK:
                raise D8CCollectionError("D8C fresh-state replicate coverage differs")
            checked = []
            for state in states:
                canonical, _raw, _digest = canonical_state_bytes(state)
                checked.append(canonical.copy())
            copied[task_id] = tuple(checked)
        self._base_suite = base_suite
        self._states_by_task = copied

    def get_task_init_states(self, task_id: int) -> list[np.ndarray]:
        if type(task_id) is not int or task_id not in D8_TASK_IDS:
            raise D8CCollectionError("D8C task id must be in 0..9")
        return [state.copy() for state in self._states_by_task[task_id]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_suite, name)


def _json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D8CCollectionError(f"{context} cannot be read") from error
    if not isinstance(value, Mapping):
        raise D8CCollectionError(f"{context} must be an object")
    return dict(value)


def _regular_file(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise D8CCollectionError(f"{context} contains a symlink component")
    try:
        metadata = absolute.stat()
    except FileNotFoundError as error:
        raise D8CCollectionError(f"{context} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise D8CCollectionError(f"{context} must be a regular file")
    return absolute.resolve(strict=True)


def parse_fresh_cluster_key(value: object) -> tuple[int, int]:
    match = _FRESH_ID.fullmatch(str(value))
    if match is None:
        raise D8CCollectionError("D8C cluster identity is not canonical fresh_confirm_v1")
    task_id, replicate_id = map(int, match.groups())
    if task_id not in D8_TASK_IDS or replicate_id not in D8_REPLICATE_IDS:
        raise D8CCollectionError("D8C cluster identity is outside the frozen schedule")
    return task_id, replicate_id


def validate_episode_id_override(value: object, *, task_id: int, replicate_id: int) -> str:
    observed_task, observed_replicate = parse_fresh_cluster_key(value)
    if observed_task != task_id or observed_replicate != replicate_id:
        raise D8CCollectionError("D8C override identity differs from task/replicate")
    result = str(value)
    if ":episode" in result or any(
        f"episode{index}" in result for index in range(40, 50)
    ):
        raise D8CCollectionError("D8C fresh identity aliases an official episode")
    return result


def validate_d8c_prerequisites(repo_root: str | Path) -> dict[str, Any]:
    """Bind D8C to the frozen readiness, states, router, and schedule."""

    root = Path(repo_root).resolve(strict=True)
    contract = load_d8_contract(root)
    schedule = load_fresh_confirmation_schedule(root)
    expected = {
        "readiness": (D8_READINESS_RELATIVE_PATH, D8_READINESS_SHA256),
        "D8A_result": (D8A_RESULT_RELATIVE_PATH, D8A_RESULT_SHA256),
        "D8A_payload": (D8A_PAYLOAD_RELATIVE_PATH, D8A_PAYLOAD_SHA256),
        "D8B_result": (D8B_RESULT_RELATIVE_PATH, D8B_RESULT_SHA256),
        "D8B_payload": (D8B_PAYLOAD_RELATIVE_PATH, D8B_PAYLOAD_SHA256),
    }
    observed = {
        name: stream_sha256(root / relative)
        for name, (relative, _digest) in expected.items()
    }
    if any(observed[name] != digest for name, (_path, digest) in expected.items()):
        raise PermissionError("D8C prerequisite artifact SHA-256 differs")
    readiness = _json_object(
        root / D8_READINESS_RELATIVE_PATH, context="D8C readiness"
    )
    bound = readiness.get("bound_artifacts", {})
    if (
        readiness.get("status") != "PASS_V3_D8A_D8B_READINESS"
        or readiness.get("source_worktree_dirty") is not False
        or readiness.get("authorization", {}).get("next_stage")
        != "D8C_PROSPECTIVE_SHADOW_COLLECTION_AND_REPLAY"
        or readiness.get("authorization", {}).get("open_episode_40_49") is not False
        or readiness.get("authorization", {}).get("active_control") is not False
        or bound.get("D8_contract_sha256") != D8_CONTRACT_SHA256
        or bound.get("D8_schedule_sha256") != D8_SCHEDULE_SHA256
        or bound.get("D8A_result_sha256") != D8A_RESULT_SHA256
        or bound.get("D8A_payload_sha256") != D8A_PAYLOAD_SHA256
        or bound.get("D8B_result_sha256") != D8B_RESULT_SHA256
        or bound.get("D8B_payload_sha256") != D8B_PAYLOAD_SHA256
        or contract.get("prospective_collection", {}).get("behavior_policy")
        != "frozen_original_A1_early_exit_controller"
    ):
        raise PermissionError("D8C readiness authorization semantics differ")
    if len(schedule) != D8_CLUSTER_COUNT:
        raise D8CCollectionError("D8C schedule count differs")
    return {
        "D8_contract_sha256": D8_CONTRACT_SHA256,
        "D8_schedule_sha256": D8_SCHEDULE_SHA256,
        "D8_readiness_sha256": D8_READINESS_SHA256,
        "D8A_result_sha256": D8A_RESULT_SHA256,
        "D8A_payload_sha256": D8A_PAYLOAD_SHA256,
        "D8B_result_sha256": D8B_RESULT_SHA256,
        "D8B_payload_sha256": D8B_PAYLOAD_SHA256,
        "clusters": len(schedule),
        "authorization": "D8C_PROSPECTIVE_SHADOW_COLLECTION_AND_REPLAY",
    }


def load_fresh_states(
    repo_root: str | Path,
) -> tuple[tuple[FreshConfirmationRecord, ...], dict[int, tuple[np.ndarray, ...]]]:
    """Load and fully authenticate the D8A payload in frozen schedule order."""

    root = Path(repo_root).resolve(strict=True)
    validate_d8c_prerequisites(root)
    schedule = load_fresh_confirmation_schedule(root)
    payload = torch.load(
        root / D8A_PAYLOAD_RELATIVE_PATH, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != D8A_PAYLOAD_SCHEMA_VERSION
        or payload.get("D8_contract_sha256") != D8_CONTRACT_SHA256
        or payload.get("D8_schedule_sha256") != D8_SCHEDULE_SHA256
        or payload.get("cluster_keys") != [record.cluster_key for record in schedule]
        or payload.get("policy_rollout_performed") is not False
        or payload.get("official_episode_identity_used") is not False
        or len(payload.get("states", [])) != D8_CLUSTER_COUNT
    ):
        raise PermissionError("D8C D8A payload semantics differ")
    expected_columns = {
        "task_id": [record.task_id for record in schedule],
        "replicate_id": [record.replicate_id for record in schedule],
        "state_seed": [record.state_seed for record in schedule],
        "policy_seed": [record.policy_seed for record in schedule],
    }
    for name, values in expected_columns.items():
        if not torch.equal(payload[name], torch.tensor(values)):
            raise PermissionError(f"D8C D8A payload {name} differs")
    states_by_task: dict[int, list[np.ndarray]] = {task: [] for task in D8_TASK_IDS}
    for record, state, expected_hash in zip(
        schedule, payload["states"], payload["state_sha256"], strict=True
    ):
        if not isinstance(state, torch.Tensor) or state.device.type != "cpu":
            raise D8CCollectionError("D8C fresh state must be a CPU tensor")
        canonical, _raw, digest = canonical_state_bytes(state.numpy())
        if digest != expected_hash:
            raise PermissionError("D8C fresh state SHA-256 differs")
        states_by_task[record.task_id].append(canonical.copy())
    frozen = {task: tuple(values) for task, values in states_by_task.items()}
    if any(len(values) != D8_CLUSTERS_PER_TASK for values in frozen.values()):
        raise D8CCollectionError("D8C fresh state task coverage differs")
    return schedule, frozen


def task_fresh_schedule(
    schedule: tuple[FreshConfirmationRecord, ...], task_id: int
) -> tuple[FreshConfirmationRecord, ...]:
    if type(task_id) is not int or task_id not in D8_TASK_IDS:
        raise D8CCollectionError("D8C task id must be in 0..9")
    selected = tuple(record for record in schedule if record.task_id == task_id)
    if tuple(record.replicate_id for record in selected) != D8_REPLICATE_IDS:
        raise D8CCollectionError("D8C task schedule must contain replicates 0..19")
    return selected


def load_fresh_task_calls(
    task_output_directory: str | Path,
    *,
    task_id: int,
    dataset_index_start: int = 0,
) -> tuple[FreshConfirmationCall, ...]:
    """Read one D8C raw manifest without opening NPZ payloads."""

    if task_id not in D8_TASK_IDS:
        raise D8CCollectionError("D8C task id must be in 0..9")
    if type(dataset_index_start) is not int or dataset_index_start < 0:
        raise D8CCollectionError("D8C dataset index start must be non-negative")
    output = Path(task_output_directory).resolve(strict=True)
    manifest = _regular_file(
        output / "teacher_calls" / "manifest.jsonl", context="D8C teacher manifest"
    )
    counters: dict[int, int] = defaultdict(int)
    previous_step: dict[int, int] = {}
    replicate_order: list[int] = []
    rows: list[FreshConfirmationCall] = []
    with manifest.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise D8CCollectionError("D8C manifest contains an empty line")
            source = json.loads(line)
            if not isinstance(source, Mapping):
                raise D8CCollectionError("D8C manifest row must be an object")
            source_task, replicate = parse_fresh_cluster_key(source.get("episode_id"))
            if source_task != task_id or source.get("task_id") != task_id:
                raise D8CCollectionError("D8C manifest task identity differs")
            if source.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION:
                raise D8CCollectionError("D8C raw cache schema differs")
            if source.get("checkpoint_sha256") != D2_CHECKPOINT_SHA256:
                raise D8CCollectionError("D8C raw cache checkpoint differs")
            if source.get("teacher_kind") != "a1_early_exit":
                raise D8CCollectionError("D8C behavior teacher kind differs")
            if not has_complete_candidate_fm_traces(source):
                raise D8CCollectionError("D8C raw cache FM trace is incomplete")
            step = source.get("step_id")
            if type(step) is not int or step < 0 or step <= previous_step.get(replicate, -1):
                raise D8CCollectionError("D8C replicate steps are not increasing")
            previous_step[replicate] = step
            if counters[replicate] == 0:
                replicate_order.append(replicate)
            ordinal = counters[replicate]
            counters[replicate] += 1
            cluster_key = str(source["episode_id"])
            rows.append(
                FreshConfirmationCall(
                    dataset_index=dataset_index_start + len(rows),
                    task_id=task_id,
                    replicate_id=replicate,
                    policy_seed=D8_POLICY_SEED_BASE + task_id * 10_000 + replicate,
                    cluster_key=cluster_key,
                    call_ordinal=ordinal,
                    step_id=step,
                    behavior_exit_layer=int(source["teacher_exit_layer"]),
                    cache_directory=manifest.parent,
                    array_path=str(source["array_path"]),
                    source_manifest_line=line_number,
                )
            )
    if (
        tuple(replicate_order) != D8_REPLICATE_IDS
        or tuple(sorted(counters)) != D8_REPLICATE_IDS
        or any(counters[replicate] < 1 for replicate in D8_REPLICATE_IDS)
    ):
        raise D8CCollectionError("D8C task manifest does not cover replicates 0..19")
    return tuple(rows)


def resolve_fresh_call_payload(call: FreshConfirmationCall) -> Path:
    parse_fresh_cluster_key(call.cluster_key)
    root = call.cache_directory.resolve(strict=True)
    if call.cache_directory.is_symlink() or not root.is_dir():
        raise D8CCollectionError("D8C cache directory must be regular")
    relative = Path(call.array_path)
    if (
        relative.is_absolute()
        or relative.suffix != ".npz"
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise D8CCollectionError("D8C cache payload path is unsafe")
    path = _regular_file(root / relative, context="D8C cache payload")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise D8CCollectionError("D8C cache payload escapes its directory") from error
    return path


def validate_d8c_gpu_contract(
    *,
    physical_gpu_index: int,
    visible_devices: str | None,
    visible_gpu_count: int,
    expected_gpu_uuid: str,
    observed_gpu_uuid: str,
) -> None:
    if physical_gpu_index not in D8C_ALLOWED_PHYSICAL_GPUS:
        raise PermissionError("D8C permits physical GPUs 0--3 only")
    if visible_devices != str(physical_gpu_index) or visible_gpu_count != 1:
        raise PermissionError("D8C requires exactly one assigned visible GPU")
    normalize = lambda value: str(value).strip().lower().removeprefix("gpu-")
    if not expected_gpu_uuid or normalize(expected_gpu_uuid) != normalize(
        observed_gpu_uuid
    ):
        raise PermissionError("D8C visible GPU UUID differs")


def hash_state(value: Any) -> str:
    _state, raw, digest = canonical_state_bytes(value)
    if digest != hashlib.sha256(raw).hexdigest():
        raise AssertionError("unreachable D8C state hash inconsistency")
    return digest


__all__ = [
    "D8A_PAYLOAD_SHA256",
    "D8B_PAYLOAD_SHA256",
    "D8C_ALLOWED_PHYSICAL_GPUS",
    "D8C_CANDIDATE_SCHEMA_VERSION",
    "D8C_COLLECTION_RESULT_SCHEMA_VERSION",
    "D8C_CONTEXT_SCHEMA_VERSION",
    "D8C_DATASET_SCHEMA_VERSION",
    "D8C_FM_STEPS",
    "D8C_RAW_TASK_SCHEMA_VERSION",
    "D8C_REPLAY_LAYERS",
    "D8C_ROLE",
    "D8C_SUITE",
    "D8CCollectionError",
    "D8_READINESS_SHA256",
    "FreshConfirmationCall",
    "FreshStateTaskSuite",
    "hash_state",
    "load_fresh_states",
    "load_fresh_task_calls",
    "parse_fresh_cluster_key",
    "resolve_fresh_call_payload",
    "task_fresh_schedule",
    "validate_d8c_gpu_contract",
    "validate_d8c_prerequisites",
    "validate_episode_id_override",
]
