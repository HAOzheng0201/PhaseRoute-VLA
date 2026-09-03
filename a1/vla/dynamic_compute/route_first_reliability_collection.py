"""Contracts for Stage-11D original-A1 development observation collection.

Only generated-state replicates 0..11 are exposed.  The frozen original A1
controller remains the behavior policy; cached tensors are observer outputs
for later CPU context construction and same-noise L13/L27 replay.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from .route_first_reliability import (
    STAGE11D_CLUSTERS_PER_TASK,
    STAGE11D_PROTOCOL_SHA256,
    STAGE11D_TASK_IDS,
    STAGE11D_TRAIN_REPLICATES,
    Stage11DRecord,
    build_stage11d_schedule,
)
from .route_first_reliability_artifacts import canonical_state_bytes, sha256_file
from .route_first_reliability_state_binding import (
    STAGE11D_STATE_BINDING_SHA256,
    load_bound_stage11d_states,
)
from .vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)


STAGE11D_COLLECTION_ROLE = "development_original_a1_observation_only"
STAGE11D_COLLECTION_SUITE = "libero_10"
STAGE11D_COLLECTION_FM_STEPS = 10
STAGE11D_COLLECTION_REPLAY_LAYERS = (13, 27)
STAGE11D_COLLECTION_ALLOWED_PHYSICAL_GPUS = tuple(range(8))
STAGE11D_COLLECTION_CLUSTER_COUNT = 120
STAGE11D_COLLECTION_CLUSTERS_PER_TASK = 12
STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH = Path(
    "runs/route_first_stage11d_development_raw"
)
STAGE11D_COLLECTION_READINESS_RELATIVE_PATH = Path(
    "results/route_first/route_first_stage11d_collection_runner_readiness.json"
)
STAGE11D_RAW_TASK_SCHEMA = "phase-route-vla.route-first-stage11d-raw-task.v1"
STAGE11D_COLLECTION_READINESS_SCHEMA = (
    "phase-route-vla.route-first-stage11d-collection-runner-readiness.v1"
)
STAGE11D_COLLECTION_READINESS_STATUS = (
    "PASS_ROUTE_FIRST_STAGE11D_COLLECTION_RUNNER_READINESS"
)
STAGE11D_A1_CHECKPOINT_SHA256 = (
    "dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
)
STAGE11D_A1_CHECKPOINT_BYTES = 33_841_175_207
STAGE11D_A1_CONFIG_SHA256 = (
    "9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca"
)
STAGE11D_A1_THRESHOLDS_SHA256 = (
    "a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796"
)
STAGE11D_A1_DATASET_STATISTICS_SHA256 = (
    "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3"
)
STAGE11D_A1_ACTION_DELTA_SHA256 = (
    "a0d0399b630953a9e0ef3b4ca09fe8a0fbde4b1ce6539ad5d911ad23fb6c812d"
)

_CLUSTER_KEY = re.compile(
    r"^libero_10:task([0-9]+):route_first_reliability_v1:replicate([0-9]+)$"
)


class Stage11DCollectionError(ValueError):
    """Raised when collection identity, artifacts, or geometry drift."""


@dataclass(frozen=True)
class Stage11DDevelopmentCall:
    task_id: int
    replicate_id: int
    cluster_key: str
    policy_seed: int
    call_ordinal: int
    step_id: int
    behavior_exit_layer: int
    cache_directory: Path
    array_path: str
    source_manifest_line: int


def development_schedule(
    schedule: Sequence[Stage11DRecord] | None = None,
) -> tuple[Stage11DRecord, ...]:
    """Return only the preregistered development-train clusters."""

    source = tuple(build_stage11d_schedule() if schedule is None else schedule)
    if len(source) == len(STAGE11D_TASK_IDS) * STAGE11D_CLUSTERS_PER_TASK:
        selected = tuple(
            record for record in source if record.split == "development_train"
        )
    elif len(source) == STAGE11D_COLLECTION_CLUSTER_COUNT:
        selected = source
    else:
        raise Stage11DCollectionError("Stage-11D schedule row count differs")
    expected = tuple(
        (task_id, replicate_id)
        for task_id in STAGE11D_TASK_IDS
        for replicate_id in STAGE11D_TRAIN_REPLICATES
    )
    observed = tuple((record.task_id, record.replicate_id) for record in selected)
    if (
        len(selected) != STAGE11D_COLLECTION_CLUSTER_COUNT
        or observed != expected
        or any(record.split != "development_train" for record in selected)
    ):
        raise Stage11DCollectionError("Stage-11D development schedule differs")
    return selected


def task_development_schedule(
    schedule: Sequence[Stage11DRecord], task_id: int
) -> tuple[Stage11DRecord, ...]:
    if type(task_id) is not int or task_id not in STAGE11D_TASK_IDS:
        raise Stage11DCollectionError("Stage-11D task id must be in 0..9")
    selected = tuple(
        record
        for record in development_schedule(schedule)
        if record.task_id == task_id
    )
    if tuple(record.replicate_id for record in selected) != STAGE11D_TRAIN_REPLICATES:
        raise Stage11DCollectionError(
            "Stage-11D task schedule must contain development replicates 0..11"
        )
    return selected


def parse_development_cluster_key(value: object) -> tuple[int, int]:
    match = _CLUSTER_KEY.fullmatch(str(value))
    if match is None:
        raise Stage11DCollectionError("Stage-11D cluster identity is not canonical")
    task_id, replicate_id = map(int, match.groups())
    if (
        task_id not in STAGE11D_TASK_IDS
        or replicate_id not in STAGE11D_TRAIN_REPLICATES
    ):
        raise Stage11DCollectionError(
            "Stage-11D collection identity is outside development_train"
        )
    return task_id, replicate_id


def validate_episode_id_override(
    value: object, *, task_id: int, replicate_id: int
) -> str:
    observed_task, observed_replicate = parse_development_cluster_key(value)
    if observed_task != task_id or observed_replicate != replicate_id:
        raise Stage11DCollectionError("Stage-11D episode identity differs")
    result = str(value)
    if ":episode" in result:
        raise Stage11DCollectionError("Generated-state identity aliases official LIBERO")
    return result


class Stage11DDevelopmentTaskSuite:
    """Proxy LIBERO while exposing only replicates 0..11 for each task."""

    def __init__(
        self,
        base_suite: Any,
        states_by_task: Mapping[int, Sequence[np.ndarray]],
    ) -> None:
        if set(states_by_task) != set(STAGE11D_TASK_IDS):
            raise Stage11DCollectionError("Stage-11D state task coverage differs")
        copied: dict[int, tuple[np.ndarray, ...]] = {}
        for task_id in STAGE11D_TASK_IDS:
            selected = tuple(states_by_task[task_id])
            if len(selected) != STAGE11D_COLLECTION_CLUSTERS_PER_TASK:
                raise Stage11DCollectionError(
                    "Stage-11D suite requires exactly 12 development states"
                )
            checked = tuple(
                canonical_state_bytes(state)[0].copy() for state in selected
            )
            copied[task_id] = checked
        self._base_suite = base_suite
        self._states_by_task = copied

    def get_task_init_states(self, task_id: int) -> list[np.ndarray]:
        if type(task_id) is not int or task_id not in STAGE11D_TASK_IDS:
            raise Stage11DCollectionError("Stage-11D task id must be in 0..9")
        return [state.copy() for state in self._states_by_task[task_id]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_suite, name)


def load_development_states(
    repo_root: str | Path,
) -> tuple[tuple[Stage11DRecord, ...], dict[int, tuple[np.ndarray, ...]], dict[str, Any]]:
    """Authenticate the frozen payload, then expose only development states."""

    schedule, all_states, attestation = load_bound_stage11d_states(repo_root)
    selected_schedule = development_schedule(schedule)
    selected_states = {
        task_id: tuple(
            state.copy()
            for state in all_states[task_id][:STAGE11D_COLLECTION_CLUSTERS_PER_TASK]
        )
        for task_id in STAGE11D_TASK_IDS
    }
    return selected_schedule, selected_states, attestation


def _object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage11DCollectionError(f"{context} is unreadable") from error
    if not isinstance(value, Mapping):
        raise Stage11DCollectionError(f"{context} must be an object")
    return dict(value)


def _regular_file(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise Stage11DCollectionError(f"{context} contains a symlink")
    try:
        metadata = absolute.stat()
    except FileNotFoundError as error:
        raise Stage11DCollectionError(f"{context} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise Stage11DCollectionError(f"{context} must be a regular file")
    return absolute.resolve(strict=True)


def validate_checkpoint_inventory(
    repo_root: str | Path, readiness: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the checkpoint identity frozen by the readiness hash pass."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    expected_directory = (root / "model/libero_exit").resolve(strict=True)
    inventory = readiness.get("checkpoint_inventory", {})
    model = inventory.get("model.pt", {})
    path = _regular_file(expected_directory / "model.pt", context="A1 model.pt")
    metadata = path.stat()
    if (
        model.get("path") != "model/libero_exit/model.pt"
        or model.get("bytes") != STAGE11D_A1_CHECKPOINT_BYTES
        or model.get("sha256") != STAGE11D_A1_CHECKPOINT_SHA256
        or model.get("device") != metadata.st_dev
        or model.get("inode") != metadata.st_ino
        or model.get("mtime_ns") != metadata.st_mtime_ns
        or model.get("ctime_ns") != metadata.st_ctime_ns
    ):
        raise PermissionError("Stage-11D A1 model identity differs from readiness")
    small = {
        "config.yaml": STAGE11D_A1_CONFIG_SHA256,
        "exit_thresholds_libero_10_exp_1.0.json": STAGE11D_A1_THRESHOLDS_SHA256,
        "dataset_statistics.json": STAGE11D_A1_DATASET_STATISTICS_SHA256,
        "exit_action_delta_matrix_libero_10_fm_steps10.json": (
            STAGE11D_A1_ACTION_DELTA_SHA256
        ),
    }
    for name, expected_sha in small.items():
        item = inventory.get(name, {})
        candidate = _regular_file(expected_directory / name, context=f"A1 {name}")
        if (
            item.get("path") != f"model/libero_exit/{name}"
            or item.get("sha256") != expected_sha
            or sha256_file(candidate) != expected_sha
            or item.get("bytes") != candidate.stat().st_size
        ):
            raise PermissionError(f"Stage-11D A1 {name} identity differs")
    return dict(inventory)


def validate_collection_readiness(repo_root: str | Path) -> dict[str, Any]:
    """Validate the exact runner authorization without opening CUDA or LIBERO."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    path = root / STAGE11D_COLLECTION_READINESS_RELATIVE_PATH
    readiness = _object(path, context="Stage-11D collection readiness")
    files = readiness.get("runner_files", {})
    access = readiness.get("access_boundary", {})
    authorization = readiness.get("authorization", {})
    if (
        readiness.get("schema_version") != STAGE11D_COLLECTION_READINESS_SCHEMA
        or readiness.get("status") != STAGE11D_COLLECTION_READINESS_STATUS
        or readiness.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
        or readiness.get("state_binding_sha256") != STAGE11D_STATE_BINDING_SHA256
        or readiness.get("schedule", {}).get("development_clusters")
        != STAGE11D_COLLECTION_CLUSTER_COUNT
        or access.get("development_train") is not True
        or access.get("calibration") is not False
        or access.get("shadow_confirmation") is not False
        or authorization.get("original_A1_observation_collection") is not True
        or authorization.get("same_noise_replay") is not False
        or authorization.get("training") is not False
        or authorization.get("active_control") is not False
        or not isinstance(files, Mapping)
        or not files
    ):
        raise PermissionError("Stage-11D collection readiness semantics differ")
    for relative, expected_sha in files.items():
        if (
            type(relative) is not str
            or type(expected_sha) is not str
            or sha256_file(root / relative) != expected_sha
        ):
            raise PermissionError("Stage-11D collection runner file hash differs")
    validate_checkpoint_inventory(root, readiness)
    return readiness


def validate_gpu_contract(
    *,
    physical_gpu_index: int,
    visible_devices: str | None,
    visible_gpu_count: int,
    expected_gpu_uuid: str,
    observed_gpu_uuid: str,
) -> None:
    if physical_gpu_index not in STAGE11D_COLLECTION_ALLOWED_PHYSICAL_GPUS:
        raise PermissionError("Stage-11D physical GPU must be in 0..7")
    if visible_devices != str(physical_gpu_index) or visible_gpu_count != 1:
        raise PermissionError("Stage-11D collection requires one visible GPU")
    normalize = lambda value: str(value).strip().lower().removeprefix("gpu-")
    if not expected_gpu_uuid or normalize(expected_gpu_uuid) != normalize(
        observed_gpu_uuid
    ):
        raise PermissionError("Stage-11D visible GPU UUID differs")


def load_development_task_calls(
    task_output_directory: str | Path, *, task_id: int
) -> tuple[Stage11DDevelopmentCall, ...]:
    """Validate one raw manifest without opening its NPZ payloads."""

    if task_id not in STAGE11D_TASK_IDS:
        raise Stage11DCollectionError("Stage-11D task id must be in 0..9")
    output = Path(task_output_directory).resolve(strict=True)
    manifest = _regular_file(
        output / "observation_calls/manifest.jsonl",
        context="Stage-11D observation manifest",
    )
    counters: dict[int, int] = defaultdict(int)
    previous_step: dict[int, int] = {}
    replicate_order: list[int] = []
    rows: list[Stage11DDevelopmentCall] = []
    with manifest.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise Stage11DCollectionError("Stage-11D manifest has an empty line")
            source = json.loads(line)
            if not isinstance(source, Mapping):
                raise Stage11DCollectionError("Stage-11D manifest row must be an object")
            source_task, replicate_id = parse_development_cluster_key(
                source.get("episode_id")
            )
            if source_task != task_id or source.get("task_id") != task_id:
                raise Stage11DCollectionError("Stage-11D manifest task differs")
            if (
                source.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION
                or source.get("checkpoint_sha256")
                != STAGE11D_A1_CHECKPOINT_SHA256
                or source.get("teacher_kind") != "frozen_original_a1_observer"
                or not has_complete_candidate_fm_traces(source)
            ):
                raise Stage11DCollectionError("Stage-11D raw cache contract differs")
            step = source.get("step_id")
            if type(step) is not int or step < 0 or step <= previous_step.get(
                replicate_id, -1
            ):
                raise Stage11DCollectionError("Stage-11D call steps are not increasing")
            previous_step[replicate_id] = step
            if counters[replicate_id] == 0:
                replicate_order.append(replicate_id)
            ordinal = counters[replicate_id]
            counters[replicate_id] += 1
            rows.append(
                Stage11DDevelopmentCall(
                    task_id=task_id,
                    replicate_id=replicate_id,
                    cluster_key=str(source["episode_id"]),
                    policy_seed=(
                        94_260_830 + task_id * 10_000 + replicate_id
                    ),
                    call_ordinal=ordinal,
                    step_id=step,
                    behavior_exit_layer=int(source["teacher_exit_layer"]),
                    cache_directory=manifest.parent,
                    array_path=str(source["array_path"]),
                    source_manifest_line=line_number,
                )
            )
    if (
        tuple(replicate_order) != STAGE11D_TRAIN_REPLICATES
        or tuple(sorted(counters)) != STAGE11D_TRAIN_REPLICATES
        or any(counters[replicate] < 1 for replicate in STAGE11D_TRAIN_REPLICATES)
    ):
        raise Stage11DCollectionError(
            "Stage-11D manifest does not cover development replicates 0..11"
        )
    return tuple(rows)


def resolve_call_payload(call: Stage11DDevelopmentCall) -> Path:
    parse_development_cluster_key(call.cluster_key)
    root = call.cache_directory.resolve(strict=True)
    relative = Path(call.array_path)
    if (
        call.cache_directory.is_symlink()
        or not root.is_dir()
        or relative.is_absolute()
        or relative.suffix != ".npz"
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Stage11DCollectionError("Stage-11D call payload path is unsafe")
    path = _regular_file(root / relative, context="Stage-11D call payload")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage11DCollectionError("Stage-11D call payload escaped cache") from error
    return path


__all__ = [
    "STAGE11D_COLLECTION_ALLOWED_PHYSICAL_GPUS",
    "STAGE11D_COLLECTION_CLUSTER_COUNT",
    "STAGE11D_COLLECTION_CLUSTERS_PER_TASK",
    "STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH",
    "STAGE11D_COLLECTION_READINESS_RELATIVE_PATH",
    "STAGE11D_COLLECTION_READINESS_SCHEMA",
    "STAGE11D_COLLECTION_READINESS_STATUS",
    "STAGE11D_RAW_TASK_SCHEMA",
    "Stage11DCollectionError",
    "Stage11DDevelopmentCall",
    "Stage11DDevelopmentTaskSuite",
    "development_schedule",
    "load_development_states",
    "load_development_task_calls",
    "parse_development_cluster_key",
    "resolve_call_payload",
    "task_development_schedule",
    "validate_checkpoint_inventory",
    "validate_collection_readiness",
    "validate_episode_id_override",
    "validate_gpu_contract",
]
