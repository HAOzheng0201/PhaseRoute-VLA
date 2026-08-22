"""Fail-closed V3-D2 development collection and tensor contracts.

The D2 pipeline is deliberately split into three auditable steps:

1. collect raw policy-call caches for the frozen ``development_v2`` grid;
2. build past-only CPU context tensors;
3. replay layers 11/13/27 with one shared flow-matching input.

Only the first two candidate actions are runtime inputs.  Layer 27 is exposed
solely to the offline target builder and can never enter the 97-D feature
builder.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np
import torch

from a1.vla.dynamic_compute.temporal_route_features import right_aligned_history
from a1.vla.dynamic_compute.vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)

from .gripper_v2_protocol import (
    ACTION_DIMENSION,
    BASE_FEATURE_DIMENSION,
    DECISION_LAYERS,
    DEVELOPMENT_EPISODES,
    FEATURE_DIMENSION,
    GRIPPER_INDEX,
    HORIZON,
    RUNTIME_CONTEXT_NAMES,
    TEACHER_LAYER,
    canonical_json_sha256,
    decode_json_bytes,
    load_protocol,
    sha256_file,
    validate_selection_document,
)


D2_SCHEMA_VERSION = "phase-route-vla.v3.d2-development-collection.v1"
D2_STATUS = "D2_DEVELOPMENT_COLLECTION_CONTRACT_FROZEN"
D2_ROLE = "development_v2"
D2_SUITE = "libero_10"
D2_SELECTION_RELATIVE_PATH = Path(
    "configs/research/v3/data_lineage/development_v2.json"
)
D2_SELECTION_SHA256 = (
    "59af8441d4207b23e4ade2dff5b987d70490e9f6ab7aff50b97255e0292436eb"
)
D2_PROTOCOL_RELATIVE_PATH = Path("configs/research/v3/gripper_v2/protocol.json")
D2_PROTOCOL_SHA256 = (
    "3a5f5ebe49ddee093dc352ab4d46f7bbfea66486bc94d12d925d4eb40d2eaad2"
)
D2_COLLECTION_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/gripper_v2/d2_collection_contract.json"
)
D2_COLLECTION_CONTRACT_FILE_SHA256 = (
    "e43b82b9a0dfe2c45f7907aec40f45ec1d02a8d8a9fa7f3f4bd1c59f270a77c3"
)
D2_CHECKPOINT_SHA256 = (
    "dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
)
D2_CHECKPOINT_CONFIG_SHA256 = (
    "9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca"
)
D2_ACTION_DELTA_SHA256 = (
    "a0d0399b630953a9e0ef3b4ca09fe8a0fbde4b1ce6539ad5d911ad23fb6c812d"
)
D2_EXIT_THRESHOLDS_SHA256 = (
    "a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796"
)
D2_DATASET_STATISTICS_SHA256 = (
    "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3"
)
D2_MODEL_ATTESTATION_SHA256 = (
    "eed935ece00719016b5ea8e49b70b53ff1f99d36c51fb8a79d9d67ffcc0a1eab"
)
D2_PHASE_CHECKPOINT_SHA256 = (
    "b601f8221d47136818d7a008eaf7cee06e1201bf514f371ae33f42cfb39515a1"
)
D2_SEED_BASE = 20260811
D2_FM_STEPS = 10
D2_HISTORY_LENGTH = 8
D2_TASK_IDS = tuple(range(10))
D2_REPLAY_LAYERS = (*DECISION_LAYERS, TEACHER_LAYER)
D2_ALLOWED_PHYSICAL_GPUS = (0, 1, 2, 3)

D2_CONTEXT_SCHEMA_VERSION = "phase-route-vla.v3.d2-context.v1"
D2_CANDIDATE_SCHEMA_VERSION = "phase-route-vla.v3.d2-candidates.v1"
D2_DATASET_SCHEMA_VERSION = "phase-route-vla.v3.d2-gripper-dataset.v1"

D2_REPLAY_SOURCE_DTYPES = {
    "projected_features": np.float32,
    "image_input_idx": np.int64,
    "instruction_summary": np.float32,
    "normalized_proprio": np.float32,
    "input_ids": np.int64,
    "attention_mask": np.bool_,
    "attention_bias": np.float32,
    "response_mask": np.bool_,
    "subsegment_ids": np.int64,
    "position_ids": np.int64,
    "action_proprio": np.float32,
    "proprio_token_idx": np.int64,
    "teacher_exit_input_x": np.float32,
    "teacher_normalized_action": np.float32,
}

_EPISODE_ID = re.compile(r"^libero_10:task([0-9]+):episode([0-9]+)$")
_MAX_SELECTION_BYTES = 1024 * 1024


class D2ContractError(ValueError):
    """Base class for fail-closed D2 contract violations."""


class D2PathError(D2ContractError):
    """Raised when a source path is unsafe or escapes its bound directory."""


class D2ArtifactError(D2ContractError):
    """Raised when a raw or derived artifact violates its schema."""


@dataclass(frozen=True)
class DevelopmentEpisode:
    task_id: int
    episode_index: int
    seed: int

    @property
    def group_key(self) -> str:
        return f"{D2_SUITE}:task{self.task_id}:episode{self.episode_index}"


@dataclass(frozen=True)
class DevelopmentCall:
    dataset_index: int
    task_id: int
    episode_index: int
    call_ordinal: int
    step_id: int
    behavior_exit_layer: int
    cache_directory: Path
    array_path: str
    source_manifest_line: int

    @property
    def group_key(self) -> str:
        return f"{D2_SUITE}:task{self.task_id}:episode{self.episode_index}"


@dataclass(frozen=True)
class PastOnlyWindow:
    proprio_history: np.ndarray
    action_history: np.ndarray
    history_mask: np.ndarray


@dataclass(frozen=True)
class GripperV2Targets:
    candidate_state: torch.Tensor
    teacher_state: torch.Tensor
    candidate_transition: torch.Tensor
    teacher_transition: torch.Tensor
    step_mismatch_bits: torch.Tensor
    transition_mismatch_bits: torch.Tensor
    occurrence: torch.Tensor
    count: torch.Tensor
    first_transition_mismatch: torch.Tensor

    def validate(self, *, rows: int) -> None:
        expected = {
            "candidate_state": (rows, 2, 8),
            "teacher_state": (rows, 8),
            "candidate_transition": (rows, 2, 7),
            "teacher_transition": (rows, 7),
            "step_mismatch_bits": (rows, 2, 8),
            "transition_mismatch_bits": (rows, 2, 7),
            "occurrence": (rows, 2, 2),
            "count": (rows, 2, 2),
            "first_transition_mismatch": (rows, 2),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise D2ArtifactError(f"D2 target {name} shape differs")
            if value.dtype not in (torch.bool, torch.int64):
                raise D2ArtifactError(f"D2 target {name} dtype differs")
        if bool((self.count[..., 0] < 0).any()) or bool(
            (self.count[..., 0] > 8).any()
        ):
            raise D2ArtifactError("D2 step count is outside 0..8")
        if bool((self.count[..., 1] < 0).any()) or bool(
            (self.count[..., 1] > 7).any()
        ):
            raise D2ArtifactError("D2 transition count is outside 0..7")
        if bool((self.first_transition_mismatch < 0).any()) or bool(
            (self.first_transition_mismatch > 7).any()
        ):
            raise D2ArtifactError("D2 transition timing is outside 0..7")


def expected_seed(task_id: int, episode_index: int) -> int:
    if type(task_id) is not int or task_id not in D2_TASK_IDS:
        raise D2ContractError("D2 task id must be in 0..9")
    if type(episode_index) is not int or episode_index not in DEVELOPMENT_EPISODES:
        raise D2ContractError("D2 episode index must be in 12..29")
    return D2_SEED_BASE + task_id * 10_000 + episode_index


def _regular_file(path: Path, *, context: str, maximum: int | None = None) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise D2PathError(f"{context} contains a symlink component")
    try:
        metadata = absolute.stat()
    except FileNotFoundError as error:
        raise D2PathError(f"{context} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise D2PathError(f"{context} must be a regular file")
    if maximum is not None and metadata.st_size > maximum:
        raise D2PathError(f"{context} exceeds its size limit")
    return absolute.resolve(strict=True)


def stream_sha256(path: str | Path) -> str:
    target = _regular_file(Path(path), context="hashed file")
    digest = hashlib.sha256()
    with target.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_model_directory(
    model_directory: str | Path,
    model_attestation_path: str | Path,
) -> dict[str, Any]:
    """Validate the local hard-linked checkpoint without rehashing 33.8 GB.

    The previously frozen attestation binds SHA-256 to device/inode/size.  A
    local hard link must share that exact identity, while all writable
    sidecars live outside the frozen ``source`` tree.
    """

    directory = Path(model_directory).resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise D2PathError("D2 runtime model directory must be a regular directory")
    attestation_path = _regular_file(
        Path(model_attestation_path), context="D2 model attestation"
    )
    if stream_sha256(attestation_path) != D2_MODEL_ATTESTATION_SHA256:
        raise D2ContractError("D2 model attestation SHA-256 differs")
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise D2ArtifactError("D2 model attestation is invalid JSON") from error
    required = {
        "schema_version": "phase-route-vla.file-sha256-attestation.v1",
        "sha256": D2_CHECKPOINT_SHA256,
    }
    if any(attestation.get(name) != value for name, value in required.items()):
        raise D2ContractError("D2 model attestation content differs")
    model_path = _regular_file(directory / "model.pt", context="D2 runtime model")
    metadata = model_path.stat()
    identity = {
        "file_size_bytes": metadata.st_size,
        "file_inode": metadata.st_ino,
        "file_device": metadata.st_dev,
    }
    if any(attestation.get(name) != value for name, value in identity.items()):
        raise D2ContractError("D2 runtime model is not the attested hard link")
    expected_files = {
        "config.yaml": D2_CHECKPOINT_CONFIG_SHA256,
        "exit_action_delta_matrix_libero_10_fm_steps10.json": (
            D2_ACTION_DELTA_SHA256
        ),
        "exit_thresholds_libero_10_exp_1.0.json": D2_EXIT_THRESHOLDS_SHA256,
        "dataset_statistics.json": D2_DATASET_STATISTICS_SHA256,
    }
    observed = {
        name: stream_sha256(directory / name) for name in expected_files
    }
    if observed != expected_files:
        raise D2ContractError("D2 runtime model sidecar SHA-256 differs")
    return {
        "directory": str(directory),
        "model_path": str(model_path),
        "model_sha256": D2_CHECKPOINT_SHA256,
        "model_identity": identity,
        "attestation_path": str(attestation_path),
        "attestation_sha256": D2_MODEL_ATTESTATION_SHA256,
        "sidecar_sha256": observed,
    }


def load_development_selection(repo_root: str | Path) -> tuple[DevelopmentEpisode, ...]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D2_SELECTION_RELATIVE_PATH
    if sha256_file(path, maximum=_MAX_SELECTION_BYTES) != D2_SELECTION_SHA256:
        raise D2ContractError("D2 development selection SHA-256 differs")
    raw = _regular_file(
        path, context="D2 development selection", maximum=_MAX_SELECTION_BYTES
    ).read_bytes()
    value = decode_json_bytes(raw, context="D2 development selection")
    validate_selection_document(
        value,
        role=D2_ROLE,
        episodes=DEVELOPMENT_EPISODES,
        expected_count=len(D2_TASK_IDS) * len(DEVELOPMENT_EPISODES),
    )
    records = tuple(
        DevelopmentEpisode(
            task_id=int(record["task_id"]),
            episode_index=int(record["episode_index"]),
            seed=int(record["seed"]),
        )
        for record in value["records"]
    )
    if any(
        record.seed != expected_seed(record.task_id, record.episode_index)
        for record in records
    ):
        raise D2ContractError("D2 selection seed formula differs")
    expected_order = tuple(
        (task, episode)
        for task in D2_TASK_IDS
        for episode in DEVELOPMENT_EPISODES
    )
    if tuple((record.task_id, record.episode_index) for record in records) != (
        expected_order
    ):
        raise D2ContractError("D2 selection order differs")
    return records


def validate_frozen_d2_inputs(repo_root: str | Path) -> dict[str, Any]:
    """Validate D1, the D0 selection, and the pre-label D2 contract."""

    root = Path(repo_root).resolve(strict=True)
    protocol_path = root / D2_PROTOCOL_RELATIVE_PATH
    if sha256_file(protocol_path, maximum=1024 * 1024) != D2_PROTOCOL_SHA256:
        raise D2ContractError("D2 bound D1 protocol SHA-256 differs")
    protocol = load_protocol(protocol_path)
    selection = load_development_selection(root)
    contract_path = root / D2_COLLECTION_CONTRACT_RELATIVE_PATH
    if stream_sha256(contract_path) != D2_COLLECTION_CONTRACT_FILE_SHA256:
        raise D2ContractError("D2 collection contract file SHA-256 differs")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract != collection_contract():
        raise D2ContractError("D2 collection contract content differs")
    return {
        "d1_protocol_status": protocol["status"],
        "d1_protocol_sha256": D2_PROTOCOL_SHA256,
        "development_selection_sha256": D2_SELECTION_SHA256,
        "development_keys": len(selection),
        "d2_contract_file_sha256": D2_COLLECTION_CONTRACT_FILE_SHA256,
        "d2_contract_canonical_sha256": collection_contract_sha256(),
    }


def task_development_window(
    selection: Sequence[DevelopmentEpisode], task_id: int
) -> tuple[DevelopmentEpisode, ...]:
    if type(task_id) is not int or task_id not in D2_TASK_IDS:
        raise D2ContractError("D2 task id must be in 0..9")
    selected = tuple(record for record in selection if record.task_id == task_id)
    if tuple(record.episode_index for record in selected) != DEVELOPMENT_EPISODES:
        raise D2ContractError("D2 task window must be exactly episodes 12..29")
    return selected


class InitialStateWindowTaskSuite:
    """Expose one contiguous state window while proxying the LIBERO suite."""

    def __init__(self, base_suite: Any, start_index: int, count: int) -> None:
        if type(start_index) is not int or start_index != DEVELOPMENT_EPISODES[0]:
            raise D2ContractError("D2 initial-state window must start at episode 12")
        if type(count) is not int or count != len(DEVELOPMENT_EPISODES):
            raise D2ContractError("D2 initial-state window must contain 18 episodes")
        self._base_suite = base_suite
        self._start_index = start_index
        self._count = count

    def get_task_init_states(self, task_id: int):
        states = self._base_suite.get_task_init_states(task_id)
        stop = self._start_index + self._count
        if len(states) < stop:
            raise D2ContractError("LIBERO task has fewer than 30 initial states")
        return states[self._start_index:stop]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_suite, name)


def global_episode_index(local_episode_index: int) -> int:
    if type(local_episode_index) is not int or not 0 <= local_episode_index < len(
        DEVELOPMENT_EPISODES
    ):
        raise D2ContractError("D2 local episode index must be in 0..17")
    return DEVELOPMENT_EPISODES[0] + local_episode_index


def validate_gpu_contract(
    *,
    physical_gpu_index: int,
    visible_devices: str | None,
    visible_gpu_count: int,
    expected_gpu_uuid: str,
    observed_gpu_uuid: str,
) -> None:
    if physical_gpu_index not in D2_ALLOWED_PHYSICAL_GPUS:
        raise PermissionError("V3-D2 permits physical GPUs 0--3 only")
    if visible_devices != str(physical_gpu_index) or visible_gpu_count != 1:
        raise PermissionError("V3-D2 requires exactly one assigned visible GPU")
    normalize = lambda value: str(value).strip().lower().removeprefix("gpu-")
    if not expected_gpu_uuid or normalize(expected_gpu_uuid) != normalize(
        observed_gpu_uuid
    ):
        raise PermissionError("V3-D2 visible GPU UUID differs")


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _parse_episode_id(value: object) -> tuple[int, int]:
    match = _EPISODE_ID.fullmatch(str(value))
    if match is None:
        raise D2ArtifactError("D2 manifest episode ID is not canonical libero_10")
    task_id, episode_index = map(int, match.groups())
    return task_id, episode_index


def load_task_calls(
    task_output_directory: str | Path,
    *,
    task_id: int,
    dataset_index_start: int = 0,
) -> tuple[DevelopmentCall, ...]:
    """Read one D2 raw manifest without opening any NPZ payload."""

    if task_id not in D2_TASK_IDS:
        raise D2ContractError("D2 task id must be in 0..9")
    if type(dataset_index_start) is not int or dataset_index_start < 0:
        raise D2ContractError("D2 dataset index start must be non-negative")
    output = Path(task_output_directory).resolve(strict=True)
    manifest = _regular_file(
        output / "teacher_calls" / "manifest.jsonl",
        context="D2 teacher manifest",
    )
    cache_directory = manifest.parent
    counters: dict[int, int] = defaultdict(int)
    previous_step: dict[int, int] = {}
    rows: list[DevelopmentCall] = []
    with manifest.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise D2ArtifactError("D2 manifest contains an empty line")
            source = json.loads(line)
            if not isinstance(source, dict):
                raise D2ArtifactError("D2 manifest row must be an object")
            source_task, episode = _parse_episode_id(source.get("episode_id"))
            if source_task != task_id or source.get("task_id") != task_id:
                raise D2ArtifactError("D2 manifest task identity differs")
            if episode not in DEVELOPMENT_EPISODES:
                raise PermissionError("D2 raw manifest contains a sealed episode")
            if source.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION:
                raise D2ArtifactError("D2 raw cache schema differs")
            if source.get("checkpoint_sha256") != D2_CHECKPOINT_SHA256:
                raise D2ArtifactError("D2 raw cache checkpoint differs")
            if source.get("teacher_kind") != "a1_early_exit":
                raise D2ArtifactError("D2 raw cache teacher kind differs")
            if not has_complete_candidate_fm_traces(source):
                raise D2ArtifactError("D2 raw cache FM trace is incomplete")
            step = source.get("step_id")
            if type(step) is not int or step < 0 or step <= previous_step.get(
                episode, -1
            ):
                raise D2ArtifactError("D2 episode steps are not strictly increasing")
            previous_step[episode] = step
            ordinal = counters[episode]
            counters[episode] += 1
            rows.append(
                DevelopmentCall(
                    dataset_index=dataset_index_start + len(rows),
                    task_id=task_id,
                    episode_index=episode,
                    call_ordinal=ordinal,
                    step_id=step,
                    behavior_exit_layer=int(source["teacher_exit_layer"]),
                    cache_directory=cache_directory,
                    array_path=str(source["array_path"]),
                    source_manifest_line=line_number,
                )
            )
    if tuple(sorted(counters)) != DEVELOPMENT_EPISODES or any(
        counters[episode] < 1 for episode in DEVELOPMENT_EPISODES
    ):
        raise D2ArtifactError("D2 task manifest does not cover all episodes 12..29")
    return tuple(rows)


def resolve_call_payload(call: DevelopmentCall) -> Path:
    if call.task_id not in D2_TASK_IDS or call.episode_index not in DEVELOPMENT_EPISODES:
        raise PermissionError("D2 cannot resolve a non-development call")
    root = call.cache_directory.resolve(strict=True)
    if call.cache_directory.is_symlink() or not root.is_dir():
        raise D2PathError("D2 cache directory must be a regular directory")
    relative = Path(call.array_path)
    if (
        relative.is_absolute()
        or relative.suffix != ".npz"
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise D2PathError("D2 cache payload path is unsafe")
    path = _regular_file(root / relative, context="D2 cache payload")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise D2PathError("D2 cache payload escapes its directory") from error
    return path


class PastOnlyHistory:
    """Maintain independent right-aligned histories for every task/episode."""

    def __init__(self) -> None:
        self._values: dict[
            tuple[int, int], deque[tuple[np.ndarray, np.ndarray]]
        ] = defaultdict(lambda: deque(maxlen=D2_HISTORY_LENGTH))
        self._next_ordinal: dict[tuple[int, int], int] = defaultdict(int)

    def window_then_commit(
        self,
        call: DevelopmentCall,
        current_proprio: Any,
        current_behavior_action: Any,
    ) -> PastOnlyWindow:
        key = (call.task_id, call.episode_index)
        if call.call_ordinal != self._next_ordinal[key]:
            raise D2ArtifactError("D2 calls are not canonical within episode")
        proprio = np.asarray(current_proprio, dtype=np.float32)
        action = np.asarray(current_behavior_action, dtype=np.float32)
        if proprio.shape != (8,) or action.shape != (8, 7):
            raise D2ArtifactError("D2 current proprio/action geometry differs")
        if not np.isfinite(proprio).all() or not np.isfinite(action).all():
            raise D2ArtifactError("D2 current proprio/action is non-finite")
        past_proprio, past_action, mask = right_aligned_history(
            list(self._values[key]),
            history_len=D2_HISTORY_LENGTH,
            proprio_dim=8,
            action_horizon=HORIZON,
            action_dim=ACTION_DIMENSION,
        )
        self._values[key].append((proprio.copy(), action.copy()))
        self._next_ordinal[key] += 1
        return PastOnlyWindow(past_proprio, past_action, mask)


def pool_visual_features(
    projected_features: Any, image_input_idx: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = np.asarray(projected_features, dtype=np.float32)
    positions = np.asarray(image_input_idx)
    if projected.shape != (5, 144, 3584) or positions.shape != (5, 144):
        raise D2ArtifactError("D2 projected visual geometry differs")
    if not np.isfinite(projected).all():
        raise D2ArtifactError("D2 projected visual features are non-finite")
    patch_mask = positions >= 0
    crop_mask = patch_mask.any(axis=1)
    if not bool(crop_mask.any()):
        raise D2ArtifactError("D2 source has no valid visual crop")
    denominator = np.maximum(patch_mask.sum(axis=1), 1).astype(np.float32)
    crop_summary = (
        (projected * patch_mask[:, :, None]).sum(axis=1) / denominator[:, None]
    )
    crop_summary[~crop_mask] = 0.0
    global_summary = projected[patch_mask].mean(axis=0)
    return (
        global_summary.astype(np.float32, copy=False),
        crop_summary.astype(np.float32, copy=False),
        crop_mask.astype(np.bool_, copy=False),
    )


def replay_batch(arrays: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Build the exact batch-size-one frozen-A1 replay inputs."""

    missing = [name for name in D2_REPLAY_SOURCE_DTYPES if name not in arrays]
    if missing:
        raise D2ArtifactError("D2 replay payload is missing: " + ", ".join(missing))
    converted = {
        name: np.asarray(arrays[name]).astype(dtype, copy=True)
        for name, dtype in D2_REPLAY_SOURCE_DTYPES.items()
    }
    fixed_shapes = {
        "projected_features": (5, 144, 3584),
        "image_input_idx": (5, 144),
        "instruction_summary": (3584,),
        "normalized_proprio": (8,),
        "teacher_exit_input_x": (8, 7),
        "teacher_normalized_action": (8, 7),
    }
    for name, shape in fixed_shapes.items():
        if converted[name].shape != shape:
            raise D2ArtifactError(f"D2 replay tensor {name} shape differs")
    for name, value in converted.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise D2ArtifactError(f"D2 replay tensor {name} is non-finite")
    batch_names = tuple(
        name
        for name in D2_REPLAY_SOURCE_DTYPES
        if name != "teacher_normalized_action"
    )
    return {
        name: torch.from_numpy(converted[name]).unsqueeze(0)
        for name in batch_names
    }


def validate_runtime_context(
    runtime_inputs: Mapping[str, torch.Tensor], *, rows: int
) -> None:
    if tuple(runtime_inputs) != RUNTIME_CONTEXT_NAMES:
        raise D2ArtifactError("D2 runtime context order or names differ")
    shapes = {
        "instruction_summary": (rows, 3584),
        "vision_crop_summary": (rows, 5, 3584),
        "vision_crop_mask": (rows, 5),
        "phase_embedding": (rows, 128),
        "phase_scalars": (rows, 3),
        "normalized_proprio": (rows, 8),
        "proprio_history": (rows, 8, 8),
        "action_history": (rows, 8, 8, 7),
        "history_mask": (rows, 8),
    }
    for name, shape in shapes.items():
        value = runtime_inputs[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise D2ArtifactError(f"D2 runtime tensor {name} shape differs")
        if name.endswith("mask"):
            if value.dtype != torch.bool:
                raise D2ArtifactError(f"D2 runtime tensor {name} must be bool")
        elif value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise D2ArtifactError(
                f"D2 runtime tensor {name} must be finite float32"
            )


def _masked_latest(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = values.shape[0]
    indices = torch.arange(mask.shape[1], device=mask.device)[None].expand(rows, -1)
    latest = torch.where(mask, indices, torch.full_like(indices, -1)).amax(dim=1)
    gathered = values[torch.arange(rows, device=values.device), latest.clamp_min(0)]
    present = latest >= 0
    shape = (rows,) + (1,) * (gathered.ndim - 1)
    return torch.where(present.reshape(shape), gathered, torch.zeros_like(gathered))


def build_legacy_82d_features(
    runtime_inputs: Mapping[str, torch.Tensor], current_candidate_action: torch.Tensor
) -> torch.Tensor:
    """Reproduce the frozen C3.54 causal 82-D context without layer leakage."""

    if not isinstance(current_candidate_action, torch.Tensor) or (
        current_candidate_action.dtype != torch.float32
        or current_candidate_action.ndim != 3
        or current_candidate_action.shape[1:] != (HORIZON, ACTION_DIMENSION)
        or not bool(torch.isfinite(current_candidate_action).all())
    ):
        raise D2ArtifactError("D2 current candidate action must be finite [B,8,7]")
    rows = int(current_candidate_action.shape[0])
    validate_runtime_context(runtime_inputs, rows=rows)
    phase_embedding = runtime_inputs["phase_embedding"]
    phase_scalars = runtime_inputs["phase_scalars"]
    proprio = runtime_inputs["normalized_proprio"]
    proprio_history = runtime_inputs["proprio_history"]
    action_history = runtime_inputs["action_history"]
    history_mask = runtime_inputs["history_mask"]
    vision = runtime_inputs["vision_crop_summary"]
    vision_mask = runtime_inputs["vision_crop_mask"]
    anchor = current_candidate_action

    previous_proprio = _masked_latest(proprio_history, history_mask)
    previous_chunk = _masked_latest(action_history, history_mask)
    mask = history_mask.float()
    count = mask.sum(dim=1).clamp_min(1.0)
    first_actions = action_history[:, :, 0, :]
    history_mean = (first_actions * mask[:, :, None]).sum(dim=1) / count[:, None]
    history_variance = (
        (first_actions - history_mean[:, None, :]).square() * mask[:, :, None]
    ).sum(dim=1) / count[:, None]
    history_std = history_variance.sqrt()
    adjacent_mask = history_mask[:, 1:] & history_mask[:, :-1]
    adjacent_delta = first_actions[:, 1:] - first_actions[:, :-1]
    adjacent_count = adjacent_mask.float().sum(dim=1).clamp_min(1.0)
    history_temporal_rms = (
        (
            adjacent_delta.square().sum(dim=2) * adjacent_mask.float()
        ).sum(dim=1)
        / (adjacent_count * first_actions.shape[-1])
    ).sqrt()
    phase_stats = torch.stack(
        (
            phase_embedding.mean(dim=1),
            phase_embedding.std(dim=1, unbiased=False),
            phase_embedding.square().mean(dim=1).sqrt(),
            phase_embedding.abs().amax(dim=1),
        ),
        dim=1,
    )
    crop_weight = vision_mask.float()
    crop_count = crop_weight.sum(dim=1).clamp_min(1.0)
    pooled_vision = (vision * crop_weight[:, :, None]).sum(dim=1) / crop_count[:, None]
    centered_vision = (vision - pooled_vision[:, None, :]) * crop_weight[:, :, None]
    vision_stats = torch.stack(
        (
            pooled_vision.mean(dim=1),
            pooled_vision.std(dim=1, unbiased=False),
            pooled_vision.square().mean(dim=1).sqrt(),
            (
                centered_vision.square().sum(dim=(1, 2))
                / (crop_count * vision.shape[2])
            ).sqrt(),
        ),
        dim=1,
    )
    scalar_context = torch.stack(
        (
            mask.mean(dim=1),
            (anchor[:, 1:] - anchor[:, :-1]).square().mean(dim=(1, 2)).sqrt(),
            anchor.square().mean(dim=(1, 2)).sqrt(),
            previous_chunk.square().mean(dim=(1, 2)).sqrt(),
            (anchor - previous_chunk).square().mean(dim=(1, 2)).sqrt(),
            history_temporal_rms,
        ),
        dim=1,
    )
    features = torch.cat(
        (
            phase_scalars,
            phase_stats,
            proprio,
            proprio - previous_proprio,
            previous_chunk[:, 0, :],
            anchor[:, 0, :],
            anchor[:, 0, :] - previous_chunk[:, 0, :],
            anchor.mean(dim=1),
            anchor.std(dim=1, unbiased=False),
            history_mean,
            history_std,
            scalar_context,
            vision_stats,
        ),
        dim=1,
    ).float()
    if features.shape != (rows, BASE_FEATURE_DIMENSION) or not bool(
        torch.isfinite(features).all()
    ):
        raise D2ArtifactError("D2 legacy causal feature geometry differs")
    return features.contiguous()


def build_gripper_v2_feature(
    runtime_inputs: Mapping[str, torch.Tensor],
    current_candidate_action: torch.Tensor,
) -> torch.Tensor:
    """Build one isolated ``[B,97]`` runtime candidate feature matrix."""

    if not isinstance(current_candidate_action, torch.Tensor) or (
        current_candidate_action.dtype != torch.float32
        or current_candidate_action.ndim != 3
        or current_candidate_action.shape[1:] != (HORIZON, ACTION_DIMENSION)
        or not bool(torch.isfinite(current_candidate_action).all())
    ):
        raise D2ArtifactError("D2 current candidate action must be finite [B,8,7]")
    base = build_legacy_82d_features(runtime_inputs, current_candidate_action)
    states = current_candidate_action[..., GRIPPER_INDEX] >= 0.0
    signs = torch.where(
        states,
        torch.ones_like(current_candidate_action[..., GRIPPER_INDEX]),
        -torch.ones_like(current_candidate_action[..., GRIPPER_INDEX]),
    )
    transitions = (states[:, 1:] != states[:, :-1]).float()
    features = torch.cat((base, signs, transitions), dim=1).float().contiguous()
    if features.shape != (
        current_candidate_action.shape[0],
        FEATURE_DIMENSION,
    ) or not bool(torch.isfinite(features).all()):
        raise D2ArtifactError("D2 Gripper-v2 feature geometry differs")
    return features


def build_gripper_v2_features(
    runtime_inputs: Mapping[str, torch.Tensor],
    current_candidate_actions: torch.Tensor,
) -> torch.Tensor:
    """Build ``[B,2,97]`` using only each isolated current candidate."""

    if not isinstance(current_candidate_actions, torch.Tensor) or (
        current_candidate_actions.dtype != torch.float32
        or current_candidate_actions.ndim != 4
        or current_candidate_actions.shape[1:] != (2, HORIZON, ACTION_DIMENSION)
        or not bool(torch.isfinite(current_candidate_actions).all())
    ):
        raise D2ArtifactError("D2 candidate actions must be finite [B,2,8,7]")
    layers = [
        build_gripper_v2_feature(
            runtime_inputs, current_candidate_actions[:, layer_index]
        )
        for layer_index, _layer in enumerate(DECISION_LAYERS)
    ]
    features = torch.stack(layers, dim=1).float().contiguous()
    if features.shape != (
        current_candidate_actions.shape[0],
        len(DECISION_LAYERS),
        FEATURE_DIMENSION,
    ) or not bool(torch.isfinite(features).all()):
        raise D2ArtifactError("D2 Gripper-v2 feature geometry differs")
    return features


def build_gripper_v2_targets(candidate_actions: torch.Tensor) -> GripperV2Targets:
    """Construct discrete labels from offline L11/L13/L27 same-noise actions."""

    if not isinstance(candidate_actions, torch.Tensor) or (
        candidate_actions.dtype != torch.float32
        or candidate_actions.ndim != 4
        or candidate_actions.shape[1:] != (3, HORIZON, ACTION_DIMENSION)
        or not bool(torch.isfinite(candidate_actions).all())
    ):
        raise D2ArtifactError("D2 replay actions must be finite [B,3,8,7]")
    states = candidate_actions[..., GRIPPER_INDEX] >= 0.0
    candidate_state = states[:, :2]
    teacher_state = states[:, 2]
    candidate_transition = candidate_state[:, :, 1:] != candidate_state[:, :, :-1]
    teacher_transition = teacher_state[:, 1:] != teacher_state[:, :-1]
    step_bits = candidate_state != teacher_state[:, None, :]
    transition_bits = candidate_transition != teacher_transition[:, None, :]
    step_count = step_bits.sum(dim=2).to(torch.int64)
    transition_count = transition_bits.sum(dim=2).to(torch.int64)
    count = torch.stack((step_count, transition_count), dim=2)
    occurrence = count > 0
    positions = torch.arange(
        1, HORIZON, dtype=torch.int64, device=candidate_actions.device
    ).view(1, 1, -1)
    sentinel = torch.full_like(positions, HORIZON)
    first = torch.where(transition_bits, positions, sentinel).amin(dim=2)
    first = torch.where(first == HORIZON, torch.zeros_like(first), first)
    result = GripperV2Targets(
        candidate_state=candidate_state.contiguous(),
        teacher_state=teacher_state.contiguous(),
        candidate_transition=candidate_transition.contiguous(),
        teacher_transition=teacher_transition.contiguous(),
        step_mismatch_bits=step_bits.contiguous(),
        transition_mismatch_bits=transition_bits.contiguous(),
        occurrence=occurrence.contiguous(),
        count=count.contiguous(),
        first_transition_mismatch=first.contiguous(),
    )
    result.validate(rows=int(candidate_actions.shape[0]))
    return result


def collection_contract() -> dict[str, Any]:
    """Return the label-independent D2 collection contract for attestation."""

    return {
        "schema_version": D2_SCHEMA_VERSION,
        "status": D2_STATUS,
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "selection": {
            "path": str(D2_SELECTION_RELATIVE_PATH),
            "sha256": D2_SELECTION_SHA256,
            "task_ids": list(D2_TASK_IDS),
            "episode_indices": list(DEVELOPMENT_EPISODES),
            "key_count": len(D2_TASK_IDS) * len(DEVELOPMENT_EPISODES),
            "seed_formula": "20260811 + task_id * 10000 + episode_index",
        },
        "inputs": {
            "d1_protocol_path": str(D2_PROTOCOL_RELATIVE_PATH),
            "d1_protocol_sha256": D2_PROTOCOL_SHA256,
            "checkpoint_sha256": D2_CHECKPOINT_SHA256,
            "checkpoint_config_sha256": D2_CHECKPOINT_CONFIG_SHA256,
            "action_delta_sha256": D2_ACTION_DELTA_SHA256,
            "exit_thresholds_sha256": D2_EXIT_THRESHOLDS_SHA256,
            "dataset_statistics_sha256": D2_DATASET_STATISTICS_SHA256,
            "model_attestation_sha256": D2_MODEL_ATTESTATION_SHA256,
            "phase_checkpoint_sha256": D2_PHASE_CHECKPOINT_SHA256,
        },
        "collection": {
            "fm_steps": D2_FM_STEPS,
            "physical_gpus_allowed": list(D2_ALLOWED_PHYSICAL_GPUS),
            "one_visible_gpu_per_process": True,
            "raw_rollout_window": [12, 30],
            "raw_cache_schema": VISION_TEACHER_CACHE_SCHEMA_VERSION,
        },
        "replay": {
            "layers": list(D2_REPLAY_LAYERS),
            "shared_fm_input_exact": True,
            "layer27_role": "offline_same_noise_consistency_label_only",
        },
        "features": {
            "runtime_context_names": list(RUNTIME_CONTEXT_NAMES),
            "legacy_dimension": BASE_FEATURE_DIMENSION,
            "gripper_sign_dimension": HORIZON,
            "transition_dimension": HORIZON - 1,
            "output_dimension": FEATURE_DIMENSION,
            "teacher_visible": False,
            "other_candidate_visible": False,
        },
        "targets": {
            "step_count_support": list(range(9)),
            "transition_count_support": list(range(8)),
            "first_transition_support": list(range(8)),
            "continuous_positive_magnitude": False,
        },
        "access_boundary": {
            "development_v2_payload": True,
            "calibration_v2_payload": False,
            "independent_test_v2_payload": False,
            "legacy_c361_row_payload": False,
            "runtime_control": False,
        },
    }


def collection_contract_sha256() -> str:
    return canonical_json_sha256(collection_contract())


__all__ = [
    name
    for name in globals()
    if name.startswith(("D2_", "Development", "PastOnly", "GripperV2"))
    or name
    in {
        "InitialStateWindowTaskSuite",
        "build_gripper_v2_feature",
        "build_gripper_v2_features",
        "build_gripper_v2_targets",
        "build_legacy_82d_features",
        "collection_contract",
        "collection_contract_sha256",
        "expected_seed",
        "global_episode_index",
        "load_development_selection",
        "load_task_calls",
        "pool_visual_features",
        "resolve_call_payload",
        "replay_batch",
        "sha256_array",
        "stream_sha256",
        "task_development_window",
        "validate_gpu_contract",
        "validate_frozen_d2_inputs",
        "validate_runtime_model_directory",
        "validate_runtime_context",
    }
]
