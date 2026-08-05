"""Build leakage-safe temporal tensors from enriched phase-cache calls."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .weak_labels import BoundaryLabelConfig, build_weak_labels


PHASE_DATASET_SCHEMA_VERSION = "phase-route-vla.phase-dataset.v1"
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}


@dataclass(frozen=True)
class PhaseDatasetConfig:
    history_len: int = 8
    split_seed: int = 20260801
    boundary_config: BoundaryLabelConfig = field(
        default_factory=lambda: BoundaryLabelConfig(
            weight_action_delta_increase=0.0,
            dilation_radius=2,
        )
    )

    def __post_init__(self) -> None:
        if self.history_len < 1:
            raise ValueError("history_len must be positive")


def _call_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return str(record["episode_id"]), int(record["step_id"])


def _require_array(
    record: Mapping[str, Any],
    name: str,
    *,
    ndim: int,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    if name not in record:
        raise KeyError(f"phase call is missing {name}")
    array = np.asarray(record[name], dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def assign_episode_splits(
    episode_task: Mapping[str, int],
    seed: int,
) -> Dict[str, int]:
    """Assign complete episodes to deterministic per-task splits."""

    by_task: Dict[int, List[str]] = {}
    for episode_id, task_id in episode_task.items():
        by_task.setdefault(int(task_id), []).append(str(episode_id))
    split_by_episode: Dict[str, int] = {}
    for task_id, episode_ids in sorted(by_task.items()):
        ordered = sorted(
            episode_ids,
            key=lambda episode_id: hashlib.sha256(
                f"{seed}:{task_id}:{episode_id}".encode()
            ).hexdigest(),
        )
        count = len(ordered)
        if count < 3:
            validation_count = 0
            test_count = 0
        else:
            validation_count = max(1, int(round(count * 0.15)))
            test_count = max(1, int(round(count * 0.15)))
            while validation_count + test_count >= count:
                if test_count > 1:
                    test_count -= 1
                elif validation_count > 1:
                    validation_count -= 1
                else:
                    break
        for index, episode_id in enumerate(ordered):
            if index < test_count:
                split = 2
            elif index < test_count + validation_count:
                split = 1
            else:
                split = 0
            split_by_episode[episode_id] = split
    return split_by_episode


def build_phase_dataset_arrays(
    phase_calls: Sequence[Mapping[str, Any]],
    telemetry_records: Sequence[Mapping[str, Any]],
    config: PhaseDatasetConfig | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Join cache/telemetry calls and construct right-aligned past windows.

    The current policy call is never included in an input history.  Windows
    are built independently per episode, so neither history nor label dilation
    can cross an episode boundary.
    """

    config = config or PhaseDatasetConfig()
    if not phase_calls or not telemetry_records:
        raise ValueError("phase calls and telemetry records must be non-empty")
    phase_by_key: Dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in phase_calls:
        key = _call_key(record)
        if key in phase_by_key:
            raise ValueError(f"duplicate phase call key {key}")
        phase_by_key[key] = record
    telemetry_by_key: Dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in telemetry_records:
        key = _call_key(record)
        if key in telemetry_by_key:
            raise ValueError(f"duplicate telemetry key {key}")
        telemetry_by_key[key] = record
    if set(phase_by_key) != set(telemetry_by_key):
        missing_phase = sorted(set(telemetry_by_key) - set(phase_by_key))
        missing_telemetry = sorted(set(phase_by_key) - set(telemetry_by_key))
        raise ValueError(
            "phase/telemetry keys differ: "
            f"missing_phase={missing_phase[:3]}, missing_telemetry={missing_telemetry[:3]}"
        )

    labels = build_weak_labels(
        list(telemetry_by_key.values()),
        config=config.boundary_config,
    )
    label_by_key = {
        (label.episode_id, label.environment_step_id): label for label in labels
    }
    ordered_keys = sorted(phase_by_key, key=lambda key: (key[0], key[1]))
    episode_ids = sorted({key[0] for key in ordered_keys})
    episode_to_index = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    episode_task: Dict[str, int] = {}
    instruction_hashes = sorted(
        {str(telemetry_by_key[key]["instruction_hash"]) for key in ordered_keys}
    )
    instruction_to_index = {
        instruction_hash: index for index, instruction_hash in enumerate(instruction_hashes)
    }

    first = phase_by_key[ordered_keys[0]]
    proprio_dim = _require_array(first, "normalized_proprio", ndim=1).shape[0]
    action_shape = _require_array(first, "normalized_action_chunk", ndim=2).shape
    visual_dim = _require_array(first, "visual_summary", ndim=1).shape[0]
    instruction_dim = _require_array(first, "instruction_summary", ndim=1).shape[0]
    if proprio_dim < 1 or min(action_shape) < 1 or visual_dim < 1 or instruction_dim < 1:
        raise ValueError("phase input dimensions must be positive")

    size = len(ordered_keys)
    history_len = config.history_len
    arrays: Dict[str, np.ndarray] = {
        "visual_summary": np.empty((size, visual_dim), dtype=np.float16),
        "instruction_summary": np.empty((size, instruction_dim), dtype=np.float16),
        "current_raw_proprio": np.empty((size, proprio_dim), dtype=np.float32),
        "current_proprio": np.empty((size, proprio_dim), dtype=np.float32),
        "proprio_history": np.zeros(
            (size, history_len, proprio_dim), dtype=np.float32
        ),
        "proprio_history_mask": np.zeros((size, history_len), dtype=np.bool_),
        "action_history": np.zeros(
            (size, history_len, *action_shape), dtype=np.float32
        ),
        "action_history_mask": np.zeros((size, history_len), dtype=np.bool_),
        "previous_executed_action": np.zeros(
            (size, action_shape[-1]), dtype=np.float32
        ),
        "previous_executed_action_mask": np.zeros((size,), dtype=np.bool_),
        "current_normalized_action_chunk": np.empty(
            (size, *action_shape), dtype=np.float32
        ),
        "progress_target": np.empty((size, 1), dtype=np.float32),
        "boundary_target": np.empty((size, 1), dtype=np.float32),
        "boundary_target_raw": np.empty((size, 1), dtype=np.float32),
        "episode_index": np.empty((size,), dtype=np.int32),
        "call_index": np.empty((size,), dtype=np.int32),
        "step_id": np.empty((size,), dtype=np.int32),
        "task_id": np.empty((size,), dtype=np.int32),
        "instruction_index": np.empty((size,), dtype=np.int32),
        "split": np.empty((size,), dtype=np.int8),
    }

    keys_by_episode: Dict[str, List[tuple[str, int]]] = {}
    for key in ordered_keys:
        keys_by_episode.setdefault(key[0], []).append(key)
    split_by_episode = assign_episode_splits(
        {
            episode_id: int(telemetry_by_key[keys[0]]["task_id"])
            for episode_id, keys in keys_by_episode.items()
        },
        seed=config.split_seed,
    )
    row_index_by_key = {key: index for index, key in enumerate(ordered_keys)}

    for episode_id, episode_keys in keys_by_episode.items():
        episode_keys.sort(key=lambda key: key[1])
        task_ids = {int(telemetry_by_key[key]["task_id"]) for key in episode_keys}
        if len(task_ids) != 1:
            raise ValueError(f"episode {episode_id} contains multiple task IDs")
        episode_task[episode_id] = next(iter(task_ids))
        for local_index, key in enumerate(episode_keys):
            row_index = row_index_by_key[key]
            phase = phase_by_key[key]
            telemetry = telemetry_by_key[key]
            label = label_by_key[key]
            raw_proprio = _require_array(
                phase, "raw_proprio", ndim=1, dtype=np.float32
            )
            normalized_proprio = _require_array(
                phase, "normalized_proprio", ndim=1, dtype=np.float32
            )
            normalized_action = _require_array(
                phase, "normalized_action_chunk", ndim=2, dtype=np.float32
            )
            visual_summary = _require_array(
                phase, "visual_summary", ndim=1, dtype=np.float16
            )
            instruction_summary = _require_array(
                phase, "instruction_summary", ndim=1, dtype=np.float16
            )
            if (
                raw_proprio.shape != (proprio_dim,)
                or normalized_proprio.shape != (proprio_dim,)
                or normalized_action.shape != action_shape
                or visual_summary.shape != (visual_dim,)
                or instruction_summary.shape != (instruction_dim,)
            ):
                raise ValueError(f"inconsistent phase input shape at {key}")

            arrays["visual_summary"][row_index] = visual_summary
            arrays["instruction_summary"][row_index] = instruction_summary
            arrays["current_raw_proprio"][row_index] = raw_proprio
            arrays["current_proprio"][row_index] = normalized_proprio
            arrays["current_normalized_action_chunk"][row_index] = normalized_action
            previous_action = np.asarray(
                phase.get("previous_action", np.empty((0,))), dtype=np.float32
            ).reshape(-1)
            if previous_action.size:
                if previous_action.shape != (action_shape[-1],):
                    raise ValueError(f"previous_action has an invalid shape at {key}")
                arrays["previous_executed_action"][row_index] = previous_action
                arrays["previous_executed_action_mask"][row_index] = True

            history_keys = episode_keys[max(0, local_index - history_len):local_index]
            history_start = history_len - len(history_keys)
            for offset, history_key in enumerate(history_keys, start=history_start):
                history_phase = phase_by_key[history_key]
                arrays["proprio_history"][row_index, offset] = _require_array(
                    history_phase,
                    "normalized_proprio",
                    ndim=1,
                    dtype=np.float32,
                )
                arrays["action_history"][row_index, offset] = _require_array(
                    history_phase,
                    "normalized_action_chunk",
                    ndim=2,
                    dtype=np.float32,
                )
                arrays["proprio_history_mask"][row_index, offset] = True
                arrays["action_history_mask"][row_index, offset] = True

            arrays["progress_target"][row_index, 0] = label.progress_target
            arrays["boundary_target"][row_index, 0] = label.boundary_target
            arrays["boundary_target_raw"][row_index, 0] = label.boundary_target_raw
            arrays["episode_index"][row_index] = episode_to_index[episode_id]
            arrays["call_index"][row_index] = local_index
            arrays["step_id"][row_index] = int(key[1])
            arrays["task_id"][row_index] = int(telemetry["task_id"])
            arrays["instruction_index"][row_index] = instruction_to_index[
                str(telemetry["instruction_hash"])
            ]
            arrays["split"][row_index] = split_by_episode[episode_id]

    split_records = {
        SPLIT_NAMES[split]: int((arrays["split"] == split).sum()) for split in SPLIT_NAMES
    }
    split_episodes = {
        SPLIT_NAMES[split]: sum(value == split for value in split_by_episode.values())
        for split in SPLIT_NAMES
    }
    metadata = {
        "schema_version": PHASE_DATASET_SCHEMA_VERSION,
        "records": size,
        "episodes": len(episode_ids),
        "history_len": history_len,
        "proprio_dim": proprio_dim,
        "action_shape": list(action_shape),
        "visual_summary_dim": visual_dim,
        "instruction_summary_dim": instruction_dim,
        "episode_ids": episode_ids,
        "episode_task": episode_task,
        "instruction_hashes": instruction_hashes,
        "split_seed": config.split_seed,
        "split_records": split_records,
        "split_episodes": split_episodes,
        "boundary_config": dict(config.boundary_config.__dict__),
        "raw_boundaries": int(arrays["boundary_target_raw"].sum()),
        "dilated_boundaries": int(arrays["boundary_target"].sum()),
        "estimator_inputs": [
            "visual_summary",
            "instruction_summary",
            "current_proprio",
            "proprio_history",
            "proprio_history_mask",
            "action_history",
            "action_history_mask",
        ],
        "analysis_only_not_estimator_inputs": [
            "current_raw_proprio",
            "previous_executed_action",
            "current_normalized_action_chunk",
            "progress_target",
            "boundary_target",
        ],
        "history_semantics": (
            "Right-aligned previous policy calls only; current call is excluded and "
            "windows never cross episode boundaries."
        ),
    }
    return arrays, metadata
