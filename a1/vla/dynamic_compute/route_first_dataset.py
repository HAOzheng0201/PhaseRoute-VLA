"""Validation and aggregation for action-free route-first teacher data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .route_first_collection import ROUTE_FIRST_COLLECTION_SCHEMA_VERSION
from .route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    ROUTE_FIRST_FEATURE_GROUPS,
    ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
    ROUTE_FIRST_LAYERS,
)


ROUTE_FIRST_AGGREGATE_SCHEMA_VERSION = "phase-route-vla.route-first-dataset.v1"
_EPISODE_PATTERN = re.compile(r"^libero_10:task(?P<task>[0-9]+):episode(?P<episode>[0-9]+)$")


class RouteFirstDatasetError(ValueError):
    """Raised when a teacher shard or aggregate violates data lineage."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    episode_id: np.ndarray,
    task_id: np.ndarray,
    step_id: np.ndarray,
    call_ordinal: np.ndarray,
) -> str:
    identity_rows = [
        [
            str(episode_id[index]),
            int(task_id[index]),
            int(step_id[index]),
            int(call_ordinal[index]),
        ]
        for index in range(features.shape[0])
    ]
    digest = hashlib.sha256()
    digest.update(features.astype(np.float32, copy=False).tobytes(order="C"))
    digest.update(teacher_layer.astype(np.int16, copy=False).tobytes(order="C"))
    digest.update(
        json.dumps(identity_rows, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class RouteFirstTeacherShard:
    path: Path
    file_sha256: str
    payload_sha256: str
    features: np.ndarray
    teacher_layer: np.ndarray
    teacher_fallback: np.ndarray
    episode_id: np.ndarray
    task_id: np.ndarray
    episode_index: np.ndarray
    step_id: np.ndarray
    call_ordinal: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


def load_route_first_teacher_shard(path: str | Path) -> RouteFirstTeacherShard:
    """Load one collector NPZ with ``allow_pickle=False`` and strict hashes."""

    target = Path(path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise RouteFirstDatasetError(f"teacher shard is not a file: {target}")
    with np.load(target, allow_pickle=False) as arrays:
        required = {
            "schema_version",
            "feature_schema_version",
            "feature_dimension",
            "feature_group_names",
            "feature_group_widths",
            "control_influence",
            "payload_sha256",
            "features",
            "teacher_layer",
            "teacher_fallback",
            "episode_id",
            "task_id",
            "step_id",
            "call_ordinal",
        }
        if set(arrays.files) != required:
            raise RouteFirstDatasetError("teacher shard fields differ")
        if arrays["schema_version"].item() != ROUTE_FIRST_COLLECTION_SCHEMA_VERSION:
            raise RouteFirstDatasetError("teacher shard schema differs")
        if arrays["feature_schema_version"].item() != ROUTE_FIRST_FEATURE_SCHEMA_VERSION:
            raise RouteFirstDatasetError("teacher feature schema differs")
        if int(arrays["feature_dimension"].item()) != ROUTE_FIRST_FEATURE_DIMENSION:
            raise RouteFirstDatasetError("teacher feature dimension differs")
        if arrays["feature_group_names"].tolist() != list(
            ROUTE_FIRST_FEATURE_GROUPS
        ) or arrays["feature_group_widths"].tolist() != list(
            ROUTE_FIRST_FEATURE_GROUPS.values()
        ):
            raise RouteFirstDatasetError("teacher feature groups differ")
        if bool(arrays["control_influence"].item()):
            raise RouteFirstDatasetError("teacher shard claims control influence")
        values = {name: arrays[name].copy() for name in required if name not in {
            "schema_version",
            "feature_schema_version",
            "feature_dimension",
            "feature_group_names",
            "feature_group_widths",
            "control_influence",
            "payload_sha256",
        }}
        claimed_payload = str(arrays["payload_sha256"].item())

    features = np.asarray(values["features"], dtype=np.float32)
    teacher = np.asarray(values["teacher_layer"], dtype=np.int16)
    fallback = np.asarray(values["teacher_fallback"], dtype=np.bool_)
    episode_id = np.asarray(values["episode_id"]).astype(np.str_)
    task_id = np.asarray(values["task_id"], dtype=np.int16)
    step_id = np.asarray(values["step_id"], dtype=np.int32)
    call_ordinal = np.asarray(values["call_ordinal"], dtype=np.int32)
    rows = int(features.shape[0]) if features.ndim == 2 else -1
    if features.shape != (rows, ROUTE_FIRST_FEATURE_DIMENSION) or rows < 1:
        raise RouteFirstDatasetError("teacher feature matrix geometry differs")
    vectors = (teacher, fallback, episode_id, task_id, step_id, call_ordinal)
    if any(value.shape != (rows,) for value in vectors):
        raise RouteFirstDatasetError("teacher shard row arrays differ")
    if not np.isfinite(features).all():
        raise RouteFirstDatasetError("teacher shard features are non-finite")
    if not set(np.unique(teacher).tolist()).issubset(ROUTE_FIRST_LAYERS):
        raise RouteFirstDatasetError("teacher shard contains an invalid layer")
    if not np.array_equal(fallback, teacher == 27):
        raise RouteFirstDatasetError("teacher fallback semantics differ")
    if np.any(task_id < 0) or np.any(step_id < 0) or np.any(call_ordinal < 0):
        raise RouteFirstDatasetError("teacher metadata must be non-negative")

    episode_index = np.empty(rows, dtype=np.int16)
    for index, identity in enumerate(episode_id.tolist()):
        match = _EPISODE_PATTERN.fullmatch(str(identity))
        if match is None:
            raise RouteFirstDatasetError(f"invalid episode identity: {identity}")
        parsed_task = int(match.group("task"))
        parsed_episode = int(match.group("episode"))
        if parsed_task != int(task_id[index]):
            raise RouteFirstDatasetError("episode identity and task_id differ")
        if parsed_episode > np.iinfo(np.int16).max:
            raise RouteFirstDatasetError("episode index exceeds int16")
        episode_index[index] = parsed_episode
    identities = list(zip(episode_id.tolist(), call_ordinal.tolist()))
    if len(set(identities)) != rows:
        raise RouteFirstDatasetError("teacher shard has duplicate policy calls")
    for identity in np.unique(episode_id):
        ordinals = np.sort(call_ordinal[episode_id == identity]).tolist()
        if ordinals != list(range(len(ordinals))):
            raise RouteFirstDatasetError("policy calls are not canonical within episode")
    recomputed = _payload_sha256(
        features, teacher, episode_id, task_id, step_id, call_ordinal
    )
    if recomputed != claimed_payload:
        raise RouteFirstDatasetError("teacher payload SHA-256 differs")
    return RouteFirstTeacherShard(
        path=target,
        file_sha256=_sha256_file(target),
        payload_sha256=recomputed,
        features=features,
        teacher_layer=teacher,
        teacher_fallback=fallback,
        episode_id=episode_id,
        task_id=task_id,
        episode_index=episode_index,
        step_id=step_id,
        call_ordinal=call_ordinal,
    )


def aggregate_route_first_teacher_shards(
    paths: Sequence[str | Path],
    *,
    expected_task_ids: Sequence[int],
    expected_episode_indices: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Validate, sort, and aggregate an exact task-by-state collection grid."""

    if not paths:
        raise RouteFirstDatasetError("at least one teacher shard is required")
    tasks = tuple(int(value) for value in expected_task_ids)
    episodes = tuple(int(value) for value in expected_episode_indices)
    if not tasks or len(set(tasks)) != len(tasks) or any(value < 0 for value in tasks):
        raise RouteFirstDatasetError("expected task ids must be unique and non-negative")
    if not episodes or len(set(episodes)) != len(episodes) or any(
        value < 0 for value in episodes
    ):
        raise RouteFirstDatasetError(
            "expected episode indices must be unique and non-negative"
        )
    resolved = [Path(path).expanduser().resolve(strict=True) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RouteFirstDatasetError("teacher shard paths must be unique")
    shards = [load_route_first_teacher_shard(path) for path in resolved]
    fields = (
        "features",
        "teacher_layer",
        "teacher_fallback",
        "episode_id",
        "task_id",
        "episode_index",
        "step_id",
        "call_ordinal",
    )
    combined = {
        name: np.concatenate([getattr(shard, name) for shard in shards], axis=0)
        for name in fields
    }
    rows = int(combined["features"].shape[0])
    identities = list(
        zip(combined["episode_id"].tolist(), combined["call_ordinal"].tolist())
    )
    if len(set(identities)) != rows:
        raise RouteFirstDatasetError("aggregate contains duplicate policy calls")
    expected_grid = {(task, episode) for task in tasks for episode in episodes}
    actual_grid = set(
        zip(combined["task_id"].tolist(), combined["episode_index"].tolist())
    )
    if actual_grid != expected_grid:
        missing = sorted(expected_grid - actual_grid)
        extra = sorted(actual_grid - expected_grid)
        raise RouteFirstDatasetError(
            f"teacher episode grid differs; missing={missing}, extra={extra}"
        )
    order = np.lexsort(
        (
            combined["call_ordinal"],
            combined["episode_index"],
            combined["task_id"],
        )
    )
    combined = {name: value[order] for name, value in combined.items()}
    counts = Counter(int(value) for value in combined["teacher_layer"])
    episode_call_counts = Counter(combined["episode_id"].tolist())
    summary: dict[str, object] = {
        "schema_version": ROUTE_FIRST_AGGREGATE_SCHEMA_VERSION,
        "feature_schema_version": ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
        "feature_dimension": ROUTE_FIRST_FEATURE_DIMENSION,
        "control_influence": False,
        "rows": rows,
        "episodes": len(actual_grid),
        "task_ids": list(tasks),
        "episode_indices": list(episodes),
        "teacher_layer_counts": {
            str(layer): int(counts.get(layer, 0)) for layer in ROUTE_FIRST_LAYERS
        },
        "policy_calls_per_episode": {
            "min": int(min(episode_call_counts.values())),
            "max": int(max(episode_call_counts.values())),
            "mean": float(rows / len(episode_call_counts)),
        },
        "sources": [
            {
                "path": str(shard.path),
                "rows": shard.rows,
                "payload_sha256": shard.payload_sha256,
                "file_sha256": shard.file_sha256,
            }
            for shard in shards
        ],
    }
    return combined, summary


def save_route_first_teacher_aggregate(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    summary: dict[str, object],
) -> dict[str, object]:
    """Publish a validated aggregate with exclusive-create semantics."""

    target = Path(path).expanduser().resolve()
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    features = arrays["features"].astype(np.float32, copy=False)
    teacher = arrays["teacher_layer"].astype(np.int16, copy=False)
    payload_sha = _payload_sha256(
        features,
        teacher,
        arrays["episode_id"],
        arrays["task_id"],
        arrays["step_id"],
        arrays["call_ordinal"],
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as output_file:
            np.savez_compressed(
                output_file,
                schema_version=np.asarray(ROUTE_FIRST_AGGREGATE_SCHEMA_VERSION),
                feature_schema_version=np.asarray(
                    ROUTE_FIRST_FEATURE_SCHEMA_VERSION
                ),
                feature_dimension=np.asarray(
                    ROUTE_FIRST_FEATURE_DIMENSION, dtype=np.int32
                ),
                feature_group_names=np.asarray(
                    tuple(ROUTE_FIRST_FEATURE_GROUPS), dtype=np.str_
                ),
                feature_group_widths=np.asarray(
                    tuple(ROUTE_FIRST_FEATURE_GROUPS.values()), dtype=np.int16
                ),
                control_influence=np.asarray(False, dtype=np.bool_),
                payload_sha256=np.asarray(payload_sha),
                source_file_sha256=np.asarray(
                    [source["file_sha256"] for source in summary["sources"]]
                ),
                **arrays,
            )
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    result = dict(summary)
    result.update(
        {
            "path": str(target),
            "payload_sha256": payload_sha,
            "file_sha256": _sha256_file(target),
        }
    )
    return result


__all__ = [
    "ROUTE_FIRST_AGGREGATE_SCHEMA_VERSION",
    "RouteFirstDatasetError",
    "RouteFirstTeacherShard",
    "aggregate_route_first_teacher_shards",
    "load_route_first_teacher_shard",
    "save_route_first_teacher_aggregate",
]
