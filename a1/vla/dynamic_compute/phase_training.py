"""Validated dataset loading and metrics for M2 phase-estimator training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from .phase_dataset import PHASE_DATASET_SCHEMA_VERSION


ESTIMATOR_INPUT_NAMES = (
    "visual_summary",
    "instruction_summary",
    "current_proprio",
    "proprio_history",
    "proprio_history_mask",
    "action_history",
    "action_history_mask",
)
TARGET_NAMES = (
    "progress_target",
    "boundary_target",
    "episode_index",
    "call_index",
)
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class PhaseDatasetBundle:
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    dataset_path: Path
    dataset_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_shape(array: np.ndarray, name: str, ndim: int, rows: int) -> None:
    if array.ndim != ndim or array.shape[0] != rows:
        raise ValueError(
            f"{name} must have {ndim} dimensions and {rows} rows, got {array.shape}"
        )


def load_phase_dataset(
    dataset_path: str | Path,
    metadata_path: str | Path,
) -> PhaseDatasetBundle:
    """Load an immutable dataset and enforce the estimator input contract."""

    dataset_path = Path(dataset_path)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != PHASE_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unexpected phase dataset schema: {metadata.get('schema_version')}")
    digest = file_sha256(dataset_path)
    if metadata.get("dataset_sha256") != digest:
        raise ValueError("Phase dataset SHA-256 does not match metadata")
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    required = set(ESTIMATOR_INPUT_NAMES) | set(TARGET_NAMES) | {"split"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Phase dataset is missing arrays: {missing}")
    rows = arrays["split"].shape[0]
    if rows < 1 or int(metadata.get("records", -1)) != rows:
        raise ValueError("Phase dataset record count is inconsistent")

    _require_shape(arrays["visual_summary"], "visual_summary", 2, rows)
    _require_shape(arrays["instruction_summary"], "instruction_summary", 2, rows)
    _require_shape(arrays["current_proprio"], "current_proprio", 2, rows)
    _require_shape(arrays["proprio_history"], "proprio_history", 3, rows)
    _require_shape(arrays["proprio_history_mask"], "proprio_history_mask", 2, rows)
    _require_shape(arrays["action_history"], "action_history", 4, rows)
    _require_shape(arrays["action_history_mask"], "action_history_mask", 2, rows)
    _require_shape(arrays["progress_target"], "progress_target", 2, rows)
    _require_shape(arrays["boundary_target"], "boundary_target", 2, rows)
    _require_shape(arrays["episode_index"], "episode_index", 1, rows)
    _require_shape(arrays["call_index"], "call_index", 1, rows)
    _require_shape(arrays["split"], "split", 1, rows)

    history_shape = arrays["proprio_history"].shape[:2]
    if arrays["proprio_history_mask"].shape != history_shape:
        raise ValueError("Proprio history and its mask are misaligned")
    if arrays["action_history"].shape[:2] != history_shape:
        raise ValueError("Action and proprio histories are misaligned")
    if arrays["action_history_mask"].shape != history_shape:
        raise ValueError("Action history and its mask are misaligned")
    if not np.array_equal(
        arrays["proprio_history_mask"].astype(bool),
        arrays["action_history_mask"].astype(bool),
    ):
        raise ValueError("Proprio/action history masks differ")
    if arrays["progress_target"].shape[1:] != (1,):
        raise ValueError("progress_target must have shape [N, 1]")
    if arrays["boundary_target"].shape[1:] != (1,):
        raise ValueError("boundary_target must have shape [N, 1]")
    if not np.isin(arrays["split"], list(SPLIT_IDS.values())).all():
        raise ValueError("Unknown split ID in phase dataset")
    if not np.isin(arrays["boundary_target"], [0.0, 1.0]).all():
        raise ValueError("boundary_target must be binary")
    if not (
        (arrays["progress_target"] >= 0.0).all()
        and (arrays["progress_target"] <= 1.0).all()
    ):
        raise ValueError("progress_target must be in [0, 1]")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value")

    for episode_index in np.unique(arrays["episode_index"]):
        episode_rows = arrays["episode_index"] == episode_index
        if np.unique(arrays["split"][episode_rows]).size != 1:
            raise ValueError(f"Episode {episode_index} crosses dataset splits")
        ordered_calls = np.sort(arrays["call_index"][episode_rows])
        if not np.array_equal(ordered_calls, np.arange(ordered_calls.size)):
            raise ValueError(f"Episode {episode_index} has non-contiguous call indices")

    for split_name, split_id in SPLIT_IDS.items():
        if not (arrays["split"] == split_id).any():
            raise ValueError(f"Phase dataset split is empty: {split_name}")
    return PhaseDatasetBundle(
        arrays=arrays,
        metadata=metadata,
        dataset_path=dataset_path,
        dataset_sha256=digest,
    )


def make_torch_batch(
    bundle: PhaseDatasetBundle,
    split_name: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Materialize only the declared estimator inputs plus supervision."""

    if split_name not in SPLIT_IDS:
        raise ValueError(f"Unknown split: {split_name}")
    indices = np.flatnonzero(bundle.arrays["split"] == SPLIT_IDS[split_name])
    batch: Dict[str, torch.Tensor] = {}
    for name in ESTIMATOR_INPUT_NAMES:
        tensor = torch.from_numpy(bundle.arrays[name][indices])
        if name.endswith("_mask"):
            tensor = tensor.to(dtype=torch.bool)
        else:
            tensor = tensor.to(dtype=torch.float32)
        batch[name] = tensor.to(device)
    for name in TARGET_NAMES:
        tensor = torch.from_numpy(bundle.arrays[name][indices])
        dtype = torch.long if name in {"episode_index", "call_index"} else torch.float32
        batch[name] = tensor.to(device=device, dtype=dtype)
    return batch


def progress_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Progress prediction/target must be aligned and non-empty")
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def boundary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError("Boundary probability/target must be aligned and non-empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Boundary threshold must be in [0, 1]")
    prediction = probability >= threshold
    positive = target == 1
    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & ~positive))
    false_negative = int(np.sum(~prediction & positive))
    true_negative = int(np.sum(~prediction & ~positive))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    bce = -np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))
    return {
        "threshold": float(threshold),
        "accuracy": float((true_positive + true_negative) / target.size),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "bce": float(bce),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def select_f1_threshold(probability: np.ndarray, target: np.ndarray) -> float:
    """Choose a boundary threshold from validation data only."""

    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    candidates = np.unique(np.concatenate([probability, np.array([0.0, 0.5, 1.0])]))
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        metrics = boundary_metrics(probability, target, float(threshold))
        key = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def baseline_metrics(
    train_progress_target: np.ndarray,
    train_boundary_target: np.ndarray,
    progress_target: np.ndarray,
    boundary_target: np.ndarray,
    *,
    seed: int,
) -> Dict[str, Mapping[str, float | int]]:
    """Compute constant-progress, majority-class and seeded random baselines."""

    progress_constant = float(np.mean(train_progress_target))
    boundary_prevalence = float(np.mean(train_boundary_target))
    count = np.asarray(boundary_target).size
    rng = np.random.default_rng(seed)
    random_prediction = (rng.random(count) < boundary_prevalence).astype(np.float64)
    majority_prediction = np.full(
        count,
        1.0 if boundary_prevalence >= 0.5 else 0.0,
        dtype=np.float64,
    )
    return {
        "constant_progress": {
            "train_mean": progress_constant,
            **progress_metrics(
                np.full_like(progress_target, progress_constant, dtype=np.float64),
                progress_target,
            ),
        },
        "majority_boundary": {
            "train_prevalence": boundary_prevalence,
            **boundary_metrics(majority_prediction, boundary_target, 0.5),
        },
        "random_boundary": {
            "seed": seed,
            "train_prevalence": boundary_prevalence,
            **boundary_metrics(random_prediction, boundary_target, 0.5),
        },
    }
