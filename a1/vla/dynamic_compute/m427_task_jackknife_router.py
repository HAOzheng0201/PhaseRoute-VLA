"""Conservative task-jackknife route13/27 ensemble for M4.27.

This module is offline-only.  It aggregates ten causal layer-13 learners and
fails closed to layer 27 whenever their lower envelope does not clear a
strictly calibrated negative maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .risk_route13_router import RiskRoute13Model


M427_ENSEMBLE_SCHEMA_VERSION = "phase-route-vla.m427-task-jackknife-ensemble.v1"
M427_AGGREGATIONS = ("min", "mean")
M427_TASKS = tuple(range(10))
_MAX_STRICT_THRESHOLD = float(np.nextafter(1.0, math.inf))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_safe13_probabilities(
    learner_probabilities: np.ndarray, *, aggregation: str
) -> np.ndarray:
    probabilities = np.asarray(learner_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("learner probabilities must have shape [N, L]")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("learner probabilities must be finite and in [0, 1]")
    if aggregation == "min":
        return probabilities.min(axis=1)
    if aggregation == "mean":
        return probabilities.mean(axis=1)
    raise ValueError(f"unsupported M4.27 aggregation: {aggregation}")


def calibrate_strict_negative_max(
    scores: np.ndarray, teacher_route: np.ndarray
) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    teacher = np.asarray(teacher_route, dtype=np.int64).reshape(-1)
    if values.shape != teacher.shape or values.size < 1:
        raise ValueError("scores and teacher routes must be aligned and non-empty")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("safe13 scores must be finite and in [0, 1]")
    negative = teacher == 27
    if not negative.any():
        raise ValueError("calibration has no required27 negative")
    return float(np.nextafter(values[negative].max(), math.inf))


def strict_route13_or_27(scores: np.ndarray, *, threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("safe13 scores must be finite and in [0, 1]")
    if (
        not math.isfinite(threshold)
        or threshold <= 0.0
        or threshold > _MAX_STRICT_THRESHOLD
    ):
        raise ValueError("strict route threshold is outside the supported range")
    return np.where(values >= threshold, 13, 27).astype(np.int64)


def episode_group_risk_metrics(
    predicted: np.ndarray,
    teacher_route: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
) -> dict[str, int | float | list[dict[str, int]]]:
    routes = np.asarray(predicted, dtype=np.int64).reshape(-1)
    teacher = np.asarray(teacher_route, dtype=np.int64).reshape(-1)
    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if not (routes.shape == teacher.shape == tasks.shape == episodes.shape):
        raise ValueError("group-risk inputs must be aligned")
    if teacher.size < 1:
        raise ValueError("group-risk inputs must be non-empty")
    if not set(np.unique(routes).tolist()).issubset({13, 27}):
        raise ValueError("predicted routes must be 13 or 27")
    positive_groups: list[dict[str, int]] = []
    error_groups: list[dict[str, int]] = []
    for task, episode in sorted(set(zip(tasks.tolist(), episodes.tolist()))):
        mask = (tasks == task) & (episodes == episode)
        required = mask & (teacher == 27)
        required_rows = int(required.sum())
        if required_rows == 0:
            continue
        descriptor = {
            "task_id": int(task),
            "episode_index": int(episode),
            "required27_rows": required_rows,
        }
        positive_groups.append(descriptor)
        false_rows = int(np.sum(required & (routes == 13)))
        if false_rows:
            error_groups.append({**descriptor, "false_shallow_rows": false_rows})
    return {
        "route27_positive_groups": len(positive_groups),
        "route27_error_groups": len(error_groups),
        "route27_group_error_rate": float(
            len(error_groups) / max(len(positive_groups), 1)
        ),
        "positive_groups": positive_groups,
        "error_groups": error_groups,
    }


def zero_error_clopper_pearson_upper(
    groups: int, *, confidence: float = 0.95
) -> float:
    if groups < 1:
        raise ValueError("at least one risk group is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return float(1.0 - (1.0 - confidence) ** (1.0 / groups))


@dataclass(frozen=True)
class TaskJackknifeRoute13Ensemble:
    learners: tuple[RiskRoute13Model, ...]
    excluded_tasks: tuple[int, ...]
    aggregation: str
    threshold: float

    def __post_init__(self) -> None:
        if self.excluded_tasks != M427_TASKS or len(self.learners) != len(M427_TASKS):
            raise ValueError("M4.27 requires exactly one learner per excluded task")
        if self.aggregation not in M427_AGGREGATIONS:
            raise ValueError("unsupported M4.27 aggregation")
        if any(model.variant != "temporal_phase_step" for model in self.learners):
            raise ValueError("M4.27 learners must use temporal_phase_step")
        if (
            not math.isfinite(self.threshold)
            or self.threshold <= 0.0
            or self.threshold > _MAX_STRICT_THRESHOLD
        ):
            raise ValueError("invalid M4.27 ensemble threshold")

    def learner_probabilities(
        self, arrays: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        probabilities = np.stack(
            [model.probabilities(arrays) for model in self.learners], axis=1
        )
        if probabilities.shape[1] != 10:
            raise RuntimeError("M4.27 learner probability grid differs")
        return probabilities

    def scores(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return aggregate_safe13_probabilities(
            self.learner_probabilities(arrays), aggregation=self.aggregation
        )

    def routes(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return strict_route13_or_27(self.scores(arrays), threshold=self.threshold)

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        descriptors = []
        for excluded_task, learner in zip(self.excluded_tasks, self.learners):
            path = root / f"exclude_task{excluded_task}.npz"
            learner.save(path, excluded_task=excluded_task)
            descriptors.append(
                {
                    "excluded_task": excluded_task,
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            )
        descriptor_path = root / "ensemble.json"
        descriptor = {
            "schema_version": M427_ENSEMBLE_SCHEMA_VERSION,
            "aggregation": self.aggregation,
            "threshold": self.threshold,
            "learners": descriptors,
        }
        descriptor_path.write_text(
            json.dumps(descriptor, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return descriptor_path

    @classmethod
    def load(cls, descriptor_path: str | Path) -> "TaskJackknifeRoute13Ensemble":
        path = Path(descriptor_path)
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        if descriptor.get("schema_version") != M427_ENSEMBLE_SCHEMA_VERSION:
            raise ValueError("unexpected M4.27 ensemble schema")
        learner_descriptors: Sequence[Mapping[str, object]] = descriptor["learners"]
        excluded_tasks = tuple(
            int(item["excluded_task"]) for item in learner_descriptors
        )
        learners = []
        for item in learner_descriptors:
            learner_path = path.parent / str(item["path"])
            if sha256_file(learner_path) != str(item["sha256"]):
                raise ValueError("M4.27 learner SHA-256 differs")
            learners.append(RiskRoute13Model.load(learner_path))
        return cls(
            tuple(learners),
            excluded_tasks,
            str(descriptor["aggregation"]),
            float(descriptor["threshold"]),
        )
