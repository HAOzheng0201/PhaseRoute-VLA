"""Leakage-safe cross-validation for the action-free route-first router."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .route_first_router import (
    RouteFirstOrdinalRouter,
    fit_route_first_affine_head,
    fit_route_first_projection,
    route_first_group_weights,
)
from .route_first_features import ROUTE_FIRST_FEATURE_DIMENSION


ROUTE_FIRST_COVERAGES = (0.01, 0.025, 0.05, 0.1, 0.15)


@dataclass(frozen=True, order=True)
class RouteFirstCandidate:
    pca_rank: int
    l2: float

    def __post_init__(self) -> None:
        if self.pca_rank < 1 or not math.isfinite(self.l2) or self.l2 <= 0.0:
            raise ValueError("route-first candidate hyperparameters are invalid")

    @property
    def name(self) -> str:
        return f"pca{self.pca_rank}_l2_{self.l2:g}"


def _aligned_training_inputs(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    teacher = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if values.shape != (teacher.size, ROUTE_FIRST_FEATURE_DIMENSION):
        raise ValueError("route-first training feature geometry differs")
    if not (teacher.shape == tasks.shape == episodes.shape) or teacher.size < 2:
        raise ValueError("route-first training arrays are misaligned or empty")
    if (
        not np.isfinite(values).all()
        or np.any(tasks < 0)
        or np.any(episodes < 0)
        or not set(np.unique(teacher).tolist()).issubset({11, 13, 27})
        or 11 not in teacher
        or 27 not in teacher
    ):
        raise ValueError("route-first training values or binary targets are invalid")
    return values, teacher, tasks, episodes


def _aligned_binary_inputs(
    score: np.ndarray, label: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(label, dtype=np.int64).reshape(-1)
    weights = np.asarray(weight, dtype=np.float64).reshape(-1)
    if not (values.shape == target.shape == weights.shape) or values.size < 2:
        raise ValueError("binary metric inputs must be aligned and non-empty")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or np.any((values < 0.0) | (values > 1.0))
        or set(np.unique(target).tolist()) != {0, 1}
    ):
        raise ValueError("binary metric inputs are invalid")
    return values, target, weights


def weighted_average_precision(
    score: np.ndarray, label: np.ndarray, weight: np.ndarray
) -> float:
    values, target, weights = _aligned_binary_inputs(score, label, weight)
    order = np.argsort(-values, kind="mergesort")
    values, target, weights = values[order], target[order], weights[order]
    positive_total = float(weights[target == 1].sum())
    true_positive = 0.0
    false_positive = 0.0
    average_precision = 0.0
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[end] == values[index]:
            end += 1
        group_positive = float(weights[index:end][target[index:end] == 1].sum())
        group_negative = float(weights[index:end][target[index:end] == 0].sum())
        true_positive += group_positive
        false_positive += group_negative
        precision = true_positive / (true_positive + false_positive)
        average_precision += (group_positive / positive_total) * precision
        index = end
    return float(average_precision)


def weighted_roc_auc(
    score: np.ndarray, label: np.ndarray, weight: np.ndarray
) -> float:
    values, target, weights = _aligned_binary_inputs(score, label, weight)
    order = np.argsort(values, kind="mergesort")
    values, target, weights = values[order], target[order], weights[order]
    positive_total = float(weights[target == 1].sum())
    negative_total = float(weights[target == 0].sum())
    concordant = 0.0
    negative_before = 0.0
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[end] == values[index]:
            end += 1
        group_positive = float(weights[index:end][target[index:end] == 1].sum())
        group_negative = float(weights[index:end][target[index:end] == 0].sum())
        concordant += group_positive * (negative_before + 0.5 * group_negative)
        negative_before += group_negative
        index = end
    return float(concordant / (positive_total * negative_total))


def false_safe_at_coverage(
    score: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    *,
    coverage: float,
) -> dict[str, float | int]:
    values, target, weights = _aligned_binary_inputs(score, label, weight)
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0,1]")
    order = np.argsort(-values, kind="mergesort")
    target_mass = float(weights.sum()) * float(coverage)
    selected: list[int] = []
    mass = 0.0
    for index in order.tolist():
        selected.append(index)
        mass += float(weights[index])
        if mass >= target_mass:
            break
    indices = np.asarray(selected, dtype=np.int64)
    selected_weight = weights[indices]
    false_mass = float(selected_weight[target[indices] == 0].sum())
    return {
        "rows": int(indices.size),
        "actual_coverage": float(selected_weight.sum() / weights.sum()),
        "false_safe_rate": float(false_mass / selected_weight.sum()),
        "precision": float(1.0 - false_mass / selected_weight.sum()),
    }


def binary_ranking_metrics(
    score: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    *,
    coverages: Iterable[float] = ROUTE_FIRST_COVERAGES,
) -> dict[str, object]:
    values, target, weights = _aligned_binary_inputs(score, label, weight)
    mass = float(weights.sum())
    prevalence = float(weights[target == 1].sum() / mass)
    average_precision = weighted_average_precision(values, target, weights)
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    nll = -float(
        np.sum(weights * (target * np.log(clipped) + (1 - target) * np.log1p(-clipped)))
        / mass
    )
    brier = float(np.sum(weights * np.square(values - target)) / mass)
    return {
        "rows": int(values.size),
        "positive_rows": int(target.sum()),
        "group_equal_prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": float(average_precision / prevalence),
        "roc_auc": weighted_roc_auc(values, target, weights),
        "nll": nll,
        "brier": brier,
        "coverage": {
            f"{float(coverage):g}": false_safe_at_coverage(
                values, target, weights, coverage=float(coverage)
            )
            for coverage in coverages
        },
    }


def ordinal_score_metrics(
    probability: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
) -> dict[str, object]:
    scores = np.asarray(probability, dtype=np.float64)
    teacher = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    if (
        scores.shape != (teacher.size, 2)
        or teacher.size < 2
        or not set(np.unique(teacher).tolist()).issubset({11, 13, 27})
        or 11 not in teacher
        or 27 not in teacher
    ):
        raise ValueError("ordinal scores and teacher layers are misaligned")
    if not np.isfinite(scores).all() or np.any(scores[:, 0] > scores[:, 1]):
        raise ValueError("ordinal scores must be finite and nested")
    weights = route_first_group_weights(task_id, episode_index)
    metrics11 = binary_ranking_metrics(scores[:, 0], teacher == 11, weights)
    metrics13 = binary_ranking_metrics(scores[:, 1], teacher <= 13, weights)
    ap11 = float(metrics11["average_precision"])
    ap13 = float(metrics13["average_precision"])
    harmonic = float(2.0 * ap11 * ap13 / max(ap11 + ap13, 1e-12))
    return {
        "safe11": metrics11,
        "safe13": metrics13,
        "macro_average_precision": float((ap11 + ap13) / 2.0),
        "harmonic_average_precision": harmonic,
        "macro_roc_auc": float(
            (float(metrics11["roc_auc"]) + float(metrics13["roc_auc"])) / 2.0
        ),
        "nested_score_violations": int(np.sum(scores[:, 0] > scores[:, 1])),
    }


def _fit_from_projection(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    projection,
    candidate: RouteFirstCandidate,
    *,
    max_iter: int,
) -> RouteFirstOrdinalRouter:
    weights = route_first_group_weights(task_id, episode_index)
    head11 = fit_route_first_affine_head(
        features,
        (teacher_layer == 11).astype(np.int64),
        weights,
        projection,
        pca_rank=candidate.pca_rank,
        l2=candidate.l2,
        max_iter=max_iter,
    )
    head13 = fit_route_first_affine_head(
        features,
        (teacher_layer <= 13).astype(np.int64),
        weights,
        projection,
        pca_rank=candidate.pca_rank,
        l2=candidate.l2,
        max_iter=max_iter,
    )
    return RouteFirstOrdinalRouter(head11, head13)


def episode_index_candidate_search(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    candidates: Sequence[RouteFirstCandidate],
    max_iter: int = 100,
) -> dict[str, object]:
    values, teacher, tasks, episodes = _aligned_training_inputs(
        features, teacher_layer, task_id, episode_index
    )
    if not candidates or len({candidate.name for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("candidate grid must be non-empty and unique")
    held_values = tuple(sorted(np.unique(episodes).tolist()))
    if len(held_values) < 2:
        raise ValueError("candidate search requires at least two episode indices")
    maximum_rank = max(candidate.pca_rank for candidate in candidates)
    oof = {
        candidate.name: np.full((teacher.size, 2), np.nan, dtype=np.float64)
        for candidate in candidates
    }
    folds: list[dict[str, object]] = []
    for held_episode in held_values:
        valid = episodes == held_episode
        train = ~valid
        train_weights = route_first_group_weights(tasks[train], episodes[train])
        projection = fit_route_first_projection(
            values[train], train_weights, maximum_rank=maximum_rank
        )
        for candidate in candidates:
            router = _fit_from_projection(
                values[train],
                teacher[train],
                tasks[train],
                episodes[train],
                projection,
                candidate,
                max_iter=max_iter,
            )
            oof[candidate.name][valid] = router.probabilities(values[valid])
        folds.append(
            {
                "held_episode_index": int(held_episode),
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "valid_safe11_rows": int(np.sum(valid & (teacher == 11))),
                "valid_safe13_rows": int(np.sum(valid & (teacher <= 13))),
            }
        )
    reports: list[dict[str, object]] = []
    for candidate in candidates:
        scores = oof[candidate.name]
        if not np.isfinite(scores).all():
            raise RuntimeError(f"OOF scores are incomplete for {candidate.name}")
        metrics = ordinal_score_metrics(scores, teacher, tasks, episodes)
        reports.append(
            {
                "name": candidate.name,
                "pca_rank": candidate.pca_rank,
                "l2": candidate.l2,
                "metrics": metrics,
            }
        )
    selected_report = max(
        reports,
        key=lambda report: (
            float(report["metrics"]["harmonic_average_precision"]),
            float(report["metrics"]["macro_roc_auc"]),
            -int(report["pca_rank"]),
            float(report["l2"]),
        ),
    )
    selected = next(
        candidate for candidate in candidates if candidate.name == selected_report["name"]
    )
    return {
        "split": "leave_one_episode_index_out",
        "folds": folds,
        "candidates": reports,
        "selected": selected,
        "selected_scores": oof[selected.name],
        "selected_metrics": selected_report["metrics"],
    }


def leave_one_task_out_scores(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    candidate: RouteFirstCandidate,
    max_iter: int = 100,
) -> dict[str, object]:
    values, teacher, tasks, episodes = _aligned_training_inputs(
        features, teacher_layer, task_id, episode_index
    )
    if np.unique(tasks).size < 2:
        raise ValueError("task audit requires at least two tasks")
    scores = np.full((teacher.size, 2), np.nan, dtype=np.float64)
    folds: list[dict[str, object]] = []
    for held_task in sorted(np.unique(tasks).tolist()):
        valid = tasks == held_task
        train = ~valid
        train_weights = route_first_group_weights(tasks[train], episodes[train])
        projection = fit_route_first_projection(
            values[train], train_weights, maximum_rank=candidate.pca_rank
        )
        router = _fit_from_projection(
            values[train],
            teacher[train],
            tasks[train],
            episodes[train],
            projection,
            candidate,
            max_iter=max_iter,
        )
        scores[valid] = router.probabilities(values[valid])
        folds.append(
            {
                "held_task_id": int(held_task),
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "valid_safe11_rows": int(np.sum(valid & (teacher == 11))),
                "valid_safe13_rows": int(np.sum(valid & (teacher <= 13))),
            }
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("leave-one-task-out scores are incomplete")
    return {
        "split": "leave_one_task_out",
        "folds": folds,
        "scores": scores,
        "metrics": ordinal_score_metrics(scores, teacher, tasks, episodes),
    }


def fit_final_route_first_router(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    candidate: RouteFirstCandidate,
    max_iter: int = 100,
) -> RouteFirstOrdinalRouter:
    values, teacher, tasks, episodes = _aligned_training_inputs(
        features, teacher_layer, task_id, episode_index
    )
    weights = route_first_group_weights(tasks, episodes)
    projection = fit_route_first_projection(
        values, weights, maximum_rank=candidate.pca_rank
    )
    return _fit_from_projection(
        values,
        teacher,
        tasks,
        episodes,
        projection,
        candidate,
        max_iter=max_iter,
    )


def ranking_gates(
    episode_metrics: Mapping[str, object],
    task_metrics: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    episode11 = episode_metrics["safe11"]
    episode13 = episode_metrics["safe13"]
    task11 = task_metrics["safe11"]
    task13 = task_metrics["safe13"]
    return {
        "episode_safe11_ap_lift": float(episode11["average_precision_lift"])
        > float(thresholds["episode_oof_safe11_ap_lift_strictly_above"]),
        "episode_safe13_ap_lift": float(episode13["average_precision_lift"])
        > float(thresholds["episode_oof_safe13_ap_lift_strictly_above"]),
        "task_safe11_ap_lift": float(task11["average_precision_lift"])
        > float(thresholds["task_oof_safe11_ap_lift_strictly_above"]),
        "task_safe13_ap_lift": float(task13["average_precision_lift"])
        > float(thresholds["task_oof_safe13_ap_lift_strictly_above"]),
        "episode_safe11_false_safe_at_1pct": float(
            episode11["coverage"]["0.01"]["false_safe_rate"]
        )
        <= float(thresholds["episode_oof_safe11_false_safe_at_1pct_at_most"]),
        "episode_safe13_false_safe_at_5pct": float(
            episode13["coverage"]["0.05"]["false_safe_rate"]
        )
        <= float(thresholds["episode_oof_safe13_false_safe_at_5pct_at_most"]),
        "task_safe11_false_safe_at_1pct": float(
            task11["coverage"]["0.01"]["false_safe_rate"]
        )
        <= float(thresholds["task_oof_safe11_false_safe_at_1pct_at_most"]),
        "task_safe13_false_safe_at_5pct": float(
            task13["coverage"]["0.05"]["false_safe_rate"]
        )
        <= float(thresholds["task_oof_safe13_false_safe_at_5pct_at_most"]),
    }


__all__ = [
    "ROUTE_FIRST_COVERAGES",
    "RouteFirstCandidate",
    "binary_ranking_metrics",
    "episode_index_candidate_search",
    "false_safe_at_coverage",
    "fit_final_route_first_router",
    "leave_one_task_out_scores",
    "ordinal_score_metrics",
    "ranking_gates",
    "weighted_average_precision",
    "weighted_roc_auc",
]
