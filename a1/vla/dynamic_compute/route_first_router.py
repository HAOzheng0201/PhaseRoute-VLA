"""Tiny action-free ordinal router for route-first dynamic computation.

Training may use a fold-local weighted PCA projection for regularization, but
every fitted head is collapsed back to one affine map over the frozen 199D
context.  Runtime therefore needs only two dot products and does not carry a
PCA dependency.  Deployment thresholds are intentionally absent: they belong
to the separately sealed states-8/9 calibration stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
)


ROUTE_FIRST_ROUTER_SCHEMA_VERSION = "phase-route-vla.route-first-router.v1"
ROUTE_FIRST_ROUTER_CALIBRATION_STATUS = "UNSET"


def _finite_matrix(values: np.ndarray, *, dimension: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] < 1:
        raise ValueError(f"features must have shape [N,{dimension}]")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return matrix


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def route_first_group_weights(
    task_id: np.ndarray, episode_index: np.ndarray
) -> np.ndarray:
    """Return row weights giving every task/episode cell equal total mass."""

    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if tasks.shape != episodes.shape or tasks.size < 1:
        raise ValueError("task and episode arrays must be aligned and non-empty")
    if np.any(tasks < 0) or np.any(episodes < 0):
        raise ValueError("task and episode identities must be non-negative")
    groups = list(zip(tasks.tolist(), episodes.tolist()))
    counts: dict[tuple[int, int], int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    weights *= weights.size / weights.sum()
    return weights


@dataclass(frozen=True)
class RouteFirstPCAProjection:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    components: np.ndarray
    score_scale: np.ndarray
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.feature_scale, dtype=np.float64).reshape(-1)
        components = np.asarray(self.components, dtype=np.float64)
        score_scale = np.asarray(self.score_scale, dtype=np.float64).reshape(-1)
        if mean.shape != (ROUTE_FIRST_FEATURE_DIMENSION,) or scale.shape != mean.shape:
            raise ValueError("route-first projection feature geometry differs")
        if (
            components.ndim != 2
            or components.shape[1] != mean.size
            or components.shape[0] != score_scale.size
            or components.shape[0] < 1
        ):
            raise ValueError("route-first PCA geometry differs")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(components).all()
            or not np.isfinite(score_scale).all()
            or np.any(scale <= 0.0)
            or np.any(score_scale <= 0.0)
            or not math.isfinite(self.epsilon)
            or self.epsilon <= 0.0
        ):
            raise ValueError("route-first PCA parameters are invalid")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "score_scale", score_scale)

    @property
    def maximum_rank(self) -> int:
        return int(self.components.shape[0])

    def transform(self, features: np.ndarray, *, rank: int) -> np.ndarray:
        values = _finite_matrix(features, dimension=self.feature_mean.size)
        rank = int(rank)
        if not 1 <= rank <= self.maximum_rank:
            raise ValueError("PCA rank is outside the fitted projection")
        standardized = (values - self.feature_mean) / self.feature_scale
        return (
            standardized @ self.components[:rank].T
        ) / self.score_scale[:rank]

    def collapse(
        self, coefficient: np.ndarray, bias: float, *, rank: int
    ) -> tuple[np.ndarray, float]:
        beta = np.asarray(coefficient, dtype=np.float64).reshape(-1)
        rank = int(rank)
        if beta.shape != (rank,) or not 1 <= rank <= self.maximum_rank:
            raise ValueError("projected coefficient geometry differs")
        standardized_weight = self.components[:rank].T @ (
            beta / self.score_scale[:rank]
        )
        raw_weight = standardized_weight / self.feature_scale
        raw_bias = float(bias) - float(self.feature_mean @ raw_weight)
        if not np.isfinite(raw_weight).all() or not math.isfinite(raw_bias):
            raise RuntimeError("collapsed route-first head is non-finite")
        return raw_weight, raw_bias


def fit_route_first_projection(
    features: np.ndarray,
    sample_weight: np.ndarray,
    *,
    maximum_rank: int,
    epsilon: float = 1e-8,
) -> RouteFirstPCAProjection:
    values = _finite_matrix(features, dimension=ROUTE_FIRST_FEATURE_DIMENSION)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if (
        weights.shape != (values.shape[0],)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("projection sample weights must be finite and positive")
    maximum_rank = int(maximum_rank)
    maximum_supported = min(values.shape[0] - 1, values.shape[1])
    if not 1 <= maximum_rank <= maximum_supported or epsilon <= 0.0:
        raise ValueError("maximum PCA rank or epsilon is invalid")
    mass = float(weights.sum())
    mean = (values * weights[:, None]).sum(axis=0) / mass
    centered = values - mean
    variance = centered**2
    variance = (variance * weights[:, None]).sum(axis=0) / mass
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < epsilon] = 1.0
    standardized = centered / scale
    weighted = standardized * np.sqrt(weights[:, None])
    _, _, right = np.linalg.svd(weighted, full_matrices=False)
    components = right[:maximum_rank].copy()
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(maximum_rank), pivots])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    scores = standardized @ components.T
    score_variance = scores**2
    score_variance = (score_variance * weights[:, None]).sum(axis=0) / mass
    score_scale = np.sqrt(np.maximum(score_variance, 0.0))
    score_scale[score_scale < epsilon] = 1.0
    return RouteFirstPCAProjection(
        feature_mean=mean,
        feature_scale=scale,
        components=components,
        score_scale=score_scale,
        epsilon=epsilon,
    )


@dataclass(frozen=True)
class RouteFirstAffineHead:
    weight: np.ndarray
    bias: float
    pca_rank: int
    l2: float
    iterations: int

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64).reshape(-1)
        if (
            weight.shape != (ROUTE_FIRST_FEATURE_DIMENSION,)
            or not np.isfinite(weight).all()
            or not math.isfinite(float(self.bias))
            or self.pca_rank < 1
            or not math.isfinite(float(self.l2))
            or self.l2 <= 0.0
            or self.iterations < 1
        ):
            raise ValueError("route-first affine head parameters are invalid")
        object.__setattr__(self, "weight", weight)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = _finite_matrix(features, dimension=self.weight.size)
        return _sigmoid(values @ self.weight + float(self.bias))


def _balanced_binary_weights(
    labels: np.ndarray, base_weight: np.ndarray
) -> np.ndarray:
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    base = np.asarray(base_weight, dtype=np.float64).reshape(-1)
    if target.shape != base.shape or set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("balanced binary fitting requires two aligned classes")
    positive_mass = float(base[target == 1].sum())
    negative_mass = float(base[target == 0].sum())
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        raise ValueError("binary class mass must be positive")
    weights = base * np.where(
        target == 1,
        0.5 / positive_mass,
        0.5 / negative_mass,
    )
    weights *= target.size / weights.sum()
    return weights


def fit_route_first_affine_head(
    features: np.ndarray,
    labels: np.ndarray,
    base_weight: np.ndarray,
    projection: RouteFirstPCAProjection,
    *,
    pca_rank: int,
    l2: float,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> RouteFirstAffineHead:
    values = _finite_matrix(features, dimension=ROUTE_FIRST_FEATURE_DIMENSION)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    base = np.asarray(base_weight, dtype=np.float64).reshape(-1)
    if target.shape != (values.shape[0],) or base.shape != target.shape:
        raise ValueError("route-first fitting arrays are misaligned")
    if l2 <= 0.0 or max_iter < 1 or tolerance <= 0.0:
        raise ValueError("route-first logistic hyperparameters are invalid")
    design_features = projection.transform(values, rank=pca_rank)
    design = np.concatenate(
        [design_features, np.ones((values.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    sample_weight = _balanced_binary_weights(target, base)
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[-1, -1] = 0.0
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        probability = _sigmoid(design @ beta)
        curvature = sample_weight * np.maximum(
            probability * (1.0 - probability), 1e-9
        )
        gradient = design.T @ (sample_weight * (probability - target))
        gradient += penalty @ beta
        hessian = design.T @ (curvature[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) <= tolerance:
            break
    raw_weight, raw_bias = projection.collapse(
        beta[:-1], float(beta[-1]), rank=pca_rank
    )
    return RouteFirstAffineHead(
        weight=raw_weight,
        bias=raw_bias,
        pca_rank=int(pca_rank),
        l2=float(l2),
        iterations=iterations,
    )


@dataclass(frozen=True)
class RouteFirstOrdinalRouter:
    head11: RouteFirstAffineHead
    head13: RouteFirstAffineHead

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Return nested ``[N,2]`` safety scores for L11 and L13."""

        probability13 = self.head13.probabilities(features)
        raw_probability11 = self.head11.probabilities(features)
        probability11 = np.minimum(raw_probability11, probability13)
        output = np.stack((probability11, probability13), axis=1)
        if not np.isfinite(output).all() or np.any(
            (output < 0.0) | (output > 1.0)
        ):
            raise RuntimeError("route-first probabilities are invalid")
        return output


def fit_route_first_ordinal_router(
    features: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    pca_rank: int,
    l2: float,
    maximum_rank: int | None = None,
    max_iter: int = 100,
    epsilon: float = 1e-8,
) -> RouteFirstOrdinalRouter:
    values = _finite_matrix(features, dimension=ROUTE_FIRST_FEATURE_DIMENSION)
    teacher = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    if teacher.shape != (values.shape[0],) or not set(np.unique(teacher)).issubset(
        {11, 13, 27}
    ):
        raise ValueError("route-first teacher layers are invalid or misaligned")
    weights = route_first_group_weights(task_id, episode_index)
    projection = fit_route_first_projection(
        values,
        weights,
        maximum_rank=max(int(pca_rank), int(maximum_rank or pca_rank)),
        epsilon=epsilon,
    )
    head11 = fit_route_first_affine_head(
        values,
        (teacher == 11).astype(np.int64),
        weights,
        projection,
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
    )
    head13 = fit_route_first_affine_head(
        values,
        (teacher <= 13).astype(np.int64),
        weights,
        projection,
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
    )
    return RouteFirstOrdinalRouter(head11=head11, head13=head13)


def save_uncalibrated_route_first_router(
    path: str | Path,
    router: RouteFirstOrdinalRouter,
    *,
    training_payload_sha256: str,
    training_file_sha256: str,
    task_ids: Sequence[int],
    episode_indices: Sequence[int],
    seed: int,
) -> None:
    """Publish score heads without deployment thresholds."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    hashes = (training_payload_sha256, training_file_sha256)
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in hashes
    ):
        raise ValueError("training hashes must be lowercase SHA-256")
    tasks = tuple(int(value) for value in task_ids)
    episodes = tuple(int(value) for value in episode_indices)
    if (
        not tasks
        or not episodes
        or list(tasks) != sorted(set(tasks))
        or list(episodes) != sorted(set(episodes))
        or any(value < 0 for value in tasks + episodes)
    ):
        raise ValueError("training task/episode grids must be sorted and unique")
    payload: Mapping[str, np.ndarray] = {
        "schema_version": np.asarray(ROUTE_FIRST_ROUTER_SCHEMA_VERSION),
        "feature_schema_version": np.asarray(ROUTE_FIRST_FEATURE_SCHEMA_VERSION),
        "feature_dimension": np.asarray(ROUTE_FIRST_FEATURE_DIMENSION, dtype=np.int32),
        "calibration_status": np.asarray(ROUTE_FIRST_ROUTER_CALIBRATION_STATUS),
        "weight11": router.head11.weight.astype(np.float32),
        "bias11": np.asarray(router.head11.bias, dtype=np.float64),
        "pca_rank11": np.asarray(router.head11.pca_rank, dtype=np.int32),
        "iterations11": np.asarray(router.head11.iterations, dtype=np.int32),
        "weight13": router.head13.weight.astype(np.float32),
        "bias13": np.asarray(router.head13.bias, dtype=np.float64),
        "pca_rank13": np.asarray(router.head13.pca_rank, dtype=np.int32),
        "iterations13": np.asarray(router.head13.iterations, dtype=np.int32),
        "l2": np.asarray(router.head11.l2, dtype=np.float64),
        "training_payload_sha256": np.asarray(training_payload_sha256),
        "training_file_sha256": np.asarray(training_file_sha256),
        "training_task_ids": np.asarray(tasks, dtype=np.int16),
        "training_episode_indices": np.asarray(episodes, dtype=np.int16),
        "seed": np.asarray(int(seed), dtype=np.int64),
    }
    if router.head11.l2 != router.head13.l2:
        raise ValueError("route-first heads must share one selected L2 value")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite {temporary}")
    try:
        with temporary.open("xb") as output_file:
            np.savez_compressed(output_file, **payload)
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def load_uncalibrated_route_first_router(
    path: str | Path,
) -> tuple[RouteFirstOrdinalRouter, dict[str, object]]:
    target = Path(path).expanduser().resolve(strict=True)
    required = {
        "schema_version",
        "feature_schema_version",
        "feature_dimension",
        "calibration_status",
        "weight11",
        "bias11",
        "pca_rank11",
        "iterations11",
        "weight13",
        "bias13",
        "pca_rank13",
        "iterations13",
        "l2",
        "training_payload_sha256",
        "training_file_sha256",
        "training_task_ids",
        "training_episode_indices",
        "seed",
    }
    with np.load(target, allow_pickle=False) as arrays:
        if set(arrays.files) != required:
            raise ValueError("route-first router fields differ")
        if str(arrays["schema_version"].item()) != ROUTE_FIRST_ROUTER_SCHEMA_VERSION:
            raise ValueError("route-first router schema differs")
        if (
            str(arrays["feature_schema_version"].item())
            != ROUTE_FIRST_FEATURE_SCHEMA_VERSION
            or int(arrays["feature_dimension"].item())
            != ROUTE_FIRST_FEATURE_DIMENSION
        ):
            raise ValueError("route-first router feature contract differs")
        if (
            str(arrays["calibration_status"].item())
            != ROUTE_FIRST_ROUTER_CALIBRATION_STATUS
        ):
            raise ValueError("route-first router is not an uncalibrated score model")
        l2 = float(arrays["l2"].item())
        head11 = RouteFirstAffineHead(
            arrays["weight11"].astype(np.float64),
            float(arrays["bias11"].item()),
            int(arrays["pca_rank11"].item()),
            l2,
            int(arrays["iterations11"].item()),
        )
        head13 = RouteFirstAffineHead(
            arrays["weight13"].astype(np.float64),
            float(arrays["bias13"].item()),
            int(arrays["pca_rank13"].item()),
            l2,
            int(arrays["iterations13"].item()),
        )
        training_payload_sha256 = str(arrays["training_payload_sha256"].item())
        training_file_sha256 = str(arrays["training_file_sha256"].item())
        training_task_ids = arrays["training_task_ids"].astype(np.int64).tolist()
        training_episode_indices = arrays["training_episode_indices"].astype(
            np.int64
        ).tolist()
        hashes = (training_payload_sha256, training_file_sha256)
        if any(
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ValueError("route-first router training hashes are invalid")
        if (
            not training_task_ids
            or not training_episode_indices
            or training_task_ids != sorted(set(training_task_ids))
            or training_episode_indices != sorted(set(training_episode_indices))
            or training_task_ids[0] < 0
            or training_episode_indices[0] < 0
        ):
            raise ValueError("route-first router training grid is invalid")
        metadata: dict[str, object] = {
            "training_payload_sha256": training_payload_sha256,
            "training_file_sha256": training_file_sha256,
            "training_task_ids": training_task_ids,
            "training_episode_indices": training_episode_indices,
            "seed": int(arrays["seed"].item()),
            "calibration_status": ROUTE_FIRST_ROUTER_CALIBRATION_STATUS,
        }
    return RouteFirstOrdinalRouter(head11, head13), metadata


__all__ = [
    "ROUTE_FIRST_ROUTER_CALIBRATION_STATUS",
    "ROUTE_FIRST_ROUTER_SCHEMA_VERSION",
    "RouteFirstAffineHead",
    "RouteFirstOrdinalRouter",
    "RouteFirstPCAProjection",
    "fit_route_first_affine_head",
    "fit_route_first_ordinal_router",
    "fit_route_first_projection",
    "load_uncalibrated_route_first_router",
    "route_first_group_weights",
    "save_uncalibrated_route_first_router",
]
