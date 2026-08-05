"""Preprocessing and fitted heads for the M4.25b temporal route router."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .causal_route_router import sigmoid_numpy


M425B_ROUTER_SCHEMA_VERSION = "phase-route-vla.m425b-temporal-route-router.v1"
M425B_VARIANTS = ("hidden_only", "step_proprio", "temporal_phase")


def _row_layer_norm(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("row-normalized features must be a finite matrix")
    mean = values.mean(axis=1, keepdims=True)
    variance = np.square(values - mean).mean(axis=1, keepdims=True)
    return (values - mean) / np.sqrt(variance + eps)


def _top_pca_components(centered: np.ndarray, rank: int) -> np.ndarray:
    """Return exact leading PCA directions, using the smaller Gram matrix.

    For the M4.25b regime the sample count is much smaller than the feature
    dimension.  Diagonalizing ``X X.T`` and recovering the right singular
    vectors is algebraically equivalent to a full SVD while avoiding repeated
    decompositions of a roughly 400 x 4,000 matrix.
    """

    values = np.asarray(centered, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("centered PCA input must be a finite matrix")
    max_rank = min(values.shape)
    if not 1 <= int(rank) <= max_rank:
        raise ValueError("PCA rank is outside the matrix dimensions")
    if values.shape[0] <= values.shape[1]:
        gram = values @ values.T
        eigenvalues, left = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][: int(rank)]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        tolerance = (
            np.finfo(np.float64).eps
            * max(values.shape)
            * max(float(singular[0]), 1.0)
        )
        if np.any(singular <= tolerance):
            # Rank-deficient edge cases need a complete orthonormal null-space
            # basis, which the dual recovery cannot determine uniquely.
            _, _, right = np.linalg.svd(values, full_matrices=False)
            components = right[: int(rank)]
        else:
            components = (left[:, order].T @ values) / singular[:, None]
            components /= np.linalg.norm(components, axis=1, keepdims=True)
    else:
        _, _, right = np.linalg.svd(values, full_matrices=False)
        components = right[: int(rank)]
    # Resolve the otherwise arbitrary component sign for reproducible hashes.
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), pivots])
    signs[signs == 0.0] = 1.0
    return components * signs[:, None]


def continuous_features(arrays: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    if variant == "step_proprio":
        return np.concatenate(
            [
                np.asarray(arrays["current_proprio"], dtype=np.float64),
                np.asarray(arrays["step_feature"], dtype=np.float64).reshape(-1, 1),
            ],
            axis=1,
        )
    if variant == "temporal_phase":
        rows = int(np.asarray(arrays["current_proprio"]).shape[0])
        return np.concatenate(
            [
                np.asarray(arrays["current_proprio"], dtype=np.float64),
                np.asarray(arrays["proprio_history"], dtype=np.float64).reshape(rows, -1),
                np.asarray(arrays["action_history"], dtype=np.float64).reshape(rows, -1),
                np.asarray(arrays["phase_scalars"], dtype=np.float64),
            ],
            axis=1,
        )
    if variant == "hidden_only":
        rows = int(np.asarray(arrays["layer11_hidden"]).shape[0])
        return np.empty((rows, 0), dtype=np.float64)
    raise ValueError(f"unsupported M4.25b variant: {variant}")


@dataclass(frozen=True)
class FeaturePreprocessor:
    variant: str
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.variant not in M425B_VARIANTS or self.layer_norm_eps <= 0.0:
            raise ValueError("invalid feature preprocessor configuration")
        mean = np.asarray(self.continuous_mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.continuous_scale, dtype=np.float64).reshape(-1)
        if mean.shape != scale.shape or not np.isfinite(mean).all():
            raise ValueError("continuous normalization arrays differ or are non-finite")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("continuous scales must be finite and positive")
        object.__setattr__(self, "continuous_mean", mean)
        object.__setattr__(self, "continuous_scale", scale)

    @classmethod
    def fit(
        cls,
        arrays: Mapping[str, np.ndarray],
        fit_mask: np.ndarray,
        *,
        variant: str,
        layer_norm_eps: float = 1e-6,
    ) -> "FeaturePreprocessor":
        mask = np.asarray(fit_mask, dtype=np.bool_).reshape(-1)
        continuous = continuous_features(arrays, variant)
        if continuous.shape[0] != mask.size or not mask.any():
            raise ValueError("preprocessor fit mask is empty or misaligned")
        if not np.isfinite(continuous).all():
            raise ValueError("continuous features contain a non-finite value")
        if continuous.shape[1]:
            mean = continuous[mask].mean(axis=0)
            scale = continuous[mask].std(axis=0)
            scale[scale < layer_norm_eps] = 1.0
        else:
            mean = np.empty((0,), dtype=np.float64)
            scale = np.empty((0,), dtype=np.float64)
        return cls(variant, mean, scale, layer_norm_eps)

    def transform(
        self, arrays: Mapping[str, np.ndarray], *, layer: int
    ) -> np.ndarray:
        if layer not in (11, 13):
            raise ValueError("router preprocessing only supports layer11/13")
        continuous = continuous_features(arrays, self.variant)
        if continuous.shape[1] != self.continuous_mean.size:
            raise ValueError("continuous feature dimension differs from preprocessor")
        standardized = (
            (continuous - self.continuous_mean) / self.continuous_scale
            if continuous.shape[1]
            else continuous
        )
        if self.variant == "step_proprio":
            output = standardized
        else:
            hidden = _row_layer_norm(
                np.asarray(arrays[f"layer{layer}_hidden"]), self.layer_norm_eps
            )
            if self.variant == "hidden_only":
                output = hidden
            else:
                stage = _row_layer_norm(
                    np.asarray(arrays["phase_stage"]), self.layer_norm_eps
                )
                history_mask = np.asarray(arrays["history_mask"], dtype=np.float64)
                output = np.concatenate(
                    [hidden, stage, standardized, history_mask], axis=1
                )
        if output.ndim != 2 or not np.isfinite(output).all():
            raise RuntimeError("router preprocessing produced invalid features")
        return output


@dataclass(frozen=True)
class ProcessedAffineHead:
    weight: np.ndarray
    bias: float
    pca_rank: int
    iterations: int

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64).reshape(-1)
        if weight.size < 1 or not np.isfinite(weight).all() or not math.isfinite(self.bias):
            raise ValueError("processed affine head parameters are invalid")
        object.__setattr__(self, "weight", weight)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.weight.size:
            raise ValueError("processed feature dimension differs from head")
        return sigmoid_numpy(values @ self.weight + float(self.bias))


def fit_processed_pca_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pca_rank: int = 64,
    l2: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
    eps: float = 1e-8,
) -> ProcessedAffineHead:
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != target.size or target.size < 3:
        raise ValueError("processed logistic inputs are invalid")
    if not np.isfinite(values).all() or set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("processed logistic requires finite features and both classes")
    if pca_rank < 1 or l2 <= 0.0 or max_iter < 1 or tolerance <= 0.0:
        raise ValueError("processed logistic hyperparameters are invalid")
    feature_mean = values.mean(axis=0)
    centered = values - feature_mean
    rank = min(int(pca_rank), values.shape[0] - 1, values.shape[1])
    components = _top_pca_components(centered, rank)
    scores = centered @ components.T
    score_scale = scores.std(axis=0)
    score_scale[score_scale < eps] = 1.0
    reduced = scores / score_scale
    design = np.concatenate(
        [reduced, np.ones((reduced.shape[0], 1), dtype=np.float64)], axis=1
    )
    positives = int(target.sum())
    negatives = int(target.size - positives)
    sample_weight = np.where(
        target == 1,
        target.size / (2.0 * positives),
        target.size / (2.0 * negatives),
    ).astype(np.float64)
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[-1, -1] = 0.0
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        probabilities = sigmoid_numpy(design @ beta)
        curvature = sample_weight * np.maximum(
            probabilities * (1.0 - probabilities), 1e-9
        )
        gradient = design.T @ (sample_weight * (probabilities - target))
        gradient += penalty @ beta
        hessian = design.T @ (curvature[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) <= tolerance:
            break
    weight = components.T @ (beta[:-1] / score_scale)
    bias = float(beta[-1] - feature_mean @ weight)
    return ProcessedAffineHead(weight, bias, rank, iterations)


@dataclass(frozen=True)
class TemporalRouteModel:
    variant: str
    preprocessor11: FeaturePreprocessor
    preprocessor13: FeaturePreprocessor
    head11: ProcessedAffineHead
    head13: ProcessedAffineHead
    threshold11: float
    threshold13: float

    def __post_init__(self) -> None:
        if self.variant not in M425B_VARIANTS:
            raise ValueError("unsupported temporal route model variant")
        for name, value in (
            ("threshold11", self.threshold11),
            ("threshold13", self.threshold13),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")

    def probabilities(
        self, arrays: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        feature11 = self.preprocessor11.transform(arrays, layer=11)
        feature13 = self.preprocessor13.transform(arrays, layer=13)
        return self.head11.probabilities(feature11), self.head13.probabilities(feature13)

    def save(self, path: str | Path, **extra: np.ndarray | str | int | float) -> None:
        payload = {
            "schema_version": np.asarray(M425B_ROUTER_SCHEMA_VERSION),
            "variant": np.asarray(self.variant),
            "threshold11": np.asarray(self.threshold11, dtype=np.float64),
            "threshold13": np.asarray(self.threshold13, dtype=np.float64),
            "layer_norm_eps": np.asarray(
                self.preprocessor11.layer_norm_eps, dtype=np.float64
            ),
            "continuous_mean11": self.preprocessor11.continuous_mean,
            "continuous_scale11": self.preprocessor11.continuous_scale,
            "continuous_mean13": self.preprocessor13.continuous_mean,
            "continuous_scale13": self.preprocessor13.continuous_scale,
            "weight11": self.head11.weight,
            "bias11": np.asarray(self.head11.bias, dtype=np.float64),
            "weight13": self.head13.weight,
            "bias13": np.asarray(self.head13.bias, dtype=np.float64),
            "pca_rank11": np.asarray(self.head11.pca_rank, dtype=np.int64),
            "pca_rank13": np.asarray(self.head13.pca_rank, dtype=np.int64),
            "iterations11": np.asarray(self.head11.iterations, dtype=np.int64),
            "iterations13": np.asarray(self.head13.iterations, dtype=np.int64),
        }
        payload.update({name: np.asarray(value) for name, value in extra.items()})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path: str | Path) -> "TemporalRouteModel":
        with np.load(Path(path), allow_pickle=False) as arrays:
            if str(arrays["schema_version"].item()) != M425B_ROUTER_SCHEMA_VERSION:
                raise ValueError("unexpected M4.25b router schema")
            variant = str(arrays["variant"].item())
            eps = float(arrays["layer_norm_eps"])
            pp11 = FeaturePreprocessor(
                variant, arrays["continuous_mean11"], arrays["continuous_scale11"], eps
            )
            pp13 = FeaturePreprocessor(
                variant, arrays["continuous_mean13"], arrays["continuous_scale13"], eps
            )
            head11 = ProcessedAffineHead(
                arrays["weight11"],
                float(arrays["bias11"]),
                int(arrays["pca_rank11"]),
                int(arrays["iterations11"]),
            )
            head13 = ProcessedAffineHead(
                arrays["weight13"],
                float(arrays["bias13"]),
                int(arrays["pca_rank13"]),
                int(arrays["iterations13"]),
            )
            return cls(
                variant,
                pp11,
                pp13,
                head11,
                head13,
                float(arrays["threshold11"]),
                float(arrays["threshold13"]),
            )
