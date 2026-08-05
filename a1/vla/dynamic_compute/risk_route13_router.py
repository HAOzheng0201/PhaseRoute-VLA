"""Risk-constrained binary route13/27 router for M4.26.

The router is intentionally offline-only.  It consumes features that exist
after transformer block 13 and predicts either a safe shallow solve at layer
13 or a fail-closed continuation to layer 27.  It never emits layer 11.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .temporal_route_router import (
    ProcessedAffineHead,
    _row_layer_norm,
    fit_processed_pca_logistic,
)


M426_ROUTER_SCHEMA_VERSION = "phase-route-vla.m426-risk-route13-router.v1"
M426_FEATURE_SCHEMA_VERSION = "phase-route-vla.m426-temporal-route-features.v1"
M426A_FEATURE_SCHEMA_VERSION = "phase-route-vla.m426a-temporal-route-features.v1"
M427_FEATURE_SCHEMA_VERSION = "phase-route-vla.m427-temporal-route-features.v1"
M428_FEATURE_SCHEMA_VERSION = "phase-route-vla.m428-temporal-route-features.v1"
M426_VARIANTS = ("hidden_only", "step_proprio", "temporal_phase_step")
M426_ROUTE_LAYERS = (13, 27)


def _continuous_features(
    arrays: Mapping[str, np.ndarray], variant: str
) -> np.ndarray:
    current = np.asarray(arrays["current_proprio"], dtype=np.float64)
    if current.ndim != 2:
        raise ValueError("current_proprio must have shape [N, P]")
    rows = current.shape[0]
    step = np.asarray(arrays["step_feature"], dtype=np.float64).reshape(rows, 1)
    if variant == "hidden_only":
        return np.empty((rows, 0), dtype=np.float64)
    if variant == "step_proprio":
        return np.concatenate([current, step], axis=1)
    if variant == "temporal_phase_step":
        return np.concatenate(
            [
                current,
                np.asarray(arrays["proprio_history"], dtype=np.float64).reshape(
                    rows, -1
                ),
                np.asarray(arrays["action_history"], dtype=np.float64).reshape(
                    rows, -1
                ),
                np.asarray(arrays["phase_scalars"], dtype=np.float64).reshape(
                    rows, -1
                ),
                step,
            ],
            axis=1,
        )
    raise ValueError(f"unsupported M4.26 variant: {variant}")


@dataclass(frozen=True)
class Route13FeaturePreprocessor:
    """Fold-local preprocessing for a single layer-13 decision head."""

    variant: str
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.variant not in M426_VARIANTS or self.layer_norm_eps <= 0.0:
            raise ValueError("invalid M4.26 feature preprocessor configuration")
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
    ) -> "Route13FeaturePreprocessor":
        mask = np.asarray(fit_mask, dtype=np.bool_).reshape(-1)
        continuous = _continuous_features(arrays, variant)
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

    def transform(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        continuous = _continuous_features(arrays, self.variant)
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
                np.asarray(arrays["layer13_hidden"]), self.layer_norm_eps
            )
            if self.variant == "hidden_only":
                output = hidden
            else:
                stage = _row_layer_norm(
                    np.asarray(arrays["phase_stage"]), self.layer_norm_eps
                )
                history_mask = np.asarray(arrays["history_mask"], dtype=np.float64)
                if history_mask.ndim != 2 or history_mask.shape[0] != hidden.shape[0]:
                    raise ValueError("history mask is misaligned")
                output = np.concatenate(
                    [hidden, stage, standardized, history_mask], axis=1
                )
        if output.ndim != 2 or not np.isfinite(output).all():
            raise RuntimeError("M4.26 preprocessing produced invalid features")
        return output


def route13_or_27(probability_safe13: np.ndarray, *, threshold: float) -> np.ndarray:
    probability = np.asarray(probability_safe13, dtype=np.float64).reshape(-1)
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("safe13 probabilities must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("route13 threshold must be in (0, 1]")
    return np.where(probability >= threshold, 13, 27).astype(np.int64)


def route13_metrics(
    predicted: np.ndarray, teacher_route: np.ndarray
) -> dict[str, float | int | dict[str, int]]:
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    teacher = np.asarray(teacher_route, dtype=np.int64).reshape(-1)
    if predicted.shape != teacher.shape or teacher.size < 1:
        raise ValueError("predicted and teacher routes must be aligned and non-empty")
    if not set(np.unique(predicted).tolist()).issubset(M426_ROUTE_LAYERS):
        raise ValueError("M4.26 predictions must only contain route13/27")
    if not set(np.unique(teacher).tolist()).issubset({11, 13, 27}):
        raise ValueError("teacher routes must be canonical 11/13/27")
    safe13 = teacher <= 13
    predicted13 = predicted == 13
    route27 = teacher == 27
    binary_correct = predicted13 == safe13
    false_shallow = predicted13 & route27
    recalled = predicted13 & safe13
    return {
        "rows": int(teacher.size),
        "binary_exact": int(binary_correct.sum()),
        "binary_exact_accuracy": float(binary_correct.mean()),
        "false_shallow": int(false_shallow.sum()),
        "false_shallow_rate": float(false_shallow.mean()),
        "route27_rows": int(route27.sum()),
        "route27_false_shallow": int(false_shallow.sum()),
        "safe13_rows": int(safe13.sum()),
        "safe13_recalled": int(recalled.sum()),
        "safe13_recall": float(recalled.sum() / max(int(safe13.sum()), 1)),
        "predicted13_coverage": float(predicted13.mean()),
        "overcompute_safe13_to27": int((safe13 & ~predicted13).sum()),
        "predicted_distribution": {
            "13": int(predicted13.sum()),
            "27": int((~predicted13).sum()),
        },
    }


@dataclass(frozen=True)
class RiskRoute13Model:
    variant: str
    preprocessor: Route13FeaturePreprocessor
    head: ProcessedAffineHead
    threshold: float

    def __post_init__(self) -> None:
        if self.variant not in M426_VARIANTS or self.preprocessor.variant != self.variant:
            raise ValueError("unsupported or inconsistent M4.26 router variant")
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
            raise ValueError("M4.26 threshold must be in (0, 1]")

    def probabilities(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.head.probabilities(self.preprocessor.transform(arrays))

    def routes(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return route13_or_27(self.probabilities(arrays), threshold=self.threshold)

    def save(self, path: str | Path, **extra: np.ndarray | str | int | float) -> None:
        payload = {
            "schema_version": np.asarray(M426_ROUTER_SCHEMA_VERSION),
            "variant": np.asarray(self.variant),
            "threshold": np.asarray(self.threshold, dtype=np.float64),
            "layer_norm_eps": np.asarray(
                self.preprocessor.layer_norm_eps, dtype=np.float64
            ),
            "continuous_mean": self.preprocessor.continuous_mean,
            "continuous_scale": self.preprocessor.continuous_scale,
            "weight": self.head.weight,
            "bias": np.asarray(self.head.bias, dtype=np.float64),
            "pca_rank": np.asarray(self.head.pca_rank, dtype=np.int64),
            "iterations": np.asarray(self.head.iterations, dtype=np.int64),
        }
        payload.update({name: np.asarray(value) for name, value in extra.items()})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path: str | Path) -> "RiskRoute13Model":
        with np.load(Path(path), allow_pickle=False) as arrays:
            if str(arrays["schema_version"].item()) != M426_ROUTER_SCHEMA_VERSION:
                raise ValueError("unexpected M4.26 router schema")
            variant = str(arrays["variant"].item())
            preprocessor = Route13FeaturePreprocessor(
                variant,
                arrays["continuous_mean"],
                arrays["continuous_scale"],
                float(arrays["layer_norm_eps"]),
            )
            head = ProcessedAffineHead(
                arrays["weight"],
                float(arrays["bias"]),
                int(arrays["pca_rank"]),
                int(arrays["iterations"]),
            )
            return cls(variant, preprocessor, head, float(arrays["threshold"]))


def fit_route13_head(
    arrays: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    *,
    variant: str,
    pca_rank: int = 64,
    l2: float = 1.0,
    max_iter: int = 100,
    layer_norm_eps: float = 1e-6,
) -> tuple[Route13FeaturePreprocessor, ProcessedAffineHead, np.ndarray]:
    mask = np.asarray(fit_mask, dtype=np.bool_).reshape(-1)
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64).reshape(-1)
    if teacher.shape != mask.shape:
        raise ValueError("teacher route and fit mask differ")
    labels = (teacher <= 13).astype(np.int64)
    preprocessor = Route13FeaturePreprocessor.fit(
        arrays, mask, variant=variant, layer_norm_eps=layer_norm_eps
    )
    features = preprocessor.transform(arrays)
    head = fit_processed_pca_logistic(
        features[mask],
        labels[mask],
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
    )
    return preprocessor, head, features
