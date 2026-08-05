"""Leakage-safe two-stage causal router for route-then-solve.

The router observes only the post-block proprio-token hidden state.  Layer 11
can stop at route 11; otherwise layer 13 can stop at route 13; every uncertain
case falls through to route 27.  Training helpers intentionally use only
NumPy so a fitted PCA/logistic model can be collapsed to one affine head per
decision point and loaded without a training-time dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


M425_ROUTER_SCHEMA_VERSION = "phase-route-vla.m425-causal-router.v1"
ROUTE_LAYERS = (11, 13, 27)


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return value


def normalize_hidden_numpy(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply parameter-free per-row LayerNorm in float64."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("features must have shape [N, D]")
    if not np.isfinite(values).all():
        raise ValueError("features contain a non-finite value")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    mean = values.mean(axis=1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=1, keepdims=True)
    return (values - mean) / np.sqrt(variance + eps)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0.0
    output = np.empty_like(logits)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


@dataclass(frozen=True)
class AffineBinaryHead:
    weight: np.ndarray
    bias: float
    pca_rank: int
    iterations: int

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64)
        if weight.ndim != 1 or weight.size < 1 or not np.isfinite(weight).all():
            raise ValueError("head weight must be a finite vector")
        if not math.isfinite(float(self.bias)):
            raise ValueError("head bias must be finite")
        object.__setattr__(self, "weight", weight)

    def probabilities(self, features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        normalized = normalize_hidden_numpy(features, eps=eps)
        if normalized.shape[1] != self.weight.shape[0]:
            raise ValueError("feature dimension differs from affine head")
        return sigmoid_numpy(normalized @ self.weight + float(self.bias))


def fit_pca_logistic_affine(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pca_rank: int = 32,
    l2: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
    eps: float = 1e-6,
) -> AffineBinaryHead:
    """Fit class-balanced ridge logistic regression and collapse its PCA."""

    values = normalize_hidden_numpy(features, eps=eps)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.shape[0] != target.shape[0] or values.shape[0] < 3:
        raise ValueError("features and labels must contain at least three aligned rows")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("binary fitting requires both classes")
    if pca_rank < 1 or l2 <= 0.0 or max_iter < 1 or tolerance <= 0.0:
        raise ValueError("invalid logistic fitting hyperparameters")

    feature_mean = values.mean(axis=0)
    centered = values - feature_mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(pca_rank), values.shape[0] - 1, values.shape[1], right.shape[0])
    components = right[:rank]
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

    normalized_weight = components.T @ (beta[:-1] / score_scale)
    bias = float(beta[-1] - feature_mean @ normalized_weight)
    if not np.isfinite(normalized_weight).all() or not math.isfinite(bias):
        raise RuntimeError("logistic fitting produced non-finite parameters")
    return AffineBinaryHead(
        weight=normalized_weight,
        bias=bias,
        pca_rank=rank,
        iterations=iterations,
    )


def calibrate_zero_false_positive(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    minimum_threshold: float = 0.5,
) -> float:
    """Choose the smallest threshold strictly above every negative score."""

    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != target.shape or scores.size < 1:
        raise ValueError("probabilities and labels must be aligned and non-empty")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    negatives = scores[target == 0]
    if negatives.size < 1:
        raise ValueError("zero-false-positive calibration requires negatives")
    threshold = max(
        float(minimum_threshold),
        float(np.nextafter(float(negatives.max()), 1.0)),
    )
    return _validate_probability(threshold, "calibrated threshold")


def sequential_routes(
    probability11: np.ndarray,
    probability13: np.ndarray,
    *,
    threshold11: float,
    threshold13: float,
) -> np.ndarray:
    p11 = np.asarray(probability11, dtype=np.float64).reshape(-1)
    p13 = np.asarray(probability13, dtype=np.float64).reshape(-1)
    if p11.shape != p13.shape or not np.isfinite(p11).all() or not np.isfinite(p13).all():
        raise ValueError("route probabilities must be aligned and finite")
    threshold11 = _validate_probability(threshold11, "threshold11")
    threshold13 = _validate_probability(threshold13, "threshold13")
    routes = np.full(p11.shape, 27, dtype=np.int64)
    routes[(p11 < threshold11) & (p13 >= threshold13)] = 13
    routes[p11 >= threshold11] = 11
    return routes


def route_metrics(predicted: np.ndarray, teacher: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    teacher = np.asarray(teacher, dtype=np.int64).reshape(-1)
    if predicted.shape != teacher.shape or predicted.size < 1:
        raise ValueError("predicted and teacher routes must be aligned and non-empty")
    if not set(np.unique(predicted)).issubset(ROUTE_LAYERS):
        raise ValueError("predicted routes contain an unsupported layer")
    if not set(np.unique(teacher)).issubset(ROUTE_LAYERS):
        raise ValueError("teacher routes contain an unsupported layer")
    false_shallow = predicted < teacher
    teacher27 = teacher == 27
    counts = {str(layer): int(np.sum(predicted == layer)) for layer in ROUTE_LAYERS}
    return {
        "rows": int(teacher.size),
        "exact": int(np.sum(predicted == teacher)),
        "exact_accuracy": float(np.mean(predicted == teacher)),
        "false_shallow": int(false_shallow.sum()),
        "false_shallow_rate": float(false_shallow.mean()),
        "teacher27_rows": int(teacher27.sum()),
        "teacher27_false_shallow": int(np.sum(false_shallow & teacher27)),
        "shallow_coverage": float(np.mean(predicted < 27)),
        "predicted_route_mean": float(predicted.mean()),
        "teacher_route_mean": float(teacher.mean()),
        "predicted_distribution": counts,
    }


@dataclass(frozen=True)
class CausalRouteRouterConfig:
    hidden_dim: int = 3584
    threshold11: float = 1.0
    threshold13: float = 1.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.layer_norm_eps <= 0.0:
            raise ValueError("invalid router dimensions or epsilon")
        _validate_probability(self.threshold11, "threshold11")
        _validate_probability(self.threshold13, "threshold13")


class CausalRouteRouter(nn.Module):
    """Two single-affine heads with conservative sequential decisions."""

    def __init__(
        self,
        config: CausalRouteRouterConfig,
        *,
        weight11: torch.Tensor,
        bias11: float,
        weight13: torch.Tensor,
        bias13: float,
    ) -> None:
        super().__init__()
        self.config = config
        for name, weight in (("weight11", weight11), ("weight13", weight13)):
            value = torch.as_tensor(weight, dtype=torch.float32).reshape(-1)
            if value.shape != (config.hidden_dim,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite hidden_dim vector")
            self.register_buffer(name, value.clone())
        self.register_buffer("bias11", torch.tensor(float(bias11), dtype=torch.float32))
        self.register_buffer("bias13", torch.tensor(float(bias13), dtype=torch.float32))

    def _probability(self, hidden: torch.Tensor, *, layer: int) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.config.hidden_dim:
            raise ValueError("router hidden must have shape [B, hidden_dim]")
        normalized = torch.nn.functional.layer_norm(
            hidden.float(),
            (self.config.hidden_dim,),
            eps=self.config.layer_norm_eps,
        )
        if layer == 11:
            return torch.sigmoid(normalized @ self.weight11 + self.bias11)
        if layer == 13:
            return torch.sigmoid(normalized @ self.weight13 + self.bias13)
        raise ValueError("router can only evaluate layer 11 or 13")

    def probability(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        return self._probability(hidden, layer=int(layer))

    def should_exit(self, layer: int, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = self._probability(hidden, layer=int(layer))
        threshold = self.config.threshold11 if int(layer) == 11 else self.config.threshold13
        return probability >= threshold, probability

    @classmethod
    def from_npz(cls, path: str | Path) -> "CausalRouteRouter":
        with np.load(Path(path), allow_pickle=False) as arrays:
            schema = str(arrays["schema_version"].item())
            if schema != M425_ROUTER_SCHEMA_VERSION:
                raise ValueError(f"unexpected router checkpoint schema: {schema}")
            config = CausalRouteRouterConfig(
                hidden_dim=int(arrays["hidden_dim"].item()),
                threshold11=float(arrays["threshold11"].item()),
                threshold13=float(arrays["threshold13"].item()),
                layer_norm_eps=float(arrays["layer_norm_eps"].item()),
            )
            return cls(
                config,
                weight11=torch.from_numpy(arrays["weight11"].astype(np.float32)),
                bias11=float(arrays["bias11"].item()),
                weight13=torch.from_numpy(arrays["weight13"].astype(np.float32)),
                bias13=float(arrays["bias13"].item()),
            )


def save_router_npz(
    path: str | Path,
    *,
    head11: AffineBinaryHead,
    head13: AffineBinaryHead,
    threshold11: float,
    threshold13: float,
    layer_norm_eps: float = 1e-6,
    extra_arrays: Mapping[str, np.ndarray | float | int | str] | None = None,
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if head11.weight.shape != head13.weight.shape:
        raise ValueError("router heads must share one hidden dimension")
    payload: dict[str, np.ndarray | float | int | str] = {
        "schema_version": M425_ROUTER_SCHEMA_VERSION,
        "hidden_dim": head11.weight.size,
        "layer_norm_eps": float(layer_norm_eps),
        "weight11": head11.weight.astype(np.float32),
        "bias11": float(head11.bias),
        "weight13": head13.weight.astype(np.float32),
        "bias13": float(head13.bias),
        "threshold11": _validate_probability(threshold11, "threshold11"),
        "threshold13": _validate_probability(threshold13, "threshold13"),
    }
    if extra_arrays:
        overlap = set(payload) & set(extra_arrays)
        if overlap:
            raise ValueError(f"extra checkpoint arrays overlap required keys: {sorted(overlap)}")
        payload.update(extra_arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
