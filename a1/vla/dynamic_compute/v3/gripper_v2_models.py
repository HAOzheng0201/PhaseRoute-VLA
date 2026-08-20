"""Deterministic CPU GLMs for the frozen V3-D1 Gripper-v2 protocol.

This module implements exactly three model families: anchored Bernoulli
logistic occurrence, zero-truncated binomial conditional count, and ordinal
cumulative-link conditional count.  Every feature head is linear and has no
free bias.  Layer-specific fold-train anchors (or ordinal cutpoints) are kept
structurally separate from the 97-D residual weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn.functional as F

from .gripper_v2_protocol import DECISION_LAYERS, FEATURE_DIMENSION


LAMBDA_GRID = (1.0e-3, 1.0e-2, 1.0e-1)
COUNT_SUPPORT_MAX = (8, 7)
TARGET_NAMES = ("step", "transition")
_EPSILON = 1.0e-12


class GripperV2ModelError(ValueError):
    """Raised when a GLM fit or prediction violates the frozen contract."""


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        values = _feature_matrix(features)
        if values.shape[1] != self.mean.numel():
            raise GripperV2ModelError("normalizer feature dimension differs")
        normalized = (values - self.mean) / self.scale
        if not bool(torch.isfinite(normalized).all()):
            raise GripperV2ModelError("normalized features are non-finite")
        return normalized


@dataclass(frozen=True)
class OccurrenceFit:
    normalizer: FeatureNormalizer
    anchor_probability: torch.Tensor
    weight: torch.Tensor
    l2_lambda: float
    final_loss: float

    def predict(self, features: torch.Tensor, candidate_layer: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer.transform(features)
        layer_index = _layer_indices(candidate_layer, rows=normalized.shape[0])
        anchor = self.anchor_probability[layer_index]
        logits = _logit(anchor) + normalized @ self.weight.T
        probability = torch.sigmoid(logits)
        _finite_probability(probability, name="occurrence")
        return probability


@dataclass(frozen=True)
class ZTBinomialFit:
    target_index: int
    support_max: int
    normalizer: FeatureNormalizer
    anchor_probability: torch.Tensor
    weight: torch.Tensor
    l2_lambda: float
    final_loss: float

    def probabilities(
        self, features: torch.Tensor, candidate_layer: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.normalizer.transform(features)
        layer_index = _layer_indices(candidate_layer, rows=normalized.shape[0])
        anchor = self.anchor_probability[layer_index]
        probability = torch.sigmoid(_logit(anchor) + normalized @ self.weight)
        return zero_truncated_binomial_probabilities(probability, self.support_max)


@dataclass(frozen=True)
class OrdinalFit:
    target_index: int
    support_max: int
    normalizer: FeatureNormalizer
    weight: torch.Tensor
    raw_base: torch.Tensor
    raw_increments: torch.Tensor
    l2_lambda: float
    final_loss: float

    @property
    def cutpoints(self) -> torch.Tensor:
        return ordered_cutpoints(self.raw_base, self.raw_increments)

    def probabilities(
        self, features: torch.Tensor, candidate_layer: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.normalizer.transform(features)
        layer_index = _layer_indices(candidate_layer, rows=normalized.shape[0])
        score = normalized @ self.weight
        return ordinal_probabilities(score, layer_index, self.cutpoints)


def _feature_matrix(features: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or features.shape[1] != FEATURE_DIMENSION
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
    ):
        raise GripperV2ModelError("features must be finite [N,97]")
    return features.detach().cpu().to(torch.float64).contiguous()


def _bool_vector(mask: torch.Tensor, *, rows: int, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (
        rows,
    ):
        raise GripperV2ModelError(f"{name} must be bool [N]")
    return mask.detach().cpu().contiguous()


def _layer_indices(candidate_layer: torch.Tensor, *, rows: int) -> torch.Tensor:
    if (
        not isinstance(candidate_layer, torch.Tensor)
        or candidate_layer.dtype != torch.long
        or candidate_layer.shape != (rows,)
    ):
        raise GripperV2ModelError("candidate layer must be int64 [N]")
    values = candidate_layer.detach().cpu()
    if not bool(((values == 11) | (values == 13)).all()):
        raise GripperV2ModelError("candidate layer must contain only 11 or 13")
    return (values == DECISION_LAYERS[1]).to(torch.long)


def fit_normalizer(features: torch.Tensor, fit_mask: torch.Tensor) -> FeatureNormalizer:
    values = _feature_matrix(features)
    selected_mask = _bool_vector(fit_mask, rows=values.shape[0], name="fit mask")
    if int(selected_mask.sum()) < 2:
        raise GripperV2ModelError("normalizer fit partition is too small")
    selected = values[selected_mask]
    mean = selected.mean(dim=0)
    scale = selected.std(dim=0, unbiased=False)
    scale = torch.where(scale >= 1.0e-8, scale, torch.ones_like(scale))
    return FeatureNormalizer(mean=mean, scale=scale)


def _logit(probability: torch.Tensor) -> torch.Tensor:
    if not bool(((probability > 0.0) & (probability < 1.0)).all()):
        raise GripperV2ModelError("anchor probability must lie inside (0,1)")
    return torch.log(probability) - torch.log1p(-probability)


def _finite_probability(probability: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(probability).all()) or not bool(
        ((probability > 0.0) & (probability < 1.0)).all()
    ):
        raise GripperV2ModelError(f"{name} probability is invalid")


def _layer_binary_anchor(
    target: torch.Tensor, layer_index: torch.Tensor, fit_mask: torch.Tensor
) -> torch.Tensor:
    anchors = torch.empty((2, target.shape[1]), dtype=torch.float64)
    for layer in range(2):
        selected = fit_mask & (layer_index == layer)
        if int(selected.sum()) < 2:
            raise GripperV2ModelError("occurrence anchor layer has insufficient rows")
        anchors[layer] = target[selected].double().mean(dim=0)
    _finite_probability(anchors, name="occurrence anchor")
    return anchors


def _run_lbfgs(
    parameters: list[torch.nn.Parameter],
    objective: Callable[[], torch.Tensor],
    *,
    max_iterations: int,
) -> float:
    if max_iterations < 1:
        raise GripperV2ModelError("LBFGS max iterations must be positive")
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=max_iterations,
        history_size=100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise GripperV2ModelError("LBFGS objective is non-finite")
        loss.backward()
        return loss

    optimizer.step(closure)
    final = objective().detach()
    if not bool(torch.isfinite(final)):
        raise GripperV2ModelError("LBFGS final loss is non-finite")
    return float(final)


def fit_occurrence_glm(
    features: torch.Tensor,
    candidate_layer: torch.Tensor,
    occurrence: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    l2_lambda: float,
    max_iterations: int = 500,
) -> OccurrenceFit:
    values = _feature_matrix(features)
    rows = values.shape[0]
    selected = _bool_vector(fit_mask, rows=rows, name="fit mask")
    layer_index = _layer_indices(candidate_layer, rows=rows)
    if (
        not isinstance(occurrence, torch.Tensor)
        or occurrence.dtype != torch.bool
        or occurrence.shape != (rows, 2)
    ):
        raise GripperV2ModelError("occurrence target must be bool [N,2]")
    if float(l2_lambda) not in LAMBDA_GRID:
        raise GripperV2ModelError("occurrence lambda is outside frozen grid")
    target = occurrence.detach().cpu()
    normalizer = fit_normalizer(values, selected)
    normalized = normalizer.transform(values)[selected]
    train_layer = layer_index[selected]
    train_target = target[selected].double()
    anchor = _layer_binary_anchor(target, layer_index, selected)
    weight = torch.nn.Parameter(
        torch.zeros((2, FEATURE_DIMENSION), dtype=torch.float64)
    )

    def objective() -> torch.Tensor:
        logits = _logit(anchor[train_layer]) + normalized @ weight.T
        data_loss = F.binary_cross_entropy_with_logits(logits, train_target)
        return data_loss + 0.5 * float(l2_lambda) * weight.square().sum()

    final_loss = _run_lbfgs([weight], objective, max_iterations=max_iterations)
    return OccurrenceFit(
        normalizer=normalizer,
        anchor_probability=anchor.detach().contiguous(),
        weight=weight.detach().contiguous(),
        l2_lambda=float(l2_lambda),
        final_loss=final_loss,
    )


def _zt_expected_count(probability: float, support_max: int) -> float:
    return support_max * probability / (1.0 - (1.0 - probability) ** support_max)


def zt_binomial_anchor_probability(mean_count: float, support_max: int) -> float:
    if not 1.0 <= float(mean_count) <= float(support_max):
        raise GripperV2ModelError("positive count mean is outside support")
    if mean_count >= support_max - 1.0e-12:
        return 1.0 - 1.0e-8
    low, high = 1.0e-8, 1.0 - 1.0e-8
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _zt_expected_count(midpoint, support_max) < mean_count:
            low = midpoint
        else:
            high = midpoint
    value = (low + high) / 2.0
    if not 0.0 < value < 1.0:
        raise GripperV2ModelError("ZT-binomial anchor inversion failed")
    return value


def zero_truncated_binomial_probabilities(
    probability: torch.Tensor, support_max: int
) -> torch.Tensor:
    if support_max not in COUNT_SUPPORT_MAX:
        raise GripperV2ModelError("ZT-binomial support must be 8 or 7")
    _finite_probability(probability, name="ZT-binomial")
    counts = torch.arange(
        1, support_max + 1, dtype=probability.dtype, device=probability.device
    )
    log_combination = (
        torch.lgamma(torch.tensor(support_max + 1.0, dtype=probability.dtype))
        - torch.lgamma(counts + 1.0)
        - torch.lgamma(support_max - counts + 1.0)
    )
    log_pmf = (
        log_combination
        + counts * torch.log(probability[:, None])
        + (support_max - counts) * torch.log1p(-probability[:, None])
    )
    log_normalizer = torch.log1p(-(1.0 - probability).pow(support_max))
    values = torch.exp(log_pmf - log_normalizer[:, None])
    values = values / values.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(values).all()) or not torch.allclose(
        values.sum(dim=1),
        torch.ones(values.shape[0], dtype=values.dtype),
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise GripperV2ModelError("ZT-binomial probabilities are invalid")
    return values


def fit_zt_binomial_glm(
    features: torch.Tensor,
    candidate_layer: torch.Tensor,
    count: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    target_index: int,
    l2_lambda: float,
    max_iterations: int = 500,
) -> ZTBinomialFit:
    values = _feature_matrix(features)
    rows = values.shape[0]
    base_mask = _bool_vector(fit_mask, rows=rows, name="fit mask")
    layer_index = _layer_indices(candidate_layer, rows=rows)
    if (
        not isinstance(count, torch.Tensor)
        or count.dtype != torch.long
        or count.shape != (rows, 2)
        or target_index not in (0, 1)
    ):
        raise GripperV2ModelError("count target must be int64 [N,2]")
    if float(l2_lambda) not in LAMBDA_GRID:
        raise GripperV2ModelError("ZT-binomial lambda is outside frozen grid")
    support_max = COUNT_SUPPORT_MAX[target_index]
    target = count[:, target_index].detach().cpu()
    selected = base_mask & (target > 0)
    if int(selected.sum()) < 4:
        raise GripperV2ModelError("ZT-binomial positive fit support is too small")
    normalizer = fit_normalizer(values, selected)
    normalized = normalizer.transform(values)[selected]
    train_layer = layer_index[selected]
    train_target = target[selected]
    anchor = torch.empty(2, dtype=torch.float64)
    for layer in range(2):
        layer_target = target[selected & (layer_index == layer)]
        if layer_target.numel() < 2:
            raise GripperV2ModelError("ZT-binomial layer support is too small")
        anchor[layer] = zt_binomial_anchor_probability(
            float(layer_target.double().mean()), support_max
        )
    weight = torch.nn.Parameter(torch.zeros(FEATURE_DIMENSION, dtype=torch.float64))

    def objective() -> torch.Tensor:
        probability = torch.sigmoid(_logit(anchor[train_layer]) + normalized @ weight)
        probabilities = zero_truncated_binomial_probabilities(
            probability, support_max
        )
        selected_probability = probabilities[
            torch.arange(train_target.numel()), train_target - 1
        ]
        nll = -torch.log(selected_probability.clamp_min(_EPSILON)).mean()
        return nll + 0.5 * float(l2_lambda) * weight.square().sum()

    final_loss = _run_lbfgs([weight], objective, max_iterations=max_iterations)
    return ZTBinomialFit(
        target_index=target_index,
        support_max=support_max,
        normalizer=normalizer,
        anchor_probability=anchor.detach().contiguous(),
        weight=weight.detach().contiguous(),
        l2_lambda=float(l2_lambda),
        final_loss=final_loss,
    )


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    if not bool((value > 0.0).all()):
        raise GripperV2ModelError("ordinal cutpoint increment must be positive")
    return value + torch.log(-torch.expm1(-value))


def initial_ordinal_cutpoints(
    count: torch.Tensor,
    layer_index: torch.Tensor,
    fit_mask: torch.Tensor,
    support_max: int,
) -> torch.Tensor:
    cutpoints = torch.empty((2, support_max - 1), dtype=torch.float64)
    for layer in range(2):
        selected = count[fit_mask & (layer_index == layer)]
        if selected.numel() < 2 or bool((selected <= 0).any()):
            raise GripperV2ModelError("ordinal layer positive support is too small")
        # Fixed Jeffreys 0.5 pseudo-counts affect initialization only; all
        # cutpoints remain trainable in the fold-train likelihood.
        histogram = torch.bincount(
            selected, minlength=support_max + 1
        )[1 : support_max + 1].double() + 0.5
        cdf = histogram.cumsum(dim=0)[:-1] / histogram.sum()
        cutpoints[layer] = _logit(cdf)
    if not bool((cutpoints[:, 1:] > cutpoints[:, :-1]).all()):
        raise GripperV2ModelError("initial ordinal cutpoints are not increasing")
    return cutpoints


def ordered_cutpoints(
    raw_base: torch.Tensor, raw_increments: torch.Tensor
) -> torch.Tensor:
    if (
        raw_base.shape != (2,)
        or raw_increments.ndim != 2
        or raw_increments.shape[0] != 2
    ):
        raise GripperV2ModelError("ordinal raw cutpoint geometry differs")
    increments = F.softplus(raw_increments)
    values = torch.cat(
        (raw_base[:, None], raw_base[:, None] + increments.cumsum(dim=1)), dim=1
    )
    if not bool(torch.isfinite(values).all()) or not bool(
        (values[:, 1:] > values[:, :-1]).all()
    ):
        raise GripperV2ModelError("ordinal cutpoints are not strictly increasing")
    return values


def ordinal_probabilities(
    score: torch.Tensor, layer_index: torch.Tensor, cutpoints: torch.Tensor
) -> torch.Tensor:
    if score.ndim != 1 or layer_index.shape != score.shape:
        raise GripperV2ModelError("ordinal score/layer geometry differs")
    if cutpoints.ndim != 2 or cutpoints.shape[0] != 2:
        raise GripperV2ModelError("ordinal cutpoint geometry differs")
    row_cutpoints = cutpoints[layer_index]
    shifted = row_cutpoints - score[:, None]
    lower = shifted[:, :-1]
    upper = shifted[:, 1:]
    difference = upper - lower
    middle_log_probability = (
        lower
        + torch.log(torch.expm1(difference))
        - F.softplus(lower)
        - F.softplus(upper)
    )
    log_probability = torch.cat(
        (
            F.logsigmoid(shifted[:, :1]),
            middle_log_probability,
            F.logsigmoid(-shifted[:, -1:]),
        ),
        dim=1,
    )
    log_probability = log_probability - torch.logsumexp(
        log_probability, dim=1, keepdim=True
    )
    probabilities = torch.exp(log_probability).clamp_min(
        torch.finfo(log_probability.dtype).tiny
    )
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    if bool((probabilities <= 0.0).any()) or not bool(
        torch.isfinite(probabilities).all()
    ):
        raise GripperV2ModelError("ordinal probabilities are invalid")
    return probabilities


def fit_ordinal_glm(
    features: torch.Tensor,
    candidate_layer: torch.Tensor,
    count: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    target_index: int,
    l2_lambda: float,
    max_iterations: int = 500,
) -> OrdinalFit:
    values = _feature_matrix(features)
    rows = values.shape[0]
    base_mask = _bool_vector(fit_mask, rows=rows, name="fit mask")
    layer_index = _layer_indices(candidate_layer, rows=rows)
    if (
        not isinstance(count, torch.Tensor)
        or count.dtype != torch.long
        or count.shape != (rows, 2)
        or target_index not in (0, 1)
    ):
        raise GripperV2ModelError("count target must be int64 [N,2]")
    if float(l2_lambda) not in LAMBDA_GRID:
        raise GripperV2ModelError("ordinal lambda is outside frozen grid")
    support_max = COUNT_SUPPORT_MAX[target_index]
    target = count[:, target_index].detach().cpu()
    selected = base_mask & (target > 0)
    if int(selected.sum()) < 4:
        raise GripperV2ModelError("ordinal positive fit support is too small")
    normalizer = fit_normalizer(values, selected)
    normalized = normalizer.transform(values)[selected]
    train_layer = layer_index[selected]
    train_target = target[selected]
    initial = initial_ordinal_cutpoints(
        target, layer_index, selected, support_max
    )
    weight = torch.nn.Parameter(torch.zeros(FEATURE_DIMENSION, dtype=torch.float64))
    raw_base = torch.nn.Parameter(initial[:, 0].clone())
    raw_increments = torch.nn.Parameter(
        _inverse_softplus(initial[:, 1:] - initial[:, :-1])
    )

    def objective() -> torch.Tensor:
        probability = ordinal_probabilities(
            normalized @ weight,
            train_layer,
            ordered_cutpoints(raw_base, raw_increments),
        )
        selected_probability = probability[
            torch.arange(train_target.numel()), train_target - 1
        ]
        nll = -torch.log(selected_probability.clamp_min(_EPSILON)).mean()
        return nll + 0.5 * float(l2_lambda) * weight.square().sum()

    final_loss = _run_lbfgs(
        [weight, raw_base, raw_increments],
        objective,
        max_iterations=max_iterations,
    )
    return OrdinalFit(
        target_index=target_index,
        support_max=support_max,
        normalizer=normalizer,
        weight=weight.detach().contiguous(),
        raw_base=raw_base.detach().contiguous(),
        raw_increments=raw_increments.detach().contiguous(),
        l2_lambda=float(l2_lambda),
        final_loss=final_loss,
    )


def conditional_nll(probabilities: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2 or count.shape != (probabilities.shape[0],):
        raise GripperV2ModelError("conditional NLL geometry differs")
    if count.dtype != torch.long or bool((count <= 0).any()) or bool(
        (count > probabilities.shape[1]).any()
    ):
        raise GripperV2ModelError("conditional count is outside positive support")
    selected = probabilities[torch.arange(count.numel()), count - 1]
    return -torch.log(selected.clamp_min(_EPSILON))


def expected_positive_count(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise GripperV2ModelError("count probabilities must be a matrix")
    support = torch.arange(
        1,
        probabilities.shape[1] + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return probabilities @ support


def ranked_probability_score(
    probabilities: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    if probabilities.ndim != 2 or count.shape != (probabilities.shape[0],):
        raise GripperV2ModelError("RPS geometry differs")
    cumulative = probabilities.cumsum(dim=1)[:, :-1]
    thresholds = torch.arange(
        1, probabilities.shape[1], dtype=count.dtype, device=count.device
    )
    observed = (count[:, None] <= thresholds[None]).to(probabilities.dtype)
    return (cumulative - observed).square().sum(dim=1) / (
        probabilities.shape[1] - 1
    )


def tie_aware_auroc(score: torch.Tensor, target: torch.Tensor) -> float:
    if score.ndim != 1 or target.shape != score.shape or target.dtype != torch.bool:
        raise GripperV2ModelError("AUROC input geometry differs")
    positive = score[target].double()
    negative = score[~target].double()
    if positive.numel() == 0 or negative.numel() == 0:
        raise GripperV2ModelError("AUROC requires both classes")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).double() + 0.5 * (comparisons == 0).double()).mean())


def one_standard_error_choice(
    cell_losses: dict[float, torch.Tensor],
) -> tuple[float, dict[str, dict[str, float]]]:
    if set(cell_losses) != set(LAMBDA_GRID):
        raise GripperV2ModelError("inner loss lambda grid differs")
    summary: dict[str, dict[str, float]] = {}
    for value in LAMBDA_GRID:
        losses = cell_losses[value].detach().cpu().double()
        if losses.ndim != 1 or losses.numel() < 2 or not bool(
            torch.isfinite(losses).all()
        ):
            raise GripperV2ModelError("inner cell losses are invalid")
        summary[str(value)] = {
            "mean": float(losses.mean()),
            "se": float(losses.std(unbiased=True) / math.sqrt(losses.numel())),
            "cells": int(losses.numel()),
        }
    minimum = min(LAMBDA_GRID, key=lambda value: (summary[str(value)]["mean"], value))
    threshold = summary[str(minimum)]["mean"] + summary[str(minimum)]["se"]
    eligible = [
        value for value in LAMBDA_GRID if summary[str(value)]["mean"] <= threshold
    ]
    return max(eligible), summary


__all__ = [
    "COUNT_SUPPORT_MAX",
    "FeatureNormalizer",
    "GripperV2ModelError",
    "LAMBDA_GRID",
    "OccurrenceFit",
    "OrdinalFit",
    "TARGET_NAMES",
    "ZTBinomialFit",
    "conditional_nll",
    "expected_positive_count",
    "fit_normalizer",
    "fit_occurrence_glm",
    "fit_ordinal_glm",
    "fit_zt_binomial_glm",
    "initial_ordinal_cutpoints",
    "one_standard_error_choice",
    "ordered_cutpoints",
    "ordinal_probabilities",
    "ranked_probability_score",
    "tie_aware_auroc",
    "zero_truncated_binomial_probabilities",
    "zt_binomial_anchor_probability",
]
