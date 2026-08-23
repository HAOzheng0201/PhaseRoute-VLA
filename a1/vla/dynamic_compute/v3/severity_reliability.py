"""Severity-aware, threshold-robust primitives for frozen V3-D6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .gripper_v2_models import FeatureNormalizer, LAMBDA_GRID, fit_normalizer
from .gripper_v2_protocol import FEATURE_DIMENSION
from .joint_reliability import (
    D5_ACTION_THRESHOLD,
    D5_CANDIDATE_LAYERS,
    RouteSummary,
    route_at_threshold,
    select_inner_threshold,
    summarize_route,
)


D6_SCHEMA_VERSION = "phase-route-vla.v3.d6-repair-contract.v1"
D6_STATUS = "D6_DEVELOPMENT_REPAIR_CONTRACT_FROZEN"
D6_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/joint_reliability/d6_repair_contract.json"
)
D6_CONTRACT_SHA256 = (
    "28185ce5431cf438d20cb7cfdfd0e20d5859b6a99f1bdafa81d18faef59fd7a1"
)
D6_SAFETY_MULTIPLIER = 0.95
D6_JACKKNIFE_ORDER_INDEX = 4
D6_MAX_SEVERITY_WEIGHT = 5.0


class D6SeverityError(ValueError):
    """Raised when D6 weights, models, or robust thresholds fail closed."""


@dataclass(frozen=True)
class SeverityWeightedFit:
    normalizer: FeatureNormalizer
    anchor_score: torch.Tensor
    weight: torch.Tensor
    l2_lambda: float
    final_loss: float

    def predict(
        self, features: torch.Tensor, candidate_layer: torch.Tensor
    ) -> torch.Tensor:
        values = self.normalizer.transform(features)
        layer_index = _layer_indices(candidate_layer, values.shape[0])
        anchor = self.anchor_score[layer_index]
        logits = _logit(anchor) + values @ self.weight.T
        score = torch.sigmoid(logits)
        if not bool(torch.isfinite(score).all()) or not bool(
            ((score > 0.0) & (score < 1.0)).all()
        ):
            raise D6SeverityError("D6 predicted score is invalid")
        return score.contiguous()


@dataclass(frozen=True)
class RobustThresholdSelection:
    feasible: bool
    full_threshold: float | None
    jackknife_thresholds: tuple[tuple[int, float], ...]
    order_statistic_threshold: float | None
    pre_shrink_threshold: float | None
    runtime_threshold: float | None
    runtime_summary: RouteSummary | None
    failure_reason: str | None


def severity_weights(full_action_distance: torch.Tensor) -> torch.Tensor:
    """Return frozen 1--5x log2 severity weights for flat candidate rows."""

    if (
        not isinstance(full_action_distance, torch.Tensor)
        or full_action_distance.device.type != "cpu"
        or full_action_distance.ndim != 1
        or not full_action_distance.is_floating_point()
        or not bool(torch.isfinite(full_action_distance).all())
    ):
        raise D6SeverityError("D6 full-action distance must be finite CPU [N]")
    distance = full_action_distance.double().clamp_min(0.0)
    ratio = (distance / D5_ACTION_THRESHOLD).clamp_min(1.0)
    value = 1.0 + torch.log2(ratio).clamp(min=0.0, max=4.0)
    if not bool(torch.isfinite(value).all()) or not bool(
        ((value >= 1.0) & (value <= D6_MAX_SEVERITY_WEIGHT)).all()
    ):
        raise D6SeverityError("D6 severity weight lies outside [1,5]")
    return value.contiguous()


def _layer_indices(candidate_layer: torch.Tensor, rows: int) -> torch.Tensor:
    if (
        candidate_layer.device.type != "cpu"
        or candidate_layer.dtype != torch.long
        or candidate_layer.shape != (rows,)
        or not bool(
            ((candidate_layer == D5_CANDIDATE_LAYERS[0]) | (candidate_layer == D5_CANDIDATE_LAYERS[1])).all()
        )
    ):
        raise D6SeverityError("D6 candidate layer must be CPU int64 L11/L13")
    return (candidate_layer == D5_CANDIDATE_LAYERS[1]).long()


def _fit_mask(mask: torch.Tensor, rows: int) -> torch.Tensor:
    if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != (rows,):
        raise D6SeverityError("D6 fit mask must be CPU bool [N]")
    if int(mask.sum()) < 4:
        raise D6SeverityError("D6 fit partition is too small")
    return mask


def _logit(value: torch.Tensor) -> torch.Tensor:
    if not bool(((value > 0.0) & (value < 1.0)).all()):
        raise D6SeverityError("D6 anchor must lie inside (0,1)")
    return torch.log(value) - torch.log1p(-value)


def _anchors(
    target: torch.Tensor,
    layer_index: torch.Tensor,
    fit_mask: torch.Tensor,
    severity_weight: torch.Tensor,
) -> torch.Tensor:
    anchors = torch.empty((2, 2), dtype=torch.float64)
    for layer in range(2):
        selected = fit_mask & (layer_index == layer)
        if int(selected.sum()) < 2:
            raise D6SeverityError("D6 layer anchor has insufficient support")
        full_weight = severity_weight[selected]
        anchors[layer, 0] = (
            full_weight * target[selected, 0].double()
        ).sum() / full_weight.sum()
        anchors[layer, 1] = target[selected, 1].double().mean()
    _logit(anchors)
    return anchors.contiguous()


def _run_lbfgs(
    parameters: list[torch.nn.Parameter],
    objective: Callable[[], torch.Tensor],
    *,
    max_iterations: int,
) -> float:
    if max_iterations < 1:
        raise D6SeverityError("D6 max iterations must be positive")
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
            raise D6SeverityError("D6 LBFGS objective is non-finite")
        loss.backward()
        return loss

    optimizer.step(closure)
    final = objective().detach()
    if not bool(torch.isfinite(final)):
        raise D6SeverityError("D6 final loss is non-finite")
    return float(final)


def fit_severity_weighted_glm(
    features: torch.Tensor,
    candidate_layer: torch.Tensor,
    unsafe_target: torch.Tensor,
    severity_weight: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    l2_lambda: float,
    max_iterations: int = 500,
) -> SeverityWeightedFit:
    rows = int(features.shape[0])
    selected = _fit_mask(fit_mask, rows)
    layer_index = _layer_indices(candidate_layer, rows)
    if (
        features.device.type != "cpu"
        or features.shape != (rows, FEATURE_DIMENSION)
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
        or unsafe_target.device.type != "cpu"
        or unsafe_target.dtype != torch.bool
        or unsafe_target.shape != (rows, 2)
        or severity_weight.device.type != "cpu"
        or severity_weight.shape != (rows,)
        or not severity_weight.is_floating_point()
        or not bool(torch.isfinite(severity_weight).all())
        or not bool(((severity_weight >= 1.0) & (severity_weight <= 5.0)).all())
    ):
        raise D6SeverityError("D6 weighted GLM input geometry differs")
    if float(l2_lambda) not in LAMBDA_GRID:
        raise D6SeverityError("D6 lambda is outside the frozen grid")
    normalizer = fit_normalizer(features, selected)
    normalized = normalizer.transform(features)[selected]
    train_layer = layer_index[selected]
    train_target = unsafe_target[selected].double()
    train_severity = severity_weight[selected].double()
    anchor = _anchors(unsafe_target, layer_index, selected, severity_weight.double())
    weight = torch.nn.Parameter(torch.zeros((2, FEATURE_DIMENSION), dtype=torch.float64))

    def objective() -> torch.Tensor:
        logits = _logit(anchor[train_layer]) + normalized @ weight.T
        full_row = F.binary_cross_entropy_with_logits(
            logits[:, 0], train_target[:, 0], reduction="none"
        )
        full_loss = (train_severity * full_row).sum() / train_severity.sum()
        gripper_loss = F.binary_cross_entropy_with_logits(
            logits[:, 1], train_target[:, 1]
        )
        data_loss = 0.5 * (full_loss + gripper_loss)
        return data_loss + 0.5 * float(l2_lambda) * weight.square().sum()

    final_loss = _run_lbfgs([weight], objective, max_iterations=max_iterations)
    return SeverityWeightedFit(
        normalizer=normalizer,
        anchor_score=anchor.detach().contiguous(),
        weight=weight.detach().contiguous(),
        l2_lambda=float(l2_lambda),
        final_loss=final_loss,
    )


def weighted_task_cell_losses(
    score: torch.Tensor,
    target: torch.Tensor,
    severity_weight: torch.Tensor,
    task_id: torch.Tensor,
) -> torch.Tensor:
    rows = int(score.shape[0])
    if (
        score.device.type != "cpu"
        or score.shape != (rows, 2)
        or not score.is_floating_point()
        or not bool(torch.isfinite(score).all())
        or not bool(((score > 0.0) & (score < 1.0)).all())
        or target.device.type != "cpu"
        or target.dtype != torch.bool
        or target.shape != (rows, 2)
        or severity_weight.device.type != "cpu"
        or severity_weight.shape != (rows,)
        or task_id.device.type != "cpu"
        or task_id.dtype != torch.long
        or task_id.shape != (rows,)
    ):
        raise D6SeverityError("D6 weighted task-cell geometry differs")
    full_nll = F.binary_cross_entropy(
        score[:, 0].double(), target[:, 0].double(), reduction="none"
    )
    gripper_nll = F.binary_cross_entropy(
        score[:, 1].double(), target[:, 1].double(), reduction="none"
    )
    cells = []
    for task in range(10):
        selected = task_id == task
        if not bool(selected.any()):
            raise D6SeverityError("D6 weighted task cell is empty")
        full = (
            severity_weight[selected].double() * full_nll[selected]
        ).sum() / severity_weight[selected].double().sum()
        cells.append(0.5 * (full + gripper_nll[selected].mean()))
    result = torch.stack(cells).contiguous()
    if result.shape != (10,) or not bool(torch.isfinite(result).all()):
        raise D6SeverityError("D6 weighted task-cell loss is invalid")
    return result


def robust_threshold_selection(
    full_action_score: torch.Tensor,
    gripper_score: torch.Tensor,
    action_consistency: torch.Tensor,
    unsafe_target: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
) -> RobustThresholdSelection:
    episodes = tuple(sorted(set(int(value) for value in episode_index.tolist())))
    if len(episodes) != 17:
        raise D6SeverityError("D6 robust threshold requires exactly 17 episodes")
    full = select_inner_threshold(
        full_action_score,
        gripper_score,
        action_consistency,
        unsafe_target,
        task_id,
        episode_index,
    )
    if not full.feasible or full.threshold is None:
        return RobustThresholdSelection(
            False, None, (), None, None, None, None, "full_inner_infeasible"
        )
    jackknife: list[tuple[int, float]] = []
    for dropped in episodes:
        mask = episode_index != dropped
        view = select_inner_threshold(
            full_action_score[mask],
            gripper_score[mask],
            action_consistency[mask],
            unsafe_target[mask],
            task_id[mask],
            episode_index[mask],
        )
        if not view.feasible or view.threshold is None:
            return RobustThresholdSelection(
                False,
                float(full.threshold),
                tuple(jackknife),
                None,
                None,
                None,
                None,
                f"jackknife_drop_episode_{dropped}_infeasible",
            )
        jackknife.append((dropped, float(view.threshold)))
    ordered = sorted(value for _, value in jackknife)
    order_statistic = ordered[D6_JACKKNIFE_ORDER_INDEX]
    pre_shrink = min(float(full.threshold), order_statistic)
    runtime = D6_SAFETY_MULTIPLIER * pre_shrink
    selected = route_at_threshold(
        full_action_score,
        gripper_score,
        action_consistency,
        threshold=runtime,
    )
    summary = summarize_route(selected, unsafe_target, task_id, episode_index)
    return RobustThresholdSelection(
        feasible=summary.feasible,
        full_threshold=float(full.threshold),
        jackknife_thresholds=tuple(jackknife),
        order_statistic_threshold=order_statistic,
        pre_shrink_threshold=pre_shrink,
        runtime_threshold=runtime,
        runtime_summary=summary,
        failure_reason=None if summary.feasible else "shrunk_runtime_infeasible",
    )


def load_d6_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D6_CONTRACT_RELATIVE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != D6_CONTRACT_SHA256:
        raise D6SeverityError("D6 frozen contract SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D6SeverityError("D6 contract cannot be read") from error
    if not isinstance(value, Mapping):
        raise D6SeverityError("D6 contract must be an object")
    contract = dict(value)
    if (
        contract.get("schema_version") != D6_SCHEMA_VERSION
        or contract.get("status") != D6_STATUS
        or contract.get("scope", {}).get("fresh_confirmation_claim_allowed") is not False
        or contract.get("severity_weight", {}).get("range") != [1.0, 5.0]
        or contract.get("robust_threshold", {}).get("fixed_safety_multiplier")
        != D6_SAFETY_MULTIPLIER
        or contract.get("robust_threshold", {}).get("jackknife_order_statistic")
        != "fifth_smallest_of_17_feasible_thresholds"
        or contract.get("authorization", {}).get("independent_test_authorized")
        is not False
    ):
        raise D6SeverityError("D6 frozen contract semantics differ")
    return contract


__all__ = [
    "D6_CONTRACT_RELATIVE_PATH",
    "D6_CONTRACT_SHA256",
    "D6_JACKKNIFE_ORDER_INDEX",
    "D6_MAX_SEVERITY_WEIGHT",
    "D6_SAFETY_MULTIPLIER",
    "D6SeverityError",
    "RobustThresholdSelection",
    "SeverityWeightedFit",
    "fit_severity_weighted_glm",
    "load_d6_contract",
    "robust_threshold_selection",
    "severity_weights",
    "weighted_task_cell_losses",
]
