"""Strict grouped nested-OOF fitting for frozen V3-D7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .epistemic_ensemble import (
    D7_FITS_PER_OUTER,
    D7_HEAD_COUNT,
    ensemble_scores,
    head_fit_masks,
)
from .gripper_v2_models import LAMBDA_GRID, one_standard_error_choice
from .joint_reliability import (
    D5_EPISODES,
    D5_FALLBACK_LAYER,
    D5DevelopmentData,
    RouteSummary,
    route_at_threshold,
    select_inner_threshold,
    summarize_route,
)
from .joint_reliability_oof import route_summary_dict
from .severity_reliability import (
    D6_SAFETY_MULTIPLIER,
    SeverityWeightedFit,
    fit_severity_weighted_glm,
    severity_weights,
    weighted_task_cell_losses,
)
from .severity_reliability_oof import severity_fit_state


D7_OOF_SCHEMA_VERSION = "phase-route-vla.v3.d7-epistemic-nested-oof-fold.v1"


class D7OOFError(ValueError):
    """Raised when D7 nested-OOF partitioning or fitting fails closed."""


@dataclass(frozen=True)
class D7ThresholdSelection:
    feasible: bool
    full_threshold: float | None
    runtime_threshold: float | None
    full_summary: RouteSummary | None
    runtime_summary: RouteSummary | None
    evaluated_thresholds: int
    failure_reason: str | None


def fit_head_ensemble(
    data: D5DevelopmentData,
    row_severity: torch.Tensor,
    base_fit_mask: torch.Tensor,
    *,
    l2_lambda: float,
    max_iterations: int = 500,
) -> tuple[SeverityWeightedFit, ...]:
    masks = head_fit_masks(base_fit_mask, data.episode_index)
    models = tuple(
        fit_severity_weighted_glm(
            data.features,
            data.candidate_layer,
            data.unsafe_target,
            row_severity,
            mask,
            l2_lambda=l2_lambda,
            max_iterations=max_iterations,
        )
        for mask in masks
    )
    if len(models) != D7_HEAD_COUNT:
        raise D7OOFError("D7 fitted head count differs")
    return models


def predict_head_ensemble(
    models: tuple[SeverityWeightedFit, ...],
    features: torch.Tensor,
    candidate_layer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(models) != D7_HEAD_COUNT:
        raise D7OOFError("D7 prediction head count differs")
    prediction = torch.stack(
        [model.predict(features, candidate_layer) for model in models]
    ).contiguous()
    upper, gripper, head_range = ensemble_scores(prediction)
    combined = torch.stack((upper, gripper), dim=1).contiguous()
    return prediction, combined, head_range, upper


def select_d7_threshold(
    combined_score: torch.Tensor,
    action_consistency: torch.Tensor,
    unsafe_target: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
) -> D7ThresholdSelection:
    selected = select_inner_threshold(
        combined_score[:, 0],
        combined_score[:, 1],
        action_consistency,
        unsafe_target,
        task_id,
        episode_index,
    )
    if not selected.feasible or selected.threshold is None:
        return D7ThresholdSelection(
            False,
            None,
            None,
            selected.summary,
            None,
            selected.evaluated_thresholds,
            "full_inner_threshold_infeasible",
        )
    runtime_threshold = D6_SAFETY_MULTIPLIER * float(selected.threshold)
    runtime_layer = route_at_threshold(
        combined_score[:, 0],
        combined_score[:, 1],
        action_consistency,
        threshold=runtime_threshold,
    )
    runtime_summary = summarize_route(
        runtime_layer, unsafe_target, task_id, episode_index
    )
    return D7ThresholdSelection(
        feasible=runtime_summary.feasible,
        full_threshold=float(selected.threshold),
        runtime_threshold=runtime_threshold,
        full_summary=selected.summary,
        runtime_summary=runtime_summary,
        evaluated_thresholds=selected.evaluated_thresholds,
        failure_reason=None if runtime_summary.feasible else "shrunk_inner_route_infeasible",
    )


def threshold_selection_dict(selection: D7ThresholdSelection) -> dict[str, Any]:
    return {
        "feasible": selection.feasible,
        "full_threshold": selection.full_threshold,
        "runtime_threshold": selection.runtime_threshold,
        "fixed_safety_multiplier": D6_SAFETY_MULTIPLIER,
        "full_summary": route_summary_dict(selection.full_summary),
        "runtime_summary": route_summary_dict(selection.runtime_summary),
        "evaluated_thresholds": selection.evaluated_thresholds,
        "failure_reason": selection.failure_reason,
    }


def fit_outer_fold(
    data: D5DevelopmentData,
    full_action_distance: torch.Tensor,
    outer_episode: int,
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Fit one immutable D7 outer fold with five-head inner ensembles."""

    data.validate()
    if outer_episode not in D5_EPISODES:
        raise D7OOFError("D7 outer episode differs")
    if (
        full_action_distance.device.type != "cpu"
        or full_action_distance.shape != (data.rows,)
        or not full_action_distance.is_floating_point()
        or not bool(torch.isfinite(full_action_distance).all())
    ):
        raise D7OOFError("D7 full-action distance geometry differs")
    row_severity = severity_weights(full_action_distance)
    inner_episodes = tuple(value for value in D5_EPISODES if value != outer_episode)
    inner_predictions = {
        value: torch.full((data.rows, 2), float("nan"), dtype=torch.float64)
        for value in LAMBDA_GRID
    }
    inner_ranges = {
        value: torch.full((data.rows,), float("nan"), dtype=torch.float64)
        for value in LAMBDA_GRID
    }
    loss_store = {value: [] for value in LAMBDA_GRID}
    fit_count = 0
    for inner_episode in inner_episodes:
        fit_mask = (data.episode_index != outer_episode) & (
            data.episode_index != inner_episode
        )
        validation_mask = data.episode_index == inner_episode
        for value in LAMBDA_GRID:
            models = fit_head_ensemble(
                data,
                row_severity,
                fit_mask,
                l2_lambda=value,
                max_iterations=max_iterations,
            )
            _, combined, head_range, _ = predict_head_ensemble(
                models,
                data.features[validation_mask],
                data.candidate_layer[validation_mask],
            )
            inner_predictions[value][validation_mask] = combined
            inner_ranges[value][validation_mask] = head_range
            loss_store[value].append(
                weighted_task_cell_losses(
                    combined,
                    data.unsafe_target[validation_mask],
                    row_severity[validation_mask],
                    data.task_id[validation_mask],
                )
            )
            fit_count += D7_HEAD_COUNT

    outer_train = data.episode_index != outer_episode
    for value in LAMBDA_GRID:
        prediction = inner_predictions[value]
        head_range = inner_ranges[value]
        if (
            not bool(torch.isfinite(prediction[outer_train]).all())
            or bool(torch.isfinite(prediction[~outer_train]).any())
            or not bool(torch.isfinite(head_range[outer_train]).all())
            or bool(torch.isfinite(head_range[~outer_train]).any())
        ):
            raise D7OOFError("D7 inner OOF assignment differs")
    cells = {
        value: torch.cat(loss_store[value]).contiguous()
        for value in LAMBDA_GRID
    }
    if any(value.numel() != 170 for value in cells.values()):
        raise D7OOFError("D7 inner task-cell count differs")
    selected_lambda, lambda_summary = one_standard_error_choice(cells)
    selected_inner = inner_predictions[selected_lambda][outer_train].contiguous()
    selected_inner_range = inner_ranges[selected_lambda][outer_train].contiguous()
    threshold_selection = select_d7_threshold(
        selected_inner,
        data.action_consistency[outer_train],
        data.unsafe_target[outer_train],
        data.task_id[outer_train],
        data.episode_index[outer_train],
    )

    outer_models = fit_head_ensemble(
        data,
        row_severity,
        outer_train,
        l2_lambda=selected_lambda,
        max_iterations=max_iterations,
    )
    fit_count += D7_HEAD_COUNT
    validation_mask = ~outer_train
    validation_indices = torch.nonzero(validation_mask, as_tuple=False).flatten()
    head_prediction, combined, head_range, _ = predict_head_ensemble(
        outer_models,
        data.features[validation_mask],
        data.candidate_layer[validation_mask],
    )
    if threshold_selection.feasible:
        if threshold_selection.runtime_threshold is None:
            raise D7OOFError("D7 feasible threshold lacks runtime value")
        selected_layer = route_at_threshold(
            combined[:, 0],
            combined[:, 1],
            data.action_consistency[validation_mask],
            threshold=threshold_selection.runtime_threshold,
        )
    else:
        selected_layer = torch.full(
            (int(validation_mask.sum()) // 2,),
            D5_FALLBACK_LAYER,
            dtype=torch.long,
        )
    if fit_count != D7_FITS_PER_OUTER:
        raise D7OOFError("D7 outer fit count differs")
    outer_masks = head_fit_masks(outer_train, data.episode_index)
    return {
        "schema_version": D7_OOF_SCHEMA_VERSION,
        "outer_episode": outer_episode,
        "validation_indices": validation_indices.contiguous(),
        "validation_head_prediction": head_prediction.contiguous(),
        "validation_score": combined.contiguous(),
        "validation_full_head_range": head_range.contiguous(),
        "selected_layer": selected_layer.contiguous(),
        "selected_lambda": selected_lambda,
        "lambda_one_standard_error": lambda_summary,
        "threshold": threshold_selection_dict(threshold_selection),
        "selected_inner_full_head_range": selected_inner_range,
        "outer_model_states": [severity_fit_state(model) for model in outer_models],
        "outer_head_fit_rows": [int(mask.sum()) for mask in outer_masks],
        "fit_count": fit_count,
    }


__all__ = [
    "D7_OOF_SCHEMA_VERSION",
    "D7OOFError",
    "D7ThresholdSelection",
    "fit_head_ensemble",
    "fit_outer_fold",
    "predict_head_ensemble",
    "select_d7_threshold",
    "threshold_selection_dict",
]
