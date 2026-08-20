"""Strict grouped nested-OOF fitting for frozen V3-D6 development selection."""

from __future__ import annotations

from typing import Any

import torch

from .gripper_v2_models import LAMBDA_GRID, one_standard_error_choice
from .joint_reliability import (
    D5_EPISODES,
    D5_FALLBACK_LAYER,
    D5DevelopmentData,
    route_at_threshold,
)
from .joint_reliability_oof import route_summary_dict
from .severity_reliability import (
    RobustThresholdSelection,
    fit_severity_weighted_glm,
    robust_threshold_selection,
    severity_weights,
    weighted_task_cell_losses,
)


D6_OOF_SCHEMA_VERSION = "phase-route-vla.v3.d6-severity-nested-oof-fold.v1"
D6_FITS_PER_OUTER = len(D5_EPISODES[:-1]) * len(LAMBDA_GRID) + 1


class D6OOFError(ValueError):
    """Raised when D6 nested-OOF partitioning or fitting fails closed."""


def severity_fit_state(model: Any) -> dict[str, Any]:
    return {
        "normalizer_mean": model.normalizer.mean.contiguous(),
        "normalizer_scale": model.normalizer.scale.contiguous(),
        "anchor_score": model.anchor_score.contiguous(),
        "weight": model.weight.contiguous(),
        "l2_lambda": float(model.l2_lambda),
        "final_loss": float(model.final_loss),
    }


def robust_selection_dict(
    selection: RobustThresholdSelection,
) -> dict[str, Any]:
    return {
        "feasible": selection.feasible,
        "full_threshold": selection.full_threshold,
        "jackknife_thresholds": {
            str(episode): threshold
            for episode, threshold in selection.jackknife_thresholds
        },
        "order_statistic_threshold": selection.order_statistic_threshold,
        "pre_shrink_threshold": selection.pre_shrink_threshold,
        "runtime_threshold": selection.runtime_threshold,
        "runtime_summary": route_summary_dict(selection.runtime_summary),
        "failure_reason": selection.failure_reason,
    }


def fit_outer_fold(
    data: D5DevelopmentData,
    full_action_distance: torch.Tensor,
    outer_episode: int,
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Fit one immutable D6 outer fold with 17 inner episode folds."""

    data.validate()
    if outer_episode not in D5_EPISODES:
        raise D6OOFError("D6 outer episode differs")
    if (
        full_action_distance.device.type != "cpu"
        or full_action_distance.shape != (data.rows,)
        or not full_action_distance.is_floating_point()
        or not bool(torch.isfinite(full_action_distance).all())
    ):
        raise D6OOFError("D6 full-action distance geometry differs")
    row_severity = severity_weights(full_action_distance)
    inner_episodes = tuple(value for value in D5_EPISODES if value != outer_episode)
    inner_predictions = {
        value: torch.full((data.rows, 2), float("nan"), dtype=torch.float64)
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
            model = fit_severity_weighted_glm(
                data.features,
                data.candidate_layer,
                data.unsafe_target,
                row_severity,
                fit_mask,
                l2_lambda=value,
                max_iterations=max_iterations,
            )
            prediction = model.predict(
                data.features[validation_mask],
                data.candidate_layer[validation_mask],
            )
            inner_predictions[value][validation_mask] = prediction
            loss_store[value].append(
                weighted_task_cell_losses(
                    prediction,
                    data.unsafe_target[validation_mask],
                    row_severity[validation_mask],
                    data.task_id[validation_mask],
                )
            )
            fit_count += 1

    outer_train = data.episode_index != outer_episode
    for value in LAMBDA_GRID:
        prediction = inner_predictions[value]
        if (
            not bool(torch.isfinite(prediction[outer_train]).all())
            or bool(torch.isfinite(prediction[~outer_train]).any())
        ):
            raise D6OOFError("D6 inner OOF assignment differs")
    cells = {
        value: torch.cat(loss_store[value]).contiguous()
        for value in LAMBDA_GRID
    }
    if any(value.numel() != 170 for value in cells.values()):
        raise D6OOFError("D6 inner task-cell count differs")
    selected_lambda, lambda_summary = one_standard_error_choice(cells)
    selected_inner = inner_predictions[selected_lambda][outer_train].contiguous()
    threshold_selection = robust_threshold_selection(
        selected_inner[:, 0],
        selected_inner[:, 1],
        data.action_consistency[outer_train],
        data.unsafe_target[outer_train],
        data.task_id[outer_train],
        data.episode_index[outer_train],
    )

    outer_model = fit_severity_weighted_glm(
        data.features,
        data.candidate_layer,
        data.unsafe_target,
        row_severity,
        outer_train,
        l2_lambda=selected_lambda,
        max_iterations=max_iterations,
    )
    fit_count += 1
    validation_mask = ~outer_train
    validation_indices = torch.nonzero(validation_mask, as_tuple=False).flatten()
    outer_prediction = outer_model.predict(
        data.features[validation_mask], data.candidate_layer[validation_mask]
    )
    if threshold_selection.feasible:
        if threshold_selection.runtime_threshold is None:
            raise D6OOFError("D6 feasible threshold lacks runtime value")
        selected_layer = route_at_threshold(
            outer_prediction[:, 0],
            outer_prediction[:, 1],
            data.action_consistency[validation_mask],
            threshold=threshold_selection.runtime_threshold,
        )
    else:
        selected_layer = torch.full(
            (int(validation_mask.sum()) // 2,),
            D5_FALLBACK_LAYER,
            dtype=torch.long,
        )
    if fit_count != D6_FITS_PER_OUTER:
        raise D6OOFError("D6 outer fit count differs")
    return {
        "schema_version": D6_OOF_SCHEMA_VERSION,
        "outer_episode": outer_episode,
        "validation_indices": validation_indices.contiguous(),
        "validation_score": outer_prediction.contiguous(),
        "selected_layer": selected_layer.contiguous(),
        "selected_lambda": selected_lambda,
        "lambda_one_standard_error": lambda_summary,
        "robust_threshold": robust_selection_dict(threshold_selection),
        "outer_model_state": severity_fit_state(outer_model),
        "fit_count": fit_count,
    }


__all__ = [
    "D6_FITS_PER_OUTER",
    "D6_OOF_SCHEMA_VERSION",
    "D6OOFError",
    "fit_outer_fold",
    "robust_selection_dict",
    "severity_fit_state",
]
