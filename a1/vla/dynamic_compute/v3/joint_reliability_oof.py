"""Strict grouped nested-OOF fitting for V3-D5 joint reliability."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .gripper_v2_models import (
    LAMBDA_GRID,
    fit_occurrence_glm,
    one_standard_error_choice,
)
from .joint_reliability import (
    D5_EPISODES,
    D5_FALLBACK_LAYER,
    D5_TASK_IDS,
    D5DevelopmentData,
    RouteSummary,
    route_at_threshold,
    select_inner_threshold,
)


D5_OOF_SCHEMA_VERSION = "phase-route-vla.v3.d5-joint-nested-oof-fold.v1"
D5_FITS_PER_OUTER = len(D5_EPISODES[:-1]) * len(LAMBDA_GRID) + 1


class D5OOFError(ValueError):
    """Raised when D5 nested-OOF partitioning or fitting fails closed."""


def binary_task_cell_losses(
    probability: torch.Tensor,
    target: torch.Tensor,
    task_id: torch.Tensor,
) -> torch.Tensor:
    """Return ten equal-weight task-cell binary NLL values."""

    if (
        probability.device.type != "cpu"
        or not probability.is_floating_point()
        or probability.ndim != 2
        or probability.shape[1] != 2
        or target.device.type != "cpu"
        or target.dtype != torch.bool
        or target.shape != probability.shape
        or task_id.device.type != "cpu"
        or task_id.dtype != torch.long
        or task_id.shape != (probability.shape[0],)
        or not bool(torch.isfinite(probability).all())
        or not bool(((probability > 0.0) & (probability < 1.0)).all())
    ):
        raise D5OOFError("D5 task-cell loss geometry differs")
    row_loss = F.binary_cross_entropy(
        probability.double(), target.double(), reduction="none"
    ).mean(dim=1)
    cells = []
    for task in D5_TASK_IDS:
        selected = task_id == task
        if not bool(selected.any()):
            raise D5OOFError("D5 task-cell loss has an empty task")
        cells.append(row_loss[selected].mean())
    result = torch.stack(cells).contiguous()
    if result.shape != (10,) or not bool(torch.isfinite(result).all()):
        raise D5OOFError("D5 task-cell loss is invalid")
    return result


def occurrence_state(model: Any) -> dict[str, Any]:
    return {
        "normalizer_mean": model.normalizer.mean.contiguous(),
        "normalizer_scale": model.normalizer.scale.contiguous(),
        "anchor_probability": model.anchor_probability.contiguous(),
        "weight": model.weight.contiguous(),
        "l2_lambda": float(model.l2_lambda),
        "final_loss": float(model.final_loss),
    }


def route_summary_dict(summary: RouteSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "early_exit_calls": summary.early_exit_calls,
        "early_exit_fraction": summary.early_exit_fraction,
        "safe_clusters": summary.safe_clusters,
        "false_safe_clusters": summary.false_safe_clusters,
        "false_safe_ucb95": summary.false_safe_ucb95,
        "per_task_early_calls": list(summary.per_task_early_calls),
        "feasible": summary.feasible,
    }


def fit_outer_fold(
    data: D5DevelopmentData,
    outer_episode: int,
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Fit one immutable D5 outer fold with 17 inner episode folds."""

    data.validate()
    if outer_episode not in D5_EPISODES:
        raise D5OOFError("D5 outer episode differs")
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
            model = fit_occurrence_glm(
                data.features,
                data.candidate_layer,
                data.unsafe_target,
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
                binary_task_cell_losses(
                    prediction,
                    data.unsafe_target[validation_mask],
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
            raise D5OOFError("D5 inner OOF assignment differs")
    cells = {
        value: torch.cat(loss_store[value]).contiguous()
        for value in LAMBDA_GRID
    }
    if any(value.numel() != 170 for value in cells.values()):
        raise D5OOFError("D5 inner task-cell count differs")
    selected_lambda, lambda_summary = one_standard_error_choice(cells)
    selected_inner = inner_predictions[selected_lambda][outer_train].contiguous()
    threshold_selection = select_inner_threshold(
        selected_inner[:, 0],
        selected_inner[:, 1],
        data.action_consistency[outer_train],
        data.unsafe_target[outer_train],
        data.task_id[outer_train],
        data.episode_index[outer_train],
    )

    outer_model = fit_occurrence_glm(
        data.features,
        data.candidate_layer,
        data.unsafe_target,
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
        assert threshold_selection.threshold is not None
        selected_layer = route_at_threshold(
            outer_prediction[:, 0],
            outer_prediction[:, 1],
            data.action_consistency[validation_mask],
            threshold=threshold_selection.threshold,
        )
    else:
        selected_layer = torch.full(
            (int(validation_mask.sum()) // 2,),
            D5_FALLBACK_LAYER,
            dtype=torch.long,
        )
    if fit_count != D5_FITS_PER_OUTER:
        raise D5OOFError("D5 outer fit count differs")
    return {
        "schema_version": D5_OOF_SCHEMA_VERSION,
        "outer_episode": outer_episode,
        "validation_indices": validation_indices.contiguous(),
        "validation_probability": outer_prediction.contiguous(),
        "selected_layer": selected_layer.contiguous(),
        "selected_lambda": selected_lambda,
        "lambda_one_standard_error": lambda_summary,
        "inner_threshold_feasible": threshold_selection.feasible,
        "inner_selected_threshold": threshold_selection.threshold,
        "inner_threshold_summary": route_summary_dict(threshold_selection.summary),
        "inner_evaluated_thresholds": threshold_selection.evaluated_thresholds,
        "outer_model_state": occurrence_state(outer_model),
        "fit_count": fit_count,
    }


__all__ = [
    "D5_FITS_PER_OUTER",
    "D5_OOF_SCHEMA_VERSION",
    "D5OOFError",
    "binary_task_cell_losses",
    "fit_outer_fold",
    "occurrence_state",
    "route_summary_dict",
]
