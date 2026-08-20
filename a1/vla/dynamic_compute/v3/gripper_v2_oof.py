"""Grouped nested-OOF fitting and evaluation for V3-D2 Gripper-v2.

The module is filesystem-free.  It consumes the frozen flattened 97-D
development dataset and keeps complete task-episode groups out of every fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .gripper_v2_models import (
    COUNT_SUPPORT_MAX,
    LAMBDA_GRID,
    conditional_nll,
    expected_positive_count,
    fit_occurrence_glm,
    fit_ordinal_glm,
    fit_zt_binomial_glm,
    one_standard_error_choice,
    ranked_probability_score,
    tie_aware_auroc,
)
from .gripper_v2_protocol import (
    DECISION_LAYERS,
    DEVELOPMENT_EPISODES,
    FEATURE_DIMENSION,
)


OOF_SCHEMA_VERSION = "phase-route-vla.v3.gripper-v2-nested-oof.v1"
TASK_IDS = tuple(range(10))
HEAD_NAMES = (
    "occurrence",
    "zt_step",
    "zt_transition",
    "ordinal_step",
    "ordinal_transition",
)
EXPECTED_FITS_PER_OUTER = 17 * 3 * 5 + 5


class GripperV2OOFError(ValueError):
    """Raised when data, folds, predictions, or a metric fail closed."""


@dataclass(frozen=True)
class DevelopmentData:
    features: torch.Tensor
    candidate_layer: torch.Tensor
    source_row: torch.Tensor
    task_id: torch.Tensor
    episode_index: torch.Tensor
    occurrence: torch.Tensor
    count: torch.Tensor
    expected_fraction: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    def validate(self) -> None:
        rows = self.rows
        if (
            self.features.device.type != "cpu"
            or self.features.ndim != 2
            or self.features.shape != (rows, FEATURE_DIMENSION)
            or not self.features.is_floating_point()
            or not bool(torch.isfinite(self.features).all())
        ):
            raise GripperV2OOFError("features must be finite CPU [N,97]")
        long_vectors = {
            "candidate_layer": self.candidate_layer,
            "source_row": self.source_row,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
        }
        for name, value in long_vectors.items():
            if value.device.type != "cpu" or value.dtype != torch.long or value.shape != (rows,):
                raise GripperV2OOFError(f"{name} must be CPU int64 [N]")
        if not bool(((self.candidate_layer == 11) | (self.candidate_layer == 13)).all()):
            raise GripperV2OOFError("candidate layers differ")
        if set(self.task_id.tolist()) != set(TASK_IDS):
            raise GripperV2OOFError("task coverage differs")
        if set(self.episode_index.tolist()) != set(DEVELOPMENT_EPISODES):
            raise GripperV2OOFError("development episode coverage differs")
        if (
            self.occurrence.device.type != "cpu"
            or self.occurrence.dtype != torch.bool
            or self.occurrence.shape != (rows, 2)
            or self.count.device.type != "cpu"
            or self.count.dtype != torch.long
            or self.count.shape != (rows, 2)
            or self.expected_fraction.device.type != "cpu"
            or not self.expected_fraction.is_floating_point()
            or self.expected_fraction.shape != (rows, 2)
            or not bool(torch.isfinite(self.expected_fraction).all())
        ):
            raise GripperV2OOFError("target geometry differs")
        support = torch.tensor(COUNT_SUPPORT_MAX, dtype=torch.long)
        if bool((self.count < 0).any()) or bool((self.count > support).any()):
            raise GripperV2OOFError("count lies outside frozen support")
        if not torch.equal(self.occurrence, self.count > 0):
            raise GripperV2OOFError("occurrence/count identity differs")
        expected = self.count.double() / support.double()
        if not torch.allclose(
            self.expected_fraction.double(), expected, rtol=0.0, atol=1.0e-7
        ):
            raise GripperV2OOFError("expected-fraction target identity differs")
        if rows % 2 or not torch.equal(self.source_row[0::2], self.source_row[1::2]):
            raise GripperV2OOFError("candidate pairs do not share source rows")
        if not torch.equal(
            self.source_row,
            torch.arange(rows // 2, dtype=torch.long).repeat_interleave(2),
        ):
            raise GripperV2OOFError("source-row identity is not contiguous paired order")
        if not torch.equal(
            self.candidate_layer.reshape(-1, 2),
            torch.tensor(DECISION_LAYERS, dtype=torch.long).expand(rows // 2, 2),
        ):
            raise GripperV2OOFError("candidate pair order differs")
        if not torch.equal(self.task_id[0::2], self.task_id[1::2]) or not torch.equal(
            self.episode_index[0::2], self.episode_index[1::2]
        ):
            raise GripperV2OOFError("candidate pair group identity differs")
        cells = set(zip(self.task_id.tolist(), self.episode_index.tolist()))
        expected_cells = {(task, episode) for task in TASK_IDS for episode in DEVELOPMENT_EPISODES}
        if cells != expected_cells:
            raise GripperV2OOFError("task-episode cell coverage differs")
        for layer in DECISION_LAYERS:
            layer_mask = self.candidate_layer == layer
            for target_index in range(2):
                values = self.count[layer_mask, target_index]
                if int((values == 0).sum()) < 100 or int((values > 0).sum()) < 100:
                    raise GripperV2OOFError("minimum zero/positive support is not met")


def development_data_from_mapping(payload: Mapping[str, Any]) -> DevelopmentData:
    required = (
        "features",
        "candidate_layer",
        "source_row",
        "task_id",
        "episode_index",
        "occurrence",
        "count",
        "expected_fraction",
    )
    if any(name not in payload for name in required):
        raise GripperV2OOFError("dataset is missing a required tensor")
    data = DevelopmentData(**{name: payload[name].detach().cpu().contiguous() for name in required})
    data.validate()
    return data


def _cell_means(
    row_loss: torch.Tensor,
    task_id: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    if row_loss.ndim != 1 or task_id.shape != row_loss.shape or not bool(torch.isfinite(row_loss).all()):
        raise GripperV2OOFError(f"{context} cell-loss geometry differs")
    cells = []
    for task in TASK_IDS:
        selected = task_id == task
        if not bool(selected.any()):
            raise GripperV2OOFError(f"{context} has an empty task cell")
        cells.append(row_loss[selected].double().mean())
    return torch.stack(cells).contiguous()


def _occurrence_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(
        probability.double(), target.double(), reduction="none"
    ).mean(dim=1)


def _append_count_cells(
    store: dict[float, list[torch.Tensor]],
    regularization_lambda: float,
    probabilities: torch.Tensor,
    count: torch.Tensor,
    task_id: torch.Tensor,
    *,
    context: str,
) -> None:
    positive = count > 0
    values = conditional_nll(probabilities[positive], count[positive])
    store[regularization_lambda].append(
        _cell_means(values, task_id[positive], context=context)
    )


def _state(model: Any) -> dict[str, Any]:
    state = {
        "normalizer_mean": model.normalizer.mean.contiguous(),
        "normalizer_scale": model.normalizer.scale.contiguous(),
        "weight": model.weight.contiguous(),
        "l2_lambda": float(model.l2_lambda),
        "final_loss": float(model.final_loss),
    }
    if hasattr(model, "anchor_probability"):
        state["anchor_probability"] = model.anchor_probability.contiguous()
    if hasattr(model, "raw_base"):
        state["raw_base"] = model.raw_base.contiguous()
        state["raw_increments"] = model.raw_increments.contiguous()
        state["cutpoints"] = model.cutpoints.contiguous()
    if hasattr(model, "target_index"):
        state["target_index"] = int(model.target_index)
        state["support_max"] = int(model.support_max)
    return state


def fit_outer_fold(
    data: DevelopmentData,
    outer_episode: int,
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """Run all 17 inner LOEO folds and one outer refit."""

    data.validate()
    if outer_episode not in DEVELOPMENT_EPISODES:
        raise GripperV2OOFError("outer episode differs")
    inner_episodes = tuple(ep for ep in DEVELOPMENT_EPISODES if ep != outer_episode)
    loss_store: dict[str, dict[float, list[torch.Tensor]]] = {
        head: {value: [] for value in LAMBDA_GRID} for head in HEAD_NAMES
    }
    fit_count = 0
    for inner_episode in inner_episodes:
        fit_mask = (data.episode_index != outer_episode) & (
            data.episode_index != inner_episode
        )
        validation_mask = data.episode_index == inner_episode
        valid_features = data.features[validation_mask]
        valid_layer = data.candidate_layer[validation_mask]
        valid_task = data.task_id[validation_mask]
        valid_occurrence = data.occurrence[validation_mask]
        valid_count = data.count[validation_mask]
        for value in LAMBDA_GRID:
            occurrence = fit_occurrence_glm(
                data.features,
                data.candidate_layer,
                data.occurrence,
                fit_mask,
                l2_lambda=value,
                max_iterations=max_iterations,
            )
            occurrence_probability = occurrence.predict(valid_features, valid_layer)
            loss_store["occurrence"][value].append(
                _cell_means(
                    _occurrence_loss(occurrence_probability, valid_occurrence),
                    valid_task,
                    context="occurrence",
                )
            )
            fit_count += 1
            for target_index, target_name in enumerate(("step", "transition")):
                zt = fit_zt_binomial_glm(
                    data.features,
                    data.candidate_layer,
                    data.count,
                    fit_mask,
                    target_index=target_index,
                    l2_lambda=value,
                    max_iterations=max_iterations,
                )
                ordinal = fit_ordinal_glm(
                    data.features,
                    data.candidate_layer,
                    data.count,
                    fit_mask,
                    target_index=target_index,
                    l2_lambda=value,
                    max_iterations=max_iterations,
                )
                _append_count_cells(
                    loss_store[f"zt_{target_name}"],
                    value,
                    zt.probabilities(valid_features, valid_layer),
                    valid_count[:, target_index],
                    valid_task,
                    context=f"zt_{target_name}",
                )
                _append_count_cells(
                    loss_store[f"ordinal_{target_name}"],
                    value,
                    ordinal.probabilities(valid_features, valid_layer),
                    valid_count[:, target_index],
                    valid_task,
                    context=f"ordinal_{target_name}",
                )
                fit_count += 2

    selected: dict[str, float] = {}
    one_se: dict[str, Any] = {}
    for head in HEAD_NAMES:
        concatenated = {
            value: torch.cat(loss_store[head][value]).contiguous()
            for value in LAMBDA_GRID
        }
        if any(values.numel() != 170 for values in concatenated.values()):
            raise GripperV2OOFError("inner one-SE cell count differs")
        selected[head], one_se[head] = one_standard_error_choice(concatenated)

    fit_mask = data.episode_index != outer_episode
    validation_mask = data.episode_index == outer_episode
    features = data.features[validation_mask]
    layers = data.candidate_layer[validation_mask]
    occurrence = fit_occurrence_glm(
        data.features,
        data.candidate_layer,
        data.occurrence,
        fit_mask,
        l2_lambda=selected["occurrence"],
        max_iterations=max_iterations,
    )
    zt_models = []
    ordinal_models = []
    for target_index, target_name in enumerate(("step", "transition")):
        zt_models.append(
            fit_zt_binomial_glm(
                data.features,
                data.candidate_layer,
                data.count,
                fit_mask,
                target_index=target_index,
                l2_lambda=selected[f"zt_{target_name}"],
                max_iterations=max_iterations,
            )
        )
        ordinal_models.append(
            fit_ordinal_glm(
                data.features,
                data.candidate_layer,
                data.count,
                fit_mask,
                target_index=target_index,
                l2_lambda=selected[f"ordinal_{target_name}"],
                max_iterations=max_iterations,
            )
        )
    fit_count += 5
    if fit_count != EXPECTED_FITS_PER_OUTER:
        raise GripperV2OOFError("outer fit count differs")

    layer_index = (layers == 13).long()
    occurrence_probability = occurrence.predict(features, layers)
    occurrence_baseline = occurrence.anchor_probability[layer_index]
    zt_probability = {
        name: model.probabilities(features, layers)
        for name, model in zip(("step", "transition"), zt_models)
    }
    ordinal_probability = {
        name: model.probabilities(features, layers)
        for name, model in zip(("step", "transition"), ordinal_models)
    }
    expected_fraction = torch.stack(
        [
            occurrence_probability[:, target] * expected_positive_count(
                ordinal_probability[name]
            ) / COUNT_SUPPORT_MAX[target]
            for target, name in enumerate(("step", "transition"))
        ],
        dim=1,
    )
    expected_baseline = torch.empty_like(expected_fraction)
    for layer_position, layer in enumerate(DECISION_LAYERS):
        train_layer = fit_mask & (data.candidate_layer == layer)
        expected_baseline[layer_index == layer_position] = (
            data.count[train_layer].double()
            / torch.tensor(COUNT_SUPPORT_MAX, dtype=torch.float64)
        ).mean(dim=0)

    predictions = {
        "row_index": torch.nonzero(validation_mask, as_tuple=False)[:, 0].long().contiguous(),
        "occurrence_probability": occurrence_probability.contiguous(),
        "occurrence_baseline": occurrence_baseline.contiguous(),
        "zt_step_probability": zt_probability["step"].contiguous(),
        "zt_transition_probability": zt_probability["transition"].contiguous(),
        "ordinal_step_probability": ordinal_probability["step"].contiguous(),
        "ordinal_transition_probability": ordinal_probability["transition"].contiguous(),
        "expected_fraction": expected_fraction.contiguous(),
        "expected_fraction_baseline": expected_baseline.contiguous(),
    }
    if not all(bool(torch.isfinite(value).all()) for value in predictions.values()):
        raise GripperV2OOFError("outer predictions are non-finite")
    return {
        "schema_version": OOF_SCHEMA_VERSION,
        "outer_episode": outer_episode,
        "inner_episodes": list(inner_episodes),
        "selected_lambda": selected,
        "one_standard_error": one_se,
        "fit_count": fit_count,
        "predictions": predictions,
        "outer_model_state": {
            "occurrence": _state(occurrence),
            "zt_step": _state(zt_models[0]),
            "zt_transition": _state(zt_models[1]),
            "ordinal_step": _state(ordinal_models[0]),
            "ordinal_transition": _state(ordinal_models[1]),
        },
    }


def final_lambda(outer_values: list[float]) -> float:
    if len(outer_values) != len(DEVELOPMENT_EPISODES) or any(
        value not in LAMBDA_GRID for value in outer_values
    ):
        raise GripperV2OOFError("outer lambda votes differ")
    counts = {value: outer_values.count(value) for value in LAMBDA_GRID}
    maximum = max(counts.values())
    return max(value for value, count in counts.items() if count == maximum)


def fit_final_models(
    data: DevelopmentData,
    lambdas: Mapping[str, float],
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    if tuple(lambdas) != HEAD_NAMES:
        raise GripperV2OOFError("final lambda head order differs")
    fit_mask = torch.ones(data.rows, dtype=torch.bool)
    models = {
        "occurrence": fit_occurrence_glm(
            data.features, data.candidate_layer, data.occurrence, fit_mask,
            l2_lambda=lambdas["occurrence"], max_iterations=max_iterations,
        )
    }
    for target_index, name in enumerate(("step", "transition")):
        models[f"zt_{name}"] = fit_zt_binomial_glm(
            data.features, data.candidate_layer, data.count, fit_mask,
            target_index=target_index, l2_lambda=lambdas[f"zt_{name}"],
            max_iterations=max_iterations,
        )
        models[f"ordinal_{name}"] = fit_ordinal_glm(
            data.features, data.candidate_layer, data.count, fit_mask,
            target_index=target_index, l2_lambda=lambdas[f"ordinal_{name}"],
            max_iterations=max_iterations,
        )
    return {name: _state(models[name]) for name in HEAD_NAMES}


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> dict[str, Any]:
    top = float(numerator.double().sum())
    bottom = float(denominator.double().sum())
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0.0:
        return {"status": "INCONCLUSIVE_ZERO_OR_NONFINITE_BASELINE", "value": None}
    return {"status": "PASS_FINITE", "value": top / bottom, "numerator": top, "denominator": bottom}


def exact_sign_test_upper_tail(improvements: int, trials: int) -> float:
    if not 0 <= improvements <= trials or trials < 1:
        raise GripperV2OOFError("sign-test counts differ")
    return sum(math.comb(trials, value) for value in range(improvements, trials + 1)) / (2**trials)


def _validate_oof_predictions(
    data: DevelopmentData, oof: Mapping[str, torch.Tensor]
) -> None:
    shapes = {
        "occurrence_probability": (data.rows, 2),
        "occurrence_baseline": (data.rows, 2),
        "zt_step_probability": (data.rows, 8),
        "zt_transition_probability": (data.rows, 7),
        "ordinal_step_probability": (data.rows, 8),
        "ordinal_transition_probability": (data.rows, 7),
        "expected_fraction": (data.rows, 2),
        "expected_fraction_baseline": (data.rows, 2),
    }
    for name, shape in shapes.items():
        value = oof[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or value.shape != shape
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise GripperV2OOFError(f"OOF {name} must be finite CPU FP64 {shape}")
    for name in (
        "occurrence_probability",
        "occurrence_baseline",
    ):
        if bool(((oof[name] <= 0.0) | (oof[name] >= 1.0)).any()):
            raise GripperV2OOFError(f"OOF {name} lies outside (0,1)")
    for name in (
        "zt_step_probability",
        "zt_transition_probability",
        "ordinal_step_probability",
        "ordinal_transition_probability",
    ):
        probability = oof[name]
        if bool((probability <= 0.0).any()) or not torch.allclose(
            probability.sum(dim=1),
            torch.ones(data.rows, dtype=torch.float64),
            rtol=1.0e-9,
            atol=1.0e-9,
        ):
            raise GripperV2OOFError(f"OOF {name} is not a positive simplex")
    for name in ("expected_fraction", "expected_fraction_baseline"):
        if bool(((oof[name] < 0.0) | (oof[name] > 1.0)).any()):
            raise GripperV2OOFError(f"OOF {name} lies outside [0,1]")
    assignment = oof["assignment_count"]
    if (
        not isinstance(assignment, torch.Tensor)
        or assignment.device.type != "cpu"
        or assignment.dtype != torch.long
        or assignment.shape != (data.rows,)
        or not assignment.is_contiguous()
        or not torch.equal(assignment, torch.ones(data.rows, dtype=torch.long))
    ):
        raise GripperV2OOFError("OOF assignment is not exactly once")


def evaluate_oof(data: DevelopmentData, oof: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    required = (
        "occurrence_probability", "occurrence_baseline", "zt_step_probability",
        "zt_transition_probability", "ordinal_step_probability",
        "ordinal_transition_probability", "expected_fraction",
        "expected_fraction_baseline", "assignment_count",
    )
    if any(name not in oof for name in required):
        raise GripperV2OOFError("OOF prediction field is missing")
    _validate_oof_predictions(data, oof)
    scope_masks = {
        "overall": torch.ones(data.rows, dtype=torch.bool),
        "layer11": data.candidate_layer == 11,
        "layer13": data.candidate_layer == 13,
    }
    occurrence_metrics: dict[str, Any] = {}
    expected_metrics: dict[str, Any] = {}
    count_metrics: dict[str, Any] = {}
    for target_index, target_name in enumerate(("step", "transition")):
        occurrence_metrics[target_name] = {}
        expected_metrics[target_name] = {}
        count_metrics[target_name] = {}
        zt = oof[f"zt_{target_name}_probability"]
        ordinal = oof[f"ordinal_{target_name}_probability"]
        for scope, mask in scope_masks.items():
            truth = data.occurrence[mask, target_index]
            probability = oof["occurrence_probability"][mask, target_index]
            baseline = oof["occurrence_baseline"][mask, target_index]
            brier = _ratio((probability - truth.double()).square(), (baseline - truth.double()).square())
            occurrence_metrics[target_name][scope] = {
                "support": int(mask.sum()),
                "positive": int(truth.sum()),
                "brier_ratio": brier,
                "brier_skill": None if brier["value"] is None else 1.0 - brier["value"],
                "auroc": tie_aware_auroc(probability, truth),
            }
            expected_metrics[target_name][scope] = _ratio(
                (oof["expected_fraction"][mask, target_index] - data.expected_fraction[mask, target_index]).square(),
                (oof["expected_fraction_baseline"][mask, target_index] - data.expected_fraction[mask, target_index]).square(),
            )
            positive = mask & (data.count[:, target_index] > 0)
            count_truth = data.count[positive, target_index]
            zt_nll = conditional_nll(zt[positive], count_truth)
            ordinal_nll = conditional_nll(ordinal[positive], count_truth)
            ratio = _ratio(ordinal_nll, zt_nll)
            count_metrics[target_name][scope] = {
                "support": int(positive.sum()),
                "conditional_nll_ratio": ratio,
                "zt_mean_nll": float(zt_nll.mean()),
                "ordinal_mean_nll": float(ordinal_nll.mean()),
                "zt_mean_discrete_crps_rps": float(
                    ranked_probability_score(zt[positive], count_truth).mean()
                ),
                "ordinal_mean_discrete_crps_rps": float(
                    ranked_probability_score(ordinal[positive], count_truth).mean()
                ),
                "zt_count_mae": float((expected_positive_count(zt[positive]) - count_truth).abs().mean()),
                "ordinal_count_mae": float((expected_positive_count(ordinal[positive]) - count_truth).abs().mean()),
            }

    outer_episodes: dict[str, Any] = {}
    improved = 0
    conclusive = 0
    for episode in DEVELOPMENT_EPISODES:
        cell_zt = []
        cell_ordinal = []
        missing = []
        for task in TASK_IDS:
            for layer in DECISION_LAYERS:
                for target_index, target_name in enumerate(("step", "transition")):
                    mask = (
                        (data.episode_index == episode)
                        & (data.task_id == task)
                        & (data.candidate_layer == layer)
                        & (data.count[:, target_index] > 0)
                    )
                    if not bool(mask.any()):
                        missing.append([task, layer, target_name])
                        continue
                    truth = data.count[mask, target_index]
                    cell_zt.append(conditional_nll(oof[f"zt_{target_name}_probability"][mask], truth).mean())
                    cell_ordinal.append(conditional_nll(oof[f"ordinal_{target_name}_probability"][mask], truth).mean())
        if missing:
            outer_episodes[str(episode)] = {"status": "INCONCLUSIVE_MISSING_POSITIVE_CELL", "missing_cells": missing}
            continue
        zt_value = float(torch.stack(cell_zt).mean())
        ordinal_value = float(torch.stack(cell_ordinal).mean())
        is_improved = ordinal_value < zt_value
        improved += int(is_improved)
        conclusive += 1
        outer_episodes[str(episode)] = {
            "status": "PASS_FINITE", "zt_mean_nll": zt_value,
            "ordinal_mean_nll": ordinal_value, "improved": is_improved,
        }
    sign_p = exact_sign_test_upper_tail(improved, 18) if conclusive == 18 else None

    occurrence_pass = all(
        metric["brier_skill"] is not None
        and metric["brier_skill"] > 0.0
        and metric["auroc"] > 0.5
        for target in occurrence_metrics.values() for metric in target.values()
    )
    expected_pass = all(
        metric["value"] is not None and metric["value"] < 1.0
        for target in expected_metrics.values() for metric in target.values()
    )
    overall_values = [
        count_metrics[target]["overall"]["conditional_nll_ratio"]["value"]
        for target in ("step", "transition")
    ]
    overall_count_pass = all(
        value is not None and value < 1.0 for value in overall_values
    )
    layer_ratios = [count_metrics[target][scope]["conditional_nll_ratio"]["value"] for target in ("step", "transition") for scope in ("layer11", "layer13")]
    strict_layer_improvements = sum(value is not None and value < 1.0 for value in layer_ratios)
    worst_layer_ratio = None if any(value is None for value in layer_ratios) else max(layer_ratios)
    robustness_pass = conclusive == 18 and improved >= 13 and sign_p is not None and sign_p <= 0.05
    full_pass = (
        occurrence_pass
        and expected_pass
        and robustness_pass
        and overall_count_pass
        and all(value is not None and value < 1.0 for value in layer_ratios)
    )
    focused_pass = (
        occurrence_pass and expected_pass and robustness_pass and overall_count_pass
        and strict_layer_improvements >= 3 and worst_layer_ratio is not None
        and worst_layer_ratio <= 1.01
    )
    return {
        "occurrence": occurrence_metrics,
        "expected_fraction": expected_metrics,
        "conditional_count": count_metrics,
        "group_robustness": {
            "by_outer_episode": outer_episodes,
            "conclusive_episodes": conclusive,
            "improved_episodes": improved,
            "exact_sign_test_upper_tail": sign_p,
            "passed": robustness_pass,
        },
        "timing_secondary": {"status": "NOT_MODELED_BY_FROZEN_GRIPPER_V2_FAMILY"},
        "gates": {
            "occurrence_pass": occurrence_pass,
            "expected_fraction_pass": expected_pass,
            "count_overall_both_targets_pass": overall_count_pass,
            "strictly_improved_layer_target_scopes": strict_layer_improvements,
            "worst_layer_target_ratio": worst_layer_ratio,
            "group_robustness_pass": robustness_pass,
            "full_pass": full_pass,
            "focused_pass_non_deployable": focused_pass,
        },
    }


__all__ = [
    "DevelopmentData", "EXPECTED_FITS_PER_OUTER", "GripperV2OOFError",
    "HEAD_NAMES", "OOF_SCHEMA_VERSION", "development_data_from_mapping",
    "evaluate_oof", "exact_sign_test_upper_tail", "final_lambda",
    "fit_final_models", "fit_outer_fold",
]
