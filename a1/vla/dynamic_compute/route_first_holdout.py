"""One-shot engineering holdout audit for a frozen route-first router."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .route_first_calibration import (
    RouteFirstThresholdRule,
    evaluate_route_first_threshold,
    route_first_safe_label,
)
from .route_first_router import route_first_group_weights


ROUTE_FIRST_HOLDOUT_PROTOCOL_SCHEMA = (
    "phase-route-vla.route-first-holdout-protocol.v1"
)
ROUTE_FIRST_HOLDOUT_PASS_STATUS = (
    "PASS_ENGINEERING_HOLDOUT_RUNTIME_INTEGRATION_READY"
)
ROUTE_FIRST_HOLDOUT_FAIL_STATUS = (
    "FAIL_ENGINEERING_HOLDOUT_ROUTE_FIRST_DISABLED"
)


def load_route_first_holdout_protocol(path: str | Path) -> dict[str, object]:
    """Load the stage-specific protocol and reject a changed data boundary."""

    target = Path(path).expanduser().resolve(strict=True)
    with target.open("r", encoding="utf-8") as input_file:
        protocol = json.load(input_file)
    required = {
        "schema_version",
        "seed",
        "frozen_calibrated_router",
        "data",
        "statistics",
        "holdout_gate",
        "diagnostics",
        "fail_closed_policy",
        "claim_boundary",
    }
    if set(protocol) != required:
        raise ValueError("route-first holdout protocol fields differ")
    if protocol["schema_version"] != ROUTE_FIRST_HOLDOUT_PROTOCOL_SCHEMA:
        raise ValueError("route-first holdout protocol schema differs")
    frozen = protocol["frozen_calibrated_router"]
    if set(frozen) != {
        "path",
        "file_sha256",
        "calibration_status_required",
        "source_router_sha256",
        "threshold11",
        "enabled11",
        "threshold13",
        "enabled13",
        "engineering_holdout_authorized",
        "active_control_authorized",
        "calibration_result_path",
        "calibration_result_sha256",
        "calibration_verification_path",
        "calibration_verification_sha256",
    }:
        raise ValueError("frozen holdout router fields differ")
    if (
        frozen["file_sha256"]
        != "ae561b77c01bd4c7eee6cc0ff91e215733662544cc1af2e5039b0a8f02c60cc2"
        or frozen["source_router_sha256"]
        != "38aaef193442a4b40e71b1d48bee42ffbe5f191cad64f99d20bd3f75df3ad3ae"
        or frozen["calibration_result_sha256"]
        != "c599ffb8280368a1014f4de0827524c3e0bc5d9ccc172ede4062600fea9d5de5"
        or frozen["calibration_verification_sha256"]
        != "34b07c89be854df10f541b8b6a3eebffcd1c7f9a7cf021276d2850f81148c7bb"
        or float(frozen["threshold11"]) != 0.9807427653025883
        or bool(frozen["enabled11"])
        or float(frozen["threshold13"]) != 0.9174261218080999
        or not bool(frozen["enabled13"])
        or not bool(frozen["engineering_holdout_authorized"])
        or bool(frozen["active_control_authorized"])
    ):
        raise ValueError("frozen holdout router binding differs")
    data = protocol["data"]
    if data != {
        "suite": "libero_10",
        "task_ids": list(range(10)),
        "engineering_holdout_episode_indices": [10, 11],
        "training_episode_indices_forbidden": list(range(8)),
        "calibration_episode_indices_forbidden": [8, 9],
        "historical_D9_episode_indices_forbidden": list(range(40, 50)),
        "teacher": "frozen_phase_route_v3",
        "control_influence": False,
        "identity_is_model_input": False,
    }:
        raise ValueError("route-first holdout data boundary differs")
    if protocol["statistics"] != {
        "episode_cells": "equal_total_weight",
        "confidence_method": (
            "one_sided_weighted_wilson_effective_sample_size"
        ),
        "confidence_level": 0.9,
        "threshold_policy": "exact_stage6_threshold_no_refit_no_movement",
        "decision_policy": "pooled_and_each_episode_index_must_pass",
    }:
        raise ValueError("route-first holdout statistics differ")
    gate = protocol["holdout_gate"]
    if set(gate) != {"pooled_safe13", "per_episode_index_safe13"}:
        raise ValueError("route-first holdout gate fields differ")
    RouteFirstThresholdRule.from_mapping(gate["pooled_safe13"], selection=False)
    RouteFirstThresholdRule.from_mapping(
        gate["per_episode_index_safe13"], selection=False
    )
    if protocol["fail_closed_policy"] != {
        "safe11_must_remain_disabled": True,
        "safe13_threshold_may_not_change": True,
        "any_pooled_or_per_episode_gate_failure_disables_runtime_integration": True,
        "holdout_failure_may_not_trigger_refit_or_recalibration": True,
        "all_disabled_route": 27,
        "active_control_not_authorized_by_holdout_alone": True,
    }:
        raise ValueError("route-first holdout fail-closed policy differs")
    return protocol


def _passes_rule(
    metrics: Mapping[str, float | int], rule: RouteFirstThresholdRule
) -> bool:
    return bool(
        float(metrics["actual_coverage"]) >= rule.minimum_coverage
        and float(metrics["actual_coverage"]) <= rule.maximum_coverage
        and float(metrics["effective_selected_rows"])
        >= rule.minimum_effective_selected_rows
        and float(metrics["empirical_false_safe_rate"])
        <= rule.maximum_empirical_false_safe_rate
        and float(metrics["false_safe_upper_bound"])
        <= rule.maximum_false_safe_upper_bound
    )


def _threshold_metrics(
    score13: np.ndarray,
    safe13: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    threshold13: float,
    confidence_level: float,
) -> dict[str, float | int]:
    return evaluate_route_first_threshold(
        score13,
        safe13,
        route_first_group_weights(task_id, episode_index),
        threshold=threshold13,
        confidence_level=confidence_level,
    )


def evaluate_route_first_holdout(
    scores: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    threshold13: float,
    enabled11: bool,
    enabled13: bool,
    expected_episode_indices: tuple[int, ...],
    pooled_rule: RouteFirstThresholdRule,
    per_episode_rule: RouteFirstThresholdRule,
    confidence_level: float,
    score_quantiles: tuple[float, ...],
) -> dict[str, object]:
    """Evaluate exact frozen routing on a holdout without changing parameters."""

    probability = np.asarray(scores, dtype=np.float64)
    teacher = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if (
        probability.ndim != 2
        or probability.shape != (teacher.size, 2)
        or tasks.shape != teacher.shape
        or episodes.shape != teacher.shape
        or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
        or np.any(probability[:, 0] > probability[:, 1])
    ):
        raise ValueError("route-first holdout arrays are invalid")
    if enabled11 or not enabled13:
        raise ValueError("holdout requires disabled L11 and enabled L13")
    if not np.isfinite(threshold13) or not 0.0 <= threshold13 <= 1.0:
        raise ValueError("holdout L13 threshold is invalid")
    expected = tuple(int(value) for value in expected_episode_indices)
    if tuple(sorted(set(episodes.tolist()))) != expected:
        raise ValueError("holdout episode indices differ")
    quantiles = tuple(float(value) for value in score_quantiles)
    if (
        not quantiles
        or tuple(sorted(set(quantiles))) != quantiles
        or quantiles[0] < 0.0
        or quantiles[-1] > 1.0
    ):
        raise ValueError("holdout score quantiles are invalid")

    safe13 = route_first_safe_label(teacher, head=13)
    pooled = _threshold_metrics(
        probability[:, 1],
        safe13,
        tasks,
        episodes,
        threshold13=threshold13,
        confidence_level=confidence_level,
    )
    pooled_pass = _passes_rule(pooled, pooled_rule)
    per_episode: dict[str, object] = {}
    episode_passes: list[bool] = []
    for episode in expected:
        selected = episodes == episode
        metrics = _threshold_metrics(
            probability[selected, 1],
            safe13[selected],
            tasks[selected],
            episodes[selected],
            threshold13=threshold13,
            confidence_level=confidence_level,
        )
        passed = _passes_rule(metrics, per_episode_rule)
        per_episode[str(episode)] = {"passed": passed, "metrics": metrics}
        episode_passes.append(passed)

    selected13 = probability[:, 1] >= threshold13
    selected_layers = np.where(selected13, 13, 27).astype(np.int16)
    raw_confusion = {
        str(layer): int(np.sum(selected13 & (teacher == layer)))
        for layer in (11, 13, 27)
    }
    per_task: dict[str, object] = {}
    for task in sorted(set(tasks.tolist())):
        selected = tasks == task
        per_task[str(task)] = _threshold_metrics(
            probability[selected, 1],
            safe13[selected],
            tasks[selected],
            episodes[selected],
            threshold13=threshold13,
            confidence_level=confidence_level,
        )
    all_passed = bool(pooled_pass and all(episode_passes))
    failures: list[str] = []
    if not pooled_pass:
        failures.append("POOLED_SAFE13_GATE_FAILED")
    failures.extend(
        f"EPISODE_INDEX_{episode}_SAFE13_GATE_FAILED"
        for episode, passed in zip(expected, episode_passes)
        if not passed
    )
    counts = Counter(int(value) for value in selected_layers)
    return {
        "status": ROUTE_FIRST_HOLDOUT_PASS_STATUS
        if all_passed
        else ROUTE_FIRST_HOLDOUT_FAIL_STATUS,
        "passed": all_passed,
        "threshold13": float(threshold13),
        "threshold_changed": False,
        "pooled_safe13": {"passed": pooled_pass, "metrics": pooled},
        "per_episode_index_safe13": per_episode,
        "failures": failures,
        "routing": {
            "selected_layer_counts": {
                "11": int(counts.get(11, 0)),
                "13": int(counts.get(13, 0)),
                "27": int(counts.get(27, 0)),
            },
            "raw_selected13_teacher_counts": raw_confusion,
        },
        "diagnostics": {
            "score13_quantiles": {
                format(value, ".12g"): float(
                    np.quantile(probability[:, 1], value)
                )
                for value in quantiles
            },
            "per_task_safe13": per_task,
        },
        "selected_layers": selected_layers,
    }


__all__ = [
    "ROUTE_FIRST_HOLDOUT_FAIL_STATUS",
    "ROUTE_FIRST_HOLDOUT_PASS_STATUS",
    "ROUTE_FIRST_HOLDOUT_PROTOCOL_SCHEMA",
    "evaluate_route_first_holdout",
    "load_route_first_holdout_protocol",
]
