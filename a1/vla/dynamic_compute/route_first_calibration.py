"""Pre-registered threshold calibration for route-first safety scores.

The score model is frozen before this module is used.  One split may select a
threshold from observed score prefixes; a disjoint confirmation split can only
accept that exact threshold or disable the corresponding head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping

import numpy as np

from .route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
)
from .route_first_router import (
    RouteFirstAffineHead,
    RouteFirstOrdinalRouter,
    route_first_group_weights,
)


ROUTE_FIRST_CALIBRATED_ROUTER_SCHEMA_VERSION = (
    "phase-route-vla.route-first-calibrated-router.v1"
)
ROUTE_FIRST_CALIBRATION_STATUS = "SET_ONE_SHOT_CONFIRMED_FAIL_CLOSED"


@dataclass(frozen=True)
class RouteFirstThresholdRule:
    minimum_coverage: float
    minimum_effective_selected_rows: float
    maximum_empirical_false_safe_rate: float
    maximum_false_safe_upper_bound: float
    maximum_coverage: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_coverage,
            self.minimum_effective_selected_rows,
            self.maximum_empirical_false_safe_rate,
            self.maximum_false_safe_upper_bound,
            self.maximum_coverage,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("route-first threshold rule must be finite")
        if (
            not 0.0 <= self.minimum_coverage <= self.maximum_coverage <= 1.0
            or self.minimum_effective_selected_rows <= 0.0
            or not 0.0 <= self.maximum_empirical_false_safe_rate <= 1.0
            or not 0.0 <= self.maximum_false_safe_upper_bound <= 1.0
        ):
            raise ValueError("route-first threshold rule is outside its domain")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object], *, selection: bool
    ) -> "RouteFirstThresholdRule":
        required = {
            "minimum_coverage",
            "minimum_effective_selected_rows",
            "maximum_empirical_false_safe_rate",
            "maximum_false_safe_upper_bound",
        }
        if selection:
            required.add("maximum_coverage")
        if set(values) != required:
            raise ValueError("route-first threshold rule fields differ")
        return cls(
            minimum_coverage=float(values["minimum_coverage"]),
            maximum_coverage=float(values.get("maximum_coverage", 1.0)),
            minimum_effective_selected_rows=float(
                values["minimum_effective_selected_rows"]
            ),
            maximum_empirical_false_safe_rate=float(
                values["maximum_empirical_false_safe_rate"]
            ),
            maximum_false_safe_upper_bound=float(
                values["maximum_false_safe_upper_bound"]
            ),
        )


def _aligned_inputs(
    score: np.ndarray, safe_label: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    safe = np.asarray(safe_label, dtype=np.bool_).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if values.shape != safe.shape or values.shape != weights.shape or values.size < 1:
        raise ValueError("route-first calibration inputs must be aligned")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any((values < 0.0) | (values > 1.0))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("route-first calibration inputs are invalid")
    return values, safe, weights


def weighted_wilson_upper_bound(
    false_safe: np.ndarray,
    sample_weight: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float]:
    errors = np.asarray(false_safe, dtype=np.bool_).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if (
        errors.shape != weights.shape
        or errors.size < 1
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not 0.5 < confidence_level < 1.0
    ):
        raise ValueError("weighted Wilson inputs are invalid")
    weight_sum = float(weights.sum())
    effective_rows = float(weight_sum**2 / np.square(weights).sum())
    rate = float(weights[errors].sum() / weight_sum)
    z_value = float(NormalDist().inv_cdf(float(confidence_level)))
    z_squared = z_value**2
    denominator = 1.0 + z_squared / effective_rows
    center = (rate + z_squared / (2.0 * effective_rows)) / denominator
    margin = z_value / denominator * math.sqrt(
        rate * (1.0 - rate) / effective_rows
        + z_squared / (4.0 * effective_rows**2)
    )
    return {
        "empirical_false_safe_rate": rate,
        "false_safe_upper_bound": float(min(1.0, center + margin)),
        "effective_selected_rows": effective_rows,
        "confidence_level": float(confidence_level),
    }


def evaluate_route_first_threshold(
    score: np.ndarray,
    safe_label: np.ndarray,
    sample_weight: np.ndarray,
    *,
    threshold: float,
    confidence_level: float,
) -> dict[str, float | int]:
    values, safe, weights = _aligned_inputs(score, safe_label, sample_weight)
    if not math.isfinite(float(threshold)) or not 0.0 <= threshold <= 1.0:
        raise ValueError("route-first threshold must be finite and in [0,1]")
    selected = values >= float(threshold)
    selected_rows = int(selected.sum())
    selected_mass = float(weights[selected].sum())
    coverage = float(selected_mass / weights.sum())
    if selected_rows == 0:
        return {
            "threshold": float(threshold),
            "selected_rows": 0,
            "actual_coverage": 0.0,
            "empirical_false_safe_rate": 1.0,
            "false_safe_upper_bound": 1.0,
            "effective_selected_rows": 0.0,
            "confidence_level": float(confidence_level),
        }
    bound = weighted_wilson_upper_bound(
        ~safe[selected], weights[selected], confidence_level=confidence_level
    )
    return {
        "threshold": float(threshold),
        "selected_rows": selected_rows,
        "actual_coverage": coverage,
        **bound,
    }


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


def select_route_first_threshold(
    score: np.ndarray,
    safe_label: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    rule: RouteFirstThresholdRule,
    confidence_level: float,
) -> dict[str, object]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    weights = route_first_group_weights(task_id, episode_index)
    _aligned_inputs(values, safe_label, weights)
    feasible: list[dict[str, float | int]] = []
    for threshold in np.unique(values)[::-1].tolist():
        metrics = evaluate_route_first_threshold(
            values,
            safe_label,
            weights,
            threshold=float(threshold),
            confidence_level=confidence_level,
        )
        if _passes_rule(metrics, rule):
            feasible.append(metrics)
    if not feasible:
        return {
            "enabled": False,
            "threshold": None,
            "reason": "NO_FEASIBLE_SELECTION_THRESHOLD",
            "candidate_thresholds": int(np.unique(values).size),
            "feasible_thresholds": 0,
            "metrics": None,
        }
    selected = max(
        feasible,
        key=lambda metrics: (
            float(metrics["actual_coverage"]),
            -float(metrics["empirical_false_safe_rate"]),
            float(metrics["threshold"]),
        ),
    )
    return {
        "enabled": True,
        "threshold": float(selected["threshold"]),
        "reason": "MAXIMUM_FEASIBLE_GROUP_EQUAL_COVERAGE",
        "candidate_thresholds": int(np.unique(values).size),
        "feasible_thresholds": len(feasible),
        "metrics": selected,
    }


def confirm_route_first_threshold(
    selection: Mapping[str, object],
    score: np.ndarray,
    safe_label: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    rule: RouteFirstThresholdRule,
    confidence_level: float,
) -> dict[str, object]:
    if not bool(selection.get("enabled")):
        return {
            "selection_enabled": False,
            "confirmed": False,
            "active_enabled": False,
            "threshold": None,
            "reason": "SELECTION_HEAD_DISABLED",
            "metrics": None,
        }
    threshold = selection.get("threshold")
    if not isinstance(threshold, (float, int)) or not math.isfinite(float(threshold)):
        raise ValueError("enabled selection must contain one finite threshold")
    weights = route_first_group_weights(task_id, episode_index)
    metrics = evaluate_route_first_threshold(
        score,
        safe_label,
        weights,
        threshold=float(threshold),
        confidence_level=confidence_level,
    )
    confirmed = _passes_rule(metrics, rule)
    return {
        "selection_enabled": True,
        "confirmed": confirmed,
        "active_enabled": confirmed,
        "threshold": float(threshold),
        "reason": "CONFIRMED_EXACT_SELECTION_THRESHOLD"
        if confirmed
        else "CONFIRMATION_GATE_FAILED_HEAD_DISABLED",
        "metrics": metrics,
    }


def route_first_safe_label(
    teacher_layer: np.ndarray, *, head: int
) -> np.ndarray:
    teacher = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    if not set(np.unique(teacher).tolist()).issubset({11, 13, 27}):
        raise ValueError("route-first teacher layers are invalid")
    if head == 11:
        return teacher == 11
    if head == 13:
        return teacher <= 13
    raise ValueError("route-first calibration head must be 11 or 13")


def route_first_confirmed_layers(
    scores: np.ndarray,
    confirmation11: Mapping[str, object],
    confirmation13: Mapping[str, object],
) -> np.ndarray:
    probability = np.asarray(scores, dtype=np.float64)
    if (
        probability.ndim != 2
        or probability.shape[1] != 2
        or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
        or np.any(probability[:, 0] > probability[:, 1])
    ):
        raise ValueError("route-first confirmation scores are invalid")
    layers = np.full(probability.shape[0], 27, dtype=np.int16)
    if bool(confirmation13.get("active_enabled")):
        threshold13 = float(confirmation13["threshold"])
        layers[probability[:, 1] >= threshold13] = 13
    if bool(confirmation11.get("active_enabled")):
        threshold11 = float(confirmation11["threshold"])
        layers[probability[:, 0] >= threshold11] = 11
    return layers


def _lowercase_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def save_calibrated_route_first_router(
    path: str | Path,
    router: RouteFirstOrdinalRouter,
    *,
    source_router_sha256: str,
    calibration_payload_sha256: str,
    calibration_file_sha256: str,
    protocol_file_sha256: str,
    selection11: Mapping[str, object],
    selection13: Mapping[str, object],
    confirmation11: Mapping[str, object],
    confirmation13: Mapping[str, object],
    engineering_holdout_authorized: bool,
) -> None:
    """Save confirmed thresholds while keeping active control unauthorized."""

    target = Path(path)
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    hashes = (
        source_router_sha256,
        calibration_payload_sha256,
        calibration_file_sha256,
        protocol_file_sha256,
    )
    if not all(_lowercase_sha256(value) for value in hashes):
        raise ValueError("calibrated route-first hashes must be lowercase SHA-256")

    def threshold_and_enabled(
        selection: Mapping[str, object], confirmation: Mapping[str, object]
    ) -> tuple[float, bool]:
        selection_enabled = bool(selection.get("enabled"))
        active_enabled = bool(confirmation.get("active_enabled"))
        if not selection_enabled:
            if active_enabled:
                raise ValueError("disabled selection cannot become active")
            return 1.0, False
        threshold = selection.get("threshold")
        if not isinstance(threshold, (float, int)) or not 0.0 <= float(
            threshold
        ) <= 1.0:
            raise ValueError("enabled selection threshold is invalid")
        confirmed_threshold = confirmation.get("threshold")
        if float(confirmed_threshold) != float(threshold):
            raise ValueError("confirmation changed the selected threshold")
        return float(threshold), active_enabled

    threshold11, enabled11 = threshold_and_enabled(selection11, confirmation11)
    threshold13, enabled13 = threshold_and_enabled(selection13, confirmation13)
    if bool(engineering_holdout_authorized) != enabled13:
        raise ValueError("engineering holdout authorization must follow safe13")
    if router.head11.l2 != router.head13.l2:
        raise ValueError("route-first calibrated heads must share L2")
    payload = {
        "schema_version": np.asarray(ROUTE_FIRST_CALIBRATED_ROUTER_SCHEMA_VERSION),
        "feature_schema_version": np.asarray(ROUTE_FIRST_FEATURE_SCHEMA_VERSION),
        "feature_dimension": np.asarray(ROUTE_FIRST_FEATURE_DIMENSION, dtype=np.int32),
        "calibration_status": np.asarray(ROUTE_FIRST_CALIBRATION_STATUS),
        "weight11": router.head11.weight.astype(np.float32),
        "bias11": np.asarray(router.head11.bias, dtype=np.float64),
        "pca_rank11": np.asarray(router.head11.pca_rank, dtype=np.int32),
        "iterations11": np.asarray(router.head11.iterations, dtype=np.int32),
        "threshold11": np.asarray(threshold11, dtype=np.float64),
        "enabled11": np.asarray(enabled11, dtype=np.bool_),
        "weight13": router.head13.weight.astype(np.float32),
        "bias13": np.asarray(router.head13.bias, dtype=np.float64),
        "pca_rank13": np.asarray(router.head13.pca_rank, dtype=np.int32),
        "iterations13": np.asarray(router.head13.iterations, dtype=np.int32),
        "threshold13": np.asarray(threshold13, dtype=np.float64),
        "enabled13": np.asarray(enabled13, dtype=np.bool_),
        "l2": np.asarray(router.head11.l2, dtype=np.float64),
        "source_router_sha256": np.asarray(source_router_sha256),
        "calibration_payload_sha256": np.asarray(calibration_payload_sha256),
        "calibration_file_sha256": np.asarray(calibration_file_sha256),
        "protocol_file_sha256": np.asarray(protocol_file_sha256),
        "selection_episode_indices": np.asarray([8], dtype=np.int16),
        "confirmation_episode_indices": np.asarray([9], dtype=np.int16),
        "engineering_holdout_authorized": np.asarray(
            engineering_holdout_authorized, dtype=np.bool_
        ),
        "active_control_authorized": np.asarray(False, dtype=np.bool_),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as output_file:
            np.savez_compressed(output_file, **payload)
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def load_calibrated_route_first_router(
    path: str | Path,
) -> tuple[RouteFirstOrdinalRouter, dict[str, object]]:
    target = Path(path).expanduser().resolve(strict=True)
    required = {
        "schema_version",
        "feature_schema_version",
        "feature_dimension",
        "calibration_status",
        "weight11",
        "bias11",
        "pca_rank11",
        "iterations11",
        "threshold11",
        "enabled11",
        "weight13",
        "bias13",
        "pca_rank13",
        "iterations13",
        "threshold13",
        "enabled13",
        "l2",
        "source_router_sha256",
        "calibration_payload_sha256",
        "calibration_file_sha256",
        "protocol_file_sha256",
        "selection_episode_indices",
        "confirmation_episode_indices",
        "engineering_holdout_authorized",
        "active_control_authorized",
    }
    with np.load(target, allow_pickle=False) as arrays:
        if set(arrays.files) != required:
            raise ValueError("calibrated route-first router fields differ")
        if (
            str(arrays["schema_version"].item())
            != ROUTE_FIRST_CALIBRATED_ROUTER_SCHEMA_VERSION
            or str(arrays["feature_schema_version"].item())
            != ROUTE_FIRST_FEATURE_SCHEMA_VERSION
            or int(arrays["feature_dimension"].item())
            != ROUTE_FIRST_FEATURE_DIMENSION
            or str(arrays["calibration_status"].item())
            != ROUTE_FIRST_CALIBRATION_STATUS
        ):
            raise ValueError("calibrated route-first schema contract differs")
        l2 = float(arrays["l2"].item())
        router = RouteFirstOrdinalRouter(
            RouteFirstAffineHead(
                arrays["weight11"].astype(np.float64),
                float(arrays["bias11"].item()),
                int(arrays["pca_rank11"].item()),
                l2,
                int(arrays["iterations11"].item()),
            ),
            RouteFirstAffineHead(
                arrays["weight13"].astype(np.float64),
                float(arrays["bias13"].item()),
                int(arrays["pca_rank13"].item()),
                l2,
                int(arrays["iterations13"].item()),
            ),
        )
        hashes = {
            name: str(arrays[name].item())
            for name in (
                "source_router_sha256",
                "calibration_payload_sha256",
                "calibration_file_sha256",
                "protocol_file_sha256",
            )
        }
        if not all(_lowercase_sha256(value) for value in hashes.values()):
            raise ValueError("calibrated route-first metadata hashes are invalid")
        threshold11 = float(arrays["threshold11"].item())
        threshold13 = float(arrays["threshold13"].item())
        enabled11 = bool(arrays["enabled11"].item())
        enabled13 = bool(arrays["enabled13"].item())
        holdout = bool(arrays["engineering_holdout_authorized"].item())
        if (
            not 0.0 <= threshold11 <= 1.0
            or not 0.0 <= threshold13 <= 1.0
            or holdout != enabled13
            or bool(arrays["active_control_authorized"].item())
            or arrays["selection_episode_indices"].astype(np.int64).tolist() != [8]
            or arrays["confirmation_episode_indices"].astype(np.int64).tolist()
            != [9]
        ):
            raise ValueError("calibrated route-first fail-closed metadata differs")
        metadata: dict[str, object] = {
            **hashes,
            "threshold11": threshold11,
            "enabled11": enabled11,
            "threshold13": threshold13,
            "enabled13": enabled13,
            "engineering_holdout_authorized": holdout,
            "active_control_authorized": False,
            "calibration_status": ROUTE_FIRST_CALIBRATION_STATUS,
        }
    return router, metadata


__all__ = [
    "ROUTE_FIRST_CALIBRATED_ROUTER_SCHEMA_VERSION",
    "ROUTE_FIRST_CALIBRATION_STATUS",
    "RouteFirstThresholdRule",
    "confirm_route_first_threshold",
    "evaluate_route_first_threshold",
    "load_calibrated_route_first_router",
    "route_first_confirmed_layers",
    "route_first_safe_label",
    "select_route_first_threshold",
    "save_calibrated_route_first_router",
    "weighted_wilson_upper_bound",
]
