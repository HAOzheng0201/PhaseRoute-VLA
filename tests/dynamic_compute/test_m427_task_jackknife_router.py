from __future__ import annotations

import numpy as np
import pytest

from a1.vla.dynamic_compute.m427_task_jackknife_router import (
    M427_TASKS,
    TaskJackknifeRoute13Ensemble,
    aggregate_safe13_probabilities,
    calibrate_strict_negative_max,
    episode_group_risk_metrics,
    strict_route13_or_27,
    zero_error_clopper_pearson_upper,
)
from a1.vla.dynamic_compute.risk_route13_router import (
    RiskRoute13Model,
    fit_route13_head,
)


def _arrays(rows: int = 20) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261127)
    teacher = np.where(np.arange(rows) % 4 == 0, 27, 13)
    signal = np.where(teacher == 27, -2.0, 2.0)
    return {
        "layer13_hidden": signal[:, None] + rng.normal(scale=0.1, size=(rows, 8)),
        "current_proprio": rng.normal(size=(rows, 2)),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": rng.random(size=(rows, 2)) > 0.5,
        "phase_stage": signal[:, None] + rng.normal(scale=0.1, size=(rows, 4)),
        "phase_scalars": rng.random(size=(rows, 3)),
        "step_feature": np.linspace(0.0, 1.0, rows),
        "teacher_route": teacher,
    }


def _model(arrays: dict[str, np.ndarray]) -> RiskRoute13Model:
    preprocessor, head, _ = fit_route13_head(
        arrays,
        np.ones(arrays["teacher_route"].shape, dtype=np.bool_),
        variant="temporal_phase_step",
        pca_rank=4,
        max_iter=30,
    )
    return RiskRoute13Model("temporal_phase_step", preprocessor, head, 1.0)


def test_min_and_mean_aggregation_are_frozen() -> None:
    probabilities = np.asarray([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_array_equal(
        aggregate_safe13_probabilities(probabilities, aggregation="min"),
        np.asarray([0.2, 0.4]),
    )
    np.testing.assert_array_equal(
        aggregate_safe13_probabilities(probabilities, aggregation="mean"),
        np.asarray([0.5, 0.5]),
    )
    with pytest.raises(ValueError):
        aggregate_safe13_probabilities(probabilities, aggregation="median")


def test_strict_negative_max_keeps_equal_negative_on_route27() -> None:
    scores = np.asarray([0.2, 0.7, 0.8])
    teacher = np.asarray([13, 27, 13])
    threshold = calibrate_strict_negative_max(scores, teacher)
    assert threshold > 0.7
    np.testing.assert_array_equal(
        strict_route13_or_27(scores, threshold=threshold),
        np.asarray([27, 27, 13]),
    )


def test_episode_group_risk_and_zero_error_bound() -> None:
    metrics = episode_group_risk_metrics(
        np.asarray([27, 13, 13, 27]),
        np.asarray([27, 27, 13, 27]),
        np.asarray([0, 0, 0, 1]),
        np.asarray([0, 0, 1, 0]),
    )
    assert metrics["route27_positive_groups"] == 2
    assert metrics["route27_error_groups"] == 1
    assert metrics["error_groups"][0]["false_shallow_rows"] == 1
    assert zero_error_clopper_pearson_upper(19) < 0.15
    assert zero_error_clopper_pearson_upper(18) > 0.15


def test_task_jackknife_ensemble_checkpoint_roundtrip(tmp_path) -> None:
    arrays = _arrays()
    model = _model(arrays)
    ensemble = TaskJackknifeRoute13Ensemble(
        tuple(model for _ in M427_TASKS), M427_TASKS, "min", 0.8
    )
    expected = ensemble.scores(arrays)
    descriptor = ensemble.save(tmp_path / "ensemble")
    loaded = TaskJackknifeRoute13Ensemble.load(descriptor)
    np.testing.assert_array_equal(loaded.scores(arrays), expected)
    assert loaded.excluded_tasks == M427_TASKS
    assert loaded.aggregation == "min"
    assert loaded.threshold == 0.8

