import numpy as np

from scripts.dynamic_compute.analyze_m416_phase_drift_risk import (
    _regression_metrics,
    nested_group_ridge_predictions,
)


def test_nested_group_ridge_generalizes_a_shared_linear_signal():
    groups = np.repeat(np.arange(4), 5)
    x = np.linspace(-2.0, 2.0, groups.size)
    features = np.column_stack((x, x**2))
    targets = 0.25 + 1.5 * x

    prediction, folds = nested_group_ridge_predictions(
        features,
        targets,
        groups,
        alphas=(0.0, 0.01, 1.0),
    )
    metrics = _regression_metrics(targets, prediction)

    assert len(folds) == 4
    assert {fold["held_task"] for fold in folds} == {0, 1, 2, 3}
    assert metrics["mae"] < 0.05
    assert metrics["pearson"] > 0.99


def test_regression_metrics_rejects_misaligned_vectors():
    try:
        _regression_metrics(np.zeros(2), np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned vectors were accepted")
