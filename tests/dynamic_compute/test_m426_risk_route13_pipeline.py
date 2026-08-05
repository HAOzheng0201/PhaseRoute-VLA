from __future__ import annotations

import numpy as np

from scripts.dynamic_compute.evaluate_m426_risk_route13_router import science_gates
from scripts.dynamic_compute.train_m426_risk_route13_router import (
    M426A_CALIBRATION_EPISODES,
    M426A_DEV_EPISODES,
    M426A_TEST_EPISODES,
    fit_variant,
)


def _grouped_arrays(episodes_per_task: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260926)
    task = np.repeat(np.arange(10), episodes_per_task)
    episode = np.tile(np.arange(episodes_per_task), 10)
    rows = task.size
    route27 = (task + 2 * episode) % 5 == 0
    teacher = np.where(route27, 27, np.where((task + episode) % 2, 13, 11))
    signal = np.where(route27, -2.0, 2.0)
    return {
        "layer13_hidden": signal[:, None] + 0.1 * rng.normal(size=(rows, 8)),
        "current_proprio": np.stack([signal, task / 9.0], axis=1),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": np.asarray(
            [[False, ep > 0] for ep in episode], dtype=np.bool_
        ),
        "phase_stage": signal[:, None] + 0.1 * rng.normal(size=(rows, 4)),
        "phase_scalars": np.stack(
            [episode / 5.0, route27.astype(float), np.full(rows, 0.1)], axis=1
        ),
        "step_feature": episode / 5.0,
        "task_id": task,
        "episode_index": episode,
        "teacher_route": teacher,
    }


def test_grouped_fit_keeps_sealed_rows_unread_and_is_fail_closed() -> None:
    arrays = _grouped_arrays()
    model, report = fit_variant(
        arrays,
        variant="temporal_phase_step",
        pca_rank=4,
        l2=1.0,
        max_iter=30,
        eps=1e-6,
    )
    assert model.variant == "temporal_phase_step"
    assert report["development_rows"] == 30
    assert report["calibration_rows"] == 10
    assert report["sealed_test_rows_not_evaluated"] == 20
    assert report["oof_metrics"]["route27_false_shallow"] == 0
    assert report["calibration_metrics"]["route27_false_shallow"] == 0
    assert len(report["oof_folds"]) == 30


def test_m426a_grouped_fit_uses_two_calibration_and_two_sealed_episodes() -> None:
    arrays = _grouped_arrays(episodes_per_task=7)
    model, report = fit_variant(
        arrays,
        variant="step_proprio",
        pca_rank=4,
        l2=1.0,
        max_iter=30,
        eps=1e-6,
        development_episodes=M426A_DEV_EPISODES,
        calibration_episodes=M426A_CALIBRATION_EPISODES,
        test_episodes=M426A_TEST_EPISODES,
    )
    assert model.variant == "step_proprio"
    assert report["development_rows"] == 30
    assert report["calibration_rows"] == 20
    assert report["sealed_test_rows_not_evaluated"] == 20
    assert report["oof_metrics"]["route27_false_shallow"] == 0
    assert report["calibration_metrics"]["route27_false_shallow"] == 0


def _method(*, exact=0.8, false_shallow=0, recall=0.5, coverage=0.5, reduction=0.2):
    return {
        "metrics": {
            "binary_exact_accuracy": exact,
            "route27_false_shallow": false_shallow,
            "route27_rows": 5,
            "safe13_recall": recall,
            "predicted13_coverage": coverage,
        },
        "estimated_latency": {"reduction_fraction": reduction},
    }


def test_science_gate_requires_safety_utility_and_control_parity() -> None:
    fit = {
        "oof_metrics": {"route27_false_shallow": 0},
        "calibration_metrics": {"route27_false_shallow": 0},
    }
    controls = {
        "hidden_only": _method(exact=0.75),
        "step_proprio": _method(exact=0.78),
    }
    gates = science_gates(_method(), controls, fit)
    assert all(gates.values())
    unsafe = science_gates(_method(false_shallow=1), controls, fit)
    assert unsafe["sealed_route27_false_shallow_zero"] is False
    assert unsafe["route27_false_shallow_not_above_controls"] is False
    weak = science_gates(_method(exact=0.76), controls, fit)
    assert weak["binary_exact_not_below_controls"] is False
