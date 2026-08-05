from __future__ import annotations

import numpy as np
import pytest

from a1.vla.dynamic_compute.risk_route13_router import (
    RiskRoute13Model,
    Route13FeaturePreprocessor,
    fit_route13_head,
    route13_metrics,
    route13_or_27,
)


def _arrays(rows: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260926)
    return {
        "layer13_hidden": rng.normal(size=(rows, 12)),
        "current_proprio": rng.normal(size=(rows, 2)),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": np.asarray(
            [[False, index > 0] for index in range(rows)], dtype=np.bool_
        ),
        "phase_stage": rng.normal(size=(rows, 4)),
        "phase_scalars": rng.random(size=(rows, 3)),
        "step_feature": np.linspace(0.0, 1.0, rows),
        "teacher_route": np.asarray([11, 13, 27] * (rows // 3)),
    }


def test_router_never_emits_route11_and_fails_closed_below_threshold() -> None:
    routes = route13_or_27(np.asarray([0.1, 0.8, 0.9]), threshold=0.8)
    np.testing.assert_array_equal(routes, np.asarray([27, 13, 13]))
    assert set(routes.tolist()) == {13, 27}
    with pytest.raises(ValueError):
        route13_or_27(np.asarray([np.nan]), threshold=0.8)


def test_metrics_treat_teacher11_as_safe_route13() -> None:
    metrics = route13_metrics(
        np.asarray([13, 13, 13, 27]), np.asarray([11, 13, 27, 11])
    )
    assert metrics["binary_exact"] == 2
    assert metrics["false_shallow"] == 1
    assert metrics["route27_false_shallow"] == 1
    assert metrics["safe13_rows"] == 3
    assert metrics["safe13_recalled"] == 2
    assert metrics["overcompute_safe13_to27"] == 1


def test_main_preprocessor_includes_step_and_uses_only_fit_rows() -> None:
    arrays = _arrays()
    mask = np.asarray([True] * 6 + [False] * 6)
    preprocessor = Route13FeaturePreprocessor.fit(
        arrays, mask, variant="temporal_phase_step"
    )
    transformed = preprocessor.transform(arrays)
    # hidden 12 + stage 4 + continuous (2+4+4+3+1) + history mask 2
    assert transformed.shape == (12, 32)
    continuous_start = 12 + 4
    continuous_end = transformed.shape[1] - 2
    np.testing.assert_allclose(
        transformed[mask, continuous_start:continuous_end].mean(axis=0),
        0.0,
        atol=1e-10,
    )
    assert not np.allclose(
        transformed[~mask, continuous_start:continuous_end].mean(axis=0), 0.0
    )


def test_route13_model_checkpoint_roundtrip(tmp_path) -> None:
    arrays = _arrays()
    fit_mask = np.ones(12, dtype=np.bool_)
    preprocessor, head, _ = fit_route13_head(
        arrays,
        fit_mask,
        variant="temporal_phase_step",
        pca_rank=5,
        max_iter=30,
    )
    model = RiskRoute13Model("temporal_phase_step", preprocessor, head, 0.75)
    expected_probability = model.probabilities(arrays)
    expected_route = model.routes(arrays)
    path = tmp_path / "m426_router.npz"
    model.save(path, checkpoint_sha256="a" * 64)
    loaded = RiskRoute13Model.load(path)
    np.testing.assert_array_equal(loaded.probabilities(arrays), expected_probability)
    np.testing.assert_array_equal(loaded.routes(arrays), expected_route)

