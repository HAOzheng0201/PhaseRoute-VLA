from __future__ import annotations

import numpy as np

from a1.vla.dynamic_compute.temporal_route_router import (
    FeaturePreprocessor,
    TemporalRouteModel,
    _top_pca_components,
    fit_processed_pca_logistic,
)
from scripts.dynamic_compute.train_m425b_temporal_router import fit_variant


def _arrays(rows: int = 8):
    rng = np.random.default_rng(7)
    return {
        "layer11_hidden": rng.normal(size=(rows, 6)),
        "layer13_hidden": rng.normal(size=(rows, 6)),
        "current_proprio": rng.normal(size=(rows, 2)),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": np.array([[False, index > 0] for index in range(rows)]),
        "phase_stage": rng.normal(size=(rows, 4)),
        "phase_scalars": rng.random(size=(rows, 3)),
        "step_feature": np.linspace(0.0, 1.0, rows),
    }


def test_temporal_preprocessor_fits_only_selected_rows() -> None:
    arrays = _arrays()
    mask = np.array([True, True, True, True, False, False, False, False])
    preprocessor = FeaturePreprocessor.fit(
        arrays, mask, variant="temporal_phase"
    )
    transformed = preprocessor.transform(arrays, layer=11)
    assert transformed.shape == (8, 6 + 4 + 2 + 4 + 4 + 3 + 2)
    continuous_start = 6 + 4
    continuous_end = transformed.shape[1] - 2
    np.testing.assert_allclose(
        transformed[mask, continuous_start:continuous_end].mean(axis=0),
        0.0,
        atol=1e-10,
    )
    assert not np.allclose(
        transformed[~mask, continuous_start:continuous_end].mean(axis=0), 0.0
    )


def test_processed_head_and_checkpoint_roundtrip(tmp_path) -> None:
    arrays = _arrays(rows=10)
    labels = np.array([0, 1] * 5)
    fit_mask = np.ones(10, dtype=bool)
    pp11 = FeaturePreprocessor.fit(arrays, fit_mask, variant="hidden_only")
    pp13 = FeaturePreprocessor.fit(arrays, fit_mask, variant="hidden_only")
    feature11 = pp11.transform(arrays, layer=11)
    feature13 = pp13.transform(arrays, layer=13)
    head11 = fit_processed_pca_logistic(feature11, labels, pca_rank=3)
    head13 = fit_processed_pca_logistic(feature13, labels, pca_rank=3)
    model = TemporalRouteModel(
        "hidden_only", pp11, pp13, head11, head13, 0.8, 0.9
    )
    expected = model.probabilities(arrays)
    path = tmp_path / "router.npz"
    model.save(path, checkpoint_sha256="a" * 64)
    loaded = TemporalRouteModel.load(path)
    actual = loaded.probabilities(arrays)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_dual_pca_matches_full_svd_subspace() -> None:
    rng = np.random.default_rng(20260804)
    values = rng.normal(size=(18, 40))
    centered = values - values.mean(axis=0)
    actual = _top_pca_components(centered, 8)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    expected = right[:8]
    np.testing.assert_allclose(
        actual.T @ actual,
        expected.T @ expected,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(actual @ actual.T, np.eye(8), atol=1e-10)


def test_grouped_fit_keeps_test_predictions_sealed() -> None:
    rows = 60
    arrays = _arrays(rows=rows)
    arrays["task_id"] = np.repeat(np.arange(10), 6)
    arrays["episode_index"] = np.tile(np.arange(6), 10)
    arrays["teacher_route"] = np.asarray(
        [
            (11, 13, 27)[(task + episode) % 3]
            for task in range(10)
            for episode in range(6)
        ]
    )
    model, report = fit_variant(
        arrays,
        variant="step_proprio",
        pca_rank=4,
        l2=1.0,
        max_iter=20,
        eps=1e-6,
    )
    assert model.variant == "step_proprio"
    assert report["development_rows"] == 30
    assert report["calibration_rows"] == 10
    assert report["sealed_test_rows_not_evaluated"] == 20
    assert report["oof_metrics"]["false_shallow"] == 0
    assert report["calibration_metrics"]["false_shallow"] == 0
