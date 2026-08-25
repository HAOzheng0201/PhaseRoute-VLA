from __future__ import annotations

import numpy as np

from a1.vla.dynamic_compute.route_first_calibration import (
    ROUTE_FIRST_CALIBRATION_STATUS,
    RouteFirstThresholdRule,
    confirm_route_first_threshold,
    load_calibrated_route_first_router,
    route_first_confirmed_layers,
    save_calibrated_route_first_router,
    select_route_first_threshold,
    weighted_wilson_upper_bound,
)
from a1.vla.dynamic_compute.route_first_router import fit_route_first_ordinal_router


def test_weighted_wilson_matches_equal_weight_zero_error_bound() -> None:
    result = weighted_wilson_upper_bound(
        np.zeros(10, dtype=np.bool_),
        np.ones(10),
        confidence_level=0.9,
    )
    z_value = 1.2815515655446008
    expected = z_value**2 / (10.0 + z_value**2)

    assert np.isclose(result["effective_selected_rows"], 10.0)
    assert result["empirical_false_safe_rate"] == 0.0
    assert np.isclose(result["false_safe_upper_bound"], expected)


def test_selection_chooses_maximum_feasible_tied_prefix_coverage() -> None:
    score = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    safe = np.asarray([True, True, False, True, False])
    task = np.zeros(5, dtype=np.int64)
    episode = np.zeros(5, dtype=np.int64)
    rule = RouteFirstThresholdRule(
        minimum_coverage=0.2,
        maximum_coverage=0.8,
        minimum_effective_selected_rows=1.0,
        maximum_empirical_false_safe_rate=0.25,
        maximum_false_safe_upper_bound=1.0,
    )

    selected = select_route_first_threshold(
        score,
        safe,
        task,
        episode,
        rule=rule,
        confidence_level=0.9,
    )

    assert selected["enabled"] is True
    assert selected["threshold"] == 0.6
    assert selected["metrics"]["selected_rows"] == 4
    assert np.isclose(selected["metrics"]["actual_coverage"], 0.8)
    assert np.isclose(
        selected["metrics"]["empirical_false_safe_rate"], 0.25
    )


def test_no_feasible_selection_disables_head() -> None:
    selected = select_route_first_threshold(
        np.asarray([0.9, 0.8, 0.7]),
        np.asarray([False, False, False]),
        np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        rule=RouteFirstThresholdRule(
            minimum_coverage=0.3,
            maximum_coverage=1.0,
            minimum_effective_selected_rows=1.0,
            maximum_empirical_false_safe_rate=0.0,
            maximum_false_safe_upper_bound=1.0,
        ),
        confidence_level=0.9,
    )

    assert selected["enabled"] is False
    assert selected["threshold"] is None
    assert selected["reason"] == "NO_FEASIBLE_SELECTION_THRESHOLD"


def test_confirmation_checks_exact_threshold_and_fails_closed() -> None:
    selection = {"enabled": True, "threshold": 0.8}
    rule = RouteFirstThresholdRule(
        minimum_coverage=0.25,
        minimum_effective_selected_rows=1.0,
        maximum_empirical_false_safe_rate=0.0,
        maximum_false_safe_upper_bound=1.0,
    )
    confirmed = confirm_route_first_threshold(
        selection,
        np.asarray([0.95, 0.85, 0.75, 0.1]),
        np.asarray([True, True, False, False]),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        rule=rule,
        confidence_level=0.9,
    )
    failed = confirm_route_first_threshold(
        selection,
        np.asarray([0.95, 0.85, 0.75, 0.1]),
        np.asarray([True, False, False, False]),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        rule=rule,
        confidence_level=0.9,
    )

    assert confirmed["threshold"] == selection["threshold"]
    assert confirmed["active_enabled"] is True
    assert failed["threshold"] == selection["threshold"]
    assert failed["active_enabled"] is False
    assert failed["reason"] == "CONFIRMATION_GATE_FAILED_HEAD_DISABLED"


def test_confirmed_layers_apply_l11_before_l13_and_otherwise_l27() -> None:
    scores = np.asarray(
        [
            [0.9, 0.95],
            [0.2, 0.9],
            [0.1, 0.5],
        ]
    )
    confirmation11 = {"active_enabled": True, "threshold": 0.8}
    confirmation13 = {"active_enabled": True, "threshold": 0.8}

    layers = route_first_confirmed_layers(
        scores, confirmation11, confirmation13
    )

    assert layers.tolist() == [11, 13, 27]


def test_calibrated_artifact_roundtrip_keeps_active_control_disabled(tmp_path) -> None:
    rng = np.random.default_rng(20260825)
    features = rng.normal(size=(120, 199)).astype(np.float32)
    latent = features[:, 0] + 0.5 * features[:, 1]
    teacher = np.where(latent > 0.8, 11, np.where(latent > -0.2, 13, 27))
    task = np.repeat(np.arange(3), 40)
    episode = np.tile(np.repeat(np.arange(4), 10), 3)
    router = fit_route_first_ordinal_router(
        features, teacher, task, episode, pca_rank=8, l2=1.0
    )
    selection11 = {"enabled": True, "threshold": 0.91}
    selection13 = {"enabled": True, "threshold": 0.73}
    confirmation11 = {
        "active_enabled": False,
        "threshold": 0.91,
    }
    confirmation13 = {
        "active_enabled": True,
        "threshold": 0.73,
    }
    path = tmp_path / "calibrated_router.npz"
    save_calibrated_route_first_router(
        path,
        router,
        source_router_sha256="a" * 64,
        calibration_payload_sha256="b" * 64,
        calibration_file_sha256="c" * 64,
        protocol_file_sha256="d" * 64,
        selection11=selection11,
        selection13=selection13,
        confirmation11=confirmation11,
        confirmation13=confirmation13,
        engineering_holdout_authorized=True,
    )

    loaded, metadata = load_calibrated_route_first_router(path)
    np.testing.assert_allclose(
        loaded.probabilities(features),
        router.probabilities(features),
        rtol=2e-5,
        atol=2e-6,
    )
    assert metadata["calibration_status"] == ROUTE_FIRST_CALIBRATION_STATUS
    assert metadata["threshold11"] == 0.91
    assert metadata["enabled11"] is False
    assert metadata["threshold13"] == 0.73
    assert metadata["enabled13"] is True
    assert metadata["engineering_holdout_authorized"] is True
    assert metadata["active_control_authorized"] is False
