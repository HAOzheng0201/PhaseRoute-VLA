from __future__ import annotations

import numpy as np

import a1.vla.dynamic_compute.route_first_training as training
from a1.vla.dynamic_compute.route_first_training import (
    RouteFirstCandidate,
    episode_index_candidate_search,
    false_safe_at_coverage,
    ranking_gates,
    weighted_average_precision,
    weighted_roc_auc,
)


def test_weighted_ranking_metrics_and_fixed_coverage() -> None:
    score = np.asarray([0.1, 0.9, 0.8, 0.2])
    label = np.asarray([0, 1, 0, 1])
    weight = np.ones(4)

    assert np.isclose(weighted_average_precision(score, label, weight), 5.0 / 6.0)
    assert np.isclose(weighted_roc_auc(score, label, weight), 0.75)
    selected = false_safe_at_coverage(
        score, label, weight, coverage=0.5
    )
    assert selected == {
        "rows": 2,
        "actual_coverage": 0.5,
        "false_safe_rate": 0.5,
        "precision": 0.5,
    }


def test_episode_index_search_fits_projection_without_held_rows(monkeypatch) -> None:
    features = np.zeros((36, 199), dtype=np.float64)
    task = np.repeat(np.arange(3), 12)
    episode = np.tile(np.repeat(np.arange(3), 4), 3)
    teacher = np.tile(np.asarray([11, 13, 27, 27]), 9)
    features[:, 0] = np.tile(np.asarray([2.0, 1.0, -1.0, -2.0]), 9)
    features[:, -1] = episode
    projection_train_episodes: list[set[int]] = []

    def fake_projection(values, weights, *, maximum_rank):
        assert values.shape[0] == weights.shape[0]
        assert maximum_rank == 4
        projection_train_episodes.append(set(values[:, -1].astype(int).tolist()))
        return object()

    class DummyRouter:
        def probabilities(self, values):
            safe13 = np.clip(0.5 + 0.15 * values[:, 0], 0.01, 0.99)
            safe11 = np.minimum(
                np.clip(0.35 + 0.15 * values[:, 0], 0.01, 0.99), safe13
            )
            return np.stack((safe11, safe13), axis=1)

    monkeypatch.setattr(training, "fit_route_first_projection", fake_projection)
    monkeypatch.setattr(
        training, "_fit_from_projection", lambda *args, **kwargs: DummyRouter()
    )
    result = episode_index_candidate_search(
        features,
        teacher,
        task,
        episode,
        candidates=(RouteFirstCandidate(4, 1.0), RouteFirstCandidate(2, 1.0)),
    )

    assert projection_train_episodes == [{1, 2}, {0, 2}, {0, 1}]
    assert result["selected"] == RouteFirstCandidate(2, 1.0)
    assert result["selected_scores"].shape == (36, 2)
    assert [fold["held_episode_index"] for fold in result["folds"]] == [0, 1, 2]


def test_ranking_gates_preserve_strict_and_non_strict_boundaries() -> None:
    def metrics(lift11, lift13, false11, false13):
        return {
            "safe11": {
                "average_precision_lift": lift11,
                "coverage": {"0.01": {"false_safe_rate": false11}},
            },
            "safe13": {
                "average_precision_lift": lift13,
                "coverage": {"0.05": {"false_safe_rate": false13}},
            },
        }

    thresholds = {
        "episode_oof_safe11_ap_lift_strictly_above": 1.25,
        "episode_oof_safe13_ap_lift_strictly_above": 1.25,
        "task_oof_safe11_ap_lift_strictly_above": 1.05,
        "task_oof_safe13_ap_lift_strictly_above": 1.05,
        "episode_oof_safe11_false_safe_at_1pct_at_most": 0.5,
        "episode_oof_safe13_false_safe_at_5pct_at_most": 0.5,
        "task_oof_safe11_false_safe_at_1pct_at_most": 0.75,
        "task_oof_safe13_false_safe_at_5pct_at_most": 0.65,
    }
    gates = ranking_gates(
        metrics(1.25, 1.26, 0.5, 0.49),
        metrics(1.06, 1.05, 0.75, 0.65),
        thresholds,
    )

    assert gates == {
        "episode_safe11_ap_lift": False,
        "episode_safe13_ap_lift": True,
        "task_safe11_ap_lift": True,
        "task_safe13_ap_lift": False,
        "episode_safe11_false_safe_at_1pct": True,
        "episode_safe13_false_safe_at_5pct": True,
        "task_safe11_false_safe_at_1pct": True,
        "task_safe13_false_safe_at_5pct": True,
    }
