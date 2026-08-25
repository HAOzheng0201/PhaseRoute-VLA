from __future__ import annotations

import numpy as np

from a1.vla.dynamic_compute.route_first_router import (
    ROUTE_FIRST_ROUTER_CALIBRATION_STATUS,
    fit_route_first_ordinal_router,
    load_uncalibrated_route_first_router,
    route_first_group_weights,
    save_uncalibrated_route_first_router,
)


def _synthetic(rows: int = 240):
    rng = np.random.default_rng(20260825)
    features = rng.normal(size=(rows, 199)).astype(np.float32)
    latent = features[:, 0] + 0.7 * features[:, 1] - 0.4 * features[:, 2]
    teacher = np.where(latent > 0.8, 11, np.where(latent > -0.3, 13, 27))
    task = np.repeat(np.arange(6), rows // 6)
    episode = np.tile(np.repeat(np.arange(4), rows // 24), 6)
    return features, teacher, task, episode


def test_group_weights_give_each_episode_equal_mass() -> None:
    task = np.asarray([0, 0, 0, 0, 1, 1])
    episode = np.asarray([0, 0, 0, 1, 0, 0])
    weights = route_first_group_weights(task, episode)

    masses = {
        (task_id, episode_id): float(
            weights[(task == task_id) & (episode == episode_id)].sum()
        )
        for task_id, episode_id in {(0, 0), (0, 1), (1, 0)}
    }
    assert len(set(round(value, 12) for value in masses.values())) == 1
    assert np.isclose(weights.mean(), 1.0)


def test_ordinal_router_is_deterministic_nested_and_separates_signal() -> None:
    features, teacher, task, episode = _synthetic()
    first = fit_route_first_ordinal_router(
        features,
        teacher,
        task,
        episode,
        pca_rank=16,
        l2=1.0,
    )
    second = fit_route_first_ordinal_router(
        features,
        teacher,
        task,
        episode,
        pca_rank=16,
        l2=1.0,
    )

    np.testing.assert_array_equal(first.head11.weight, second.head11.weight)
    np.testing.assert_array_equal(first.head13.weight, second.head13.weight)
    probability = first.probabilities(features)
    assert probability.shape == (features.shape[0], 2)
    assert np.all(probability[:, 0] <= probability[:, 1])
    safe13 = teacher <= 13
    assert probability[safe13, 1].mean() > probability[~safe13, 1].mean()
    safe11 = teacher == 11
    assert probability[safe11, 0].mean() > probability[~safe11, 0].mean()


def test_uncalibrated_router_roundtrip_has_no_thresholds(tmp_path) -> None:
    features, teacher, task, episode = _synthetic()
    router = fit_route_first_ordinal_router(
        features,
        teacher,
        task,
        episode,
        pca_rank=12,
        l2=3.0,
    )
    path = tmp_path / "route_first_router.npz"
    save_uncalibrated_route_first_router(
        path,
        router,
        training_payload_sha256="a" * 64,
        training_file_sha256="b" * 64,
        task_ids=range(6),
        episode_indices=range(4),
        seed=20260825,
    )

    loaded, metadata = load_uncalibrated_route_first_router(path)
    np.testing.assert_allclose(
        loaded.probabilities(features),
        router.probabilities(features),
        rtol=2e-5,
        atol=2e-6,
    )
    assert metadata["calibration_status"] == ROUTE_FIRST_ROUTER_CALIBRATION_STATUS
    assert metadata["training_task_ids"] == list(range(6))
    with np.load(path, allow_pickle=False) as arrays:
        assert "threshold11" not in arrays.files
        assert "threshold13" not in arrays.files
