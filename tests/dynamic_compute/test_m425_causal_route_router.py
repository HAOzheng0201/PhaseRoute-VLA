import numpy as np
import torch

from a1.vla.dynamic_compute.causal_route_router import (
    CausalRouteRouter,
    CausalRouteRouterConfig,
    calibrate_zero_false_positive,
    fit_pca_logistic_affine,
    route_metrics,
    save_router_npz,
    sequential_routes,
)


def test_affine_fit_is_deterministic_and_separates_synthetic_data(tmp_path):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 12))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(np.int64)
    first = fit_pca_logistic_affine(features, labels, pca_rank=12, l2=1.0)
    second = fit_pca_logistic_affine(features, labels, pca_rank=12, l2=1.0)
    np.testing.assert_array_equal(first.weight, second.weight)
    assert first.bias == second.bias
    assert np.mean((first.probabilities(features) >= 0.5) == labels) >= 0.85

    path = tmp_path / "router.npz"
    save_router_npz(
        path,
        head11=first,
        head13=first,
        threshold11=0.6,
        threshold13=0.7,
    )
    router = CausalRouteRouter.from_npz(path)
    torch_prob = router.probability(11, torch.from_numpy(features.astype(np.float32)))
    np.testing.assert_allclose(
        torch_prob.detach().numpy(),
        first.probabilities(features),
        rtol=2e-5,
        atol=2e-6,
    )


def test_zero_false_positive_calibration_and_sequential_fail_closed():
    probabilities11 = np.asarray([0.9, 0.8, 0.3, 0.4, 0.2])
    labels11 = np.asarray([1, 1, 0, 0, 0])
    threshold11 = calibrate_zero_false_positive(probabilities11, labels11)
    assert threshold11 > 0.4
    assert not np.any(probabilities11[labels11 == 0] >= threshold11)

    probabilities13 = np.asarray([0.1, 0.2, 0.9, 0.8, 0.3])
    routes = sequential_routes(
        probabilities11,
        probabilities13,
        threshold11=threshold11,
        threshold13=0.7,
    )
    assert routes.tolist() == [11, 11, 13, 13, 27]
    metrics = route_metrics(routes, np.asarray([11, 11, 13, 27, 27]))
    assert metrics["false_shallow"] == 1
    assert metrics["teacher27_false_shallow"] == 1
    assert metrics["predicted_distribution"] == {"11": 2, "13": 2, "27": 1}


def test_router_rejects_unsupported_layer():
    head = fit_pca_logistic_affine(
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [-1.0, 2.0]]),
        np.asarray([0, 0, 1, 1]),
        pca_rank=1,
    )
    router = CausalRouteRouter(
        config=CausalRouteRouterConfig(hidden_dim=2),
        weight11=torch.from_numpy(head.weight.astype(np.float32)),
        bias11=head.bias,
        weight13=torch.from_numpy(head.weight.astype(np.float32)),
        bias13=head.bias,
    )
    try:
        router.probability(12, torch.zeros(1, 2))
    except ValueError as error:
        assert "layer 11 or 13" in str(error)
    else:
        raise AssertionError("unsupported layer should fail closed")
