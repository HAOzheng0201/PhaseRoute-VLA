from __future__ import annotations

import torch

from a1.vla.dynamic_compute.v3.final_router import (
    FinalFiveHeadRouter,
    FinalRouterError,
    final_router_from_mapping,
    final_router_state,
    severity_fit_from_state,
)
from a1.vla.dynamic_compute.v3.gripper_v2_models import FeatureNormalizer
from a1.vla.dynamic_compute.v3.severity_reliability import SeverityWeightedFit


def _model(offset: float) -> SeverityWeightedFit:
    return SeverityWeightedFit(
        normalizer=FeatureNormalizer(
            mean=torch.zeros(97, dtype=torch.float64),
            scale=torch.ones(97, dtype=torch.float64),
        ),
        anchor_score=torch.tensor(
            [[0.1 + offset, 0.2], [0.3 + offset, 0.4]], dtype=torch.float64
        ),
        weight=torch.zeros((2, 97), dtype=torch.float64),
        l2_lambda=0.01,
        final_loss=0.5 + offset,
    )


def test_final_router_round_trip_preserves_predictions() -> None:
    router = FinalFiveHeadRouter(
        models=tuple(_model(0.01 * index) for index in range(5)),
        full_threshold=0.5,
        runtime_threshold=0.475,
    )
    state = final_router_state(router)
    loaded = final_router_from_mapping(state)
    features = torch.zeros((4, 97), dtype=torch.float64)
    layers = torch.tensor([11, 13, 11, 13])
    original = router.predict(features, layers)
    restored = loaded.predict(features, layers)
    assert all(torch.equal(left, right) for left, right in zip(original, restored))
    assert torch.allclose(
        restored[1][:, 0],
        torch.tensor([0.14, 0.34, 0.14, 0.34], dtype=torch.float64),
    )
    assert torch.allclose(
        restored[1][:, 1],
        torch.tensor([0.2, 0.4, 0.2, 0.4], dtype=torch.float64),
    )


def test_final_router_rejects_wrong_head_lambda_and_threshold() -> None:
    state = final_router_state(
        FinalFiveHeadRouter(
            models=tuple(_model(0.0) for _ in range(5)),
            full_threshold=0.4,
            runtime_threshold=0.38,
        )
    )
    state["head_states"][0]["l2_lambda"] = 0.1
    try:
        severity_fit_from_state(state["head_states"][0])
    except FinalRouterError:
        pass
    else:
        raise AssertionError("wrong D8B lambda was accepted")

    state["head_states"][0]["l2_lambda"] = 0.01
    state["runtime_threshold"] = 0.39
    try:
        final_router_from_mapping(state)
    except FinalRouterError:
        pass
    else:
        raise AssertionError("wrong D8B threshold shrink was accepted")
