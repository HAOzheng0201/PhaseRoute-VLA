from types import SimpleNamespace

import numpy as np
import torch

from scripts.dynamic_compute.collect_m425_causal_route_features import (
    collect_one,
    extract_proprio_hidden,
)
from scripts.dynamic_compute.evaluate_m425_causal_router import evaluate_predictions
from scripts.dynamic_compute.train_m425_causal_router import fit_grouped_router


def test_feature_extractor_uses_proprio_position_and_frozen_hidden_indices():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    selected = extract_proprio_hidden(hidden, torch.tensor([[1], [4]]))
    torch.testing.assert_close(selected[0], hidden[0, 1])
    torch.testing.assert_close(selected[1], hidden[1, 4])

    class DummyModel:
        def forward(self, **kwargs):
            assert kwargs["output_hidden_states"] is True
            assert kwargs["exit_id"] == 13
            states = tuple(
                torch.full((1, 5, 3), float(index)) for index in range(14)
            )
            return SimpleNamespace(
                exit_layer=13,
                attn_key_values=[(torch.zeros(1), torch.zeros(1))] * 14,
                hidden_states=states,
                last_hidden_state=torch.full((1, 5, 3), 13.0),
            )

    layer11, layer13 = collect_one(
        DummyModel(),
        {
            "proprio_token_idx": torch.tensor([[2]]),
            "output_hidden_states": False,
        },
        amp_enabled=False,
    )
    torch.testing.assert_close(layer11, torch.full((1, 3), 12.0))
    torch.testing.assert_close(layer13, torch.full((1, 3), 13.0))


def test_grouped_fit_never_predicts_oof_shallower_on_separable_tasks():
    rng = np.random.default_rng(11)
    features11 = []
    features13 = []
    routes = []
    tasks = []
    for task in range(8):
        for route, center in ((11, -3.0), (13, 0.0), (27, 3.0)):
            for _ in range(3):
                vector = rng.normal(scale=0.1, size=16)
                vector[0] += center
                vector[1] += -3.0 if route == 13 else 3.0
                features11.append(vector)
                features13.append(vector)
                routes.append(route)
                tasks.append(task)
    fit = fit_grouped_router(
        np.asarray(features11),
        np.asarray(features13),
        np.asarray(routes),
        np.asarray(tasks),
        pca_rank=8,
    )
    assert fit["development_rows"] == 72
    assert fit["oof_metrics"]["false_shallow"] == 0
    assert fit["threshold11"] >= 0.5
    assert fit["threshold13"] >= 0.5


def test_estimated_latency_uses_frozen_route_means():
    predicted = np.asarray([11, 13, 27, 11])
    teacher = np.asarray([11, 13, 27, 13])
    metrics = evaluate_predictions(
        predicted,
        teacher,
        route_latency_ms={11: 2.0, 13: 3.0, 27: 5.0},
        full_latency_ms=5.0,
    )
    assert metrics["false_shallow"] == 1
    assert metrics["estimated_cuda_latency_ms"]["mean"] == 3.0
    assert metrics["estimated_reduction_vs_full_mean"] == 0.4
