from __future__ import annotations
from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from a1.vla.dynamic_compute.v3 import gripper_v2_models as gm  # noqa: E402


def _synthetic(rows: int = 160):
    generator = torch.Generator().manual_seed(20260820)
    features = torch.randn(rows, 97, generator=generator)
    layers = torch.tensor([11, 13], dtype=torch.long).repeat(rows // 2)
    step_score = 0.9 * features[:, 82] - 0.7 * features[:, 90]
    transition_score = 0.8 * features[:, 89] + 0.6 * features[:, 96]
    occurrence = torch.stack((step_score > 0, transition_score > 0), dim=1)
    step_count = torch.where(
        occurrence[:, 0], 1 + torch.remainder((step_score.abs() * 3).long(), 8), 0
    )
    transition_count = torch.where(
        occurrence[:, 1],
        1 + torch.remainder((transition_score.abs() * 3).long(), 7),
        0,
    )
    count = torch.stack((step_count, transition_count), dim=1).long()
    fit = torch.ones(rows, dtype=torch.bool)
    return features, layers, occurrence, count, fit


def test_zt_binomial_anchor_and_probabilities_are_valid() -> None:
    for support in (7, 8):
        for mean in (1.1, 2.5, float(support) - 0.1):
            probability = gm.zt_binomial_anchor_probability(mean, support)
            values = gm.zero_truncated_binomial_probabilities(
                torch.tensor([probability], dtype=torch.float64), support
            )
            expected = gm.expected_positive_count(values).item()
            assert expected == pytest.approx(mean, abs=1e-8)
            assert values.sum().item() == pytest.approx(1.0)
            assert bool((values > 0).all())


def test_ordered_cutpoints_and_probabilities_are_strict() -> None:
    raw_base = torch.tensor([-1.0, 0.5], dtype=torch.float64)
    raw_increment = torch.zeros((2, 6), dtype=torch.float64)
    cutpoints = gm.ordered_cutpoints(raw_base, raw_increment)
    assert cutpoints.shape == (2, 7)
    assert bool((cutpoints[:, 1:] > cutpoints[:, :-1]).all())
    score = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    layer = torch.tensor([0, 1, 0], dtype=torch.long)
    probability = gm.ordinal_probabilities(score, layer, cutpoints)
    assert probability.shape == (3, 8)
    assert torch.allclose(probability.sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert bool((probability > 0).all())


def test_occurrence_fit_uses_zero_initialized_no_bias_residual() -> None:
    features, layers, occurrence, _, fit = _synthetic()
    model = gm.fit_occurrence_glm(
        features,
        layers,
        occurrence,
        fit,
        l2_lambda=0.01,
        max_iterations=80,
    )
    probability = model.predict(features, layers)
    assert model.weight.shape == (2, 97)
    assert probability.shape == (features.shape[0], 2)
    assert bool(((probability > 0) & (probability < 1)).all())
    assert torch.nn.functional.binary_cross_entropy(
        probability.float(), occurrence.float()
    ) < 0.45


@pytest.mark.parametrize("target_index", [0, 1])
def test_count_models_fit_positive_support_and_sum_to_one(target_index: int) -> None:
    features, layers, _, count, fit = _synthetic()
    baseline = gm.fit_zt_binomial_glm(
        features,
        layers,
        count,
        fit,
        target_index=target_index,
        l2_lambda=0.01,
        max_iterations=60,
    )
    ordinal = gm.fit_ordinal_glm(
        features,
        layers,
        count,
        fit,
        target_index=target_index,
        l2_lambda=0.01,
        max_iterations=60,
    )
    baseline_probability = baseline.probabilities(features, layers)
    ordinal_probability = ordinal.probabilities(features, layers)
    support = gm.COUNT_SUPPORT_MAX[target_index]
    assert baseline_probability.shape == (features.shape[0], support)
    assert ordinal_probability.shape == (features.shape[0], support)
    assert torch.allclose(
        baseline_probability.sum(dim=1),
        torch.ones(features.shape[0], dtype=torch.float64),
    )
    assert torch.allclose(
        ordinal_probability.sum(dim=1),
        torch.ones(features.shape[0], dtype=torch.float64),
    )
    assert ordinal.cutpoints.shape == (2, support - 1)
    assert bool((ordinal.cutpoints[:, 1:] > ordinal.cutpoints[:, :-1]).all())


def test_conditional_metrics_and_tie_aware_auroc() -> None:
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64
    )
    count = torch.tensor([1, 3], dtype=torch.long)
    assert bool((gm.conditional_nll(probabilities, count) > 0).all())
    assert bool((gm.ranked_probability_score(probabilities, count) >= 0).all())
    assert gm.tie_aware_auroc(
        torch.tensor([0.0, 0.5, 0.5, 1.0]),
        torch.tensor([False, False, True, True]),
    ) == pytest.approx(0.875)


def test_one_standard_error_selects_largest_eligible_lambda() -> None:
    losses = {
        0.001: torch.tensor([0.90, 1.10, 1.00], dtype=torch.float64),
        0.01: torch.tensor([0.91, 1.09, 1.00], dtype=torch.float64),
        0.1: torch.tensor([0.95, 1.05, 1.00], dtype=torch.float64),
    }
    selected, summary = gm.one_standard_error_choice(losses)
    assert selected == 0.1
    assert summary["0.1"]["cells"] == 3


def test_invalid_lambda_layer_and_zero_count_fail_closed() -> None:
    features, layers, occurrence, count, fit = _synthetic()
    with pytest.raises(gm.GripperV2ModelError, match="lambda"):
        gm.fit_occurrence_glm(
            features, layers, occurrence, fit, l2_lambda=1.0, max_iterations=2
        )
    invalid_layers = layers.clone()
    invalid_layers[0] = 27
    with pytest.raises(gm.GripperV2ModelError, match="11 or 13"):
        gm.fit_occurrence_glm(
            features,
            invalid_layers,
            occurrence,
            fit,
            l2_lambda=0.01,
            max_iterations=2,
        )
    with pytest.raises(gm.GripperV2ModelError, match="positive support"):
        gm.conditional_nll(torch.full((2, 3), 1 / 3), torch.tensor([0, 1]))
