from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from a1.vla.dynamic_compute.v3 import shadow_signal_adapter as sa  # noqa: E402


def synthetic_state() -> sa.FrozenLegacySignalState:
    return sa.FrozenLegacySignalState(
        feature_mean=torch.zeros((2, 82), dtype=torch.float64),
        feature_scale=torch.ones((2, 82), dtype=torch.float64),
        motion_anchor=sa.D4A_MOTION_THRESHOLDS.clone(),
        tail_anchor=sa.D4A_TAIL_ANCHORS[:, None].clone(),
        motion_weight=torch.zeros((2, 82), dtype=torch.float64),
        motion_correction=torch.zeros((2, 2), dtype=torch.float64),
        tail_weight=torch.zeros((1, 82), dtype=torch.float64),
        tail_model_correction=torch.zeros((2, 1), dtype=torch.float64),
        tail_conformal_correction=sa.D4A_TAIL_CORRECTIONS.clone(),
    )


def test_zero_residual_hits_frozen_boundaries_and_passes() -> None:
    features = torch.zeros((4, 97), dtype=torch.float32)
    layers = torch.tensor([11, 13, 11, 13])
    signals = sa.adapt_shadow_signals(synthetic_state(), features, layers)
    assert torch.equal(
        signals.motion_prediction,
        sa.D4A_MOTION_THRESHOLDS[[0, 1, 0, 1]],
    )
    assert torch.equal(
        signals.tail_upper, sa.D4A_TAIL_BUDGETS[[0, 1, 0, 1]]
    )
    assert bool(signals.motion_safe.all())
    assert bool(signals.tail_ucb_safe.all())


def test_positive_residual_vetoes_without_compensation() -> None:
    state = synthetic_state()
    state = sa.FrozenLegacySignalState(
        **{
            **state.__dict__,
            "motion_weight": torch.ones((2, 82), dtype=torch.float64),
            "tail_weight": torch.ones((1, 82), dtype=torch.float64),
        }
    )
    features = torch.ones((2, 97), dtype=torch.float32)
    layers = torch.tensor([11, 13])
    signals = sa.adapt_shadow_signals(state, features, layers)
    assert not bool(signals.motion_safe.any())
    assert not bool(signals.tail_ucb_safe.any())


def test_suffix_never_changes_legacy_predictions() -> None:
    first = torch.zeros((2, 97), dtype=torch.float32)
    second = first.clone()
    second[:, 82:] = torch.arange(15, dtype=torch.float32)
    layers = torch.tensor([11, 13])
    left = sa.adapt_shadow_signals(synthetic_state(), first, layers)
    right = sa.adapt_shadow_signals(synthetic_state(), second, layers)
    assert torch.equal(left.motion_prediction, right.motion_prediction)
    assert torch.equal(left.tail_upper, right.tail_upper)


def test_nonfinite_shape_layer_and_threshold_source_drift_fail() -> None:
    features = torch.zeros((2, 97), dtype=torch.float32)
    layers = torch.tensor([11, 13])
    with pytest.raises(sa.D4ASignalError):
        sa.adapt_shadow_signals(synthetic_state(), features[:, :96], layers)
    features[0, 0] = torch.nan
    with pytest.raises(sa.D4ASignalError):
        sa.adapt_shadow_signals(synthetic_state(), features, layers)
    with pytest.raises(sa.D4ASignalError):
        sa.adapt_shadow_signals(
            synthetic_state(), torch.zeros((2, 97)), torch.tensor([11, 27])
        )
    state = synthetic_state()
    state = sa.FrozenLegacySignalState(
        **{
            **state.__dict__,
            "motion_anchor": state.motion_anchor + 1e-12,
        }
    )
    with pytest.raises(sa.D4ASignalError):
        state.validate()


def test_action_cosine_distance_matches_a1_horizon_reduction() -> None:
    first = torch.zeros((2, 8, 7), dtype=torch.float32)
    second = torch.zeros_like(first)
    first[:, :, 0] = 1.0
    second[0, :, 0] = 1.0
    second[1, :, 1] = 1.0
    distance = sa.mean_action_cosine_distance(
        first.contiguous(), second.contiguous()
    )
    assert torch.equal(distance, torch.tensor([0.0, 1.0]))
    with pytest.raises(sa.D4ASignalError):
        sa.mean_action_cosine_distance(first.double(), second.double())
