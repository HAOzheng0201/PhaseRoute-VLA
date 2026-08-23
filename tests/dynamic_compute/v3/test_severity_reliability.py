from __future__ import annotations

from pathlib import Path

import pytest
import torch

from a1.vla.dynamic_compute.v3 import severity_reliability as sr


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_contract_admits_reuse_but_forbids_fresh_claim() -> None:
    contract = sr.load_d6_contract(REPO_ROOT)
    assert contract["scope"]["same_development_v2_payload_reused"] is True
    assert contract["scope"]["fresh_confirmation_claim_allowed"] is False
    assert contract["scope"]["independent_test_v2_access_allowed"] is False
    assert contract["scope"]["active_control_allowed"] is False
    assert contract["claim_boundary"]["D6_result_is_unbiased_method_comparison"] is False


def test_frozen_severity_weight_examples_and_cap() -> None:
    threshold = 0.00390625
    distance = torch.tensor(
        [0.0, threshold, 2 * threshold, 4 * threshold, 8 * threshold, 16 * threshold, 64 * threshold],
        dtype=torch.float64,
    )
    weight = sr.severity_weights(distance)
    assert weight.tolist() == pytest.approx([1, 1, 2, 3, 4, 5, 5])
    with pytest.raises(sr.D6SeverityError, match="finite"):
        sr.severity_weights(torch.tensor([float("nan")]))


def test_weighted_anchor_emphasizes_severe_full_action_positive() -> None:
    rows = 40
    features = torch.zeros((rows, 97), dtype=torch.float64)
    layer = torch.tensor([11, 13]).repeat(rows // 2)
    target = torch.zeros((rows, 2), dtype=torch.bool)
    target[0, 0] = True
    target[1, 0] = True
    target[2::4, 1] = True
    target[3::4, 1] = True
    severity = torch.ones(rows, dtype=torch.float64)
    severity[0] = 5.0
    severity[1] = 5.0
    model = sr.fit_severity_weighted_glm(
        features,
        layer,
        target,
        severity,
        torch.ones(rows, dtype=torch.bool),
        l2_lambda=0.1,
        max_iterations=20,
    )
    unweighted_l11 = float(target[layer == 11, 0].double().mean())
    assert float(model.anchor_score[0, 0]) > unweighted_l11
    assert model.weight.shape == (2, 97)
    assert torch.equal(model.weight, torch.zeros_like(model.weight))


def test_robust_threshold_uses_fifth_smallest_then_fixed_shrink() -> None:
    identities = [(task, episode) for task in range(10) for episode in range(12, 29)]
    calls = len(identities)
    task = torch.tensor([value[0] for value in identities], dtype=torch.long)
    episode = torch.tensor([value[1] for value in identities], dtype=torch.long)
    full = torch.ones((calls, 2), dtype=torch.float64)
    full[:, 0] = torch.tensor(
        [0.01 + (ep - 12) * 0.001 + task_id * 1.0e-5 for task_id, ep in identities],
        dtype=torch.float64,
    )
    gripper = torch.zeros_like(full)
    consistency = torch.zeros((calls, 2), dtype=torch.bool)
    consistency[:, 0] = True
    unsafe = torch.zeros((calls, 2, 2), dtype=torch.bool)
    selection = sr.robust_threshold_selection(
        full, gripper, consistency, unsafe, task, episode
    )
    assert selection.feasible is True
    assert len(selection.jackknife_thresholds) == 17
    ordered = sorted(value for _, value in selection.jackknife_thresholds)
    assert selection.order_statistic_threshold == ordered[4]
    assert selection.pre_shrink_threshold == min(
        selection.full_threshold, selection.order_statistic_threshold
    )
    assert selection.runtime_threshold == pytest.approx(
        0.95 * selection.pre_shrink_threshold
    )
    assert selection.runtime_summary is not None
    assert selection.runtime_summary.safe_clusters >= 60
