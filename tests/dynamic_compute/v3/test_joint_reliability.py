from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from a1.vla.dynamic_compute.v3 import joint_reliability as jr


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_frozen_contract_is_development_only_and_non_compensating() -> None:
    contract = jr.load_d5_contract(REPO_ROOT)
    assert contract["scope"]["pre_target_inspection_design"] is True
    assert contract["scope"]["d5_joint_target_distribution_opened_at_freeze"] is False
    assert contract["scope"]["calibration_v2_reuse_for_repair_or_selection"] is False
    assert contract["scope"]["independent_test_v2_access_allowed"] is False
    assert contract["scope"]["active_control_allowed"] is False
    assert contract["routing"]["non_compensating_and"] is True
    assert contract["routing"]["legacy_motion_and_tail"] == (
        "diagnostic_only_not_runtime_hard_veto"
    )


def test_route_priority_and_fail_closed_signals() -> None:
    full = torch.tensor([[0.01, 0.02], [float("nan"), 0.01]], dtype=torch.float64)
    gripper = torch.tensor([[0.01, 0.01], [0.01, 0.01]], dtype=torch.float64)
    consistency = torch.tensor([[True, True], [True, True]])
    selected = jr.route_at_threshold(
        full, gripper, consistency, threshold=0.015
    )
    assert selected.tolist() == [11, 13]
    gripper[1, 1] = jr.D5_GRIPPER_THRESHOLD + 1.0e-3
    selected = jr.route_at_threshold(
        full, gripper, consistency, threshold=0.015
    )
    assert selected.tolist() == [11, 27]


def test_all_three_gates_are_required() -> None:
    full = torch.zeros((3, 2), dtype=torch.float64)
    gripper = torch.zeros((3, 2), dtype=torch.float64)
    consistency = torch.ones((3, 2), dtype=torch.bool)
    consistency[0, 0] = False
    gripper[1, 0] = jr.D5_GRIPPER_THRESHOLD + 1.0e-6
    full[2, 0] = 0.2
    selected = jr.route_at_threshold(
        full, gripper, consistency, threshold=0.1
    )
    assert selected.tolist() == [13, 13, 13]


def test_action_distance_matches_horizon_mean_cosine() -> None:
    candidate = torch.zeros((2, 8, 7), dtype=torch.float32)
    reference = torch.zeros_like(candidate)
    candidate[0, :, 0] = 1.0
    reference[0, :, 0] = 1.0
    candidate[1, :, 0] = 1.0
    reference[1, :, 1] = 1.0
    distance = jr.mean_action_cosine_distance(candidate, reference)
    assert distance.tolist() == pytest.approx([0.0, 1.0])


def test_exact_threshold_selection_avoids_one_late_false_cluster() -> None:
    calls = 61
    scores = torch.full((calls, 2), 2.0, dtype=torch.float64)
    scores[:, 0] = torch.arange(1, calls + 1, dtype=torch.float64) / 100.0
    gripper = torch.zeros_like(scores)
    consistency = torch.ones((calls, 2), dtype=torch.bool)
    unsafe = torch.zeros((calls, 2, 2), dtype=torch.bool)
    unsafe[-1, 0, 0] = True
    task = torch.tensor([index % 10 for index in range(calls)], dtype=torch.long)
    episode = torch.tensor([12 + index // 10 for index in range(calls)], dtype=torch.long)
    selection = jr.select_inner_threshold(
        scores, gripper, consistency, unsafe, task, episode
    )
    assert selection.feasible is True
    assert selection.threshold == pytest.approx(0.60)
    assert selection.summary is not None
    assert selection.summary.early_exit_calls == 60
    assert selection.summary.safe_clusters == 60
    assert selection.summary.false_safe_clusters == 0
    assert selection.summary.false_safe_ucb95 <= 0.05


def test_contract_file_is_valid_json_without_nan() -> None:
    path = REPO_ROOT / jr.D5_CONTRACT_RELATIVE_PATH
    raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    assert value["model"]["trainable_feature_parameter_count"] == 194
    assert "NaN" not in raw
