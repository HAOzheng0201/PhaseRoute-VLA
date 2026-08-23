from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / (
    "configs/research/v3/gripper_v2/d4a_signal_adapter_contract.json"
)


def load_contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_thresholds_are_frozen_before_shadow_and_not_searchable() -> None:
    contract = load_contract()
    assert contract["status"] == "D4A_SIGNAL_ADAPTER_CONTRACT_FROZEN"
    assert contract["scope"]["pre_shadow_distribution_freeze"] is True
    assert contract["scope"]["formal_shadow_allowed_before_adapter_attestation"] is False
    assert contract["motion_gate"]["threshold_search_on_v3_allowed"] is False
    assert contract["tail_ucb_gate"]["threshold_search_on_v3_allowed"] is False
    assert contract["scope"]["active_control_allowed"] is False
    assert contract["scope"]["independent_test_allowed"] is False


def test_tail_budgets_are_exact_anchor_plus_correction() -> None:
    tail = load_contract()["tail_ucb_gate"]
    for layer in (11, 13):
        key = str(layer)
        assert tail["tail_budgets"][key] == (
            tail["q90_anchors"][key] + tail["conformal_corrections"][key]
        )


def test_adapter_cannot_use_teacher_truth_or_identity_at_runtime() -> None:
    contract = load_contract()
    assert contract["feature_adapter"]["legacy_slice"] == [0, 82]
    assert contract["feature_adapter"]["task_episode_identity_runtime_visible"] is False
    assert contract["feature_adapter"]["layer27_runtime_visible"] is False
    assert contract["formal_shadow_truth"]["truth_used_for_decision"] is False
    assert contract["action_consistency_adapter"][
        "candidate_to_l27_truth_is_runtime_visible"
    ] is False
