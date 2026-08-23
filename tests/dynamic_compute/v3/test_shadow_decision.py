from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from a1.vla.dynamic_compute.v3 import shadow_decision as sd  # noqa: E402


def candidate(layer: int, **overrides):
    values = {
        "original_action_consistency": True,
        "motion_safe": True,
        "tail_ucb_safe": True,
        "gripper_score": sd.D4_GRIPPER_THRESHOLD,
    }
    values.update(overrides)
    return sd.ShadowCandidateSignals(layer=layer, **values)


def test_contract_freezes_four_way_and_and_prohibitions() -> None:
    contract = sd.load_d4_contract(REPO_ROOT)
    assert contract["status"] == sd.D4_STATUS
    assert contract["decision"]["priority"] == [11, 13, 27]
    assert contract["decision"]["non_compensating_and"] is True
    assert contract["motion_gate"]["required"] is True
    assert contract["tail_ucb_gate"]["required"] is True
    assert contract["scope"]["active_control_allowed"] is False
    assert contract["scope"]["independent_test_allowed"] is False
    assert contract["scope"][
        "shadow_execution_allowed_before_signal_attestation"
    ] is False


@pytest.mark.parametrize(
    "field,value,reason",
    [
        (
            "original_action_consistency",
            False,
            "failed_original_action_consistency",
        ),
        ("motion_safe", False, "failed_motion_safe"),
        ("tail_ucb_safe", False, "failed_tail_ucb_safe"),
        (
            "gripper_score",
            sd.D4_GRIPPER_THRESHOLD + 1e-12,
            "failed_gripper_safe",
        ),
    ],
)
def test_every_gate_is_non_compensating(field, value, reason) -> None:
    first = candidate(11, **{field: value})
    decision = sd.decide_shadow(first, candidate(13))
    assert first.route_safe is False
    assert reason in first.veto_reasons
    assert decision.selected_layer == 13


def test_priority_is_l11_then_l13_then_defer() -> None:
    both = sd.decide_shadow(candidate(11), candidate(13))
    assert both.selected_layer == 11
    assert both.disposition == "SHADOW_L11"
    only_l13 = sd.decide_shadow(
        candidate(11, motion_safe=False), candidate(13)
    )
    assert only_l13.selected_layer == 13
    defer = sd.decide_shadow(
        candidate(11, motion_safe=False),
        candidate(13, tail_ucb_safe=False),
    )
    assert defer.selected_layer == 27
    assert defer.disposition == "DEFER_L27"
    assert defer.would_early_exit is False


@pytest.mark.parametrize(
    "field",
    [
        "original_action_consistency",
        "motion_safe",
        "tail_ucb_safe",
        "gripper_score",
    ],
)
def test_missing_signals_fail_closed(field: str) -> None:
    first = candidate(11, **{field: None})
    second = candidate(13, **{field: None})
    decision = sd.decide_shadow(first, second)
    assert decision.selected_layer == 27
    assert any(reason.startswith("missing_") for reason in first.veto_reasons)


def test_nonfinite_gripper_scores_fail_closed() -> None:
    for value in (math.nan, math.inf, -math.inf):
        signals = candidate(11, gripper_score=value)
        assert signals.route_safe is False
        assert "nonfinite_gripper_score" in signals.veto_reasons


def test_wrong_layer_order_and_boolean_coercion_are_rejected() -> None:
    with pytest.raises(sd.D4ShadowError):
        candidate(27)
    with pytest.raises(sd.D4ShadowError):
        candidate(11, motion_safe=1)
    with pytest.raises(sd.D4ShadowError):
        sd.decide_shadow(candidate(13), candidate(11))


def test_record_has_no_action_and_summary_accounting_is_exact() -> None:
    decisions = [
        sd.decide_shadow(candidate(11), candidate(13)),
        sd.decide_shadow(candidate(11, motion_safe=False), candidate(13)),
        sd.decide_shadow(
            candidate(11, motion_safe=False),
            candidate(13, tail_ucb_safe=False),
        ),
    ]
    record = decisions[0].to_record()
    assert record["returns_action"] is False
    assert "action" not in record
    summary = sd.summarize_shadow_decisions(decisions)
    assert summary["selection_counts"] == {"11": 1, "13": 1, "27": 1}
    assert summary["estimated_rp_pep_fm_calls"] == 4 + 5 + 7
    assert summary["early_exit_fraction"] == pytest.approx(2 / 3)
    assert summary["active_control"] is False
    assert summary["measured_latency"] is False
