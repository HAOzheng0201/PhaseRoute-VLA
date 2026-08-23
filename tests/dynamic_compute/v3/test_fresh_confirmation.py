from __future__ import annotations

from pathlib import Path

import pytest

from a1.vla.dynamic_compute.v3 import fresh_confirmation as fc


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_frozen_contract_keeps_confirmation_shadow_and_test_sealed() -> None:
    contract = fc.load_d8_contract(REPO_ROOT)
    assert contract["fresh_schedule"]["required_clusters"] == 200
    assert contract["D7_final_router_finalization"]["head_count"] == 5
    assert contract["D7_final_router_finalization"]["lambda"] == 0.01
    assert contract["confirmation_gate"]["minimum_safe_clusters"] == 120
    assert contract["scope"]["shadow_decision_only"] is True
    assert contract["authorization"]["open_episode_40_49_authorized"] is False
    assert (
        contract["authorization"][
            "fresh_policy_rollout_authorized_on_contract_validation_alone"
        ]
        is False
    )


def test_fresh_schedule_expands_to_200_unique_non_episode_records() -> None:
    records = fc.load_fresh_confirmation_schedule(REPO_ROOT)
    assert len(records) == 200
    assert len({record.cluster_key for record in records}) == 200
    assert len({record.state_seed for record in records}) == 200
    assert len({record.policy_seed for record in records}) == 200
    assert not ({record.state_seed for record in records} & {record.policy_seed for record in records})
    assert records[0] == fc.FreshConfirmationRecord(
        task_id=0,
        replicate_id=0,
        state_seed=30_260_821,
        policy_seed=40_260_821,
    )
    assert records[-1] == fc.FreshConfirmationRecord(
        task_id=9,
        replicate_id=19,
        state_seed=30_350_840,
        policy_seed=40_350_840,
    )
    assert all("episode" not in record.cluster_key for record in records)


def _summary(**overrides: object) -> fc.D8ConfirmationSummary:
    value: dict[str, object] = {
        "total_clusters": 200,
        "clusters_per_task": (20,) * 10,
        "safe_clusters": 200,
        "safe_clusters_per_task": (20,) * 10,
        "policy_calls": 7000,
        "early_exit_calls": 900,
        "early_exit_calls_per_task": (90,) * 10,
        "false_safe_clusters": 3,
        "false_full_action_clusters": 3,
        "false_gripper_calls": 0,
        "severe_false_full_action_clusters": 0,
        "nondegenerate_row_fraction": 1.0,
        "estimated_fm_reduction_fraction": 0.35,
        "all_candidate_rows_and_policy_calls_accounted_for": True,
        "all_predictions_finite": True,
    }
    value.update(overrides)
    return fc.D8ConfirmationSummary(**value)  # type: ignore[arg-type]


def test_confirmation_gate_passes_only_when_all_frozen_checks_pass() -> None:
    summary = _summary()
    assert summary.false_safe_ucb95 == pytest.approx(0.03830970849856934)
    assert summary.passes
    assert all(summary.gate_checks().values())


def test_confirmation_gate_exact_ucb_and_severe_veto_fail_closed() -> None:
    sparse = _summary(
        safe_clusters=120,
        safe_clusters_per_task=(12,) * 10,
        false_safe_clusters=2,
        false_full_action_clusters=2,
    )
    assert sparse.false_safe_ucb95 == pytest.approx(0.05153371368637799)
    assert not sparse.passes
    assert not sparse.gate_checks()["false_safe_exact_ucb95_at_most_5_percent"]

    severe = _summary(severe_false_full_action_clusters=1)
    assert not severe.passes
    assert not severe.gate_checks()[
        "severe_false_full_action_clusters_at_most_zero"
    ]


def test_schedule_seed_geometry_rejects_invalid_identity() -> None:
    with pytest.raises(fc.D8ProtocolError, match="task id"):
        fc.expected_state_seed(10, 0)
    with pytest.raises(fc.D8ProtocolError, match="replicate id"):
        fc.expected_policy_seed(0, 20)
