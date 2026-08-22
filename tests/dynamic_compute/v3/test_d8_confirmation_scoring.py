from __future__ import annotations

import torch

from a1.vla.dynamic_compute.v3 import d8_confirmation_scoring as d8d


def _payload() -> dict:
    calls = 200
    task = torch.arange(10).repeat_interleave(20)
    replicate = torch.arange(20).repeat(10)
    keys = [
        f"libero_10:task{int(task[i])}:fresh_confirm_v1:replicate{int(replicate[i])}"
        for i in range(calls)
    ]
    return {
        "features": torch.zeros((2 * calls, 97)),
        "candidate_layer": torch.tensor([11, 13]).repeat(calls),
        "source_row": torch.arange(calls).repeat_interleave(2),
        "task_id": task.repeat_interleave(2),
        "replicate_id": replicate.repeat_interleave(2),
        "cluster_keys": [key for key in keys for _ in range(2)],
        "call_ordinal": torch.zeros(calls, dtype=torch.long).repeat_interleave(2),
        "step_id": torch.zeros(calls, dtype=torch.long).repeat_interleave(2),
        "action_consistency": torch.ones(2 * calls, dtype=torch.bool),
        "unsafe_target": torch.zeros((2 * calls, 2), dtype=torch.bool),
        "full_action_distance": torch.zeros(2 * calls, dtype=torch.float64),
    }


def _prediction(rows: int) -> torch.Tensor:
    prediction = torch.empty((5, rows, 2), dtype=torch.float64)
    for head in range(5):
        prediction[head, :, 0] = 0.10 + 0.01 * head
        prediction[head, :, 1] = 0.01 + 0.01 * head
    return prediction


def _score(payload: dict) -> d8d.D8ScoringResult:
    data = d8d.confirmation_data_from_mapping(payload)
    return d8d.score_frozen_router_predictions(
        data,
        _prediction(data.rows),
        runtime_threshold=0.5,
        gripper_threshold=0.05,
        action_consistency_threshold=0.00390625,
        behavior_fm_calls=1400,
    )


def test_d8d_scoring_applies_l11_priority_and_passes_exact_gate() -> None:
    scored = _score(_payload())
    assert torch.equal(scored.selected_layer, torch.full((200,), 11))
    assert scored.summary.safe_clusters == 200
    assert scored.summary.early_exit_calls == 200
    assert scored.summary.nondegenerate_row_fraction == 1.0
    assert scored.summary.estimated_fm_reduction_fraction > 0.4
    checks = scored.summary.gate_checks()
    assert all(checks.values())


def test_d8d_three_small_false_full_clusters_pass_but_four_fail_veto() -> None:
    payload = _payload()
    for call in range(3):
        row = 2 * call
        payload["unsafe_target"][row, 0] = True
        payload["full_action_distance"][row] = 0.004
    three = _score(payload)
    assert three.summary.false_safe_clusters == 3
    assert three.summary.false_full_action_clusters == 3
    assert three.summary.severe_false_full_action_clusters == 0
    assert all(three.summary.gate_checks().values())

    row = 2 * 3
    payload["unsafe_target"][row, 0] = True
    payload["full_action_distance"][row] = 0.004
    four = _score(payload)
    assert four.summary.false_safe_clusters == 4
    assert four.summary.false_safe_ucb95 < 0.05
    assert not four.summary.gate_checks()[
        "false_full_action_clusters_at_most_three"
    ]


def test_d8d_prefers_safe_l13_and_records_severe_and_gripper_vetoes() -> None:
    payload = _payload()
    data = d8d.confirmation_data_from_mapping(payload)
    prediction = _prediction(data.rows)
    prediction[:, 0, 0] = 0.9
    scored = d8d.score_frozen_router_predictions(
        data,
        prediction,
        runtime_threshold=0.5,
        gripper_threshold=0.05,
        action_consistency_threshold=0.00390625,
        behavior_fm_calls=1400,
    )
    assert scored.selected_layer[0] == 13

    payload["unsafe_target"][1, 0] = True
    payload["unsafe_target"][1, 1] = True
    payload["full_action_distance"][1] = 0.02
    data = d8d.confirmation_data_from_mapping(payload)
    failed = d8d.score_frozen_router_predictions(
        data,
        prediction,
        runtime_threshold=0.5,
        gripper_threshold=0.05,
        action_consistency_threshold=0.00390625,
        behavior_fm_calls=1400,
    )
    assert failed.summary.false_gripper_calls == 1
    assert failed.summary.severe_false_full_action_clusters == 1
    checks = failed.summary.gate_checks()
    assert not checks["false_gripper_calls_at_most_zero"]
    assert not checks["severe_false_full_action_clusters_at_most_zero"]


def test_d8d_rejects_pair_identity_drift() -> None:
    payload = _payload()
    payload["candidate_layer"][1] = 11
    try:
        d8d.confirmation_data_from_mapping(payload)
    except d8d.D8DScoringError:
        pass
    else:
        raise AssertionError("D8D accepted malformed candidate pairing")
