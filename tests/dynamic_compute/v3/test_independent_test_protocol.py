from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from a1.vla.dynamic_compute.v3 import independent_test_protocol as d9


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_d9_contract_freezes_paired_active_test_without_opening_states() -> None:
    contract = d9.load_d9_contract(REPO_ROOT)
    assert contract["status"] == d9.D9_STATUS
    assert contract["paired_evaluation"]["arms"] == list(d9.D9_ARMS)
    assert contract["paired_evaluation"]["total_required_rollouts"] == 200
    assert contract["paired_evaluation"]["physical_gpu_allowlist"] == [0, 1, 2, 3]
    assert contract["paired_evaluation"]["GPU_4_to_7_allowed"] is False
    assert (
        contract["authorization"]["on_contract_validation_pass"]
        == "D9A_RUNTIME_ADAPTER_IMPLEMENTATION_AND_D8_PARITY_ONLY"
    )
    assert (
        contract["authorization"][
            "test_sample_or_state_access_on_contract_validation_pass"
        ]
        is False
    )


def test_d9_selection_has_exact_pairs_seeds_order_and_arm_balance() -> None:
    records = d9.load_d9_selection_metadata(REPO_ROOT)
    assert len(records) == 100
    assert records[0] == d9.D9TestRecord(0, 40, 20_260_851)
    assert records[-1] == d9.D9TestRecord(9, 49, 20_350_860)
    assert records[0].arm_order == d9.D9_ARMS
    assert records[1].arm_order == tuple(reversed(d9.D9_ARMS))
    assert {record.physical_gpu_index for record in records} == {0, 1, 2, 3}
    assert sum(record.arm_order[0] == d9.D9_ARMS[0] for record in records) == 50


def test_d9_primary_gate_is_success_retention_efficiency_and_safety() -> None:
    gate = d9.load_d9_contract(REPO_ROOT)["primary_gate"]
    assert gate["all_criteria_are_conjunctive"] is True
    assert gate["PhaseRoute_success_rate_at_least"] == 0.75
    assert gate["PhaseRoute_minus_A1_success_rate_at_least"] == -0.05
    assert gate["paired_task_stratified_bootstrap"]["lower_bound_at_least"] == -0.10
    assert gate["measured_FM_calls_per_policy_call_reduction_at_least"] == 0.25
    assert gate["false_safe_cluster_exact_CP_UCB95_at_most"] == 0.05
    assert gate["false_gripper_calls_at_most"] == 0
    assert gate["severe_false_full_action_clusters_at_most"] == 0


def test_d9_selection_rejects_seed_or_order_mutation() -> None:
    path = REPO_ROOT / d9.D9_SELECTION_RELATIVE_PATH
    selection = json.loads(path.read_text(encoding="utf-8"))
    wrong_seed = copy.deepcopy(selection)
    wrong_seed["records"][0]["seed"] += 1
    with pytest.raises(d9.D9ProtocolError, match="record value"):
        d9.records_from_selection(wrong_seed)
    wrong_order = copy.deepcopy(selection)
    wrong_order["records"][0], wrong_order["records"][1] = (
        wrong_order["records"][1],
        wrong_order["records"][0],
    )
    with pytest.raises(d9.D9ProtocolError, match="ordering or coverage"):
        d9.records_from_selection(wrong_order)


def test_d9_validator_source_cannot_open_test_samples_or_run_control() -> None:
    source = (
        REPO_ROOT
        / "scripts/dynamic_compute/v3/validate_v3_d9_independent_test_contract.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "torch.load",
        "numpy.load",
        "pickle.load",
        "get_task_init_states",
        "eval_libero_early_exit",
        "nvidia-smi",
    ):
        assert forbidden not in source
