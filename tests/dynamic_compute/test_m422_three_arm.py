import pytest

from scripts.dynamic_compute.summarize_m422_three_arm import build_summary


def _result(policy, outcomes, indices=(27, 28)):
    if policy == "full_depth":
        model_class = "a1.vla.affordvla.AffordVLA"
        early_enabled = False
        productive_enabled = False
    else:
        model_class = "a1.vla.affordvla_early_exit.AffordVLAEarlyExit"
        early_enabled = True
        productive_enabled = policy == "rp_pep"
    rows = []
    for episode_idx, success in zip(indices, outcomes, strict=True):
        rows.append(
            {
                "status": "PASS",
                "task_id": 0,
                "episode_idx": episode_idx,
                "episode_seed": 100 + episode_idx,
                "initial_state_sha256": f"{episode_idx:064x}",
                "success": success,
                "policy_calls": 2,
                "action_chunk_sha256": ["a" * 64, "b" * 64],
                "latency_ms_by_call": [100.0, 110.0],
                "fm_calls_total": 10 if policy == "early_exit" else 6,
                "exit_layer_sequence": [11, 13] if policy != "full_depth" else None,
            }
        )
    return {
        "status": "PASS",
        "scope": (
            "m420b_rp_pep_closed_loop_shard"
            if policy == "rp_pep"
            else "m418_persistent_closed_loop_counterfactual_shard"
        ),
        "policy": policy,
        "model_class": model_class,
        "early_exit_enabled": early_enabled,
        "productive_exit_enabled": productive_enabled,
        "vision_aggregation_enabled": False,
        "telemetry_errors": 0,
        "checkpoint_sha256": "c" * 64,
        "task_suite": "libero_spatial",
        "seed": 100,
        "episodes_per_task": len(indices),
        "episode_indices": list(indices),
        "fm_steps": 10,
        "episode_records": rows,
    }


def _items(early_outcomes, full_outcomes, indices=(27, 28)):
    return (
        [("early", _result("early_exit", early_outcomes, indices))],
        [("rp", _result("rp_pep", early_outcomes, indices))],
        [("full", _result("full_depth", full_outcomes, indices))],
    )


def test_three_arm_summary_counts_suspected_failure_and_requests_followup():
    result = build_summary(
        *_items([True, False], [True, True]),
        expected_task_ids=(0,),
        expected_episode_indices=(27, 28),
    )

    assert result["status"] == "PASS"
    assert result["successes"] == {
        "early_exit": 1,
        "rp_pep": 1,
        "full_depth": 2,
    }
    assert result["outcome_counts_early_vs_full"]["both_succeed"] == 1
    assert result["outcome_counts_early_vs_full"]["early_exit_failure_suspected"] == 1
    assert result["failures_fixed_by_full_depth"] == 1
    assert result["cross_seed_followup_required"] is True
    assert all(result["gates"].values())


def test_early_rp_action_mismatch_fails_equivalence_gate():
    early, rp_pep, full = _items([True, True], [True, True])
    rp_pep[0][1]["episode_records"][0]["action_chunk_sha256"][0] = "x" * 64

    result = build_summary(
        early,
        rp_pep,
        full,
        expected_task_ids=(0,),
        expected_episode_indices=(27, 28),
    )

    assert result["status"] == "FAIL"
    assert result["early_rp_equivalence"]["action_chunk_sha256_mismatches"] == 1
    assert not result["gates"]["early_rp_trajectory_equivalence"]


def test_incomplete_expected_grid_fails_without_changing_outcomes():
    result = build_summary(
        *_items([True, True], [True, True]),
        expected_task_ids=(0,),
        expected_episode_indices=(27, 28, 29),
    )

    assert result["status"] == "FAIL"
    assert not result["gates"]["complete_expected_three_arm_grid"]


def test_full_depth_semantics_are_fail_closed():
    early, rp_pep, full = _items([True, True], [True, True])
    full[0][1]["early_exit_enabled"] = True

    with pytest.raises(ValueError, match="semantic mismatch"):
        build_summary(
            early,
            rp_pep,
            full,
            expected_task_ids=(0,),
            expected_episode_indices=(27, 28),
        )


def test_episode_metadata_mismatch_is_rejected():
    early, rp_pep, full = _items([True, True], [True, True])
    full[0][1]["episode_records"][1]["initial_state_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="initial_state_sha256"):
        build_summary(
            early,
            rp_pep,
            full,
            expected_task_ids=(0,),
            expected_episode_indices=(27, 28),
        )
