from scripts.dynamic_compute.summarize_m418_paired_counterfactual import (
    build_summary,
    exact_mcnemar_p_value,
    wilson_interval,
)


def _result(policy, outcomes):
    model_class = (
        "a1.vla.affordvla_early_exit.AffordVLAEarlyExit"
        if policy == "early_exit"
        else "a1.vla.affordvla.AffordVLA"
    )
    return {
        "status": "PASS",
        "scope": "m418_persistent_closed_loop_counterfactual_shard",
        "policy": policy,
        "model_class": model_class,
        "checkpoint_sha256": "a" * 64,
        "task_suite": "libero_spatial",
        "seed": 9,
        "episodes_per_task": len(outcomes),
        "episode_start_index": 0,
        "episode_indices": list(range(len(outcomes))),
        "fm_steps": 10,
        "episode_records": [
            {
                "status": "PASS",
                "task_id": 0,
                "episode_idx": index,
                "episode_seed": 9 + index,
                "initial_state_sha256": f"{index:064x}",
                "success": success,
                "policy_calls": 10 + index,
                "latency_ms_total": 100.0,
                "exit_mean_ratio": 0.5 if policy == "early_exit" else None,
            }
            for index, success in enumerate(outcomes)
        ],
    }


def test_wilson_interval_handles_small_counts_without_zero_width():
    interval = wilson_interval(1, 1)

    assert interval["estimate"] == 1.0
    assert 0.0 < interval["lower"] < interval["upper"] == 1.0
    assert wilson_interval(0, 0) is None


def test_exact_mcnemar_uses_only_discordant_pairs():
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(1, 0) == 1.0
    assert exact_mcnemar_p_value(6, 0) == 0.03125


def test_paired_summary_counts_causal_failure_candidates_and_readiness():
    early = [("early", _result("early_exit", [True, False, False, True]))]
    full = [("full", _result("full_depth", [True, True, False, True]))]

    result = build_summary(
        early,
        full,
        min_risk_positive_episodes=1,
        min_risk_negative_episodes=3,
    )

    assert result["outcome_counts"] == {
        "both_succeed": 2,
        "both_fail": 1,
        "early_exit_failure_suspected": 1,
        "full_depth_regression_or_trajectory_difference": 0,
    }
    assert result["observed_early_exit_failures"] == 2
    assert result["failures_fixed_by_full_depth"] == 1
    assert result["attributable_fraction_among_early_failures"]["estimate"] == 0.5
    assert result["risk_training_ready"] is False
