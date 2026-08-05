from scripts.dynamic_compute.summarize_m419_discordant_replays import build_summary


EPISODES = [2, 14, 22]


def _result(policy, seed, successes):
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
        "task_ids": [5],
        "seed": seed,
        "episodes_per_task": 3,
        "episode_indices": EPISODES,
        "fm_steps": 10,
        "episode_records": [
            {
                "status": "PASS",
                "task_id": 5,
                "episode_idx": episode_idx,
                "episode_seed": seed + 50_000 + episode_idx,
                "initial_state_sha256": f"{episode_idx:064x}",
                "success": success,
                "policy_calls": 10,
                "exit_mean_ratio": 0.5 if policy == "early_exit" else None,
            }
            for episode_idx, success in zip(EPISODES, successes)
        ],
    }


def test_replay_summary_applies_preregistered_two_of_three_rule():
    seeds = [1000, 2000, 3000]
    early_successes = [
        [False, True, True],
        [False, True, True],
        [True, True, True],
    ]
    full_successes = [
        [True, False, True],
        [True, True, False],
        [True, False, False],
    ]
    early = [
        (f"early-{seed}", _result("early_exit", seed, outcomes))
        for seed, outcomes in zip(seeds, early_successes)
    ]
    full = [
        (f"full-{seed}", _result("full_depth", seed, outcomes))
        for seed, outcomes in zip(seeds, full_successes)
    ]

    result = build_summary(
        early,
        full,
        min_repeat_seeds=3,
        min_expected_matches=2,
    )

    assert result["outcome_counts"] == {
        "both_succeed": 3,
        "both_fail": 0,
        "early_exit_failure_suspected": 2,
        "full_depth_regression_or_trajectory_difference": 4,
    }
    assert result["causal_positive_state_replicated"] is True
    assert result["all_original_discordances_replicated"] is True
    assert all(
        summary["replicated"] for summary in result["state_summaries"].values()
    )
    assert result["risk_head_training_recommended"] is False


def test_opposite_discordance_blocks_replication():
    seeds = [1000, 2000, 3000]
    early = [
        (f"early-{seed}", _result("early_exit", seed, [True, True, True]))
        for seed in seeds
    ]
    full = [
        (
            f"full-{seed}",
            _result(
                "full_depth",
                seed,
                [False, True, True] if seed == 1000 else [True, True, True],
            ),
        )
        for seed in seeds
    ]

    result = build_summary(
        early,
        full,
        min_repeat_seeds=3,
        min_expected_matches=2,
    )

    episode2 = result["state_summaries"]["2"]
    assert episode2["expected_matches"] == 0
    assert episode2["opposite_discordance_matches"] == 1
    assert episode2["replicated"] is False
    assert result["causal_positive_state_replicated"] is False
    assert result["decision"] == "causal_positive_not_replicated_stop_risk_head_route"
