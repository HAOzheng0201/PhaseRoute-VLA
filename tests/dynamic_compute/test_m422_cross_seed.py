from scripts.dynamic_compute.summarize_m422_cross_seed import build_summary


SEEDS = (100, 200, 300)


def _result(policy, seed, success):
    full = policy == "full_depth"
    return {
        "status": "PASS",
        "scope": "m418_persistent_closed_loop_counterfactual_shard",
        "policy": policy,
        "model_class": (
            "a1.vla.affordvla.AffordVLA"
            if full
            else "a1.vla.affordvla_early_exit.AffordVLAEarlyExit"
        ),
        "early_exit_enabled": not full,
        "productive_exit_enabled": False,
        "vision_aggregation_enabled": False,
        "telemetry_errors": 0,
        "checkpoint_sha256": "c" * 64,
        "task_suite": "libero_spatial",
        "task_ids": [4],
        "seed": seed,
        "episodes_per_task": 1,
        "episode_indices": [29],
        "fm_steps": 10,
        "episode_records": [
            {
                "status": "PASS",
                "task_id": 4,
                "episode_idx": 29,
                "episode_seed": seed + 40_029,
                "initial_state_sha256": "a" * 64,
                "success": success,
                "policy_calls": 10,
                "exit_layer_sequence": [11] * 10 if not full else None,
            }
        ],
    }


def _items(early_successes, full_successes):
    early = [
        (f"early-{seed}", _result("early_exit", seed, success))
        for seed, success in zip(SEEDS, early_successes, strict=True)
    ]
    full = [
        (f"full-{seed}", _result("full_depth", seed, success))
        for seed, success in zip(SEEDS, full_successes, strict=True)
    ]
    return early, full


def _summarize(early_successes, full_successes):
    return build_summary(
        *_items(early_successes, full_successes),
        expected_base_seeds=SEEDS,
        task_id=4,
        episode_idx=29,
        expected_outcome="early_exit_failure_suspected",
        min_expected_matches=2,
    )


def test_two_of_three_expected_direction_replicates():
    result = _summarize([False, False, True], [True, True, True])

    assert result["expected_matches"] == 2
    assert result["opposite_discordance_matches"] == 0
    assert result["replicated"] is True
    assert result["decision"] == "discordant_state_replicated"


def test_zero_of_three_expected_direction_does_not_replicate():
    result = _summarize([True, True, True], [True, True, True])

    assert result["expected_matches"] == 0
    assert result["replicated"] is False


def test_opposite_discordance_blocks_replication():
    result = _summarize([False, False, True], [True, True, False])

    assert result["expected_matches"] == 2
    assert result["opposite_discordance_matches"] == 1
    assert result["replicated"] is False
