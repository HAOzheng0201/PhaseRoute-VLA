from scripts.dynamic_compute.summarize_m420b_paired_rollouts import build_summary


def _result(policy, task_ids, episode_indices=(27, 28)):
    records = []
    for task_id in task_ids:
        for episode_idx in episode_indices:
            records.append(
                {
                    "status": "PASS",
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "episode_seed": 20264804 + task_id * 10_000 + episode_idx,
                    "initial_state_sha256": f"state-{task_id}-{episode_idx}",
                    "success": task_id % 2 == 0,
                    "policy_calls": 2,
                    "action_chunk_sha256": ["a" * 64, "b" * 64],
                    "exit_layer_sequence": [11, 13],
                    "fm_calls_total": 15 if policy == "early_exit" else 9,
                    "latency_ms_by_call": (
                        [100.0, 100.0] if policy == "early_exit" else [70.0, 70.0]
                    ),
                    "latency_ms_total": 200.0 if policy == "early_exit" else 140.0,
                    "wall_seconds": 1.0,
                }
            )
    return {
        "status": "PASS",
        "policy": policy,
        "telemetry_errors": 0,
        "checkpoint_sha256": "c" * 64,
        "task_suite": "libero_spatial",
        "seed": 20264804,
        "episodes_per_task": len(episode_indices),
        "episode_indices": list(episode_indices),
        "fm_steps": 10,
        "episode_records": records,
    }


def test_complete_exact_grid_passes_all_closed_loop_gates():
    baseline = [("baseline", _result("early_exit", range(10)))]
    sparse = [("sparse", _result("rp_pep", range(10)))]

    result = build_summary(baseline, sparse)

    assert result["status"] == "PASS"
    assert result["paired_episodes"] == 20
    assert result["total_rollouts"] == 40
    assert all(result["gates"].values())


def test_one_action_hash_mismatch_fails_trajectory_gate():
    baseline_result = _result("early_exit", range(10))
    sparse_result = _result("rp_pep", range(10))
    sparse_result["episode_records"][0]["action_chunk_sha256"][1] = "x" * 64

    result = build_summary([("baseline", baseline_result)], [("sparse", sparse_result)])

    assert result["status"] == "FAIL"
    assert result["equivalence"]["action_chunk_sha256_mismatches"] == 1
    assert not result["gates"]["trajectory_equivalence"]


def test_expanded_episode_grid_is_explicitly_supported():
    indices = (29, 30, 31)
    baseline = [("baseline", _result("early_exit", range(10), indices))]
    sparse = [("sparse", _result("rp_pep", range(10), indices))]

    result = build_summary(
        baseline,
        sparse,
        expected_episode_indices=indices,
    )

    assert result["status"] == "PASS"
    assert result["paired_episodes"] == 30
    assert result["total_rollouts"] == 60
    assert result["expected_episode_indices"] == [29, 30, 31]
