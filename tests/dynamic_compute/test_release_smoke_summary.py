from scripts.dynamic_compute.summarize_release_smoke import (
    EXPECTED_FM_CALLS,
    EXPECTED_TASKS_BY_GPU,
    parse_expected_gpu_uuids,
    summarize,
)


TEST_GPU_UUIDS = {gpu: f"GPU-00000000-0000-0000-0000-00000000000{gpu}" for gpu in range(4)}


def _shard(gpu: int):
    tasks = EXPECTED_TASKS_BY_GPU[gpu]
    records = []
    for task in tasks:
        records.append(
            {
                "status": "PASS",
                "task_id": task,
                "episode_idx": 30,
                "success": task != 4,
                "policy_calls": 2,
                "initial_state_sha256": f"state-{task}",
                "exit_layer_sequence": [11, 27],
                "exit_layer_counts": {"11": 1, "27": 1},
                "fm_calls_by_policy_call": [
                    EXPECTED_FM_CALLS[11],
                    EXPECTED_FM_CALLS[27],
                ],
                "fm_calls_total": EXPECTED_FM_CALLS[11] + EXPECTED_FM_CALLS[27],
            }
        )
    return {
        "status": "PASS",
        "scope": "m420b_rp_pep_closed_loop_shard",
        "policy": "rp_pep",
        "model_class": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
        "productive_exit_enabled": True,
        "vision_aggregation_enabled": False,
        "checkpoint_sha256": (
            "dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
        ),
        "task_suite": "libero_spatial",
        "task_ids": list(tasks),
        "episode_indices": [30],
        "seed": 20261329,
        "fm_steps": 10,
        "completed_episodes": len(tasks),
        "telemetry_errors": 0,
        "policy_calls": len(tasks) * 2,
        "physical_gpu_uuid_visible": TEST_GPU_UUIDS[gpu][4:],
        "physical_gpu_uuid_nvidia_smi": TEST_GPU_UUIDS[gpu],
        "episode_records": records,
    }


def test_release_smoke_accepts_task_failure_as_valid_rollout():
    result = summarize(
        [_shard(gpu) for gpu in range(4)],
        expected_gpu_uuids=TEST_GPU_UUIDS,
        episode_index=30,
        seed=20261329,
    )

    assert result["status"] == "PASS"
    assert result["completed_episodes"] == 10
    assert result["successes"] == 9
    assert result["global_checks"]["ten_tasks_exactly_once"] is True


def test_release_smoke_rejects_wrong_fm_formula():
    shards = [_shard(gpu) for gpu in range(4)]
    shards[0]["episode_records"][0]["fm_calls_by_policy_call"][0] = 99

    result = summarize(
        shards,
        expected_gpu_uuids=TEST_GPU_UUIDS,
        episode_index=30,
        seed=20261329,
    )

    assert result["status"] == "FAIL"
    assert result["checks_by_gpu"]["0"]["rp_pep_fm_formula"] is False


def test_expected_gpu_uuid_parser_requires_complete_unique_front_four():
    parsed = parse_expected_gpu_uuids(
        [f"{gpu}={uuid}" for gpu, uuid in TEST_GPU_UUIDS.items()]
    )

    assert parsed == TEST_GPU_UUIDS
