from scripts.dynamic_compute.summarize_m417_full_depth_attribution import (
    build_summary,
    outcome_name,
)


def _early(task_id, success):
    return {
        "status": "PASS",
        "checkpoint_sha256": "a" * 64,
        "task_suite": "libero_spatial",
        "task_id": task_id,
        "seed": 7,
        "completed_episodes": 1,
        "successes": int(success),
        "telemetry_calls": 10,
        "latency_ms_mean": 8.0,
        "mean_exit_ratio": 0.5,
    }


def _full(task_id, success):
    return {
        "status": "PASS",
        "scope": "m417_full_depth_no_early_exit_control",
        "model_class": "a1.vla.affordvla.AffordVLA",
        "checkpoint_sha256": "a" * 64,
        "task_suite": "libero_spatial",
        "task_id": task_id,
        "seed": 7,
        "completed_episodes": 1,
        "successes": int(success),
        "policy_calls": 11,
        "latency_ms_mean": 4.0,
        "peak_cuda_memory_bytes": 100,
        "episode_seeds": [task_id * 10_000 + 7],
        "initial_state_sha256": ["b" * 64],
        "fm_steps": 10,
    }


def test_outcome_matrix_is_explicit():
    assert outcome_name(True, True) == "both_succeed"
    assert outcome_name(False, False) == "both_fail"
    assert outcome_name(False, True) == "early_exit_failure_suspected"
    assert outcome_name(True, False) == "full_depth_regression_or_trajectory_difference"


def test_summary_counts_failures_fixed_by_full_depth():
    early = [("early0", _early(0, False)), ("early1", _early(1, True))]
    full = [("full0", _full(0, True)), ("full1", _full(1, True))]

    result = build_summary(early, full)

    assert result["observed_early_exit_failures"] == 1
    assert result["failures_fixed_by_full_depth"] == 1
    assert result["early_exit_attributable_failure_fraction"] == 1.0
    assert result["full_depth_path_validated"] is True
    assert result["rows"][0]["latency_ratio_full_over_early"] == 0.5
