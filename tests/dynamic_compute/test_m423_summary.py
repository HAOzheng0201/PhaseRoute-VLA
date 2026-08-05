from scripts.dynamic_compute.summarize_m423_full_depth_challenge import (
    EXPECTED_ORDERS,
    summarize_profiles,
)


UUIDS = [f"GPU-{index:064x}" for index in range(4)]


def _profile(gpu: int, policy: str, *, action_mismatch: bool = False):
    latency = {"early_exit": 100.0, "rp_pep": 70.0, "full_depth": 50.0}[policy]
    fm_calls = {"early_exit": 8, "rp_pep": 5, "full_depth": 1}[policy]
    layers = {"early_exit": 14, "rp_pep": 14, "full_depth": 28}[policy]
    timed = []
    components = []
    for record in range(12):
        teacher_exit = (11, 13, 27)[record // 4]
        action_sha = f"same-{record}"
        if policy == "full_depth" or (policy == "rp_pep" and action_mismatch and record == 0):
            action_sha = f"{policy}-{record}"
        for repeat in range(2):
            timed.append(
                {
                    "task_id": record % 4,
                    "episode_id": f"ep-{record}",
                    "step_id": record * 8 + 10,
                    "teacher_exit_layer": teacher_exit,
                    "repeat": repeat,
                    "exit_layer": teacher_exit if policy != "full_depth" else 31,
                    "action_sha256": action_sha,
                    "cuda_latency_ms": latency,
                    "wall_latency_ms": latency + 1,
                    "fm_calls": fm_calls,
                    "transformer_layers_executed": layers,
                }
            )
        components.append(
            {
                "task_id": record % 4,
                "episode_id": f"ep-{record}",
                "step_id": record * 8 + 10,
                "teacher_exit_layer": teacher_exit,
                "cuda_latency_ms": latency + 2,
                "transformer_ms": 20.0 if policy != "full_depth" else 40.0,
                "fm_head_ms": 45.0 if policy == "rp_pep" else (25.0 if policy == "full_depth" else 70.0),
                "instrumented_other_ms": 7.0,
            }
        )
    position = EXPECTED_ORDERS[gpu].index(policy) + 1
    return {
        "status": "PASS",
        "scope": "m423_fixed_observation_policy_profile",
        "policy": policy,
        "physical_gpu_index": gpu,
        "order_position": position,
        "physical_gpu_uuid_nvidia_smi": UUIDS[gpu],
        "physical_gpu_uuid_visible": UUIDS[gpu][4:],
        "selection_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "memory_bytes": {
            "timed_peak_allocated": {"early_exit": 25, "rp_pep": 20, "full_depth": 30}[policy],
            "timed_peak_reserved": 40,
            "component_peak_allocated": 30,
            "component_peak_reserved": 40,
        },
        "local_checks": {
            "full_depth_single_solve": True,
            "component_events_consistent": True,
        },
        "timed_samples": timed,
        "component_samples": components,
    }


def _items(**kwargs):
    return [
        (f"gpu{gpu}-{policy}", _profile(gpu, policy, **kwargs))
        for gpu in range(4)
        for policy in EXPECTED_ORDERS[gpu]
    ]


def _m422():
    return {
        "scope": "m422_three_arm_full_depth_attribution_summary",
        "records": 50,
        "successes": {"early_exit": 49, "rp_pep": 49, "full_depth": 50},
    }


def test_summary_passes_engineering_but_rejects_false_pareto_claim():
    result = summarize_profiles(_items(), m422_result=_m422())

    assert result["status"] == "PASS"
    assert result["audit_counters"]["early_rp_pep_action_mismatches"] == 0
    assert result["policy_summary"]["full_depth"]["fm_calls"]["per_call"]["mean"] == 1
    assert result["pareto_challenge"]["status"] == "NOT_MET"
    assert result["pareto_challenge"]["overall_pareto_improvement"] is False
    assert result["diagnosis"]["rp_pep_extra_fm_solves_per_call_vs_full_depth"] == 4
    assert result["by_actual_early_exit_layer"]["11"]["paired_samples"] == 32
    assert (
        result["by_actual_early_exit_layer"]["11"]["paired_cuda_latency_ratio"]
        ["rp_pep_over_full"]["median"]
        == 1.4
    )


def test_summary_surfaces_early_rp_pep_action_mismatch():
    result = summarize_profiles(_items(action_mismatch=True), m422_result=_m422())
    assert result["status"] == "FAIL"
    assert result["audit_counters"]["early_rp_pep_action_mismatches"] == 8
