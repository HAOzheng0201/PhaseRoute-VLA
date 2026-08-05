from scripts.dynamic_compute.summarize_m424_oracle_challenge import (
    EXPECTED_ORDERS,
    summarize_oracle_challenge,
)


UUIDS = [f"GPU-{index:064x}" for index in range(4)]


def _profile(gpu: int, policy: str, *, mismatch: bool = False):
    oracle = policy == "oracle_rts"
    timed = []
    components = []
    for record in range(12):
        route = (11, 13, 27)[record // 4]
        for repeat in range(2):
            timed.append(
                {
                    "cache_dir": f"/cache/task{record % 4}",
                    "array_path": f"arrays/call_{record:06d}.npz",
                    "episode_id": f"ep-{record}",
                    "task_id": record % 4,
                    "step_id": record * 8 + 10,
                    "teacher_exit_layer": route,
                    "repeat": repeat,
                    "route_layer": route if oracle else None,
                    "transformer_layers_executed": route + 1 if oracle else 28,
                    "cuda_latency_ms": 35.0 if oracle else 50.0,
                    "wall_latency_ms": 36.0 if oracle else 51.0,
                    "fm_calls": 1,
                    "fm_steps": 10,
                    "rng_burns": {11: 6, 13: 7, 27: 14}[route] if oracle else 0,
                    "original_fm_calls": {11: 7, 13: 8, 27: 15}[route] if oracle else 1,
                    "action_exact": not mismatch,
                }
            )
        components.append(
            {
                "cache_dir": f"/cache/task{record % 4}",
                "array_path": f"arrays/call_{record:06d}.npz",
                "episode_id": f"ep-{record}",
                "task_id": record % 4,
                "step_id": record * 8 + 10,
                "teacher_exit_layer": route,
                "route_layer": route if oracle else None,
                "transformer_layers_executed": route + 1 if oracle else 28,
                "cuda_latency_ms": 36.0 if oracle else 51.0,
                "transformer_ms": 5.0 if oracle else 10.0,
                "fm_head_ms": 29.0 if oracle else 39.0,
                "instrumented_other_ms": 2.0,
            }
        )
    position = EXPECTED_ORDERS[gpu].index(policy) + 1
    return {
        "status": "PASS",
        "scope": (
            "m424_oracle_route_then_solve_profile"
            if oracle
            else "m423_fixed_observation_policy_profile"
        ),
        "policy": policy,
        "physical_gpu_index": gpu,
        "order_position": position,
        "physical_gpu_uuid_nvidia_smi": UUIDS[gpu],
        "physical_gpu_uuid_visible": UUIDS[gpu][4:],
        "selection_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "route_source_sha256": "c" * 64 if oracle else None,
        "memory_bytes": {
            "timed_peak_allocated": 100,
            "timed_peak_reserved": 120,
            "component_peak_allocated": 100,
            "component_peak_reserved": 120,
        },
        "local_checks": {"component_events_consistent": True},
        "timed_samples": timed,
        "component_samples": components,
    }


def _items(**kwargs):
    return [
        (f"gpu{gpu}-{policy}", _profile(gpu, policy, **(kwargs if policy == "oracle_rts" else {})))
        for gpu in range(4)
        for policy in EXPECTED_ORDERS[gpu]
    ]


M422 = {
    "scope": "m422_three_arm_full_depth_attribution_summary",
    "paired_states": 50,
    "successes": {"early_exit": 49, "full_depth": 50},
}


def test_oracle_summary_marks_fast_exact_ceiling_viable():
    result = summarize_oracle_challenge(_items(), m422_result=M422)
    assert result["status"] == "PASS"
    assert result["audit_counters"]["oracle_action_mismatches"] == 0
    assert result["paired_oracle_over_full"]["median"] == 0.7
    assert result["oracle_ceiling"]["status"] == "VIABLE"
    assert result["task_effect_boundary"]["overall_task_pareto_claimed"] is False


def test_oracle_summary_fails_engineering_on_action_mismatch():
    result = summarize_oracle_challenge(_items(mismatch=True), m422_result=M422)
    assert result["status"] == "FAIL"
    assert result["oracle_ceiling"]["status"] == "NOT_VIABLE"
    assert result["audit_counters"]["oracle_action_mismatches"] == 96
