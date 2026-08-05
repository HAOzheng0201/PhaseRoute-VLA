from scripts.dynamic_compute.replay_m420b_rp_pep import evaluate_gate, normalize_gpu_uuid


def _row(*, exact=True, sparse_ms=70.0, sparse_calls=4):
    return {
        "exit_match": True,
        "action_exact": exact,
        "action_max_abs_error": 0.0 if exact else 1e-3,
        "gripper_direction_mismatches": 0,
        "finite": True,
        "fm_formula_match": True,
        "retained_trace_match": True,
        "threshold_event_match": True,
        "telemetry_errors": 0,
        "baseline": {
            "fm_calls": 7,
            "cuda_latency_ms": 100.0,
            "wall_latency_ms": 105.0,
        },
        "rp_pep": {
            "fm_calls": sparse_calls,
            "cuda_latency_ms": sparse_ms,
            "wall_latency_ms": sparse_ms + 5.0,
        },
    }


def test_gate_passes_strict_equivalence_and_preregistered_savings():
    result = evaluate_gate([_row(), _row(sparse_ms=75.0)])

    assert result["status"] == "PASS"
    assert result["gates"]["strict_equivalence"]
    assert result["fm_solver_calls"]["reduction_fraction"] > 0.35
    assert result["cuda_policy_latency"]["reduction_fraction"] > 0.15


def test_gate_fails_on_one_nonexact_action_even_when_efficiency_passes():
    result = evaluate_gate([_row(), _row(exact=False)])

    assert result["status"] == "FAIL"
    assert result["counters"]["action_nonexact"] == 1
    assert not result["gates"]["strict_equivalence"]


def test_gpu_uuid_normalization_accepts_nvidia_smi_prefix_only():
    raw = "00000000-0000-0000-0000-000000000000"
    assert normalize_gpu_uuid(f"GPU-{raw}") == normalize_gpu_uuid(raw)
