import pytest

from scripts.summarize_route_first_stage9_pair import build_summary


PROTOCOL_SHA = "a" * 64
GPU_UUID = "GPU-535e41e1-a1ac-af65-a015-fc281644709e"


def _runtime():
    return {
        "policy_calls": 2,
        "prepared_calls": 2,
        "committed_calls": 2,
        "error_count": 0,
        "selected_layers": {"11": 0, "13": 1, "27": 1},
    }


def _measurement(mean, p50, p95):
    return {
        "latency_ms": {
            "policy_wall": {
                "count": 2,
                "sum": mean * 2,
                "mean": mean,
                "p50": p50,
                "p95": p95,
                "max": max(p50, p95),
            }
        }
    }


def _episode(success=True):
    return {
        "task_id": 0,
        "episode_index": 12,
        "seed": 20260838,
        "success": success,
        "wall_seconds": 10.0,
    }


def _candidate():
    attestation = {
        "schema_version": "phase-route-vla.route-first-stage9-candidate-arm.v1",
        "status": "PASS",
        "method": "candidate_first_v3",
        "arm_position": 1,
        "protocol_sha256": PROTOCOL_SHA,
        "gpu_uuid": GPU_UUID,
    }
    evaluation = {
        "schema_version": "phase-route-vla.libero-evaluation-summary.v1",
        "method": "phase_route_v3",
        "suite": "libero_10",
        "task_ids": [0],
        "episode_indices": [12],
        "seed_base": 20260826,
        "total_episodes": 1,
        "total_successes": 1,
        "episodes": [_episode()],
        "runtime": _runtime(),
        "stage1_measurement": _measurement(100.0, 90.0, 120.0),
    }
    return attestation, evaluation


def _route():
    attestation = {
        "schema_version": "phase-route-vla.route-first-active-attestation.v1",
        "status": "PASS",
        "protocol_sha256": PROTOCOL_SHA,
        "selection_error": None,
    }
    runtime = _runtime()
    runtime["route_first_integrity"] = {
        "valid_calls_with_exactly_one_fm": 2,
        "fm_invocations": 2,
    }
    evaluation = {
        "schema_version": "phase-route-vla.route-first-active-evaluation.v1",
        "method": "route_first_stage8",
        "arm_position": 2,
        "protocol_sha256": PROTOCOL_SHA,
        "suite": "libero_10",
        "task_ids": [0],
        "episode_indices": [12],
        "seed_base": 20260826,
        "total_episodes": 1,
        "total_successes": 1,
        "gpu": {"expected_uuid": GPU_UUID},
        "episodes": [_episode()],
        "runtime": runtime,
        "stage1_measurement": _measurement(60.0, 50.0, 80.0),
    }
    return attestation, evaluation


def test_builds_identity_checked_stage9_pair():
    candidate_attestation, candidate_evaluation = _candidate()
    route_attestation, route_evaluation = _route()

    result = build_summary(
        candidate_attestation,
        candidate_evaluation,
        route_attestation,
        route_evaluation,
    )

    assert result["status"] == "PASS"
    comparison = result["descriptive_comparison"]
    assert comparison["policy_wall_mean_reduction_fraction"] == pytest.approx(0.4)
    assert comparison["policy_wall_mean_speedup"] == pytest.approx(5 / 3)
    assert result["next_gate"]["state13_pilot_protocol_gate_unlocked"] is True
    assert result["claim_boundary"]["formal_wall_clock_speedup_claim"] is False


@pytest.mark.parametrize("drift", ["protocol", "gpu", "seed"])
def test_rejects_paired_identity_drift(drift):
    candidate_attestation, candidate_evaluation = _candidate()
    route_attestation, route_evaluation = _route()
    if drift == "protocol":
        route_attestation["protocol_sha256"] = "b" * 64
    elif drift == "gpu":
        route_evaluation["gpu"]["expected_uuid"] = "GPU-other"
    else:
        route_evaluation["episodes"][0]["seed"] += 1

    with pytest.raises(ValueError):
        build_summary(
            candidate_attestation,
            candidate_evaluation,
            route_attestation,
            route_evaluation,
        )


def test_failed_exact_fm_gate_does_not_unlock_state13():
    candidate_attestation, candidate_evaluation = _candidate()
    route_attestation, route_evaluation = _route()
    route_evaluation["runtime"]["route_first_integrity"][
        "valid_calls_with_exactly_one_fm"
    ] = 1

    result = build_summary(
        candidate_attestation,
        candidate_evaluation,
        route_attestation,
        route_evaluation,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["route_every_call_exactly_one_fm"] is False
    assert result["next_gate"]["state13_pilot_protocol_gate_unlocked"] is False


def test_failed_route_episode_does_not_unlock_state13():
    candidate_attestation, candidate_evaluation = _candidate()
    route_attestation, route_evaluation = _route()
    route_evaluation["total_successes"] = 0
    route_evaluation["episodes"][0]["success"] = False

    result = build_summary(
        candidate_attestation,
        candidate_evaluation,
        route_attestation,
        route_evaluation,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["route_episode_success"] is False
    assert result["next_gate"]["state13_pilot_protocol_gate_unlocked"] is False
