from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from a1.vla.dynamic_compute.route_first_stage10 import (
    METHODS,
    PROTOCOL_SHA256,
    SCHEDULE_SHA256,
    STATE_PAYLOAD_SCHEMA,
    load_schedule,
    sha256_file,
)
from a1.vla.dynamic_compute.route_first_stage10_active import (
    ACTIVE_TRIPLET_SCHEMA,
    PROTECTED_CODE_SHA256,
    Stage10ActiveError,
    aggregate_triplets,
    select_arm,
    summarize_measurement_records,
    summarize_policy_records,
    validate_state_payload_mapping,
    validate_triplet_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _payload():
    schedule = load_schedule(REPO_ROOT)
    states = [
        np.array([float(index), float(index + 1)], dtype=np.float64)
        for index in range(len(schedule))
    ]
    from a1.vla.dynamic_compute.route_first_stage10 import canonical_state_bytes

    digests = [canonical_state_bytes(state)[2] for state in states]
    records = [
        {
            "task_id": spec.task_id,
            "replicate_id": spec.replicate_id,
            "cluster_key": spec.cluster_key,
            "arm_order": list(spec.arm_order),
            "state_seed": spec.state_seed,
            "policy_seed": spec.policy_seed,
            "state_dimension": 2,
            "state_sha256": digest,
        }
        for spec, digest in zip(schedule, digests, strict=True)
    ]
    payload = {
        "schema_version": STATE_PAYLOAD_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "source_git_commit": "generation-commit",
        "task_id": [item.task_id for item in schedule],
        "replicate_id": [item.replicate_id for item in schedule],
        "state_seed": [item.state_seed for item in schedule],
        "policy_seed": [item.policy_seed for item in schedule],
        "cluster_keys": [item.cluster_key for item in schedule],
        "arm_orders": [list(item.arm_order) for item in schedule],
        "state_sha256": digests,
        "states": states,
        "determinism_passes": 2,
        "initial_task_success_all_false": True,
        "official_episode_identity_used": False,
        "policy_rollout_performed": False,
    }
    return schedule, payload, {"source_git_commit": "generation-commit", "records": records}


def _telemetry(spec, layer: int, route: bool = False):
    events = []
    if route:
        events = [
            {
                "event": "route_first_selected_action",
                "layer_idx": layer,
                "fm_calls": 1,
                "fail_reason": None,
            },
            {
                "event": "phase_route_decision",
                "selected_layer": layer,
                "fm_calls": 1,
            },
            {
                "event": "exit_candidate",
                "evaluated": True,
                "layer_idx": layer,
                "should_exit": True,
                "fm_calls": 1,
            },
        ]
    return {
        "schema_version": "phase-route-vla.telemetry.v1",
        "episode_id": spec.cluster_key,
        "task_id": spec.task_id,
        "step_id": 10,
        "exit_layer": layer,
        "fm_calls": 3 if route else 2,
        "fm_steps_total": 10 if route else 20,
        "extra": {"exit_events": events},
    }


def _measurement(spec, layer: int | None, latency: float):
    modes = {
        "original_a1": "original_a1",
        "candidate_first_v3": "phase_route_v3",
        "route_first_stage8": "route_first_stage8",
    }
    return {
        "schema_version": "phase-route-vla.stage1.measurement.v1",
        "measurement_is_control_input": False,
        "d9_protected_source_modified": False,
        "mode": modes[spec.method],
        "context": {"episode_id": spec.cluster_key, "task_id": spec.task_id},
        "selected_layer": layer,
        "policy_wall_latency_ms": latency,
        "action_finite": True,
        "action_shape": [8, 7],
        "error": None,
    }


def _triplet(spec, *, route_latency: float = 70.0):
    latency = {
        "original_a1": 90.0,
        "candidate_first_v3": 100.0,
        "route_first_stage8": route_latency,
    }
    layer = {
        "original_a1": "L11",
        "candidate_first_v3": "L13",
        "route_first_stage8": "L13",
    }
    arms = {}
    for position, method in enumerate(spec.arm_order, start=1):
        arms[method] = {
            "method": method,
            "arm_position": position,
            "success": True,
            "environment_steps": 100,
            "policy_calls": 10,
            "selected_layer_counts": {layer[method]: 10},
            "route_exactly_one_fm_calls": (
                10 if method == "route_first_stage8" else 0
            ),
            "policy_p50_ms": latency[method],
            "policy_seed": spec.policy_seed,
            "state_sha256": "a" * 64,
            "source_git_commit": "runner-commit",
            "gpu_uuid": "GPU-test",
            "evidence_valid": True,
        }
    return {
        "schema_version": ACTIVE_TRIPLET_SCHEMA,
        "status": "COMPLETE_ROUTE_FIRST_STAGE10_TRIPLET",
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "cluster_key": spec.cluster_key,
        "state_seed": spec.state_seed,
        "policy_seed": spec.policy_seed,
        "arm_order": list(spec.arm_order),
        "arms": arms,
    }


def test_active_arm_selection_is_exact_and_counterbalanced() -> None:
    schedule = load_schedule(REPO_ROOT)
    for spec in schedule:
        for position, method in enumerate(spec.arm_order, start=1):
            selected = select_arm(
                REPO_ROOT,
                task_id=spec.task_id,
                replicate_id=spec.replicate_id,
                method=method,
                arm_position=position,
            )
            assert selected.policy_seed == spec.policy_seed
    with pytest.raises(Stage10ActiveError, match="frozen arm order"):
        select_arm(
            REPO_ROOT,
            task_id=0,
            replicate_id=0,
            method="route_first_stage8",
            arm_position=1,
        )


def test_state_payload_mapping_validates_every_record_and_raw_sha() -> None:
    schedule, payload, attestation = _payload()
    states = validate_state_payload_mapping(schedule, payload, attestation)
    assert len(states) == 60
    assert all(state.dtype == np.dtype("<f8") for state in states)

    changed = deepcopy(payload)
    changed["states"][0][0] += 1.0
    with pytest.raises(Stage10ActiveError, match="mismatch"):
        validate_state_payload_mapping(schedule, changed, attestation)


def test_state_payload_rejects_policy_seed_or_order_drift() -> None:
    schedule, payload, attestation = _payload()
    payload["policy_seed"][0] += 1
    with pytest.raises(Stage10ActiveError, match="policy_seed"):
        validate_state_payload_mapping(schedule, payload, attestation)

    _, payload, attestation = _payload()
    payload["arm_orders"][0].reverse()
    with pytest.raises(Stage10ActiveError, match="arm orders"):
        validate_state_payload_mapping(schedule, payload, attestation)


def test_route_first_exactly_one_fm_uses_events_not_top_level_counter() -> None:
    spec = select_arm(
        REPO_ROOT,
        task_id=0,
        replicate_id=0,
        method="route_first_stage8",
        arm_position=3,
    )
    summary = summarize_policy_records([_telemetry(spec, 13, route=True)], spec=spec)
    assert summary["telemetry_fm_calls"] == 3
    assert summary["route_exactly_one_fm_calls"] == 1
    assert summary["route_exactly_one_fm_fraction"] == 1.0

    broken = _telemetry(spec, 13, route=True)
    broken["extra"]["exit_events"][0]["fm_calls"] = 2
    with pytest.raises(Stage10ActiveError, match="exactly one FM"):
        summarize_policy_records([broken], spec=spec)


def test_measurement_summary_uses_policy_wall_latency_and_validates_mode() -> None:
    spec = select_arm(
        REPO_ROOT,
        task_id=0,
        replicate_id=0,
        method="candidate_first_v3",
        arm_position=2,
    )
    records = [
        _measurement(spec, 13, value) for value in (10.0, 30.0, 20.0)
    ]
    for index, record in enumerate(records):
        record["context"]["call_ordinal"] = index
    summary = summarize_measurement_records(
        records, spec=spec, expected_policy_calls=3
    )
    assert summary["mean_ms"] == 20.0
    assert summary["p50_ms"] == 20.0
    assert summary["p95_ms"] == 30.0

    records[0]["mode"] = "route_first_stage8"
    with pytest.raises(Stage10ActiveError, match="measurement record"):
        summarize_measurement_records(records, spec=spec, expected_policy_calls=3)


def test_complete_60_triplet_aggregate_passes_preregistered_gate() -> None:
    schedule = load_schedule(REPO_ROOT)
    triplets = [_triplet(spec) for spec in schedule]
    result = aggregate_triplets(schedule, triplets)
    assert result["status"] == "PASS_ROUTE_FIRST_STAGE10_FRESH_ACTIVE_CONFIRMATION"
    assert result["active_rollouts"] == 180
    assert result["success_counts"] == {method: 60 for method in METHODS}
    assert result["within_triplet_latency_ratios"][
        "route_to_candidate_episode_p50_median"
    ] == pytest.approx(0.7)
    assert all(result["gates"].values())


def test_aggregate_cannot_pass_incomplete_or_slow_route_evidence() -> None:
    schedule = load_schedule(REPO_ROOT)
    with pytest.raises(Stage10ActiveError, match="all 60"):
        aggregate_triplets(schedule, [_triplet(item) for item in schedule[:-1]])

    result = aggregate_triplets(schedule, [_triplet(item, route_latency=95.0) for item in schedule])
    assert result["status"].startswith("INCOMPLETE")
    assert not result["gates"][
        "route_candidate_episode_p50_ratio_median_at_most_0_80"
    ]


def test_triplet_rejects_cross_gpu_or_commit_pairing() -> None:
    spec = load_schedule(REPO_ROOT)[0]
    record = _triplet(spec)
    validate_triplet_record(record, spec=spec)
    record["arms"]["original_a1"]["gpu_uuid"] = "GPU-other"
    with pytest.raises(Stage10ActiveError, match="pairing"):
        validate_triplet_record(record, spec=spec)


def test_protected_historical_code_is_still_byte_exact() -> None:
    assert {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in PROTECTED_CODE_SHA256
    } == PROTECTED_CODE_SHA256


def test_readiness_artifact_sha_constants_are_well_formed() -> None:
    from scripts.freeze_route_first_stage10_runner_readiness import (
        EXPECTED_ARTIFACTS,
    )

    assert all(
        len(item["sha256"]) == 64
        and set(item["sha256"]).issubset(set("0123456789abcdef"))
        for item in EXPECTED_ARTIFACTS.values()
    )


def test_runner_never_reads_official_init_states_and_loads_payload_safely() -> None:
    runner = (REPO_ROOT / "scripts/run_route_first_stage10_arm.py").read_text(
        encoding="utf-8"
    )
    contract = (
        REPO_ROOT
        / "a1/vla/dynamic_compute/route_first_stage10_active.py"
    ).read_text(encoding="utf-8")
    supervisor = (
        REPO_ROOT / "scripts/run_route_first_stage10_triplet.py"
    ).read_text(encoding="utf-8")
    assert "get_task_init_states" not in runner
    assert 'map_location="cpu", weights_only=True' in contract
    assert "for position, method in enumerate(spec.arm_order" in supervisor
    assert "minimum_free_memory_40000_mib" in (
        REPO_ROOT / "scripts/validate_route_first_stage10_preflight.py"
    ).read_text(encoding="utf-8")
