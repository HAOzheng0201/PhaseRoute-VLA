from __future__ import annotations

import json
from pathlib import Path

import pytest

from a1.vla.dynamic_compute.route_first_active_protocol import (
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
    RouteFirstActiveProtocolError,
    load_route_first_active_protocol,
    validate_route_first_active_selection,
)
from scripts.validate_route_first_active_run import validate_run


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "configs/route_first_active_pilot_protocol.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


def test_protocol_authorizes_only_frozen_smoke_and_alternating_pilot_arms() -> None:
    protocol = load_route_first_active_protocol(PROTOCOL_PATH, REPO_ROOT)
    smoke = validate_route_first_active_selection(
        protocol,
        experiment_stage="engineering_smoke",
        task_spec="0",
        episode_spec="12",
        arm_position=2,
        seed=20260826,
    )
    assert smoke.task_ids == (0,)
    assert smoke.episode_indices == (12,)

    even = validate_route_first_active_selection(
        protocol,
        experiment_stage="paired_pilot",
        task_spec="2",
        episode_spec="13",
        arm_position=2,
        seed=20260826,
    )
    odd = validate_route_first_active_selection(
        protocol,
        experiment_stage="paired_pilot",
        task_spec="3",
        episode_spec="13",
        arm_position=1,
        seed=20260826,
    )
    assert (even.arm_position, odd.arm_position) == (2, 1)

    with pytest.raises(RouteFirstActiveProtocolError, match="smoke arm"):
        validate_route_first_active_selection(
            protocol,
            experiment_stage="engineering_smoke",
            task_spec="0",
            episode_spec="13",
            arm_position=2,
            seed=20260826,
        )
    with pytest.raises(RouteFirstActiveProtocolError, match="arm order"):
        validate_route_first_active_selection(
            protocol,
            experiment_stage="paired_pilot",
            task_spec="3",
            episode_spec="13",
            arm_position=2,
            seed=20260826,
        )


def _valid_runtime_record() -> dict:
    return {
        "schema_version": "phase-route-vla.route-first-active-runtime.v1",
        "runtime_mode": "route_first_l13_l27",
        "prepared": True,
        "committed": True,
        "errors": [],
        "selected_layer": 13,
        "route_first_target_layer": 13,
        "route_first_scores": [0.1, 0.95],
        "events": [
            {
                "event": "exit_candidate",
                "layer_idx": 3,
                "evaluated": False,
                "should_exit": False,
                "fm_calls": 0,
            },
            {
                "event": "route_first_selected_action",
                "layer_idx": 13,
                "fm_calls": 1,
                "fail_reason": None,
            },
            {
                "event": "phase_route_decision",
                "selected_layer": 13,
                "fm_calls": 1,
            },
            {
                "event": "exit_candidate",
                "layer_idx": 13,
                "evaluated": True,
                "should_exit": True,
                "fm_calls": 1,
            },
        ],
    }


def _valid_measurement() -> dict:
    return {
        "schema_version": "phase-route-vla.stage1.measurement.v1",
        "mode": "route_first_stage8",
        "measurement_is_control_input": False,
        "d9_protected_source_modified": False,
        "selected_layer": 13,
        "action_finite": True,
        "action_shape": [8, 7],
        "policy_wall_latency_ms": 50.0,
        "error": None,
    }


def _active_run_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "preflight.json",
        {
            "status": "PASS",
            "scope": "route_first_stage9_active_preflight",
            "simulator_episode_opened": False,
            "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
            "expected_gpu_uuid": "GPU-abcd",
        },
    )
    runtime_summary = {
        "records": 1,
        "policy_calls": 1,
        "prepared_calls": 1,
        "committed_calls": 1,
        "error_count": 0,
        "records_with_errors": 0,
        "selected_layers": {"11": 0, "13": 1, "27": 0},
        "route_first_integrity": {
            "valid_calls_with_exactly_one_fm": 1,
            "fm_invocations": 1,
            "valid_calls_with_fm_calls_equal_one_fraction": 1.0,
        },
    }
    _write_json(
        run_dir / "evaluation_summary.json",
        {
            "schema_version": "phase-route-vla.route-first-active-evaluation.v1",
            "method": "route_first_stage8",
            "experiment_stage": "engineering_smoke",
            "arm_position": 2,
            "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
            "task_ids": [0],
            "episode_indices": [12],
            "seed_base": 20260826,
            "total_episodes": 1,
            "total_successes": 1,
            "gpu": {
                "expected_uuid": "GPU-abcd",
                "visible_uuid": "abcd",
            },
            "runtime": runtime_summary,
            "stage1_measurement": {
                "records": 1,
                "records_with_errors": 0,
                "records_with_nonfinite_actions": 0,
                "records_without_action_audit": 0,
            },
            "active_latency_ms": {
                "count": 1,
                "mean": 50.0,
                "p50": 50.0,
                "p90": 50.0,
            },
            "gates": {
                "runtime_integrity": True,
                "measurement_integrity": True,
            },
        },
    )
    _write_jsonl(run_dir / "phase_route_runtime.jsonl", [_valid_runtime_record()])
    _write_jsonl(run_dir / "policy_telemetry.jsonl", [{"exit_layer": 13}])
    _write_jsonl(run_dir / "stage1_measurement.jsonl", [_valid_measurement()])
    (run_dir / "stdout.log").write_text("complete\n", encoding="utf-8")
    (run_dir / "command.sh").write_text("frozen command\n", encoding="utf-8")
    return run_dir


def test_active_validator_requires_one_fm_and_finite_action_per_call(tmp_path) -> None:
    run_dir = _active_run_fixture(tmp_path)
    result = validate_run(run_dir, repo_root=REPO_ROOT)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())

    record = _valid_runtime_record()
    record["events"][-1]["fm_calls"] = 2
    _write_jsonl(run_dir / "phase_route_runtime.jsonl", [record])
    failed = validate_run(run_dir, repo_root=REPO_ROOT)
    assert failed["status"] == "FAIL"
    assert failed["checks"]["every_runtime_call_exactly_one_fm"] is False


def test_launcher_keeps_state13_closed_until_smoke_is_separately_authorized() -> None:
    launcher = (REPO_ROOT / "scripts/run_libero_route_first_active.sh").read_text(
        encoding="utf-8"
    )
    assert 'episode_indices="${EPISODE_INDICES:-12}"' in launcher
    assert 'if [[ "${experiment_stage}" != "engineering_smoke" ]]' in launcher
    assert "State 13 remains closed" in launcher
