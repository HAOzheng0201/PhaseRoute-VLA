from __future__ import annotations

import json
from pathlib import Path

from a1.vla.dynamic_compute.route_first_active_protocol import (
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
)
from scripts.validate_phase_route_v3_run import validate_run as validate_v3_run
from scripts.validate_route_first_stage9_candidate_arm import validate_candidate_arm


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _candidate_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    _write_json(
        run_dir / "stage9_preflight.json",
        {
            "status": "PASS",
            "scope": "route_first_stage9_active_preflight",
            "simulator_episode_opened": False,
            "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
            "expected_gpu_uuid": "GPU-abcd",
        },
    )
    _write_json(
        run_dir / "preflight.json",
        {
            "status": "PASS",
            "scope": "phase_route_v3_release_preflight",
            "expected_gpu_uuid": "GPU-abcd",
            "cuda": {"visible_uuid": "abcd"},
        },
    )
    runtime = {
        "records": 1,
        "policy_calls": 1,
        "prepared_calls": 1,
        "committed_calls": 1,
        "error_count": 0,
        "records_with_errors": 0,
        "selected_layers": {"11": 0, "13": 1, "27": 0},
    }
    measurement = {
        "records": 1,
        "records_with_errors": 0,
        "records_with_nonfinite_actions": 0,
        "records_without_action_audit": 0,
        "latency_ms": {"policy_wall": {"count": 1, "p50": 100.0}},
    }
    _write_json(
        run_dir / "evaluation_summary.json",
        {
            "schema_version": "phase-route-vla.libero-evaluation-summary.v1",
            "method": "phase_route_v3",
            "suite": "libero_10",
            "task_ids": [0],
            "episode_indices": [12],
            "seed_base": 20260826,
            "total_episodes": 1,
            "total_successes": 1,
            "success_rate": 1.0,
            "runtime": runtime,
            "stage1_measurement": measurement,
        },
    )
    _write_jsonl(run_dir / "phase_route_runtime.jsonl", [{"selected_layer": 13}])
    _write_jsonl(run_dir / "policy_telemetry.jsonl", [{"exit_layer": 13}])
    _write_jsonl(
        run_dir / "stage1_measurement.jsonl",
        [{"selected_layer": 13, "action_finite": True}],
    )
    (run_dir / "stdout.log").write_text("complete\n", encoding="utf-8")
    (run_dir / "command.sh").write_text("frozen command\n", encoding="utf-8")
    v3_attestation = validate_v3_run(run_dir)
    assert v3_attestation["status"] == "PASS"
    _write_json(run_dir / "run_attestation.json", v3_attestation)
    return run_dir


def test_candidate_arm_binds_v3_run_to_stage9_smoke(tmp_path) -> None:
    run_dir = _candidate_fixture(tmp_path)
    result = validate_candidate_arm(run_dir, repo_root=REPO_ROOT)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["method"] == "candidate_first_v3"
    assert result["arm_position"] == 1


def test_candidate_arm_rejects_state_drift_even_if_general_v3_run_passes(
    tmp_path,
) -> None:
    run_dir = _candidate_fixture(tmp_path)
    evaluation_path = run_dir / "evaluation_summary.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["episode_indices"] = [11]
    _write_json(evaluation_path, evaluation)

    assert validate_v3_run(run_dir)["status"] == "PASS"
    result = validate_candidate_arm(run_dir, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert result["checks"]["state_is_frozen_smoke_state"] is False


def test_candidate_launcher_is_state12_only_and_always_measured() -> None:
    launcher = (
        REPO_ROOT / "scripts/run_libero_route_first_stage9_candidate.sh"
    ).read_text(encoding="utf-8")
    assert 'episode_indices="${EPISODE_INDICES:-12}"' in launcher
    assert '"${episode_indices}" != "12"' in launcher
    assert "--measurement-output" in launcher
    assert "validate_route_first_active_preflight.py" in launcher
    assert "validate_route_first_stage9_candidate_arm.py" in launcher
