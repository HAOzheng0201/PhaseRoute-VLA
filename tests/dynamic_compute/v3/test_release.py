from __future__ import annotations

import json
from pathlib import Path

import pytest

from a1.vla.dynamic_compute.v3.release import (
    PHASE_STATE_SHA256,
    parse_index_spec,
    summarize_runtime_records,
    validate_general_release_selection,
    validate_phase_route_v3_release,
)
from scripts.validate_phase_route_v3_run import validate_run
from scripts.validate_phase_route_v3_release import validate_physical_gpu_index


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_v3_preflight_preserves_default_gpu_scope_and_allows_explicit_all8() -> None:
    validate_physical_gpu_index(3, 4)
    validate_physical_gpu_index(7, 8)
    with pytest.raises(ValueError, match="0-3"):
        validate_physical_gpu_index(4, 4)
    with pytest.raises(ValueError, match="must be 4 or 8"):
        validate_physical_gpu_index(0, 16)


def test_clean_clone_v3_release_gate_loads_exact_frozen_payloads() -> None:
    result = validate_phase_route_v3_release(
        REPO_ROOT,
        require_backbone=False,
        validate_payloads=True,
    )

    assert result["status"] == "PASS"
    assert result["deployment_authorized"] is False
    assert result["payload_validation"]["router_models"] == 5
    assert result["payload_validation"]["phase_state_sha256"] == PHASE_STATE_SHA256
    assert result["historical_validation"]["legacy_evidence"] == {
        "verified": True,
        "verified_count": 28,
        "total_bytes": 1_737_937,
        "manifest_sha256": (
            "4ae5b617525a1f575f62700ab46434a1c9e8b20b9d13863b7ae8787f74c0ea6a"
        ),
    }
    assert all(result["checks"].values())


def test_runtime_summary_counts_only_frozen_route_layers() -> None:
    summary = summarize_runtime_records(
        (
            {"selected_layer": 11, "fallback": False, "errors": []},
            {"selected_layer": 13, "fallback": False, "errors": []},
            {"selected_layer": 27, "fallback": True, "errors": ["bad context"]},
        )
    )

    assert summary["records"] == 3
    assert summary["selected_layers"] == {"11": 1, "13": 1, "27": 1}
    assert summary["early_exit_records"] == 2
    assert summary["fallback_records"] == 1
    assert summary["records_with_errors"] == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    (("0", (0,)), ("0,2-4", (0, 2, 3, 4)), ("9,1", (9, 1))),
)
def test_release_index_spec_is_explicit_and_order_preserving(text, expected) -> None:
    assert parse_index_spec(text, name="task_ids") == expected


@pytest.mark.parametrize("text", ("", "2-1", "1,1", "a", "1,,2"))
def test_release_index_spec_rejects_ambiguous_values(text) -> None:
    with pytest.raises(ValueError):
        parse_index_spec(text, name="episode_indices")


def test_general_release_selection_refuses_consumed_d9_states() -> None:
    with pytest.raises(ValueError, match="refuses consumed D9"):
        validate_general_release_selection("0", "39-40")


def test_general_release_selection_accepts_engineering_smoke() -> None:
    assert validate_general_release_selection("0,2-3", "0,3") == (
        (0, 2, 3),
        (0, 3),
    )


@pytest.mark.parametrize("measurement_enabled", (False, True))
def test_completed_general_run_attestation_is_machine_checkable(
    tmp_path, measurement_enabled
) -> None:
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {"status": "PASS", "scope": "phase_route_v3_release_preflight"}
        ),
        encoding="utf-8",
    )
    runtime = {
        "records": 1,
        "policy_calls": 1,
        "prepared_calls": 1,
        "committed_calls": 1,
        "error_count": 0,
        "records_with_errors": 0,
        "selected_layers": {"11": 1, "13": 0, "27": 0},
    }
    evaluation = {
        "schema_version": "phase-route-vla.libero-evaluation-summary.v1",
        "method": "phase_route_v3",
        "suite": "libero_10",
        "task_ids": [0],
        "episode_indices": [0],
        "total_episodes": 1,
        "total_successes": 1,
        "success_rate": 1.0,
        "runtime": runtime,
    }
    if measurement_enabled:
        evaluation["stage1_measurement"] = {
            "records": 1,
            "records_with_errors": 0,
            "records_with_nonfinite_actions": 0,
            "records_without_action_audit": 0,
        }
        (tmp_path / "stage1_measurement.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    (tmp_path / "evaluation_summary.json").write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )
    (tmp_path / "phase_route_runtime.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "policy_telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "stdout.log").write_text("complete\n", encoding="utf-8")

    result = validate_run(tmp_path)

    assert result["status"] == "PASS"
    assert result["scope"] == "general_simulator_run_not_D9_retest"
    assert all(result["checks"].values())
