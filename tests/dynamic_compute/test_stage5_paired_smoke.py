import pytest

from scripts.summarize_stage5_paired_smoke import (
    build_summary,
    parse_baseline_log,
)


def _telemetry(task_id, layers, instruction_hash="same"):
    return [
        {
            "schema_version": "phase-route-vla.telemetry.v1",
            "episode_id": f"libero_10:task{task_id}:episode0",
            "task_id": task_id,
            "instruction_hash": instruction_hash,
            "exit_layer": layer,
            "fm_calls": 5 if layer < 27 else 7,
            "latency_ms": 10.0 + layer,
        }
        for layer in layers
    ]


def _log():
    return """\
Task 0: first task
Episode seed: 20260823
Exit layers this episode: [11, 27]
Success: True
Episode duration: 2.00s
Task 1: second task
Episode seed: 20270823
Exit layers this episode: [27]
Success: True
Episode duration: 3.00s
"""


def _phase_summary():
    return {
        "schema_version": "phase-route-vla.libero-evaluation-summary.v1",
        "method": "phase_route_v3",
        "total_successes": 1,
        "telemetry_errors": 0,
        "episodes": [
            {
                "task_id": 0,
                "episode_index": 0,
                "seed": 20260823,
                "success": True,
                "policy_calls": 1,
                "selected_layers": {"13": 1},
                "wall_seconds": 1.5,
            },
            {
                "task_id": 1,
                "episode_index": 0,
                "seed": 20270823,
                "success": False,
                "policy_calls": 2,
                "selected_layers": {"13": 1, "27": 1},
                "wall_seconds": 4.0,
            },
        ],
    }


def test_builds_identity_checked_descriptive_summary():
    baseline = _telemetry(0, [11, 27]) + _telemetry(1, [27])
    phase = _telemetry(0, [13]) + _telemetry(1, [13, 27])

    result = build_summary(
        parse_baseline_log(_log()),
        baseline,
        _phase_summary(),
        phase,
        expected_task_ids=(0, 1),
    )

    assert result["status"] == "PASS"
    assert result["paired_outcomes"] == {
        "both_success": 1,
        "baseline_only_success": 1,
        "phase_only_success": 0,
        "both_failure": 0,
    }
    assert result["original_a1"]["policy_calls"] == 3
    assert result["phase_route_v3"]["policy_calls"] == 3
    assert result["claim_boundary"]["formal_speedup_claim"] is False


def test_rejects_log_telemetry_exit_layer_drift():
    baseline = _telemetry(0, [27, 27]) + _telemetry(1, [27])
    phase = _telemetry(0, [13]) + _telemetry(1, [13, 27])

    with pytest.raises(ValueError, match="log/telemetry exit layers differ"):
        build_summary(
            parse_baseline_log(_log()),
            baseline,
            _phase_summary(),
            phase,
            expected_task_ids=(0, 1),
        )


def test_rejects_instruction_mismatch_as_failed_integrity_check():
    baseline = _telemetry(0, [11, 27]) + _telemetry(1, [27])
    phase = _telemetry(0, [13], "different") + _telemetry(1, [13, 27])

    result = build_summary(
        parse_baseline_log(_log()),
        baseline,
        _phase_summary(),
        phase,
        expected_task_ids=(0, 1),
    )

    assert result["status"] == "FAIL"
    assert not result["checks"]["instruction_alignment"]
