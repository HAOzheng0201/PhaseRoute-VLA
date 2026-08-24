from __future__ import annotations

import pytest

from scripts.summarize_stage1_five_arm_smoke import (
    build_summary,
    latency_summary,
    reduction,
    summarize_telemetry,
)


CHECKPOINT_SHA = "d" * 64
METHODS = ("fixed_l11", "fixed_l13", "fixed_l27", "original_a1", "phase_route_v3")


def _method(*, fm_calls: float = 1.0, wall_ms: float = 10.0) -> dict:
    return {
        "success": True,
        "policy_calls": 1,
        "fm_calls_per_policy_call": fm_calls,
        "policy_wall_latency_ms": {"mean": wall_ms},
        "episode_wall_seconds": wall_ms / 1000.0,
        "instruction_hashes": ["instruction"],
    }


def _bindings() -> dict[str, dict[str, str]]:
    output = {
        name: {
            "GPU_UUID": "GPU-11111111-2222-3333-4444-555555555555",
            "CHECKPOINT": "/checkpoint",
        }
        for name in METHODS
    }
    output["original_a1"]["CHECKPOINT_SHA256"] = CHECKPOINT_SHA
    output["phase_route_v3"]["CHECKPOINT_SHA256"] = CHECKPOINT_SHA
    return output


def test_latency_summary_uses_nearest_rank_and_strict_finite_values() -> None:
    summary = latency_summary([4.0, 1.0, 3.0, 2.0])
    assert summary == {
        "count": 4,
        "sum": 10.0,
        "mean": 2.5,
        "p50": 2.0,
        "p95": 4.0,
        "max": 4.0,
    }
    with pytest.raises(ValueError, match="finite"):
        latency_summary([float("nan")])


def test_telemetry_summary_uses_zero_based_layer_indices_as_depths() -> None:
    rows = [
        {
            "schema_version": "phase-route-vla.telemetry.v1",
            "action_shape": [1, 8, 7],
            "exit_layer": 11,
            "fm_calls": 1,
            "latency_ms": 5.0,
            "instruction_hash": "same",
        },
        {
            "schema_version": "phase-route-vla.telemetry.v1",
            "action_shape": [1, 8, 7],
            "exit_layer": 27,
            "fm_calls": 3,
            "latency_ms": 7.0,
            "instruction_hash": "same",
        },
    ]
    summary = summarize_telemetry(rows)
    assert summary["selected_layer_index_mean"] == 19.0
    assert summary["executed_depth_ratio_to_l27"] == pytest.approx(20.0 / 28.0)
    assert summary["fm_calls_per_policy_call"] == 2.0


def test_five_arm_summary_checks_identity_and_keeps_negative_speed_result() -> None:
    methods = {name: _method() for name in METHODS}
    methods["original_a1"] = _method(fm_calls=8.0, wall_ms=100.0)
    methods["phase_route_v3"] = _method(fm_calls=6.0, wall_ms=140.0)
    result = build_summary(
        methods,
        _bindings(),
        task_id=0,
        episode_index=0,
        seed=1,
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    comparison = result["descriptive_comparisons"]["phase_vs_original_a1"]
    assert result["status"] == "PASS"
    assert comparison["fm_calls_per_policy_call_reduction_fraction"] == 0.25
    assert comparison["policy_wall_mean_reduction_fraction"] == pytest.approx(-0.4)

    mismatched = _bindings()
    mismatched["fixed_l11"]["GPU_UUID"] = "GPU-other"
    failed = build_summary(
        methods,
        mismatched,
        task_id=0,
        episode_index=0,
        seed=1,
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert failed["status"] == "FAIL"
    assert failed["checks"]["same_physical_gpu_uuid"] is False


def test_reduction_rejects_nonpositive_reference() -> None:
    with pytest.raises(ValueError, match="positive"):
        reduction(1.0, 0.0)
