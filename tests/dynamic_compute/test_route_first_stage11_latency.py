from __future__ import annotations

import pytest

from scripts.analyze_route_first_stage11_latency import (
    CallRecord,
    Stage11LatencyError,
    _comparison,
    _method_summary,
    latency_summary,
    route_overlay_for_call,
)


def _record(
    method: str,
    layer: int,
    wall: float,
    *,
    fm_calls: int = 1,
    authoritative_fm_calls: int | None = None,
    components=None,
) -> CallRecord:
    return CallRecord(
        method=method,
        task_id=0,
        replicate_id=0,
        arm_position=1,
        call_ordinal=0,
        layer=layer,
        telemetry_fm_event_sum=fm_calls,
        authoritative_fm_calls=(
            fm_calls if authoritative_fm_calls is None else authoritative_fm_calls
        ),
        policy_wall_ms=wall,
        policy_cuda_ms=wall - 0.1,
        components=components or {},
    )


def _route_components() -> dict[str, tuple[float, ...]]:
    return {
        "runtime_begin": (1.0,),
        "visual_capture": (2.0,),
        "runtime_prepare": (10.0,),
        "selected_action_route": (0.5,),
        "runtime_commit": (0.5,),
        "phase_estimator": (4.0,),
        "router_predict": (1.0,),
        # adapter_begin is nested in runtime_begin/runtime_prepare and must not
        # be added to the non-overlapping overlay.
        "adapter_begin": (0.2, 1.5),
    }


def test_latency_summary_is_strict_and_uses_nearest_rank() -> None:
    result = latency_summary([4.0, 1.0, 3.0, 2.0])
    assert result["mean"] == 2.5
    assert result["p50"] == 2.0
    assert result["p90"] == 4.0
    assert result["p95"] == 4.0
    with pytest.raises(Stage11LatencyError, match="finite"):
        latency_summary([float("nan")])


def test_route_overlay_does_not_double_count_nested_prepare_events() -> None:
    record = _record(
        "route_first_stage8", 13, 100.0, components=_route_components()
    )
    result = route_overlay_for_call(record)
    assert result["instrumented_route_overlay"] == 14.0
    assert result["runtime_prepare_other"] == 5.0
    assert result["uninstrumented_policy_residual"] == 86.0
    assert result["instrumented_route_overlay_fraction"] == 0.14


def test_route_overlay_rejects_missing_or_repeated_top_level_events() -> None:
    missing = _route_components()
    del missing["runtime_commit"]
    with pytest.raises(Stage11LatencyError, match="missing"):
        route_overlay_for_call(
            _record("route_first_stage8", 13, 100.0, components=missing)
        )

    repeated = _route_components()
    repeated["runtime_begin"] = (1.0, 2.0)
    with pytest.raises(Stage11LatencyError, match="not unique"):
        route_overlay_for_call(
            _record("route_first_stage8", 13, 100.0, components=repeated)
        )


def test_descriptive_comparison_keeps_coverage_and_path_ratios_separate() -> None:
    methods = {
        "original_a1": _method_summary(
            [
                _record("original_a1", 11, 80.0, fm_calls=7),
                _record("original_a1", 27, 200.0, fm_calls=15),
            ]
        ),
        "candidate_first_v3": _method_summary(
            [_record("candidate_first_v3", 27, 150.0, fm_calls=3)]
        ),
        "route_first_stage8": _method_summary(
            [
                _record(
                    "route_first_stage8",
                    13,
                    70.0,
                    fm_calls=3,
                    authoritative_fm_calls=1,
                    components=_route_components(),
                ),
                _record(
                    "route_first_stage8",
                    27,
                    90.0,
                    fm_calls=3,
                    authoritative_fm_calls=1,
                    components=_route_components(),
                ),
            ]
        ),
    }
    result = _comparison(methods)
    assert result["route_L13_vs_A1_L11_policy_wall_p50_ratio_descriptive"] == 0.875
    assert result["route_L27_vs_A1_L27_policy_wall_p50_ratio_descriptive"] == 0.45
    assert result["A1_L11_share"] == 0.5
    assert result["route_L13_share"] == 0.5
    assert result["interpretation"]["stage10_does_not_identify_a_safe_new_threshold"]
    route = methods["route_first_stage8"]
    assert route["telemetry_fm_event_sum"]["total"] == 6
    assert route["authoritative_fm_calls"]["total"] == 2
