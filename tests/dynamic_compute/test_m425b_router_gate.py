from __future__ import annotations

from scripts.dynamic_compute.evaluate_m425b_temporal_router import science_gates


def _method(*, exact=0.5, false_shallow=0, coverage=0.4, reduction=0.2):
    return {
        "metrics": {
            "exact_accuracy": exact,
            "false_shallow": false_shallow,
            "teacher27_false_shallow": false_shallow,
            "teacher27_rows": 5,
            "shallow_coverage": coverage,
        },
        "estimated_latency": {"reduction_fraction": reduction},
    }


def test_temporal_router_gate_requires_every_safety_and_utility_condition() -> None:
    fit = {
        "oof_metrics": {"false_shallow": 0},
        "calibration_metrics": {"false_shallow": 0},
    }
    gates = science_gates(
        _method(),
        {"hidden_only": _method(exact=0.4), "step_proprio": _method(exact=0.45)},
        fit,
    )
    assert all(gates.values())

    unsafe = science_gates(
        _method(false_shallow=1),
        {"hidden_only": _method(exact=0.4), "step_proprio": _method(exact=0.45)},
        fit,
    )
    assert unsafe["sealed_false_shallow_zero"] is False
    assert unsafe["sealed_route27_false_shallow_zero"] is False


def test_temporal_router_gate_rejects_control_regression() -> None:
    fit = {
        "oof_metrics": {"false_shallow": 0},
        "calibration_metrics": {"false_shallow": 0},
    }
    gates = science_gates(
        _method(exact=0.4),
        {"hidden_only": _method(exact=0.45), "step_proprio": _method(exact=0.35)},
        fit,
    )
    assert gates["exact_not_below_controls"] is False
