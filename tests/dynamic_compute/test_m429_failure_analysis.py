import numpy as np
import pytest

from scripts.dynamic_compute.analyze_m429_router_failure import (
    cosine_action_distance,
    score_tail_summary,
    threshold_scan,
    trace_diagnostics,
)


def test_cosine_action_distance_identical_and_orthogonal():
    first = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (8, 1))
    second = first.copy()
    orthogonal = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (8, 1))

    assert cosine_action_distance(first, second) == pytest.approx(0.0)
    assert cosine_action_distance(first, orthogonal) == pytest.approx(1.0)


def test_threshold_scan_exposes_safety_coverage_tradeoff():
    scores = np.array([0.95, 0.995, 0.98, 0.993], dtype=np.float64)
    teacher = np.array([13, 13, 27, 27], dtype=np.int64)
    task = np.array([0, 0, 1, 1], dtype=np.int64)
    episode = np.array([20, 20, 20, 21], dtype=np.int64)

    rows = threshold_scan(scores, teacher, task, episode, [0.97, 0.994])

    assert rows[0]["false_shallow_rows"] == 2
    assert rows[1]["false_shallow_rows"] == 0
    assert rows[0]["predicted13_coverage"] > rows[1]["predicted13_coverage"]


def test_score_tail_summary_detects_overlap():
    summary = score_tail_summary(
        np.array([0.8, 0.9, 0.99, 0.995]),
        np.array([13, 13, 27, 27]),
    )
    assert summary["required27"]["max"] == pytest.approx(0.995)
    assert summary["high_score_tail_overlaps"] is True


def test_trace_diagnostics_marks_future_layers_unavailable():
    layers = np.array([1, 3, 11, 13, 27], dtype=np.int16)
    actions = np.zeros((5, 8, 7), dtype=np.float32)
    actions[..., 0] = 1.0
    actions[4, ..., :2] = np.array([0.0, 1.0])
    shard = {
        "fm_trace_roles": np.ones(5, dtype=np.uint8),
        "fm_trace_layers": layers,
        "fm_trace_input_x": np.zeros_like(actions),
        "fm_trace_output_action": actions,
    }

    result = trace_diagnostics(shard)

    assert result["causal_candidate_layers_at_route13"] == [1, 3, 11, 13]
    assert result["future_candidate_layers_unavailable_at_route13"] == [27]
    assert result["adjacent_candidate_diagnostics"][-1]["available_at_layer13"] is False
