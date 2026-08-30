from __future__ import annotations

import numpy as np

from scripts.analyze_route_first_stage11c_coverage import (
    classify_diagnosis,
    summarize_live_scores,
    summarize_teacher_scores,
)


GRID = (0.5, 0.8, 0.9)


def test_live_summary_reports_raw_and_equal_task_coverage() -> None:
    scores = []
    tasks = []
    for task_id in range(10):
        count = task_id + 1
        scores.extend(([0.95] if task_id % 2 == 0 else [0.1]) * count)
        tasks.extend([task_id] * count)
    values = np.asarray(scores)
    selected = np.where(values >= 0.9, 13, 27)
    result = summarize_live_scores(
        values,
        np.asarray(tasks),
        selected,
        threshold_grid=GRID,
        frozen_threshold=0.9,
    )
    frozen = result["threshold_curve_descriptive_only"][-1]
    assert frozen["selected_rows"] == 25
    assert frozen["raw_policy_call_coverage"] == 25 / 55
    assert frozen["equal_task_coverage"] == 0.5
    assert result["observed_L13_calls"] == 25


def test_teacher_summary_uses_direct_teacher_safe13_ceiling() -> None:
    tasks = np.repeat(np.arange(10), 2)
    episodes = np.tile([8, 9], 10)
    teacher = np.where(tasks % 2 == 0, 13, 27)
    score13 = np.where(teacher <= 13, 0.95, 0.85)
    scores = np.stack((np.zeros_like(score13), score13), axis=1).astype(float)
    result = summarize_teacher_scores(
        scores,
        teacher,
        tasks,
        episodes,
        threshold_grid=GRID,
        confidence_level=0.9,
    )
    assert result["teacher_safe13_group_equal_ceiling"] == 0.5
    assert result["maximum_unsafe_score13"] == 0.85
    frozen = result["threshold_curve_descriptive_only"][-1]
    assert frozen["actual_coverage"] == 0.5
    assert frozen["empirical_false_safe_rate"] == 0.0


def test_classifier_rejects_threshold_only_when_all_rules_hold() -> None:
    live = {
        "threshold_curve_descriptive_only": [
            {"threshold": 0.8, "raw_policy_call_coverage": 0.15}
        ]
    }
    calibration = {"teacher_safe13_group_equal_ceiling": 0.16}
    holdout = {
        "teacher_safe13_group_equal_ceiling": 0.15,
        "maximum_unsafe_score13": 0.999,
        "threshold_curve_descriptive_only": [
            {"threshold": 0.8, "false_safe_upper_bound": 0.24}
        ],
    }
    stage11b = {"routing_usage": {"decoder_block_reduction_fraction": 0.05}}
    rules = {
        "maximum_teacher_safe13_ceiling_for_low_ceiling": 0.2,
        "threshold_probe": 0.8,
        "maximum_live_raw_coverage_for_weak_relaxation": 0.2,
        "minimum_holdout_false_safe_upper_for_unsafe_relaxation": 0.2,
        "minimum_unsafe_score_for_high_confidence_overlap": 0.99,
        "maximum_current_decoder_block_reduction": 0.1,
    }
    status, checks = classify_diagnosis(
        live, calibration, holdout, stage11b, rules=rules
    )
    assert status == "THRESHOLD_ONLY_NOT_VIABLE_NEW_DEVELOPMENT_TARGET_REQUIRED"
    assert all(checks.values())
    holdout["maximum_unsafe_score13"] = 0.8
    status, checks = classify_diagnosis(
        live, calibration, holdout, stage11b, rules=rules
    )
    assert status == "THRESHOLD_ONLY_REMAINS_DIAGNOSTIC_CANDIDATE"
    assert checks["unsafe_examples_have_high_confidence_overlap"] is False


def test_protocol_forbids_stage10_and_new_threshold_selection() -> None:
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[2]
    path = root / "configs/research/route_first_stage11c_coverage_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["decision"]["emits_new_threshold"] is False
    assert protocol["decision"]["authorizes_runtime_change"] is False
    assert all(protocol["forbidden"].values())
    assert "stage10" not in json.dumps(protocol["inputs"]).lower()
