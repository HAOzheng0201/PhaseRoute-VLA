from scripts.dynamic_compute.summarize_m416_risk_gate import build_gate_summary


def _analysis(metrics, fold_r2):
    return {
        "status": "PASS",
        "records": 100,
        "tasks": list(range(len(fold_r2))),
        "constant_nested_loto_metrics": {"mae": 1.0},
        "feature_analyses": {
            "phase_stage": {
                "nested_loto_metrics": metrics,
                "folds": [
                    {"held_task": index, "metrics": {"r2": value}}
                    for index, value in enumerate(fold_r2)
                ],
            }
        },
    }


def test_gate_recommends_only_a_cross_task_predictive_risk_head():
    result = build_gate_summary(
        _analysis({"mae": 0.8, "pearson": 0.6, "r2": 0.3}, [0.2] * 5),
        feature_name="phase_stage",
        min_mae_relative_improvement=0.1,
        min_pearson=0.4,
        min_r2=0.2,
        min_nonnegative_task_r2_fraction=0.8,
    )

    assert result["status"] == "PASS"
    assert result["online_rollout_recommended"] is True
    assert all(result["gates"].values())


def test_gate_rejects_negative_held_task_r2():
    result = build_gate_summary(
        _analysis({"mae": 0.8, "pearson": 0.6, "r2": 0.3}, [0.2, -0.1, -0.2]),
        feature_name="phase_stage",
        min_mae_relative_improvement=0.1,
        min_pearson=0.4,
        min_r2=0.2,
        min_nonnegative_task_r2_fraction=0.8,
    )

    assert result["status"] == "PASS"
    assert result["online_rollout_recommended"] is False
    assert result["gates"]["nonnegative_task_r2_fraction"] is False
