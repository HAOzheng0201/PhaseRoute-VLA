from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from a1.vla.dynamic_compute.route_first_calibration import RouteFirstThresholdRule
from a1.vla.dynamic_compute.route_first_holdout import (
    ROUTE_FIRST_HOLDOUT_FAIL_STATUS,
    ROUTE_FIRST_HOLDOUT_PASS_STATUS,
    evaluate_route_first_holdout,
    load_route_first_holdout_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _rule(
    *,
    minimum_coverage: float = 0.1,
    minimum_rows: float = 2.0,
    maximum_false_safe: float = 0.25,
) -> RouteFirstThresholdRule:
    return RouteFirstThresholdRule(
        minimum_coverage=minimum_coverage,
        minimum_effective_selected_rows=minimum_rows,
        maximum_empirical_false_safe_rate=maximum_false_safe,
        maximum_false_safe_upper_bound=1.0,
    )


def _evaluate(
    scores: np.ndarray,
    teacher: np.ndarray,
    episodes: np.ndarray,
    *,
    pooled_rule: RouteFirstThresholdRule | None = None,
    per_episode_rule: RouteFirstThresholdRule | None = None,
) -> dict[str, object]:
    rows = teacher.size
    return evaluate_route_first_holdout(
        scores,
        teacher,
        np.arange(rows, dtype=np.int64) % 4,
        episodes,
        threshold13=0.8,
        enabled11=False,
        enabled13=True,
        expected_episode_indices=(10, 11),
        pooled_rule=pooled_rule or _rule(),
        per_episode_rule=per_episode_rule or _rule(),
        confidence_level=0.9,
        score_quantiles=(0.0, 0.5, 1.0),
    )


def test_repository_holdout_protocol_binds_stage6_artifacts() -> None:
    protocol = load_route_first_holdout_protocol(
        REPO_ROOT / "configs/route_first_holdout_protocol.json"
    )

    frozen = protocol["frozen_calibrated_router"]
    assert frozen["enabled11"] is False
    assert frozen["enabled13"] is True
    assert frozen["threshold13"] == 0.9174261218080999
    assert protocol["data"]["engineering_holdout_episode_indices"] == [10, 11]


def test_holdout_passes_only_when_pooled_and_both_states_pass() -> None:
    score13 = np.asarray([0.95, 0.9, 0.4, 0.3, 0.96, 0.85, 0.2, 0.1])
    scores = np.column_stack((score13 * 0.5, score13))
    teacher = np.asarray([13, 11, 27, 27, 13, 13, 27, 27])
    episodes = np.asarray([10, 10, 10, 10, 11, 11, 11, 11])

    result = _evaluate(scores, teacher, episodes)

    assert result["status"] == ROUTE_FIRST_HOLDOUT_PASS_STATUS
    assert result["passed"] is True
    assert result["threshold_changed"] is False
    assert result["routing"]["selected_layer_counts"] == {
        "11": 0,
        "13": 4,
        "27": 4,
    }
    assert result["routing"]["raw_selected13_teacher_counts"] == {
        "11": 1,
        "13": 3,
        "27": 0,
    }


def test_one_bad_state_fails_closed_even_when_pooled_gate_is_lax() -> None:
    score13 = np.asarray([0.95, 0.9, 0.2, 0.1, 0.96, 0.85, 0.2, 0.1])
    scores = np.column_stack((score13 * 0.5, score13))
    teacher = np.asarray([13, 13, 27, 27, 13, 27, 27, 27])
    episodes = np.asarray([10, 10, 10, 10, 11, 11, 11, 11])

    result = _evaluate(
        scores,
        teacher,
        episodes,
        pooled_rule=_rule(maximum_false_safe=0.5),
        per_episode_rule=_rule(maximum_false_safe=0.2),
    )

    assert result["pooled_safe13"]["passed"] is True
    assert result["per_episode_index_safe13"]["10"]["passed"] is True
    assert result["per_episode_index_safe13"]["11"]["passed"] is False
    assert result["status"] == ROUTE_FIRST_HOLDOUT_FAIL_STATUS
    assert result["passed"] is False
    assert result["failures"] == ["EPISODE_INDEX_11_SAFE13_GATE_FAILED"]


@pytest.mark.parametrize(
    ("enabled11", "enabled13"),
    [(True, True), (False, False)],
)
def test_holdout_rejects_changed_enabled_heads(
    enabled11: bool, enabled13: bool
) -> None:
    scores = np.asarray([[0.1, 0.9], [0.1, 0.9]])
    with pytest.raises(ValueError, match="disabled L11 and enabled L13"):
        evaluate_route_first_holdout(
            scores,
            np.asarray([13, 13]),
            np.asarray([0, 0]),
            np.asarray([10, 11]),
            threshold13=0.8,
            enabled11=enabled11,
            enabled13=enabled13,
            expected_episode_indices=(10, 11),
            pooled_rule=_rule(minimum_rows=1.0),
            per_episode_rule=_rule(minimum_rows=1.0),
            confidence_level=0.9,
            score_quantiles=(0.0, 1.0),
        )


def test_holdout_rejects_wrong_episode_indices_and_quantiles() -> None:
    scores = np.asarray([[0.1, 0.9], [0.1, 0.9]])
    common = dict(
        scores=scores,
        teacher_layer=np.asarray([13, 13]),
        task_id=np.asarray([0, 0]),
        episode_index=np.asarray([10, 12]),
        threshold13=0.8,
        enabled11=False,
        enabled13=True,
        expected_episode_indices=(10, 11),
        pooled_rule=_rule(minimum_rows=1.0),
        per_episode_rule=_rule(minimum_rows=1.0),
        confidence_level=0.9,
        score_quantiles=(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="episode indices differ"):
        evaluate_route_first_holdout(**common)

    common["episode_index"] = np.asarray([10, 11])
    common["score_quantiles"] = (0.5, 0.1)
    with pytest.raises(ValueError, match="quantiles are invalid"):
        evaluate_route_first_holdout(**common)
