from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from a1.vla.dynamic_compute.v3 import gripper_v2_calibration as gc  # noqa: E402
from a1.vla.dynamic_compute.v3.gripper_v2_models import (  # noqa: E402
    ordered_cutpoints,
)


def test_frozen_contract_score_and_access_boundary_are_exact() -> None:
    contract = gc.load_d3_contract(REPO_ROOT)
    assert contract["status"] == gc.D3_STATUS
    assert contract["score"]["name"] == "step_any_mismatch_probability"
    assert contract["score"]["source_target_index"] == 0
    assert contract["score"][
        "one_global_threshold_shared_across_layers_tasks_and_time"
    ] is True
    assert contract["frozen_model"]["refit_on_calibration"] is False
    assert contract["lineage"]["independent_test_access_allowed"] is False
    assert contract["threshold_selection"]["always_defer_is_valid"] is False
    assert contract["gate"]["on_pass_active_control_authorized"] is False
    assert contract["gate"]["on_pass_independent_test_authorized"] is False
    observed = json.loads(
        (
            REPO_ROOT
            / "configs/research/v3/gripper_v2/d3_calibration_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert observed == contract


def test_calibration_selection_and_seed_formula_are_exact() -> None:
    selection = gc.load_calibration_selection(REPO_ROOT)
    assert len(selection) == 100
    assert selection[0] == gc.CalibrationEpisode(0, 30, 20260841)
    assert selection[-1] == gc.CalibrationEpisode(9, 39, 20350850)
    for task in range(10):
        window = gc.task_calibration_window(selection, task)
        assert [record.episode_index for record in window] == list(range(30, 40))
        assert all(
            record.seed
            == gc.expected_calibration_seed(task, record.episode_index)
            for record in window
        )
    with pytest.raises(gc.D3CalibrationError):
        gc.expected_calibration_seed(0, 40)


def test_exact_clopper_pearson_boundary_is_not_wald() -> None:
    assert gc.clopper_pearson_upper(0, 100) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 100.0)
    )
    assert gc.clopper_pearson_upper(0, 58) > 0.05
    assert gc.clopper_pearson_upper(0, 59) < 0.05
    assert gc.clopper_pearson_upper(1, 100) < 0.05
    assert gc.clopper_pearson_upper(2, 100) > 0.05
    assert gc.clopper_pearson_upper(0, 0) == 1.0
    with pytest.raises(gc.D3CalibrationError):
        gc.clopper_pearson_upper(2, 1)


def _calibration_rows(*, unsafe_clusters: set[int] | None = None):
    unsafe_clusters = unsafe_clusters or set()
    task = torch.arange(100, dtype=torch.long) // 10
    episode = 30 + torch.arange(100, dtype=torch.long) % 10
    first_score = (torch.arange(100, dtype=torch.float64) + 1.0) / 1000.0
    second_score = 0.5 + first_score
    score = torch.cat((first_score, second_score))
    task_id = torch.cat((task, task))
    episode_index = torch.cat((episode, episode))
    candidate_layer = torch.cat(
        (torch.full((100,), 11), torch.full((100,), 13))
    ).long()
    step = torch.zeros(200, dtype=torch.bool)
    for cluster in unsafe_clusters:
        step[cluster] = True
    transition = torch.zeros_like(step)
    return {
        "score": score,
        "step_mismatch": step,
        "transition_mismatch": transition,
        "task_id": task_id,
        "episode_index": episode_index,
        "candidate_layer": candidate_layer,
    }


def test_threshold_maximizes_cluster_coverage_and_uses_smaller_tie() -> None:
    result = gc.select_global_threshold(**_calibration_rows())
    assert result["status"] == "PASS_V3_D3_CALIBRATION_GATE"
    selected = result["selected"]
    assert selected is not None
    assert selected["threshold"] == pytest.approx(0.1)
    assert selected["safe_clusters"] == 100
    assert selected["false_safe_clusters"] == 0
    assert selected["safe_cluster_coverage"] == 1.0
    assert selected["false_safe_cluster_ucb95"] < 0.05
    assert result["feasible_threshold_count"] > 0
    assert result["checks"]["single_global_threshold_only"] is True


def test_one_of_100_false_clusters_passes_but_all_unsafe_fails() -> None:
    one = gc.select_global_threshold(**_calibration_rows(unsafe_clusters={99}))
    assert one["status"] == "PASS_V3_D3_CALIBRATION_GATE"
    assert one["selected"]["safe_clusters"] == 100
    assert one["selected"]["false_safe_clusters"] == 1
    assert one["selected"]["false_safe_cluster_ucb95"] < 0.05
    all_unsafe = gc.select_global_threshold(
        **_calibration_rows(unsafe_clusters=set(range(100)))
    )
    assert all_unsafe["status"] == "NEGATIVE_V3_D3_CALIBRATION_GATE"
    assert all_unsafe["selected"] is None
    assert all_unsafe["checks"]["always_defer_not_accepted"] is False


def test_missing_cluster_transition_leakage_and_nonfinite_fail_closed() -> None:
    values = _calibration_rows()
    keep = torch.ones(200, dtype=torch.bool)
    keep[[99, 199]] = False
    missing = {name: value[keep] for name, value in values.items()}
    with pytest.raises(gc.D3CalibrationError, match="100 calibration clusters"):
        gc.select_global_threshold(**missing)
    leaked = _calibration_rows()
    leaked["transition_mismatch"][0] = True
    with pytest.raises(gc.D3CalibrationError, match="step-contained"):
        gc.select_global_threshold(**leaked)
    invalid = _calibration_rows()
    invalid["score"][0] = math.nan
    with pytest.raises(gc.D3CalibrationError, match="finite FP64"):
        gc.select_global_threshold(**invalid)


def _ordinal_state(target: int) -> dict[str, object]:
    support_max = (8, 7)[target]
    raw_base = torch.zeros(2, dtype=torch.float64)
    raw_increments = torch.zeros(
        2, support_max - 2, dtype=torch.float64
    )
    return {
        "normalizer_mean": torch.zeros(97, dtype=torch.float64),
        "normalizer_scale": torch.ones(97, dtype=torch.float64),
        "weight": torch.zeros(97, dtype=torch.float64),
        "l2_lambda": 0.1,
        "final_loss": 0.0,
        "raw_base": raw_base,
        "raw_increments": raw_increments,
        "cutpoints": ordered_cutpoints(raw_base, raw_increments),
        "target_index": target,
        "support_max": support_max,
    }


def test_frozen_state_scoring_uses_step_occurrence_not_posthoc_score() -> None:
    state = {
        "occurrence": {
            "normalizer_mean": torch.zeros(97, dtype=torch.float64),
            "normalizer_scale": torch.ones(97, dtype=torch.float64),
            "weight": torch.zeros(2, 97, dtype=torch.float64),
            "l2_lambda": 0.01,
            "final_loss": 0.0,
            "anchor_probability": torch.tensor(
                [[0.2, 0.3], [0.4, 0.5]], dtype=torch.float64
            ),
        },
        "zt_step": {},
        "zt_transition": {},
        "ordinal_step": _ordinal_state(0),
        "ordinal_transition": _ordinal_state(1),
    }
    predictions = gc.score_calibration_features(
        state,
        torch.zeros(2, 97, dtype=torch.float64),
        torch.tensor([11, 13], dtype=torch.long),
    )
    assert torch.equal(
        predictions["score"], torch.tensor([0.2, 0.4], dtype=torch.float64)
    )
    assert predictions["occurrence_probability"].shape == (2, 2)
    assert predictions["ordinal_step_probability"].shape == (2, 8)
    assert predictions["ordinal_transition_probability"].shape == (2, 7)
    assert predictions["ordinal_expected_fraction"].shape == (2, 2)
    assert bool(torch.isfinite(predictions["ordinal_expected_fraction"]).all())
