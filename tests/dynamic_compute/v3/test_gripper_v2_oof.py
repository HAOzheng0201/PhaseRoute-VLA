from __future__ import annotations

import pytest
import torch

from a1.vla.dynamic_compute.v3 import gripper_v2_oof as go


@pytest.fixture(scope="module")
def development_data() -> go.DevelopmentData:
    generator = torch.Generator().manual_seed(20260820)
    features = []
    layers = []
    source_rows = []
    tasks = []
    episodes = []
    call_codes = []
    source_row = 0
    for task in range(10):
        for episode in range(12, 30):
            for call in range(12):
                features.append(torch.randn(2, 97, generator=generator))
                layers.extend((11, 13))
                source_rows.extend((source_row, source_row))
                tasks.extend((task, task))
                episodes.extend((episode, episode))
                call_codes.extend((2 * call, 2 * call + 1))
                source_row += 1
    feature = torch.cat(features)
    layer = torch.tensor(layers, dtype=torch.long)
    task = torch.tensor(tasks, dtype=torch.long)
    episode = torch.tensor(episodes, dtype=torch.long)
    code = torch.tensor(call_codes, dtype=torch.long)
    occurrence = torch.stack(
        (
            (code + task + episode).remainder(3) != 0,
            (code + 2 * task + episode).remainder(4) != 0,
        ),
        dim=1,
    )
    count = torch.stack(
        (
            torch.where(occurrence[:, 0], 1 + (code + task).remainder(8), 0),
            torch.where(occurrence[:, 1], 1 + (code + episode).remainder(7), 0),
        ),
        dim=1,
    ).long()
    result = go.DevelopmentData(
        features=feature,
        candidate_layer=layer,
        source_row=torch.tensor(source_rows, dtype=torch.long),
        task_id=task,
        episode_index=episode,
        occurrence=occurrence,
        count=count,
        expected_fraction=count.float() / torch.tensor([8.0, 7.0]),
    )
    result.validate()
    return result


def _probabilities(count: torch.Tensor, support: int, *, good: bool) -> torch.Tensor:
    rows = count.numel()
    values = torch.full(
        (rows, support), (0.05 if good else 1.0) / support, dtype=torch.float64
    )
    if good:
        values[torch.arange(rows), count.clamp_min(1) - 1] = 0.95
        values /= values.sum(dim=1, keepdim=True)
    return values.contiguous()


def test_development_data_enforces_pair_and_group_contract(
    development_data: go.DevelopmentData,
) -> None:
    assert development_data.rows == 4320
    invalid_layer = development_data.candidate_layer.clone()
    invalid_layer[1] = 11
    with pytest.raises(go.GripperV2OOFError, match="pair order"):
        go.DevelopmentData(
            **{
                **development_data.__dict__,
                "candidate_layer": invalid_layer,
            }
        ).validate()


def test_one_outer_fold_has_exact_nested_fit_and_cell_counts(
    development_data: go.DevelopmentData,
) -> None:
    fold = go.fit_outer_fold(development_data, 12, max_iterations=2)
    assert fold["fit_count"] == 260
    assert fold["outer_episode"] == 12
    assert len(fold["inner_episodes"]) == 17
    assert set(fold["selected_lambda"]) == set(go.HEAD_NAMES)
    assert fold["predictions"]["row_index"].numel() == 240
    for summary in fold["one_standard_error"].values():
        assert all(item["cells"] == 170 for item in summary.values())


def test_metrics_and_strict_gates_pass_oracle_predictions(
    development_data: go.DevelopmentData,
) -> None:
    data = development_data
    occurrence_probability = torch.where(
        data.occurrence, torch.tensor(0.9), torch.tensor(0.1)
    ).double()
    oof = {
        "occurrence_probability": occurrence_probability.contiguous(),
        "occurrence_baseline": torch.full(
            (data.rows, 2), 0.5, dtype=torch.float64
        ),
        "zt_step_probability": _probabilities(data.count[:, 0], 8, good=False),
        "zt_transition_probability": _probabilities(
            data.count[:, 1], 7, good=False
        ),
        "ordinal_step_probability": _probabilities(
            data.count[:, 0], 8, good=True
        ),
        "ordinal_transition_probability": _probabilities(
            data.count[:, 1], 7, good=True
        ),
        "expected_fraction": data.expected_fraction.double().contiguous(),
        "expected_fraction_baseline": torch.full(
            (data.rows, 2), 0.5, dtype=torch.float64
        ),
        "assignment_count": torch.ones(data.rows, dtype=torch.long),
    }
    result = go.evaluate_oof(data, oof)
    assert result["gates"]["full_pass"] is True
    assert result["gates"]["focused_pass_non_deployable"] is True
    assert result["group_robustness"]["improved_episodes"] == 18
    assert result["group_robustness"]["exact_sign_test_upper_tail"] == pytest.approx(
        1 / 2**18
    )


def test_final_lambda_uses_stronger_regularization_to_break_mode_tie() -> None:
    assert go.final_lambda([0.001] * 6 + [0.01] * 6 + [0.1] * 6) == 0.1
    with pytest.raises(go.GripperV2OOFError, match="votes"):
        go.final_lambda([0.1] * 17)


def test_sign_test_gate_boundary_is_exact() -> None:
    assert go.exact_sign_test_upper_tail(13, 18) == pytest.approx(0.048126220703125)
    assert go.exact_sign_test_upper_tail(12, 18) > 0.05
