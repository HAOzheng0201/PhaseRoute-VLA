from __future__ import annotations

import torch

from a1.vla.dynamic_compute.v3 import epistemic_ensemble_oof as eo
from a1.vla.dynamic_compute.v3.epistemic_ensemble import D7_FITS_PER_OUTER
from a1.vla.dynamic_compute.v3.joint_reliability import D5DevelopmentData


def synthetic_development_data() -> tuple[D5DevelopmentData, torch.Tensor]:
    identities = [
        (task, episode) for task in range(10) for episode in range(12, 30)
    ]
    calls = len(identities)
    rows = 2 * calls
    generator = torch.Generator().manual_seed(20260821)
    features = 0.01 * torch.randn((rows, 97), generator=generator)
    layer = torch.tensor([11, 13], dtype=torch.long).repeat(calls)
    source = torch.arange(calls).repeat_interleave(2)
    task = torch.tensor([value[0] for value in identities]).repeat_interleave(2)
    episode = torch.tensor([value[1] for value in identities]).repeat_interleave(2)
    unsafe = torch.zeros((rows, 2), dtype=torch.bool)
    unsafe[:, 0] = task == 0
    unsafe[:, 1] = (task == 1) & (((episode - 12) % 5) == 0)
    features[:, 0] += 3.0 * unsafe[:, 0].double()
    features[:, 1] += 3.0 * unsafe[:, 1].double()
    distance = torch.where(
        unsafe[:, 0],
        torch.full((rows,), 0.0625),
        torch.full((rows,), 0.001),
    )
    data = D5DevelopmentData(
        features=features,
        candidate_layer=layer,
        source_row=source,
        task_id=task,
        episode_index=episode,
        action_consistency=torch.ones(rows, dtype=torch.bool),
        unsafe_target=unsafe,
    )
    data.validate()
    return data, distance


def test_outer_fold_fits_five_head_nested_oof_without_outer_leakage() -> None:
    data, distance = synthetic_development_data()
    result = eo.fit_outer_fold(data, distance, 12, max_iterations=2)
    expected = torch.nonzero(data.episode_index == 12, as_tuple=False).flatten()
    assert result["schema_version"] == eo.D7_OOF_SCHEMA_VERSION
    assert result["fit_count"] == D7_FITS_PER_OUTER == 260
    assert torch.equal(result["validation_indices"], expected)
    assert result["validation_head_prediction"].shape == (5, 20, 2)
    assert result["validation_score"].shape == (20, 2)
    assert result["validation_full_head_range"].shape == (20,)
    assert result["selected_layer"].shape == (10,)
    assert len(result["outer_model_states"]) == 5
    assert result["outer_head_fit_rows"][0] == 340
    assert result["threshold"]["fixed_safety_multiplier"] == 0.95
    assert bool(torch.isfinite(result["selected_inner_full_head_range"]).all())


def test_threshold_selection_applies_fixed_shrink_and_remains_feasible() -> None:
    identities = [
        (task, episode) for task in range(10) for episode in range(13, 30)
    ]
    calls = len(identities)
    task = torch.tensor([item[0] for item in identities]).repeat_interleave(2)
    episode = torch.tensor([item[1] for item in identities]).repeat_interleave(2)
    full = torch.tensor(
        [
            0.01 + 0.001 * (item[1] - 13) + 1.0e-5 * item[0] + 0.005 * candidate
            for item in identities
            for candidate in range(2)
        ],
        dtype=torch.float64,
    )
    combined = torch.stack((full, torch.full_like(full, 0.01)), dim=1)
    selection = eo.select_d7_threshold(
        combined,
        torch.ones(2 * calls, dtype=torch.bool),
        torch.zeros((2 * calls, 2), dtype=torch.bool),
        task,
        episode,
    )
    assert selection.feasible is True
    assert selection.runtime_threshold == 0.95 * selection.full_threshold
    assert selection.runtime_summary is not None
    assert selection.runtime_summary.feasible is True
