from __future__ import annotations

import torch

from a1.vla.dynamic_compute.v3 import severity_reliability_oof as so
from a1.vla.dynamic_compute.v3.joint_reliability import D5DevelopmentData


def synthetic_development_data() -> tuple[D5DevelopmentData, torch.Tensor]:
    identities = [
        (task, episode, call)
        for task in range(10)
        for episode in range(12, 30)
        for call in range(2)
    ]
    calls = len(identities)
    rows = 2 * calls
    generator = torch.Generator().manual_seed(20260821)
    features = 0.03 * torch.randn((rows, 97), generator=generator)
    layer = torch.tensor([11, 13], dtype=torch.long).repeat(calls)
    source = torch.arange(calls).repeat_interleave(2)
    task = torch.tensor([value[0] for value in identities]).repeat_interleave(2)
    episode = torch.tensor([value[1] for value in identities]).repeat_interleave(2)
    consistency = torch.ones(rows, dtype=torch.bool)
    unsafe = torch.zeros((rows, 2), dtype=torch.bool)
    # Binary support in every fold without making the robust threshold unsafe.
    unsafe[:, 0] = (source % 61) == 0
    unsafe[:, 1] = (source % 73) == 0
    features[:, 0] = unsafe[:, 0].double() + 0.01 * features[:, 0]
    features[:, 1] = unsafe[:, 1].double() + 0.01 * features[:, 1]
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
        action_consistency=consistency,
        unsafe_target=unsafe,
    )
    data.validate()
    return data, distance


def test_fit_outer_fold_is_grouped_and_uses_robust_threshold() -> None:
    data, distance = synthetic_development_data()
    result = so.fit_outer_fold(data, distance, 12, max_iterations=5)
    expected = torch.nonzero(data.episode_index == 12, as_tuple=False).flatten()
    assert result["schema_version"] == so.D6_OOF_SCHEMA_VERSION
    assert result["fit_count"] == so.D6_FITS_PER_OUTER == 52
    assert torch.equal(result["validation_indices"], expected)
    assert result["validation_score"].shape == (expected.numel(), 2)
    assert result["selected_layer"].shape == (expected.numel() // 2,)
    robust = result["robust_threshold"]
    assert robust["feasible"] is True
    assert len(robust["jackknife_thresholds"]) == 17
    assert robust["runtime_threshold"] == 0.95 * robust["pre_shrink_threshold"]
    assert result["outer_model_state"]["weight"].shape == (2, 97)
