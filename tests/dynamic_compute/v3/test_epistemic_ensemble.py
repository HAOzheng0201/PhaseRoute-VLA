from __future__ import annotations

from pathlib import Path

import pytest
import torch

from a1.vla.dynamic_compute.v3 import epistemic_ensemble as ee


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_frozen_contract_counts_and_claim_boundary() -> None:
    contract = ee.load_d7_contract(REPO_ROOT)
    assert contract["epistemic_ensemble"]["head_count"] == 5
    assert contract["epistemic_ensemble"]["trainable_feature_parameter_count"] == 970
    assert contract["nested_oof"]["fits_per_outer"] == 260
    assert contract["nested_oof"]["total_model_fits"] == 4680
    assert contract["claim_boundary"]["D7_result_is_fresh_confirmation"] is False
    assert contract["authorization"]["independent_test_authorized"] is False


def test_delete_group_masks_exclude_base_validation_and_exactly_one_group() -> None:
    episode = torch.arange(12, 30, dtype=torch.long).repeat_interleave(4)
    base = episode != 12
    masks = ee.head_fit_masks(base, episode)
    assert len(masks) == 5
    assert torch.equal(masks[0], base)
    assert not any(bool(mask[episode == 12].any()) for mask in masks)
    included = torch.stack(masks).long().sum(dim=0)
    assert bool((included[base] == 4).all())
    assert bool((included[~base] == 0).all())
    group = ee.delete_group_index(episode)
    for head in range(1, 5):
        assert not bool(masks[head][base & (group == head - 1)].any())
        assert bool(masks[head][base & (group != head - 1)].all())


def test_ensemble_score_uses_full_max_but_head_zero_gripper() -> None:
    prediction = torch.tensor(
        [
            [[0.10, 0.03], [0.20, 0.04]],
            [[0.30, 0.70], [0.10, 0.80]],
            [[0.20, 0.60], [0.40, 0.90]],
            [[0.15, 0.50], [0.30, 0.60]],
            [[0.25, 0.40], [0.25, 0.50]],
        ],
        dtype=torch.float64,
    )
    full, gripper, head_range = ee.ensemble_scores(prediction)
    assert full.tolist() == pytest.approx([0.30, 0.40])
    assert gripper.tolist() == pytest.approx([0.03, 0.04])
    assert head_range.tolist() == pytest.approx([0.20, 0.30])


def test_delete_group_geometry_fails_closed() -> None:
    with pytest.raises(ee.D7ProtocolError, match="12--29"):
        ee.delete_group_index(torch.tensor([30], dtype=torch.long))
