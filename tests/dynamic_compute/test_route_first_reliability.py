from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from a1.vla.dynamic_compute.route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
)
from a1.vla.dynamic_compute import route_first_reliability as reliability


REPO_ROOT = Path(__file__).resolve().parents[2]


def _runtime_inputs(rows: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    mask = torch.tensor([[False] * 8, [False, True, True, True, False, False, False, False]])
    if rows != 2:
        mask = torch.zeros(rows, 8, dtype=torch.bool)
    return {
        "instruction_summary": torch.randn(rows, 3584, generator=generator),
        "vision_crop_summary": torch.randn(rows, 5, 3584, generator=generator),
        "vision_crop_mask": torch.tensor([[True, True, True, True, False]] * rows),
        "phase_embedding": torch.randn(rows, 128, generator=generator),
        "phase_scalars": torch.rand(rows, 3, generator=generator),
        "normalized_proprio": torch.randn(rows, 8, generator=generator),
        "proprio_history": torch.randn(rows, 8, 8, generator=generator),
        "action_history": torch.randn(rows, 8, 8, 7, generator=generator),
        "history_mask": mask,
    }


def test_protocol_and_schedule_are_frozen_and_disjoint() -> None:
    result = reliability.validate_stage11d_protocol(REPO_ROOT)
    schedule = reliability.build_stage11d_schedule()
    assert result["clusters"] == 200
    assert result["GPU_collection_authorized"] is False
    assert len(schedule) == 200
    assert len({record.cluster_key for record in schedule}) == 200
    assert not ({record.state_seed for record in schedule} & {record.policy_seed for record in schedule})
    assert sum(record.split == "development_train" for record in schedule) == 120
    assert sum(record.split == "calibration" for record in schedule) == 40
    assert sum(record.split == "shadow_confirmation" for record in schedule) == 40


def test_protocol_explicitly_forbids_current_experiment_execution() -> None:
    path = REPO_ROOT / reliability.STAGE11D_PROTOCOL_RELATIVE_PATH
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["authorization"] == {
        "current_stage": [
            "protocol_validation",
            "CPU_synthetic_target_tests",
            "runner_implementation_without_environment_execution",
        ],
        "new_state_generation_now": False,
        "GPU_collection_now": False,
        "training_now": False,
        "active_environment_control_now": False,
        "next_authorization_after_clean_tests_and_readiness": (
            "generate_200_new_states_then_collect_original_A1_observation_only_data"
        ),
    }


def test_identical_actions_are_safe_and_gripper_xor_is_unsafe() -> None:
    actions = torch.ones(2, 2, 8, 7, dtype=torch.float32)
    actions[1, 0, 3, 6] = -1.0
    targets = reliability.build_l13_reliability_targets(actions)
    assert targets.safe13.tolist() == [True, False]
    assert targets.gripper_step_unsafe.tolist() == [False, True]
    assert targets.joint_unsafe.tolist() == [False, True]
    assert targets.full_action_distance[0].item() == pytest.approx(0.0, abs=1e-12)


def test_cosine_target_matches_horizon_mean_and_strict_threshold() -> None:
    actions = torch.zeros(1, 2, 8, 7, dtype=torch.float64)
    actions[:, :, :, 0] = 1.0
    angle = torch.tensor(0.2, dtype=torch.float64)
    actions[:, 0, :, 0] = torch.cos(angle)
    actions[:, 0, :, 1] = torch.sin(angle)
    targets = reliability.build_l13_reliability_targets(actions)
    expected = 1.0 - torch.cos(angle).item()
    assert targets.full_action_distance.item() == pytest.approx(expected)
    assert targets.full_action_unsafe.item() is (expected > 0.00390625)


def test_reliability_batch_features_are_action_free() -> None:
    runtime = _runtime_inputs()
    safe_actions = torch.ones(2, 2, 8, 7)
    changed_actions = safe_actions.clone()
    changed_actions[:, 0, :, 0] = -1.0
    first = reliability.build_stage11d_reliability_batch(runtime, safe_actions)
    second = reliability.build_stage11d_reliability_batch(runtime, changed_actions)
    assert first.features.shape == (2, ROUTE_FIRST_FEATURE_DIMENSION)
    assert torch.equal(first.features, second.features)
    assert first.targets.safe13.all()
    assert not second.targets.safe13.any()


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(2, 3, 8, 7),
        torch.zeros(2, 2, 7, 7),
        torch.zeros(2, 2, 8, 6),
        torch.zeros(0, 2, 8, 7),
        torch.zeros(2, 2, 8, 7, dtype=torch.int64),
    ],
)
def test_target_builder_rejects_geometry_drift(value: torch.Tensor) -> None:
    with pytest.raises(reliability.Stage11DReliabilityError):
        reliability.build_l13_reliability_targets(value)


def test_target_builder_rejects_nonfinite_values() -> None:
    actions = torch.zeros(1, 2, 8, 7)
    actions[0, 0, 0, 0] = torch.nan
    with pytest.raises(reliability.Stage11DReliabilityError):
        reliability.build_l13_reliability_targets(actions)
