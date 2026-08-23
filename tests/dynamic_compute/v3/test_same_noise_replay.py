from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from a1.vla.dynamic_compute.v3.same_noise_replay import (
    D9C_COLLECTION_SHA256,
    D9D_ACTION_THRESHOLD,
    D9D_EXPECTED_ROWS,
    D9D_REPLAY_LAYERS,
    D9D_SEVERE_RATIO,
    D9DReplayError,
    build_call_truth,
    hash_online_action,
    validate_d9c_collection,
    validate_gpu_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _actions() -> torch.Tensor:
    actions = torch.zeros(3, 8, 7, dtype=torch.float32)
    actions[:, :, 0] = 1.0
    actions[0, :, 0] = 0.0
    actions[0, :, 1] = 1.0
    actions[0, 3, 6] = -1.0
    actions[1, :, 0] = 1.0
    actions[2, :, 0] = 1.0
    actions[2, :, 6] = 1.0
    return actions


def test_build_call_truth_uses_selected_layer_and_l27_only() -> None:
    actions = _actions()
    online = actions[0].clone()
    truth = build_call_truth(
        actions, selected_layer=11, online_selected_action=online
    )
    assert truth.selected_candidate_index == 0
    assert truth.selected_replay_max_abs_error == 0.0
    assert truth.selected_replay_bit_exact
    expected = float(
        (
            1.0
            - torch.nn.functional.cosine_similarity(
                actions[0].double(), actions[2].double(), dim=-1, eps=1.0e-8
            )
        )
        .mean()
        .item()
    )
    assert math.isclose(truth.full_action_distance, expected, abs_tol=1e-12)
    assert truth.full_action_unsafe
    assert truth.gripper_unsafe
    assert truth.severe_full_action


def test_l27_truth_is_safe_and_nonsevere() -> None:
    actions = _actions()
    truth = build_call_truth(
        actions, selected_layer=27, online_selected_action=actions[2].clone()
    )
    assert truth.selected_candidate_index == 2
    assert math.isclose(truth.full_action_distance, 0.0, abs_tol=1e-12)
    assert not truth.full_action_unsafe
    assert not truth.gripper_unsafe
    assert not truth.severe_full_action


def test_truth_uses_actual_online_selected_action_not_quantized_replay() -> None:
    actions = _actions()
    online = actions[2].clone()
    truth = build_call_truth(
        actions, selected_layer=11, online_selected_action=online
    )
    assert truth.selected_replay_max_abs_error > 0.0
    assert not truth.selected_replay_bit_exact
    assert math.isclose(truth.full_action_distance, 0.0, abs_tol=1e-12)
    assert not truth.full_action_unsafe
    assert not truth.gripper_unsafe
    assert not truth.severe_full_action


def test_thresholds_are_strict_greater_than() -> None:
    assert D9D_ACTION_THRESHOLD == 0.00390625
    assert D9D_SEVERE_RATIO == 4.0
    assert D9D_REPLAY_LAYERS == (11, 13, 27)


def test_truth_rejects_wrong_geometry_and_nonfinite() -> None:
    actions = _actions()
    with pytest.raises(D9DReplayError):
        build_call_truth(
            actions[:, :, :6], selected_layer=11, online_selected_action=actions[0]
        )
    actions[0, 0, 0] = float("nan")
    with pytest.raises(D9DReplayError):
        build_call_truth(
            actions, selected_layer=11, online_selected_action=torch.zeros(8, 7)
        )


def test_online_action_hash_has_d9c_domain_and_float32_normalization() -> None:
    action = np.arange(56, dtype=np.float32).reshape(8, 7)
    assert hash_online_action(action) == hash_online_action(torch.from_numpy(action))
    assert hash_online_action(action) == hash_online_action(action.astype(np.float64))


def test_front_four_gpu_contract() -> None:
    validate_gpu_contract(
        shard_index=2,
        physical_gpu_index=2,
        visible_devices="2",
        visible_gpu_count=1,
        expected_gpu_uuid="GPU-abc",
        observed_gpu_uuid="abc",
    )
    with pytest.raises(D9DReplayError):
        validate_gpu_contract(
            shard_index=2,
            physical_gpu_index=6,
            visible_devices="6",
            visible_gpu_count=1,
            expected_gpu_uuid="GPU-abc",
            observed_gpu_uuid="abc",
        )


def test_frozen_d9c_collection_authorizes_replay_only() -> None:
    result = validate_d9c_collection(REPO_ROOT)
    assert result["sha256"] == D9C_COLLECTION_SHA256
    assert result["cache_rows"] == D9D_EXPECTED_ROWS
    assert len(result["arm_payload_binding"]) == 100
