from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from a1.vla.dynamic_compute.v3 import development_collection as dc  # noqa: E402


def _runtime(rows: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260820)
    values = {
        "instruction_summary": torch.randn(rows, 3584, generator=generator),
        "vision_crop_summary": torch.randn(rows, 5, 3584, generator=generator),
        "vision_crop_mask": torch.tensor([[True, True, True, True, False]] * rows),
        "phase_embedding": torch.randn(rows, 128, generator=generator),
        "phase_scalars": torch.rand(rows, 3, generator=generator),
        "normalized_proprio": torch.randn(rows, 8, generator=generator),
        "proprio_history": torch.randn(rows, 8, 8, generator=generator),
        "action_history": torch.randn(rows, 8, 8, 7, generator=generator),
        "history_mask": torch.tensor(
            [[False, False, False, True, True, True, True, True]] * rows
        ),
    }
    return values


def test_collection_contract_is_label_independent_and_exact() -> None:
    contract = dc.collection_contract()
    assert contract["status"] == dc.D2_STATUS
    assert contract["selection"]["episode_indices"] == list(range(12, 30))
    assert contract["replay"]["layers"] == [11, 13, 27]
    assert contract["features"]["output_dimension"] == 97
    assert contract["features"]["teacher_visible"] is False
    assert contract["access_boundary"] == {
        "development_v2_payload": True,
        "calibration_v2_payload": False,
        "independent_test_v2_payload": False,
        "legacy_c361_row_payload": False,
        "runtime_control": False,
    }
    assert len(dc.collection_contract_sha256()) == 64
    path = REPO_ROOT / "configs/research/v3/gripper_v2/d2_collection_contract.json"
    assert json.loads(path.read_text(encoding="utf-8")) == contract


def test_frozen_selection_and_seed_formula() -> None:
    audit = dc.validate_frozen_d2_inputs(REPO_ROOT)
    assert audit["development_keys"] == 180
    assert audit["d1_protocol_status"] == "D1_GRIPPER_V2_PROTOCOL_FROZEN"
    selection = dc.load_development_selection(REPO_ROOT)
    assert len(selection) == 180
    assert selection[0] == dc.DevelopmentEpisode(0, 12, 20260823)
    assert selection[-1] == dc.DevelopmentEpisode(9, 29, 20350840)
    for task in range(10):
        window = dc.task_development_window(selection, task)
        assert [record.episode_index for record in window] == list(range(12, 30))
        assert all(
            record.seed == dc.expected_seed(task, record.episode_index)
            for record in window
        )


def test_initial_state_window_and_global_identity() -> None:
    class Suite:
        n_tasks = 10

        def get_task_init_states(self, task_id: int):
            return np.arange(50 * 2).reshape(50, 2) + task_id * 1000

        def get_task(self, task_id: int):
            return f"task-{task_id}"

    window = dc.InitialStateWindowTaskSuite(Suite(), 12, 18)
    assert window.n_tasks == 10
    assert window.get_task(3) == "task-3"
    assert np.array_equal(
        window.get_task_init_states(0), np.arange(100).reshape(50, 2)[12:30]
    )
    assert [dc.global_episode_index(index) for index in range(18)] == list(
        range(12, 30)
    )
    with pytest.raises(dc.D2ContractError):
        dc.InitialStateWindowTaskSuite(Suite(), 0, 18)
    with pytest.raises(dc.D2ContractError):
        dc.global_episode_index(18)


def test_gpu_contract_rejects_back_four_and_uuid_mismatch() -> None:
    dc.validate_gpu_contract(
        physical_gpu_index=2,
        visible_devices="2",
        visible_gpu_count=1,
        expected_gpu_uuid="GPU-abc",
        observed_gpu_uuid="abc",
    )
    with pytest.raises(PermissionError, match="0--3"):
        dc.validate_gpu_contract(
            physical_gpu_index=4,
            visible_devices="4",
            visible_gpu_count=1,
            expected_gpu_uuid="abc",
            observed_gpu_uuid="abc",
        )
    with pytest.raises(PermissionError, match="UUID"):
        dc.validate_gpu_contract(
            physical_gpu_index=0,
            visible_devices="0",
            visible_gpu_count=1,
            expected_gpu_uuid="abc",
            observed_gpu_uuid="def",
        )


def test_past_only_history_resets_and_excludes_current() -> None:
    history = dc.PastOnlyHistory()
    first = dc.DevelopmentCall(0, 0, 12, 0, 10, 11, Path("."), "a.npz", 1)
    second = dc.DevelopmentCall(1, 0, 12, 1, 18, 13, Path("."), "b.npz", 2)
    other = dc.DevelopmentCall(2, 1, 12, 0, 10, 11, Path("."), "c.npz", 1)
    proprio = np.arange(8, dtype=np.float32)
    action = np.arange(56, dtype=np.float32).reshape(8, 7)
    first_window = history.window_then_commit(first, proprio, action)
    second_window = history.window_then_commit(second, proprio + 1, action + 1)
    other_window = history.window_then_commit(other, proprio + 2, action + 2)
    assert not first_window.history_mask.any()
    assert second_window.history_mask.tolist() == [False] * 7 + [True]
    assert np.array_equal(second_window.action_history[-1], action)
    assert not np.array_equal(second_window.action_history[-1], action + 1)
    assert not other_window.history_mask.any()


def test_visual_pool_geometry_and_mask() -> None:
    features = np.zeros((5, 144, 3584), dtype=np.float32)
    positions = np.full((5, 144), -1, dtype=np.int64)
    positions[:4] = np.arange(144)
    for crop in range(4):
        features[crop] = crop + 1
    global_summary, crop_summary, crop_mask = dc.pool_visual_features(
        features, positions
    )
    assert global_summary.shape == (3584,)
    assert crop_summary.shape == (5, 3584)
    assert crop_mask.tolist() == [True, True, True, True, False]
    assert torch.allclose(torch.from_numpy(global_summary), torch.full((3584,), 2.5))
    assert not crop_summary[4].any()


def test_replay_batch_has_exact_batch_one_geometry() -> None:
    arrays = {
        "projected_features": np.zeros((5, 144, 3584), dtype=np.float16),
        "image_input_idx": np.zeros((5, 144), dtype=np.int32),
        "instruction_summary": np.zeros((3584,), dtype=np.float16),
        "normalized_proprio": np.zeros((8,), dtype=np.float32),
        "input_ids": np.zeros((12,), dtype=np.int64),
        "attention_mask": np.ones((12,), dtype=np.bool_),
        "attention_bias": np.zeros((12, 12), dtype=np.float32),
        "response_mask": np.zeros((12,), dtype=np.bool_),
        "subsegment_ids": np.zeros((12,), dtype=np.int64),
        "position_ids": np.arange(12, dtype=np.int64),
        "action_proprio": np.zeros((8,), dtype=np.float32),
        "proprio_token_idx": np.array([3], dtype=np.int64),
        "teacher_exit_input_x": np.zeros((8, 7), dtype=np.float32),
        "teacher_normalized_action": np.zeros((8, 7), dtype=np.float32),
    }
    batch = dc.replay_batch(arrays)
    assert "teacher_normalized_action" not in batch
    assert batch["projected_features"].shape == (1, 5, 144, 3584)
    assert batch["teacher_exit_input_x"].shape == (1, 8, 7)
    arrays.pop("position_ids")
    with pytest.raises(dc.D2ArtifactError, match="position_ids"):
        dc.replay_batch(arrays)


def test_97d_features_are_current_candidate_only() -> None:
    runtime = _runtime()
    generator = torch.Generator().manual_seed(7)
    candidates = torch.randn(2, 2, 8, 7, generator=generator)
    features = dc.build_gripper_v2_features(runtime, candidates)
    assert features.shape == (2, 2, 97)
    expected_sign = torch.where(
        candidates[..., 6] >= 0,
        torch.ones_like(candidates[..., 6]),
        -torch.ones_like(candidates[..., 6]),
    )
    assert torch.equal(features[..., 82:90], expected_sign)
    expected_transition = (
        (candidates[..., 6] >= 0)[:, :, 1:]
        != (candidates[..., 6] >= 0)[:, :, :-1]
    ).float()
    assert torch.equal(features[..., 90:97], expected_transition)

    changed = candidates.clone()
    changed[:, 1] *= -1
    changed_features = dc.build_gripper_v2_features(runtime, changed)
    assert torch.equal(features[:, 0], changed_features[:, 0])
    assert not torch.equal(features[:, 1], changed_features[:, 1])


def test_discrete_target_truth_table_and_no_continuous_magnitude() -> None:
    actions = torch.zeros(2, 3, 8, 7, dtype=torch.float32)
    actions[..., 6] = -1
    actions[0, 0, 3, 6] = 1
    actions[0, 1, :, 6] = 1
    actions[1, 2, 4:, 6] = 1
    actions[1, 0, 5:, 6] = 1
    actions[1, 1, 2:, 6] = 1
    targets = dc.build_gripper_v2_targets(actions)
    assert targets.count[0, 0].tolist() == [1, 2]
    assert targets.first_transition_mismatch[0, 0].item() == 3
    assert targets.count[0, 1].tolist() == [8, 0]
    assert targets.occurrence[0, 1].tolist() == [True, False]
    assert targets.first_transition_mismatch[1].tolist() == [4, 2]
    assert all(
        tensor.dtype in (torch.bool, torch.int64)
        for tensor in targets.__dict__.values()
    )


def test_runtime_and_target_reject_leakage_geometry_and_nonfinite() -> None:
    runtime = _runtime(1)
    candidates = torch.zeros(1, 2, 8, 7)
    leaked = dict(runtime)
    leaked["layer27_candidate_action"] = torch.zeros(1, 8, 7)
    with pytest.raises(dc.D2ArtifactError, match="order or names"):
        dc.build_gripper_v2_features(leaked, candidates)
    with pytest.raises(dc.D2ArtifactError, match=r"\[B,3,8,7\]"):
        dc.build_gripper_v2_targets(candidates)
    actions = torch.zeros(1, 3, 8, 7)
    actions[0, 2, 0, 6] = float("nan")
    with pytest.raises(dc.D2ArtifactError, match="finite"):
        dc.build_gripper_v2_targets(actions)


def _manifest_row(task: int, episode: int, step: int, array_path: str) -> dict:
    shapes = {
        "fm_trace_layers": [3],
        "fm_trace_roles": [3],
        "fm_trace_steps": [3],
        "fm_trace_input_x": [3, 8, 7],
        "fm_trace_output_action": [3, 8, 7],
    }
    return {
        "schema_version": dc.VISION_TEACHER_CACHE_SCHEMA_VERSION,
        "episode_id": f"libero_10:task{task}:episode{episode}",
        "step_id": step,
        "task_id": task,
        "teacher_kind": "a1_early_exit",
        "checkpoint_sha256": dc.D2_CHECKPOINT_SHA256,
        "teacher_exit_layer": 27,
        "fm_calls": 3,
        "fm_trace_count": 3,
        "candidate_trace_count": 3,
        "comparison_trace_count": 0,
        "candidate_layers": [11, 13, 27],
        "shapes": shapes,
        "array_path": array_path,
    }


def test_manifest_accepts_only_development_libero_long(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher_calls"
    arrays = teacher / "arrays"
    arrays.mkdir(parents=True)
    rows = []
    for episode in range(12, 30):
        path = arrays / f"call_{episode:06d}.npz"
        np.savez(path, value=np.array([episode]))
        rows.append(_manifest_row(0, episode, 10, f"arrays/{path.name}"))
    (teacher / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    calls = dc.load_task_calls(tmp_path, task_id=0)
    assert len(calls) == 18
    assert [call.episode_index for call in calls] == list(range(12, 30))
    assert all(dc.resolve_call_payload(call).is_file() for call in calls)

    rows[0]["episode_id"] = "libero_spatial:task0:episode12"
    (teacher / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(dc.D2ArtifactError, match="canonical libero_10"):
        dc.load_task_calls(tmp_path, task_id=0)
