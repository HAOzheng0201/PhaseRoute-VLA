import json

import numpy as np
import torch

from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.phase_observer import SafePhaseObserver


def _checkpoint(path):
    torch.manual_seed(3)
    config = PhaseEstimatorConfig(
        visual_summary_dim=4,
        instruction_dim=5,
        proprio_dim=3,
        action_horizon=2,
        action_dim=2,
        visual_proj_dim=4,
        instruction_proj_dim=4,
        proprio_proj_dim=3,
        action_proj_dim=3,
        gru_hidden_dim=5,
        stage_dim=4,
    )
    model = PhaseStateEstimator(config)
    torch.save(
        {
            "schema_version": "phase-route-vla.phase-estimator-checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "model_config": dict(config.__dict__),
            "dataset_sha256": "a" * 64,
            "validation_boundary_threshold": 0.6,
        },
        path,
    )


def _call(observer, step_id, action_value):
    return observer.log_call(
        context={"episode_id": "episode-0", "step_id": step_id, "task_id": 0},
        instruction="move the object",
        raw_proprio=np.zeros(3, dtype=np.float32),
        normalized_proprio=np.full(3, step_id / 10, dtype=np.float32),
        previous_action=None,
        normalized_action_chunk=np.full((2, 2), action_value, dtype=np.float32),
        action_chunk=np.full((2, 2), action_value, dtype=np.float32),
        visual_summary=np.ones(4, dtype=np.float32),
        instruction_summary=np.ones(5, dtype=np.float32),
        visual_token_count=2,
        instruction_token_count=3,
    )


def test_observer_is_causal_and_records_history_without_controlling_exit(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint_path)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = SafePhaseObserver(checkpoint_path, first_path, device="cpu", history_len=2)
    second = SafePhaseObserver(checkpoint_path, second_path, device="cpu", history_len=2)
    try:
        assert _call(first, 0, 1.0)
        assert _call(second, 0, 999.0)
        assert _call(first, 1, 2.0)
    finally:
        first.close()
        second.close()

    first_records = [json.loads(line) for line in first_path.read_text().splitlines()]
    second_record = json.loads(second_path.read_text().splitlines()[0])
    assert first_records[0]["progress"] == second_record["progress"]
    assert first_records[0]["boundary_prob"] == second_record["boundary_prob"]
    assert [record["history_count"] for record in first_records] == [0, 1]
    assert all(record["observer_only"] for record in first_records)
    assert not any(record["controls_early_exit"] for record in first_records)
    assert first.error_count == 0


def test_observer_contains_invalid_input_errors(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint_path)
    observer = SafePhaseObserver(
        checkpoint_path,
        tmp_path / "observer.jsonl",
        device="cpu",
    )
    try:
        ok = observer.log_call(
            context={"episode_id": "episode", "step_id": 0},
            instruction="task",
            raw_proprio=np.zeros(3),
            normalized_proprio=np.zeros(99),
            previous_action=None,
            normalized_action_chunk=np.zeros((2, 2)),
            action_chunk=np.zeros((2, 2)),
            visual_summary=np.zeros(4),
            instruction_summary=np.zeros(5),
            visual_token_count=2,
            instruction_token_count=2,
        )
    finally:
        observer.close()
    assert not ok
    assert observer.error_count == 1
    assert "invalid shape" in observer.last_error
