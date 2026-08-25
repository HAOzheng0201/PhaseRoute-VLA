from __future__ import annotations

import numpy as np
import pytest
import torch

from a1.vla.dynamic_compute.route_first_collection import RouteFirstTeacherCollector
from a1.vla.dynamic_compute.route_first_dataset import (
    ROUTE_FIRST_AGGREGATE_SCHEMA_VERSION,
    RouteFirstDatasetError,
    aggregate_route_first_teacher_shards,
    load_route_first_teacher_shard,
    save_route_first_teacher_aggregate,
)


def _context(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "instruction_summary": torch.randn(1, 3584, generator=generator),
        "vision_crop_summary": torch.randn(1, 5, 3584, generator=generator),
        "vision_crop_mask": torch.ones(1, 5, dtype=torch.bool),
        "phase_embedding": torch.randn(1, 128, generator=generator),
        "phase_scalars": torch.rand(1, 3, generator=generator),
        "normalized_proprio": torch.randn(1, 8, generator=generator),
        "proprio_history": torch.zeros(1, 8, 8),
        "action_history": torch.zeros(1, 8, 8, 7),
        "history_mask": torch.zeros(1, 8, dtype=torch.bool),
    }


class _Adapter:
    def begin_policy_call(self, runtime_inputs):
        return None


class _Runtime:
    def __init__(self) -> None:
        self.adapter = _Adapter()
        self._current = {"context": {}}

    def record_route_event(self, event_name, payload):
        return None


def _write_shard(path, task: int, episode: int, layers: tuple[int, ...]):
    runtime = _Runtime()
    collector = RouteFirstTeacherCollector(runtime)
    collector.install()
    runtime.adapter.begin_policy_call(None)
    for ordinal, layer in enumerate(layers):
        runtime._current["context"] = {
            "episode_id": f"libero_10:task{task}:episode{episode}",
            "task_id": task,
            "step_id": 10 + ordinal * 8,
            "call_ordinal": ordinal,
        }
        runtime.adapter.begin_policy_call(_context(task * 100 + episode * 10 + ordinal))
        runtime.record_route_event(
            "phase_route_decision",
            {"selected_layer": layer, "fallback": layer == 27},
        )
    collector.uninstall()
    return collector.save(path)


def test_load_route_first_teacher_shard_checks_identity_and_hash(tmp_path):
    path = tmp_path / "task2_state3.npz"
    published = _write_shard(path, 2, 3, (11, 13, 27))

    shard = load_route_first_teacher_shard(path)

    assert shard.rows == 3
    assert shard.file_sha256 == published["file_sha256"]
    assert shard.payload_sha256 == published["payload_sha256"]
    assert shard.task_id.tolist() == [2, 2, 2]
    assert shard.episode_index.tolist() == [3, 3, 3]
    assert shard.call_ordinal.tolist() == [0, 1, 2]


def test_aggregate_route_first_teacher_shards_enforces_exact_grid(tmp_path):
    task1 = tmp_path / "task1.npz"
    task0 = tmp_path / "task0.npz"
    _write_shard(task1, 1, 0, (27,))
    _write_shard(task0, 0, 0, (13, 27))

    arrays, summary = aggregate_route_first_teacher_shards(
        [task1, task0], expected_task_ids=(0, 1), expected_episode_indices=(0,)
    )

    assert arrays["task_id"].tolist() == [0, 0, 1]
    assert arrays["call_ordinal"].tolist() == [0, 1, 0]
    assert summary["rows"] == 3
    assert summary["episodes"] == 2
    assert summary["teacher_layer_counts"] == {"11": 0, "13": 1, "27": 2}

    output = tmp_path / "aggregate" / "teacher.npz"
    result = save_route_first_teacher_aggregate(output, arrays, summary)
    assert result["path"] == str(output.resolve())
    with np.load(output, allow_pickle=False) as saved:
        assert saved["schema_version"].item() == ROUTE_FIRST_AGGREGATE_SCHEMA_VERSION
        assert saved["features"].shape == (3, 199)
        assert saved["episode_index"].tolist() == [0, 0, 0]


def test_aggregate_route_first_teacher_shards_rejects_missing_episode(tmp_path):
    path = tmp_path / "task0.npz"
    _write_shard(path, 0, 0, (27,))

    with pytest.raises(RouteFirstDatasetError, match="episode grid differs"):
        aggregate_route_first_teacher_shards(
            [path], expected_task_ids=(0, 1), expected_episode_indices=(0,)
        )
