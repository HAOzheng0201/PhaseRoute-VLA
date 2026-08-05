import json
from pathlib import Path

import numpy as np
import torch

from a1.vla.dynamic_compute.phase_cache import (
    PHASE_CACHE_SCHEMA_VERSION,
    NullPhaseCacheWriter,
    SafePhaseCacheWriter,
    emit_phase_signal_summary,
    summarize_instruction_embeddings,
    summarize_visual_embeddings,
)


def test_instruction_summary_excludes_visual_response_proprio_and_padding():
    embeddings = torch.arange(1 * 7 * 2, dtype=torch.float32).reshape(1, 7, 2)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, -1]])
    image_input_idx = torch.tensor([[[1, 2]]])
    response_mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0]])
    proprio_token_idx = torch.tensor([[4]])

    summary, counts = summarize_instruction_embeddings(
        embeddings,
        input_ids,
        image_input_idx=image_input_idx,
        response_mask=response_mask,
        proprio_token_idx=proprio_token_idx,
    )

    # Only positions 0 and 3 remain.
    torch.testing.assert_close(summary, embeddings[:, [0, 3]].mean(dim=1))
    assert counts.tolist() == [2]


def test_visual_summary_ignores_padded_crop_features():
    features = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[100.0, 200.0], [300.0, 400.0]]]]
    )
    indices = torch.tensor([[[4, 5], [-1, -1]]])

    summary, counts = summarize_visual_embeddings(features, indices)

    torch.testing.assert_close(summary, torch.tensor([[2.0, 3.0]]))
    assert counts.tolist() == [2]


def test_phase_callback_failure_is_contained():
    def broken_callback(payload):
        del payload
        raise RuntimeError("collector failed")

    assert emit_phase_signal_summary(broken_callback, {"x": torch.ones(1)}) is False


def test_null_writer_has_no_filesystem_side_effect(tmp_path: Path):
    writer = NullPhaseCacheWriter()
    assert writer.log_call(output_dir=tmp_path) is False
    writer.close()
    assert list(tmp_path.iterdir()) == []


def test_phase_cache_writer_persists_versioned_aligned_shard(tmp_path: Path):
    writer = SafePhaseCacheWriter(tmp_path / "cache", summary_dtype="float16")
    accepted = writer.log_call(
        context={"episode_id": "suite:task0:episode0", "step_id": 10, "task_id": 0},
        instruction="pick up the bowl",
        raw_proprio=np.arange(8, dtype=np.float32),
        normalized_proprio=np.linspace(-1, 1, 8, dtype=np.float32),
        previous_action=None,
        normalized_action_chunk=np.zeros((8, 7), dtype=np.float32),
        action_chunk=np.ones((8, 7), dtype=np.float32),
        visual_summary=np.arange(6, dtype=np.float32),
        instruction_summary=np.arange(8, dtype=np.float32),
        visual_token_count=576,
        instruction_token_count=24,
    )
    writer.close()

    assert accepted is True
    assert writer.error_count == 0
    manifest = json.loads((tmp_path / "cache" / "manifest.jsonl").read_text())
    assert manifest["schema_version"] == PHASE_CACHE_SCHEMA_VERSION
    assert manifest["episode_id"] == "suite:task0:episode0"
    assert manifest["previous_action_present"] is False
    assert manifest["summary_counts"] == {
        "visual_tokens": 576,
        "instruction_tokens": 24,
    }
    shard = np.load(tmp_path / "cache" / manifest["array_path"])
    np.testing.assert_array_equal(shard["action_chunk"], np.ones((8, 7)))
    assert shard["visual_summary"].dtype == np.float16
    assert shard["instruction_summary"].dtype == np.float16


def test_writer_refuses_to_overwrite_existing_shard(tmp_path: Path):
    output_dir = tmp_path / "cache"
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    np.savez(arrays_dir / "call_000000.npz", sentinel=np.ones(1))
    writer = SafePhaseCacheWriter(output_dir)

    accepted = writer.log_call(
        context={"episode_id": "episode", "step_id": 0},
        instruction="instruction",
        raw_proprio=np.ones(8),
        normalized_proprio=np.ones(8),
        previous_action=None,
        normalized_action_chunk=np.ones((8, 7)),
        action_chunk=np.ones((8, 7)),
        visual_summary=np.ones(4),
        instruction_summary=np.ones(4),
        visual_token_count=1,
        instruction_token_count=1,
    )

    assert accepted is False
    assert writer.error_count == 1
    assert "refusing to overwrite" in writer.last_error
