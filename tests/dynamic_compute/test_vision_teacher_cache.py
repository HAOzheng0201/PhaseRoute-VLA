import json
from pathlib import Path

import numpy as np
import torch

from a1.vla.dynamic_compute.vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    NullVisionTeacherCacheWriter,
    SafeVisionTeacherCacheWriter,
    emit_flow_matching_trace,
    emit_vision_teacher_features,
    has_complete_candidate_fm_traces,
)


def _writer_kwargs():
    features = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    return {
        "context": {"episode_id": "suite:task0:episode0", "step_id": 10, "task_id": 0},
        "instruction": "pick up the bowl",
        "projected_features": features,
        "image_input_idx": np.array([[2, 3, 4], [3, 4, -1]], dtype=np.int32),
        "instruction_summary": np.arange(4, dtype=np.float32),
        "normalized_proprio": np.arange(8, dtype=np.float32),
        "teacher_normalized_action": np.zeros((8, 7), dtype=np.float32),
        "teacher_action": np.ones((8, 7), dtype=np.float32),
        "cpu_rng_state": np.arange(16, dtype=np.uint8),
        "cuda_rng_state": np.arange(32, dtype=np.uint8),
        "teacher_exit_layer": 13,
        "fm_calls": 7,
        "fm_steps_total": 70,
        "input_ids": np.arange(9, dtype=np.int64),
        "attention_mask": np.ones(9, dtype=bool),
        "attention_bias": np.empty((0,), dtype=np.float32),
        "response_mask": np.zeros(9, dtype=bool),
        "subsegment_ids": np.empty((0,), dtype=np.int64),
        "position_ids": np.arange(9, dtype=np.int64),
        "action_proprio": np.arange(8, dtype=np.float32)[None],
        "proprio_token_idx": np.array([7], dtype=np.int64),
        "teacher_exit_input_x": np.full((8, 7), 0.5, dtype=np.float32),
        "teacher_exit_trace_action": np.zeros((8, 7), dtype=np.float32),
        "fm_trace_count": 2,
        "fm_trace_layers": np.array([12, 13], dtype=np.int16),
        "fm_trace_roles": np.array([0, 1], dtype=np.uint8),
        "fm_trace_steps": np.array([10, 10], dtype=np.int16),
        "fm_trace_input_x": np.stack(
            (
                np.full((8, 7), -0.5, dtype=np.float32),
                np.full((8, 7), 0.5, dtype=np.float32),
            )
        ),
        "fm_trace_output_action": np.zeros((2, 8, 7), dtype=np.float32),
    }


def test_feature_callback_failure_is_contained():
    def broken_callback(payload):
        del payload
        raise RuntimeError("collector failed")

    assert emit_vision_teacher_features(broken_callback, {"x": torch.ones(1)}) is False


def test_fm_trace_callback_failure_is_contained():
    def broken_callback(payload):
        del payload
        raise RuntimeError("collector failed")

    assert emit_flow_matching_trace(broken_callback, {"input_x": torch.ones(1)}) is False


def test_null_writer_has_no_filesystem_side_effect(tmp_path: Path):
    writer = NullVisionTeacherCacheWriter()
    assert writer.log_call(output_dir=tmp_path) is False
    writer.close()
    assert list(tmp_path.iterdir()) == []


def test_writer_persists_aligned_projected_features_and_metadata(tmp_path: Path):
    writer = SafeVisionTeacherCacheWriter(
        tmp_path / "cache",
        feature_dtype="float16",
        checkpoint_sha256="a" * 64,
    )
    accepted = writer.log_call(**_writer_kwargs())
    writer.close()

    assert accepted is True
    assert writer.error_count == 0
    metadata = json.loads((tmp_path / "cache" / "manifest.jsonl").read_text())
    assert metadata["schema_version"] == VISION_TEACHER_CACHE_SCHEMA_VERSION
    assert metadata["source_projected_tokens"] == 5
    assert metadata["unique_visual_slots"] == 3
    assert metadata["valid_crop_count"] == 2
    assert metadata["teacher_exit_layer"] == 13
    with np.load(tmp_path / "cache" / metadata["array_path"]) as shard:
        assert shard["projected_features"].dtype == np.float16
        assert shard["projected_features"].shape == (2, 3, 4)
        np.testing.assert_array_equal(shard["teacher_action"], np.ones((8, 7)))
        np.testing.assert_array_equal(
            shard["cuda_rng_state"], np.arange(32, dtype=np.uint8)
        )
        assert shard["input_ids"].shape == (9,)
        assert shard["teacher_exit_input_x"].shape == (8, 7)
    assert metadata["sequence_length"] == 9
    assert metadata["fm_trace_count"] == 2
    assert metadata["candidate_trace_count"] == 1
    assert metadata["comparison_trace_count"] == 1
    assert metadata["candidate_layers"] == [13]
    assert metadata["teacher_trace_max_abs_error"] == 0.0


def test_writer_rejects_shape_mismatch_without_partial_manifest(tmp_path: Path):
    writer = SafeVisionTeacherCacheWriter(tmp_path / "cache")
    kwargs = _writer_kwargs()
    kwargs["image_input_idx"] = np.zeros((2, 2), dtype=np.int32)

    assert writer.log_call(**kwargs) is False
    assert writer.error_count == 1
    assert "image_input_idx" in writer.last_error
    assert not writer.manifest_path.exists()


def test_writer_refuses_to_overwrite_existing_shard(tmp_path: Path):
    output_dir = tmp_path / "cache"
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    np.savez(arrays_dir / "call_000000.npz", sentinel=np.ones(1))
    writer = SafeVisionTeacherCacheWriter(output_dir)

    assert writer.log_call(**_writer_kwargs()) is False
    assert writer.error_count == 1
    assert "refusing to overwrite" in writer.last_error


def test_writer_rejects_misaligned_exit_trace(tmp_path: Path):
    writer = SafeVisionTeacherCacheWriter(tmp_path / "cache")
    kwargs = _writer_kwargs()
    kwargs["teacher_exit_trace_action"] = np.ones((8, 7), dtype=np.float32)

    assert writer.log_call(**kwargs) is False
    assert "does not align" in writer.last_error


def test_writer_accepts_collator_time_and_sample_proprio_axes(tmp_path: Path):
    writer = SafeVisionTeacherCacheWriter(tmp_path / "cache")
    kwargs = _writer_kwargs()
    kwargs["action_proprio"] = np.arange(8, dtype=np.float32).reshape(1, 1, 8)

    assert writer.log_call(**kwargs) is True
    assert writer.error_count == 0


def test_writer_persists_multiple_increasing_candidate_traces(tmp_path: Path):
    writer = SafeVisionTeacherCacheWriter(tmp_path / "cache")
    kwargs = _writer_kwargs()
    kwargs.update(
        teacher_exit_layer=15,
        fm_trace_count=4,
        fm_trace_layers=np.array([10, 11, 13, 15], dtype=np.int16),
        fm_trace_roles=np.array([0, 1, 1, 1], dtype=np.uint8),
        fm_trace_steps=np.full(4, 10, dtype=np.int16),
        fm_trace_input_x=np.stack(
            (
                np.zeros((8, 7), dtype=np.float32),
                np.ones((8, 7), dtype=np.float32),
                np.full((8, 7), 2.0, dtype=np.float32),
                np.full((8, 7), 0.5, dtype=np.float32),
            )
        ),
        fm_trace_output_action=np.zeros((4, 8, 7), dtype=np.float32),
    )

    assert writer.log_call(**kwargs) is True
    metadata = json.loads(writer.manifest_path.read_text())
    assert metadata["candidate_layers"] == [11, 13, 15]
    assert metadata["candidate_trace_count"] == 3


def test_writer_rejects_duplicate_candidate_layers(tmp_path: Path):
    writer = SafeVisionTeacherCacheWriter(tmp_path / "cache")
    kwargs = _writer_kwargs()
    kwargs["fm_trace_layers"] = np.array([13, 13], dtype=np.int16)
    kwargs["fm_trace_roles"] = np.array([1, 1], dtype=np.uint8)

    assert writer.log_call(**kwargs) is False
    assert "unique" in writer.last_error


def _complete_trace_record(*, terminal_accounting: bool):
    candidate_count = 14
    trace_count = 15
    return {
        "fm_calls": trace_count if terminal_accounting else candidate_count,
        "fm_trace_count": trace_count,
        "candidate_trace_count": candidate_count,
        "comparison_trace_count": 1,
        "candidate_layers": list(range(1, 28, 2)),
        "teacher_exit_layer": 27,
        "shapes": {
            "fm_trace_layers": [trace_count],
            "fm_trace_roles": [trace_count],
            "fm_trace_steps": [trace_count],
            "fm_trace_input_x": [trace_count, 8, 7],
            "fm_trace_output_action": [trace_count, 8, 7],
        },
    }


def test_complete_trace_accepts_both_accounting_conventions():
    assert has_complete_candidate_fm_traces(
        _complete_trace_record(terminal_accounting=False)
    )
    assert has_complete_candidate_fm_traces(
        _complete_trace_record(terminal_accounting=True)
    )


def test_complete_trace_rejects_unaccounted_or_wrong_terminal_trace():
    record = _complete_trace_record(terminal_accounting=True)
    record["comparison_trace_count"] = 0
    assert not has_complete_candidate_fm_traces(record)
    record = _complete_trace_record(terminal_accounting=True)
    record["candidate_layers"][-1] = 25
    assert not has_complete_candidate_fm_traces(record)
