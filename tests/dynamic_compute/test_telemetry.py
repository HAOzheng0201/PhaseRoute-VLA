import json
from pathlib import Path

import pytest
import torch

from a1.vla.dynamic_compute.config import (
    DynamicComputeConfig,
    PhaseCacheConfig,
    TelemetryConfig,
)
from a1.vla.dynamic_compute.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    DynamicComputeTelemetry,
    NullTelemetryLogger,
    SafeJSONLTelemetryLogger,
    build_policy_call_telemetry,
    instruction_hash,
)


def test_all_dynamic_compute_features_are_disabled_by_default():
    config = DynamicComputeConfig()

    assert config.enabled is False
    assert config.phase_enabled is False
    assert config.vision_aggregation_enabled is False
    assert config.joint_budget_enabled is False
    assert config.reliable_exit_enabled is False
    assert config.lfp_enabled is False
    assert config.dynamic_fm_steps_enabled is False
    assert config.telemetry.enabled is False
    assert config.phase_cache.enabled is False


def test_enabled_telemetry_requires_an_output_path():
    with pytest.raises(ValueError, match="output_path"):
        TelemetryConfig(enabled=True)


def test_enabled_phase_cache_requires_an_output_directory():
    with pytest.raises(ValueError, match="output_dir"):
        PhaseCacheConfig(enabled=True)


def test_instruction_hash_is_stable_and_does_not_store_instruction():
    instruction = "Put the red mug on the plate."
    digest = instruction_hash(instruction)

    assert digest == instruction_hash(instruction)
    assert len(digest) == 16
    assert instruction not in digest


def test_jsonl_logger_writes_strict_versioned_records(tmp_path: Path):
    output_path = tmp_path / "telemetry.jsonl"
    record = DynamicComputeTelemetry(
        episode_id="episode-1",
        step_id=8,
        task_id=3,
        instruction_hash=instruction_hash("pick up the bowl"),
        candidate_exit_layers=[1, 3, 5, 27],
        action_delta_by_exit=[0.2, 0.1],
        exit_layer=5,
        fm_calls=3,
        fm_steps_total=30,
        latency_ms=12.5,
        action_shape=[1, 8, 7],
        action_dtype="torch.bfloat16",
        extra={"nonfinite_is_sanitized": float("nan")},
    )

    with SafeJSONLTelemetryLogger(output_path, flush_every=1) as logger:
        assert logger.log(record) is True
        assert logger.records_written == 1
        assert logger.error_count == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == TELEMETRY_SCHEMA_VERSION
    assert payload["episode_id"] == "episode-1"
    assert payload["candidate_exit_layers"] == [1, 3, 5, 27]
    assert payload["action_shape"] == [1, 8, 7]
    assert payload["extra"]["nonfinite_is_sanitized"] is None


def test_disabled_logger_has_no_filesystem_side_effect(tmp_path: Path):
    output_path = tmp_path / "must_not_exist.jsonl"
    logger = NullTelemetryLogger()

    assert logger.log(DynamicComputeTelemetry(step_id=0)) is False
    logger.close()
    assert not output_path.exists()


def test_logging_failure_is_contained(tmp_path: Path):
    # Opening a directory as a file fails, but the logger must not raise.
    logger = SafeJSONLTelemetryLogger(tmp_path)

    assert logger.log(DynamicComputeTelemetry(step_id=0)) is False
    assert logger.error_count == 1
    assert "IsADirectoryError" in logger.last_error


def test_side_channel_does_not_modify_action_tensor(tmp_path: Path):
    action = torch.randn(1, 8, 7)
    baseline = action.clone()

    with SafeJSONLTelemetryLogger(tmp_path / "telemetry.jsonl", flush_every=1) as logger:
        accepted = logger.log(
            DynamicComputeTelemetry(
                action_shape=list(action.shape),
                action_dtype=str(action.dtype),
            )
        )

    assert accepted is True
    torch.testing.assert_close(action, baseline, rtol=0, atol=0)


def test_full_tensor_is_rejected_without_gpu_copy(tmp_path: Path):
    logger = SafeJSONLTelemetryLogger(tmp_path / "telemetry.jsonl")
    record = DynamicComputeTelemetry(extra={"raw_tensor": torch.ones(2)})

    assert logger.log(record) is False
    assert "precomputed summary" in logger.last_error
    assert not (tmp_path / "telemetry.jsonl").exists()


def test_policy_call_builder_aligns_exit_metrics_and_context():
    record = build_policy_call_telemetry(
        context={
            "episode_id": "suite:task2:episode4",
            "step_id": 24,
            "task_id": 2,
            "previous_action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.2, 1.0],
        },
        instruction="put the bowl on the plate",
        raw_proprio=[0.0, 0.1, 0.2, 0.0, 0.0, 0.0, -0.03, 0.03],
        active_token_count=648,
        n_layers=28,
        visual_token_count=576,
        candidate_exit_layers=[1, 3, 5],
        telemetry_events=[
            {
                "event": "exit_candidate",
                "layer_idx": 1,
                "evaluated": True,
                "should_exit": False,
                "action_delta": 0.4,
                "fm_calls": 1,
                "fm_steps": 10,
            },
            {
                "event": "exit_candidate",
                "layer_idx": 3,
                "evaluated": True,
                "should_exit": True,
                "action_delta": 0.2,
                "fm_calls": 1,
                "fm_steps": 10,
            },
        ],
        latency_ms=123.0,
        action_shape=[1, 8, 7],
        action_dtype="torch.bfloat16",
        normalization_key="libero_spatial_no_noops",
    )

    assert record.exit_layer == 3
    assert record.action_delta_by_exit == [0.4, 0.2, None]
    assert record.fm_calls == 2
    assert record.fm_steps_total == 20
    assert record.active_tokens_by_layer == [648] * 28
    assert record.extra["visual_tokens"] == 576
    assert record.gripper_state == pytest.approx(0.06)
    assert record.translation_speed == pytest.approx((0.1**2 + 0.2**2 + 0.3**2) ** 0.5)


def test_policy_call_uses_compacted_llm_length_from_vision_event():
    record = build_policy_call_telemetry(
        context={"episode_id": "episode"},
        instruction="move the bowl",
        raw_proprio=None,
        active_token_count=648,
        n_layers=3,
        visual_token_count=576,
        candidate_exit_layers=[],
        telemetry_events=[
            {
                "event": "vision_aggregation",
                "status": "compressed",
                "original_tokens": 576,
                "kept_tokens": 64,
                "original_active_tokens": 648,
                "active_tokens": 136,
                "original_llm_sequence_length": 680,
                "llm_sequence_length": 168,
            }
        ],
        latency_ms=10.0,
        action_shape=[1, 8, 7],
        action_dtype="torch.bfloat16",
        normalization_key="libero_spatial_no_noops",
    )

    assert record.active_tokens_by_layer == [136, 136, 136]
    assert record.extra["vision_aggregation"]["kept_tokens"] == 64
