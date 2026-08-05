from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from a1.vla.dynamic_compute.learnable_vision_aggregation import (
    LearnableVisionAggregationConfig,
    LearnableVisionAggregator,
)
from a1.vla.dynamic_compute.learnable_vision_runtime import (
    DISTILLED_EFA_CHECKPOINT_SCHEMA_VERSION,
    load_distilled_vision_aggregator,
)


def _checkpoint(path: Path, *, schema=DISTILLED_EFA_CHECKPOINT_SCHEMA_VERSION):
    config = LearnableVisionAggregationConfig(
        hidden_dim=16,
        attention_dim=8,
        output_tokens=4,
        num_heads=2,
        max_crops=2,
        max_patches_per_crop=4,
        min_tokens_per_crop=1,
    )
    model = LearnableVisionAggregator(config)
    torch.save(
        {
            "schema_version": schema,
            "aggregator_config": asdict(config),
            "aggregator_state_dict": model.state_dict(),
            "teacher_checkpoint_sha256": "a" * 64,
        },
        path,
    )
    return model


def test_runtime_loads_only_distilled_aggregator(tmp_path: Path):
    expected = _checkpoint(tmp_path / "efa.pt")
    loaded = load_distilled_vision_aggregator(
        tmp_path / "efa.pt",
        device="cpu",
        expected_hidden_dim=16,
        expected_teacher_checkpoint_sha256="a" * 64,
    )

    assert loaded.config.output_tokens == 4
    assert loaded.teacher_checkpoint_sha256 == "a" * 64
    assert loaded.model.training is False
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value)


def test_runtime_rejects_warmup_or_wrong_teacher(tmp_path: Path):
    _checkpoint(tmp_path / "wrong-schema.pt", schema="warmup")
    with pytest.raises(ValueError, match="schema"):
        load_distilled_vision_aggregator(
            tmp_path / "wrong-schema.pt", device="cpu"
        )

    _checkpoint(tmp_path / "wrong-teacher.pt")
    with pytest.raises(ValueError, match="fingerprints"):
        load_distilled_vision_aggregator(
            tmp_path / "wrong-teacher.pt",
            device="cpu",
            expected_teacher_checkpoint_sha256="b" * 64,
        )
