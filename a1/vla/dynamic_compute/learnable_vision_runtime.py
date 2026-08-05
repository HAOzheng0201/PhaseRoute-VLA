"""Inference loader for a frozen-A1-distilled learnable vision aggregator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .learnable_vision_aggregation import (
    LearnableVisionAggregationConfig,
    LearnableVisionAggregator,
)


DISTILLED_EFA_CHECKPOINT_SCHEMA_VERSION = (
    "phase-route-vla.frozen-a1-efa-distillation.v1"
)


@dataclass(frozen=True)
class LoadedLearnableVisionAggregator:
    model: LearnableVisionAggregator
    checkpoint_path: Path
    teacher_checkpoint_sha256: str
    config: LearnableVisionAggregationConfig


def load_distilled_vision_aggregator(
    checkpoint_path: Union[str, Path],
    *,
    device: Union[str, torch.device],
    expected_hidden_dim: Optional[int] = None,
    expected_teacher_checkpoint_sha256: Optional[str] = None,
) -> LoadedLearnableVisionAggregator:
    """Load only aggregator weights; never deserialize or duplicate frozen A1."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != DISTILLED_EFA_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unexpected distilled EFA checkpoint schema")
    teacher_hash = str(checkpoint.get("teacher_checkpoint_sha256", ""))
    if len(teacher_hash) != 64:
        raise ValueError("distilled EFA checkpoint has no valid teacher fingerprint")
    int(teacher_hash, 16)
    if (
        expected_teacher_checkpoint_sha256 is not None
        and teacher_hash != expected_teacher_checkpoint_sha256
    ):
        raise ValueError("distilled EFA and runtime A1 fingerprints differ")
    config = LearnableVisionAggregationConfig(**checkpoint["aggregator_config"])
    if expected_hidden_dim is not None and config.hidden_dim != expected_hidden_dim:
        raise ValueError("distilled EFA hidden size does not match runtime A1")
    model = LearnableVisionAggregator(config)
    model.load_state_dict(checkpoint["aggregator_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return LoadedLearnableVisionAggregator(
        model=model,
        checkpoint_path=path,
        teacher_checkpoint_sha256=teacher_hash,
        config=config,
    )
