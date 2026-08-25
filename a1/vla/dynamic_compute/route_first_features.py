"""Action-free context features for experimental route-first depth routing.

The frozen PhaseRoute-V3 router consumes a candidate action and therefore
cannot decide depth before the flow-matching action head runs.  This module
defines a separate, causal feature contract that uses only signals available
after the vision backbone and before decoder layer 0.  It intentionally does
not accept task/episode identities, future observations, or candidate actions.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch

from a1.vla.dynamic_compute.v3.development_collection import (
    validate_runtime_context,
)


ROUTE_FIRST_FEATURE_SCHEMA_VERSION = "phase-route-vla.route-first-context.v1"
ROUTE_FIRST_LAYERS = (11, 13, 27)

# Keep the group order frozen: checkpoints and collected arrays bind to it.
ROUTE_FIRST_FEATURE_GROUPS = OrderedDict(
    (
        ("phase_embedding", 128),
        ("phase_scalars", 3),
        ("normalized_proprio", 8),
        ("proprio_delta", 8),
        ("previous_first_action", 7),
        ("history_first_action_mean", 7),
        ("history_first_action_std", 7),
        ("history_scalars", 3),
        ("global_vision_stats", 4),
        ("instruction_stats", 4),
        ("per_crop_vision_stats", 15),
        ("vision_crop_mask", 5),
    )
)
ROUTE_FIRST_FEATURE_DIMENSION = sum(ROUTE_FIRST_FEATURE_GROUPS.values())


class RouteFirstFeatureError(ValueError):
    """Raised when an action-free runtime context violates the contract."""


def route_first_feature_slices() -> Mapping[str, slice]:
    """Return immutable-by-convention group slices for audit and ablation."""

    offset = 0
    groups: dict[str, slice] = {}
    for name, width in ROUTE_FIRST_FEATURE_GROUPS.items():
        groups[name] = slice(offset, offset + width)
        offset += width
    if offset != ROUTE_FIRST_FEATURE_DIMENSION:
        raise RuntimeError("route-first feature group widths are inconsistent")
    return groups


def _masked_latest(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = values.shape[0]
    indices = torch.arange(mask.shape[1], device=mask.device)[None].expand(rows, -1)
    latest = torch.where(mask, indices, torch.full_like(indices, -1)).amax(dim=1)
    gathered = values[torch.arange(rows, device=values.device), latest.clamp_min(0)]
    present = latest >= 0
    shape = (rows,) + (1,) * (gathered.ndim - 1)
    return torch.where(present.reshape(shape), gathered, torch.zeros_like(gathered))


def _summary_stats(values: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            values.mean(dim=1),
            values.std(dim=1, unbiased=False),
            values.square().mean(dim=1).sqrt(),
            values.abs().amax(dim=1),
        ),
        dim=1,
    )


def build_route_first_context_features(
    runtime_inputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Build the causal ``[B,199]`` feature used by route-first students.

    ``runtime_inputs`` is the same nine-tensor pre-action context prepared by
    the frozen online phase estimator.  Candidate actions and layer IDs are
    absent from both the function signature and the resulting feature.
    """

    if not isinstance(runtime_inputs, Mapping):
        raise RouteFirstFeatureError("runtime_inputs must be a mapping")
    instruction = runtime_inputs.get("instruction_summary")
    if not isinstance(instruction, torch.Tensor) or instruction.ndim != 2:
        raise RouteFirstFeatureError("instruction_summary must be a matrix")
    rows = int(instruction.shape[0])
    try:
        validate_runtime_context(runtime_inputs, rows=rows)
    except Exception as error:
        raise RouteFirstFeatureError(str(error)) from error

    phase_embedding = runtime_inputs["phase_embedding"]
    phase_scalars = runtime_inputs["phase_scalars"]
    proprio = runtime_inputs["normalized_proprio"]
    proprio_history = runtime_inputs["proprio_history"]
    action_history = runtime_inputs["action_history"]
    history_mask = runtime_inputs["history_mask"]
    vision = runtime_inputs["vision_crop_summary"]
    vision_mask = runtime_inputs["vision_crop_mask"]

    previous_proprio = _masked_latest(proprio_history, history_mask)
    previous_chunk = _masked_latest(action_history, history_mask)
    first_actions = action_history[:, :, 0, :]
    history_weight = history_mask.float()
    history_count = history_weight.sum(dim=1).clamp_min(1.0)
    history_mean = (
        first_actions * history_weight[:, :, None]
    ).sum(dim=1) / history_count[:, None]
    history_variance = (
        (first_actions - history_mean[:, None, :]).square()
        * history_weight[:, :, None]
    ).sum(dim=1) / history_count[:, None]
    history_std = history_variance.sqrt()
    adjacent_mask = history_mask[:, 1:] & history_mask[:, :-1]
    adjacent_delta = first_actions[:, 1:] - first_actions[:, :-1]
    adjacent_count = adjacent_mask.float().sum(dim=1).clamp_min(1.0)
    history_temporal_rms = (
        (
            adjacent_delta.square().sum(dim=2) * adjacent_mask.float()
        ).sum(dim=1)
        / (adjacent_count * first_actions.shape[-1])
    ).sqrt()
    history_scalars = torch.stack(
        (
            history_weight.mean(dim=1),
            previous_chunk.square().mean(dim=(1, 2)).sqrt(),
            history_temporal_rms,
        ),
        dim=1,
    )

    crop_weight = vision_mask.float()
    crop_count = crop_weight.sum(dim=1).clamp_min(1.0)
    pooled_vision = (
        vision * crop_weight[:, :, None]
    ).sum(dim=1) / crop_count[:, None]
    centered_vision = (
        vision - pooled_vision[:, None, :]
    ) * crop_weight[:, :, None]
    global_vision_stats = torch.stack(
        (
            pooled_vision.mean(dim=1),
            pooled_vision.std(dim=1, unbiased=False),
            pooled_vision.square().mean(dim=1).sqrt(),
            (
                centered_vision.square().sum(dim=(1, 2))
                / (crop_count * vision.shape[2])
            ).sqrt(),
        ),
        dim=1,
    )
    per_crop_vision_stats = torch.stack(
        (
            vision.mean(dim=2),
            vision.std(dim=2, unbiased=False),
            vision.square().mean(dim=2).sqrt(),
        ),
        dim=2,
    ) * crop_weight[:, :, None]
    per_crop_vision_stats = per_crop_vision_stats.reshape(rows, -1)

    groups = (
        phase_embedding,
        phase_scalars,
        proprio,
        proprio - previous_proprio,
        previous_chunk[:, 0, :],
        history_mean,
        history_std,
        history_scalars,
        global_vision_stats,
        _summary_stats(instruction),
        per_crop_vision_stats,
        crop_weight,
    )
    features = torch.cat(groups, dim=1).float().contiguous()
    if features.shape != (rows, ROUTE_FIRST_FEATURE_DIMENSION):
        raise RouteFirstFeatureError("route-first feature geometry differs")
    if not bool(torch.isfinite(features).all()):
        raise RouteFirstFeatureError("route-first features must be finite")
    return features


__all__ = [
    "ROUTE_FIRST_FEATURE_DIMENSION",
    "ROUTE_FIRST_FEATURE_GROUPS",
    "ROUTE_FIRST_FEATURE_SCHEMA_VERSION",
    "ROUTE_FIRST_LAYERS",
    "RouteFirstFeatureError",
    "build_route_first_context_features",
    "route_first_feature_slices",
]
