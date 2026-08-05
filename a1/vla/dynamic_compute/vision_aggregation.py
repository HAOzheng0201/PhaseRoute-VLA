"""Static visual-token aggregation and multimodal sequence compaction.

This module implements the training-free M4 baseline.  It operates on the
projected visual features (the same hidden size as the LLM), pools them into a
smaller crop-balanced token bank, and removes the unused visual positions from
every sequence-aligned field before the LLM blocks run.

The feature is deliberately opt-in.  Disabled and keep-all calls are handled
by the caller without modifying the original A1 tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StaticVisionAggregationConfig:
    """Configuration for the M4 training-free EFA-Lite baseline."""

    enabled: bool = False
    keep_tokens: int = 64
    bank_tokens: Optional[int] = None
    min_tokens_per_crop: int = 4
    fail_open: bool = True
    preserve_position_ids: bool = True

    def __post_init__(self) -> None:
        if self.keep_tokens < 1:
            raise ValueError("keep_tokens must be at least 1")
        if self.bank_tokens is not None and self.bank_tokens < self.keep_tokens:
            raise ValueError("bank_tokens must be greater than or equal to keep_tokens")
        if self.min_tokens_per_crop < 1:
            raise ValueError("min_tokens_per_crop must be at least 1")


@dataclass(frozen=True)
class AggregatedVision:
    """Projected visual tokens and their representative sequence positions."""

    features: torch.Tensor
    sequence_positions: torch.Tensor
    valid_mask: torch.Tensor
    crop_ids: torch.Tensor
    source_counts: torch.Tensor
    original_counts: torch.Tensor
    bank_counts: torch.Tensor
    kept_counts: torch.Tensor

    @property
    def compression_applied(self) -> bool:
        return bool(torch.any(self.kept_counts < self.original_counts).item())


@dataclass(frozen=True)
class CompactedSequence:
    """All sequence-aligned tensors after removing unused visual positions."""

    embeddings: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    attention_bias: Optional[torch.Tensor]
    response_mask: Optional[torch.Tensor]
    subsegment_ids: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    image_input_idx: torch.Tensor
    proprio_token_idx: Optional[torch.Tensor]
    append_last_valid_logits: Optional[torch.Tensor]
    old_to_new: torch.Tensor
    lengths: torch.Tensor


def _allocate_tokens(
    source_counts: list[int],
    target_tokens: int,
    min_tokens_per_crop: int,
) -> list[int]:
    """Allocate an exact token budget while protecting every valid crop."""

    if not source_counts or any(count < 1 for count in source_counts):
        raise ValueError("source_counts must contain positive integers")
    total = sum(source_counts)
    if target_tokens > total:
        target_tokens = total
    required = min_tokens_per_crop * len(source_counts)
    if target_tokens < required:
        raise ValueError(
            f"keep budget {target_tokens} is smaller than the per-crop minimum {required}"
        )
    if any(count < min_tokens_per_crop for count in source_counts):
        raise ValueError("a valid crop has fewer tokens than min_tokens_per_crop")

    allocation = [min_tokens_per_crop for _ in source_counts]
    remaining = target_tokens - sum(allocation)
    while remaining:
        capacities = [count - used for count, used in zip(source_counts, allocation)]
        capacity_total = sum(capacities)
        if capacity_total <= 0:
            raise RuntimeError("token allocation exhausted its source capacity")

        raw_additions = [remaining * capacity / capacity_total for capacity in capacities]
        additions = [
            min(capacity, int(math.floor(raw)))
            for capacity, raw in zip(capacities, raw_additions)
        ]
        added = sum(additions)
        for index, addition in enumerate(additions):
            allocation[index] += addition
        remaining -= added

        if remaining:
            order = sorted(
                range(len(capacities)),
                key=lambda index: (
                    raw_additions[index] - math.floor(raw_additions[index]),
                    source_counts[index] - allocation[index],
                    -index,
                ),
                reverse=True,
            )
            progressed = False
            for index in order:
                if remaining == 0:
                    break
                if allocation[index] < source_counts[index]:
                    allocation[index] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("could not finish deterministic token allocation")

    return allocation


def _factor_grid(token_count: int, side: int) -> Optional[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for h, w in ((height, width), (width, height)):
            if h <= side and w <= side:
                candidates.append((h, w))
    if not candidates:
        return None
    return min(candidates, key=lambda shape: (abs(shape[0] - shape[1]), -shape[0]))


def _pool_contiguous(
    features: torch.Tensor,
    positions: torch.Tensor,
    target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    boundaries = torch.linspace(
        0,
        features.shape[0],
        steps=target_tokens + 1,
        device=features.device,
        dtype=torch.float64,
    ).floor().to(torch.long)
    pooled: list[torch.Tensor] = []
    representatives: list[torch.Tensor] = []
    for index in range(target_tokens):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = start + 1
        pooled.append(features[start:end].mean(dim=0))
        representatives.append(positions[start + (end - start - 1) // 2])
    return torch.stack(pooled), torch.stack(representatives)


def _pool_crop(
    features: torch.Tensor,
    positions: torch.Tensor,
    target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatially pool one crop and select one source position per output bin."""

    source_tokens = features.shape[0]
    if target_tokens == source_tokens:
        return features, positions
    if not 0 < target_tokens < source_tokens:
        raise ValueError("target_tokens must be in [1, source_tokens]")

    side = math.isqrt(source_tokens)
    grid = _factor_grid(target_tokens, side) if side * side == source_tokens else None
    if grid is None:
        return _pool_contiguous(features, positions, target_tokens)

    output_height, output_width = grid
    feature_grid = features.reshape(side, side, features.shape[-1])
    position_grid = positions.reshape(side, side)
    if side % output_height == 0 and side % output_width == 0:
        pooled = feature_grid.reshape(
            output_height,
            side // output_height,
            output_width,
            side // output_width,
            features.shape[-1],
        ).mean(dim=(1, 3))
    else:
        pooled = F.adaptive_avg_pool2d(
            feature_grid.permute(2, 0, 1).unsqueeze(0),
            (output_height, output_width),
        )[0].permute(1, 2, 0)

    row_edges = torch.linspace(
        0,
        side,
        output_height + 1,
        dtype=torch.float64,
        device=positions.device,
    ).floor().long()
    col_edges = torch.linspace(
        0,
        side,
        output_width + 1,
        dtype=torch.float64,
        device=positions.device,
    ).floor().long()
    center_rows = row_edges[:-1] + (row_edges[1:] - row_edges[:-1] - 1) // 2
    center_cols = col_edges[:-1] + (col_edges[1:] - col_edges[:-1] - 1) // 2
    representative_positions = position_grid[
        center_rows[:, None], center_cols[None, :]
    ]
    return pooled.reshape(-1, features.shape[-1]), representative_positions.reshape(-1)


def _rank_single_token_bank(
    scores: torch.Tensor,
    crop_ids: torch.Tensor,
    min_tokens_per_crop: int,
) -> torch.Tensor:
    """Return one deterministic full ranking with crop-protected prefixes."""

    selected: list[torch.Tensor] = []
    selected_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    for crop_id in torch.unique(crop_ids, sorted=True):
        crop_indices = torch.nonzero(crop_ids == crop_id, as_tuple=False).flatten()
        if crop_indices.numel() < min_tokens_per_crop:
            raise ValueError("token bank cannot satisfy min_tokens_per_crop")
        local_order = torch.argsort(scores[crop_indices], descending=True, stable=True)
        protected = crop_indices[local_order[:min_tokens_per_crop]]
        selected.extend(protected.unbind())
        selected_mask[protected] = True

    remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
    if remaining.numel():
        remaining_order = torch.argsort(scores[remaining], descending=True, stable=True)
        selected.extend(remaining[remaining_order].unbind())
    return torch.stack(selected)


def rank_token_bank(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    crop_ids: torch.Tensor,
    min_tokens_per_crop: int,
) -> torch.Tensor:
    """Rank a padded token bank; valid prefixes are nested for every budget."""

    if features.ndim != 3:
        raise ValueError("features must have shape [B, M, D]")
    if valid_mask.shape != features.shape[:2] or crop_ids.shape != features.shape[:2]:
        raise ValueError("valid_mask and crop_ids must align with features")
    rankings = torch.full_like(crop_ids, -1)
    for batch_index in range(features.shape[0]):
        valid_indices = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("each sample must contain at least one valid token")
        valid_features = features[batch_index, valid_indices]
        if not torch.isfinite(valid_features).all():
            raise ValueError("visual token bank contains a non-finite value")
        scores = torch.linalg.vector_norm(valid_features.float(), dim=-1)
        local_ranking = _rank_single_token_bank(
            scores,
            crop_ids[batch_index, valid_indices],
            min_tokens_per_crop,
        )
        rankings[batch_index, :valid_indices.numel()] = valid_indices[local_ranking]
    return rankings


def aggregate_projected_vision(
    image_features: torch.Tensor,
    image_input_idx: torch.Tensor,
    config: StaticVisionAggregationConfig,
) -> AggregatedVision:
    """Build a crop-balanced projected-token bank and take a ranked prefix."""

    if not config.enabled:
        raise ValueError("aggregate_projected_vision requires an enabled config")
    if image_features.ndim != 4:
        raise ValueError("image_features must have shape [B, C, M, D]")
    if image_input_idx.shape != image_features.shape[:3]:
        raise ValueError("image_input_idx must have shape [B, C, M]")

    batch_size, crop_count, _, hidden_dim = image_features.shape
    sample_features: list[torch.Tensor] = []
    sample_positions: list[torch.Tensor] = []
    sample_crop_ids: list[torch.Tensor] = []
    source_counts_all: list[int] = []
    original_counts: list[int] = []
    bank_counts: list[int] = []
    kept_counts: list[int] = []

    for batch_index in range(batch_size):
        flat_features = image_features[batch_index].reshape(-1, hidden_dim)
        flat_positions = image_input_idx[batch_index].reshape(-1)
        flat_crop_ids = torch.arange(
            crop_count,
            dtype=torch.long,
            device=image_features.device,
        )[:, None].expand_as(image_input_idx[batch_index]).reshape(-1)
        valid = flat_positions >= 0
        flat_features = flat_features[valid]
        flat_positions = flat_positions[valid]
        flat_crop_ids = flat_crop_ids[valid]
        if flat_positions.numel() == 0:
            raise ValueError("each sample must contain at least one valid visual crop")
        if not torch.isfinite(flat_features).all():
            raise ValueError("projected visual features contain a non-finite value")

        # Molmo's overlapping crops can map several projected features to the
        # same LLM sequence slot.  A1 inserts them with advanced indexed
        # assignment, whose observable checkpoint behavior is last-write-wins.
        # Canonicalize to those effective unique slots before reducing width.
        source_count = int(flat_positions.numel())
        order = torch.argsort(flat_positions, stable=True)
        sorted_positions = flat_positions[order]
        last_for_position = torch.ones_like(sorted_positions, dtype=torch.bool)
        if sorted_positions.numel() > 1:
            last_for_position[:-1] = sorted_positions[:-1] != sorted_positions[1:]
        effective_indices = order[last_for_position]
        effective_features = flat_features[effective_indices]
        effective_positions = flat_positions[effective_indices]
        effective_crop_ids = flat_crop_ids[effective_indices]

        valid_crops: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for crop_id in torch.unique(effective_crop_ids, sorted=True):
            crop_mask = effective_crop_ids == crop_id
            valid_crops.append(
                (
                    int(crop_id.item()),
                    effective_features[crop_mask],
                    effective_positions[crop_mask],
                )
            )

        source_counts = [item[1].shape[0] for item in valid_crops]
        original_count = sum(source_counts)
        keep_count = min(config.keep_tokens, original_count)
        bank_count = min(config.bank_tokens or keep_count, original_count)

        if keep_count < config.min_tokens_per_crop * len(valid_crops):
            raise ValueError("keep_tokens cannot satisfy the per-crop minimum")
        bank_allocation = _allocate_tokens(
            source_counts,
            bank_count,
            config.min_tokens_per_crop,
        )

        bank_features: list[torch.Tensor] = []
        bank_positions: list[torch.Tensor] = []
        bank_crop_ids: list[torch.Tensor] = []
        for (crop_index, features, positions), crop_budget in zip(
            valid_crops, bank_allocation
        ):
            pooled_features, representative_positions = _pool_crop(
                features,
                positions,
                crop_budget,
            )
            bank_features.append(pooled_features)
            bank_positions.append(representative_positions)
            bank_crop_ids.append(
                torch.full(
                    (crop_budget,),
                    crop_index,
                    dtype=torch.long,
                    device=image_features.device,
                )
            )

        features = torch.cat(bank_features)
        positions = torch.cat(bank_positions)
        crop_ids = torch.cat(bank_crop_ids)
        if keep_count < bank_count:
            scores = torch.linalg.vector_norm(features.float(), dim=-1)
            ranking = _rank_single_token_bank(
                scores,
                crop_ids,
                config.min_tokens_per_crop,
            )
            chosen = ranking[:keep_count]
            features = features[chosen]
            positions = positions[chosen]
            crop_ids = crop_ids[chosen]

        sequence_order = torch.argsort(positions, stable=True)
        features = features[sequence_order]
        positions = positions[sequence_order]
        crop_ids = crop_ids[sequence_order]
        if torch.unique(positions).numel() != positions.numel():
            raise RuntimeError("aggregated visual tokens selected duplicate sequence positions")

        sample_features.append(features)
        sample_positions.append(positions)
        sample_crop_ids.append(crop_ids)
        source_counts_all.append(source_count)
        original_counts.append(original_count)
        bank_counts.append(bank_count)
        kept_counts.append(keep_count)

    max_kept = max(kept_counts)
    output_features = image_features.new_zeros((batch_size, max_kept, hidden_dim))
    output_positions = image_input_idx.new_full((batch_size, max_kept), -1)
    output_crop_ids = image_input_idx.new_full((batch_size, max_kept), -1)
    output_valid = torch.zeros(
        (batch_size, max_kept), dtype=torch.bool, device=image_features.device
    )
    for batch_index, count in enumerate(kept_counts):
        output_features[batch_index, :count] = sample_features[batch_index]
        output_positions[batch_index, :count] = sample_positions[batch_index]
        output_crop_ids[batch_index, :count] = sample_crop_ids[batch_index]
        output_valid[batch_index, :count] = True

    return AggregatedVision(
        features=output_features,
        sequence_positions=output_positions,
        valid_mask=output_valid,
        crop_ids=output_crop_ids,
        source_counts=torch.tensor(
            source_counts_all, dtype=torch.long, device=image_features.device
        ),
        original_counts=torch.tensor(
            original_counts, dtype=torch.long, device=image_features.device
        ),
        bank_counts=torch.tensor(
            bank_counts, dtype=torch.long, device=image_features.device
        ),
        kept_counts=torch.tensor(kept_counts, dtype=torch.long, device=image_features.device),
    )


def _validate_sequence_tensor(
    tensor: Optional[torch.Tensor],
    batch_size: int,
    sequence_length: int,
    name: str,
) -> None:
    if tensor is None:
        return
    if tensor.shape[0] != batch_size or tensor.shape[-1] != sequence_length:
        raise ValueError(f"{name} must align with the batch and sequence dimensions")


def _gather_last_dim(
    tensor: torch.Tensor,
    gather_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    fill_value: int | float | bool,
) -> torch.Tensor:
    index = gather_indices.clamp_min(0)
    gathered = torch.gather(tensor, -1, index)
    return torch.where(valid_mask, gathered, torch.as_tensor(fill_value, device=tensor.device))


def _compact_square_tensor(
    tensor: torch.Tensor,
    gather_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Gather both sequence axes of a 3-D/4-D attention tensor."""

    if tensor.shape[-2:] != (sequence_length, sequence_length):
        raise ValueError("square attention tensor does not match sequence length")
    original_ndim = tensor.ndim
    if original_ndim == 2:
        tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
    elif original_ndim in {3, 4}:
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, *tensor.shape[1:])
        elif tensor.shape[0] != batch_size:
            raise ValueError("attention tensor batch dimension cannot be broadcast")
    else:
        raise ValueError("attention tensor must have rank 2, 3, or 4")

    output_rows: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        indices = gather_indices[batch_index].clamp_min(0)
        sample = tensor[batch_index]
        sample = sample.index_select(-2, indices).index_select(-1, indices)
        square_valid = valid_mask[batch_index].unsqueeze(-1) & valid_mask[
            batch_index
        ].unsqueeze(-2)
        while square_valid.ndim < sample.ndim:
            square_valid = square_valid.unsqueeze(0)
        sample = torch.where(square_valid, sample, torch.zeros((), device=sample.device, dtype=sample.dtype))
        output_rows.append(sample)
    output = torch.stack(output_rows)
    if original_ndim == 2 and batch_size == 1:
        return output[0]
    return output


def _remap_positions(
    positions: Optional[torch.Tensor],
    old_to_new: torch.Tensor,
    name: str,
) -> Optional[torch.Tensor]:
    if positions is None:
        return None
    if positions.shape[0] != old_to_new.shape[0]:
        raise ValueError(f"{name} batch dimension does not match")
    flat = positions.reshape(positions.shape[0], -1)
    valid = flat >= 0
    if valid.any() and int(flat[valid].max().item()) >= old_to_new.shape[1]:
        raise ValueError(f"{name} contains an out-of-range position")
    mapped = torch.full(
        flat.shape,
        -1,
        dtype=torch.long,
        device=old_to_new.device,
    )
    safe = flat.clamp_min(0).to(dtype=torch.long)
    gathered = torch.gather(old_to_new, 1, safe)
    mapped[valid] = gathered[valid]
    if (mapped[valid] < 0).any():
        raise ValueError(f"{name} refers to a token removed during visual compaction")
    return mapped.reshape_as(positions)


def compact_multimodal_sequence(
    embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    original_image_positions: torch.Tensor,
    kept_image_positions: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    subsegment_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    proprio_token_idx: Optional[torch.Tensor] = None,
    append_last_valid_logits: Optional[torch.Tensor] = None,
    preserve_position_ids: bool = True,
) -> CompactedSequence:
    """Remove unselected visual positions from every sequence-aligned tensor."""

    if embeddings.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("embeddings and input_ids must have shapes [B, S, D] and [B, S]")
    batch_size, sequence_length, hidden_dim = embeddings.shape
    if input_ids.shape != (batch_size, sequence_length):
        raise ValueError("input_ids must align with embeddings")
    if original_image_positions.shape[0] != batch_size:
        raise ValueError("original_image_positions batch dimension does not match")
    if kept_image_positions.shape[0] != batch_size:
        raise ValueError("kept_image_positions batch dimension does not match")

    for tensor, name in (
        (response_mask, "response_mask"),
        (subsegment_ids, "subsegment_ids"),
        (position_ids, "position_ids"),
    ):
        _validate_sequence_tensor(tensor, batch_size, sequence_length, name)
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            _validate_sequence_tensor(
                attention_mask, batch_size, sequence_length, "attention_mask"
            )
        elif attention_mask.ndim == 3:
            if attention_mask.shape != (batch_size, sequence_length, sequence_length):
                raise ValueError("3-D attention_mask must have shape [B, S, S]")
        else:
            raise ValueError("attention_mask must have rank 2 or 3 before model conversion")

    keep_mask = torch.ones(
        (batch_size, sequence_length), dtype=torch.bool, device=embeddings.device
    )
    original_flat = original_image_positions.reshape(batch_size, -1)
    kept_flat = kept_image_positions.reshape(batch_size, -1)
    for batch_index in range(batch_size):
        original = original_flat[batch_index]
        original = original[original >= 0]
        kept = kept_flat[batch_index]
        kept = kept[kept >= 0]
        if original.numel() == 0:
            raise ValueError("each sample must contain original visual positions")
        if int(original.max().item()) >= sequence_length:
            raise ValueError("original_image_positions contains an out-of-range position")
        # Overlapping Molmo crops legitimately refer to the same LLM slot.
        # Sequence compaction removes slots, so operate on the unique set.
        original = torch.unique(original, sorted=True)
        if torch.unique(kept).numel() != kept.numel():
            raise ValueError("kept_image_positions contains duplicates")
        if kept.numel() and int(kept.max().item()) >= sequence_length:
            raise ValueError("kept_image_positions contains an out-of-range position")
        if kept.numel() and not torch.isin(kept, original).all():
            raise ValueError("kept_image_positions must be a subset of original positions")
        keep_mask[batch_index, original] = False
        keep_mask[batch_index, kept] = True

    lengths = keep_mask.sum(dim=1)
    compacted_length = int(lengths.max().item())
    gather_indices = torch.full(
        (batch_size, compacted_length),
        -1,
        dtype=torch.long,
        device=embeddings.device,
    )
    valid_mask = torch.zeros_like(gather_indices, dtype=torch.bool)
    old_to_new = torch.full(
        (batch_size, sequence_length),
        -1,
        dtype=torch.long,
        device=embeddings.device,
    )
    for batch_index in range(batch_size):
        indices = torch.nonzero(keep_mask[batch_index], as_tuple=False).flatten()
        count = indices.numel()
        gather_indices[batch_index, :count] = indices
        valid_mask[batch_index, :count] = True
        old_to_new[batch_index, indices] = torch.arange(count, device=embeddings.device)

    embedding_indices = gather_indices.clamp_min(0).unsqueeze(-1).expand(
        -1, -1, hidden_dim
    )
    compacted_embeddings = torch.gather(embeddings, 1, embedding_indices)
    compacted_embeddings = torch.where(
        valid_mask.unsqueeze(-1), compacted_embeddings, torch.zeros_like(compacted_embeddings)
    )
    compacted_input_ids = _gather_last_dim(
        input_ids,
        gather_indices,
        valid_mask,
        -1,
    )

    compacted_attention_mask: Optional[torch.Tensor]
    if attention_mask is None:
        compacted_attention_mask = None
    elif attention_mask.ndim == 2:
        compacted_attention_mask = _gather_last_dim(
            attention_mask,
            gather_indices,
            valid_mask,
            False if attention_mask.dtype == torch.bool else 0,
        )
    else:
        compacted_attention_mask = _compact_square_tensor(
            attention_mask,
            gather_indices,
            valid_mask,
            batch_size,
            sequence_length,
        )

    compacted_attention_bias = (
        None
        if attention_bias is None
        else _compact_square_tensor(
            attention_bias,
            gather_indices,
            valid_mask,
            batch_size,
            sequence_length,
        )
    )
    compacted_response_mask = (
        None
        if response_mask is None
        else _gather_last_dim(response_mask, gather_indices, valid_mask, 0)
    )
    compacted_subsegment_ids = (
        None
        if subsegment_ids is None
        else _gather_last_dim(subsegment_ids, gather_indices, valid_mask, 0)
    )
    compacted_position_ids = (
        None
        if position_ids is None
        else _gather_last_dim(position_ids, gather_indices, valid_mask, 0)
    )
    if compacted_position_ids is not None and not preserve_position_ids:
        valid_tokens = compacted_input_ids != -1
        compacted_position_ids = torch.clamp(
            torch.cumsum(valid_tokens.to(torch.long), dim=-1) - 1,
            min=0,
        )

    mapped_image_positions = _remap_positions(
        kept_image_positions, old_to_new, "kept_image_positions"
    )
    assert mapped_image_positions is not None
    mapped_proprio = _remap_positions(
        proprio_token_idx, old_to_new, "proprio_token_idx"
    )
    mapped_append = _remap_positions(
        append_last_valid_logits, old_to_new, "append_last_valid_logits"
    )

    return CompactedSequence(
        embeddings=compacted_embeddings,
        input_ids=compacted_input_ids,
        attention_mask=compacted_attention_mask,
        attention_bias=compacted_attention_bias,
        response_mask=compacted_response_mask,
        subsegment_ids=compacted_subsegment_ids,
        position_ids=compacted_position_ids,
        image_input_idx=mapped_image_positions,
        proprio_token_idx=mapped_proprio,
        append_last_valid_logits=mapped_append,
        old_to_new=old_to_new,
        lengths=lengths,
    )


__all__ = [
    "AggregatedVision",
    "CompactedSequence",
    "StaticVisionAggregationConfig",
    "aggregate_projected_vision",
    "compact_multimodal_sequence",
    "rank_token_bank",
]
