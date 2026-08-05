"""Learnable EFA-Lite warmup module over cached A1 projected features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_aggregation import (
    AggregatedVision,
    StaticVisionAggregationConfig,
    aggregate_projected_vision,
)


@dataclass(frozen=True)
class LearnableVisionAggregationConfig:
    hidden_dim: int = 3584
    attention_dim: int = 128
    output_tokens: int = 144
    num_heads: int = 8
    max_crops: int = 5
    max_patches_per_crop: int = 144
    min_tokens_per_crop: int = 4
    proprio_dim: int = 8
    action_horizon: int = 8
    action_dim: int = 7
    dropout: float = 0.0
    residual_gate_init: float = 0.01

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.attention_dim < 1:
            raise ValueError("hidden dimensions must be positive")
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if self.output_tokens < 1:
            raise ValueError("output_tokens must be positive")
        if self.max_crops < 1 or self.max_patches_per_crop < 1:
            raise ValueError("crop geometry must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must be in [0, 1)")


@dataclass
class LearnableAggregationOutput:
    aggregated: AggregatedVision
    anchor_features: torch.Tensor
    attention_weights: torch.Tensor
    importance_scores: torch.Tensor
    residual_scale: torch.Tensor
    latent_tokens: torch.Tensor


@dataclass
class EFAWarmupOutput:
    aggregation: LearnableAggregationOutput
    predicted_action: torch.Tensor
    predicted_crop_means: torch.Tensor
    valid_crop_mask: torch.Tensor


def _atanh(value: float) -> float:
    if value == 0.0:
        return 0.0
    return float(torch.atanh(torch.tensor(value, dtype=torch.float64)).item())


@torch.no_grad()
def reparameterize_residual_scale(
    aggregator: "LearnableVisionAggregator",
    target_scale: float,
) -> dict[str, float]:
    """Raise the residual gate without changing the current EFA function.

    Warmup can drive the scalar residual gate close to zero, which also scales
    down action-distillation gradients into the output projection.  Rescaling
    that projection inversely while increasing the gate preserves the exact
    residual product at initialization but removes the gradient bottleneck.
    """

    if not 0.0 < target_scale < 1.0:
        raise ValueError("target residual scale must be in (0, 1)")
    current = float(torch.tanh(aggregator.residual_gate_logit.float()).item())
    if not current > 0.0:
        raise ValueError("function-preserving reparameterization requires a positive gate")
    ratio = current / target_scale
    aggregator.output_projection.weight.mul_(ratio)
    if aggregator.output_projection.bias is not None:
        aggregator.output_projection.bias.mul_(ratio)
    aggregator.residual_gate_logit.copy_(
        torch.tensor(
            _atanh(target_scale),
            dtype=aggregator.residual_gate_logit.dtype,
            device=aggregator.residual_gate_logit.device,
        )
    )
    actual = float(torch.tanh(aggregator.residual_gate_logit.float()).item())
    return {
        "original_scale": current,
        "target_scale": target_scale,
        "actual_scale": actual,
        "output_projection_rescale": ratio,
        "gradient_scale_gain": target_scale / current,
    }


class LearnableVisionAggregator(nn.Module):
    """Instruction-conditioned cross-attention residual around static pool144."""

    def __init__(self, config: LearnableVisionAggregationConfig):
        super().__init__()
        self.config = config
        dim = config.attention_dim
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.input_projection = nn.Linear(config.hidden_dim, dim)
        self.crop_embedding = nn.Embedding(config.max_crops, dim)
        self.patch_embedding = nn.Embedding(config.max_patches_per_crop, dim)
        self.instruction_norm = nn.LayerNorm(config.hidden_dim)
        self.instruction_film = nn.Linear(config.hidden_dim, 2 * dim)
        self.queries = nn.Parameter(torch.empty(config.output_tokens, dim))
        self.cross_attention = nn.MultiheadAttention(
            dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.latent_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, config.hidden_dim)
        self.importance_head = nn.Linear(dim, 1)
        self.residual_gate_logit = nn.Parameter(
            torch.tensor(_atanh(config.residual_gate_init), dtype=torch.float32)
        )
        nn.init.normal_(self.queries, std=0.02)

    def _validate_inputs(
        self,
        projected_features: torch.Tensor,
        image_input_idx: torch.Tensor,
        instruction_summary: torch.Tensor,
        keep_tokens: int,
    ) -> None:
        if projected_features.ndim != 4:
            raise ValueError("projected_features must have shape [B, C, M, D]")
        batch_size, crops, patches, hidden = projected_features.shape
        if hidden != self.config.hidden_dim:
            raise ValueError("projected feature hidden size does not match config")
        if crops > self.config.max_crops or patches > self.config.max_patches_per_crop:
            raise ValueError("projected feature crop geometry exceeds config")
        if image_input_idx.shape != (batch_size, crops, patches):
            raise ValueError("image_input_idx must align with projected_features")
        if instruction_summary.shape != (batch_size, hidden):
            raise ValueError("instruction_summary must have shape [B, D]")
        if not 0 < keep_tokens <= self.config.output_tokens:
            raise ValueError("keep_tokens must be in [1, output_tokens]")

    @staticmethod
    def _nested_selection(
        scores: torch.Tensor,
        crop_ids: torch.Tensor,
        keep_tokens: int,
        min_tokens_per_crop: int,
    ) -> torch.Tensor:
        protected = []
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        for crop_id in torch.unique(crop_ids, sorted=True):
            crop_indices = torch.nonzero(crop_ids == crop_id, as_tuple=False).flatten()
            count = min(min_tokens_per_crop, int(crop_indices.numel()))
            order = torch.argsort(scores[crop_indices], descending=True, stable=True)
            chosen = crop_indices[order[:count]]
            protected.extend(chosen.unbind())
            selected_mask[chosen] = True
        if len(protected) > keep_tokens:
            raise ValueError("keep_tokens cannot satisfy the anchor crop minimum")
        remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        if remaining.numel():
            order = torch.argsort(scores[remaining], descending=True, stable=True)
            protected.extend(remaining[order].unbind())
        return torch.stack(protected)[:keep_tokens]

    def forward(
        self,
        projected_features: torch.Tensor,
        image_input_idx: torch.Tensor,
        instruction_summary: torch.Tensor,
        *,
        keep_tokens: Optional[int] = None,
    ) -> LearnableAggregationOutput:
        keep_tokens = keep_tokens or self.config.output_tokens
        self._validate_inputs(
            projected_features,
            image_input_idx,
            instruction_summary,
            keep_tokens,
        )
        batch_size, crops, patches, _ = projected_features.shape
        valid = image_input_idx >= 0
        if (~valid.any(dim=(1, 2))).any():
            raise ValueError("every sample must have at least one valid projected token")
        if not torch.isfinite(projected_features[valid]).all():
            raise ValueError("projected_features contains a non-finite valid token")

        flat = projected_features.reshape(batch_size, crops * patches, -1)
        flat_valid = valid.reshape(batch_size, crops * patches)
        tokens = self.input_projection(self.input_norm(flat))
        crop_ids = torch.arange(crops, device=flat.device)[:, None].expand(
            crops, patches
        )
        patch_ids = torch.arange(patches, device=flat.device)[None, :].expand(
            crops, patches
        )
        tokens = tokens + self.crop_embedding(crop_ids.reshape(-1))[None]
        tokens = tokens + self.patch_embedding(patch_ids.reshape(-1))[None]
        gamma, beta = self.instruction_film(
            self.instruction_norm(instruction_summary)
        ).chunk(2, dim=-1)
        tokens = tokens * (1.0 + 0.1 * torch.tanh(gamma[:, None])) + beta[:, None]

        queries = self.queries[None].expand(batch_size, -1, -1)
        latent, attention = self.cross_attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=~flat_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        latent = self.latent_norm(latent)
        residual = self.output_projection(latent)
        importance = self.importance_head(latent).squeeze(-1)

        anchor_config = StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=self.config.output_tokens,
            bank_tokens=self.config.output_tokens,
            min_tokens_per_crop=self.config.min_tokens_per_crop,
            fail_open=False,
        )
        with torch.no_grad():
            anchor = aggregate_projected_vision(
                projected_features,
                image_input_idx,
                anchor_config,
            )
        if not torch.all(anchor.kept_counts == self.config.output_tokens):
            raise ValueError("the source cannot provide output_tokens anchor positions")
        scale = torch.tanh(self.residual_gate_logit)
        full_features = anchor.features + scale.to(residual.dtype) * residual

        if keep_tokens == self.config.output_tokens:
            output = AggregatedVision(
                features=full_features,
                sequence_positions=anchor.sequence_positions,
                valid_mask=anchor.valid_mask,
                crop_ids=anchor.crop_ids,
                source_counts=anchor.source_counts,
                original_counts=anchor.original_counts,
                bank_counts=anchor.bank_counts,
                kept_counts=anchor.kept_counts,
            )
        else:
            selected_features = []
            selected_positions = []
            selected_crop_ids = []
            for batch_index in range(batch_size):
                chosen = self._nested_selection(
                    importance[batch_index],
                    anchor.crop_ids[batch_index],
                    keep_tokens,
                    self.config.min_tokens_per_crop,
                )
                order = torch.argsort(
                    anchor.sequence_positions[batch_index, chosen], stable=True
                )
                chosen = chosen[order]
                selected_features.append(full_features[batch_index, chosen])
                selected_positions.append(anchor.sequence_positions[batch_index, chosen])
                selected_crop_ids.append(anchor.crop_ids[batch_index, chosen])
            output = AggregatedVision(
                features=torch.stack(selected_features),
                sequence_positions=torch.stack(selected_positions),
                valid_mask=torch.ones(
                    (batch_size, keep_tokens),
                    dtype=torch.bool,
                    device=flat.device,
                ),
                crop_ids=torch.stack(selected_crop_ids),
                source_counts=anchor.source_counts,
                original_counts=anchor.original_counts,
                bank_counts=anchor.bank_counts,
                kept_counts=torch.full(
                    (batch_size,),
                    keep_tokens,
                    dtype=torch.long,
                    device=flat.device,
                ),
            )
        return LearnableAggregationOutput(
            aggregated=output,
            anchor_features=anchor.features,
            attention_weights=attention,
            importance_scores=importance,
            residual_scale=scale,
            latent_tokens=latent,
        )


class LearnableEFAWarmupModel(nn.Module):
    """Auxiliary cache-only objectives used before frozen-A1 distillation."""

    def __init__(self, config: LearnableVisionAggregationConfig):
        super().__init__()
        self.config = config
        self.aggregator = LearnableVisionAggregator(config)
        dim = config.attention_dim
        self.proprio_projection = nn.Linear(config.proprio_dim, dim)
        self.instruction_projection = nn.Linear(config.hidden_dim, dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, config.action_horizon * config.action_dim),
        )
        self.crop_queries = nn.Parameter(torch.empty(config.max_crops, dim))
        self.crop_decoder = nn.MultiheadAttention(
            dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.crop_output = nn.Linear(dim, config.hidden_dim)
        nn.init.normal_(self.crop_queries, std=0.02)

    def forward(
        self,
        projected_features: torch.Tensor,
        image_input_idx: torch.Tensor,
        instruction_summary: torch.Tensor,
        normalized_proprio: torch.Tensor,
    ) -> EFAWarmupOutput:
        aggregation = self.aggregator(
            projected_features,
            image_input_idx,
            instruction_summary,
        )
        latent_mean = aggregation.latent_tokens.mean(dim=1)
        action_features = torch.cat(
            (
                latent_mean,
                self.instruction_projection(instruction_summary),
                self.proprio_projection(normalized_proprio),
            ),
            dim=-1,
        )
        predicted_action = self.action_head(action_features).reshape(
            projected_features.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )
        crop_queries = self.crop_queries[None].expand(
            projected_features.shape[0], -1, -1
        )
        crop_latent, _ = self.crop_decoder(
            crop_queries,
            aggregation.latent_tokens,
            aggregation.latent_tokens,
            need_weights=False,
        )
        predicted_crop_means = self.crop_output(crop_latent)
        valid_crop_mask = (image_input_idx >= 0).any(dim=-1)
        return EFAWarmupOutput(
            aggregation=aggregation,
            predicted_action=predicted_action,
            predicted_crop_means=predicted_crop_means,
            valid_crop_mask=valid_crop_mask,
        )


def efa_warmup_loss(
    output: EFAWarmupOutput,
    projected_features: torch.Tensor,
    image_input_idx: torch.Tensor,
    teacher_action: torch.Tensor,
    *,
    action_weight: float = 1.0,
    crop_weight: float = 0.5,
    anchor_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute auxiliary action, all-view coverage and anchor-preservation loss."""

    action_loss = F.smooth_l1_loss(output.predicted_action, teacher_action)
    valid = image_input_idx >= 0
    weights = valid.unsqueeze(-1).to(projected_features.dtype)
    crop_counts = valid.sum(dim=-1).clamp_min(1).unsqueeze(-1)
    crop_means = (projected_features * weights).sum(dim=-2) / crop_counts
    crop_mask = output.valid_crop_mask
    predicted = output.predicted_crop_means[crop_mask].float()
    target = crop_means[crop_mask].float()
    crop_cosine = (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()
    target_scale = target.square().mean(dim=-1).sqrt().clamp_min(1e-6)
    crop_mse = (
        (predicted - target).square().mean(dim=-1) / target_scale.square()
    ).mean()
    crop_loss = crop_cosine + 0.1 * crop_mse
    anchor = output.aggregation.anchor_features.float()
    aggregated = output.aggregation.aggregated.features.float()
    anchor_loss = (1.0 - F.cosine_similarity(aggregated, anchor, dim=-1)).mean()
    total = (
        action_weight * action_loss
        + crop_weight * crop_loss
        + anchor_weight * anchor_loss
    )
    return total, {
        "action": action_loss.detach(),
        "crop": crop_loss.detach(),
        "anchor": anchor_loss.detach(),
        "total": total.detach(),
    }
