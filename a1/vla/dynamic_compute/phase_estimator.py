"""Lightweight temporal PhaseStateEstimator for PhaseRoute-VLA M2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PhaseEstimatorConfig:
    visual_summary_dim: int = 3584
    instruction_dim: int = 3584
    proprio_dim: int = 8
    action_horizon: int = 8
    action_dim: int = 7
    visual_proj_dim: int = 256
    instruction_proj_dim: int = 256
    proprio_proj_dim: int = 128
    action_proj_dim: int = 128
    gru_hidden_dim: int = 256
    stage_dim: int = 128

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass
class PhaseState:
    stage_embedding: torch.Tensor
    progress: torch.Tensor
    boundary_prob: torch.Tensor
    uncertainty: torch.Tensor
    next_hidden: torch.Tensor


@dataclass(frozen=True)
class PhaseLossConfig:
    lambda_progress: float = 1.0
    lambda_boundary: float = 2.0
    lambda_order: float = 0.5
    order_margin: float = 0.02

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class PhaseStateEstimator(nn.Module):
    """Estimate progress and phase boundaries without controlling A1 in M2."""

    def __init__(self, config: PhaseEstimatorConfig | None = None):
        super().__init__()
        self.config = config or PhaseEstimatorConfig()
        cfg = self.config
        self.visual_projector = _mlp(
            cfg.visual_summary_dim,
            cfg.visual_proj_dim,
            cfg.visual_proj_dim,
        )
        self.instruction_projector = _mlp(
            cfg.instruction_dim,
            cfg.instruction_proj_dim,
            cfg.instruction_proj_dim,
        )
        self.proprio_projector = _mlp(
            cfg.proprio_dim,
            cfg.proprio_proj_dim,
            cfg.proprio_proj_dim,
        )
        self.history_proprio_projector = _mlp(
            cfg.proprio_dim,
            cfg.proprio_proj_dim,
            cfg.proprio_proj_dim,
        )
        self.action_projector = _mlp(
            cfg.action_horizon * cfg.action_dim,
            cfg.action_proj_dim,
            cfg.action_proj_dim,
        )
        temporal_input_dim = cfg.proprio_proj_dim + cfg.action_proj_dim
        self.temporal_fusion = _mlp(
            temporal_input_dim,
            cfg.gru_hidden_dim,
            cfg.gru_hidden_dim,
        )
        self.gru_cell = nn.GRUCell(cfg.gru_hidden_dim, cfg.gru_hidden_dim)
        fusion_dim = (
            cfg.visual_proj_dim
            + cfg.instruction_proj_dim
            + cfg.proprio_proj_dim
            + cfg.gru_hidden_dim
        )
        self.stage_projector = _mlp(fusion_dim, cfg.gru_hidden_dim, cfg.stage_dim)
        self.progress_head = nn.Linear(cfg.stage_dim, 1)
        self.boundary_head = nn.Linear(cfg.stage_dim, 1)

    @property
    def parameter_dtype(self) -> torch.dtype:
        return self.progress_head.weight.dtype

    def _validate_inputs(
        self,
        visual_summary: torch.Tensor,
        instruction_summary: torch.Tensor,
        current_proprio: torch.Tensor,
        proprio_history: torch.Tensor,
        proprio_history_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_history_mask: torch.Tensor,
        previous_hidden: Optional[torch.Tensor],
    ) -> tuple[int, int]:
        cfg = self.config
        if visual_summary.ndim != 2 or visual_summary.shape[1] != cfg.visual_summary_dim:
            raise ValueError("visual_summary must have shape [B, visual_summary_dim]")
        batch_size = visual_summary.shape[0]
        if instruction_summary.shape != (batch_size, cfg.instruction_dim):
            raise ValueError("instruction_summary has an invalid shape")
        if current_proprio.shape != (batch_size, cfg.proprio_dim):
            raise ValueError("current_proprio has an invalid shape")
        if proprio_history.ndim != 3 or proprio_history.shape[0] != batch_size:
            raise ValueError("proprio_history must have shape [B, H, proprio_dim]")
        history_len = proprio_history.shape[1]
        if proprio_history.shape[2] != cfg.proprio_dim:
            raise ValueError("proprio_history has an invalid proprio dimension")
        if proprio_history_mask.shape != (batch_size, history_len):
            raise ValueError("proprio_history_mask has an invalid shape")
        expected_action_shape = (
            batch_size,
            history_len,
            cfg.action_horizon,
            cfg.action_dim,
        )
        if action_history.shape != expected_action_shape:
            raise ValueError("action_history has an invalid shape")
        if action_history_mask.shape != (batch_size, history_len):
            raise ValueError("action_history_mask has an invalid shape")
        if not torch.equal(
            proprio_history_mask.to(torch.bool),
            action_history_mask.to(torch.bool),
        ):
            raise ValueError("proprio/action history masks must be aligned")
        if previous_hidden is not None and previous_hidden.shape != (
            1,
            batch_size,
            cfg.gru_hidden_dim,
        ):
            raise ValueError("previous_hidden must have shape [1, B, gru_hidden_dim]")
        return batch_size, history_len

    def forward(
        self,
        *,
        visual_summary: torch.Tensor,
        instruction_summary: torch.Tensor,
        current_proprio: torch.Tensor,
        proprio_history: torch.Tensor,
        proprio_history_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_history_mask: torch.Tensor,
        previous_hidden: Optional[torch.Tensor] = None,
    ) -> PhaseState:
        batch_size, history_len = self._validate_inputs(
            visual_summary,
            instruction_summary,
            current_proprio,
            proprio_history,
            proprio_history_mask,
            action_history,
            action_history_mask,
            previous_hidden,
        )
        dtype = self.parameter_dtype
        visual_summary = visual_summary.to(dtype=dtype)
        instruction_summary = instruction_summary.to(dtype=dtype)
        current_proprio = current_proprio.to(dtype=dtype)
        proprio_history = proprio_history.to(dtype=dtype)
        action_history = action_history.to(dtype=dtype)
        history_mask = action_history_mask.to(device=action_history.device, dtype=torch.bool)

        if previous_hidden is None:
            hidden = torch.zeros(
                batch_size,
                self.config.gru_hidden_dim,
                device=visual_summary.device,
                dtype=dtype,
            )
        else:
            hidden = previous_hidden[0].to(device=visual_summary.device, dtype=dtype)

        flat_actions = action_history.reshape(batch_size, history_len, -1)
        for index in range(history_len):
            history_proprio = self.history_proprio_projector(
                proprio_history[:, index]
            )
            history_action = self.action_projector(flat_actions[:, index])
            temporal_input = self.temporal_fusion(
                torch.cat([history_proprio, history_action], dim=-1)
            )
            candidate_hidden = self.gru_cell(temporal_input, hidden)
            step_mask = history_mask[:, index].unsqueeze(-1)
            hidden = torch.where(step_mask, candidate_hidden, hidden)

        fused = torch.cat(
            [
                self.visual_projector(visual_summary),
                self.instruction_projector(instruction_summary),
                self.proprio_projector(current_proprio),
                hidden,
            ],
            dim=-1,
        )
        stage_embedding = self.stage_projector(fused)
        progress = torch.sigmoid(self.progress_head(stage_embedding))
        boundary_prob = torch.sigmoid(self.boundary_head(stage_embedding))
        probability = boundary_prob.clamp(1e-6, 1.0 - 1e-6)
        uncertainty = -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log(1.0 - probability)
        ) / math.log(2.0)
        return PhaseState(
            stage_embedding=stage_embedding,
            progress=progress,
            boundary_prob=boundary_prob,
            uncertainty=uncertainty,
            next_hidden=hidden.unsqueeze(0),
        )


def phase_estimator_loss(
    phase_state: PhaseState,
    *,
    progress_target: torch.Tensor,
    boundary_target: torch.Tensor,
    episode_index: Optional[torch.Tensor] = None,
    call_index: Optional[torch.Tensor] = None,
    config: PhaseLossConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute progress, boundary and within-episode temporal-order losses."""

    config = config or PhaseLossConfig()
    if progress_target.shape != phase_state.progress.shape:
        raise ValueError("progress_target must match predicted progress")
    if boundary_target.shape != phase_state.boundary_prob.shape:
        raise ValueError("boundary_target must match predicted boundary probability")
    progress_target = progress_target.to(
        device=phase_state.progress.device,
        dtype=phase_state.progress.dtype,
    )
    boundary_target = boundary_target.to(
        device=phase_state.boundary_prob.device,
        dtype=phase_state.boundary_prob.dtype,
    )
    progress_loss = F.l1_loss(phase_state.progress, progress_target)
    boundary_loss = F.binary_cross_entropy(
        phase_state.boundary_prob.clamp(1e-6, 1.0 - 1e-6),
        boundary_target,
    )
    order_loss = phase_state.progress.new_zeros(())
    order_pairs = 0
    if episode_index is not None or call_index is not None:
        if episode_index is None or call_index is None:
            raise ValueError("episode_index and call_index must be provided together")
        if episode_index.ndim != 1 or call_index.shape != episode_index.shape:
            raise ValueError("episode_index/call_index must have shape [B]")
        predicted = phase_state.progress[:, 0]
        same_episode = episode_index[:, None] == episode_index[None, :]
        ordered = call_index[:, None] < call_index[None, :]
        pair_mask = same_episode & ordered
        if pair_mask.any():
            differences = predicted[None, :] - predicted[:, None]
            order_loss = F.relu(config.order_margin - differences[pair_mask]).mean()
            order_pairs = int(pair_mask.sum().item())

    total = (
        config.lambda_progress * progress_loss
        + config.lambda_boundary * boundary_loss
        + config.lambda_order * order_loss
    )
    return {
        "total": total,
        "progress": progress_loss,
        "boundary": boundary_loss,
        "order": order_loss,
        "order_pairs": torch.tensor(
            order_pairs,
            device=total.device,
            dtype=torch.int64,
        ),
    }
