"""Transparent phase-to-profile rule used only for the M3 hypothesis test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .phase_estimator import PhaseState


@dataclass(frozen=True)
class PhaseRuleConfig:
    boundary_threshold: float = 0.6
    uncertainty_threshold: float = 0.6
    rapid_progress_threshold: float = 0.15
    low_motion_threshold: float = 0.3

    def __post_init__(self) -> None:
        for name in ("boundary_threshold", "uncertainty_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.rapid_progress_threshold < 0.0:
            raise ValueError("rapid_progress_threshold must be nonnegative")
        if self.low_motion_threshold < 0.0:
            raise ValueError("low_motion_threshold must be nonnegative")


@dataclass
class BudgetSelection:
    profile_id: torch.Tensor
    profile_probs: torch.Tensor
    reasons: Tuple[str, ...]


class TransparentPhaseBudgetController:
    """Select B0--B3 without learned parameters or hidden teacher calls."""

    num_profiles = 4

    def __init__(self, config: PhaseRuleConfig | None = None):
        self.config = config or PhaseRuleConfig()

    def __call__(
        self,
        phase_state: PhaseState,
        *,
        progress_delta: torch.Tensor | None = None,
        motion_speed: torch.Tensor | None = None,
    ) -> BudgetSelection:
        progress = phase_state.progress
        boundary = phase_state.boundary_prob
        uncertainty = phase_state.uncertainty
        if progress.ndim != 2 or progress.shape[1:] != (1,):
            raise ValueError("phase progress must have shape [B, 1]")
        if boundary.shape != progress.shape or uncertainty.shape != progress.shape:
            raise ValueError("phase scalar outputs must have aligned [B, 1] shapes")
        batch_size = progress.shape[0]
        if progress_delta is None:
            progress_delta = torch.zeros_like(progress)
        if motion_speed is None:
            motion_speed = torch.full_like(progress, float("inf"))
        if progress_delta.shape != progress.shape or motion_speed.shape != progress.shape:
            raise ValueError("progress_delta and motion_speed must have shape [B, 1]")

        config = self.config
        profile_id = torch.zeros(
            batch_size,
            device=progress.device,
            dtype=torch.long,
        )
        low_motion = motion_speed <= config.low_motion_threshold
        rapid_progress = progress_delta.abs() >= config.rapid_progress_threshold
        high_risk = (
            (boundary >= config.boundary_threshold)
            | (uncertainty >= config.uncertainty_threshold)
        )
        profile_id = torch.where(low_motion[:, 0], 1, profile_id)
        profile_id = torch.where(rapid_progress[:, 0], 2, profile_id)
        profile_id = torch.where(high_risk[:, 0], 3, profile_id)
        probabilities = torch.nn.functional.one_hot(
            profile_id,
            num_classes=self.num_profiles,
        ).to(dtype=progress.dtype)
        reasons = []
        for index in range(batch_size):
            if bool(high_risk[index, 0]):
                reasons.append("boundary_or_uncertainty")
            elif bool(rapid_progress[index, 0]):
                reasons.append("rapid_progress")
            elif bool(low_motion[index, 0]):
                reasons.append("low_motion")
            else:
                reasons.append("default")
        return BudgetSelection(
            profile_id=profile_id,
            profile_probs=probabilities,
            reasons=tuple(reasons),
        )
