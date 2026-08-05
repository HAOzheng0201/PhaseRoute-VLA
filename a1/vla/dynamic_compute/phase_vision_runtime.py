"""Phase-profile-aware visual-width routing for M4.11.

The router is deliberately opt-in.  It wraps an already loaded learnable EFA
module and conservatively preserves all original visual tokens for selected
high-risk profiles (B3 by default).  Lower-risk profiles continue to use the
base EFA unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Optional, Sequence
import weakref

import torch
from torch import nn

from .vision_aggregation import (
    AggregatedVision,
    StaticVisionAggregationConfig,
    aggregate_projected_vision,
)


@dataclass(frozen=True)
class PhaseVisionProfile:
    """The current causal phase profile exposed to the visual router."""

    name: Optional[str]
    reason: Optional[str] = None
    uncertainty: Optional[float] = None


@dataclass(frozen=True)
class PhaseRoutedVisionAggregation:
    """Aggregation output plus auditable route metadata."""

    aggregated: AggregatedVision
    profile_name: Optional[str]
    profile_reason: Optional[str]
    profile_uncertainty: Optional[float]
    route: str
    full_token_fallback: bool
    full_token_trigger: Optional[str]
    aggregation_kind: str = "phase_profile_learned"


PhaseProfileProvider = Callable[[], Any]


def make_exit_controller_profile_provider(
    exit_controller: Any,
) -> Callable[[], PhaseVisionProfile]:
    """Read the active profile without retaining A1 through a module cycle."""

    controller_ref = weakref.ref(exit_controller)

    def provider() -> PhaseVisionProfile:
        controller = controller_ref()
        if controller is None:
            return PhaseVisionProfile(name=None, reason="controller_released")
        budget = getattr(controller, "resolved_budget", None)
        profile = getattr(budget, "profile", None)
        return PhaseVisionProfile(
            name=getattr(profile, "name", None),
            reason=getattr(controller, "phase_profile_reason", None),
        )

    return provider


def make_phase_runtime_profile_provider(
    phase_runtime: Any,
) -> Callable[[], PhaseVisionProfile]:
    """Read the current estimator plan without enabling phase depth control."""

    runtime_ref = weakref.ref(phase_runtime)

    def provider() -> PhaseVisionProfile:
        runtime = runtime_ref()
        if runtime is None:
            return PhaseVisionProfile(name=None, reason="runtime_released")
        plan = getattr(runtime, "current_plan", None)
        budget = getattr(plan, "budget", None)
        profile = getattr(budget, "profile", None)
        selection = getattr(plan, "selection", None)
        reasons = getattr(selection, "reasons", ())
        phase_state = getattr(plan, "phase_state", None)
        return PhaseVisionProfile(
            name=getattr(profile, "name", None),
            reason=reasons[0] if reasons else None,
            uncertainty=_optional_scalar(
                getattr(phase_state, "uncertainty", None)
            ),
        )

    return provider


def _optional_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("phase uncertainty must contain exactly one value")
        value = value.detach().to(device="cpu", dtype=torch.float32).item()
    result = float(value)
    # Invalid estimator output must route conservatively rather than silently
    # allowing learned compression.
    return result if math.isfinite(result) else 1.0


def _normalize_profile(value: Any) -> PhaseVisionProfile:
    if value is None:
        return PhaseVisionProfile(name=None, reason="missing_phase_plan")
    if isinstance(value, PhaseVisionProfile):
        return value
    if isinstance(value, str):
        return PhaseVisionProfile(name=value)
    if isinstance(value, Mapping):
        return PhaseVisionProfile(
            name=value.get("name") or value.get("profile_name"),
            reason=value.get("reason") or value.get("profile_reason"),
            uncertainty=_optional_scalar(
                value.get("uncertainty", value.get("phase_uncertainty"))
            ),
        )
    profile = getattr(value, "profile", value)
    return PhaseVisionProfile(
        name=getattr(profile, "name", None),
        reason=getattr(value, "reason", None),
        uncertainty=_optional_scalar(getattr(value, "uncertainty", None)),
    )


def _keep_all_projected_vision(
    projected_features: torch.Tensor,
    image_input_idx: torch.Tensor,
) -> AggregatedVision:
    if image_input_idx.ndim != 3:
        raise ValueError("image_input_idx must have shape [B, C, M]")
    valid_counts = (image_input_idx >= 0).sum(dim=(1, 2))
    if (valid_counts < 1).any():
        raise ValueError("every sample must contain a valid projected token")
    keep_tokens = int(valid_counts.max().item())
    aggregated = aggregate_projected_vision(
        projected_features,
        image_input_idx,
        StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=keep_tokens,
            bank_tokens=keep_tokens,
            min_tokens_per_crop=1,
            fail_open=False,
        ),
    )
    if aggregated.compression_applied:
        raise RuntimeError("keep-all phase route unexpectedly compressed visual tokens")
    return aggregated


class PhaseProfileVisionRouter(nn.Module):
    """Use full visual width for high-risk profiles and EFA otherwise."""

    def __init__(
        self,
        base_aggregator: nn.Module,
        *,
        profile_provider: PhaseProfileProvider,
        full_token_profiles: Sequence[str] = ("B3",),
        full_token_on_missing_profile: bool = True,
        full_token_hold_calls: int = 0,
        full_token_uncertainty_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not callable(profile_provider):
            raise TypeError("profile_provider must be callable")
        profiles = frozenset(str(name) for name in full_token_profiles)
        if not profiles or any(not name for name in profiles):
            raise ValueError("full_token_profiles must contain non-empty names")
        self.base_aggregator = base_aggregator
        self._profile_provider = profile_provider
        self.full_token_profiles = profiles
        self.full_token_on_missing_profile = bool(full_token_on_missing_profile)
        if full_token_hold_calls < 0:
            raise ValueError("full_token_hold_calls must be nonnegative")
        self.full_token_hold_calls = int(full_token_hold_calls)
        if full_token_uncertainty_threshold is not None:
            threshold = float(full_token_uncertainty_threshold)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "full_token_uncertainty_threshold must be within [0, 1]"
                )
            self.full_token_uncertainty_threshold = threshold
        else:
            self.full_token_uncertainty_threshold = None
        self._remaining_full_token_hold_calls = 0

    def reset_route_state(self) -> None:
        """Reset hysteresis state at an episode boundary."""

        self._remaining_full_token_hold_calls = 0

    def forward(
        self,
        projected_features: torch.Tensor,
        image_input_idx: torch.Tensor,
        instruction_summary: torch.Tensor,
    ) -> PhaseRoutedVisionAggregation:
        profile = _normalize_profile(self._profile_provider())
        profile_triggered = profile.name in self.full_token_profiles
        missing_triggered = (
            profile.name is None and self.full_token_on_missing_profile
        )
        uncertainty_triggered = (
            self.full_token_uncertainty_threshold is not None
            and profile.uncertainty is not None
            and profile.uncertainty >= self.full_token_uncertainty_threshold
        )
        direct_full_tokens = (
            profile_triggered or missing_triggered or uncertainty_triggered
        )
        direct_trigger = (
            "profile"
            if profile_triggered
            else "missing_profile"
            if missing_triggered
            else "uncertainty"
            if uncertainty_triggered
            else None
        )
        held_full_tokens = False
        if direct_full_tokens:
            self._remaining_full_token_hold_calls = self.full_token_hold_calls
        elif self._remaining_full_token_hold_calls > 0:
            held_full_tokens = True
            self._remaining_full_token_hold_calls -= 1
        use_full_tokens = direct_full_tokens or held_full_tokens
        if use_full_tokens:
            aggregated = _keep_all_projected_vision(
                projected_features,
                image_input_idx,
            )
            route = "full_token_hold" if held_full_tokens else "full_token"
        else:
            output = self.base_aggregator(
                projected_features,
                image_input_idx,
                instruction_summary,
            )
            aggregated = getattr(output, "aggregated", output)
            if not isinstance(aggregated, AggregatedVision):
                raise TypeError("base aggregator did not return AggregatedVision")
            route = "learned_efa"
        return PhaseRoutedVisionAggregation(
            aggregated=aggregated,
            profile_name=profile.name,
            profile_reason=profile.reason,
            profile_uncertainty=profile.uncertainty,
            route=route,
            full_token_fallback=use_full_tokens,
            full_token_trigger="hold" if held_full_tokens else direct_trigger,
        )


__all__ = [
    "PhaseProfileVisionRouter",
    "PhaseRoutedVisionAggregation",
    "PhaseVisionProfile",
    "make_exit_controller_profile_provider",
    "make_phase_runtime_profile_provider",
]
