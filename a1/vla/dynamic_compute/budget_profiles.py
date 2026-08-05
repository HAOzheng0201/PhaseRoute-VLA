"""Discrete width/depth budget profiles and legal-exit resolution for M3+."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    agg_tokens: int | None
    visual_keep_ratio: float
    min_exit_fraction: float
    exit_threshold_scale: float
    fm_steps_per_exit: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Budget profile name cannot be empty")
        if self.agg_tokens is not None and self.agg_tokens < 1:
            raise ValueError("agg_tokens must be positive when provided")
        if not 0.0 < self.visual_keep_ratio <= 1.0:
            raise ValueError("visual_keep_ratio must be in (0, 1]")
        if not 0.0 <= self.min_exit_fraction <= 1.0:
            raise ValueError("min_exit_fraction must be in [0, 1]")
        if self.exit_threshold_scale <= 0.0:
            raise ValueError("exit_threshold_scale must be positive")
        if self.fm_steps_per_exit < 1:
            raise ValueError("fm_steps_per_exit must be positive")


@dataclass(frozen=True)
class ResolvedBudget:
    profile_id: int
    profile: BudgetProfile
    min_exit_rank: int
    min_exit_layer: int
    eligible_exit_layers: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.profile_id < 0:
            raise ValueError("profile_id must be nonnegative")
        if not self.eligible_exit_layers:
            raise ValueError("eligible_exit_layers cannot be empty")
        if not 0 <= self.min_exit_rank < len(self.eligible_exit_layers):
            raise ValueError("min_exit_rank is outside eligible exits")
        if self.min_exit_layer != self.eligible_exit_layers[self.min_exit_rank]:
            raise ValueError("min_exit_layer and rank disagree")


def m3_depth_profiles(fm_steps_per_exit: int = 2) -> Tuple[BudgetProfile, ...]:
    """Return the four-profile M3 depth-only schedule.

    A1's calibrated LIBERO thresholds make layers 15--25 diagnostic
    candidates rather than viable exits (their thresholds are ``-1e8``).
    Keeping B3 at rank 5 / layer 11 for the standard 14-exit schedule avoids
    silently turning a noisy boundary call into a forced layer-27 exit.  The
    shallower minima still skip unproductive flow-matching probes.
    """

    return (
        BudgetProfile("B0", None, 1.0, 0.00, 1.25, fm_steps_per_exit),
        BudgetProfile("B1", None, 1.0, 0.10, 1.00, fm_steps_per_exit),
        BudgetProfile("B2", None, 1.0, 0.25, 0.90, fm_steps_per_exit),
        BudgetProfile("B3", None, 1.0, 0.35, 1.00, fm_steps_per_exit),
    )


class BudgetProfileResolver:
    """Map fractional minimum depth to the actual A1 candidate-exit list."""

    def __init__(
        self,
        profiles: Sequence[BudgetProfile],
        eligible_exit_layers: Iterable[int],
    ):
        self.profiles = tuple(profiles)
        if not self.profiles:
            raise ValueError("profiles cannot be empty")
        names = [profile.name for profile in self.profiles]
        if len(set(names)) != len(names):
            raise ValueError("profile names must be unique")
        layers = tuple(int(layer) for layer in eligible_exit_layers)
        if not layers:
            raise ValueError("eligible_exit_layers cannot be empty")
        if any(layer < 0 for layer in layers):
            raise ValueError("eligible exit layers must be nonnegative")
        if tuple(sorted(set(layers))) != layers:
            raise ValueError("eligible exit layers must be unique and increasing")
        self.eligible_exit_layers = layers
        self._resolved = tuple(
            self._resolve_profile(profile_id, profile)
            for profile_id, profile in enumerate(self.profiles)
        )
        ranks = [budget.min_exit_rank for budget in self._resolved]
        if ranks != sorted(ranks):
            raise ValueError("profiles must have nondecreasing minimum exit depth")

    def _resolve_profile(
        self,
        profile_id: int,
        profile: BudgetProfile,
    ) -> ResolvedBudget:
        max_rank = len(self.eligible_exit_layers) - 1
        rank = min(
            max_rank,
            int(math.ceil(profile.min_exit_fraction * max_rank)),
        )
        return ResolvedBudget(
            profile_id=profile_id,
            profile=profile,
            min_exit_rank=rank,
            min_exit_layer=self.eligible_exit_layers[rank],
            eligible_exit_layers=self.eligible_exit_layers,
        )

    def resolve(self, profile_id: int) -> ResolvedBudget:
        if not 0 <= profile_id < len(self._resolved):
            raise ValueError(f"Unknown profile_id: {profile_id}")
        return self._resolved[profile_id]

    def resolve_name(self, profile_name: str) -> ResolvedBudget:
        for budget in self._resolved:
            if budget.profile.name == profile_name:
                return budget
        raise ValueError(f"Unknown profile name: {profile_name}")

    @property
    def resolved_profiles(self) -> Tuple[ResolvedBudget, ...]:
        return self._resolved
