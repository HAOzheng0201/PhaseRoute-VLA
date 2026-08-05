"""Validated plans for RNG-preserving productive-exit pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class ProductiveExitPlan:
    """A fixed sparse exit schedule plus exact RNG-stream preservation rules."""

    original_exit_layers: tuple[int, ...]
    eligible_exit_layers: tuple[int, ...]
    comparison_reference_by_exit: tuple[tuple[int, int], ...]
    rng_burns_by_exit: tuple[tuple[int, int], ...]
    name: str = "rp_pep"

    def __post_init__(self) -> None:
        original = self.original_exit_layers
        eligible = self.eligible_exit_layers
        if not original or tuple(sorted(set(original))) != original:
            raise ValueError("original_exit_layers must be unique and increasing")
        if not eligible or tuple(sorted(set(eligible))) != eligible:
            raise ValueError("eligible_exit_layers must be unique and increasing")
        if not set(eligible).issubset(original):
            raise ValueError("eligible exits must be a subset of original exits")
        if eligible[-1] != original[-1]:
            raise ValueError("productive plan must retain the final exit")
        references = dict(self.comparison_reference_by_exit)
        burns = dict(self.rng_burns_by_exit)
        if len(references) != len(self.comparison_reference_by_exit):
            raise ValueError("comparison references must have unique exit keys")
        if len(burns) != len(self.rng_burns_by_exit):
            raise ValueError("RNG burn rules must have unique exit keys")
        if not set(references).issubset(eligible) or not set(burns).issubset(eligible):
            raise ValueError("productive rules must target eligible exits")
        for exit_layer, reference_layer in references.items():
            if not 0 <= reference_layer < exit_layer:
                raise ValueError("comparison reference must precede its exit")
        if any(count < 0 for count in burns.values()):
            raise ValueError("RNG burn counts must be nonnegative")

    @property
    def comparison_references(self) -> dict[int, int]:
        return dict(self.comparison_reference_by_exit)

    @property
    def rng_burns(self) -> dict[int, int]:
        return dict(self.rng_burns_by_exit)

    def comparison_reference(self, exit_layer: int) -> Optional[int]:
        return self.comparison_references.get(int(exit_layer))

    def rng_burn_count(self, exit_layer: int) -> int:
        return self.rng_burns.get(int(exit_layer), 0)

    def validate_thresholds(
        self,
        thresholds: Mapping[int, float],
        *,
        lower_is_easier: bool,
    ) -> None:
        normalized = {int(layer): float(value) for layer, value in thresholds.items()}
        missing = set(self.original_exit_layers) - set(normalized)
        if missing:
            raise ValueError(f"thresholds are missing original exits: {sorted(missing)}")
        if not lower_is_easier:
            raise ValueError("RP-PEP v1 requires a lower-is-easier exit metric")
        pruned = set(self.original_exit_layers) - set(self.eligible_exit_layers)
        unsafe = {
            layer: normalized[layer]
            for layer in pruned
            if normalized[layer] > 0.0
        }
        if unsafe:
            raise ValueError(f"pruned exits have positive thresholds: {unsafe}")
        if normalized[self.original_exit_layers[-1]] <= 0.0:
            raise ValueError("final exit threshold must remain a positive fallback")

    def select_eligible_thresholds(
        self,
        thresholds: Mapping[int, float],
        *,
        lower_is_easier: bool,
    ) -> tuple[float, ...]:
        """Validate the frozen full schedule and select the retained values."""

        self.validate_thresholds(
            thresholds,
            lower_is_easier=lower_is_easier,
        )
        normalized = {int(layer): float(value) for layer, value in thresholds.items()}
        return tuple(normalized[layer] for layer in self.eligible_exit_layers)


def a1_fm10_rp_pep_plan(original_exit_layers: Sequence[int]) -> ProductiveExitPlan:
    """Build the preregistered A1-FM10 RP-PEP schedule."""

    original = tuple(int(layer) for layer in original_exit_layers)
    required = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27)
    if original != required:
        raise ValueError(f"RP-PEP v1 requires A1 exits {required}, got {original}")
    return ProductiveExitPlan(
        original_exit_layers=original,
        eligible_exit_layers=(3, 11, 13, 27),
        comparison_reference_by_exit=((3, 1), (11, 9), (27, 25)),
        rng_burns_by_exit=((3, 1), (11, 2), (27, 5)),
    )


__all__ = [
    "ProductiveExitPlan",
    "a1_fm10_rp_pep_plan",
]
