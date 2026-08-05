"""Deterministic temporal hysteresis for A1 exit-depth decisions.

The controller is deliberately independent from model features and success
labels.  It only receives the exit layer proposed by the unmodified A1 exit
criterion and may conservatively keep that proposal at a previously latched
deeper layer.  It never routes shallower than the raw proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ExitDepthHysteresisConfig:
    """Configuration for conservative exit-depth downgrade hysteresis."""

    enabled: bool = False
    release_after_shallow_calls: int = 2
    max_latched_layer: Optional[int] = None

    def __post_init__(self) -> None:
        if self.release_after_shallow_calls < 1:
            raise ValueError("release_after_shallow_calls must be positive")
        if self.max_latched_layer is not None and self.max_latched_layer < 0:
            raise ValueError("max_latched_layer must be nonnegative")


@dataclass(frozen=True)
class ExitDepthHysteresisDecision:
    """One auditable transformation from a raw to a routed exit layer."""

    raw_layer: int
    routed_layer: int
    reason: str
    latched_layer_before: Optional[int]
    latched_layer_after: Optional[int]
    pending_shallow_layer: Optional[int]
    pending_shallow_calls: int


class ExitDepthHysteresis:
    """Require repeated evidence before moving a latched exit shallower."""

    def __init__(
        self,
        config: ExitDepthHysteresisConfig,
        eligible_exit_layers: Sequence[int],
    ) -> None:
        layers = tuple(int(layer) for layer in eligible_exit_layers)
        if not layers:
            raise ValueError("eligible_exit_layers must not be empty")
        if tuple(sorted(set(layers))) != layers:
            raise ValueError("eligible_exit_layers must be unique and increasing")
        if layers[0] < 0:
            raise ValueError("eligible_exit_layers must be nonnegative")
        if (
            config.max_latched_layer is not None
            and config.max_latched_layer not in layers
        ):
            raise ValueError("max_latched_layer must be an eligible exit layer")
        self.config = config
        self.eligible_exit_layers = layers
        self.final_exit_layer = layers[-1]
        self._eligible_set = frozenset(layers)
        self.episodes_reset = 0
        self.decisions = 0
        self.reset_episode()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def latched_layer(self) -> Optional[int]:
        return self._latched_layer

    @property
    def pending_shallow_layer(self) -> Optional[int]:
        return self._pending_shallow_layer

    @property
    def pending_shallow_calls(self) -> int:
        return self._pending_shallow_calls

    def reset_episode(self) -> None:
        """Clear all temporal state at a real environment episode boundary."""

        self._latched_layer: Optional[int] = None
        self._pending_shallow_layer: Optional[int] = None
        self._pending_shallow_calls = 0
        self.episodes_reset += 1

    def _may_latch(self, layer: int) -> bool:
        cap = self.config.max_latched_layer
        return cap is None or layer <= cap

    def _clear_pending(self) -> None:
        self._pending_shallow_layer = None
        self._pending_shallow_calls = 0

    def route(self, proposed_layer: int) -> ExitDepthHysteresisDecision:
        """Route one raw A1 proposal; call exactly once per policy call."""

        raw = int(proposed_layer)
        if raw not in self._eligible_set:
            raise ValueError(f"proposed layer {raw} is not an eligible exit")
        before = self._latched_layer

        if not self.enabled:
            routed = raw
            reason = "disabled"
        elif raw == self.final_exit_layer:
            # The final output must never be blocked.  With a finite latch cap,
            # do not turn one final-layer excursion into several full-depth calls.
            routed = raw
            reason = "final_exit_passthrough"
            if self._may_latch(raw):
                self._latched_layer = raw
            self._clear_pending()
        elif before is None:
            routed = raw
            reason = "initial_proposal"
            if self._may_latch(raw):
                self._latched_layer = raw
            self._clear_pending()
        elif raw > before:
            routed = raw
            reason = "deeper_upgrade"
            if self._may_latch(raw):
                self._latched_layer = raw
            self._clear_pending()
        elif raw == before:
            routed = raw
            reason = "latched_match"
            self._clear_pending()
        else:
            if self._pending_shallow_layer == raw:
                self._pending_shallow_calls += 1
            else:
                self._pending_shallow_layer = raw
                self._pending_shallow_calls = 1
            if (
                self._pending_shallow_calls
                >= self.config.release_after_shallow_calls
            ):
                routed = raw
                reason = "shallow_release"
                self._latched_layer = raw
                self._clear_pending()
            else:
                routed = before
                reason = "shallow_deferred"

        if routed < raw:
            raise RuntimeError("depth hysteresis routed shallower than raw A1")
        self.decisions += 1
        return ExitDepthHysteresisDecision(
            raw_layer=raw,
            routed_layer=routed,
            reason=reason,
            latched_layer_before=before,
            latched_layer_after=self._latched_layer,
            pending_shallow_layer=self._pending_shallow_layer,
            pending_shallow_calls=self._pending_shallow_calls,
        )


__all__ = [
    "ExitDepthHysteresis",
    "ExitDepthHysteresisConfig",
    "ExitDepthHysteresisDecision",
]
