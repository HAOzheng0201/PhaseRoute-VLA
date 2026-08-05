"""Metric-aware, phase-protected exit policy for the M3 depth-only ablation."""

from __future__ import annotations

from dataclasses import dataclass

from .budget_profiles import ResolvedBudget
from .phase_estimator import PhaseState


@dataclass(frozen=True)
class PhaseAwareExitPolicyConfig:
    enforce_min_depth: bool = True
    scale_threshold: bool = True
    protect_boundary: bool = True
    boundary_threshold: float = 0.6
    uncertainty_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.boundary_threshold <= 1.0:
            raise ValueError("boundary_threshold must be in [0, 1]")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str
    layer_idx: int
    exit_rank: int
    min_exit_layer: int
    base_threshold: float
    adjusted_threshold: float
    metric_value: float | None
    threshold_passed: bool | None
    forced_final: bool


class PhaseAwareExitPolicy:
    """Apply minimum depth, metric-aware threshold scaling and risk protection."""

    def __init__(self, config: PhaseAwareExitPolicyConfig | None = None):
        self.config = config or PhaseAwareExitPolicyConfig()

    @staticmethod
    def adjust_threshold(
        base_threshold: float,
        scale: float,
        *,
        lower_is_easier: bool,
    ) -> float:
        """Scale an exit threshold while preserving its ease direction.

        A1 currently uses a distance (`leq=True`), where multiplying by a
        larger scale makes exiting easier.  For a bounded similarity
        (`leq=False`), the same semantic scale expands/contracts the distance
        from the ideal similarity 1.0 instead.
        """

        if scale <= 0.0:
            raise ValueError("threshold scale must be positive")
        if lower_is_easier:
            return float(base_threshold * scale)
        return float(1.0 - (1.0 - base_threshold) * scale)

    def should_exit(
        self,
        *,
        layer_idx: int,
        exit_rank: int,
        metric_value: float | None,
        phase_state: PhaseState,
        budget: ResolvedBudget,
        base_threshold: float,
        lower_is_easier: bool,
        is_final_exit: bool,
    ) -> ExitDecision:
        if phase_state.progress.shape != (1, 1):
            raise ValueError("M3 online exit policy currently requires batch size 1")
        if layer_idx not in budget.eligible_exit_layers:
            raise ValueError(f"Layer {layer_idx} is not an eligible exit")
        if budget.eligible_exit_layers[exit_rank] != layer_idx:
            raise ValueError("exit_rank does not match layer_idx")
        adjusted_threshold = (
            self.adjust_threshold(
                base_threshold,
                budget.profile.exit_threshold_scale,
                lower_is_easier=lower_is_easier,
            )
            if self.config.scale_threshold
            else float(base_threshold)
        )
        if is_final_exit:
            return ExitDecision(
                should_exit=True,
                reason="forced_final",
                layer_idx=layer_idx,
                exit_rank=exit_rank,
                min_exit_layer=budget.min_exit_layer,
                base_threshold=float(base_threshold),
                adjusted_threshold=adjusted_threshold,
                metric_value=metric_value,
                threshold_passed=None,
                forced_final=True,
            )
        if self.config.enforce_min_depth and exit_rank < budget.min_exit_rank:
            return ExitDecision(
                should_exit=False,
                reason="below_min_depth",
                layer_idx=layer_idx,
                exit_rank=exit_rank,
                min_exit_layer=budget.min_exit_layer,
                base_threshold=float(base_threshold),
                adjusted_threshold=adjusted_threshold,
                metric_value=None,
                threshold_passed=None,
                forced_final=False,
            )
        boundary = float(phase_state.boundary_prob[0, 0].detach().cpu())
        uncertainty = float(phase_state.uncertainty[0, 0].detach().cpu())
        if self.config.protect_boundary and (
            boundary >= self.config.boundary_threshold
            or uncertainty >= self.config.uncertainty_threshold
        ):
            return ExitDecision(
                should_exit=False,
                reason="phase_risk_guard",
                layer_idx=layer_idx,
                exit_rank=exit_rank,
                min_exit_layer=budget.min_exit_layer,
                base_threshold=float(base_threshold),
                adjusted_threshold=adjusted_threshold,
                metric_value=metric_value,
                threshold_passed=None,
                forced_final=False,
            )
        if metric_value is None:
            return ExitDecision(
                should_exit=False,
                reason="missing_previous_action",
                layer_idx=layer_idx,
                exit_rank=exit_rank,
                min_exit_layer=budget.min_exit_layer,
                base_threshold=float(base_threshold),
                adjusted_threshold=adjusted_threshold,
                metric_value=None,
                threshold_passed=None,
                forced_final=False,
            )
        threshold_passed = (
            metric_value <= adjusted_threshold
            if lower_is_easier
            else metric_value >= adjusted_threshold
        )
        return ExitDecision(
            should_exit=bool(threshold_passed),
            reason="threshold_passed" if threshold_passed else "threshold_failed",
            layer_idx=layer_idx,
            exit_rank=exit_rank,
            min_exit_layer=budget.min_exit_layer,
            base_threshold=float(base_threshold),
            adjusted_threshold=adjusted_threshold,
            metric_value=float(metric_value),
            threshold_passed=bool(threshold_passed),
            forced_final=False,
        )


def phase_exit_policy_config_for_ablation(
    mode: str,
) -> PhaseAwareExitPolicyConfig:
    """Build the three clean M3 ablations used by the LIBERO runner."""

    if mode == "min_depth":
        return PhaseAwareExitPolicyConfig(
            enforce_min_depth=True,
            scale_threshold=False,
            protect_boundary=False,
        )
    if mode == "threshold":
        return PhaseAwareExitPolicyConfig(
            enforce_min_depth=False,
            scale_threshold=True,
            protect_boundary=False,
        )
    if mode == "joint":
        return PhaseAwareExitPolicyConfig(
            enforce_min_depth=True,
            scale_threshold=True,
            # The edge-triggered B3 minimum is the boundary guard.  Applying
            # a second absolute-probability guard here would force layer 27
            # whenever the weak boundary head remains saturated.
            protect_boundary=False,
        )
    raise ValueError(f"Unknown phase-depth ablation mode: {mode}")
