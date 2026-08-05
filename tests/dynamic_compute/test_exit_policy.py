import pytest
import torch

from a1.vla.dynamic_compute.budget_profiles import (
    BudgetProfileResolver,
    m3_depth_profiles,
)
from a1.vla.dynamic_compute.exit_policy import (
    PhaseAwareExitPolicy,
    PhaseAwareExitPolicyConfig,
    phase_exit_policy_config_for_ablation,
)
from a1.vla.dynamic_compute.phase_estimator import PhaseState


def _phase(boundary=0.1, uncertainty=0.1):
    return PhaseState(
        stage_embedding=torch.zeros(1, 4),
        progress=torch.tensor([[0.5]]),
        boundary_prob=torch.tensor([[boundary]]),
        uncertainty=torch.tensor([[uncertainty]]),
        next_hidden=torch.zeros(1, 1, 4),
    )


def test_minimum_depth_and_final_exit_are_hard_constraints():
    exits = (1, 3, 5, 7, 9)
    budget = BudgetProfileResolver(m3_depth_profiles(), exits).resolve(2)
    policy = PhaseAwareExitPolicy()

    early = policy.should_exit(
        layer_idx=1,
        exit_rank=0,
        metric_value=0.0,
        phase_state=_phase(),
        budget=budget,
        base_threshold=0.1,
        lower_is_easier=True,
        is_final_exit=False,
    )
    final = policy.should_exit(
        layer_idx=9,
        exit_rank=4,
        metric_value=999.0,
        phase_state=_phase(boundary=1.0, uncertainty=1.0),
        budget=budget,
        base_threshold=0.1,
        lower_is_easier=True,
        is_final_exit=True,
    )
    assert not early.should_exit and early.reason == "below_min_depth"
    assert final.should_exit and final.reason == "forced_final"


def test_boundary_guard_blocks_aggressive_exit():
    exits = (1, 3, 5)
    budget = BudgetProfileResolver(m3_depth_profiles(), exits).resolve(0)
    decision = PhaseAwareExitPolicy().should_exit(
        layer_idx=1,
        exit_rank=0,
        metric_value=0.0,
        phase_state=_phase(boundary=0.8),
        budget=budget,
        base_threshold=0.1,
        lower_is_easier=True,
        is_final_exit=False,
    )
    assert not decision.should_exit
    assert decision.reason == "phase_risk_guard"


def test_threshold_scaling_preserves_distance_and_similarity_ease_direction():
    policy = PhaseAwareExitPolicy(
        PhaseAwareExitPolicyConfig(protect_boundary=False)
    )
    exits = (1, 3, 5)
    easy_budget = BudgetProfileResolver(m3_depth_profiles(), exits).resolve(0)

    distance = policy.should_exit(
        layer_idx=1,
        exit_rank=0,
        metric_value=0.11,
        phase_state=_phase(),
        budget=easy_budget,
        base_threshold=0.1,
        lower_is_easier=True,
        is_final_exit=False,
    )
    similarity = policy.should_exit(
        layer_idx=1,
        exit_rank=0,
        metric_value=0.89,
        phase_state=_phase(),
        budget=easy_budget,
        base_threshold=0.9,
        lower_is_easier=False,
        is_final_exit=False,
    )
    assert distance.adjusted_threshold == 0.125
    assert distance.should_exit
    assert similarity.adjusted_threshold == 0.875
    assert similarity.should_exit


def test_phase_exit_ablation_configs_are_orthogonal():
    min_depth = phase_exit_policy_config_for_ablation("min_depth")
    threshold = phase_exit_policy_config_for_ablation("threshold")
    joint = phase_exit_policy_config_for_ablation("joint")

    assert (
        min_depth.enforce_min_depth,
        min_depth.scale_threshold,
        min_depth.protect_boundary,
    ) == (True, False, False)
    assert (
        threshold.enforce_min_depth,
        threshold.scale_threshold,
        threshold.protect_boundary,
    ) == (False, True, False)
    assert (
        joint.enforce_min_depth,
        joint.scale_threshold,
        joint.protect_boundary,
    ) == (True, True, False)

    with pytest.raises(ValueError, match="Unknown phase-depth"):
        phase_exit_policy_config_for_ablation("unknown")
