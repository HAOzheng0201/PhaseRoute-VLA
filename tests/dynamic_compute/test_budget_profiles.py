import pytest

from a1.vla.dynamic_compute.budget_profiles import (
    BudgetProfileResolver,
    m3_depth_profiles,
)


def test_m3_profiles_keep_full_vision_and_resolve_to_legal_exits():
    exits = tuple(range(1, 28, 2))
    resolver = BudgetProfileResolver(m3_depth_profiles(), exits)
    resolved = resolver.resolved_profiles

    assert [budget.profile.name for budget in resolved] == ["B0", "B1", "B2", "B3"]
    assert all(budget.profile.visual_keep_ratio == 1.0 for budget in resolved)
    assert all(budget.profile.agg_tokens is None for budget in resolved)
    assert [budget.min_exit_rank for budget in resolved] == [0, 2, 4, 5]
    assert all(budget.min_exit_layer in exits for budget in resolved)
    assert resolver.resolve_name("B3") == resolver.resolve(3)


def test_profile_resolver_rejects_invalid_exit_lists_and_ids():
    profiles = m3_depth_profiles()
    with pytest.raises(ValueError, match="unique and increasing"):
        BudgetProfileResolver(profiles, [3, 1])
    resolver = BudgetProfileResolver(profiles, [1, 3, 5])
    with pytest.raises(ValueError, match="Unknown profile_id"):
        resolver.resolve(4)
