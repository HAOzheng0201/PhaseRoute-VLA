from types import SimpleNamespace

import pytest
import torch
from torch import nn

from a1.vla.dynamic_compute.phase_vision_runtime import (
    PhaseProfileVisionRouter,
    PhaseVisionProfile,
    make_exit_controller_profile_provider,
    make_phase_runtime_profile_provider,
)
from a1.vla.dynamic_compute.vision_aggregation import (
    StaticVisionAggregationConfig,
    aggregate_projected_vision,
)


def _fixture():
    features = torch.arange(32, dtype=torch.float32).reshape(1, 1, 8, 4)
    positions = torch.arange(8, dtype=torch.long).reshape(1, 1, 8) + 2
    instruction = torch.ones(1, 4)
    return features, positions, instruction


class _RecordingAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, positions, instruction):
        assert instruction.shape == (1, 4)
        self.calls += 1
        return aggregate_projected_vision(
            features,
            positions,
            StaticVisionAggregationConfig(
                enabled=True,
                keep_tokens=4,
                min_tokens_per_crop=1,
                fail_open=False,
            ),
        )


@pytest.mark.parametrize("profile_name", ["B0", "B1", "B2"])
def test_lower_risk_profiles_call_base_efa(profile_name):
    base = _RecordingAggregator()
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=lambda: PhaseVisionProfile(profile_name, "low_risk"),
    )

    output = router(*_fixture())

    assert base.calls == 1
    assert output.route == "learned_efa"
    assert output.full_token_fallback is False
    assert output.profile_name == profile_name
    assert output.aggregated.kept_counts.tolist() == [4]
    assert output.aggregated.compression_applied is True


def test_b3_bypasses_base_efa_and_preserves_all_tokens():
    base = _RecordingAggregator()
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=lambda: PhaseVisionProfile("B3", "boundary"),
    )

    output = router(*_fixture())

    assert base.calls == 0
    assert output.route == "full_token"
    assert output.full_token_fallback is True
    assert output.profile_reason == "boundary"
    assert output.aggregated.kept_counts.tolist() == [8]
    assert output.aggregated.original_counts.tolist() == [8]
    assert output.aggregated.compression_applied is False


def test_contact_schedule_can_protect_b1_and_b3():
    base = _RecordingAggregator()
    current = {"name": "B1"}
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=lambda: PhaseVisionProfile(current["name"], "contact"),
        full_token_profiles=("B1", "B3"),
    )

    b1_output = router(*_fixture())
    current["name"] = "B0"
    b0_output = router(*_fixture())

    assert base.calls == 1
    assert b1_output.route == "full_token"
    assert b1_output.aggregated.kept_counts.tolist() == [8]
    assert b0_output.route == "learned_efa"
    assert b0_output.aggregated.kept_counts.tolist() == [4]


def test_full_token_hysteresis_holds_then_resets_at_episode_boundary():
    base = _RecordingAggregator()
    current = {"name": "B3"}
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=lambda: PhaseVisionProfile(current["name"]),
        full_token_profiles=("B3",),
        full_token_hold_calls=2,
    )

    direct = router(*_fixture())
    current["name"] = "B0"
    held_1 = router(*_fixture())
    held_2 = router(*_fixture())
    released = router(*_fixture())
    router.reset_route_state()
    after_reset = router(*_fixture())

    assert [direct.route, held_1.route, held_2.route, released.route] == [
        "full_token",
        "full_token_hold",
        "full_token_hold",
        "learned_efa",
    ]
    assert after_reset.route == "learned_efa"
    assert base.calls == 2


def test_uncertainty_trigger_holds_then_resets_at_episode_boundary():
    base = _RecordingAggregator()
    current = {"uncertainty": 0.059}
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=lambda: PhaseVisionProfile(
            "B0", "default", current["uncertainty"]
        ),
        full_token_profiles=("B3",),
        full_token_hold_calls=2,
        full_token_uncertainty_threshold=0.06,
    )

    compressed = router(*_fixture())
    current["uncertainty"] = 0.06
    triggered = router(*_fixture())
    current["uncertainty"] = 0.01
    held = router(*_fixture())
    router.reset_route_state()
    after_reset = router(*_fixture())

    assert compressed.route == "learned_efa"
    assert compressed.full_token_trigger is None
    assert triggered.route == "full_token"
    assert triggered.full_token_trigger == "uncertainty"
    assert triggered.profile_uncertainty == pytest.approx(0.06)
    assert held.route == "full_token_hold"
    assert held.full_token_trigger == "hold"
    assert after_reset.route == "learned_efa"
    assert base.calls == 2


def test_uncertainty_is_ignored_when_threshold_is_disabled():
    base = _RecordingAggregator()
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=lambda: PhaseVisionProfile("B0", uncertainty=1.0),
    )

    output = router(*_fixture())

    assert output.route == "learned_efa"
    assert base.calls == 1


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_uncertainty_threshold_must_be_a_probability(threshold):
    with pytest.raises(ValueError):
        PhaseProfileVisionRouter(
            _RecordingAggregator(),
            profile_provider=lambda: PhaseVisionProfile("B0"),
            full_token_uncertainty_threshold=threshold,
        )


def test_missing_phase_plan_fails_safe_to_full_tokens():
    base = _RecordingAggregator()
    router = PhaseProfileVisionRouter(base, profile_provider=lambda: None)

    output = router(*_fixture())

    assert base.calls == 0
    assert output.profile_name is None
    assert output.profile_reason == "missing_phase_plan"
    assert output.route == "full_token"


def test_exit_controller_provider_reads_current_plan_without_registering_controller():
    controller = nn.Module()
    controller.resolved_budget = SimpleNamespace(
        profile=SimpleNamespace(name="B3")
    )
    controller.phase_profile_reason = "uncertainty"
    base = _RecordingAggregator()
    router = PhaseProfileVisionRouter(
        base,
        profile_provider=make_exit_controller_profile_provider(controller),
    )

    profile = router._profile_provider()

    assert profile == PhaseVisionProfile("B3", "uncertainty")
    assert list(router.children()) == [base]


def test_phase_runtime_provider_supports_width_only_routing():
    class _Runtime:
        pass

    runtime = _Runtime()
    runtime.current_plan = SimpleNamespace(
        budget=SimpleNamespace(profile=SimpleNamespace(name="B1")),
        selection=SimpleNamespace(reasons=("low_motion",)),
        phase_state=SimpleNamespace(uncertainty=torch.tensor([[0.25]])),
    )
    provider = make_phase_runtime_profile_provider(runtime)

    assert provider() == PhaseVisionProfile("B1", "low_motion", 0.25)
    runtime.current_plan = None
    assert provider().name is None
