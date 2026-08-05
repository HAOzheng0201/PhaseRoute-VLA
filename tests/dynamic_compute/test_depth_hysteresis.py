import pytest

from a1.vla.dynamic_compute.depth_hysteresis import (
    ExitDepthHysteresis,
    ExitDepthHysteresisConfig,
)


LAYERS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27)


def _router(release_calls=2, *, enabled=True, cap=13):
    return ExitDepthHysteresis(
        ExitDepthHysteresisConfig(
            enabled=enabled,
            release_after_shallow_calls=release_calls,
            max_latched_layer=cap,
        ),
        LAYERS,
    )


def test_deeper_upgrade_is_immediate_and_shallow_release_needs_same_target():
    router = _router(release_calls=2)

    assert router.route(11).routed_layer == 11
    upgrade = router.route(13)
    first_shallow = router.route(11)
    changed_target = router.route(9)
    second_same_target = router.route(9)

    assert upgrade.routed_layer == 13
    assert upgrade.reason == "deeper_upgrade"
    assert first_shallow.routed_layer == 13
    assert first_shallow.reason == "shallow_deferred"
    assert changed_target.routed_layer == 13
    assert changed_target.pending_shallow_calls == 1
    assert second_same_target.routed_layer == 9
    assert second_same_target.reason == "shallow_release"
    assert router.latched_layer == 9


def test_final_exit_passes_through_without_latching_above_cap():
    router = _router(release_calls=3, cap=13)

    router.route(13)
    final = router.route(27)
    after_final = router.route(11)

    assert final.routed_layer == 27
    assert final.reason == "final_exit_passthrough"
    assert final.latched_layer_after == 13
    assert after_final.routed_layer == 13


def test_episode_reset_clears_latch_and_pending_evidence():
    router = _router(release_calls=2)
    router.route(13)
    router.route(11)

    router.reset_episode()
    decision = router.route(11)

    assert decision.routed_layer == 11
    assert decision.reason == "initial_proposal"
    assert decision.pending_shallow_calls == 0


def test_disabled_router_is_identity_and_does_not_create_a_latch():
    router = _router(enabled=False)
    decisions = [router.route(layer) for layer in (11, 13, 11, 27)]

    assert [item.routed_layer for item in decisions] == [11, 13, 11, 27]
    assert {item.reason for item in decisions} == {"disabled"}
    assert router.latched_layer is None


@pytest.mark.parametrize(
    "config_kwargs,layers,error",
    [
        ({"release_after_shallow_calls": 0}, LAYERS, "positive"),
        ({"max_latched_layer": -1}, LAYERS, "nonnegative"),
        ({"max_latched_layer": 12}, LAYERS, "eligible"),
        ({}, (1, 3, 3), "unique and increasing"),
        ({}, (), "must not be empty"),
    ],
)
def test_invalid_configurations_are_rejected(config_kwargs, layers, error):
    with pytest.raises(ValueError, match=error):
        ExitDepthHysteresis(ExitDepthHysteresisConfig(**config_kwargs), layers)


def test_routing_rejects_non_candidate_layer():
    router = _router()
    with pytest.raises(ValueError, match="not an eligible exit"):
        router.route(12)
