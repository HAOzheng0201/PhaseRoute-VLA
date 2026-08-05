import pytest

from scripts.dynamic_compute.profile_m424_oracle_route_then_solve import (
    ORACLE_RNG_BURNS,
    load_frozen_routes,
)


def _route_source(*, mismatch: bool = False):
    rows = []
    for repeat in (0, 1):
        rows.append(
            {
                "cache_dir": "/cache/task0",
                "array_path": "arrays/call_000000.npz",
                "episode_id": "suite:task0:episode0",
                "task_id": 0,
                "step_id": 10,
                "teacher_exit_layer": 13,
                "repeat": repeat,
                "exit_layer": 11,
                "fm_calls": 7,
                "action_sha256": "b" * 64 if not mismatch or repeat == 0 else "c" * 64,
            }
        )
    return {
        "scope": "m423_fixed_observation_policy_profile",
        "status": "PASS",
        "policy": "early_exit",
        "checkpoint_sha256": "a" * 64,
        "selection_sha256": "d" * 64,
        "timed_samples": rows,
    }


def test_oracle_burn_schedule_is_original_calls_minus_one():
    assert ORACLE_RNG_BURNS == {3: 2, 11: 6, 13: 7, 27: 14}


def test_frozen_route_loader_validates_and_collapses_repeats():
    routes = load_frozen_routes(
        _route_source(),
        checkpoint_sha256="a" * 64,
        expected_selection_sha256="d" * 64,
    )
    assert len(routes) == 1
    route = next(iter(routes.values()))
    assert route == {
        "route_layer": 11,
        "original_fm_calls": 7,
        "rng_burns": 6,
        "expected_action_sha256": "b" * 64,
    }


def test_frozen_route_loader_rejects_repeat_action_mismatch():
    with pytest.raises(ValueError, match="repeats disagree"):
        load_frozen_routes(
            _route_source(mismatch=True),
            checkpoint_sha256="a" * 64,
            expected_selection_sha256="d" * 64,
        )
