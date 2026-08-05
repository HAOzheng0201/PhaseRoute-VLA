from __future__ import annotations

import numpy as np
import pytest

from a1.vla.dynamic_compute.temporal_route_features import (
    canonical_teacher_route,
    parse_episode_index,
    right_aligned_history,
)


@pytest.mark.parametrize(
    ("raw", "route"),
    [
        (1, 11),
        (3, 11),
        (11, 11),
        (13, 13),
        (15, 27),
        (27, 27),
    ],
)
def test_teacher_exit_maps_to_first_non_shallower_route(raw: int, route: int) -> None:
    assert canonical_teacher_route(raw) == route
    assert canonical_teacher_route(raw) >= raw


@pytest.mark.parametrize("raw", [0, 2, 12, 14, 28])
def test_teacher_exit_mapping_rejects_non_a1_layers(raw: int) -> None:
    with pytest.raises(ValueError, match="unsupported raw A1 exit"):
        canonical_teacher_route(raw)


def test_parse_episode_index_requires_canonical_suffix() -> None:
    assert parse_episode_index("libero_spatial:task8:episode5") == 5
    with pytest.raises(ValueError, match="canonical suffix"):
        parse_episode_index("episode5:extra")


def test_history_is_right_aligned_and_past_only() -> None:
    history = [
        (np.full((2,), index, dtype=np.float32), np.full((3, 1), index + 10))
        for index in range(2)
    ]
    proprio, action, mask = right_aligned_history(
        history,
        history_len=4,
        proprio_dim=2,
        action_horizon=3,
        action_dim=1,
    )
    np.testing.assert_array_equal(mask, [False, False, True, True])
    np.testing.assert_array_equal(proprio[:2], 0.0)
    np.testing.assert_array_equal(proprio[2], [0.0, 0.0])
    np.testing.assert_array_equal(proprio[3], [1.0, 1.0])
    np.testing.assert_array_equal(action[2], 10.0)
    np.testing.assert_array_equal(action[3], 11.0)


def test_history_rejects_bad_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="invalid shape"):
        right_aligned_history(
            [(np.zeros((3,)), np.zeros((3, 1)))],
            history_len=2,
            proprio_dim=2,
            action_horizon=3,
            action_dim=1,
        )
    with pytest.raises(ValueError, match="non-finite"):
        right_aligned_history(
            [(np.array([0.0, np.nan]), np.zeros((3, 1)))],
            history_len=2,
            proprio_dim=2,
            action_horizon=3,
            action_dim=1,
        )
