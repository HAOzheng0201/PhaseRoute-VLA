from __future__ import annotations

from a1.vla.dynamic_compute.risk_route13_router import (
    M426A_FEATURE_SCHEMA_VERSION,
    M426_FEATURE_SCHEMA_VERSION,
)
from scripts.dynamic_compute.build_m426_temporal_features import (
    EXPECTED_EPISODES,
    EXPECTED_SEED,
    M426A_EXPECTED_EPISODES,
    M426A_EXPECTED_SEED,
    M426A_ROLE_BY_EPISODE,
    ROLE_BY_EPISODE,
)


def test_m426a_is_a_new_frozen_seven_episode_protocol() -> None:
    assert EXPECTED_SEED == 20260926
    assert EXPECTED_EPISODES == tuple(range(6))
    assert ROLE_BY_EPISODE[3] == "calibration"
    assert ROLE_BY_EPISODE[4] == "test"

    assert M426A_EXPECTED_SEED == 20261026
    assert M426A_EXPECTED_EPISODES == tuple(range(7))
    assert [
        index
        for index, role in M426A_ROLE_BY_EPISODE.items()
        if role == "development"
    ] == [0, 1, 2]
    assert [
        index
        for index, role in M426A_ROLE_BY_EPISODE.items()
        if role == "calibration"
    ] == [3, 4]
    assert [
        index for index, role in M426A_ROLE_BY_EPISODE.items() if role == "test"
    ] == [5, 6]
    assert M426A_FEATURE_SCHEMA_VERSION != M426_FEATURE_SCHEMA_VERSION
