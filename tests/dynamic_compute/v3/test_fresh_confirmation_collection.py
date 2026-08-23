from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from a1.vla.dynamic_compute.rollout_identity import resolve_policy_episode_id
from a1.vla.dynamic_compute.v3 import fresh_confirmation_collection as d8c


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_d8c_prerequisites_and_fresh_state_suite_are_exact() -> None:
    audit = d8c.validate_d8c_prerequisites(REPO_ROOT)
    schedule, states = d8c.load_fresh_states(REPO_ROOT)
    assert audit["authorization"] == "D8C_PROSPECTIVE_SHADOW_COLLECTION_AND_REPLAY"
    assert len(schedule) == 200
    assert set(states) == set(range(10))
    assert all(len(values) == 20 for values in states.values())

    class FakeSuite:
        label = "forwarded"

    suite = d8c.FreshStateTaskSuite(FakeSuite(), states)
    observed = suite.get_task_init_states(3)
    assert suite.label == "forwarded"
    assert len(observed) == 20
    assert all(state.dtype == np.float64 and state.ndim == 1 for state in observed)
    observed[0][0] += 1.0
    assert not np.array_equal(observed[0], suite.get_task_init_states(3)[0])


def test_fresh_identity_never_aliases_an_official_episode() -> None:
    key = "libero_10:task4:fresh_confirm_v1:replicate17"
    assert d8c.parse_fresh_cluster_key(key) == (4, 17)
    assert d8c.validate_episode_id_override(key, task_id=4, replicate_id=17) == key
    for bad in (
        "libero_10:task4:episode40",
        "libero_10:task4:fresh_confirm_v1:replicate20",
        "libero_10:task3:fresh_confirm_v1:replicate17",
    ):
        with pytest.raises(d8c.D8CCollectionError):
            d8c.validate_episode_id_override(bad, task_id=4, replicate_id=17)


def test_rollout_identity_preserves_legacy_default_and_accepts_fresh_override() -> None:
    assert (
        resolve_policy_episode_id("libero_10", 2, 7)
        == "libero_10:task2:episode7"
    )
    key = "libero_10:task2:fresh_confirm_v1:replicate7"
    assert resolve_policy_episode_id("libero_10", 2, 7, key) == key
    assert all(f"episode{index}" not in key for index in range(40, 50))
    with pytest.raises(ValueError, match="nonempty"):
        resolve_policy_episode_id("libero_10", 2, 7, "")


def test_rollout_source_uses_one_resolved_identity_for_all_observers() -> None:
    source = (
        REPO_ROOT / "robot_experiments/libero/eval_libero_early_exit.py"
    ).read_text(encoding="utf-8")
    assert "episode_id = resolve_policy_episode_id(" in source
    assert source.count('"episode_id": episode_id') == 1
    assert "episode_id_override: Optional[str] = None" in source
