from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_route_first_stage11b_profile import (
    PROTOCOL_SCHEMA,
    _normalize_uuid,
    _parse_task_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stage11b_protocol_uses_only_previously_opened_development_state() -> None:
    protocol = json.loads(
        (REPO_ROOT / "configs/research/route_first_stage11b_profile_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["schema_version"] == PROTOCOL_SCHEMA
    assert protocol["status"] == "FROZEN_DEVELOPMENT_PROFILE_NOT_RUN"
    assert protocol["data_role"] == {
        "suite": "libero_10",
        "official_init_state": 0,
        "state_role": "previously_opened_development_state",
        "final_stage10_fresh_states_used": False,
        "threshold_or_router_training_allowed_in_stage11b": False,
    }
    shards = protocol["schedule"]["full_shards"]
    flattened = [task for name in sorted(shards) for task in shards[name]]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))
    assert protocol["method"]["controller_or_router_change"] is False
    assert protocol["method"]["measurement_is_control_input"] is False


def test_task_parser_and_gpu_uuid_normalization_are_strict() -> None:
    assert _parse_task_ids("0,4,8") == (0, 4, 8)
    with pytest.raises(Exception, match="unique"):
        _parse_task_ids("0,0")
    with pytest.raises(Exception, match="0..9"):
        _parse_task_ids("10")
    assert _normalize_uuid("GPU-ABC") == "abc"
    assert _normalize_uuid("abc") == "abc"


def test_runner_reuses_stage10_sparse_controller_constructor() -> None:
    source = (
        REPO_ROOT / "scripts/run_route_first_stage11b_profile.py"
    ).read_text(encoding="utf-8")
    assert "from scripts.run_route_first_stage10_arm import _sparse_controller" in source
    assert "base_controller = _sparse_controller(cfg, model, device)" in source
    assert "initialize_exit_controller" not in source
