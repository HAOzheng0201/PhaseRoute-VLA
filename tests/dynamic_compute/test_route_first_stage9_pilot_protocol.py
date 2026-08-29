from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from a1.vla.dynamic_compute.route_first_active_protocol import (
    ROUTE_FIRST_ACTIVE_BASE_SEED,
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
)
from a1.vla.dynamic_compute.route_first_stage9_pilot_protocol import (
    STAGE9_CANDIDATE_METHOD,
    STAGE9_PILOT_EPISODE_INDEX,
    STAGE9_ROUTE_FIRST_METHOD,
    STAGE9_STATE12_GATE_SHA256,
    Stage9PilotProtocolError,
    authorize_stage9_pilot_arm,
    expected_stage9_pilot_arm_order,
    load_stage9_state12_unlock,
    sha256_file,
)
from scripts.validate_route_first_stage9_pilot_prelaunch import validate_prelaunch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPO_ROOT / "configs/route_first_active_pilot_protocol.json"
GPU_UUID = "GPU-535e41e1-a1ac-af65-a015-fc281644709e"


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _stage9_preflight() -> dict[str, object]:
    physical = {"index": 4, "uuid": GPU_UUID}
    return {
        "schema_version": "phase-route-vla.route-first-active-preflight.v1",
        "status": "PASS",
        "scope": "route_first_stage9_active_preflight",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "simulator_episode_opened": False,
        "repo_root": str(REPO_ROOT),
        "git_commit": _head(),
        "worktree_dirty": False,
        "physical_gpu_index": 4,
        "expected_gpu_uuid": GPU_UUID,
        "physical_gpu_before_cuda_smoke": physical,
        "physical_gpu_after_cuda_smoke": physical,
        "checks": {"all_frozen_inputs_exact": True},
    }


def _v3_preflight() -> dict[str, object]:
    return {
        "schema_version": "phase-route-vla.v3.preflight.v1",
        "status": "PASS",
        "scope": "phase_route_v3_release_preflight",
        "repo_root": str(REPO_ROOT),
        "git_commit": _head(),
        "worktree_dirty": False,
        "physical_gpu_index": 4,
        "expected_gpu_uuid": GPU_UUID,
        "checks": {"release_exact": True},
    }


def _args(
    tmp_path: Path,
    *,
    method: str,
    task_id: int,
    arm_position: int,
    with_v3: bool,
) -> argparse.Namespace:
    stage9_path = tmp_path / f"stage9_{method}.json"
    _write_json(stage9_path, _stage9_preflight())
    v3_path = tmp_path / f"v3_{method}.json"
    if with_v3:
        _write_json(v3_path, _v3_preflight())
    return argparse.Namespace(
        repo_root=REPO_ROOT,
        protocol=PROTOCOL,
        method=method,
        task_id=task_id,
        episode_index=STAGE9_PILOT_EPISODE_INDEX,
        arm_position=arm_position,
        seed=ROUTE_FIRST_ACTIVE_BASE_SEED,
        physical_gpu_index=4,
        expected_gpu_uuid=GPU_UUID,
        stage9_preflight=stage9_path,
        v3_preflight=v3_path if with_v3 else None,
        output=tmp_path / "unused.json",
    )


def test_state12_unlock_is_exact_and_authorizes_state13_only() -> None:
    gate_path = REPO_ROOT / "results/route_first/route_first_stage9_state12_pair.json"
    gate = load_stage9_state12_unlock(REPO_ROOT)
    assert sha256_file(gate_path) == STAGE9_STATE12_GATE_SHA256
    assert gate["status"] == "PASS"
    assert gate["next_gate"]["state13_pilot_protocol_gate_unlocked"] is True
    selection, _, loaded_gate = authorize_stage9_pilot_arm(
        repo_root=REPO_ROOT,
        protocol_path=PROTOCOL,
        method=STAGE9_CANDIDATE_METHOD,
        task_id=0,
        episode_index=13,
        arm_position=1,
        seed=ROUTE_FIRST_ACTIVE_BASE_SEED,
    )
    assert selection.episode_seed == 20260839
    assert loaded_gate == gate


@pytest.mark.parametrize("task_id", range(10))
def test_pilot_order_alternates_by_task(task_id: int) -> None:
    expected = (
        (STAGE9_CANDIDATE_METHOD, STAGE9_ROUTE_FIRST_METHOD)
        if task_id % 2 == 0
        else (STAGE9_ROUTE_FIRST_METHOD, STAGE9_CANDIDATE_METHOD)
    )
    assert expected_stage9_pilot_arm_order(task_id) == expected


@pytest.mark.parametrize(
    ("episode_index", "arm_position", "seed"),
    [(12, 1, 20260826), (13, 2, 20260826), (13, 1, 20260827)],
)
def test_protocol_rejects_state_order_or_seed_drift(
    episode_index: int, arm_position: int, seed: int
) -> None:
    with pytest.raises(Stage9PilotProtocolError):
        authorize_stage9_pilot_arm(
            repo_root=REPO_ROOT,
            protocol_path=PROTOCOL,
            method=STAGE9_CANDIDATE_METHOD,
            task_id=0,
            episode_index=episode_index,
            arm_position=arm_position,
            seed=seed,
        )


def test_prelaunch_binds_candidate_and_route_evidence(tmp_path: Path) -> None:
    candidate = validate_prelaunch(
        _args(
            tmp_path,
            method=STAGE9_CANDIDATE_METHOD,
            task_id=0,
            arm_position=1,
            with_v3=True,
        )
    )
    route = validate_prelaunch(
        _args(
            tmp_path,
            method=STAGE9_ROUTE_FIRST_METHOD,
            task_id=0,
            arm_position=2,
            with_v3=False,
        )
    )
    assert candidate["status"] == route["status"] == "PASS"
    assert all(candidate["checks"].values())
    assert all(route["checks"].values())
    assert candidate["simulator_episode_opened"] is False


def test_prelaunch_fails_closed_on_v3_gpu_drift(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        method=STAGE9_CANDIDATE_METHOD,
        task_id=0,
        arm_position=1,
        with_v3=True,
    )
    value = json.loads(args.v3_preflight.read_text(encoding="utf-8"))
    value["expected_gpu_uuid"] = "GPU-other"
    _write_json(args.v3_preflight, value)
    result = validate_prelaunch(args)
    assert result["status"] == "FAIL"
    assert result["checks"]["candidate_v3_gpu_uuid"] is False
