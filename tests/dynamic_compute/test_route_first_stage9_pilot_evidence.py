from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from a1.vla.dynamic_compute.route_first_active_protocol import (
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
)
from scripts.summarize_route_first_stage9_pilot_task import summarize_dirs
from scripts.validate_route_first_stage9_pilot_arm import validate_arm
from scripts.validate_route_first_stage9_pilot_prelaunch import validate_prelaunch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPO_ROOT / "configs/route_first_active_pilot_protocol.json"
GPU_UUID = "GPU-535e41e1-a1ac-af65-a015-fc281644709e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value, allow_nan=False) + "\n" for value in values),
        encoding="utf-8",
    )


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


def _inventory(run_dir: Path, filenames: tuple[str, ...]) -> dict[str, object]:
    return {
        name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha(run_dir / name)}
        for name in filenames
    }


def _arm_fixture(
    tmp_path: Path,
    *,
    method: str,
    task_id: int = 0,
    exact_route_fm: bool = True,
) -> Path:
    candidate = method == "candidate_first_v3"
    arm_position = (1 if task_id % 2 == 0 else 2) if candidate else (
        2 if task_id % 2 == 0 else 1
    )
    run_dir = tmp_path / f"task{task_id}_{method}"
    run_dir.mkdir()
    stage9_path = run_dir / "stage9_preflight.json"
    _write_json(stage9_path, _stage9_preflight())
    preflight_path = run_dir / "preflight.json"
    _write_json(preflight_path, _v3_preflight() if candidate else _stage9_preflight())
    args = argparse.Namespace(
        repo_root=REPO_ROOT,
        protocol=PROTOCOL,
        method=method,
        task_id=task_id,
        episode_index=13,
        arm_position=arm_position,
        seed=20260826,
        physical_gpu_index=4,
        expected_gpu_uuid=GPU_UUID,
        stage9_preflight=stage9_path,
        v3_preflight=preflight_path if candidate else None,
        output=run_dir / "unused.json",
    )
    prelaunch = validate_prelaunch(args)
    assert prelaunch["status"] == "PASS"
    _write_json(run_dir / "prelaunch.json", prelaunch)

    calls = 2
    layers = {"11": 0, "13": 1, "27": 1}
    runtime = {
        "records": calls,
        "policy_calls": calls,
        "prepared_calls": calls,
        "committed_calls": calls,
        "error_count": 0,
        "records_with_errors": 0,
        "selected_layers": layers,
    }
    if not candidate:
        runtime["route_first_integrity"] = {
            "valid_calls_with_exactly_one_fm": calls if exact_route_fm else 1,
            "fm_invocations": calls if exact_route_fm else 1,
            "decoder_blocks": 54,
            "calls_with_route_errors": 0,
        }
    evaluation = {
        "schema_version": (
            "phase-route-vla.libero-evaluation-summary.v1"
            if candidate
            else "phase-route-vla.route-first-active-evaluation.v1"
        ),
        "method": "phase_route_v3" if candidate else "route_first_stage8",
        "experiment_stage": None if candidate else "paired_pilot",
        "arm_position": None if candidate else arm_position,
        "suite": "libero_10",
        "task_ids": [task_id],
        "episode_indices": [13],
        "seed_base": 20260826,
        "total_episodes": 1,
        "total_successes": 1,
        "episodes": [
            {
                "task_id": task_id,
                "episode_index": 13,
                "seed": 20260839 + task_id * 10_000,
                "success": True,
                "wall_seconds": 10.0,
            }
        ],
        "runtime": runtime,
        "stage1_measurement": {
            "records": calls,
            "records_with_errors": 0,
            "records_with_nonfinite_actions": 0,
            "records_without_action_audit": 0,
        },
    }
    if not candidate:
        evaluation.update(
            {
                "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
                "gpu": {
                    "physical_index": 4,
                    "expected_uuid": GPU_UUID,
                    "visible_uuid": GPU_UUID.removeprefix("GPU-"),
                },
            }
        )
    _write_json(run_dir / "evaluation_summary.json", evaluation)
    _write_jsonl(
        run_dir / "phase_route_runtime.jsonl",
        [{"selected_layer": 13}, {"selected_layer": 27}],
    )
    _write_jsonl(run_dir / "policy_telemetry.jsonl", [{"call": 0}, {"call": 1}])
    mode = "phase_route_v3" if candidate else "route_first_stage8"
    _write_jsonl(
        run_dir / "stage1_measurement.jsonl",
        [
            {
                "mode": mode,
                "context": {"task_id": task_id},
                "selected_layer": 13,
                "policy_wall_latency_ms": 100.0 if candidate else 70.0,
                "action_finite": True,
                "error": None,
            },
            {
                "mode": mode,
                "context": {"task_id": task_id},
                "selected_layer": 27,
                "policy_wall_latency_ms": 110.0 if candidate else 80.0,
                "action_finite": True,
                "error": None,
            },
        ],
    )
    (run_dir / "stdout.log").write_text("complete\n", encoding="utf-8")
    (run_dir / "command.sh").write_text("frozen command\n", encoding="utf-8")
    _write_json(
        run_dir / "gpu_postflight.json",
        {
            "schema_version": (
                "phase-route-vla.route-first-stage9-pilot-gpu-postflight.v1"
            ),
            "status": "PASS",
            "expected_gpu_uuid": GPU_UUID,
            "physical_gpu": {"index": 4, "uuid": GPU_UUID},
            "compute_processes": [],
            "checks": {"physical_uuid_matches": True, "no_process": True},
        },
    )
    existing_files = (
        "preflight.json",
        "evaluation_summary.json",
        "phase_route_runtime.jsonl",
        "policy_telemetry.jsonl",
        "stage1_measurement.jsonl",
        "stdout.log",
    ) + (("command.sh",) if not candidate else ())
    _write_json(
        run_dir / "run_attestation.json",
        {
            "schema_version": (
                "phase-route-vla.v3.general-run-attestation.v1"
                if candidate
                else "phase-route-vla.route-first-active-attestation.v1"
            ),
            "status": "PASS",
            "protocol_sha256": (
                None if candidate else ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256
            ),
            "run_dir": str(run_dir.resolve()),
            "checks": {"original_attestor_pass": True},
            "artifacts": _inventory(run_dir, existing_files),
        },
    )
    return run_dir


@pytest.mark.parametrize("method", ["candidate_first_v3", "route_first_stage8"])
def test_pilot_arm_seals_complete_original_attestation_chain(
    tmp_path: Path, method: str
) -> None:
    run_dir = _arm_fixture(tmp_path, method=method)
    result = validate_arm(run_dir, repo_root=REPO_ROOT)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["protocol_sha256"] == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256


def test_pilot_arm_rejects_raw_artifact_drift_after_original_attestation(
    tmp_path: Path,
) -> None:
    run_dir = _arm_fixture(tmp_path, method="candidate_first_v3")
    with (run_dir / "stage1_measurement.jsonl").open("a", encoding="utf-8") as output:
        output.write("\n")
    result = validate_arm(run_dir, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert result["checks"]["existing_attestation_artifacts_exact"] is False


def test_route_arm_rejects_less_than_one_fm_per_call(tmp_path: Path) -> None:
    run_dir = _arm_fixture(
        tmp_path, method="route_first_stage8", exact_route_fm=False
    )
    result = validate_arm(run_dir, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert result["checks"]["route_every_call_exactly_one_fm"] is False


def test_task_pair_verifies_both_sealed_arms_and_order(tmp_path: Path) -> None:
    candidate_dir = _arm_fixture(tmp_path, method="candidate_first_v3")
    route_dir = _arm_fixture(tmp_path, method="route_first_stage8")
    for run_dir in (candidate_dir, route_dir):
        _write_json(
            run_dir / "stage9_pilot_arm_attestation.json",
            validate_arm(run_dir, repo_root=REPO_ROOT),
        )
    result = summarize_dirs(candidate_dir, route_dir)
    assert result["status"] == "PASS"
    assert result["arm_order"] == ["candidate_first_v3", "route_first_stage8"]
    assert result["descriptive_comparison"][
        "policy_wall_p50_ratio_route_to_candidate"
    ] == pytest.approx(0.7)


def test_task_pair_rejects_artifact_drift_after_pilot_seal(tmp_path: Path) -> None:
    candidate_dir = _arm_fixture(tmp_path, method="candidate_first_v3")
    route_dir = _arm_fixture(tmp_path, method="route_first_stage8")
    for run_dir in (candidate_dir, route_dir):
        _write_json(
            run_dir / "stage9_pilot_arm_attestation.json",
            validate_arm(run_dir, repo_root=REPO_ROOT),
        )
    (route_dir / "stdout.log").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact mismatch"):
        summarize_dirs(candidate_dir, route_dir)
