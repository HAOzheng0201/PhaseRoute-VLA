from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from a1.vla.dynamic_compute.route_first_active_protocol import (
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
)
from scripts.aggregate_route_first_stage9_pilot import aggregate_pairs


REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_SCHEMA = "phase-route-vla.route-first-stage9-pilot-arm.v1"
PAIR_SCHEMA = "phase-route-vla.route-first-stage9-pilot-task-pair.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _write_measurements(path: Path, values: tuple[float, float]) -> None:
    path.write_text(
        "".join(
            json.dumps({"policy_wall_latency_ms": value}) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _arm(run_dir: Path, method: str, values: tuple[float, float]) -> str:
    run_dir.mkdir()
    measurement = run_dir / "stage1_measurement.jsonl"
    _write_measurements(measurement, values)
    attestation = {
        "schema_version": ARM_SCHEMA,
        "status": "PASS",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "method": method,
        "artifacts": {
            measurement.name: {
                "bytes": measurement.stat().st_size,
                "sha256": _sha(measurement),
            }
        },
    }
    path = run_dir / "stage9_pilot_arm_attestation.json"
    _write_json(path, attestation)
    return _sha(path)


def _pair_grid(
    tmp_path: Path,
    *,
    route_successes: int = 10,
    route_values: tuple[float, float] = (70.0, 80.0),
) -> list[Path]:
    paths = []
    for task_id in range(10):
        candidate_dir = tmp_path / f"task{task_id}_candidate"
        route_dir = tmp_path / f"task{task_id}_route"
        candidate_sha = _arm(
            candidate_dir, "candidate_first_v3", (100.0, 110.0)
        )
        route_sha = _arm(route_dir, "route_first_stage8", route_values)
        candidate_result = {
            "success": True,
            "policy_calls": 2,
            "policy_wall_latency_ms": {"p50": 100.0, "mean": 105.0},
        }
        route_success = task_id < route_successes
        route_result = {
            "success": route_success,
            "policy_calls": 2,
            "policy_wall_latency_ms": {
                "p50": min(route_values),
                "mean": sum(route_values) / 2,
            },
            "route_exact_fm_invocations": 2,
            "route_decoder_blocks": 54,
        }
        pair = {
            "schema_version": PAIR_SCHEMA,
            "status": "PASS",
            "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
            "task_id": task_id,
            "episode_index": 13,
            "arm_order": (
                ["candidate_first_v3", "route_first_stage8"]
                if task_id % 2 == 0
                else ["route_first_stage8", "candidate_first_v3"]
            ),
            "paired_outcome": (
                "both_success" if route_success else "candidate_only_success"
            ),
            "candidate_first": candidate_result,
            "route_first": route_result,
            "checks": {"sealed_pair": True},
            "run_dirs": {
                "candidate_first": str(candidate_dir.resolve()),
                "route_first": str(route_dir.resolve()),
            },
            "input_attestations": {
                "candidate_first": {
                    "path": str(
                        candidate_dir.resolve()
                        / "stage9_pilot_arm_attestation.json"
                    ),
                    "sha256": candidate_sha,
                },
                "route_first": {
                    "path": str(
                        route_dir.resolve()
                        / "stage9_pilot_arm_attestation.json"
                    ),
                    "sha256": route_sha,
                },
            },
        }
        path = tmp_path / f"task{task_id}_pair.json"
        _write_json(path, pair)
        paths.append(path)
    return paths


def test_global_pilot_passes_frozen_success_latency_and_fm_gates(
    tmp_path: Path,
) -> None:
    result = aggregate_pairs(_pair_grid(tmp_path))
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["candidate_first"]["successes"] == 10
    assert result["route_first"]["successes"] == 10
    assert result["descriptive_comparison"][
        "policy_wall_median_ratio_route_to_candidate"
    ] == pytest.approx(0.7)


def test_global_pilot_retains_failures_and_applies_success_guardrail(
    tmp_path: Path,
) -> None:
    result = aggregate_pairs(_pair_grid(tmp_path, route_successes=7))
    assert result["status"] == "FAIL"
    assert result["route_first"]["successes"] == 7
    assert result["checks"]["route_success_guardrail"] is False
    assert result["next_gate"]["fresh_state_confirmation_authorized"] is False


def test_global_pilot_applies_pooled_p50_latency_gate(tmp_path: Path) -> None:
    result = aggregate_pairs(
        _pair_grid(tmp_path, route_values=(95.0, 100.0))
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["route_median_latency_ratio_at_most_0_90"] is False


def test_global_pilot_rejects_raw_measurement_drift(tmp_path: Path) -> None:
    pairs = _pair_grid(tmp_path)
    run_dir = tmp_path / "task0_route"
    with (run_dir / "stage1_measurement.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps({"policy_wall_latency_ms": 1.0}) + "\n")
    with pytest.raises(ValueError, match="raw artifact drift"):
        aggregate_pairs(pairs)


def test_state13_launchers_preserve_frozen_schedule_and_attestors() -> None:
    candidate = (
        REPO_ROOT / "scripts/run_libero_route_first_stage9_pilot_candidate.sh"
    ).read_text(encoding="utf-8")
    route = (
        REPO_ROOT / "scripts/run_libero_route_first_stage9_pilot_route.sh"
    ).read_text(encoding="utf-8")
    task = (
        REPO_ROOT / "scripts/run_libero_route_first_stage9_pilot_task.sh"
    ).read_text(encoding="utf-8")
    assert 'episode_index="${EPISODE_INDEX:-13}"' in candidate
    assert 'episode_index="${EPISODE_INDEX:-13}"' in route
    assert 'seed="${SEED:-20260826}"' in candidate
    assert 'seed="${SEED:-20260826}"' in route
    assert '--output "${run_dir}/preflight.json"' in candidate
    assert 'ln "${run_dir}/stage9_preflight.json" "${run_dir}/preflight.json"' in route
    assert "validate_phase_route_v3_run.py" in candidate
    assert "validate_route_first_active_run.py" in route
    assert "validate_route_first_stage9_pilot_arm.py" in candidate
    assert "validate_route_first_stage9_pilot_arm.py" in route
    assert "task_id % 2" in task
    assert "summarize_route_first_stage9_pilot_task.py" in task
