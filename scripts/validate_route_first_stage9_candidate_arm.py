#!/usr/bin/env python3
"""Bind one candidate-first V3 arm to the frozen Stage-9 smoke protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_active_protocol import (  # noqa: E402
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
    load_route_first_active_protocol,
)
from scripts.validate_phase_route_v3_run import validate_run as validate_v3_run  # noqa: E402


CANDIDATE_METHOD = "candidate_first_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _normalize_uuid(value: Any) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def validate_candidate_arm(
    run_dir: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    repository = Path(repo_root).resolve(strict=True)
    protocol = load_route_first_active_protocol(
        repository / "configs/route_first_active_pilot_protocol.json",
        repository,
    )
    paths = {
        "stage9_preflight.json": root / "stage9_preflight.json",
        "preflight.json": root / "preflight.json",
        "evaluation_summary.json": root / "evaluation_summary.json",
        "run_attestation.json": root / "run_attestation.json",
        "phase_route_runtime.jsonl": root / "phase_route_runtime.jsonl",
        "policy_telemetry.jsonl": root / "policy_telemetry.jsonl",
        "stage1_measurement.jsonl": root / "stage1_measurement.jsonl",
        "stdout.log": root / "stdout.log",
        "command.sh": root / "command.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing candidate-arm artifacts: {missing}")

    stage9_preflight = _load_object(paths["stage9_preflight.json"])
    v3_preflight = _load_object(paths["preflight.json"])
    evaluation = _load_object(paths["evaluation_summary.json"])
    stored_v3_attestation = _load_object(paths["run_attestation.json"])
    computed_v3_attestation = validate_v3_run(root)
    runtime = evaluation.get("runtime")
    measurement = evaluation.get("stage1_measurement")
    smoke = protocol.get("schedule", {}).get("engineering_smoke", {})
    preflight_uuid = stage9_preflight.get("expected_gpu_uuid")
    v3_cuda = v3_preflight.get("cuda", {})
    checks = {
        "protocol_exact": stage9_preflight.get("protocol_sha256")
        == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "stage9_preflight_pass": stage9_preflight.get("status") == "PASS",
        "stage9_preflight_scope": stage9_preflight.get("scope")
        == "route_first_stage9_active_preflight",
        "stage9_preflight_no_episode": stage9_preflight.get("simulator_episode_opened")
        is False,
        "v3_preflight_pass": v3_preflight.get("status") == "PASS",
        "v3_preflight_scope": v3_preflight.get("scope")
        == "phase_route_v3_release_preflight",
        "gpu_uuid_same_across_preflights": bool(preflight_uuid)
        and _normalize_uuid(preflight_uuid)
        == _normalize_uuid(v3_preflight.get("expected_gpu_uuid"))
        == _normalize_uuid(v3_cuda.get("visible_uuid")),
        "candidate_is_first_preregistered_arm": isinstance(smoke, Mapping)
        and tuple(smoke.get("arm_order", ()))
        == (CANDIDATE_METHOD, "route_first_stage8"),
        "task_is_frozen_smoke_task": evaluation.get("task_ids") == [0]
        and smoke.get("task_ids") == [0],
        "state_is_frozen_smoke_state": evaluation.get("episode_indices") == [12]
        and smoke.get("episode_indices") == [12],
        "base_seed_frozen": evaluation.get("seed_base") == 20260826,
        "evaluation_method": evaluation.get("method") == "phase_route_v3",
        "one_episode": evaluation.get("total_episodes") == 1,
        "v3_attestation_stored_pass": stored_v3_attestation.get("status") == "PASS",
        "v3_attestation_recomputed_pass": computed_v3_attestation.get("status")
        == "PASS",
        "runtime_error_free": isinstance(runtime, Mapping)
        and runtime.get("error_count") == 0
        and runtime.get("records_with_errors") == 0
        and runtime.get("policy_calls") == runtime.get("prepared_calls")
        == runtime.get("committed_calls"),
        "measurement_complete": isinstance(measurement, Mapping)
        and isinstance(runtime, Mapping)
        and measurement.get("records") == runtime.get("policy_calls")
        and measurement.get("records_with_errors") == 0
        and measurement.get("records_with_nonfinite_actions") == 0
        and measurement.get("records_without_action_audit") == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "phase-route-vla.route-first-stage9-candidate-arm.v1",
        "status": status,
        "scope": "stage9_engineering_smoke_candidate_first_arm",
        "method": CANDIDATE_METHOD,
        "arm_position": 1,
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "run_dir": str(root),
        "gpu_uuid": preflight_uuid,
        "result": {
            "task_ids": evaluation.get("task_ids"),
            "episode_indices": evaluation.get("episode_indices"),
            "total_successes": evaluation.get("total_successes"),
            "policy_calls": runtime.get("policy_calls") if isinstance(runtime, Mapping) else None,
            "selected_layers": runtime.get("selected_layers")
            if isinstance(runtime, Mapping)
            else None,
            "latency_ms": measurement.get("latency_ms")
            if isinstance(measurement, Mapping)
            else None,
        },
        "artifacts": {
            name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = run_dir / "stage9_arm_attestation.json"
    temporary = output_path.with_name(output_path.name + ".incomplete")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    result = validate_candidate_arm(run_dir, repo_root=args.repo_root)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(output_path)
    print(output, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
