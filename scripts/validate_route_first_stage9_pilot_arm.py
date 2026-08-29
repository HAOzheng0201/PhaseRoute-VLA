#!/usr/bin/env python3
"""Seal one completed candidate-first or route-first state-13 pilot arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_stage9_pilot_protocol import (  # noqa: E402
    STAGE9_CANDIDATE_METHOD,
    STAGE9_ROUTE_FIRST_METHOD,
    authorize_stage9_pilot_arm,
)
from a1.vla.dynamic_compute.route_first_active_protocol import (  # noqa: E402
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
)


SCHEMA = "phase-route-vla.route-first-stage9-pilot-arm.v1"
PRELAUNCH_SCHEMA = "phase-route-vla.route-first-stage9-pilot-prelaunch.v1"
POSTFLIGHT_SCHEMA = "phase-route-vla.route-first-stage9-pilot-gpu-postflight.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return tuple(rows)


def normalize_uuid(value: Any) -> str:
    result = str(value).strip().lower()
    return result[4:] if result.startswith("gpu-") else result


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        float(record["policy_wall_latency_ms"])
        for record in records
        if isinstance(record.get("policy_wall_latency_ms"), (int, float))
        and not isinstance(record.get("policy_wall_latency_ms"), bool)
        and math.isfinite(float(record["policy_wall_latency_ms"]))
        and float(record["policy_wall_latency_ms"]) > 0.0
    ]
    if len(values) != len(records) or not values:
        raise ValueError("pilot measurements contain invalid policy latency")
    return {
        "count": len(values),
        "sum": math.fsum(values),
        "mean": statistics.fmean(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": max(values),
    }


def _expected_existing_attestation_schema(method: str) -> str:
    if method == STAGE9_CANDIDATE_METHOD:
        return "phase-route-vla.v3.general-run-attestation.v1"
    if method == STAGE9_ROUTE_FIRST_METHOD:
        return "phase-route-vla.route-first-active-attestation.v1"
    raise ValueError(f"unknown pilot method: {method}")


def _attested_artifacts_are_exact(
    attestation: Mapping[str, Any], root: Path
) -> bool:
    artifacts = attestation.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return False
    for filename, expected in artifacts.items():
        if not isinstance(filename, str) or not isinstance(expected, Mapping):
            return False
        path = root / filename
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            return False
    return True


def _prelaunch_artifact_is_exact(
    prelaunch: Mapping[str, Any], name: str, path: Path
) -> bool:
    artifacts = prelaunch.get("artifacts")
    metadata = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    return bool(
        isinstance(metadata, Mapping)
        and path.is_file()
        and sha256_file(path) == metadata.get("sha256")
    )


def validate_arm(
    run_dir: str | Path, *, repo_root: str | Path = REPO_ROOT
) -> dict[str, Any]:
    root = Path(run_dir).resolve(strict=True)
    repository = Path(repo_root).resolve(strict=True)
    paths = {
        "prelaunch.json": root / "prelaunch.json",
        "preflight.json": root / "preflight.json",
        "gpu_postflight.json": root / "gpu_postflight.json",
        "evaluation_summary.json": root / "evaluation_summary.json",
        "run_attestation.json": root / "run_attestation.json",
        "phase_route_runtime.jsonl": root / "phase_route_runtime.jsonl",
        "policy_telemetry.jsonl": root / "policy_telemetry.jsonl",
        "stage1_measurement.jsonl": root / "stage1_measurement.jsonl",
        "stdout.log": root / "stdout.log",
        "command.sh": root / "command.sh",
        "stage9_preflight.json": root / "stage9_preflight.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pilot arm artifacts: {missing}")
    prelaunch = load_object(paths["prelaunch.json"])
    postflight = load_object(paths["gpu_postflight.json"])
    evaluation = load_object(paths["evaluation_summary.json"])
    existing_attestation = load_object(paths["run_attestation.json"])
    measurements = load_jsonl(paths["stage1_measurement.jsonl"])
    telemetry = load_jsonl(paths["policy_telemetry.jsonl"])
    runtime_records = load_jsonl(paths["phase_route_runtime.jsonl"])
    selection = prelaunch.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("pilot prelaunch selection is missing")
    method = str(selection.get("method"))
    task_id = int(selection.get("task_id", -1))
    episode_index = int(selection.get("episode_index", -1))
    arm_position = int(selection.get("arm_position", -1))
    seed = int(selection.get("base_seed", -1))
    authorized, _, _ = authorize_stage9_pilot_arm(
        repo_root=repository,
        protocol_path=repository / "configs/route_first_active_pilot_protocol.json",
        method=method,
        task_id=task_id,
        episode_index=episode_index,
        arm_position=arm_position,
        seed=seed,
    )
    episodes = evaluation.get("episodes")
    runtime = evaluation.get("runtime")
    measurement_summary = evaluation.get("stage1_measurement")
    if (
        not isinstance(episodes, list)
        or len(episodes) != 1
        or not isinstance(episodes[0], Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(measurement_summary, Mapping)
    ):
        raise ValueError("pilot evaluation evidence is incomplete")
    episode = episodes[0]
    calls = int(runtime.get("policy_calls", -1))
    selected = runtime.get("selected_layers")
    if not isinstance(selected, Mapping):
        raise ValueError("pilot selected-layer counts are missing")
    layers = {str(layer): int(selected.get(str(layer), 0)) for layer in (11, 13, 27)}
    latencies = latency_summary(measurements)
    expected_evaluation_schema = (
        "phase-route-vla.libero-evaluation-summary.v1"
        if method == STAGE9_CANDIDATE_METHOD
        else "phase-route-vla.route-first-active-evaluation.v1"
    )
    expected_evaluation_method = (
        "phase_route_v3"
        if method == STAGE9_CANDIDATE_METHOD
        else STAGE9_ROUTE_FIRST_METHOD
    )
    expected_mode = expected_evaluation_method
    evaluation_gpu = evaluation.get("gpu", {})
    expected_gpu_uuid = prelaunch.get("expected_gpu_uuid")
    route_integrity = runtime.get("route_first_integrity", {})
    postflight_gpu = postflight.get("physical_gpu")
    stage9_artifact_exact = _prelaunch_artifact_is_exact(
        prelaunch, "stage9_preflight", paths["stage9_preflight.json"]
    )
    v3_artifact_exact = bool(
        method != STAGE9_CANDIDATE_METHOD
        or _prelaunch_artifact_is_exact(
            prelaunch, "v3_preflight", paths["preflight.json"]
        )
    )
    route_exact_fm = bool(
        method != STAGE9_ROUTE_FIRST_METHOD
        or (
            isinstance(route_integrity, Mapping)
            and route_integrity.get("valid_calls_with_exactly_one_fm") == calls
            and route_integrity.get("fm_invocations") == calls
            and route_integrity.get("calls_with_route_errors") == 0
        )
    )
    checks = {
        "prelaunch_pass": prelaunch.get("schema_version") == PRELAUNCH_SCHEMA
        and prelaunch.get("status") == "PASS"
        and prelaunch.get("protocol_sha256") == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256
        and prelaunch.get("simulator_episode_opened") is False
        and prelaunch.get("state13_open_authorized_for_this_arm") is True
        and isinstance(prelaunch.get("checks"), Mapping)
        and bool(prelaunch.get("checks"))
        and all(value is True for value in prelaunch["checks"].values()),
        "prelaunch_inputs_unchanged": stage9_artifact_exact and v3_artifact_exact,
        "selection_preregistered": authorized.task_id == task_id,
        "postflight_pass": postflight.get("schema_version") == POSTFLIGHT_SCHEMA
        and postflight.get("status") == "PASS",
        "postflight_no_compute_process": not postflight.get("compute_processes"),
        "postflight_gpu_identity": bool(
            isinstance(postflight_gpu, Mapping)
            and postflight_gpu.get("index") == prelaunch.get("physical_gpu_index")
            and normalize_uuid(postflight_gpu.get("uuid"))
            == normalize_uuid(expected_gpu_uuid)
            and isinstance(postflight.get("checks"), Mapping)
            and bool(postflight.get("checks"))
            and all(value is True for value in postflight["checks"].values())
        ),
        "existing_attestation_pass": existing_attestation.get("schema_version")
        == _expected_existing_attestation_schema(method)
        and existing_attestation.get("status") == "PASS"
        and isinstance(existing_attestation.get("checks"), Mapping)
        and bool(existing_attestation.get("checks"))
        and all(value is True for value in existing_attestation["checks"].values()),
        "existing_attestation_identity": existing_attestation.get("run_dir")
        == str(root)
        and (
            method != STAGE9_ROUTE_FIRST_METHOD
            or existing_attestation.get("protocol_sha256")
            == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256
        ),
        "existing_attestation_artifacts_exact": _attested_artifacts_are_exact(
            existing_attestation, root
        ),
        "evaluation_schema": evaluation.get("schema_version")
        == expected_evaluation_schema,
        "evaluation_method": evaluation.get("method") == expected_evaluation_method,
        "evaluation_protocol": method != STAGE9_ROUTE_FIRST_METHOD
        or evaluation.get("protocol_sha256") == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "pilot_stage": (
            evaluation.get("experiment_stage") == "paired_pilot"
            if method == STAGE9_ROUTE_FIRST_METHOD
            else True
        ),
        "arm_position": (
            evaluation.get("arm_position") == arm_position
            if method == STAGE9_ROUTE_FIRST_METHOD
            else True
        ),
        "task_state_seed": evaluation.get("task_ids") == [task_id]
        and evaluation.get("episode_indices") == [episode_index]
        and evaluation.get("seed_base") == seed
        and episode.get("task_id") == task_id
        and episode.get("episode_index") == episode_index
        and episode.get("seed") == authorized.episode_seed,
        "one_episode": evaluation.get("total_episodes") == 1,
        "success_bounded": evaluation.get("total_successes") in (0, 1)
        and bool(episode.get("success"))
        == bool(evaluation.get("total_successes")),
        "runtime_error_free": runtime.get("error_count") == 0,
        "calls_all_committed": calls > 0
        and runtime.get("prepared_calls") == calls
        and runtime.get("committed_calls") == calls,
        "route_counts_complete": sum(layers.values()) == calls,
        "runtime_records_complete": len(runtime_records) == calls
        and all(
            record.get("selected_layer") in (11, 13, 27)
            for record in runtime_records
        ),
        "measurement_complete": len(measurements) == calls
        and measurement_summary.get("records") == calls
        and measurement_summary.get("records_with_errors") == 0
        and measurement_summary.get("records_with_nonfinite_actions") == 0
        and measurement_summary.get("records_without_action_audit") == 0
        and all(
            record.get("mode") == expected_mode
            and record.get("selected_layer") in (11, 13, 27)
            and record.get("action_finite") is True
            and record.get("error") is None
            and isinstance(record.get("context"), Mapping)
            and record["context"].get("task_id") == task_id
            for record in measurements
        ),
        "telemetry_complete": len(telemetry) == calls,
        "gpu_uuid_same_before_after": normalize_uuid(expected_gpu_uuid)
        == normalize_uuid(postflight.get("expected_gpu_uuid")),
        "route_evaluation_gpu_bound": method != STAGE9_ROUTE_FIRST_METHOD
        or (
            isinstance(evaluation_gpu, Mapping)
            and evaluation_gpu.get("physical_index")
            == prelaunch.get("physical_gpu_index")
            and normalize_uuid(evaluation_gpu.get("expected_uuid"))
            == normalize_uuid(expected_gpu_uuid)
            == normalize_uuid(evaluation_gpu.get("visible_uuid"))
        ),
        "route_l11_disabled": method != STAGE9_ROUTE_FIRST_METHOD or layers["11"] == 0,
        "route_every_call_exactly_one_fm": route_exact_fm,
    }
    artifacts = {
        filename: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for filename, path in paths.items()
    }
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "stage9_state13_preregistered_pilot_arm",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "method": method,
        "arm_position": arm_position,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "run_dir": str(root),
        "selection": {
            "task_id": task_id,
            "episode_index": episode_index,
            "base_seed": seed,
            "episode_seed": authorized.episode_seed,
            "gpu_uuid": expected_gpu_uuid,
        },
        "result": {
            "success": bool(episode.get("success")),
            "policy_calls": calls,
            "selected_layers": layers,
            "policy_wall_latency_ms": latencies,
            "episode_wall_seconds": float(episode.get("wall_seconds")),
            "route_exact_fm_invocations": (
                int(route_integrity.get("fm_invocations"))
                if method == STAGE9_ROUTE_FIRST_METHOD
                else None
            ),
            "route_decoder_blocks": (
                int(route_integrity.get("decoder_blocks"))
                if method == STAGE9_ROUTE_FIRST_METHOD
                else None
            ),
        },
        "checks": checks,
        "artifacts": artifacts,
        "claim_boundary": {
            "single_pilot_arm_only": True,
            "task_failure_is_retained_not_replaced": True,
            "formal_speedup_claim": False,
            "deployment_authorized": False,
        },
    }


def main() -> None:
    args = parse_args()
    output = args.run_dir / "stage9_pilot_arm_attestation.json"
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = validate_arm(args.run_dir, repo_root=args.repo_root)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
