#!/usr/bin/env python3
"""Seal one completed preregistered route-first active arm."""

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
    ROUTE_FIRST_ACTIVE_METHOD,
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
    load_route_first_active_protocol,
    validate_route_first_active_selection,
)


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
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSONL objects: {path}")
        result.append(value)
    return tuple(result)


def _normalize_uuid(value: Any) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def _runtime_call_is_valid(record: Mapping[str, Any]) -> bool:
    events = record.get("events")
    if not isinstance(events, list):
        return False
    evaluated = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "exit_candidate"
        and event.get("evaluated") is True
    ]
    selected_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "route_first_selected_action"
    ]
    decisions = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "phase_route_decision"
    ]
    errors = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event")
        in ("route_first_action_error", "route_first_action_rejected")
    ]
    layer = record.get("selected_layer")
    scores = record.get("route_first_scores")
    return bool(
        record.get("schema_version")
        == "phase-route-vla.route-first-active-runtime.v1"
        and record.get("runtime_mode") == "route_first_l13_l27"
        and record.get("prepared") is True
        and record.get("committed") is True
        and not record.get("errors")
        and layer in (13, 27)
        and record.get("route_first_target_layer") == layer
        and isinstance(scores, list)
        and len(scores) == 2
        and len(evaluated) == 1
        and evaluated[0].get("layer_idx") == layer
        and evaluated[0].get("should_exit") is True
        and evaluated[0].get("fm_calls") == 1
        and len(selected_events) == 1
        and selected_events[0].get("layer_idx") == layer
        and selected_events[0].get("fm_calls") == 1
        and selected_events[0].get("fail_reason") is None
        and len(decisions) == 1
        and decisions[0].get("selected_layer") == layer
        and decisions[0].get("fm_calls") == 1
        and not errors
    )


def _measurement_call_is_valid(record: Mapping[str, Any], layer: Any) -> bool:
    return bool(
        record.get("schema_version") == "phase-route-vla.stage1.measurement.v1"
        and record.get("mode") == ROUTE_FIRST_ACTIVE_METHOD
        and record.get("measurement_is_control_input") is False
        and record.get("d9_protected_source_modified") is False
        and record.get("selected_layer") == layer
        and record.get("action_finite") is True
        and record.get("action_shape") == [8, 7]
        and record.get("error") is None
        and isinstance(record.get("policy_wall_latency_ms"), (int, float))
        and not isinstance(record.get("policy_wall_latency_ms"), bool)
        and float(record["policy_wall_latency_ms"]) >= 0.0
    )


def validate_run(
    run_dir: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    repository = Path(repo_root).resolve(strict=True)
    paths = {
        "preflight.json": root / "preflight.json",
        "evaluation_summary.json": root / "evaluation_summary.json",
        "phase_route_runtime.jsonl": root / "phase_route_runtime.jsonl",
        "policy_telemetry.jsonl": root / "policy_telemetry.jsonl",
        "stage1_measurement.jsonl": root / "stage1_measurement.jsonl",
        "stdout.log": root / "stdout.log",
        "command.sh": root / "command.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing active run artifacts: {missing}")

    protocol = load_route_first_active_protocol(
        repository / "configs/route_first_active_pilot_protocol.json",
        repository,
    )
    preflight = load_object(paths["preflight.json"])
    evaluation = load_object(paths["evaluation_summary.json"])
    runtime_records = load_jsonl(paths["phase_route_runtime.jsonl"])
    telemetry_records = load_jsonl(paths["policy_telemetry.jsonl"])
    measurement_records = load_jsonl(paths["stage1_measurement.jsonl"])
    runtime = evaluation.get("runtime")
    measurement = evaluation.get("stage1_measurement")
    latency = evaluation.get("active_latency_ms")
    gates = evaluation.get("gates")
    if not all(isinstance(item, dict) for item in (runtime, measurement, latency, gates)):
        raise ValueError("evaluation summary is missing active evidence objects")

    selection_error = None
    try:
        validate_route_first_active_selection(
            protocol,
            experiment_stage=str(evaluation.get("experiment_stage")),
            task_spec=",".join(str(item) for item in evaluation.get("task_ids", ())),
            episode_spec=",".join(
                str(item) for item in evaluation.get("episode_indices", ())
            ),
            arm_position=int(evaluation.get("arm_position", -1)),
            seed=int(evaluation.get("seed_base", -1)),
        )
    except Exception as error:
        selection_error = f"{type(error).__name__}: {error}"

    layer_counts = runtime.get("selected_layers", {})
    route_integrity = runtime.get("route_first_integrity", {})
    runtime_call_checks = tuple(_runtime_call_is_valid(row) for row in runtime_records)
    measurement_call_checks = tuple(
        _measurement_call_is_valid(measurement_row, runtime_row.get("selected_layer"))
        for measurement_row, runtime_row in zip(measurement_records, runtime_records)
    )
    expected_gpu_uuid = preflight.get("expected_gpu_uuid")
    evaluation_gpu = evaluation.get("gpu", {})
    checks = {
        "protocol_exact": evaluation.get("protocol_sha256")
        == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "selection_preregistered": selection_error is None,
        "preflight_pass": preflight.get("status") == "PASS",
        "preflight_scope": preflight.get("scope")
        == "route_first_stage9_active_preflight",
        "preflight_no_episode": preflight.get("simulator_episode_opened") is False,
        "preflight_protocol": preflight.get("protocol_sha256")
        == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "evaluation_schema": evaluation.get("schema_version")
        == "phase-route-vla.route-first-active-evaluation.v1",
        "evaluation_method": evaluation.get("method") == ROUTE_FIRST_ACTIVE_METHOD,
        "episodes_positive": int(evaluation.get("total_episodes", 0)) > 0,
        "successes_bounded": 0
        <= int(evaluation.get("total_successes", -1))
        <= int(evaluation.get("total_episodes", -1)),
        "gpu_uuid_bound": _normalize_uuid(expected_gpu_uuid)
        == _normalize_uuid(evaluation_gpu.get("expected_uuid"))
        == _normalize_uuid(evaluation_gpu.get("visible_uuid")),
        "runtime_records_match": len(runtime_records)
        == int(runtime.get("records", -1))
        == int(runtime.get("policy_calls", -2)),
        "all_calls_prepared": len(runtime_records)
        == int(runtime.get("prepared_calls", -1)),
        "all_calls_committed": len(runtime_records)
        == int(runtime.get("committed_calls", -1)),
        "runtime_error_free": int(runtime.get("error_count", -1)) == 0
        and int(runtime.get("records_with_errors", -1)) == 0,
        "l11_permanently_disabled": int(layer_counts.get("11", -1)) == 0,
        "route_counts_complete": len(runtime_records)
        == int(layer_counts.get("13", -1)) + int(layer_counts.get("27", -1)),
        "every_runtime_call_exactly_one_fm": bool(runtime_records)
        and all(runtime_call_checks),
        "route_integrity_matches": int(
            route_integrity.get("valid_calls_with_exactly_one_fm", -1)
        )
        == len(runtime_records)
        and int(route_integrity.get("fm_invocations", -1)) == len(runtime_records)
        and float(
            route_integrity.get(
                "valid_calls_with_fm_calls_equal_one_fraction", -1.0
            )
        )
        == 1.0,
        "telemetry_one_per_call": len(telemetry_records) == len(runtime_records),
        "measurement_one_per_call": len(measurement_records) == len(runtime_records),
        "every_action_audited": bool(measurement_records)
        and len(measurement_call_checks) == len(runtime_records)
        and all(measurement_call_checks),
        "measurement_summary_complete": int(measurement.get("records", -1))
        == len(runtime_records)
        and int(measurement.get("records_with_errors", -1)) == 0
        and int(measurement.get("records_with_nonfinite_actions", -1)) == 0
        and int(measurement.get("records_without_action_audit", -1)) == 0,
        "latency_summary_complete": int(latency.get("count", -1))
        == len(runtime_records)
        and all(latency.get(name) is not None for name in ("mean", "p50", "p90")),
        "runner_gates_pass": gates.get("runtime_integrity") is True
        and gates.get("measurement_integrity") is True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "phase-route-vla.route-first-active-attestation.v1",
        "status": status,
        "scope": "stage9_preregistered_engineering_active_control",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "run_dir": str(root),
        "selection_error": selection_error,
        "artifacts": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "evaluation": {
            "method": evaluation.get("method"),
            "experiment_stage": evaluation.get("experiment_stage"),
            "arm_position": evaluation.get("arm_position"),
            "suite": evaluation.get("suite"),
            "task_ids": evaluation.get("task_ids"),
            "episode_indices": evaluation.get("episode_indices"),
            "total_episodes": evaluation.get("total_episodes"),
            "total_successes": evaluation.get("total_successes"),
            "runtime": runtime,
            "active_latency_ms": latency,
        },
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = run_dir / "run_attestation.json"
    temporary = output_path.with_name(output_path.name + ".incomplete")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    result = validate_run(run_dir, repo_root=args.repo_root)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(output_path)
    print(output, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
