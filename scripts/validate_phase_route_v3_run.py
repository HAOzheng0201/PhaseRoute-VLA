#!/usr/bin/env python3
"""Seal one completed general-purpose PhaseRoute-V3 simulator run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
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


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    preflight_path = root / "preflight.json"
    evaluation_path = root / "evaluation_summary.json"
    runtime_path = root / "phase_route_runtime.jsonl"
    telemetry_path = root / "policy_telemetry.jsonl"
    stdout_path = root / "stdout.log"
    required_paths = (
        preflight_path,
        evaluation_path,
        runtime_path,
        telemetry_path,
        stdout_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing run artifacts: {missing}")

    preflight = load_object(preflight_path)
    evaluation = load_object(evaluation_path)
    measurement = evaluation.get("stage1_measurement")
    measurement_path = root / "stage1_measurement.jsonl"
    if measurement is not None and not isinstance(measurement, dict):
        raise ValueError("stage1_measurement must be an object or null")
    if isinstance(measurement, dict) and not measurement_path.is_file():
        raise FileNotFoundError(f"missing Stage-1 measurement: {measurement_path}")
    paths = required_paths + (
        (measurement_path,) if isinstance(measurement, dict) else ()
    )
    runtime = evaluation.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("evaluation summary has no runtime object")
    runtime_records = line_count(runtime_path)
    telemetry_records = line_count(telemetry_path)
    layer_counts = runtime.get("selected_layers", {})
    measurement_records = (
        line_count(measurement_path) if isinstance(measurement, dict) else None
    )
    checks = {
        "preflight_pass": preflight.get("status") == "PASS",
        "preflight_v3_scope": preflight.get("scope")
        == "phase_route_v3_release_preflight",
        "evaluation_schema": evaluation.get("schema_version")
        == "phase-route-vla.libero-evaluation-summary.v1",
        "evaluation_method": evaluation.get("method") == "phase_route_v3",
        "episodes_positive": int(evaluation.get("total_episodes", 0)) > 0,
        "successes_bounded": 0
        <= int(evaluation.get("total_successes", -1))
        <= int(evaluation.get("total_episodes", -1)),
        "runtime_records_match": runtime_records == int(runtime.get("records", -1)),
        "runtime_calls_match": runtime_records == int(runtime.get("policy_calls", -1)),
        "all_calls_prepared": runtime_records == int(runtime.get("prepared_calls", -1)),
        "all_calls_committed": runtime_records == int(runtime.get("committed_calls", -1)),
        "runtime_error_free": int(runtime.get("error_count", -1)) == 0
        and int(runtime.get("records_with_errors", -1)) == 0,
        "route_counts_complete": runtime_records
        == sum(int(layer_counts.get(str(layer), -1)) for layer in (11, 13, 27)),
        "telemetry_present": telemetry_records > 0,
        "measurement_complete_or_disabled": (
            measurement is None
            or (
                measurement_records == int(measurement.get("records", -1))
                and measurement_records == runtime_records
                and int(measurement.get("records_with_errors", -1)) == 0
                and int(measurement.get("records_with_nonfinite_actions", -1)) == 0
                and int(measurement.get("records_without_action_audit", -1)) == 0
            )
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "phase-route-vla.v3.general-run-attestation.v1",
        "status": status,
        "scope": "general_simulator_run_not_D9_retest",
        "research_simulation_only": True,
        "deployment_authorized": False,
        "run_dir": str(root),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        },
        "evaluation": {
            "suite": evaluation.get("suite"),
            "task_ids": evaluation.get("task_ids"),
            "episode_indices": evaluation.get("episode_indices"),
            "total_episodes": evaluation.get("total_episodes"),
            "total_successes": evaluation.get("total_successes"),
            "success_rate": evaluation.get("success_rate"),
            "runtime": runtime,
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
    result = validate_run(run_dir)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(output_path)
    print(output, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
