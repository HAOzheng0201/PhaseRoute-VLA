#!/usr/bin/env python3
"""Revalidate and seal one completed Stage 10 arm directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402


sha256_file = CONTRACT.sha256_file
validate_local_state_artifacts = CONTRACT.validate_local_state_artifacts
ACTIVE_ARM_SCHEMA = ACTIVE.ACTIVE_ARM_SCHEMA
ARM_ATTESTATION_SCHEMA = ACTIVE.ARM_ATTESTATION_SCHEMA
POSTFLIGHT_SCHEMA = ACTIVE.POSTFLIGHT_SCHEMA
PREFLIGHT_SCHEMA = ACTIVE.PREFLIGHT_SCHEMA
Stage10ActiveError = ACTIVE.Stage10ActiveError
normalize_gpu_uuid = ACTIVE.normalize_gpu_uuid
read_jsonl = ACTIVE.read_jsonl
select_arm = ACTIVE.select_arm
summarize_measurement_records = ACTIVE.summarize_measurement_records
summarize_policy_records = ACTIVE.summarize_policy_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm_dir", type=Path)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=("original_a1", "candidate_first_v3", "route_first_stage8"),
        required=True,
    )
    parser.add_argument("--arm-position", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser.parse_args()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ActiveError(f"JSON object required: {path}")
    return dict(value)


def _inventory(root: Path) -> tuple[dict[str, Any], ...]:
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in (
            "arm_attestation.json",
            "arm_attestation.sha256",
        ) or relative.endswith(".incomplete"):
            continue
        result.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(result)


def validate_arm(args: argparse.Namespace) -> dict[str, Any]:
    root = args.arm_dir.resolve(strict=True)
    spec = select_arm(
        REPO_ROOT,
        task_id=args.task_id,
        replicate_id=args.replicate_id,
        method=args.method,
        arm_position=args.arm_position,
    )
    required = {
        name: root / name
        for name in (
            "command.txt",
            "stdout.log",
            "preflight.json",
            "policy_telemetry.jsonl",
            "stage1_measurement.jsonl",
            "episode.log",
            "result.json",
            "result.sha256",
            "gpu_postflight.json",
        )
    }
    if spec.method != "original_a1":
        required["phase_route_runtime.jsonl"] = root / "phase_route_runtime.jsonl"
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise Stage10ActiveError(f"arm evidence is missing: {missing}")
    result = _object(required["result.json"])
    expected_result_sha = required["result.sha256"].read_text(
        encoding="utf-8"
    ).split()[0]
    observed_result_sha = sha256_file(required["result.json"])
    if expected_result_sha != observed_result_sha:
        raise Stage10ActiveError("arm result SHA-256 sidecar differs")
    preflight = _object(required["preflight.json"])
    postflight = _object(required["gpu_postflight.json"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip()
    local = validate_local_state_artifacts(REPO_ROOT)
    attestation_records = local["attestation"]["records"]
    state_records = [
        item
        for item in attestation_records
        if item.get("task_id") == spec.task_id
        and item.get("replicate_id") == spec.replicate_id
    ]
    if len(state_records) != 1:
        raise Stage10ActiveError("bound state attestation selection differs")
    telemetry = read_jsonl(required["policy_telemetry.jsonl"])
    measurements = read_jsonl(required["stage1_measurement.jsonl"])
    policy = summarize_policy_records(telemetry, spec=spec)
    latency = summarize_measurement_records(
        measurements, spec=spec, expected_policy_calls=policy["policy_calls"]
    )
    gpu = result.get("gpu", {})
    state = result.get("state_evidence", {})
    identity_ok = bool(
        result.get("schema_version") == ACTIVE_ARM_SCHEMA
        and result.get("status") == "COMPLETE_ROUTE_FIRST_STAGE10_ACTIVE_ARM"
        and result.get("method") == spec.method
        and result.get("task_id") == spec.task_id
        and result.get("replicate_id") == spec.replicate_id
        and result.get("cluster_key") == spec.cluster_key
        and result.get("state_seed") == spec.state_seed
        and result.get("policy_seed") == spec.policy_seed
        and result.get("arm_order") == list(spec.arm_order)
        and result.get("arm_position") == spec.arm_position
        and type(result.get("success")) is bool
        and type(result.get("environment_steps")) is int
        and result.get("environment_steps") >= 0
        and result.get("source_git_commit") == commit
        and result.get("source_worktree_dirty") is False
        and result.get("policy_accounting") == policy
        and result.get("policy_latency_ms") == latency
        and state.get("state_sha256") == state_records[0]["state_sha256"]
        and state.get("payload_sha256")
        == local["binding"]["local_state_payload"]["sha256"]
        and gpu.get("physical_index") == args.physical_gpu_index
        and normalize_gpu_uuid(gpu.get("uuid"))
        == normalize_gpu_uuid(args.expected_gpu_uuid)
    )
    preflight_ok = bool(
        preflight.get("schema_version") == PREFLIGHT_SCHEMA
        and preflight.get("status") == "PASS"
        and preflight.get("source_git_commit") == commit
        and preflight.get("cluster_key") == spec.cluster_key
        and preflight.get("method") == spec.method
        and preflight.get("arm_position") == spec.arm_position
        and preflight.get("physical_gpu_index") == args.physical_gpu_index
        and normalize_gpu_uuid(preflight.get("expected_gpu_uuid"))
        == normalize_gpu_uuid(args.expected_gpu_uuid)
        and all(preflight.get("checks", {}).values())
    )
    postflight_ok = bool(
        postflight.get("schema_version") == POSTFLIGHT_SCHEMA
        and postflight.get("status") == "PASS"
        and not postflight.get("compute_processes")
        and postflight.get("physical_gpu", {}).get("index")
        == args.physical_gpu_index
        and normalize_gpu_uuid(postflight.get("expected_gpu_uuid"))
        == normalize_gpu_uuid(args.expected_gpu_uuid)
        and all(postflight.get("checks", {}).values())
    )
    checks = {
        "worktree_clean": worktree == "",
        "identity_and_result_exact": identity_ok,
        "preflight_exact": preflight_ok,
        "postflight_exact": postflight_ok,
        "policy_telemetry_exact": policy["policy_calls"] > 0,
        "measurement_exact": latency["records"] == policy["policy_calls"],
        "route_exactly_one_fm": (
            policy["route_exactly_one_fm_calls"] == policy["policy_calls"]
            if spec.method == "route_first_stage8"
            else True
        ),
        "valid_task_failure_retained": True,
        "outcome_based_replacement_absent": True,
    }
    inventory = _inventory(root)
    return {
        "schema_version": ARM_ATTESTATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "cluster_key": spec.cluster_key,
        "method": spec.method,
        "arm_position": spec.arm_position,
        "arm_order": list(spec.arm_order),
        "policy_seed": spec.policy_seed,
        "state_sha256": state_records[0]["state_sha256"],
        "source_git_commit": commit,
        "physical_gpu_index": args.physical_gpu_index,
        "gpu_uuid": args.expected_gpu_uuid,
        "success": result["success"],
        "environment_steps": result["environment_steps"],
        "policy_calls": policy["policy_calls"],
        "selected_layer_counts": policy["selected_layer_counts"],
        "route_exactly_one_fm_calls": policy["route_exactly_one_fm_calls"],
        "policy_p50_ms": latency["p50_ms"],
        "result_sha256": observed_result_sha,
        "artifact_inventory": list(inventory),
        "artifact_count": len(inventory),
        "checks": checks,
        "claim_boundary": {
            "raw_arm_evidence_only": True,
            "stage10_gate_evaluated": False,
            "deployment_authorized": False,
        },
    }


def main() -> None:
    args = parse_args()
    output = args.arm_dir.resolve() / "arm_attestation.json"
    sidecar = output.with_name("arm_attestation.sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite arm attestation")
    value = validate_arm(args)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  arm_attestation.json\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if value["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
