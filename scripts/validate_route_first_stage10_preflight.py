#!/usr/bin/env python3
"""No-state-open, no-CUDA preflight immediately before one Stage 10 arm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


# Preflight uses only nvidia-smi.  Optional imports reached through ``a1`` must
# never create a CUDA context before the active arm is authorized.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402


PROTOCOL_SHA256 = CONTRACT.PROTOCOL_SHA256
SCHEDULE_SHA256 = CONTRACT.SCHEDULE_SHA256
validate_local_state_artifacts = CONTRACT.validate_local_state_artifacts
MINIMUM_FREE_MEMORY_MIB = ACTIVE.MINIMUM_FREE_MEMORY_MIB
PREFLIGHT_SCHEMA = ACTIVE.PREFLIGHT_SCHEMA
load_runner_readiness = ACTIVE.load_runner_readiness
normalize_gpu_uuid = ACTIVE.normalize_gpu_uuid
select_arm = ACTIVE.select_arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run(command: list[str]) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return subprocess.run(
                command, check=True, capture_output=True, text=True
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(f"read-only command failed after three attempts: {last_error}")


def gpu_snapshot(index: int) -> dict[str, Any]:
    lines = _run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("physical GPU query did not return exactly one row")
    fields = [item.strip() for item in lines[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError("physical GPU query format differs")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_used_mib": int(fields[3]),
        "memory_total_mib": int(fields[4]),
        "memory_free_mib": int(fields[4]) - int(fields[3]),
        "utilization_gpu_percent": int(fields[5]),
    }


def compute_processes(expected_uuid: str) -> list[dict[str, Any]]:
    rows = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    ).strip().splitlines()
    result = []
    for row in rows:
        if not row.strip():
            continue
        fields = [item.strip() for item in row.split(",", 3)]
        if len(fields) != 4 or normalize_gpu_uuid(fields[0]) != normalize_gpu_uuid(
            expected_uuid
        ):
            continue
        result.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": int(fields[3]),
            }
        )
    return result


def validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.physical_gpu_index not in range(8):
        raise ValueError("physical GPU index must be in 0..7")
    spec = select_arm(
        REPO_ROOT,
        task_id=args.task_id,
        replicate_id=args.replicate_id,
        method=args.method,
        arm_position=args.arm_position,
    )
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    readiness_error = None
    state_error = None
    readiness = None
    local = None
    try:
        readiness = load_runner_readiness(REPO_ROOT)
    except Exception as error:
        readiness_error = f"{type(error).__name__}: {error}"
    try:
        # Exact bytes are hashed, but fresh_states.pt is deliberately not
        # deserialized by preflight.
        local = validate_local_state_artifacts(REPO_ROOT)
    except Exception as error:
        state_error = f"{type(error).__name__}: {error}"
    gpu = gpu_snapshot(args.physical_gpu_index)
    processes = compute_processes(args.expected_gpu_uuid)
    checks = {
        "worktree_clean": worktree == "",
        "runner_readiness_exact": readiness is not None and readiness_error is None,
        "state_binding_bytes_exact_without_deserialization": local is not None
        and state_error is None,
        "physical_index_matches": gpu["index"] == args.physical_gpu_index,
        "physical_uuid_matches": normalize_gpu_uuid(gpu["uuid"])
        == normalize_gpu_uuid(args.expected_gpu_uuid),
        "no_external_compute_process": not processes,
        "minimum_free_memory_40000_mib": gpu["memory_free_mib"]
        >= MINIMUM_FREE_MEMORY_MIB,
        "fresh_state_payload_not_opened": True,
        "simulator_environment_not_created": True,
        "model_not_loaded": True,
        "cuda_not_initialized": True,
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "source_git_commit": commit,
        "source_worktree_dirty": bool(worktree),
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "cluster_key": spec.cluster_key,
        "state_seed": spec.state_seed,
        "policy_seed": spec.policy_seed,
        "arm_order": list(spec.arm_order),
        "method": spec.method,
        "arm_position": spec.arm_position,
        "physical_gpu_index": args.physical_gpu_index,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "physical_gpu": gpu,
        "external_compute_processes": processes,
        "runner_readiness_error": readiness_error,
        "state_binding_error": state_error,
        "checks": checks,
        "access_ledger": {
            "fresh_state_payload_deserialized": False,
            "LIBERO_environment_created": False,
            "model_loaded": False,
            "CUDA_initialized": False,
            "active_rollouts": 0,
        },
        "research_simulation_only": True,
        "deployment_authorized": False,
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    result = validate_preflight(args)
    _write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
