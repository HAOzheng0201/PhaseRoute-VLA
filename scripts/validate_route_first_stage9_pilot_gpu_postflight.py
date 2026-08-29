#!/usr/bin/env python3
"""Fail closed if another compute process is present after one pilot arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any


SCHEMA = "phase-route-vla.route-first-stage9-pilot-gpu-postflight.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_uuid(value: Any) -> str:
    result = str(value).strip().lower()
    return result[4:] if result.startswith("gpu-") else result


def _run(command: list[str]) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(
        f"nvidia-smi failed after three read-only attempts: {last_error}"
    )


def _parse_memory(value: str) -> int:
    return int(value.strip().removesuffix(" MiB").strip())


def audit_gpu(index: int, expected_uuid: str) -> dict[str, Any]:
    gpu_rows = _run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).strip().splitlines()
    if len(gpu_rows) != 1:
        raise RuntimeError("physical GPU query did not return exactly one row")
    fields = [item.strip() for item in gpu_rows[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError("unexpected physical GPU query format")
    gpu = {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_used_mib": int(fields[3]),
        "memory_total_mib": int(fields[4]),
        "utilization_gpu_percent": int(fields[5]),
    }
    process_rows = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    ).strip().splitlines()
    processes = []
    for row in process_rows:
        if not row.strip():
            continue
        parts = [item.strip() for item in row.split(",", 3)]
        if len(parts) != 4 or normalize_uuid(parts[0]) != normalize_uuid(expected_uuid):
            continue
        processes.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": _parse_memory(parts[3]),
            }
        )
    checks = {
        "physical_index_matches": gpu["index"] == index,
        "physical_uuid_matches": normalize_uuid(gpu["uuid"])
        == normalize_uuid(expected_uuid),
        "no_compute_process_after_arm": not processes,
    }
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "stage9_state13_pilot_post_arm_gpu_audit",
        "research_simulation_only": True,
        "deployment_authorized": False,
        "physical_gpu": gpu,
        "expected_gpu_uuid": expected_uuid,
        "compute_processes": processes,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_name(
        args.output.name + ".incomplete"
    ).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit_gpu(args.physical_gpu_index, args.expected_gpu_uuid)
    temporary = args.output.with_name(args.output.name + ".incomplete")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
