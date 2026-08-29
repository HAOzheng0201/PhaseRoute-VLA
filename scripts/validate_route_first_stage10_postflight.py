#!/usr/bin/env python3
"""Audit the physical GPU after one Stage 10 arm process exits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE  # noqa: E402
from scripts.validate_route_first_stage10_preflight import (  # noqa: E402
    compute_processes,
    gpu_snapshot,
)


POSTFLIGHT_SCHEMA = ACTIVE.POSTFLIGHT_SCHEMA
normalize_gpu_uuid = ACTIVE.normalize_gpu_uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit(index: int, expected_uuid: str) -> dict[str, Any]:
    gpu = gpu_snapshot(index)
    processes = compute_processes(expected_uuid)
    checks = {
        "physical_index_matches": gpu["index"] == index,
        "physical_uuid_matches": normalize_gpu_uuid(gpu["uuid"])
        == normalize_gpu_uuid(expected_uuid),
        "no_compute_process_after_arm": not processes,
    }
    return {
        "schema_version": POSTFLIGHT_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "physical_gpu": gpu,
        "expected_gpu_uuid": expected_uuid,
        "compute_processes": processes,
        "checks": checks,
        "research_simulation_only": True,
        "deployment_authorized": False,
    }


def main() -> None:
    args = parse_args()
    target = args.output.resolve()
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    value = audit(args.physical_gpu_index, args.expected_gpu_uuid)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if value["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
