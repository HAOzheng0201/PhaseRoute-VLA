#!/usr/bin/env python3
"""Validate Stage-11D without opening data, a simulator, or CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_reliability import (  # noqa: E402
    build_stage11d_schedule,
    validate_stage11d_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_once(path: Path, payload: dict[str, object]) -> None:
    target = path.expanduser().resolve()
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    protocol = validate_stage11d_protocol(REPO_ROOT)
    schedule = build_stage11d_schedule()
    payload: dict[str, object] = {
        "schema_version": "phase-route-vla.route-first-stage11d-readiness.v1",
        "status": "PASS_ROUTE_FIRST_STAGE11D_PROTOCOL_READINESS",
        "protocol": protocol,
        "schedule": {
            "clusters": len(schedule),
            "tasks": len({record.task_id for record in schedule}),
            "replicates_per_task": 20,
            "unique_cluster_keys": len({record.cluster_key for record in schedule}),
            "unique_state_seeds": len({record.state_seed for record in schedule}),
            "unique_policy_seeds": len({record.policy_seed for record in schedule}),
            "state_policy_seed_overlap": len(
                {record.state_seed for record in schedule}
                & {record.policy_seed for record in schedule}
            ),
        },
        "execution": {
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "data_payload_opened": False,
            "simulator_opened": False,
            "GPU_queried_or_initialized": False,
            "training_run": False,
            "active_control_run": False,
        },
        "next_stage": "IMPLEMENT_GENERATION_COLLECTION_AND_REPLAY_RUNNERS_WITHOUT_EXECUTION",
    }
    if args.output is not None:
        _write_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
