#!/usr/bin/env python3
"""Publish CPU-only readiness for the Stage-11D generated-state runner."""

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
    STAGE11D_PROTOCOL_SHA256,
    build_stage11d_schedule,
    validate_stage11d_protocol,
)
from a1.vla.dynamic_compute.route_first_reliability_artifacts import (  # noqa: E402
    STAGE11D_STATE_PASSES,
    STAGE11D_STATE_RECORDS_RELATIVE_PATH,
    STAGE11D_STATES_RELATIVE_PATH,
    sha256_file,
)


RUNNER_FILES = (
    "a1/vla/dynamic_compute/route_first_reliability.py",
    "a1/vla/dynamic_compute/route_first_reliability_artifacts.py",
    "configs/research/route_first_stage11d_reliability_protocol.json",
    "scripts/dynamic_compute/route_first_stage11d/README.md",
    "scripts/dynamic_compute/route_first_stage11d/generate_state_record.py",
    "scripts/dynamic_compute/route_first_stage11d/generate_states.py",
    "scripts/dynamic_compute/route_first_stage11d/aggregate_states.py",
    "scripts/validate_route_first_stage11d_state_runner.py",
    "tests/dynamic_compute/test_route_first_reliability.py",
    "tests/dynamic_compute/test_route_first_reliability_states.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
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
    sources = {
        relative: (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in RUNNER_FILES
        if relative.endswith(".py")
    }
    worker = sources[
        "scripts/dynamic_compute/route_first_stage11d/generate_state_record.py"
    ]
    preflight_runners = (
        "scripts/dynamic_compute/route_first_stage11d/generate_state_record.py",
        "scripts/dynamic_compute/route_first_stage11d/generate_states.py",
    )
    cpu_runners = (*preflight_runners, "scripts/dynamic_compute/route_first_stage11d/aggregate_states.py")
    combined = "\n".join(sources[relative] for relative in cpu_runners)
    checks = {
        "protocol_is_frozen": protocol["protocol_sha256"]
        == STAGE11D_PROTOCOL_SHA256,
        "schedule_is_200_unique_clusters": len(schedule) == 200
        and len({record.cluster_key for record in schedule}) == 200,
        "two_isolated_generation_passes": STAGE11D_STATE_PASSES == (1, 2),
        "worker_has_exactly_one_explicit_reset": worker.count(
            "environment.env.reset()"
        )
        == 1,
        "preflight_paths_exist": all(
            "--preflight-only" in sources[relative]
            for relative in preflight_runners
        ),
        "official_fixed_state_loader_absent": "get_task_init_states" not in combined,
        "model_checkpoint_loader_absent": "model.pt" not in combined,
        "state_output_absent": not (
            REPO_ROOT / STAGE11D_STATE_RECORDS_RELATIVE_PATH
        ).exists()
        and not (REPO_ROOT / STAGE11D_STATES_RELATIVE_PATH).exists(),
        "protected_A1_files_not_runner_inputs": all(
            name not in combined
            for name in (
                "a1/vla/value_net.py",
                "robot_experiments/libero/exit_vla_utils.py",
                "robot_experiments/libero/eval_libero_early_exit.py",
            )
        ),
        "CUDA_disabled_by_all_runners": all(
            'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in sources[relative]
            for relative in cpu_runners
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("Stage-11D state runner failed static readiness")
    payload: dict[str, object] = {
        "schema_version": "phase-route-vla.route-first-stage11d-state-runner-readiness.v1",
        "status": "PASS_ROUTE_FIRST_STAGE11D_STATE_RUNNER_READINESS",
        "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
        "runner_files": {
            relative: sha256_file(REPO_ROOT / relative) for relative in RUNNER_FILES
        },
        "schedule": {
            "clusters": len(schedule),
            "passes": list(STAGE11D_STATE_PASSES),
            "isolated_processes": len(schedule) * len(STAGE11D_STATE_PASSES),
            "split_counts": protocol["split_counts"],
        },
        "checks": checks,
        "execution": {
            "state_generated": False,
            "state_payload_opened": False,
            "LIBERO_simulator_opened": False,
            "model_loaded": False,
            "policy_action_sampled": False,
            "GPU_queried_or_initialized": False,
        },
        "authorization": {
            "state_generation": True,
            "original_A1_collection": False,
            "same_noise_replay": False,
            "training": False,
            "active_control": False,
        },
        "next_stage": "RUN_PREFLIGHT_THEN_GENERATE_AND_AGGREGATE_200_STATES_ON_CPU",
    }
    _write_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
