#!/usr/bin/env python3
"""Generate one Stage-11D reset state in an isolated CPU process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MUJOCO_GL"] = "osmesa"
REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "robot_experiments/libero/LIBERO"
for path in (REPO_ROOT, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.vla.dynamic_compute.route_first_reliability import (  # noqa: E402
    STAGE11D_PROTOCOL_SHA256,
    build_stage11d_schedule,
    validate_stage11d_protocol,
)
from a1.vla.dynamic_compute.route_first_reliability_artifacts import (  # noqa: E402
    STAGE11D_LIBERO_COMMIT,
    STAGE11D_STATE_PASSES,
    STAGE11D_STATE_RECORDS_RELATIVE_PATH,
    STAGE11D_STATE_RECORD_SCHEMA,
    canonical_state_bytes,
    sha256_file,
    validate_state_runner_readiness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", type=int, choices=STAGE11D_STATE_PASSES, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _expected(args: argparse.Namespace):
    try:
        return next(
            record
            for record in build_stage11d_schedule()
            if record.task_id == args.task_id
            and record.replicate_id == args.replicate_id
        )
    except StopIteration as error:
        raise PermissionError("Stage-11D task/replicate is outside schedule") from error


def _expected_output(args: argparse.Namespace) -> Path:
    records = REPO_ROOT / STAGE11D_STATE_RECORDS_RELATIVE_PATH
    incomplete_root = records.with_name(records.name + ".incomplete")
    return (
        incomplete_root
        / f"pass{args.pass_id}"
        / f"task{args.task_id:02d}_replicate{args.replicate_id:02d}"
    ).resolve()


def run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("Stage-11D state generation is CPU-only")
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise PermissionError("Stage-11D state generation requires OSMesa")
    if git_output(REPO_ROOT, "status", "--porcelain=v1"):
        raise PermissionError("Stage-11D state generation requires clean worktree")
    protocol = validate_stage11d_protocol(REPO_ROOT)
    expected = _expected(args)
    output = args.output_dir.resolve()
    temporary = output.with_name(output.name + ".incomplete")
    if output != _expected_output(args):
        raise PermissionError("Stage-11D state record output path differs")
    if output.exists() or temporary.exists():
        raise FileExistsError("Stage-11D state record refuses to overwrite evidence")
    if (
        git_output(LIBERO_ROOT, "rev-parse", "HEAD") != STAGE11D_LIBERO_COMMIT
        or git_output(LIBERO_ROOT, "status", "--porcelain=v1")
    ):
        raise PermissionError("Stage-11D LIBERO gitlink differs or is dirty")
    readiness = validate_state_runner_readiness(REPO_ROOT)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_ROUTE_FIRST_STAGE11D_STATE_RECORD_PREFLIGHT",
                    "task_id": expected.task_id,
                    "replicate_id": expected.replicate_id,
                    "split": expected.split,
                    "pass_id": args.pass_id,
                    "state_seed": expected.state_seed,
                    "policy_seed": expected.policy_seed,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "readiness_status": readiness["status"],
                    "output_absent": True,
                    "state_generated": False,
                    "simulator_opened": False,
                    "gpu_query_or_initialization": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.libero.envs.env_wrapper import ControlEnv

    task_names = libero_task_map["libero_10"]
    if len(task_names) != 10:
        raise PermissionError("Stage-11D LIBERO-10 task map differs")
    task_name = task_names[args.task_id]
    bddl_path = (
        LIBERO_ROOT
        / "libero/libero/bddl_files/libero_10"
        / f"{task_name}.bddl"
    ).resolve(strict=True)

    random.seed(expected.state_seed)
    np.random.seed(expected.state_seed)
    torch.manual_seed(expected.state_seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    environment = None
    try:
        environment = ControlEnv(
            bddl_file_name=str(bddl_path),
            initialization_noise=None,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            hard_reset=True,
        )
        environment.seed(expected.state_seed)
        environment.env.reset()  # One reset, no wrapper retry/replacement loop.
        initial_success = bool(environment.check_success())
        state, state_bytes, state_hash = canonical_state_bytes(
            environment.get_sim_state()
        )
    finally:
        if environment is not None:
            environment.close()
    if initial_success:
        raise RuntimeError(
            "Stage-11D generated an initially solved state; retain failure, no replacement"
        )

    temporary.mkdir(parents=True, exist_ok=False)
    state_path = temporary / "state.bin"
    state_path.write_bytes(state_bytes)
    result = {
        "schema_version": STAGE11D_STATE_RECORD_SCHEMA,
        "status": "PASS_ROUTE_FIRST_STAGE11D_STATE_RECORD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "libero_gitlink_commit": STAGE11D_LIBERO_COMMIT,
        "pass_id": args.pass_id,
        "task_id": expected.task_id,
        "replicate_id": expected.replicate_id,
        "split": expected.split,
        "cluster_key": expected.cluster_key,
        "task_name": task_name,
        "bddl_relative_path": str(bddl_path.relative_to(REPO_ROOT)),
        "bddl_sha256": sha256_file(bddl_path),
        "state_seed": expected.state_seed,
        "policy_seed": expected.policy_seed,
        "state_dtype": "float64-le",
        "state_rank": int(state.ndim),
        "state_dimension": int(state.size),
        "state_nbytes": len(state_bytes),
        "state_payload": state_path.name,
        "state_sha256": state_hash,
        "initial_task_success": initial_success,
        "explicit_reset_attempts": 1,
        "model_checkpoint_loaded": False,
        "policy_action_sampled": False,
        "official_episode_identity_used": False,
        "historical_generated_state_payload_opened": False,
        "gpu_query_or_initialization": 0,
        "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
        "runner_readiness_sha256": sha256_file(
            REPO_ROOT
            / "results/route_first/route_first_stage11d_state_runner_readiness.json"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = temporary / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (temporary / "result.sha256").write_text(
        f"{sha256_file(result_path)}  result.json\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temporary), str(output))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    run(parse_args())
