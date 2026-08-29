#!/usr/bin/env python3
"""Generate one Stage 10 state record in an isolated CPU process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
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
REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "robot_experiments/libero/LIBERO"
for path in (REPO_ROOT, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONTRACT_PATH = REPO_ROOT / "a1/vla/dynamic_compute/route_first_stage10.py"
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_phase_route_stage10_contract", CONTRACT_PATH
)
if CONTRACT_SPEC is None or CONTRACT_SPEC.loader is None:
    raise ImportError(f"cannot load Stage 10 contract: {CONTRACT_PATH}")
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = CONTRACT
CONTRACT_SPEC.loader.exec_module(CONTRACT)

PROTOCOL_SHA256 = CONTRACT.PROTOCOL_SHA256
SCHEDULE_SHA256 = CONTRACT.SCHEDULE_SHA256
STATE_PASSES = CONTRACT.STATE_PASSES
STATE_RECORD_SCHEMA = CONTRACT.STATE_RECORD_SCHEMA
canonical_state_bytes = CONTRACT.canonical_state_bytes
load_schedule = CONTRACT.load_schedule
sha256_file = CONTRACT.sha256_file


LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OUTPUT_ROOT = Path("runs/route_first_stage10_state_records.incomplete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", type=int, required=True, choices=STATE_PASSES)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("Stage 10 state generation is CPU-only")
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise PermissionError("Stage 10 state generation requires OSMesa")
    if git_output(REPO_ROOT, "status", "--porcelain=v1"):
        raise PermissionError("Stage 10 state generation requires a clean worktree")
    schedule = load_schedule(REPO_ROOT)
    try:
        spec = next(
            item
            for item in schedule
            if item.task_id == args.task_id
            and item.replicate_id == args.replicate_id
        )
    except StopIteration as error:
        raise PermissionError(
            "Stage 10 task/replicate is outside the schedule"
        ) from error
    expected = (
        REPO_ROOT
        / OUTPUT_ROOT
        / f"pass{args.pass_id}"
        / f"task{args.task_id:02d}_replicate{args.replicate_id:02d}"
    ).resolve()
    output = args.output_dir.resolve()
    temporary = output.with_name(output.name + ".incomplete")
    if output != expected:
        raise PermissionError("Stage 10 state record output path differs")
    if output.exists() or temporary.exists():
        raise FileExistsError("Stage 10 state record refuses to overwrite evidence")
    if (
        git_output(LIBERO_ROOT, "rev-parse", "HEAD") != LIBERO_COMMIT
        or git_output(LIBERO_ROOT, "status", "--porcelain=v1")
    ):
        raise PermissionError("Stage 10 LIBERO gitlink differs or is dirty")

    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.libero.envs.env_wrapper import ControlEnv

    task_names = libero_task_map["libero_10"]
    if len(task_names) != 10:
        raise PermissionError("Stage 10 LIBERO-10 task map differs")
    task_name = task_names[args.task_id]
    bddl_path = (
        LIBERO_ROOT
        / "libero/libero/bddl_files/libero_10"
        / f"{task_name}.bddl"
    ).resolve(strict=True)

    random.seed(spec.state_seed)
    np.random.seed(spec.state_seed)
    torch.manual_seed(spec.state_seed)
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
        environment.seed(spec.state_seed)
        environment.env.reset()
        initial_success = bool(environment.check_success())
        state, state_bytes, state_hash = canonical_state_bytes(
            environment.get_sim_state()
        )
    finally:
        if environment is not None:
            environment.close()
    if initial_success:
        raise RuntimeError("Stage 10 generated an initially solved state")

    temporary.mkdir(parents=True, exist_ok=False)
    state_path = temporary / "state.bin"
    state_path.write_bytes(state_bytes)
    result = {
        "schema_version": STATE_RECORD_SCHEMA,
        "status": "PASS_ROUTE_FIRST_STAGE10_STATE_RECORD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "libero_gitlink_commit": LIBERO_COMMIT,
        "pass_id": args.pass_id,
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "cluster_key": spec.cluster_key,
        "arm_order": list(spec.arm_order),
        "task_name": task_name,
        "bddl_relative_path": str(bddl_path.relative_to(REPO_ROOT)),
        "bddl_sha256": sha256_file(bddl_path),
        "state_seed": spec.state_seed,
        "policy_seed": spec.policy_seed,
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
        "gpu_query_or_initialization": 0,
        "protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
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
    main()
