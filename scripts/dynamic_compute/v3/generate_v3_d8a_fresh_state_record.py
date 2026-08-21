#!/usr/bin/env python3
"""Generate one immutable D8A state record in one isolated process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = REPO_ROOT / "robot_experiments/libero/LIBERO"
LIBERO_PYTHON_ROOT = LIBERO_ROOT / "libero"
for path in (REPO_ROOT, LIBERO_PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.d8_artifacts import (  # noqa: E402
    D8A_RECORD_SCHEMA_VERSION,
    canonical_state_bytes,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CONTRACT_SHA256,
    D8_REPLICATE_IDS,
    D8_SCHEDULE_SHA256,
    D8_TASK_IDS,
    load_d8_contract,
    load_fresh_confirmation_schedule,
)


LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
D8_CONTRACT_VALIDATION = Path(
    "results/v3/v3_d8_fresh_confirmation_contract_validation.json"
)
D8_CONTRACT_VALIDATION_SHA256 = (
    "ccda03321468f78eb483b0fe276b7d3eed4e92653968abd53ba83c4986f60f1f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", type=int, required=True, choices=(1, 2))
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D8A state generation is CPU-only")
    if git_output(REPO_ROOT, "status", "--porcelain=v1"):
        raise PermissionError("V3-D8A state generation requires a clean worktree")
    if (
        args.task_id not in D8_TASK_IDS
        or args.replicate_id not in D8_REPLICATE_IDS
    ):
        raise PermissionError("V3-D8A task or replicate id differs")
    load_d8_contract(REPO_ROOT)
    records = load_fresh_confirmation_schedule(REPO_ROOT)
    expected = next(
        record
        for record in records
        if record.task_id == args.task_id
        and record.replicate_id == args.replicate_id
    )
    validation_path = REPO_ROOT / D8_CONTRACT_VALIDATION
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        sha256(validation_path) != D8_CONTRACT_VALIDATION_SHA256
        or validation.get("status")
        != "PASS_V3_D8_FRESH_CONFIRMATION_CONTRACT_FROZEN"
        or validation.get("authorization", {}).get("on_contract_validation_pass")
        != "D8A_FRESH_STATE_GENERATION_AND_D8B_FINAL_ROUTER_FINALIZATION_ONLY"
    ):
        raise PermissionError("V3-D8A contract validation evidence differs")
    if (
        git_output(LIBERO_ROOT, "rev-parse", "HEAD") != LIBERO_COMMIT
        or git_output(LIBERO_ROOT, "status", "--porcelain=v1")
    ):
        raise PermissionError("V3-D8A LIBERO gitlink differs or is dirty")

    expected_output = (
        REPO_ROOT
        / "reports/v3_d8_fresh_states_records.incomplete"
        / f"pass{args.pass_id}"
        / f"task{args.task_id:02d}_replicate{args.replicate_id:02d}"
    ).resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D8A record output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D8A refuses to overwrite state evidence")
    output.parent.mkdir(parents=True, exist_ok=True)

    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.libero.envs.env_wrapper import ControlEnv

    task_names = libero_task_map["libero_10"]
    if len(task_names) != 10:
        raise PermissionError("V3-D8A LIBERO-10 task map differs")
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
        environment.env.reset()  # Exactly one explicit reset; no wrapper retry loop.
        initial_success = bool(environment.check_success())
        state, state_bytes, state_hash = canonical_state_bytes(
            environment.get_sim_state()
        )
    finally:
        if environment is not None:
            environment.close()
    if initial_success:
        raise RuntimeError("V3-D8A generated an initially solved state; fail closed")

    incomplete.mkdir(parents=True, exist_ok=False)
    state_path = incomplete / "state.bin"
    state_path.write_bytes(state_bytes)
    result = {
        "status": "PASS_V3_D8A_FRESH_STATE_RECORD",
        "schema_version": D8A_RECORD_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "libero_gitlink_commit": LIBERO_COMMIT,
        "pass_id": args.pass_id,
        "task_id": expected.task_id,
        "replicate_id": expected.replicate_id,
        "cluster_key": expected.cluster_key,
        "task_name": task_name,
        "bddl_relative_path": str(bddl_path.relative_to(REPO_ROOT)),
        "bddl_sha256": sha256(bddl_path),
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
        "gpu_query_or_initialization": 0,
        "D8_contract_sha256": D8_CONTRACT_SHA256,
        "D8_schedule_sha256": D8_SCHEDULE_SHA256,
        "D8_contract_validation_sha256": D8_CONTRACT_VALIDATION_SHA256,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{sha256(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    run(parse_args())
