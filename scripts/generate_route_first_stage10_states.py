#!/usr/bin/env python3
"""Run the two-pass isolated Stage 10 fresh-state generation schedule."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
load_schedule = CONTRACT.load_schedule
sha256_file = CONTRACT.sha256_file


OUTPUT = Path("runs/route_first_stage10_state_records")
WORKER = REPO_ROOT / "scripts/generate_route_first_stage10_state_record.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def run_one(pass_id: int, spec, root: Path) -> dict:
    output = (
        root
        / f"pass{pass_id}"
        / f"task{spec.task_id:02d}_replicate{spec.replicate_id:02d}"
    )
    command = [
        sys.executable,
        str(WORKER),
        "--pass-id",
        str(pass_id),
        "--task-id",
        str(spec.task_id),
        "--replicate-id",
        str(spec.replicate_id),
        "--output-dir",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MUJOCO_GL": "osmesa",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(spec.state_seed),
            "LIBERO_CONFIG_PATH": str(root / "libero_config"),
            "NUMBA_CACHE_DIR": str(root / "numba_cache"),
            "MPLCONFIGDIR": str(root / "matplotlib_cache"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = (
        root
        / "logs"
        / f"pass{pass_id}_task{spec.task_id:02d}_replicate{spec.replicate_id:02d}.log"
    )
    log_path.write_text(
        shlex.join(command) + "\n\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"Stage 10 state record failed: pass={pass_id} task={spec.task_id} "
            f"replicate={spec.replicate_id}; see {log_path}"
        )
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "pass_id": pass_id,
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "result_sha256": sha256_file(result_path),
        "state_sha256": result["state_sha256"],
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("Stage 10 state generation is CPU-only")
    if not 1 <= args.max_workers <= 32:
        raise ValueError("max-workers must be in 1..32")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("Stage 10 state generation requires a clean worktree")
    schedule = load_schedule(REPO_ROOT)
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("Stage 10 state generation refuses to overwrite evidence")
    incomplete.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "numba_cache", "matplotlib_cache", "libero_config"):
        (incomplete / name).mkdir()
    benchmark_root = REPO_ROOT / "robot_experiments/libero/LIBERO/libero/libero"
    (incomplete / "libero_config/config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {benchmark_root.parent / 'datasets'}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    started = time.perf_counter()
    completed_records = []
    for pass_id in STATE_PASSES:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(run_one, pass_id, spec, incomplete)
                for spec in schedule
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                completed_records.append(future.result())
                if index % 10 == 0 or index == len(futures):
                    print(
                        f"Stage10 state pass {pass_id}: "
                        f"{index}/{len(futures)} complete",
                        flush=True,
                    )
    result = {
        "schema_version": "phase-route-vla.route-first-stage10-state-generation-run.v1",
        "status": "PASS_ROUTE_FIRST_STAGE10_STATE_GENERATION_PENDING_AGGREGATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "processes": len(completed_records),
        "generation_passes": len(STATE_PASSES),
        "records_per_pass": len(schedule),
        "max_parallel_processes": args.max_workers,
        "gpu_query_or_initialization": 0,
        "model_checkpoint_loaded": False,
        "policy_action_sampled": False,
        "official_states_0_to_49_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "record_result_sha256": {
            (
                f"pass{item['pass_id']}:task{item['task_id']}:"
                f"replicate{item['replicate_id']}"
            ): item["result_sha256"]
            for item in sorted(
                completed_records,
                key=lambda value: (
                    value["pass_id"],
                    value["task_id"],
                    value["replicate_id"],
                ),
            )
        },
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{sha256_file(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
