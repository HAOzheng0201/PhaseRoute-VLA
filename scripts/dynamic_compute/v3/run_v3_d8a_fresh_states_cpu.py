#!/usr/bin/env python3
"""Run all 400 isolated D8A generation processes and retain their logs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    load_d8_contract,
    load_fresh_confirmation_schedule,
)


OUTPUT = Path("reports/v3_d8_fresh_states_records")
WORKER = REPO_ROOT / "scripts/dynamic_compute/v3/generate_v3_d8a_fresh_state_record.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def run_one(pass_id: int, task_id: int, replicate_id: int, root: Path) -> dict:
    output = root / f"pass{pass_id}" / f"task{task_id:02d}_replicate{replicate_id:02d}"
    command = [
        sys.executable,
        str(WORKER),
        "--pass-id",
        str(pass_id),
        "--task-id",
        str(task_id),
        "--replicate-id",
        str(replicate_id),
        "--output-dir",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MUJOCO_GL": "osmesa",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(30_260_821 + task_id * 10_000 + replicate_id),
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
    log = root / "logs" / f"pass{pass_id}_task{task_id:02d}_replicate{replicate_id:02d}.log"
    log.write_text(
        shlex.join(command) + "\n\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"D8A record failed: pass={pass_id} task={task_id} "
            f"replicate={replicate_id}; see {log}"
        )
    result_path = output / "result.json"
    return {
        "pass_id": pass_id,
        "task_id": task_id,
        "replicate_id": replicate_id,
        "result_sha256": sha256(result_path),
        "state_sha256": json.loads(result_path.read_text(encoding="utf-8"))[
            "state_sha256"
        ],
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D8A runner is CPU-only")
    if not 1 <= args.max_workers <= 32:
        raise ValueError("V3-D8A max workers must be in 1..32")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D8A runner requires a clean worktree")
    load_d8_contract(REPO_ROOT)
    records = load_fresh_confirmation_schedule(REPO_ROOT)
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D8A refuses to overwrite generation records")
    incomplete.mkdir(parents=True, exist_ok=False)
    (incomplete / "logs").mkdir()
    (incomplete / "numba_cache").mkdir()
    (incomplete / "matplotlib_cache").mkdir()
    config_root = incomplete / "libero_config"
    config_root.mkdir()
    benchmark_root = REPO_ROOT / "robot_experiments/libero/LIBERO/libero/libero"
    (config_root / "config.yaml").write_text(
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
    for pass_id in (1, 2):
        futures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for record in records:
                futures.append(
                    executor.submit(
                        run_one,
                        pass_id,
                        record.task_id,
                        record.replicate_id,
                        incomplete,
                    )
                )
            for index, future in enumerate(as_completed(futures), start=1):
                completed_records.append(future.result())
                if index % 20 == 0 or index == len(futures):
                    print(f"D8A pass {pass_id}: {index}/{len(futures)} complete", flush=True)
    result = {
        "status": "PASS_V3_D8A_ISOLATED_GENERATION_COMPLETE_PENDING_AGGREGATION",
        "schema_version": "phase-route-vla.v3.d8a-generation-run.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "processes": len(completed_records),
        "generation_passes": 2,
        "records_per_pass": len(records),
        "max_parallel_processes": args.max_workers,
        "gpu_query_or_initialization": 0,
        "model_checkpoint_loaded": False,
        "policy_action_sampled": False,
        "official_episode_40_49_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
        "record_result_sha256": {
            f"pass{item['pass_id']}:task{item['task_id']}:replicate{item['replicate_id']}": item[
                "result_sha256"
            ]
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
        f"{sha256(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
