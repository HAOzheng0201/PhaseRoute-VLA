#!/usr/bin/env python3
"""Launch the frozen 60-triplet Stage 10 schedule on explicitly idle GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402
from scripts.validate_route_first_stage10_preflight import (  # noqa: E402
    compute_processes,
    gpu_snapshot,
)


load_schedule = CONTRACT.load_schedule
MINIMUM_FREE_MEMORY_MIB = ACTIVE.MINIMUM_FREE_MEMORY_MIB
load_runner_readiness = ACTIVE.load_runner_readiness


TRIPLET_RUNNER = REPO_ROOT / "scripts/run_route_first_stage10_triplet.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-indices",
        required=True,
        help="comma-separated physical GPUs verified idle immediately before launch",
    )
    parser.add_argument(
        "--python-bin", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "model/libero_exit"
    )
    return parser.parse_args()


def _indices(value: str) -> tuple[int, ...]:
    tokens = tuple(item.strip() for item in value.split(","))
    if not tokens or any(not item.isdigit() for item in tokens):
        raise ValueError("GPU indices must be comma-separated integers")
    result = tuple(int(item) for item in tokens)
    if len(result) != len(set(result)) or any(item not in range(8) for item in result):
        raise ValueError("GPU indices must be unique values in 0..7")
    return result


def main() -> None:
    args = parse_args()
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip():
        raise PermissionError("Stage 10 launcher requires a clean worktree")
    load_runner_readiness(REPO_ROOT)
    gpu_indices = _indices(args.gpu_indices)
    gpu_bindings = []
    for index in gpu_indices:
        snapshot = gpu_snapshot(index)
        processes = compute_processes(snapshot["uuid"])
        if processes or snapshot["memory_free_mib"] < MINIMUM_FREE_MEMORY_MIB:
            raise RuntimeError(f"GPU {index} is not idle with 40000 MiB free")
        gpu_bindings.append((index, snapshot["uuid"]))
    schedule_queue: queue.Queue[Any] = queue.Queue()
    for spec in load_schedule(REPO_ROOT):
        schedule_queue.put(spec)
    stop = threading.Event()
    failures = []
    failure_lock = threading.Lock()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_root = REPO_ROOT / "runs/route_first_stage10_launch_logs" / timestamp
    log_root.mkdir(parents=True, exist_ok=False)
    python = str(args.python_bin.resolve(strict=True))
    checkpoint = str(args.checkpoint.resolve(strict=True))

    def worker(index: int, uuid: str) -> None:
        while not stop.is_set():
            try:
                spec = schedule_queue.get_nowait()
            except queue.Empty:
                return
            output = (
                REPO_ROOT
                / "runs/route_first_stage10_active"
                / f"task{spec.task_id:02d}_replicate{spec.replicate_id:02d}"
            )
            command = [
                python,
                str(TRIPLET_RUNNER),
                "--task-id",
                str(spec.task_id),
                "--replicate-id",
                str(spec.replicate_id),
                "--physical-gpu-index",
                str(index),
                "--expected-gpu-uuid",
                uuid,
                "--checkpoint",
                checkpoint,
                "--python-bin",
                python,
            ]
            if output.exists():
                command.append("--resume")
            log_path = log_root / (
                f"gpu{index}_task{spec.task_id:02d}_replicate{spec.replicate_id:02d}.log"
            )
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            with log_path.open("x", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[GPU{index}] {line}", end="", flush=True)
            returncode = process.wait()
            schedule_queue.task_done()
            if returncode:
                with failure_lock:
                    failures.append(
                        {
                            "task_id": spec.task_id,
                            "replicate_id": spec.replicate_id,
                            "gpu_index": index,
                            "gpu_uuid": uuid,
                            "returncode": returncode,
                            "log": str(log_path.relative_to(REPO_ROOT)),
                        }
                    )
                stop.set()
                return

    threads = [
        threading.Thread(target=worker, args=binding, daemon=False)
        for binding in gpu_bindings
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    launch_result = {
        "status": "PASS_STAGE10_LAUNCH_COMPLETE" if not failures else "ABORT_STAGE10_LAUNCH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "gpu_bindings": [
            {"physical_index": index, "uuid": uuid} for index, uuid in gpu_bindings
        ],
        "remaining_unlaunched_triplets": schedule_queue.qsize(),
        "failures": failures,
        "interim_metrics_computed": False,
        "outcome_based_retry": False,
    }
    result_path = log_root / "launch_result.json"
    result_path.write_text(
        json.dumps(launch_result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(launch_result, ensure_ascii=False, indent=2, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
