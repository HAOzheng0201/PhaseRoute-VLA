#!/usr/bin/env python3
"""Launch all ten Stage-11D original-A1 development collection tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_reliability_collection import (  # noqa: E402
    STAGE11D_COLLECTION_CLUSTER_COUNT,
    STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH,
    development_schedule,
    load_development_states,
    validate_collection_readiness,
)


LOG_ROOT = Path("runs/route_first_stage11d_development_launch_logs")
WORKER = Path(
    "scripts/dynamic_compute/route_first_stage11d/collect_original_a1_task.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--maximum-memory-used-mib", type=int, default=500)
    parser.add_argument("--maximum-utilization-percent", type=int, default=5)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=40_000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    records = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise RuntimeError("Unexpected nvidia-smi inventory row")
        records.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_used_mib": int(parts[3]),
                "memory_total_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )
    if tuple(record["index"] for record in records) != tuple(range(8)):
        raise RuntimeError("Stage-11D expected physical GPUs 0..7")
    return records


def _select_gpus(inventory: list[dict[str, Any]], args: argparse.Namespace):
    idle = [
        record
        for record in inventory
        if record["memory_used_mib"] <= args.maximum_memory_used_mib
        and record["utilization_percent"] <= args.maximum_utilization_percent
        and record["memory_total_mib"] - record["memory_used_mib"]
        >= args.minimum_free_memory_mib
    ]
    idle.sort(key=lambda record: (record["memory_used_mib"], record["index"]))
    selected = idle[: args.max_parallel]
    if not selected:
        raise RuntimeError("No idle GPU satisfies the Stage-11D launch gates")
    return selected


def _worker_environment(gpu: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    hf_home = environment.get("HF_HOME") or str(REPO_ROOT.parent / "hf_cache")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "MUJOCO_EGL_DEVICE_ID": str(gpu["index"]),
            "DATA_DIR": str(REPO_ROOT),
            "HF_HOME": hf_home,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLA_CONFIG_YAML": "libero_simulation.yaml",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    prefixes = [
        str(REPO_ROOT / "robot_experiments/libero/LIBERO"),
        str(REPO_ROOT),
    ]
    if environment.get("PYTHONPATH"):
        prefixes.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(prefixes)
    return environment


def _worker_command(
    task_id: int, gpu: dict[str, Any], *, minimum_free_memory_mib: int
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / WORKER),
        "--task-id",
        str(task_id),
        "--physical-gpu-index",
        str(gpu["index"]),
        "--expected-gpu-uuid",
        str(gpu["uuid"]),
        "--minimum-free-memory-mib",
        str(minimum_free_memory_mib),
        "--output-dir",
        str(
            REPO_ROOT
            / STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH
            / f"task{task_id:02d}"
        ),
    ]


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_parallel <= 8:
        raise ValueError("--max-parallel must be in 1..8")
    if _git("status", "--porcelain=v1"):
        raise PermissionError("Stage-11D launch requires a clean worktree")
    readiness = validate_collection_readiness(REPO_ROOT)
    schedule, _states, attestation = load_development_states(REPO_ROOT)
    selected = development_schedule(schedule)
    output_root = REPO_ROOT / STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH
    log_root = REPO_ROOT / LOG_ROOT
    output_absent = not output_root.exists()
    logs_absent = not log_root.exists()
    if not output_absent or not logs_absent:
        raise FileExistsError("Stage-11D launch refuses existing outputs or logs")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_ROUTE_FIRST_STAGE11D_COLLECTION_LAUNCH_PREFLIGHT",
                    "source_git_commit": _git("rev-parse", "HEAD"),
                    "readiness_status": readiness["status"],
                    "state_payload_sha256": attestation["payload_sha256"],
                    "development_clusters": len(selected),
                    "tasks": list(range(10)),
                    "replicates_per_task": 12,
                    "max_parallel": args.max_parallel,
                    "output_absent": output_absent,
                    "logs_absent": logs_absent,
                    "GPU_queried": False,
                    "model_loaded": False,
                    "LIBERO_environment_opened": False,
                    "collection_started": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return
    if len(selected) != STAGE11D_COLLECTION_CLUSTER_COUNT:
        raise RuntimeError("Stage-11D development cluster count differs")

    inventory = _gpu_inventory()
    gpus = _select_gpus(inventory, args)
    log_root.mkdir(parents=True, exist_ok=False)
    (log_root / "source_git_commit.txt").write_text(
        _git("rev-parse", "HEAD") + "\n", encoding="utf-8"
    )
    (log_root / "gpu_preflight.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed: list[dict[str, Any]] = []
    for batch_start in range(0, 10, len(gpus)):
        batch_tasks = list(range(batch_start, min(batch_start + len(gpus), 10)))
        processes = []
        for task_id, gpu in zip(
            batch_tasks, gpus[: len(batch_tasks)], strict=True
        ):
            log_path = log_root / f"task{task_id:02d}.log"
            log_file = log_path.open("xb")
            command = _worker_command(
                task_id,
                gpu,
                minimum_free_memory_mib=args.minimum_free_memory_mib,
            )
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=_worker_environment(gpu),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append((task_id, gpu, process, log_file, log_path, command))
            print(
                f"started task={task_id} gpu={gpu['index']} pid={process.pid} "
                f"log={log_path}",
                flush=True,
            )
        while any(process.poll() is None for _, _, process, *_ in processes):
            alive = sum(process.poll() is None for _, _, process, *_ in processes)
            print(f"batch_start={batch_start} alive_workers={alive}", flush=True)
            time.sleep(15)
        failed = False
        for task_id, gpu, process, log_file, log_path, command in processes:
            log_file.close()
            completed.append(
                {
                    "task_id": task_id,
                    "physical_gpu_index": gpu["index"],
                    "gpu_uuid": gpu["uuid"],
                    "return_code": process.returncode,
                    "log": str(log_path.relative_to(REPO_ROOT)),
                    "command": command,
                }
            )
            if process.returncode != 0:
                failed = True
                print(f"failed task={task_id}; retained {log_path}", flush=True)
            else:
                print(f"completed task={task_id}", flush=True)
        if failed:
            raise RuntimeError(
                "Stage-11D collection stopped after worker failure; no retry allowed"
            )

    result = {
        "status": "PASS_ROUTE_FIRST_STAGE11D_COLLECTION_LAUNCH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": _git("rev-parse", "HEAD"),
        "readiness_status": readiness["status"],
        "state_payload_sha256": attestation["payload_sha256"],
        "development_clusters": len(selected),
        "workers": completed,
        "access_ledger": {
            "development_tasks_launched": 10,
            "calibration_tasks_launched": 0,
            "shadow_confirmation_tasks_launched": 0,
            "same_noise_replay_launched": False,
            "training_launched": False,
            "new_router_active_control_launched": False,
        },
    }
    (log_root / "launch_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS_ROUTE_FIRST_STAGE11D_COLLECTION_LAUNCH", flush=True)


if __name__ == "__main__":
    main()
