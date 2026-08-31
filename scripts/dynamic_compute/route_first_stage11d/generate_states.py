#!/usr/bin/env python3
"""Run 400 isolated Stage-11D reset-state processes and retain all logs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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

from a1.vla.dynamic_compute.route_first_reliability import (  # noqa: E402
    STAGE11D_PROTOCOL_SHA256,
    build_stage11d_schedule,
    validate_stage11d_protocol,
)
from a1.vla.dynamic_compute.route_first_reliability_artifacts import (  # noqa: E402
    STAGE11D_STATE_PASSES,
    STAGE11D_STATE_RECORDS_RELATIVE_PATH,
    STAGE11D_STATE_RUN_SCHEMA,
    sha256_file,
    validate_state_runner_readiness,
)


WORKER = Path(__file__).with_name("generate_state_record.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _worker_command(pass_id: int, task_id: int, replicate_id: int, root: Path) -> list[str]:
    output = root / f"pass{pass_id}" / f"task{task_id:02d}_replicate{replicate_id:02d}"
    return [
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


def _run_one(pass_id: int, record, root: Path) -> dict[str, object]:
    command = _worker_command(pass_id, record.task_id, record.replicate_id, root)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MUJOCO_GL": "osmesa",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(record.state_seed),
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
    log = (
        root
        / "logs"
        / f"pass{pass_id}_task{record.task_id:02d}_replicate{record.replicate_id:02d}.log"
    )
    log.write_text(shlex.join(command) + "\n\n" + completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"Stage-11D state failed pass={pass_id} task={record.task_id} "
            f"replicate={record.replicate_id}; retained log={log}"
        )
    result_path = Path(command[-1]) / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "pass_id": pass_id,
        "task_id": record.task_id,
        "replicate_id": record.replicate_id,
        "result_sha256": sha256_file(result_path),
        "state_sha256": result["state_sha256"],
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("Stage-11D state runner is CPU-only")
    if not 1 <= args.max_workers <= 32:
        raise ValueError("Stage-11D max workers must be in 1..32")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("Stage-11D state runner requires clean worktree")
    protocol = validate_stage11d_protocol(REPO_ROOT)
    readiness = validate_state_runner_readiness(REPO_ROOT)
    schedule = build_stage11d_schedule()
    output = REPO_ROOT / STAGE11D_STATE_RECORDS_RELATIVE_PATH
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("Stage-11D refuses to overwrite state records")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_ROUTE_FIRST_STAGE11D_STATE_RUN_PREFLIGHT",
                    "protocol_sha256": protocol["protocol_sha256"],
                    "readiness_status": readiness["status"],
                    "records_per_pass": len(schedule),
                    "passes": list(STAGE11D_STATE_PASSES),
                    "isolated_processes": len(schedule) * len(STAGE11D_STATE_PASSES),
                    "max_workers": args.max_workers,
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
    for pass_id in STAGE11D_STATE_PASSES:
        futures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for record in schedule:
                futures.append(executor.submit(_run_one, pass_id, record, incomplete))
            for index, future in enumerate(as_completed(futures), start=1):
                completed_records.append(future.result())
                if index % 20 == 0 or index == len(futures):
                    print(
                        f"Stage11D pass {pass_id}: {index}/{len(futures)} complete",
                        flush=True,
                    )
    result = {
        "schema_version": STAGE11D_STATE_RUN_SCHEMA,
        "status": "PASS_ROUTE_FIRST_STAGE11D_STATE_RUN_PENDING_AGGREGATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
        "runner_readiness_sha256": sha256_file(
            REPO_ROOT
            / "results/route_first/route_first_stage11d_state_runner_readiness.json"
        ),
        "processes": len(completed_records),
        "generation_passes": len(STAGE11D_STATE_PASSES),
        "records_per_pass": len(schedule),
        "max_parallel_processes": args.max_workers,
        "gpu_query_or_initialization": 0,
        "model_checkpoint_loaded": False,
        "policy_action_sampled": False,
        "official_states_0_to_49_opened": False,
        "historical_generated_state_payload_opened": False,
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
        f"{sha256_file(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
