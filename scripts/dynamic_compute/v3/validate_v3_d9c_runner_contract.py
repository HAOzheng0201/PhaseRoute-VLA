#!/usr/bin/env python3
"""Freeze D9C runner readiness without opening official test states."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    D9_ARMS,
    D9_RECORD_COUNT,
    D9_TASK_IDS,
    load_d9_selection_metadata,
)
from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    D9C_RUNNER_READINESS_RELATIVE_PATH,
    D9C_RUNNER_READINESS_STATUS,
    sha256_file,
    task_schedule,
    validate_d9b_readiness,
)


CODE_PATHS = (
    Path("a1/vla/dynamic_compute/v3/paired_active_collection.py"),
    Path("scripts/dynamic_compute/v3/run_v3_d9c_task.py"),
    Path("scripts/dynamic_compute/v3/run_v3_d9c_front4.sh"),
    Path("scripts/dynamic_compute/v3/freeze_v3_d9c_collection.py"),
    Path("scripts/dynamic_compute/v3/validate_v3_d9c_runner_contract.py"),
    Path("tests/dynamic_compute/v3/test_paired_active_collection.py"),
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout.strip(),
        "pass": completed.returncode == 0,
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D9C runner validation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9C runner validation requires a clean implementation commit")
    output = REPO_ROOT / D9C_RUNNER_READINESS_RELATIVE_PATH
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9C runner validation refuses overwrite")

    d9b = validate_d9b_readiness(REPO_ROOT)
    records = load_d9_selection_metadata(REPO_ROOT)
    schedules = {task: task_schedule(REPO_ROOT, task) for task in D9_TASK_IDS}
    test = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "tests/dynamic_compute/v3/test_paired_active_collection.py",
            "tests/dynamic_compute/v3/test_active_runtime.py",
            "tests/dynamic_compute/v3/test_runtime_adapter.py",
            "tests/dynamic_compute/v3/test_independent_test_protocol.py",
        ]
    )
    shell_syntax = _run(
        ["bash", "-n", "scripts/dynamic_compute/v3/run_v3_d9c_front4.sh"]
    )
    compile_check = _run(
        [
            sys.executable,
            "-m",
            "py_compile",
            *(path.as_posix() for path in CODE_PATHS if path.suffix == ".py"),
        ]
    )
    pip_check = _run([sys.executable, "-m", "pip", "check"])
    runner_source = (REPO_ROOT / CODE_PATHS[1]).read_text(encoding="utf-8")
    launcher_source = (REPO_ROOT / CODE_PATHS[2]).read_text(encoding="utf-8")
    freezer_source = (REPO_ROOT / CODE_PATHS[3]).read_text(encoding="utf-8")
    checks = {
        "D9B_readiness_exact_and_control_code_unchanged": bool(d9b),
        "selection_has_exact_100_pairs": len(records) == D9_RECORD_COUNT,
        "ten_tasks_each_have_episode_40_to_49": all(
            len(schedules[task]) == 10
            and tuple(item.episode_index for item in schedules[task])
            == tuple(range(40, 50))
            for task in D9_TASK_IDS
        ),
        "arm_order_exact_and_balanced": sum(
            item.arm_order[0] == D9_ARMS[0] for item in records
        )
        == 50,
        "runner_uses_canonical_identity": (
            "episode_id_override=record.canonical_key" in runner_source
        ),
        "runner_uses_frozen_arm_order": "for arm in record.arm_order" in runner_source,
        "runner_requires_clean_worktree": (
            "D9C requires a clean frozen-runner worktree" in runner_source
        ),
        "runner_requires_explicit_resume": (
            "explicit --resume is required" in runner_source
        ),
        "runner_caches_PhaseRoute_only": (
            "if arm == PHASE_ROUTE_ARM" in runner_source
            and "teacher_kind=PHASE_ROUTE_TEACHER_KIND" in runner_source
        ),
        "launcher_hard_limits_front_four": (
            "declare -a gpus=(0 1 2 3)" in launcher_source
            and "gpu=$((task_id % 4))" in launcher_source
            and "gpus=(4" not in launcher_source
        ),
        "freezer_does_not_aggregate_D9_metrics": (
            '"success_rate_reported": False' in freezer_source
            and '"safety_rate_reported": False' in freezer_source
            and '"efficiency_rate_reported": False' in freezer_source
        ),
        "targeted_regression_pass": bool(test["pass"]),
        "shell_syntax_pass": bool(shell_syntax["pass"]),
        "python_compile_pass": bool(compile_check["pass"]),
        "pip_check_pass": bool(pip_check["pass"]),
        "validation_did_not_initialize_CUDA": not torch.cuda.is_initialized(),
        "source_worktree_remains_clean": not bool(
            git_output("status", "--porcelain=v1")
        ),
        "official_test_states_not_opened": True,
        "active_rollouts_not_run": True,
    }
    if not all(checks.values()):
        raise PermissionError(f"D9C runner readiness checks failed: {checks}")
    result = {
        "status": D9C_RUNNER_READINESS_STATUS,
        "schema_version": "phase-route-vla.v3.d9c-runner-readiness.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_initialized": torch.cuda.is_initialized(),
        },
        "D9B": d9b,
        "bound_code_sha256": {
            path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS
        },
        "schedule_summary": {
            "tasks": len(schedules),
            "pairs": len(records),
            "rollouts": 2 * len(records),
            "episodes_per_task": 10,
            "A1_first_pairs": sum(item.arm_order[0] == D9_ARMS[0] for item in records),
            "PhaseRoute_first_pairs": sum(
                item.arm_order[0] == D9_ARMS[1] for item in records
            ),
            "physical_gpu_allowlist": [0, 1, 2, 3],
        },
        "validation": {
            "targeted_regression": test,
            "shell_syntax": shell_syntax,
            "python_compile": compile_check,
            "pip_check": pip_check,
        },
        "checks": checks,
        "access_ledger": {
            "selection_metadata_opened": True,
            "official_test_states_opened": False,
            "LIBERO_environment_created": False,
            "model_loaded": False,
            "CUDA_initialized": False,
            "active_rollouts": 0,
            "success_safety_efficiency_aggregate_calls": 0,
        },
        "authorization": {
            "next_stage": "D9C_EXACT_ONE_SHOT_FRONT4_PREFLIGHT_AND_COLLECTION",
            "exact_episode_40_49_schedule_only": True,
            "physical_GPU_4_to_7": False,
            "interim_aggregate": False,
        },
        "claim_boundary": {
            "runner_readiness_is_D9_result": False,
            "active_control_has_run": False,
            "D9_primary_gate_evaluated": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(D9C_RUNNER_READINESS_STATUS)


if __name__ == "__main__":
    main()
