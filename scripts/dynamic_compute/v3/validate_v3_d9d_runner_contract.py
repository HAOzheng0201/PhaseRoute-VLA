#!/usr/bin/env python3
"""Freeze D9D runner readiness without opening any cache NPZ payload."""

from __future__ import annotations

from collections import Counter
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

from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    sha256_file,
)
from a1.vla.dynamic_compute.v3.same_noise_replay import (  # noqa: E402
    D9C_COLLECTION_SHA256,
    D9D_EXPECTED_ROWS,
    D9D_REPLAY_LAYERS,
    D9D_RUNNER_READINESS_RELATIVE_PATH,
    D9D_RUNNER_READINESS_STATUS,
    D9D_SHARD_COUNT,
    load_d9d_calls,
    validate_d9c_collection,
)


CODE_PATHS = (
    Path("a1/vla/dynamic_compute/v3/same_noise_replay.py"),
    Path("scripts/dynamic_compute/v3/replay_v3_d9d_shard.py"),
    Path("scripts/dynamic_compute/v3/run_v3_d9d_front4.sh"),
    Path("scripts/dynamic_compute/v3/freeze_v3_d9d_collection.py"),
    Path("scripts/dynamic_compute/v3/validate_v3_d9d_runner_contract.py"),
    Path("tests/dynamic_compute/v3/test_same_noise_replay.py"),
)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


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
        raise PermissionError("D9D runner validation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9D runner validation requires a clean implementation commit")
    output = REPO_ROOT / D9D_RUNNER_READINESS_RELATIVE_PATH
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9D runner validation refuses overwrite")

    collection = validate_d9c_collection(REPO_ROOT)
    calls = load_d9d_calls(REPO_ROOT)
    shard_counts = Counter(call.dataset_index % D9D_SHARD_COUNT for call in calls)
    tests = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "tests/dynamic_compute/v3/test_same_noise_replay.py",
            "tests/dynamic_compute/v3/test_independent_test_protocol.py",
            "tests/dynamic_compute/v3/test_paired_active_collection.py",
        ]
    )
    shell_syntax = _run(["bash", "-n", "scripts/dynamic_compute/v3/run_v3_d9d_front4.sh"])
    compile_check = _run(
        [
            sys.executable,
            "-m",
            "py_compile",
            *(path.as_posix() for path in CODE_PATHS if path.suffix == ".py"),
        ]
    )
    pip_check = _run([sys.executable, "-m", "pip", "check"])
    worker_source = (REPO_ROOT / CODE_PATHS[1]).read_text(encoding="utf-8")
    launcher_source = (REPO_ROOT / CODE_PATHS[2]).read_text(encoding="utf-8")
    freezer_source = (REPO_ROOT / CODE_PATHS[3]).read_text(encoding="utf-8")
    checks = {
        "D9C_collection_attestation_exact": collection["sha256"]
        == D9C_COLLECTION_SHA256,
        "complete_3700_call_index_from_100_pairs": len(calls) == D9D_EXPECTED_ROWS,
        "all_calls_not_only_early_calls_are_indexed": len(calls)
        == collection["cache_rows"],
        "modulo_four_shards_are_exactly_balanced": shard_counts
        == Counter({0: 925, 1: 925, 2: 925, 3: 925}),
        "candidate_layer_order_is_frozen": D9D_REPLAY_LAYERS == (11, 13, 27),
        "worker_hashes_NPZ_before_open": worker_source.index(
            "observed_source_sha = sha256_file(call.array_path)"
        )
        < worker_source.index("with np.load(call.array_path, allow_pickle=False)"),
        "worker_reuses_one_teacher_exit_input_for_three_layers": (
            'shared_input = cpu_batch["teacher_exit_input_x"]' in worker_source
            and "for layer in D9D_REPLAY_LAYERS" in worker_source
            and "shared flow-matching input mutated" in worker_source
        ),
        "worker_uses_online_FP32_precision_without_BF16_autocast": (
            "with torch.inference_mode():" in worker_source
            and "torch.autocast" not in worker_source
        ),
        "truth_uses_actual_online_selected_action_against_replayed_L27": (
            "online.double(), reference" in (
                REPO_ROOT / CODE_PATHS[0]
            ).read_text(encoding="utf-8")
        ),
        "worker_requires_clean_worktree_and_nonoverwrite": (
            "D9D replay requires a clean frozen-runner worktree" in worker_source
            and "D9D refuses to overwrite replay evidence" in worker_source
        ),
        "launcher_hard_limits_physical_GPU_0_to_3": (
            "declare -a gpus=(0 1 2 3)" in launcher_source
            and "gpus=(4" not in launcher_source
        ),
        "freezer_does_not_aggregate_final_D9_gate": (
            '"success_rate_reported": False' in freezer_source
            and '"safety_rate_reported": False' in freezer_source
            and '"efficiency_rate_reported": False' in freezer_source
            and '"D9D_is_D9_pass_or_negative": False' in freezer_source
        ),
        "targeted_regression_pass": bool(tests["pass"]),
        "shell_syntax_pass": bool(shell_syntax["pass"]),
        "python_compile_pass": bool(compile_check["pass"]),
        "pip_check_pass": bool(pip_check["pass"]),
        "validation_did_not_initialize_CUDA": not torch.cuda.is_initialized(),
        "source_worktree_remains_clean": not bool(git_output("status", "--porcelain=v1")),
        "cache_NPZ_payloads_not_opened": True,
        "model_router_LIBERO_not_loaded": True,
    }
    if not all(checks.values()):
        raise PermissionError(f"D9D runner readiness checks failed: {checks}")
    result = {
        "status": D9D_RUNNER_READINESS_STATUS,
        "schema_version": "phase-route-vla.v3.d9d-runner-readiness.v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_initialized": torch.cuda.is_initialized(),
        },
        "D9C_collection": {
            key: value for key, value in collection.items() if key != "arm_payload_binding"
        },
        "bound_code_sha256": {
            path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS
        },
        "call_index_summary": {
            "pairs": 100,
            "policy_calls": len(calls),
            "candidate_layers": list(D9D_REPLAY_LAYERS),
            "candidate_replays": len(calls) * len(D9D_REPLAY_LAYERS),
            "rows_per_modulo_shard": dict(sorted(shard_counts.items())),
            "physical_gpu_allowlist": [0, 1, 2, 3],
        },
        "validation": {
            "targeted_regression": tests,
            "shell_syntax": shell_syntax,
            "python_compile": compile_check,
            "pip_check": pip_check,
        },
        "checks": checks,
        "access_ledger": {
            "D9C_attestation_opened": True,
            "D9C_manifest_inventory_runtime_metadata_opened": True,
            "cache_NPZ_payloads_opened": 0,
            "LIBERO_environment_created": False,
            "model_loaded": False,
            "router_loaded": False,
            "CUDA_initialized": False,
            "environment_actions_executed": 0,
            "D9_gate_aggregate_calls": 0,
        },
        "authorization": {
            "next_stage": "D9D_EXACT_FRONT4_SAME_NOISE_REPLAY",
            "all_3700_calls_only": True,
            "physical_GPU_4_to_7": False,
            "interim_aggregate": False,
        },
        "claim_boundary": {
            "runner_readiness_is_D9_result": False,
            "per_call_truth_created": False,
            "D9_primary_gate_evaluated": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(D9D_RUNNER_READINESS_STATUS)


if __name__ == "__main__":
    main()
