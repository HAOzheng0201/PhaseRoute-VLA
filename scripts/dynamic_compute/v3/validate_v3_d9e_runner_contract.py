#!/usr/bin/env python3
"""Freeze D9E one-shot aggregation readiness without opening outcomes/truth."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    D9_CONTRACT_SHA256,
    load_d9_contract,
)
from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    sha256_file,
)
from a1.vla.dynamic_compute.v3.same_noise_replay import (  # noqa: E402
    D9C_COLLECTION_SHA256,
)


OUTPUT = Path("results/v3/v3_d9e_runner_readiness.json")
STATUS = "PASS_V3_D9E_FROZEN_AGGREGATE_RUNNER_READINESS"
D9C_ATTESTATION = Path("results/v3/v3_d9c_collection_attestation.json")
D9D_ATTESTATION = Path("results/v3/v3_d9d_collection_attestation.json")
D9D_ATTESTATION_SHA256 = (
    "f8b3421948ca6c8ccfda6837afde9cfec0a7dbd6cee61987eb03e2dee2f6ea65"
)
CODE_PATHS = (
    Path("a1/vla/dynamic_compute/v3/independent_test_aggregate.py"),
    Path("scripts/dynamic_compute/v3/aggregate_v3_d9e_final.py"),
    Path("scripts/dynamic_compute/v3/validate_v3_d9e_runner_contract.py"),
    Path("tests/dynamic_compute/v3/test_independent_test_aggregate.py"),
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _run(command: list[str]) -> dict[str, Any]:
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
        raise PermissionError("D9E runner validation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError(
            "D9E runner validation requires a clean implementation commit"
        )
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9E runner validation refuses overwrite")

    load_d9_contract(REPO_ROOT)
    d9c_sha = sha256_file(REPO_ROOT / D9C_ATTESTATION)
    d9d_sha = sha256_file(REPO_ROOT / D9D_ATTESTATION)
    tests = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "tests/dynamic_compute/v3/test_independent_test_aggregate.py",
            "tests/dynamic_compute/v3/test_independent_test_protocol.py",
            "tests/dynamic_compute/v3/test_paired_active_collection.py",
            "tests/dynamic_compute/v3/test_same_noise_replay.py",
        ]
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
    aggregate_source = (REPO_ROOT / CODE_PATHS[0]).read_text(encoding="utf-8")
    runner_source = (REPO_ROOT / CODE_PATHS[1]).read_text(encoding="utf-8")
    checks = {
        "D9_contract_exact": D9_CONTRACT_SHA256
        == "eea74662357d39737a3ac84b2d59059150ac4f098c6bddbfe695ba1ed64e59d3",
        "D9C_collection_attestation_SHA_exact": d9c_sha == D9C_COLLECTION_SHA256,
        "D9D_collection_attestation_SHA_exact": d9d_sha
        == D9D_ATTESTATION_SHA256,
        "bootstrap_resamples_seed_and_linear_quantile_frozen": all(
            token in aggregate_source
            for token in (
                "D9_BOOTSTRAP_RESAMPLES = 100_000",
                "D9_BOOTSTRAP_SEED = 60_260_821",
                'np.quantile(samples, D9_BOOTSTRAP_PERCENTILE, method="linear")',
            )
        ),
        "safe_cluster_is_unique_episode_with_real_early_exit": (
            "early_keys.add(truth.canonical_key)" in aggregate_source
        ),
        "false_safe_is_any_full_or_gripper_unsafe_early_call": (
            "unsafe = truth.full_action_unsafe or truth.gripper_unsafe"
            in aggregate_source
        ),
        "head_range_denominator_is_all_evaluated_runtime_candidate_rows": (
            "nondegenerate / len(head_ranges)" in aggregate_source
            and "fraction_above_1e_6_all_runtime_candidate_rows"
            in aggregate_source
        ),
        "FM_efficiency_uses_each_arm_FM_calls_per_policy_call": (
            "1.0 - phase_fm_per_call / a1_fm_per_call" in aggregate_source
        ),
        "all_primary_gate_checks_are_conjunctive": (
            "if all(gate_checks.values())" in aggregate_source
        ),
        "one_cross_pair_aggregate_call_in_runner": runner_source.count(
            "aggregate_independent_test(pairs, truths)"
        )
        == 1,
        "runner_is_CPU_only_clean_and_nonoverwriting": all(
            token in runner_source
            for token in (
                'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"',
                "D9E requires a clean frozen-runner worktree",
                "D9E refuses to overwrite one-shot evidence",
            )
        ),
        "runner_authenticates_D9C_D9D_and_readiness_before_outcome_load": (
            runner_source.index("readiness = _readiness()")
            < runner_source.index("pairs, pair_records = _load_pairs(d9c)")
            and runner_source.index("d9c, d9d = _authenticate_attestations()")
            < runner_source.index("truths = _load_truths(d9d)")
        ),
        "runner_reports_early_exit_failure_without_causal_claim": (
            '"early_exit_and_failure_cooccurrence_proves_causation": False'
            in runner_source
        ),
        "runner_records_missing_online_router_predict_latency": (
            '"five_head_router_predict_CPU_latency_ms": None' in aggregate_source
            and '"five_head_router_predict_latency_not_instrumented_online": True'
            in aggregate_source
        ),
        "targeted_synthetic_regression_pass": bool(tests["pass"]),
        "python_compile_pass": bool(compile_check["pass"]),
        "pip_check_pass": bool(pip_check["pass"]),
        "validation_did_not_initialize_CUDA": not torch.cuda.is_initialized(),
        "source_worktree_remains_clean": not bool(
            git_output("status", "--porcelain=v1")
        ),
        "D9C_success_values_not_opened": True,
        "D9D_truth_payloads_not_opened": True,
    }
    if not all(checks.values()):
        raise PermissionError(f"D9E runner readiness checks failed: {checks}")
    result = {
        "status": STATUS,
        "schema_version": "phase-route-vla.v3.d9e-runner-readiness.v1",
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
        "input_attestation_sha256": {
            D9C_ATTESTATION.as_posix(): d9c_sha,
            D9D_ATTESTATION.as_posix(): d9d_sha,
        },
        "bound_code_sha256": {
            path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS
        },
        "frozen_statistical_definitions": {
            "pairs": 100,
            "pairs_per_task": 10,
            "bootstrap_resamples": 100_000,
            "bootstrap_seed": 60_260_821,
            "bootstrap_percentile": 0.05,
            "bootstrap_quantile_method": "numpy_linear",
            "safe_cluster": "unique task_episode with at least one selected L11_or_L13 call",
            "false_safe_cluster": "safe cluster with any full_action_or_gripper_unsafe early call",
            "head_range_denominator": "all actually evaluated L11_or_L13 runtime candidate rows",
            "efficiency": "one_minus_PhaseRoute_FM_per_call_divided_by_A1_FM_per_call",
            "all_primary_criteria_conjunctive": True,
        },
        "validation": {
            "targeted_regression": tests,
            "python_compile": compile_check,
            "pip_check": pip_check,
        },
        "checks": checks,
        "access_ledger": {
            "D9_contract_opened": True,
            "D9C_attestation_bytes_hashed": True,
            "D9D_attestation_bytes_hashed": True,
            "D9C_attestation_JSON_opened": False,
            "D9C_pair_or_arm_files_opened": 0,
            "D9C_success_values_opened": False,
            "D9D_attestation_JSON_opened": False,
            "D9D_truth_payloads_opened": 0,
            "cross_pair_aggregate_calls": 0,
            "D9_primary_gate_calls": 0,
            "LIBERO_environment_created": False,
            "model_or_router_loaded": False,
            "CUDA_initialized": False,
        },
        "authorization": {
            "next_stage": "D9E_ONE_SHOT_FINAL_AGGREGATE",
            "exact_D9C_and_D9D_evidence_only": True,
            "second_independent_test_or_result_tuning": False,
        },
        "claim_boundary": {
            "readiness_is_D9_result": False,
            "success_safety_efficiency_seen": False,
            "D9_primary_gate_evaluated": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(STATUS)


if __name__ == "__main__":
    main()
