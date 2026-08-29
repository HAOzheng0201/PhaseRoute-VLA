#!/usr/bin/env python3
"""Freeze the tested Stage 10 runner before any fresh-state deserialization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402


PROTOCOL_SHA256 = CONTRACT.PROTOCOL_SHA256
SCHEDULE_SHA256 = CONTRACT.SCHEDULE_SHA256
STATE_BINDING_SHA256 = CONTRACT.STATE_BINDING_SHA256
STATE_RESULT_SHA256 = CONTRACT.STATE_RESULT_SHA256
sha256_file = CONTRACT.sha256_file
validate_local_state_artifacts = CONTRACT.validate_local_state_artifacts
PROTECTED_CODE_SHA256 = ACTIVE.PROTECTED_CODE_SHA256
RUNNER_READINESS_RELATIVE_PATH = ACTIVE.RUNNER_READINESS_RELATIVE_PATH
RUNNER_READINESS_SCHEMA = ACTIVE.RUNNER_READINESS_SCHEMA
RUNNER_READINESS_STATUS = ACTIVE.RUNNER_READINESS_STATUS


BOUND_CODE = (
    "a1/vla/dynamic_compute/route_first_stage10.py",
    "a1/vla/dynamic_compute/route_first_stage10_active.py",
    "a1/vla/dynamic_compute/route_first_controller.py",
    "a1/vla/dynamic_compute/route_first_runtime.py",
    "a1/vla/dynamic_compute/productive_exit.py",
    "a1/vla/dynamic_compute/stage1_measurement.py",
    "a1/vla/dynamic_compute/telemetry.py",
    "a1/vla/dynamic_compute/v3/active_runtime.py",
    "a1/vla/dynamic_compute/v3/runtime_adapter.py",
    "robot_experiments/libero/stage1_vla_utils.py",
    "scripts/_route_first_stage10_contracts.py",
    "scripts/run_route_first_stage10_arm.py",
    "scripts/run_route_first_stage10_triplet.py",
    "scripts/launch_route_first_stage10_active.py",
    "scripts/validate_route_first_stage10_preflight.py",
    "scripts/validate_route_first_stage10_postflight.py",
    "scripts/validate_route_first_stage10_arm.py",
    "scripts/aggregate_route_first_stage10_active.py",
    "scripts/freeze_route_first_stage10_runner_readiness.py",
    "tests/dynamic_compute/test_route_first_stage10_protocol.py",
    "tests/dynamic_compute/test_route_first_stage10_active.py",
    *PROTECTED_CODE_SHA256.keys(),
)

EXPECTED_ARTIFACTS = {
    "model/libero_exit/model.pt": {
        "bytes": 33_841_175_207,
        "sha256": "dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f",
        "rehash_each_preflight": False,
    },
    "model/libero_exit/config.yaml": {
        "bytes": 8_369,
        "sha256": "9365d0a6ca6379a7787aaf46e170a7945f084c359560463edc14726965b04ca",
        "rehash_each_preflight": True,
    },
    "model/libero_exit/dataset_statistics.json": {
        "bytes": 11_871,
        "sha256": "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3",
        "rehash_each_preflight": True,
    },
    "model/libero_exit/exit_thresholds_libero_10_exp_1.0.json": {
        "bytes": 236,
        "sha256": "a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796",
        "rehash_each_preflight": True,
    },
    "artifacts/phase_route_v3/final_router.pt": {
        "bytes": 22_290,
        "sha256": "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830",
        "rehash_each_preflight": True,
    },
    "artifacts/phase_route_v3/phase_estimator.pt": {
        "bytes": 11_344_688,
        "sha256": "b601f8221d47136818d7a008eaf7cee06e1201bf514f371ae33f42cfb39515a1",
        "rehash_each_preflight": True,
    },
    "runs/route_first_calibration_stage6/router_calibrated.npz": {
        "sha256": "ae561b77c01bd4c7eee6cc0ff91e215733662544cc1af2e5039b0a8f02c60cc2",
        "rehash_each_preflight": True,
    },
    "results/route_first/route_first_stage7_holdout.json": {
        "sha256": "d9780a5e4765ee9a80165eb790b99b4e9e85fcb1ae6d34ae006ddb72ce48f258",
        "rehash_each_preflight": True,
    },
    "configs/route_first_stage10_fresh_confirmation_protocol.json": {
        "sha256": PROTOCOL_SHA256,
        "rehash_each_preflight": True,
    },
    "configs/route_first_stage10_fresh_schedule.json": {
        "sha256": SCHEDULE_SHA256,
        "rehash_each_preflight": True,
    },
    "configs/route_first_stage10_fresh_state_binding.json": {
        "sha256": STATE_BINDING_SHA256,
        "rehash_each_preflight": True,
    },
    "results/route_first/route_first_stage10_fresh_states.json": {
        "sha256": STATE_RESULT_SHA256,
        "rehash_each_preflight": True,
    },
}


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "PYTHONNOUSERSITE": "1",
            "DATA_DIR": str(REPO_ROOT),
            "VLA_CONFIG_YAML": "libero_simulation.yaml",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": output[-12_000:],
        "pass": completed.returncode == 0,
    }


def main() -> None:
    output = REPO_ROOT / RUNNER_READINESS_RELATIVE_PATH
    temporary = output.with_name(output.name + ".incomplete")
    if output.exists() or temporary.exists():
        raise FileExistsError("runner readiness refuses to overwrite evidence")
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip()
    if worktree:
        raise PermissionError("runner readiness requires a clean implementation commit")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    local = validate_local_state_artifacts(REPO_ROOT)
    # Bind ignored local files after exact byte validation, but never call
    # torch.load here.  Scientific active access starts only in the arm runner.
    expected = dict(EXPECTED_ARTIFACTS)
    expected[local["binding"]["local_state_attestation"]["path"]] = {
        "sha256": local["binding"]["local_state_attestation"]["sha256"],
        "rehash_each_preflight": True,
    }
    expected[local["binding"]["local_state_payload"]["path"]] = {
        "bytes": local["binding"]["local_state_payload"]["bytes"],
        "sha256": local["binding"]["local_state_payload"]["sha256"],
        "rehash_each_preflight": True,
    }
    bound_artifacts = {}
    artifact_checks = {}
    for relative, specification in expected.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Stage 10 artifact is missing: {relative}")
        stat = path.stat()
        digest = sha256_file(path)
        expected_size = specification.get("bytes", stat.st_size)
        matches = bool(
            stat.st_size == expected_size and digest == specification["sha256"]
        )
        artifact_checks[relative] = matches
        bound_artifacts[relative] = {
            "bytes": stat.st_size,
            "sha256": digest,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
            "rehash_each_preflight": specification["rehash_each_preflight"],
        }
    bound_code = {relative: sha256_file(REPO_ROOT / relative) for relative in BOUND_CODE}
    protected_exact = all(
        bound_code.get(relative) == digest
        for relative, digest in PROTECTED_CODE_SHA256.items()
    )
    python = sys.executable
    targeted = _run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/dynamic_compute/test_route_first_stage10_protocol.py",
            "tests/dynamic_compute/test_route_first_stage10_active.py",
        ]
    )
    full = _run([python, "-m", "pytest", "-q", "tests"])
    compile_check = _run(
        [python, "-m", "py_compile", *[str(REPO_ROOT / item) for item in BOUND_CODE if item.endswith(".py")]]
    )
    diff_check = _run(["git", "diff", "--check"])
    pip_check = _run([python, "-m", "pip", "check"])
    clean_after = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip() == ""
    checks = {
        "source_worktree_clean": not worktree,
        "protected_historical_code_exact": protected_exact,
        "all_bound_artifacts_exact": all(artifact_checks.values()),
        "state_payload_bytes_validated_not_deserialized": True,
        "targeted_contract_tests_pass": targeted["pass"],
        "full_tests_directory_pass": full["pass"],
        "python_compile_pass": compile_check["pass"],
        "git_diff_check_pass": diff_check["pass"],
        "pip_check_pass": pip_check["pass"],
        "validation_did_not_initialize_cuda": True,
        "source_worktree_remains_clean": clean_after,
        "active_rollouts_not_started": True,
    }
    result = {
        "schema_version": RUNNER_READINESS_SCHEMA,
        "status": RUNNER_READINESS_STATUS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": source_commit,
        "source_worktree_dirty": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "bound_code_sha256": bound_code,
        "bound_artifacts": bound_artifacts,
        "artifact_checks": artifact_checks,
        "validation": {
            "targeted_contract_tests": targeted,
            "full_tests_directory": full,
            "python_compile": compile_check,
            "git_diff_check": diff_check,
            "pip_check": pip_check,
        },
        "checks": checks,
        "access_ledger": {
            "fresh_state_payload_opened": False,
            "fresh_state_payload_bytes_hashed": True,
            "fresh_state_payload_deserialized": False,
            "LIBERO_environment_created": False,
            "model_loaded": False,
            "CUDA_initialized": False,
            "active_rollouts": 0,
        },
        "authorization": {
            "next_stage": "EXACT_STAGE10_60_TRIPLET_180_ARM_ACTIVE_CONFIRMATION",
            "minimum_free_memory_mib_before_each_arm": 40_000,
            "same_gpu_uuid_within_triplet": True,
            "outcome_based_retry": False,
            "deployment_authorized": False,
        },
        "claim_boundary": {
            "runner_readiness_is_active_result": False,
            "fresh_state_payload_deserialized": False,
            "active_control_has_run": False,
            "stage10_gate_evaluated": False,
        },
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 10 runner readiness checks failed: {checks}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
