#!/usr/bin/env python3
"""Freeze D9B only after current-code parity and a real-model dry-run pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

from a1.vla.dynamic_compute.v3.active_runtime import sha256_file  # noqa: E402
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_ACTION_DELTA_SHA256,
    D2_CHECKPOINT_CONFIG_SHA256,
    D2_CHECKPOINT_SHA256,
    D2_DATASET_STATISTICS_SHA256,
    D2_EXIT_THRESHOLDS_SHA256,
    D2_MODEL_ATTESTATION_SHA256,
    D2_PHASE_CHECKPOINT_SHA256,
    validate_runtime_model_directory,
)
from a1.vla.dynamic_compute.v3.final_router import (  # noqa: E402
    final_router_from_mapping,
)
from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    D9_CONTRACT_RELATIVE_PATH,
    D9_CONTRACT_SHA256,
    D9_SELECTION_RELATIVE_PATH,
    D9_SELECTION_SHA256,
    load_d9_contract,
)
from a1.vla.dynamic_compute.v3.runtime_adapter import (  # noqa: E402
    D9A_RUNTIME_STATUS,
    frozen_router_sha256,
    route_cached_candidate_pairs,
)


SCHEMA_VERSION = "phase-route-vla.v3.d9b-readiness-attestation.v1"
PASS_STATUS = "PASS_V3_D9B_READINESS_FOR_ONE_SHOT_PAIRED_ACTIVE_TEST"
OUTPUT = Path("results/v3/v3_d9b_readiness_attestation.json")
D9A_RESULT = Path("results/v3/v3_d9a_runtime_adapter_validation.json")
D9A_RESULT_SHA256 = (
    "fbf450a2beaab07e558e8e6d961bf7799b080e4afe98626d7ff477343d434acf"
)
CONTEXT = Path("reports/v3_d8_fresh_context/fresh_context.pt")
DATASET = Path("reports/v3_d8_fresh_dataset/fresh_confirmation_dataset.pt")
ROUTER = Path("reports/v3_d8_final_router/final_router.pt")
SCORING = Path("reports/v3_d8_confirmation/confirmation_scoring.pt")
EXPECTED_INPUT_SHA256 = {
    CONTEXT: "3941ea81f1387da819f5ab9c12612bb3aa954d90d2b7e26dd9a7dfc3994b3785",
    DATASET: "411b3d68b2e4326573722a616b5fcf7862fbcc6b85f499be7cdf0877a8889327",
    ROUTER: "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830",
    SCORING: "b225ebec9bfd55044a5b856dd09ad9b5b14278164172d93d525d10309472ffba",
}
CODE_PATHS = (
    Path("a1/vla/dynamic_compute/v3/active_runtime.py"),
    Path("a1/vla/dynamic_compute/v3/runtime_adapter.py"),
    Path("a1/vla/dynamic_compute/v3/development_collection.py"),
    Path("a1/vla/dynamic_compute/v3/final_router.py"),
    Path("a1/vla/dynamic_compute/v3/independent_test_protocol.py"),
    Path("a1/vla/value_net.py"),
    Path("a1/vla/affordvla_early_exit.py"),
    Path("robot_experiments/libero/exit_vla_utils.py"),
    Path("robot_experiments/libero/eval_libero_early_exit.py"),
    Path("tests/dynamic_compute/v3/test_active_runtime.py"),
    Path("tests/dynamic_compute/v3/test_runtime_adapter.py"),
    Path("tests/dynamic_compute/test_productive_exit.py"),
    Path("scripts/dynamic_compute/v3/validate_v3_d9b_model_dry_run.py"),
    Path("scripts/dynamic_compute/v3/freeze_v3_d9b_readiness.py"),
)
REGRESSION_PATHS = (
    "tests/dynamic_compute/v3",
    "tests/dynamic_compute/test_productive_exit.py",
    "tests/dynamic_compute/test_phase_estimator.py",
    "tests/dynamic_compute/test_phase_depth_runtime.py",
    "tests/dynamic_compute/test_phase_cache.py",
    "tests/dynamic_compute/test_rollout_telemetry.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-attestation", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _current_parity() -> dict[str, Any]:
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise PermissionError(f"D9B parity input differs: {relative}")
    context = torch.load(
        REPO_ROOT / CONTEXT, map_location="cpu", weights_only=False
    )
    dataset = torch.load(
        REPO_ROOT / DATASET, map_location="cpu", weights_only=False
    )
    router_payload = torch.load(
        REPO_ROOT / ROUTER, map_location="cpu", weights_only=False
    )
    expected = torch.load(
        REPO_ROOT / SCORING, map_location="cpu", weights_only=False
    )
    router = final_router_from_mapping(router_payload)
    before = frozen_router_sha256(router)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observed = route_cached_candidate_pairs(
        router,
        context["runtime_inputs"],
        dataset["candidate_actions"][:, :2],
        dataset["action_consistency"].reshape(-1, 2),
    )
    after = frozen_router_sha256(router)
    head_error = float(
        (observed.five_head_prediction - expected["five_head_prediction"])
        .abs()
        .max()
    )
    selected_matches = int(
        (observed.selected_layer == expected["selected_layer"]).sum()
    )
    safe_matches = int(
        (observed.candidate_safe == expected["candidate_safe"]).sum()
    )
    return {
        "policy_calls": int(observed.selected_layer.numel()),
        "candidate_rows": int(observed.candidate_safe.numel()),
        "selected_layer_exact_matches": selected_matches,
        "candidate_safe_exact_matches": safe_matches,
        "five_head_prediction_max_abs_error": head_error,
        "router_state_sha256_before": before,
        "router_state_sha256_after": after,
        "selection_counts": {
            f"L{layer}": int((observed.selected_layer == layer).sum())
            for layer in (11, 13, 27)
        },
        "pass": (
            observed.selected_layer.numel() == 7140
            and observed.candidate_safe.numel() == 14280
            and selected_matches == 7140
            and safe_matches == 14280
            and head_error <= 1.0e-12
            and before == after
        ),
    }


def _regression() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        *REGRESSION_PATHS,
    ]
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
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D9B readiness freeze is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9B readiness requires a clean implementation commit")
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9B refuses to overwrite readiness evidence")

    contract = load_d9_contract(REPO_ROOT)
    if sha256_file(REPO_ROOT / D9_CONTRACT_RELATIVE_PATH) != D9_CONTRACT_SHA256:
        raise PermissionError("D9 contract hash differs")
    if sha256_file(REPO_ROOT / D9_SELECTION_RELATIVE_PATH) != D9_SELECTION_SHA256:
        raise PermissionError("D9 selection metadata hash differs")
    d9a = _json(REPO_ROOT / D9A_RESULT)
    if (
        sha256_file(REPO_ROOT / D9A_RESULT) != D9A_RESULT_SHA256
        or d9a.get("status") != D9A_RUNTIME_STATUS
    ):
        raise PermissionError("D9A evidence differs")

    dry_path = args.dry_run_result.resolve(strict=True)
    dry = _json(dry_path)
    current_commit = git_output("rev-parse", "HEAD")
    if (
        dry.get("status") != "PASS_V3_D9B_REAL_MODEL_NON_CONTROL_DRY_RUN"
        or dry.get("source_git_commit") != current_commit
        or not all(dry.get("checks", {}).values())
        or dry.get("access_ledger", {}).get("active_control") is not False
        or dry.get("access_ledger", {}).get("environment_steps") != 0
        or dry.get("access_ledger", {}).get("episode_40_49_init_states_opened")
        is not False
    ):
        raise PermissionError("D9B real-model dry-run evidence differs")
    physical_gpu = dry["environment"]["physical_gpu"]
    if (
        physical_gpu.get("physical_index") not in (0, 1, 2, 3)
        or not str(physical_gpu.get("uuid", "")).startswith("GPU-")
        or dry["environment"].get("visible_device_count") != 1
    ):
        raise PermissionError("D9B dry-run GPU binding differs")

    model_audit = validate_runtime_model_directory(
        args.checkpoint.resolve(strict=True),
        args.model_attestation.resolve(strict=True),
    )
    phase_path = args.phase_checkpoint.resolve(strict=True)
    if sha256_file(phase_path) != D2_PHASE_CHECKPOINT_SHA256:
        raise PermissionError("D9B phase checkpoint differs")
    parity = _current_parity()
    regression = _regression()
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    gitlink = git_output("ls-tree", "HEAD", "robot_experiments/libero/LIBERO")
    checks = {
        "D9_contract_and_selection_metadata_exact": True,
        "D9A_frozen_evidence_exact": True,
        "current_adapter_D8_parity_exact": parity["pass"],
        "real_model_non_control_dry_run_pass": True,
        "dry_run_bound_to_clean_current_commit": dry["source_git_commit"]
        == current_commit,
        "A1_model_and_sidecars_attested": model_audit["model_sha256"]
        == D2_CHECKPOINT_SHA256,
        "phase_checkpoint_exact": sha256_file(phase_path)
        == D2_PHASE_CHECKPOINT_SHA256,
        "D8_router_exact": sha256_file(REPO_ROOT / ROUTER)
        == EXPECTED_INPUT_SHA256[ROUTER],
        "front_four_physical_GPU_and_UUID_bound": physical_gpu["physical_index"]
        in (0, 1, 2, 3),
        "LIBERO_gitlink_exact": gitlink.startswith(
            "160000 commit 8f1084e3132a39270c3a13ebe37270a43ece2a01"
        ),
        "CPU_regression_pass": regression["pass"],
        "pip_check_pass": pip_check.returncode == 0,
        "readiness_process_did_not_initialize_CUDA": not torch.cuda.is_initialized(),
        "official_episode_40_49_state_not_opened": True,
        "active_control_not_run": True,
        "no_fit_threshold_normalizer_or_feature_change": True,
    }
    if not all(checks.values()):
        raise PermissionError(f"D9B readiness checks failed: {checks}")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9B implementation worktree changed during readiness")

    code_sha = {
        path.as_posix(): sha256_file(REPO_ROOT / path) for path in CODE_PATHS
    }
    result = {
        "status": PASS_STATUS,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_initialized": torch.cuda.is_initialized(),
            "dry_run_physical_GPU": physical_gpu,
        },
        "bound_code_sha256": code_sha,
        "bound_artifacts": {
            "D9_contract": {
                "path": D9_CONTRACT_RELATIVE_PATH.as_posix(),
                "sha256": D9_CONTRACT_SHA256,
            },
            "D9_selection_metadata": {
                "path": D9_SELECTION_RELATIVE_PATH.as_posix(),
                "sha256": D9_SELECTION_SHA256,
            },
            "D9A_result": {
                "path": D9A_RESULT.as_posix(),
                "sha256": D9A_RESULT_SHA256,
            },
            "dry_run_result": {
                "path": str(dry_path),
                "sha256": sha256_file(dry_path),
            },
            "A1_model": model_audit,
            "A1_config_sha256": D2_CHECKPOINT_CONFIG_SHA256,
            "A1_action_delta_sha256": D2_ACTION_DELTA_SHA256,
            "A1_exit_thresholds_sha256": D2_EXIT_THRESHOLDS_SHA256,
            "A1_dataset_statistics_sha256": D2_DATASET_STATISTICS_SHA256,
            "A1_model_attestation_sha256": D2_MODEL_ATTESTATION_SHA256,
            "phase_checkpoint": {
                "path": str(phase_path),
                "sha256": D2_PHASE_CHECKPOINT_SHA256,
                "state_sha256": dry["runtime_artifacts"]["phase_state_sha256"],
            },
            "D8_router": {
                "path": ROUTER.as_posix(),
                "sha256": EXPECTED_INPUT_SHA256[ROUTER],
                "state_sha256": parity["router_state_sha256_after"],
            },
            "LIBERO_gitlink": gitlink,
        },
        "current_code_D8_parity": parity,
        "real_model_dry_run_summary": {
            "policy_calls": dry["policy_calls"],
            "runtime_counters": dry["runtime_counters"],
            "elapsed_seconds": dry["elapsed_seconds"],
        },
        "regression": regression,
        "pip_check": {
            "exit_code": pip_check.returncode,
            "output": pip_check.stdout.strip(),
        },
        "readiness_checks": checks,
        "access_ledger": {
            "independent_test_selection_metadata_opened": True,
            "independent_test_sample_state_payload_opened": False,
            "LIBERO_episode_40_49_init_states_opened": False,
            "already_analyzed_D8_cached_inputs_opened": 4,
            "real_model_synthetic_policy_calls": 2,
            "LIBERO_environment_created": False,
            "environment_steps": 0,
            "active_control": False,
            "fit_calls": 0,
            "threshold_normalizer_or_feature_changes": 0,
        },
        "authorization": {
            "authorized": contract["authorization"]["on_D9B_readiness_pass"],
            "exact_schedule_only": True,
            "open_episode_40_49_only_under_frozen_D9C_runner": True,
            "additional_test_tuning_or_second_independent_test": False,
            "deployment": False,
        },
        "claim_boundary": {
            "readiness_is_independent_test_result": False,
            "synthetic_dry_run_is_closed_loop_success": False,
            "D9C_has_run": False,
            "active_control_has_run": False,
            "superiority_or_noninferiority_claim_authorized": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
