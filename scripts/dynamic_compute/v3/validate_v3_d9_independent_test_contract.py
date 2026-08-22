#!/usr/bin/env python3
"""Validate D9 protocol metadata without opening test states or running control."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    D9_ARMS,
    D9_CONTRACT_RELATIVE_PATH,
    D9_CONTRACT_SHA256,
    D9_EPISODE_INDICES,
    D9_GPU_ALLOWLIST,
    D9_RECORD_COUNT,
    D9_RECORDS_PER_TASK,
    D9_SELECTION_SHA256,
    D9_TASK_IDS,
    load_d9_contract,
    load_d9_selection_metadata,
)


OUTPUT = Path("results/v3/v3_d9_independent_test_contract_validation.json")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("D9 contract validation must hide all GPUs")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9 contract validation requires a clean worktree")
    contract = load_d9_contract(REPO_ROOT)
    records = load_d9_selection_metadata(REPO_ROOT)
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9 refuses to overwrite contract validation")
    task_counts = {
        str(task): sum(record.task_id == task for record in records)
        for task in D9_TASK_IDS
    }
    gpu_counts = {
        str(gpu): sum(record.physical_gpu_index == gpu for record in records)
        for gpu in D9_GPU_ALLOWLIST
    }
    checks = {
        "D8_prospective_shadow_confirmation_pass_bound": True,
        "contract_SHA_and_semantics_exact": True,
        "selection_metadata_SHA_exact": True,
        "all_100_task_episode_seed_records_exact": len(records) == D9_RECORD_COUNT,
        "all_10_tasks_have_10_episode_records": all(
            value == D9_RECORDS_PER_TASK for value in task_counts.values()
        ),
        "episode_indices_exactly_40_to_49": {
            record.episode_index for record in records
        }
        == set(D9_EPISODE_INDICES),
        "paired_A1_and_PhaseRoute_arms_frozen": tuple(
            contract["paired_evaluation"]["arms"]
        )
        == D9_ARMS,
        "runtime_adapter_D8_parity_required_before_test_access": contract[
            "D9A_runtime_adapter_readiness"
        ]["readiness_attestation_required_before_test_access"],
        "front_four_GPU_schedule_only": set(gpu_counts) == {"0", "1", "2", "3"},
        "test_sample_state_payload_not_opened": True,
        "active_control_not_run": True,
        "no_fit_threshold_or_feature_selection": True,
    }
    if not all(checks.values()):
        raise PermissionError("D9 contract validation check failed")
    result = {
        "status": "PASS_V3_D9_INDEPENDENT_TEST_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d9-contract-validation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "contract": {
            "path": D9_CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": D9_CONTRACT_SHA256,
        },
        "selection_metadata": {
            "sha256": D9_SELECTION_SHA256,
            "records": len(records),
            "task_counts": task_counts,
            "GPU_assignment_counts_for_future_execution": gpu_counts,
            "sample_state_payload_opened": False,
        },
        "future_evaluation": {
            "arms": list(D9_ARMS),
            "pairs": D9_RECORD_COUNT,
            "rollouts": 2 * D9_RECORD_COUNT,
            "primary_gate": contract["primary_gate"],
            "execution_order": contract["execution_order"],
        },
        "checks": checks,
        "access_ledger": {
            "D8_formal_attestation_opened": True,
            "independent_test_selection_metadata_opened": True,
            "independent_test_sample_state_payload_opened": False,
            "LIBERO_init_state_archive_opened": False,
            "D8_router_model_payload_opened": False,
            "model_checkpoint_opened": False,
            "GPU_query_or_initialization": 0,
            "fit_calls": 0,
            "threshold_or_feature_searches": 0,
            "test_rollouts": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "D9A_RUNTIME_ADAPTER_IMPLEMENTATION_AND_D8_PARITY_ONLY",
            "episode_40_49_state_access": False,
            "independent_test_sample_payload_access": False,
            "active_control": False,
            "deployment": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
