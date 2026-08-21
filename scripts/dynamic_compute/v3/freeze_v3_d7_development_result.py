#!/usr/bin/env python3
"""Freeze the immutable V3-D7 reused-development selection attestation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT = Path("reports/v3_d7_development_oof/result.json")
REPORT_SHA256 = (
    "600370bf978450afc8756cfe7929b36b33ed9d7da716a463902e13c2d0ab3ea9"
)
PAYLOAD = Path("reports/v3_d7_development_oof/development_epistemic_nested_oof.pt")
PAYLOAD_SHA256 = (
    "ada55c17e7bbf7c6a5833c2a832c77f13249a9fd3c7aff6d6e0c842dd242a35d"
)
SOURCE_GIT_COMMIT = "ffc141297a8e5ee10a74688203c1643e158de36b"
D7_CONTRACT_SHA256 = (
    "7e1f8934e33ae33493b950eabc1142c1f6cd7103ef7b4ad735d6c8b13a5afdea"
)
D7_CONTRACT_VALIDATION_SHA256 = (
    "31dc77519a1ae7b03210a23301f553ca632a90df33eedb3dfcfc17b76386b829"
)
OUTPUT = Path("results/v3/v3_d7_formal_development_result.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D7 formal report must be a JSON object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D7 result freeze requires clean worktree")
    report_path = REPO_ROOT / REPORT
    payload_path = REPO_ROOT / PAYLOAD
    if sha256(report_path) != REPORT_SHA256 or sha256(payload_path) != PAYLOAD_SHA256:
        raise PermissionError("V3-D7 formal report or payload SHA differs")
    report = json_object(report_path)
    gate_checks = report.get("gate_checks", {})
    claim_boundary = report.get("claim_boundary", {})
    access_ledger = report.get("access_ledger", {})
    if (
        report.get("status") != "PROMISING_V3_D7_REUSED_DEVELOPMENT_SELECTION"
        or report.get("schema_version")
        != "phase-route-vla.v3.d7-reused-development-result.v1"
        or report.get("source_git_commit") != SOURCE_GIT_COMMIT
        or report.get("source_worktree_dirty") is not False
        or report.get("role") != "development_v2_reused_for_D7_method_selection"
        or report.get("payload_sha256") != PAYLOAD_SHA256
        or not isinstance(gate_checks, Mapping)
        or not gate_checks
        or not all(value is True for value in gate_checks.values())
        or report.get("next_stage", {}).get("authorized")
        != "FRESH_CONFIRMATION_DATA_PROTOCOL_DESIGN_ONLY"
        or report.get("next_stage", {}).get("independent_test_authorized") is not False
        or report.get("next_stage", {}).get("active_control_authorized") is not False
        or not isinstance(claim_boundary, Mapping)
        or not claim_boundary
        or not all(value is False for value in claim_boundary.values())
        or access_ledger.get("calibration_v2_payload_opened") is not False
        or access_ledger.get("independent_test_payload_opened") is not False
        or access_ledger.get("gpu_query_or_initialization") != 0
        or access_ledger.get("fresh_rollout") is not False
        or access_ledger.get("active_control") is not False
    ):
        raise PermissionError("V3-D7 promising result semantics differ")
    if (
        report.get("rows") != 13042
        or report.get("policy_calls") != 6521
        or report.get("clusters") != 180
        or report.get("outer_fold_count") != 18
        or report.get("inner_fold_count_per_outer") != 17
        or report.get("head_count") != 5
        or report.get("fits_per_outer") != 260
        or report.get("total_model_fits") != 4680
        or report.get("infeasible_outer_folds") != 0
        or set(report.get("selected_lambdas", {}).values()) != {0.01}
        or report.get("input_sha256", {}).get("D7_contract") != D7_CONTRACT_SHA256
        or report.get("input_sha256", {}).get("D7_contract_validation")
        != D7_CONTRACT_VALIDATION_SHA256
    ):
        raise PermissionError("V3-D7 training or data geometry differs")

    selection = report["selection"]
    metric = report["OOF_metrics"]
    head_range = report["full_action_head_range"]
    if (
        selection.get("false_safe_clusters") != 2
        or selection.get("false_full_action_calls") != 2
        or selection.get("false_full_action_clusters") != 2
        or selection.get("false_gripper_calls") != 0
        or head_range.get("rows_above_1e-6") != 13042
        or head_range.get("fraction_above_1e-6") != 1.0
        or report.get("locked_comparison", {}).get("comparison_is_unbiased_or_fresh")
        is not False
    ):
        raise PermissionError("V3-D7 selection or ensemble semantics differ")

    result = {
        "schema_version": "phase-route-vla.v3.d7-formal-development-attestation.v1",
        "status": report["status"],
        "run_timestamp_utc": report["timestamp_utc"],
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": report["source_git_commit"],
        "freeze_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "development_v2_reused_after_D5_D6_for_D7_method_selection",
        "data_boundary": {
            "task_ids": list(range(10)),
            "episode_indices": list(range(12, 30)),
            "clusters": report["clusters"],
            "policy_calls": report["policy_calls"],
            "candidate_rows": report["rows"],
            "development_data_is_fresh_confirmation": False,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "fresh_rollout": False,
        },
        "training_audit": {
            "outer_folds": report["outer_fold_count"],
            "inner_folds_per_outer": report["inner_fold_count_per_outer"],
            "ensemble_heads": report["head_count"],
            "fits_per_outer": report["fits_per_outer"],
            "total_model_fits": report["total_model_fits"],
            "infeasible_outer_folds": report["infeasible_outer_folds"],
            "selected_lambdas": report["selected_lambdas"],
            "thresholds": report["thresholds"],
            "gpu_query_or_initialization": 0,
        },
        "predictive_metrics": {
            "full_action_epistemic_upper_AUROC": metric[
                "full_action_epistemic_upper_score"
            ]["auroc"],
            "full_action_epistemic_upper_log_loss": metric[
                "full_action_epistemic_upper_score"
            ]["binary_log_loss"],
            "head0_gripper_step_AUROC": metric[
                "head0_gripper_step_probability"
            ]["auroc"],
            "head0_gripper_step_log_loss": metric[
                "head0_gripper_step_probability"
            ]["binary_log_loss"],
        },
        "uncertainty_diagnostics": {
            "full_action_head_range": head_range,
            "all_rows_non_degenerate_above_1e-6": True,
        },
        "formal_selection": {
            "L11_calls": selection["L11"],
            "L13_calls": selection["L13"],
            "L27_calls": selection["L27"],
            "early_exit_calls": selection["early_exit_calls"],
            "early_exit_fraction": selection["early_exit_fraction"],
            "safe_clusters": selection["safe_clusters"],
            "false_safe_clusters": selection["false_safe_clusters"],
            "false_safe_cluster_ucb95": selection["false_safe_cluster_ucb95"],
            "maximum_allowed_ucb95": 0.05,
            "false_full_action_calls": selection["false_full_action_calls"],
            "false_full_action_clusters": selection[
                "false_full_action_clusters"
            ],
            "false_gripper_calls": selection["false_gripper_calls"],
            "false_records": selection["false_records"],
            "per_task_early_calls": selection["per_task_early_calls"],
            "gate_checks": gate_checks,
        },
        "estimated_efficiency": report["estimated_efficiency"],
        "locked_comparison": report["locked_comparison"],
        "artifacts": {
            "report_path": REPORT.as_posix(),
            "report_sha256": REPORT_SHA256,
            "payload_path": PAYLOAD.as_posix(),
            "payload_sha256": PAYLOAD_SHA256,
            "D7_contract_sha256": D7_CONTRACT_SHA256,
            "D7_contract_validation_sha256": D7_CONTRACT_VALIDATION_SHA256,
            "D5_formal_sha256": report["input_sha256"]["D5_formal"],
            "D6_formal_sha256": report["input_sha256"]["D6_formal"],
            "dataset_result_sha256": report["input_sha256"]["dataset_result"],
            "dataset_payload_sha256": report["input_sha256"]["dataset_payload"],
            "outer_fold_result_sha256": report["input_sha256"][
                "outer_fold_results"
            ],
        },
        "interpretation": {
            "all_outer_shrunk_thresholds_feasible": True,
            "ensemble_is_non_degenerate": True,
            "preregistered_development_gate_passed": True,
            "false_clusters_reduced_from_four_to_two_on_reused_development": True,
            "D6_task0_episode14_and_task9_episode29_errors_rejected": True,
            "remaining_errors_are_full_action_only": True,
            "result_is_promising_method_selection": True,
            "result_is_fresh_confirmation": False,
            "comparison_supports_superiority": False,
            "posthoc_D7_repair_allowed": False,
        },
        "authorization": {
            "next_stage": "FRESH_CONFIRMATION_DATA_PROTOCOL_DESIGN_ONLY",
            "open_episode_40_49_independent_test": False,
            "reuse_episode_30_39_for_repair_or_confirmation": False,
            "active_control_authorized": False,
            "deployment_authorized": False,
        },
        "claim_boundary": claim_boundary,
    }
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D7 refuses to overwrite formal result")
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(incomplete), str(output))
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
