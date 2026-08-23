#!/usr/bin/env python3
"""Freeze the immutable V3-D5 formal development result attestation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT = Path("reports/v3_d5_development_oof/result.json")
REPORT_SHA256 = (
    "bddd8fdbbf53f5d8270ee13012dc6f29d5481ca6c5e1c4dde4aacb85cd3ca2bf"
)
PAYLOAD = Path(
    "reports/v3_d5_development_oof/development_joint_nested_oof.pt"
)
PAYLOAD_SHA256 = (
    "db8235f568c26ec918ebce413e12bb8326a66e3e79a0063e77476d9058a899ed"
)
OUTPUT = Path("results/v3/v3_d5_formal_development_result.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D5 formal report must be a JSON object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 result freeze requires clean worktree")
    report_path = REPO_ROOT / REPORT
    payload_path = REPO_ROOT / PAYLOAD
    if sha256(report_path) != REPORT_SHA256 or sha256(payload_path) != PAYLOAD_SHA256:
        raise PermissionError("V3-D5 formal report or payload SHA differs")
    report = json_object(report_path)
    if (
        report.get("status") != "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
        or report.get("source_worktree_dirty") is not False
        or report.get("payload_sha256") != PAYLOAD_SHA256
        or report.get("gate_checks", {}).get(
            "false_safe_cluster_exact_ucb95_at_most_5_percent"
        )
        is not False
        or report.get("next_stage", {}).get("authorized")
        != "D5_NEGATIVE_RESULT_ANALYSIS_ONLY"
    ):
        raise PermissionError("V3-D5 negative result semantics differ")
    selection = report["selection"]
    metric = report["OOF_metrics"]
    result = {
        "schema_version": "phase-route-vla.v3.d5-formal-development-attestation.v1",
        "status": report["status"],
        "run_timestamp_utc": report["timestamp_utc"],
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": report["source_git_commit"],
        "freeze_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "development_v2",
        "data_boundary": {
            "task_ids": list(range(10)),
            "episode_indices": list(range(12, 30)),
            "clusters": report["clusters"],
            "policy_calls": report["policy_calls"],
            "candidate_rows": report["rows"],
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
        },
        "training_audit": {
            "outer_folds": report["outer_fold_count"],
            "inner_folds_per_outer": report["inner_fold_count_per_outer"],
            "fits_per_outer": report["fits_per_outer"],
            "total_model_fits": report["total_model_fits"],
            "infeasible_outer_folds": report["infeasible_outer_folds"],
            "selected_lambdas": report["selected_lambdas"],
            "selected_thresholds": report["selected_thresholds"],
        },
        "predictive_metrics": {
            "full_action_unsafe_AUROC": metric["full_action_unsafe"]["auroc"],
            "full_action_unsafe_log_loss": metric["full_action_unsafe"][
                "binary_log_loss"
            ],
            "gripper_step_unsafe_AUROC": metric["gripper_step_unsafe"]["auroc"],
            "gripper_step_unsafe_log_loss": metric["gripper_step_unsafe"][
                "binary_log_loss"
            ],
        },
        "formal_gate": {
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
            "false_gripper_calls": selection["false_gripper_calls"],
            "gate_checks": report["gate_checks"],
        },
        "estimated_efficiency": report["estimated_efficiency"],
        "artifacts": {
            "report_path": REPORT.as_posix(),
            "report_sha256": REPORT_SHA256,
            "payload_path": PAYLOAD.as_posix(),
            "payload_sha256": PAYLOAD_SHA256,
            "dataset_result_sha256": report["input_sha256"]["dataset_result"],
            "dataset_payload_sha256": report["input_sha256"]["dataset_payload"],
            "outer_fold_result_sha256": report["input_sha256"][
                "outer_fold_results"
            ],
        },
        "interpretation": {
            "predictive_signal_is_strong": metric["full_action_unsafe"]["auroc"]
            > 0.9,
            "gripper_gate_observed_false_safe_calls": selection[
                "false_gripper_calls"
            ],
            "exact_statistical_gate_passed": False,
            "posthoc_threshold_repair_allowed": False,
            "result_is_promising_but_negative": True,
        },
        "authorization": {
            "next_stage": "D5_NEGATIVE_RESULT_ANALYSIS_ONLY",
            "reuse_D3_calibration_for_repair": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
            "deployment_authorized": False,
        },
        "claim_boundary": report["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite formal result")
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
