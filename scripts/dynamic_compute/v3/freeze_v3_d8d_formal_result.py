#!/usr/bin/env python3
"""Freeze the already-computed D8D report without recomputing its gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.d8_confirmation_scoring import (  # noqa: E402
    D8D_PAYLOAD_SCHEMA_VERSION,
    D8D_RESULT_SCHEMA_VERSION,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    stream_sha256,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CLUSTER_COUNT,
)


REPORT = Path("reports/v3_d8_confirmation/result.json")
REPORT_SHA256 = (
    "5c43ce8f77ada57737bbebc4abcbaa0274f0924e5a87ff62735d9b2ed8122c53"
)
PAYLOAD = Path("reports/v3_d8_confirmation/confirmation_scoring.pt")
PAYLOAD_SHA256 = (
    "b225ebec9bfd55044a5b856dd09ad9b5b14278164172d93d525d10309472ffba"
)
ERROR_RECORDS = Path("reports/v3_d8_confirmation/false_safe_records.jsonl")
ERROR_RECORDS_SHA256 = (
    "c58a0122621b4a90f4502076eeea014fc3dec94d7e18667dbf737a18d4abd947"
)
OUTPUT = Path("results/v3/v3_d8_formal_confirmation_result.json")
SOURCE_SCORING_COMMIT = "013530b2b5c3e0435369a83db7353ec5f56593c4"
EXPECTED_STATUS = "PASS_V3_D8_PROSPECTIVE_SHADOW_CONFIRMATION"
EXPECTED_CALLS = 7140
EXPECTED_ROWS = 14280


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8D JSON must be an object: {path}")
    return dict(value)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D8D formal freeze is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8D formal freeze requires a clean worktree")
    report_path = REPO_ROOT / REPORT
    payload_path = REPO_ROOT / PAYLOAD
    error_path = REPO_ROOT / ERROR_RECORDS
    if (
        stream_sha256(report_path) != REPORT_SHA256
        or stream_sha256(payload_path) != PAYLOAD_SHA256
        or stream_sha256(error_path) != ERROR_RECORDS_SHA256
    ):
        raise PermissionError("D8D formal evidence SHA-256 differs")
    report = json_object(report_path)
    confirmation = report.get("confirmation", {})
    safety = report.get("safety_audit", {})
    ledger = report.get("access_ledger", {})
    next_stage = report.get("next_stage", {})
    checks = report.get("gate_checks", {})
    if (
        report.get("status") != EXPECTED_STATUS
        or report.get("schema_version") != D8D_RESULT_SCHEMA_VERSION
        or report.get("source_git_commit") != SOURCE_SCORING_COMMIT
        or report.get("source_worktree_dirty") is not False
        or len(checks) != 14
        or not all(value is True for value in checks.values())
        or confirmation.get("clusters") != D8_CLUSTER_COUNT
        or confirmation.get("policy_calls") != EXPECTED_CALLS
        or confirmation.get("safe_clusters") != D8_CLUSTER_COUNT
        or confirmation.get("false_safe_clusters") != 1
        or confirmation.get("false_full_action_clusters") != 1
        or confirmation.get("false_gripper_calls") != 0
        or confirmation.get("severe_false_full_action_clusters") != 0
        or safety.get("false_safe_records") != 1
        or ledger.get("confirmation_gate_evaluations") != 1
        or ledger.get("model_refits") != 0
        or ledger.get("threshold_or_feature_searches") != 0
        or ledger.get("gpu_query_or_initialization") != 0
        or ledger.get("official_episode_40_49_opened") is not False
        or ledger.get("active_control") is not False
        or next_stage.get("authorized")
        != "INDEPENDENT_TEST_V2_PROTOCOL_DESIGN_ONLY"
        or next_stage.get("open_episode_40_49_authorized") is not False
    ):
        raise PermissionError("D8D formal report semantics differ")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != D8D_PAYLOAD_SCHEMA_VERSION
        or payload["five_head_prediction"].shape != (5, EXPECTED_ROWS, 2)
        or payload["combined_score"].shape != (EXPECTED_ROWS, 2)
        or payload["full_action_head_range"].shape != (EXPECTED_ROWS,)
        or payload["candidate_safe"].shape != (EXPECTED_ROWS,)
        or payload["selected_layer"].shape != (EXPECTED_CALLS,)
        or payload.get("gate_checks") != checks
    ):
        raise PermissionError("D8D scoring payload geometry differs")
    if (
        not bool(torch.isfinite(payload["five_head_prediction"]).all())
        or not bool(torch.isfinite(payload["combined_score"]).all())
        or payload.get("active_control") is not False
        or payload.get("official_episode_40_49_opened") is not False
        or payload.get("router_refit_or_threshold_selection") is not False
    ):
        raise PermissionError("D8D scoring payload boundary differs")
    error_lines = [
        json.loads(line)
        for line in error_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(error_lines) != 1
        or error_lines[0].get("cluster_key")
        != "libero_10:task4:fresh_confirm_v1:replicate18"
        or error_lines[0].get("gripper_step_unsafe") is not False
        or error_lines[0].get("severe_full_action_false") is not False
    ):
        raise PermissionError("D8D false-safe record differs")

    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D8D refuses to overwrite formal attestation")
    result = {
        "status": EXPECTED_STATUS,
        "schema_version": "phase-route-vla.v3.d8-formal-confirmation-attestation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "prospective_generated_state_shadow_confirmation",
        "suite": "libero_10",
        "formal_report": {
            "path": REPORT.as_posix(),
            "sha256": REPORT_SHA256,
            "source_scoring_commit": SOURCE_SCORING_COMMIT,
        },
        "artifacts": {
            "prediction_payload": PAYLOAD.as_posix(),
            "prediction_payload_sha256": PAYLOAD_SHA256,
            "false_safe_records": ERROR_RECORDS.as_posix(),
            "false_safe_records_sha256": ERROR_RECORDS_SHA256,
        },
        "confirmation": confirmation,
        "selection": report["selection"],
        "safety_audit": {
            "false_safe_calls": safety["false_safe_calls"],
            "false_full_action_calls": safety["false_full_action_calls"],
            "false_gripper_calls": safety["false_gripper_calls"],
            "severe_false_full_action_calls": safety[
                "severe_false_full_action_calls"
            ],
            "false_safe_cluster_keys": safety["false_safe_cluster_keys"],
            "full_action_truth_threshold": safety["full_action_truth_threshold"],
            "severe_ratio_threshold": safety["severe_ratio_threshold"],
        },
        "estimated_efficiency": report["estimated_efficiency"],
        "gate_checks": checks,
        "input_sha256": report["input_sha256"],
        "access_ledger": {
            **ledger,
            "formal_gate_recomputed_during_freeze": False,
            "formal_report_and_payload_authenticated": True,
        },
        "authorization": next_stage,
        "claim_boundary": report["claim_boundary"],
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(
        f"{stream_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
