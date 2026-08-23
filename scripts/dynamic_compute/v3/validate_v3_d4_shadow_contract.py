#!/usr/bin/env python3
"""Validate the frozen V3-D4.0 shadow contract without opening payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3.shadow_decision import (  # noqa: E402
    D4_ACTION_CONSISTENCY_THRESHOLD,
    D4_CONTRACT_RELATIVE_PATH,
    D4_CONTRACT_SHA256,
    D4_GRIPPER_THRESHOLD,
    D4_PRIORITY,
    D4_STATUS,
    ShadowCandidateSignals,
    decide_shadow,
    load_d4_contract,
)


DEFAULT_OUTPUT = Path("results/v3/v3_d4_shadow_contract_validation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(layer: int) -> ShadowCandidateSignals:
    return ShadowCandidateSignals(
        layer=layer,
        original_action_consistency=True,
        motion_safe=True,
        tail_ucb_safe=True,
        gripper_score=D4_GRIPPER_THRESHOLD,
    )


def main() -> None:
    args = parse_args()
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D4 contract validation requires a clean worktree")
    contract = load_d4_contract(REPO_ROOT)
    prerequisites = contract["prerequisite"]
    bound_hashes = {}
    for key in (
        "d3_attestation",
        "d3_calibration_result",
        "d3_dataset_payload",
        "d3_predictions",
    ):
        path = REPO_ROOT / prerequisites[f"{key}_path"]
        observed = sha256(path)
        expected = prerequisites[f"{key}_sha256"]
        if observed != expected:
            raise PermissionError(f"V3-D4 prerequisite hash differs: {key}")
        bound_hashes[key] = observed
    d3_attestation = json.loads(
        (REPO_ROOT / prerequisites["d3_attestation_path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        d3_attestation.get("status") != prerequisites["required_d3_status"]
        or d3_attestation.get("authorization", {}).get(
            "d4_shadow_decision_only_authorized"
        )
        is not True
        or d3_attestation.get("authorization", {}).get(
            "active_control_authorized"
        )
        is not False
        or d3_attestation.get("authorization", {}).get(
            "independent_test_authorized"
        )
        is not False
    ):
        raise PermissionError("V3-D4 prerequisite authorization differs")
    priority = decide_shadow(_safe(11), _safe(13))
    fail_closed = decide_shadow(
        ShadowCandidateSignals(11, True, None, True, D4_GRIPPER_THRESHOLD),
        ShadowCandidateSignals(13, True, True, None, D4_GRIPPER_THRESHOLD),
    )
    result = {
        "status": "PASS_V3_D4_SHADOW_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d4-contract-validation.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "contract": {
            "status": D4_STATUS,
            "path": D4_CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": D4_CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "prerequisite_sha256": bound_hashes,
        "decision_audit": {
            "formula": contract["decision"]["formula"],
            "priority": list(D4_PRIORITY),
            "both_safe_selects": priority.selected_layer,
            "missing_motion_or_tail_selects": fail_closed.selected_layer,
            "action_consistency_threshold": D4_ACTION_CONSISTENCY_THRESHOLD,
            "gripper_threshold": D4_GRIPPER_THRESHOLD,
            "returns_action": False,
            "active_control": False,
        },
        "access_ledger": {
            "d3_attestation_json_opened": True,
            "d3_artifacts_byte_hashed": 4,
            "d3_tensor_payload_deserialized": 0,
            "independent_test_selection_opened": False,
            "independent_test_sample_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "model_fit_or_threshold_selection": 0,
            "shadow_distribution_viewed": False,
        },
        "readiness": {
            "decision_engine": "PASS",
            "motion_runtime_threshold_frozen": False,
            "tail_runtime_threshold_frozen": False,
            "formal_shadow_execution_authorized": False,
            "reason": (
                "motion and tail-UCB signal adapter thresholds require separate "
                "pre-distribution attestation"
            ),
        },
        "next_stage": {
            "authorized": (
                "V3-D4A_SIGNAL_ADAPTER_IMPLEMENTATION_AND_ATTESTATION_ONLY"
            ),
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
        "claim_boundary": {
            "shadow_decision_run": False,
            "active_control_run": False,
            "independent_test_run": False,
            "deployment_authorized": False,
            "superiority_claim_authorized": False,
        },
    }
    output = args.output.resolve()
    expected_output = (REPO_ROOT / DEFAULT_OUTPUT).resolve()
    if output != expected_output:
        raise PermissionError("V3-D4 validation output path differs")
    sidecar = output.with_suffix(".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("V3-D4 refuses to overwrite validation evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
