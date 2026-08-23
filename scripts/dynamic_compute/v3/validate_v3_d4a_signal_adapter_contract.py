#!/usr/bin/env python3
"""Validate V3-D4A pre-shadow signal thresholds without tensor loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = Path(
    "configs/research/v3/gripper_v2/d4a_signal_adapter_contract.json"
)
CONTRACT_SHA256 = (
    "4c0a53521eaeac2be845cbafaba80c51596a89257e93bcfd78247df684aad13a"
)
OUTPUT_PATH = Path("results/v3/v3_d4a_signal_adapter_contract_validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D4A contract validation requires clean worktree")
    contract_file = REPO_ROOT / CONTRACT_PATH
    if sha256(contract_file) != CONTRACT_SHA256:
        raise PermissionError("V3-D4A contract SHA-256 differs")
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "D4A_SIGNAL_ADAPTER_CONTRACT_FROZEN"
        or contract.get("scope", {}).get("pre_shadow_distribution_freeze")
        is not True
        or contract.get("scope", {}).get("active_control_allowed") is not False
        or contract.get("scope", {}).get("independent_test_allowed") is not False
        or contract.get("motion_gate", {}).get(
            "threshold_search_on_v3_allowed"
        )
        is not False
        or contract.get("tail_ucb_gate", {}).get(
            "threshold_search_on_v3_allowed"
        )
        is not False
    ):
        raise PermissionError("V3-D4A contract semantics differ")
    prerequisite = contract["prerequisite"]
    d4_path = REPO_ROOT / prerequisite["d4_contract_validation_path"]
    if sha256(d4_path) != prerequisite["d4_contract_validation_sha256"]:
        raise PermissionError("V3-D4A D4 validation hash differs")
    d4_result = json.loads(d4_path.read_text(encoding="utf-8"))
    if d4_result.get("status") != prerequisite["required_d4_validation_status"]:
        raise PermissionError("V3-D4A D4 validation status differs")
    legacy = contract["legacy_evidence"]
    source_root = Path(legacy["source_root"]).resolve(strict=True)
    artifact_hashes = {}
    for key in ("checkpoint", "tail_calibration"):
        path = source_root / legacy[f"{key}_path"]
        observed = sha256(path)
        if observed != legacy[f"{key}_sha256"]:
            raise PermissionError(f"V3-D4A legacy artifact hash differs: {key}")
        artifact_hashes[key] = observed
    motion = contract["motion_gate"]["thresholds"]
    tail = contract["tail_ucb_gate"]
    if not all(
        float(motion[str(layer)][name]) > 0.0
        for layer in (11, 13)
        for name in ("translation_rms", "rotation_rms")
    ):
        raise PermissionError("V3-D4A motion threshold domain differs")
    for layer in (11, 13):
        expected = float(tail["q90_anchors"][str(layer)]) + float(
            tail["conformal_corrections"][str(layer)]
        )
        if expected != float(tail["tail_budgets"][str(layer)]):
            raise PermissionError("V3-D4A tail budget addition differs")
    result = {
        "status": "PASS_V3_D4A_SIGNAL_ADAPTER_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d4a-contract-validation.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "contract": {
            "path": CONTRACT_PATH.as_posix(),
            "sha256": CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "bound_sha256": {
            "d4_contract_validation": prerequisite[
                "d4_contract_validation_sha256"
            ],
            **artifact_hashes,
        },
        "frozen_thresholds": {
            "motion": motion,
            "tail": tail["tail_budgets"],
            "action_consistency": contract["action_consistency_adapter"][
                "threshold"
            ],
            "selected_from_v3_shadow_distribution": False,
        },
        "access_ledger": {
            "d4_validation_json_opened": True,
            "legacy_artifacts_byte_hashed": 2,
            "legacy_tensor_artifacts_deserialized": 0,
            "v3_calibration_tensor_payload_opened": False,
            "shadow_distribution_viewed": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "model_fit_or_threshold_search": 0,
        },
        "next_stage": {
            "authorized": "V3-D4A_SIGNAL_ADAPTER_IMPLEMENTATION_ONLY",
            "formal_shadow_authorized": False,
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
        "claim_boundary": {
            "adapter_implemented": False,
            "shadow_decision_run": False,
            "deployment_authorized": False,
            "superiority_claim_authorized": False,
        },
    }
    output = REPO_ROOT / OUTPUT_PATH
    sidecar = output.with_suffix(".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("V3-D4A refuses to overwrite validation evidence")
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = sha256(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
