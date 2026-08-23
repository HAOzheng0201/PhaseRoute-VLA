#!/usr/bin/env python3
"""Validate the frozen V3-D6 repair contract using metadata only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = Path("configs/research/v3/joint_reliability/d6_repair_contract.json")
CONTRACT_SHA256 = (
    "28185ce5431cf438d20cb7cfdfd0e20d5859b6a99f1bdafa81d18faef59fd7a1"
)
OUTPUT = Path("results/v3/v3_d6_repair_contract_validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D6 metadata must be a JSON object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def authenticated_json(path_text: str, expected: str) -> dict[str, Any]:
    path = REPO_ROOT / path_text
    if path.suffix != ".json" or sha256(path) != expected:
        raise PermissionError(f"V3-D6 authenticated JSON differs: {path_text}")
    return json_object(path)


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D6 contract validation requires clean worktree")
    contract_path = REPO_ROOT / CONTRACT
    if sha256(contract_path) != CONTRACT_SHA256:
        raise PermissionError("V3-D6 contract SHA differs")
    contract = json_object(contract_path)
    if (
        contract.get("schema_version") != "phase-route-vla.v3.d6-repair-contract.v1"
        or contract.get("status") != "D6_DEVELOPMENT_REPAIR_CONTRACT_FROZEN"
        or contract.get("scope", {}).get("same_development_v2_payload_reused") is not True
        or contract.get("scope", {}).get("fresh_confirmation_claim_allowed") is not False
        or contract.get("scope", {}).get("calibration_v2_reuse_for_repair_or_selection") is not False
        or contract.get("scope", {}).get("independent_test_v2_access_allowed") is not False
        or contract.get("scope", {}).get("active_control_allowed") is not False
    ):
        raise PermissionError("V3-D6 scope semantics differ")

    prerequisite = contract["prerequisite"]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", prerequisite["d5_freeze_commit"], "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise PermissionError("V3-D6 D5 freeze commit is not an ancestor")
    d5_contract = authenticated_json(
        prerequisite["d5_contract_path"], prerequisite["d5_contract_sha256"]
    )
    d5_formal = authenticated_json(
        prerequisite["d5_formal_result_path"], prerequisite["d5_formal_result_sha256"]
    )
    d5_analysis = authenticated_json(
        prerequisite["d5_negative_analysis_path"],
        prerequisite["d5_negative_analysis_sha256"],
    )
    if (
        d5_contract.get("status") != "D5_DEVELOPMENT_CONTRACT_FROZEN"
        or d5_formal.get("status") != prerequisite["required_d5_status"]
        or d5_analysis.get("status") != prerequisite["required_analysis_status"]
        or d5_analysis.get("authorization", {}).get("next_stage")
        != prerequisite["authorized_next_stage"]
    ):
        raise PermissionError("V3-D6 D5 prerequisite semantics differ")

    lineage = contract["lineage"]
    dataset_result = authenticated_json(
        lineage["D5_dataset_result_path"], lineage["D5_dataset_result_sha256"]
    )
    if (
        dataset_result.get("status") != "PASS_V3_D5_DEVELOPMENT_DATASET"
        or dataset_result.get("payload_sha256") != lineage["D5_dataset_payload_sha256"]
        or dataset_result.get("policy_calls") != lineage["policy_call_count"]
        or dataset_result.get("candidate_rows") != lineage["candidate_row_count"]
        or dataset_result.get("clusters") != lineage["cluster_count"]
    ):
        raise PermissionError("V3-D6 dataset metadata differs")

    severity = contract["severity_weight"]
    model = contract["model"]
    threshold = contract["robust_threshold"]
    criteria = contract["development_selection_criteria"]
    if (
        severity.get("formula")
        != "1+clamp(log2(max(ratio,1)),minimum=0,maximum=4)"
        or severity.get("range") != [1.0, 5.0]
        or model.get("trainable_feature_parameter_count") != 194
        or model.get("l2_lambda_grid") != [0.001, 0.01, 0.1]
        or threshold.get("jackknife_order_statistic")
        != "fifth_smallest_of_17_feasible_thresholds"
        or threshold.get("fixed_safety_multiplier") != 0.95
        or threshold.get("threshold_reoptimization_after_multiplier") is not False
        or criteria.get("false_safe_cluster_ucb_at_most") != 0.05
        or criteria.get("criteria_met_is_fresh_confirmation") is not False
    ):
        raise PermissionError("V3-D6 model or robust threshold semantics differ")

    result = {
        "status": "PASS_V3_D6_REPAIR_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d6-contract-validation.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "contract": {
            "path": CONTRACT.as_posix(),
            "sha256": CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "bound_sha256": {
            "D5_contract": prerequisite["d5_contract_sha256"],
            "D5_formal_result": prerequisite["d5_formal_result_sha256"],
            "D5_negative_analysis": prerequisite["d5_negative_analysis_sha256"],
            "D5_dataset_result": lineage["D5_dataset_result_sha256"],
            "D5_dataset_payload": lineage["D5_dataset_payload_sha256"],
        },
        "frozen_repair": {
            "severity_weight_formula": severity["formula"],
            "severity_weight_range": severity["range"],
            "model_family": model["family"],
            "jackknife_views": 17,
            "jackknife_order_statistic": threshold["jackknife_order_statistic"],
            "safety_multiplier": threshold["fixed_safety_multiplier"],
        },
        "access_ledger": {
            "JSON_metadata_files_opened": 5,
            "tensor_payloads_deserialized": 0,
            "model_fits": 0,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "D6_DEVELOPMENT_SELECTION_IMPLEMENTATION_AND_RUN",
            "fresh_confirmation": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D6 refuses to overwrite contract validation")
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
