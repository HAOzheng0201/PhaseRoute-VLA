#!/usr/bin/env python3
"""Validate the frozen V3-D5 contract using JSON metadata only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = Path(
    "configs/research/v3/joint_reliability/d5_development_contract.json"
)
CONTRACT_SHA256 = (
    "e0a584e76f03d0f1b43cd5bbd3477ee2e3694f5425642868b3ec563edd52a29f"
)
OUTPUT_PATH = Path("results/v3/v3_d5_development_contract_validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D5 metadata must be a JSON object: {path}")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def authenticated_json(entry: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    path = REPO_ROOT / str(entry["path"])
    if path.suffix != ".json" or sha256(path) != entry["sha256"]:
        raise PermissionError(f"D5 authenticated JSON differs: {context}")
    return json_object(path)


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 contract validation requires clean worktree")
    contract_file = REPO_ROOT / CONTRACT_PATH
    if sha256(contract_file) != CONTRACT_SHA256:
        raise PermissionError("V3-D5 contract SHA-256 differs")
    contract = json_object(contract_file)
    scope = contract.get("scope", {})
    routing = contract.get("routing", {})
    if (
        contract.get("schema_version")
        != "phase-route-vla.v3.d5-development-contract.v1"
        or contract.get("status") != "D5_DEVELOPMENT_CONTRACT_FROZEN"
        or scope.get("pre_target_inspection_design") is not True
        or scope.get("d5_joint_target_distribution_opened_at_freeze") is not False
        or scope.get("calibration_v2_reuse_for_repair_or_selection") is not False
        or scope.get("independent_test_v2_access_allowed") is not False
        or scope.get("active_control_allowed") is not False
        or routing.get("legacy_motion_and_tail")
        != "diagnostic_only_not_runtime_hard_veto"
        or routing.get("non_compensating_and") is not True
    ):
        raise PermissionError("V3-D5 frozen scope or routing semantics differ")

    prerequisite = contract["prerequisite"]
    if subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            prerequisite["d4b_freeze_commit"],
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0:
        raise PermissionError("V3-D5 D4B freeze commit is not an ancestor")
    d4b_result = authenticated_json(
        {
            "path": prerequisite["d4b_result_path"],
            "sha256": prerequisite["d4b_result_sha256"],
        },
        context="D4B formal result",
    )
    d4b_analysis = authenticated_json(
        {
            "path": prerequisite["d4b_analysis_path"],
            "sha256": prerequisite["d4b_analysis_sha256"],
        },
        context="D4B negative analysis",
    )
    if (
        d4b_result.get("status") != "NEGATIVE_V3_D4B_FORMAL_SHADOW_GATE"
        or d4b_analysis.get("status") != prerequisite["required_analysis_status"]
        or d4b_analysis.get("authorization", {}).get("active_control_authorized")
        is not False
    ):
        raise PermissionError("V3-D5 D4B prerequisite semantics differ")

    lineage = contract["lineage"]
    selection_path = REPO_ROOT / lineage["selection_path"]
    if sha256(selection_path) != lineage["selection_sha256"]:
        raise PermissionError("V3-D5 development selection SHA differs")
    selection = json_object(selection_path)
    records = selection.get("records", [])
    identities = [
        (int(record["task_id"]), int(record["episode_index"]))
        for record in records
    ]
    expected_identities = [
        (task, episode) for task in range(10) for episode in range(12, 30)
    ]
    if identities != expected_identities:
        raise PermissionError("V3-D5 development selection identities differ")

    inputs = contract["authenticated_inputs"]
    d2_formal = authenticated_json(
        inputs["d2_formal_attestation"], context="D2 formal attestation"
    )
    context_result = authenticated_json(
        inputs["d2_context_result"], context="D2 context result"
    )
    dataset_result = authenticated_json(
        inputs["d2_dataset_result"], context="D2 dataset result"
    )
    d3_formal = authenticated_json(
        {
            "path": inputs["frozen_gripper_calibration"]["formal_attestation_path"],
            "sha256": inputs["frozen_gripper_calibration"]["formal_attestation_sha256"],
        },
        context="D3 formal attestation",
    )
    d3_result = authenticated_json(
        {
            "path": inputs["frozen_gripper_calibration"]["result_path"],
            "sha256": inputs["frozen_gripper_calibration"]["result_sha256"],
        },
        context="D3 calibration result",
    )
    if (
        d2_formal.get("status") != "PASS_V3_D2_FULL_DEVELOPMENT_GATE"
        or context_result.get("status") != "PASS_V3_D2_CONTEXT"
        or context_result.get("payload_sha256")
        != inputs["d2_context_result"]["payload_sha256"]
        or dataset_result.get("status") != "PASS_V3_D2_DATASET"
        or dataset_result.get("payload_sha256")
        != inputs["d2_dataset_result"]["payload_sha256"]
        or d3_formal.get("status") != "PASS_V3_D3_CALIBRATION_GATE"
        or d3_result.get("selected_threshold")
        != inputs["frozen_gripper_calibration"]["score_threshold"]
    ):
        raise PermissionError("V3-D5 D2/D3 metadata semantics differ")

    candidate_hashes: dict[str, str] = {}
    candidate_rows = 0
    for shard in range(4):
        entry = inputs["d2_candidate_results"][str(shard)]
        result = authenticated_json(entry, context=f"D2 candidate shard {shard}")
        if (
            result.get("status") != "PASS_V3_D2_CANDIDATE_SHARD"
            or result.get("shard_index") != shard
            or result.get("payload_sha256") != entry["payload_sha256"]
            or result.get("physical_gpu_index") != shard
        ):
            raise PermissionError(f"V3-D5 candidate shard {shard} metadata differs")
        candidate_rows += int(result["rows"])
        candidate_hashes[str(shard)] = entry["sha256"]
    if candidate_rows != lineage["policy_call_count"]:
        raise PermissionError("V3-D5 candidate row count differs")

    model = contract["model"]
    folds = contract["nested_oof"]
    threshold = contract["inner_threshold_selection"]
    gate = contract["formal_development_gate"]
    if (
        model.get("family") != "two_target_layer_anchored_logistic_glm"
        or model.get("trainable_feature_parameter_count") != 194
        or model.get("trainable_feature_parameter_cap") != 256
        or model.get("l2_lambda_grid") != [0.001, 0.01, 0.1]
        or folds.get("outer_folds") != 18
        or folds.get("inner_folds_per_outer") != 17
        or folds.get("normalizer_anchor_model_and_threshold_fit_without_outer_episode")
        is not True
        or threshold.get("minimum_safe_clusters") != 60
        or threshold.get("false_safe_cluster_ucb_at_most") != 0.05
        or gate.get("minimum_safe_clusters") != 60
        or gate.get("minimum_early_exit_call_fraction") != 0.05
        or gate.get("maximum_infeasible_outer_folds") != 0
    ):
        raise PermissionError("V3-D5 model, fold, or statistical gate differs")

    result = {
        "status": "PASS_V3_D5_DEVELOPMENT_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d5-contract-validation.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "contract": {
            "path": CONTRACT_PATH.as_posix(),
            "sha256": CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "lineage": {
            "role": lineage["role"],
            "clusters": len(identities),
            "policy_calls": lineage["policy_call_count"],
            "candidate_rows": lineage["candidate_row_count"],
            "episodes": lineage["episode_indices"],
        },
        "bound_sha256": {
            "d4b_result": prerequisite["d4b_result_sha256"],
            "d4b_analysis": prerequisite["d4b_analysis_sha256"],
            "d2_formal": inputs["d2_formal_attestation"]["sha256"],
            "d2_context_result": inputs["d2_context_result"]["sha256"],
            "d2_dataset_result": inputs["d2_dataset_result"]["sha256"],
            "d2_candidate_results": candidate_hashes,
            "d3_formal": inputs["frozen_gripper_calibration"]["formal_attestation_sha256"],
            "d3_result": inputs["frozen_gripper_calibration"]["result_sha256"],
        },
        "frozen_design": {
            "target_axes": contract["offline_targets"]["target_axes"],
            "model_family": model["family"],
            "outer_folds": folds["outer_folds"],
            "inner_folds_per_outer": folds["inner_folds_per_outer"],
            "gripper_threshold": routing["gripper_safe"]["threshold"],
            "minimum_safe_clusters": gate["minimum_safe_clusters"],
            "false_safe_ucb95_max": gate["false_safe_cluster_ucb_at_most"],
        },
        "access_ledger": {
            "JSON_metadata_files_opened": 13,
            "tensor_payloads_deserialized": 0,
            "d5_joint_target_distribution_opened": False,
            "calibration_v2_payload_reopened": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "model_fits": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "D5_DEVELOPMENT_ONLY_NESTED_OOF_TRAINING",
            "calibration_v2_repair_authorized": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT_PATH
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite contract evidence")
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
