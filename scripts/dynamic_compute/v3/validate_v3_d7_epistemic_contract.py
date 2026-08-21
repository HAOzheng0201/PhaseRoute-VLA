#!/usr/bin/env python3
"""Validate frozen V3-D7 contract using metadata only."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = Path(
    "configs/research/v3/joint_reliability/d7_epistemic_ensemble_contract.json"
)
CONTRACT_SHA256 = (
    "7e1f8934e33ae33493b950eabc1142c1f6cd7103ef7b4ad735d6c8b13a5afdea"
)
D6_FORMAL = Path("results/v3/v3_d6_formal_development_result.json")
D6_FORMAL_SHA256 = (
    "c8bda5b40afb93c5fe815e71224da1e0f99570e4b73970e4cf8489b78fd62fc6"
)
D6_ANALYSIS = Path("results/v3/v3_d6_negative_analysis.json")
D6_ANALYSIS_SHA256 = (
    "e3005e3dd51b5f712c034607d1130180a7d79e7f8354f7298b7840751f2b9fd7"
)
D6_CONTRACT = Path("configs/research/v3/joint_reliability/d6_repair_contract.json")
D6_CONTRACT_SHA256 = (
    "28185ce5431cf438d20cb7cfdfd0e20d5859b6a99f1bdafa81d18faef59fd7a1"
)
D5_DATASET_RESULT = Path("reports/v3_d5_development_dataset/result.json")
D5_DATASET_RESULT_SHA256 = (
    "7b4facd767594974359bef11edec83bbe3df66c3ee4c5c3981814992f792186d"
)
OUTPUT = Path("results/v3/v3_d7_epistemic_contract_validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D7 validation metadata must be an object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D7 contract validation requires clean worktree")
    paths = {
        "contract": (CONTRACT, CONTRACT_SHA256),
        "D6_formal": (D6_FORMAL, D6_FORMAL_SHA256),
        "D6_analysis": (D6_ANALYSIS, D6_ANALYSIS_SHA256),
        "D6_contract": (D6_CONTRACT, D6_CONTRACT_SHA256),
        "D5_dataset_result": (D5_DATASET_RESULT, D5_DATASET_RESULT_SHA256),
    }
    metadata = {}
    for name, (relative, expected) in paths.items():
        path = REPO_ROOT / relative
        if sha256(path) != expected:
            raise PermissionError(f"V3-D7 {name} SHA differs")
        metadata[name] = json_object(path)
    contract = metadata["contract"]
    analysis = metadata["D6_analysis"]
    if (
        contract.get("schema_version")
        != "phase-route-vla.v3.d7-epistemic-ensemble-contract.v1"
        or contract.get("status") != "D7_EPISTEMIC_ENSEMBLE_CONTRACT_FROZEN"
        or contract.get("epistemic_ensemble", {}).get("head_count") != 5
        or contract.get("epistemic_ensemble", {}).get(
            "full_action_runtime_score"
        )
        != "maximum_full_action_score_across_five_heads"
        or contract.get("nested_oof", {}).get("total_model_fits") != 4680
        or contract.get("threshold", {}).get("jackknife_threshold_views") != 0
        or contract.get("development_selection_criteria", {}).get(
            "false_full_action_clusters_at_most"
        )
        != 3
        or contract.get("authorization", {}).get("independent_test_authorized")
        is not False
        or analysis.get("status") != "PASS_V3_D6_NEGATIVE_RESULT_ANALYSIS"
        or analysis.get("authorization", {}).get("next_stage")
        != "D7_PROTOCOL_DESIGN_ONLY_USING_D5_D6_AS_REUSED_DEVELOPMENT_EVIDENCE"
    ):
        raise PermissionError("V3-D7 frozen contract semantics differ")
    result = {
        "status": "PASS_V3_D7_EPISTEMIC_ENSEMBLE_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d7-contract-validation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "contract": {
            "path": CONTRACT.as_posix(),
            "sha256": CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "bound_sha256": {
            "D6_formal": D6_FORMAL_SHA256,
            "D6_negative_analysis": D6_ANALYSIS_SHA256,
            "D6_contract": D6_CONTRACT_SHA256,
            "D5_dataset_result": D5_DATASET_RESULT_SHA256,
        },
        "frozen_design": {
            "head_count": 5,
            "feature_parameters": 970,
            "full_action_score": "maximum_across_five_heads",
            "gripper_score": "head_0_only",
            "fits_per_outer": 260,
            "total_model_fits": 4680,
            "minimum_early_exit_fraction": 0.10,
            "maximum_false_full_action_clusters": 3,
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
        "authorization": contract["authorization"],
        "claim_boundary": contract["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D7 refuses to overwrite contract validation")
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(incomplete), str(output))
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
