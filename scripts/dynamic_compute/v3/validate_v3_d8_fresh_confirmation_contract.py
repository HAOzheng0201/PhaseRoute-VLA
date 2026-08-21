#!/usr/bin/env python3
"""Validate the frozen V3-D8 contract without opening fresh evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3 import fresh_confirmation as fc  # noqa: E402
from a1.vla.dynamic_compute.v3.gripper_v2_calibration import (  # noqa: E402
    clopper_pearson_upper,
)


CONTRACT = fc.D8_CONTRACT_RELATIVE_PATH
CONTRACT_SHA256 = fc.D8_CONTRACT_SHA256
SCHEDULE = fc.D8_SCHEDULE_RELATIVE_PATH
SCHEDULE_SHA256 = fc.D8_SCHEDULE_SHA256
D7_FORMAL = Path("results/v3/v3_d7_formal_development_result.json")
D7_FORMAL_SHA256 = (
    "4c6d267bb40d2a2b01b92ffa662d0ffb487fb09e1640ca37fa2a10ad8b1a1a07"
)
D7_CONTRACT = Path(
    "configs/research/v3/joint_reliability/d7_epistemic_ensemble_contract.json"
)
D7_CONTRACT_SHA256 = (
    "7e1f8934e33ae33493b950eabc1142c1f6cd7103ef7b4ad735d6c8b13a5afdea"
)
D7_VALIDATION = Path("results/v3/v3_d7_epistemic_contract_validation.json")
D7_VALIDATION_SHA256 = (
    "31dc77519a1ae7b03210a23301f553ca632a90df33eedb3dfcfc17b76386b829"
)
D7_AGGREGATE_RESULT = Path("reports/v3_d7_development_oof/result.json")
D7_AGGREGATE_RESULT_SHA256 = (
    "600370bf978450afc8756cfe7929b36b33ed9d7da716a463902e13c2d0ab3ea9"
)
D7_AGGREGATE_PAYLOAD = Path(
    "reports/v3_d7_development_oof/development_epistemic_nested_oof.pt"
)
D7_AGGREGATE_PAYLOAD_SHA256 = (
    "ada55c17e7bbf7c6a5833c2a832c77f13249a9fd3c7aff6d6e0c842dd242a35d"
)
D0_AUDIT = Path("results/v3/v3_d0_data_lineage_audit.json")
D0_AUDIT_SHA256 = (
    "64d1159b3941fe1e7b806da981a0f47297758dcc2cad87d4e283d03db3a71c4b"
)
OUTPUT = Path("results/v3/v3_d8_fresh_confirmation_contract_validation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D8 validation metadata must be an object")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D8 contract validation requires clean worktree")
    json_inputs = {
        "contract": (CONTRACT, CONTRACT_SHA256),
        "schedule": (SCHEDULE, SCHEDULE_SHA256),
        "D7_formal": (D7_FORMAL, D7_FORMAL_SHA256),
        "D7_contract": (D7_CONTRACT, D7_CONTRACT_SHA256),
        "D7_validation": (D7_VALIDATION, D7_VALIDATION_SHA256),
        "D7_aggregate_result": (
            D7_AGGREGATE_RESULT,
            D7_AGGREGATE_RESULT_SHA256,
        ),
        "D0_audit": (D0_AUDIT, D0_AUDIT_SHA256),
    }
    metadata = {}
    for name, (relative, expected) in json_inputs.items():
        path = REPO_ROOT / relative
        if sha256(path) != expected:
            raise PermissionError(f"V3-D8 {name} SHA differs")
        metadata[name] = json_object(path)
    payload_path = REPO_ROOT / D7_AGGREGATE_PAYLOAD
    if sha256(payload_path) != D7_AGGREGATE_PAYLOAD_SHA256:
        raise PermissionError("V3-D8 D7 aggregate payload SHA differs")

    contract = fc.load_d8_contract(REPO_ROOT)
    records = fc.load_fresh_confirmation_schedule(REPO_ROOT)
    d7_formal = metadata["D7_formal"]
    d7_result = metadata["D7_aggregate_result"]
    d0 = metadata["D0_audit"]
    availability = d0.get("availability_audit", {})
    gate = contract.get("confirmation_gate", {})
    boundary = gate.get("exact_boundary_audit", {})
    if (
        d7_formal.get("status")
        != "PROMISING_V3_D7_REUSED_DEVELOPMENT_SELECTION"
        or d7_formal.get("authorization", {}).get("next_stage")
        != "FRESH_CONFIRMATION_DATA_PROTOCOL_DESIGN_ONLY"
        or d7_formal.get("authorization", {}).get(
            "open_episode_40_49_independent_test"
        )
        is not False
        or d7_result.get("status")
        != "PROMISING_V3_D7_REUSED_DEVELOPMENT_SELECTION"
        or not all(d7_result.get("gate_checks", {}).values())
        or d7_result.get("payload_sha256") != D7_AGGREGATE_PAYLOAD_SHA256
        or d0.get("status") != "PASS_NO_KNOWN_HIT"
        or availability.get("status") != "PASS_STATIC_LIBERO_LONG_AVAILABILITY"
        or availability.get("task_count") != 10
        or availability.get("minimum_initial_states") != 50
        or len(records) != fc.D8_CLUSTER_COUNT
        or boundary.get("one_false_of_120_safe_ucb95")
        != clopper_pearson_upper(1, 120)
        or boundary.get("two_false_of_120_safe_ucb95")
        != clopper_pearson_upper(2, 120)
        or boundary.get("four_false_of_200_safe_ucb95")
        != clopper_pearson_upper(4, 200)
        or boundary.get("five_false_of_200_safe_ucb95")
        != clopper_pearson_upper(5, 200)
        or contract.get("fresh_state_generation", {}).get("generation_passes") != 2
        or contract.get("D7_final_router_finalization", {}).get(
            "final_model_fits"
        )
        != 5
        or contract.get("prospective_collection", {}).get(
            "D7_shadow_decision_applied_to_environment"
        )
        is not False
        or contract.get("authorization", {}).get(
            "fresh_policy_rollout_authorized_on_contract_validation_alone"
        )
        is not False
        or contract.get("authorization", {}).get("open_episode_40_49_authorized")
        is not False
    ):
        raise PermissionError("V3-D8 frozen protocol semantics differ")

    result = {
        "status": "PASS_V3_D8_FRESH_CONFIRMATION_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d8-contract-validation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "contract": {
            "path": CONTRACT.as_posix(),
            "sha256": CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "schedule": {
            "path": SCHEDULE.as_posix(),
            "sha256": SCHEDULE_SHA256,
            "schedule_id": metadata["schedule"]["schedule_id"],
            "records": len(records),
            "tasks": len(fc.D8_TASK_IDS),
            "replicates_per_task": len(fc.D8_REPLICATE_IDS),
            "state_seed_min": min(record.state_seed for record in records),
            "state_seed_max": max(record.state_seed for record in records),
            "policy_seed_min": min(record.policy_seed for record in records),
            "policy_seed_max": max(record.policy_seed for record in records),
        },
        "bound_sha256": {
            "D0_audit": D0_AUDIT_SHA256,
            "D7_contract": D7_CONTRACT_SHA256,
            "D7_contract_validation": D7_VALIDATION_SHA256,
            "D7_formal_result": D7_FORMAL_SHA256,
            "D7_aggregate_result": D7_AGGREGATE_RESULT_SHA256,
            "D7_aggregate_payload": D7_AGGREGATE_PAYLOAD_SHA256,
        },
        "frozen_design": {
            "fresh_generated_state_clusters": 200,
            "state_generation_passes": 2,
            "official_episode_identity_used": False,
            "final_D7_heads": 5,
            "final_D7_lambda": 0.01,
            "final_model_fits": 5,
            "final_threshold_uses_D7_outer_OOF_only": True,
            "minimum_safe_clusters": 120,
            "minimum_early_exit_fraction": 0.10,
            "maximum_false_safe_ucb95": 0.05,
            "maximum_false_full_action_clusters": 3,
            "maximum_false_gripper_calls": 0,
            "maximum_severe_false_clusters": 0,
            "minimum_estimated_FM_reduction": 0.30,
            "confirmation_is_shadow_only": True,
        },
        "access_ledger": {
            "JSON_metadata_files_opened": len(json_inputs),
            "binary_payloads_hashed": 1,
            "tensor_payloads_deserialized": 0,
            "fresh_states_generated": 0,
            "final_model_fits": 0,
            "fresh_policy_rollouts": 0,
            "fresh_candidate_labels_opened": False,
            "calibration_v2_payload_opened": False,
            "independent_test_v2_payload_opened": False,
            "official_episode_40_49_opened": False,
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
        raise FileExistsError("V3-D8 refuses to overwrite contract validation")
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(incomplete), str(output))
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
