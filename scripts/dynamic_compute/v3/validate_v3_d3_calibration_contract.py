#!/usr/bin/env python3
"""Validate the pre-label V3-D3 calibration contract and D2 bindings."""

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

from a1.vla.dynamic_compute.v3.gripper_v2_calibration import (  # noqa: E402
    D3_CONTRACT_RELATIVE_PATH,
    D3_CONTRACT_SHA256,
    D3_STATUS,
    clopper_pearson_upper,
    load_calibration_selection,
    load_d3_contract,
    load_frozen_d2_final_state,
    validate_d3_prerequisites,
)


DEFAULT_OUTPUT = Path("results/v3/v3_d3_calibration_contract_validation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    args = parse_args()
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D3 contract validation requires a clean worktree")
    contract = load_d3_contract(REPO_ROOT)
    prerequisite = validate_d3_prerequisites(REPO_ROOT)
    selection = load_calibration_selection(REPO_ROOT)
    final_state = load_frozen_d2_final_state(REPO_ROOT)
    result = {
        "status": "PASS_V3_D3_CALIBRATION_CONTRACT_FROZEN",
        "schema_version": "phase-route-vla.v3.d3-contract-validation.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "contract": {
            "status": D3_STATUS,
            "path": D3_CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": D3_CONTRACT_SHA256,
            "contract_id": contract["contract_id"],
        },
        "prerequisite": prerequisite,
        "calibration_selection": {
            "clusters": len(selection),
            "first_key": selection[0].group_key,
            "last_key": selection[-1].group_key,
        },
        "frozen_model": {
            "heads": list(final_state),
            "primary_parameter_count": contract["frozen_model"][
                "primary_parameter_count"
            ],
            "final_lambdas": contract["frozen_model"]["final_lambdas"],
            "refit_on_calibration": False,
        },
        "exact_ucb_boundary_audit": {
            "zero_of_58": clopper_pearson_upper(0, 58),
            "zero_of_59": clopper_pearson_upper(0, 59),
            "one_of_100": clopper_pearson_upper(1, 100),
            "two_of_100": clopper_pearson_upper(2, 100),
        },
        "access_ledger": {
            "development_final_model_payload_opened": True,
            "calibration_selection_metadata_opened": True,
            "calibration_sample_payload_opened": False,
            "independent_test_selection_metadata_opened": True,
            "independent_test_sample_payload_opened": False,
        },
        "claim_boundary": {
            "runtime_threshold_selected": False,
            "shadow_decision_run": False,
            "active_control_run": False,
            "independent_test_run": False,
            "deployment_authorized": False,
            "superiority_claim_authorized": False,
        },
        "next_stage": {
            "authorized": "V3-D3_CALIBRATION_V2_COLLECTION_AND_THRESHOLD_ONLY",
            "episode_indices": list(range(30, 40)),
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
    }
    output = args.output.resolve()
    expected = (REPO_ROOT / DEFAULT_OUTPUT).resolve()
    if output != expected:
        raise PermissionError("V3-D3 validation output path differs")
    sidecar = output.with_suffix(".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("V3-D3 refuses to overwrite validation evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
