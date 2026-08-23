#!/usr/bin/env python3
"""Fit and freeze the development-only final five-head D8B router."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.epistemic_ensemble import (  # noqa: E402
    D7_HEAD_COUNT,
    head_fit_masks,
)
from a1.vla.dynamic_compute.v3.epistemic_ensemble_oof import (  # noqa: E402
    fit_head_ensemble,
    predict_head_ensemble,
    select_d7_threshold,
    threshold_selection_dict,
)
from a1.vla.dynamic_compute.v3.final_router import (  # noqa: E402
    D8B_L2_LAMBDA,
    D8B_PAYLOAD_SCHEMA_VERSION,
    D8B_RESULT_SCHEMA_VERSION,
    FinalFiveHeadRouter,
    final_router_state,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CONTRACT_SHA256,
    D8_SCHEDULE_SHA256,
    load_d8_contract,
)
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_CONTRACT_SHA256,
    development_data_from_mapping,
)
from a1.vla.dynamic_compute.v3.severity_reliability import (  # noqa: E402
    severity_weights,
)


D5_DATASET_RESULT = Path("reports/v3_d5_development_dataset/result.json")
D5_DATASET_RESULT_SHA256 = (
    "7b4facd767594974359bef11edec83bbe3df66c3ee4c5c3981814992f792186d"
)
D5_DATASET_PAYLOAD_SHA256 = (
    "cf40a9802e37d2335668db7f7e24194a3316d552183151cc780fecb5424137df"
)
D5_DATASET_SCHEMA = "phase-route-vla.v3.d5-joint-development-dataset.v1"
D7_OOF_RESULT = Path("reports/v3_d7_development_oof/result.json")
D7_OOF_RESULT_SHA256 = (
    "600370bf978450afc8756cfe7929b36b33ed9d7da716a463902e13c2d0ab3ea9"
)
D7_OOF_PAYLOAD_SHA256 = (
    "ada55c17e7bbf7c6a5833c2a832c77f13249a9fd3c7aff6d6e0c842dd242a35d"
)
D7_OOF_PAYLOAD_SCHEMA = "phase-route-vla.v3.d7-reused-development-payload.v1"
D8_CONTRACT_VALIDATION = Path(
    "results/v3/v3_d8_fresh_confirmation_contract_validation.json"
)
D8_CONTRACT_VALIDATION_SHA256 = (
    "ccda03321468f78eb483b0fe276b7d3eed4e92653968abd53ba83c4986f60f1f"
)
OUTPUT = Path("reports/v3_d8_final_router")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8B JSON must be an object: {path}")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D8B fitting is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D8B fitting requires a clean worktree")
    contract = load_d8_contract(REPO_ROOT)
    finalization = contract["D7_final_router_finalization"]
    if (
        finalization.get("head_count") != D7_HEAD_COUNT
        or finalization.get("lambda") != D8B_L2_LAMBDA
        or finalization.get("final_model_fits") != D7_HEAD_COUNT
        or finalization.get("confirmation_state_or_rollout_access_allowed") is not False
        or finalization.get("gpu_allowed") is not False
    ):
        raise PermissionError("V3-D8B frozen finalization semantics differ")
    validation_path = REPO_ROOT / D8_CONTRACT_VALIDATION
    validation = json_object(validation_path)
    if (
        sha256(validation_path) != D8_CONTRACT_VALIDATION_SHA256
        or validation.get("status")
        != "PASS_V3_D8_FRESH_CONFIRMATION_CONTRACT_FROZEN"
    ):
        raise PermissionError("V3-D8B contract validation evidence differs")

    d5_result_path = REPO_ROOT / D5_DATASET_RESULT
    d5_result = json_object(d5_result_path)
    if (
        sha256(d5_result_path) != D5_DATASET_RESULT_SHA256
        or d5_result.get("status") != "PASS_V3_D5_DEVELOPMENT_DATASET"
        or d5_result.get("payload_sha256") != D5_DATASET_PAYLOAD_SHA256
    ):
        raise PermissionError("V3-D8B D5 dataset metadata differs")
    d5_payload_path = d5_result_path.parent / str(d5_result["payload"])
    if sha256(d5_payload_path) != D5_DATASET_PAYLOAD_SHA256:
        raise PermissionError("V3-D8B D5 dataset payload SHA differs")
    dataset = torch.load(d5_payload_path, map_location="cpu", weights_only=True)
    if (
        dataset.get("schema_version") != D5_DATASET_SCHEMA
        or dataset.get("contract_sha256") != D5_CONTRACT_SHA256
        or dataset.get("calibration_or_test_payload_opened") is not False
        or dataset.get("layer27_runtime_visible") is not False
        or "full_action_distance" not in dataset
    ):
        raise PermissionError("V3-D8B D5 dataset payload semantics differ")
    data = development_data_from_mapping(dataset)
    full_action_distance = dataset["full_action_distance"].detach().cpu().contiguous()

    d7_result_path = REPO_ROOT / D7_OOF_RESULT
    d7_result = json_object(d7_result_path)
    if (
        sha256(d7_result_path) != D7_OOF_RESULT_SHA256
        or d7_result.get("status")
        != "PROMISING_V3_D7_REUSED_DEVELOPMENT_SELECTION"
        or d7_result.get("payload_sha256") != D7_OOF_PAYLOAD_SHA256
        or not all(d7_result.get("gate_checks", {}).values())
    ):
        raise PermissionError("V3-D8B D7 OOF metadata differs")
    d7_payload_path = d7_result_path.parent / str(d7_result["payload"])
    if sha256(d7_payload_path) != D7_OOF_PAYLOAD_SHA256:
        raise PermissionError("V3-D8B D7 OOF payload SHA differs")
    d7_payload = torch.load(d7_payload_path, map_location="cpu", weights_only=True)
    if (
        d7_payload.get("schema_version") != D7_OOF_PAYLOAD_SCHEMA
        or d7_payload.get("calibration_or_test_payload_opened") is not False
        or d7_payload.get("active_control") is not False
        or not torch.equal(d7_payload["task_id"], data.task_id)
        or not torch.equal(d7_payload["episode_index"], data.episode_index)
        or not torch.equal(d7_payload["candidate_layer"], data.candidate_layer)
        or not torch.equal(d7_payload["unsafe_target"], data.unsafe_target)
    ):
        raise PermissionError("V3-D8B D7 OOF payload semantics differ")

    torch.manual_seed(50_260_821)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    threshold = select_d7_threshold(
        d7_payload["OOF_score"],
        data.action_consistency,
        data.unsafe_target,
        data.task_id,
        data.episode_index,
    )
    if (
        not threshold.feasible
        or threshold.full_threshold is None
        or threshold.runtime_threshold is None
    ):
        raise RuntimeError("V3-D8B final OOF threshold is infeasible")
    base_mask = torch.ones(data.rows, dtype=torch.bool)
    row_severity = severity_weights(full_action_distance)
    models = fit_head_ensemble(
        data,
        row_severity,
        base_mask,
        l2_lambda=D8B_L2_LAMBDA,
        max_iterations=500,
    )
    router = FinalFiveHeadRouter(
        models=models,
        full_threshold=threshold.full_threshold,
        runtime_threshold=threshold.runtime_threshold,
    )
    router.validate()
    head_prediction, combined, head_range, _ = predict_head_ensemble(
        models, data.features, data.candidate_layer
    )
    masks = head_fit_masks(base_mask, data.episode_index)
    if (
        head_prediction.shape != (D7_HEAD_COUNT, data.rows, 2)
        or combined.shape != (data.rows, 2)
        or not bool(torch.isfinite(head_range).all())
    ):
        raise RuntimeError("V3-D8B fitted router prediction geometry differs")

    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D8B refuses to overwrite final router")
    incomplete.mkdir(parents=True, exist_ok=False)
    payload_path = incomplete / "final_router.pt"
    state = final_router_state(router)
    torch.save(
        {
            "schema_version": D8B_PAYLOAD_SCHEMA_VERSION,
            "D8_contract_sha256": D8_CONTRACT_SHA256,
            "D8_schedule_sha256": D8_SCHEDULE_SHA256,
            **state,
            "feature_dimension": data.features.shape[1],
            "head_count": len(models),
            "fit_rows_per_head": [int(mask.sum()) for mask in masks],
            "development_rows": data.rows,
            "development_calls": data.calls,
            "threshold_source": "frozen_D7_outer_OOF_scores_and_development_truth",
            "confirmation_state_or_rollout_accessed": False,
            "calibration_or_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
        },
        payload_path,
    )
    threshold_dict = threshold_selection_dict(threshold)
    result = {
        "status": "PASS_V3_D8B_FINAL_ROUTER_FROZEN",
        "schema_version": D8B_RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "development_only_final_router_for_prospective_confirmation",
        "head_count": len(models),
        "lambda": D8B_L2_LAMBDA,
        "final_model_fits": len(models),
        "fit_rows_per_head": [int(mask.sum()) for mask in masks],
        "threshold": threshold_dict,
        "gripper_threshold": router.gripper_threshold,
        "action_consistency_threshold": router.action_consistency_threshold,
        "development_diagnostic": {
            "rows": data.rows,
            "calls": data.calls,
            "head_range_rows_above_1e-6": int((head_range > 1.0e-6).sum()),
            "head_range_fraction_above_1e-6": float(
                (head_range > 1.0e-6).double().mean()
            ),
            "all_predictions_finite": bool(torch.isfinite(combined).all()),
        },
        "payload": payload_path.name,
        "payload_sha256": sha256(payload_path),
        "input_sha256": {
            "D8_contract": D8_CONTRACT_SHA256,
            "D8_schedule": D8_SCHEDULE_SHA256,
            "D8_contract_validation": D8_CONTRACT_VALIDATION_SHA256,
            "D5_dataset_result": D5_DATASET_RESULT_SHA256,
            "D5_dataset_payload": D5_DATASET_PAYLOAD_SHA256,
            "D7_OOF_result": D7_OOF_RESULT_SHA256,
            "D7_OOF_payload": D7_OOF_PAYLOAD_SHA256,
        },
        "access_ledger": {
            "development_payloads_opened": 2,
            "development_final_model_fits": len(models),
            "confirmation_state_or_rollout_accessed": False,
            "fresh_policy_rollout": False,
            "calibration_or_test_payload_opened": False,
            "official_episode_40_49_opened": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "claim_boundary": {
            "development_diagnostic_is_fresh_confirmation": False,
            "router_is_closed_loop_validated": False,
            "superiority_claim_authorized": False,
            "deployment_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{sha256(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
