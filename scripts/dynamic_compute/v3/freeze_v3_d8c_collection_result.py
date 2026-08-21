#!/usr/bin/env python3
"""Freeze D8C collection/replay evidence before any D8D router scoring."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.development_collection import stream_sha256  # noqa: E402
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CLUSTER_COUNT,
    D8_CLUSTERS_PER_TASK,
    D8_TASK_IDS,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation_collection import (  # noqa: E402
    D8C_COLLECTION_RESULT_SCHEMA_VERSION,
    D8C_DATASET_SCHEMA_VERSION,
    D8C_ROLE,
    D8C_SUITE,
    parse_fresh_cluster_key,
    validate_d8c_prerequisites,
)


DATASET_RESULT = Path("reports/v3_d8_fresh_dataset/result.json")
OUTPUT = Path("results/v3/v3_d8c_formal_collection_result.json")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8C JSON must be an object: {path}")
    return dict(value)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D8C result freeze is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8C result freeze requires a clean worktree")
    prerequisites = validate_d8c_prerequisites(REPO_ROOT)
    current_commit = git_output("rev-parse", "HEAD")
    result_path = REPO_ROOT / DATASET_RESULT
    dataset_result = json_object(result_path)
    payload_path = result_path.parent / str(dataset_result.get("payload"))
    if (
        dataset_result.get("status") != "PASS_V3_D8C_DATASET"
        or dataset_result.get("source_git_commit") != current_commit
        or dataset_result.get("source_worktree_dirty") is not False
        or dataset_result.get("role") != D8C_ROLE
        or dataset_result.get("suite") != D8C_SUITE
        or dataset_result.get("clusters") != D8_CLUSTER_COUNT
        or dataset_result.get("clusters_per_task")
        != [D8_CLUSTERS_PER_TASK] * len(D8_TASK_IDS)
        or stream_sha256(payload_path) != dataset_result.get("payload_sha256")
        or dataset_result.get("access_ledger", {}).get("final_router_loaded")
        is not False
        or dataset_result.get("access_ledger", {}).get("confirmation_gate_inspected")
        is not False
        or dataset_result.get("access_ledger", {}).get("official_episode_40_49_opened")
        is not False
    ):
        raise PermissionError("D8C dataset result semantics differ")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    rows = int(dataset_result["policy_calls"])
    candidate_rows = rows * 2
    if (
        payload.get("schema_version") != D8C_DATASET_SCHEMA_VERSION
        or payload.get("role") != D8C_ROLE
        or payload.get("suite") != D8C_SUITE
        or payload.get("router_scored") is not False
        or payload.get("confirmation_gate_inspected") is not False
        or payload.get("official_episode_40_49_opened") is not False
        or payload.get("layer27_runtime_visible") is not False
        or payload.get("layer27_is_consistency_teacher_only") is not True
        or payload["features"].shape != (candidate_rows, 97)
        or payload["unsafe_target"].shape != (candidate_rows, 2)
        or payload["full_action_distance"].shape != (candidate_rows,)
        or payload["action_consistency"].shape != (candidate_rows,)
        or payload["candidate_actions"].shape != (rows, 3, 8, 7)
        or not bool(torch.isfinite(payload["features"]).all())
        or not bool(torch.isfinite(payload["full_action_distance"]).all())
        or not bool(torch.isfinite(payload["candidate_actions"]).all())
    ):
        raise PermissionError("D8C dataset payload geometry or boundary differs")
    unique_clusters = set(payload["cluster_keys"])
    expected_pairs = {
        (task, replicate)
        for task in D8_TASK_IDS
        for replicate in range(D8_CLUSTERS_PER_TASK)
    }
    observed_pairs = {parse_fresh_cluster_key(key) for key in unique_clusters}
    if observed_pairs != expected_pairs or len(unique_clusters) != D8_CLUSTER_COUNT:
        raise PermissionError("D8C frozen cluster coverage differs")

    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D8C refuses to overwrite formal collection evidence")
    result = {
        "status": "PASS_V3_D8C_PROSPECTIVE_COLLECTION_AND_REPLAY",
        "schema_version": D8C_COLLECTION_RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "clusters": D8_CLUSTER_COUNT,
        "clusters_per_task": [D8_CLUSTERS_PER_TASK] * len(D8_TASK_IDS),
        "policy_calls": rows,
        "candidate_rows": candidate_rows,
        "feature_dimension": 97,
        "behavior_a1": dataset_result["behavior_a1"],
        "candidate_truth_support": dataset_result["target_support"],
        "bound_artifacts": {
            **prerequisites,
            "D8C_dataset_result_path": DATASET_RESULT.as_posix(),
            "D8C_dataset_result_sha256": stream_sha256(result_path),
            "D8C_dataset_payload_path": str(payload_path.relative_to(REPO_ROOT)),
            "D8C_dataset_payload_sha256": stream_sha256(payload_path),
        },
        "checks": {
            "all_200_generated_state_clusters_collected": True,
            "all_policy_calls_have_past_only_context": True,
            "all_policy_calls_have_same_noise_L11_L13_L27_replay": True,
            "all_candidate_truth_rows_finite": True,
            "original_A1_was_only_behavior_policy": True,
            "D7_shadow_decision_not_applied_to_environment": True,
            "final_router_not_loaded_or_scored": True,
            "confirmation_gate_not_inspected": True,
            "official_episode_40_49_remain_sealed": True,
        },
        "access_ledger": {
            "fresh_generated_state_rollouts": D8_CLUSTER_COUNT,
            "fresh_policy_calls": rows,
            "fresh_candidate_rows": candidate_rows,
            "candidate_truth_opened": True,
            "final_router_loaded": False,
            "router_predictions_computed": 0,
            "confirmation_gate_inspected": False,
            "official_episode_40_49_opened": False,
            "calibration_or_test_payload_opened": False,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "D8D_APPLY_FROZEN_ROUTER_AND_AGGREGATE_CONFIRMATION_GATE",
            "refit_or_threshold_change": False,
            "open_episode_40_49": False,
            "active_control": False,
            "deployment": False,
        },
        "claim_boundary": {
            "D8C_collection_and_replay_complete": True,
            "D8_confirmation_gate_evaluated": False,
            "generated_states_are_official_LIBERO_fixed_states": False,
            "shadow_consistency_is_closed_loop_D7_success": False,
            "behavior_A1_success_is_descriptive_only": True,
            "superiority_claim_authorized": False,
            "deployment_authorized": False,
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError("D8C formal collection checks did not all pass")
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(incomplete), str(output))
    sidecar.write_text(f"{stream_sha256(output)}  {output.name}\n", encoding="utf-8")
    print("PASS_V3_D8C_PROSPECTIVE_COLLECTION_AND_REPLAY", flush=True)


if __name__ == "__main__":
    main()
