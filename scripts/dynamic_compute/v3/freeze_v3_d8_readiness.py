#!/usr/bin/env python3
"""Bind immutable D8A state and D8B router payloads before any D8C rollout."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.d8_artifacts import (  # noqa: E402
    D8A_PAYLOAD_SCHEMA_VERSION,
    D8A_RESULT_SCHEMA_VERSION,
)
from a1.vla.dynamic_compute.v3.final_router import (  # noqa: E402
    D8B_PAYLOAD_SCHEMA_VERSION,
    D8B_RESULT_SCHEMA_VERSION,
    final_router_from_mapping,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CLUSTER_COUNT,
    D8_CONTRACT_SHA256,
    D8_SCHEDULE_SHA256,
    load_d8_contract,
    load_fresh_confirmation_schedule,
)


D8A_RESULT = Path("reports/v3_d8_fresh_states/result.json")
D8B_RESULT = Path("reports/v3_d8_final_router/result.json")
OUTPUT = Path("results/v3/v3_d8_readiness_attestation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8 readiness JSON must be an object: {path}")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D8 readiness validation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D8 readiness freeze requires a clean worktree")
    contract = load_d8_contract(REPO_ROOT)
    schedule = load_fresh_confirmation_schedule(REPO_ROOT)
    current_commit = git_output("rev-parse", "HEAD")

    d8a_result_path = REPO_ROOT / D8A_RESULT
    d8a_result = json_object(d8a_result_path)
    d8a_payload_path = d8a_result_path.parent / str(d8a_result.get("payload"))
    if (
        d8a_result.get("status") != "PASS_V3_D8A_FRESH_STATES_FROZEN"
        or d8a_result.get("schema_version") != D8A_RESULT_SCHEMA_VERSION
        or d8a_result.get("source_git_commit") != current_commit
        or d8a_result.get("source_worktree_dirty") is not False
        or d8a_result.get("payload_sha256") != sha256(d8a_payload_path)
        or d8a_result.get("audit", {}).get("byte_identical_records")
        != D8_CLUSTER_COUNT
        or d8a_result.get("access_ledger", {}).get("fresh_policy_rollout") is not False
        or d8a_result.get("access_ledger", {}).get("official_episode_40_49_opened")
        is not False
    ):
        raise PermissionError("V3-D8 readiness D8A result differs")
    d8a_payload = torch.load(d8a_payload_path, map_location="cpu", weights_only=True)
    if (
        d8a_payload.get("schema_version") != D8A_PAYLOAD_SCHEMA_VERSION
        or d8a_payload.get("D8_contract_sha256") != D8_CONTRACT_SHA256
        or d8a_payload.get("D8_schedule_sha256") != D8_SCHEDULE_SHA256
        or d8a_payload.get("policy_rollout_performed") is not False
        or d8a_payload.get("official_episode_identity_used") is not False
        or len(d8a_payload.get("states", [])) != D8_CLUSTER_COUNT
        or d8a_payload.get("cluster_keys") != [item.cluster_key for item in schedule]
        or not torch.equal(
            d8a_payload["state_seed"], torch.tensor([item.state_seed for item in schedule])
        )
        or not torch.equal(
            d8a_payload["policy_seed"], torch.tensor([item.policy_seed for item in schedule])
        )
    ):
        raise PermissionError("V3-D8 readiness D8A payload semantics differ")
    for state, expected_hash in zip(
        d8a_payload["states"], d8a_payload["state_sha256"]
    ):
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.float64
            or state.ndim != 1
            or not bool(torch.isfinite(state).all())
            or hashlib.sha256(
                np.ascontiguousarray(state.numpy().astype("<f8", copy=False)).tobytes()
            ).hexdigest()
            != expected_hash
        ):
            raise PermissionError("V3-D8 readiness fresh state payload differs")

    d8b_result_path = REPO_ROOT / D8B_RESULT
    d8b_result = json_object(d8b_result_path)
    d8b_payload_path = d8b_result_path.parent / str(d8b_result.get("payload"))
    if (
        d8b_result.get("status") != "PASS_V3_D8B_FINAL_ROUTER_FROZEN"
        or d8b_result.get("schema_version") != D8B_RESULT_SCHEMA_VERSION
        or d8b_result.get("source_git_commit") != current_commit
        or d8b_result.get("source_worktree_dirty") is not False
        or d8b_result.get("payload_sha256") != sha256(d8b_payload_path)
        or d8b_result.get("access_ledger", {}).get(
            "confirmation_state_or_rollout_accessed"
        )
        is not False
        or d8b_result.get("access_ledger", {}).get("official_episode_40_49_opened")
        is not False
    ):
        raise PermissionError("V3-D8 readiness D8B result differs")
    d8b_payload = torch.load(d8b_payload_path, map_location="cpu", weights_only=True)
    if (
        d8b_payload.get("schema_version") != D8B_PAYLOAD_SCHEMA_VERSION
        or d8b_payload.get("D8_contract_sha256") != D8_CONTRACT_SHA256
        or d8b_payload.get("D8_schedule_sha256") != D8_SCHEDULE_SHA256
        or d8b_payload.get("confirmation_state_or_rollout_accessed") is not False
        or d8b_payload.get("calibration_or_test_payload_opened") is not False
    ):
        raise PermissionError("V3-D8 readiness D8B payload semantics differ")
    router = final_router_from_mapping(d8b_payload)
    router.validate()
    if (
        router.full_threshold != d8b_result.get("threshold", {}).get("full_threshold")
        or router.runtime_threshold
        != d8b_result.get("threshold", {}).get("runtime_threshold")
    ):
        raise PermissionError("V3-D8 readiness router threshold differs")

    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("V3-D8 refuses to overwrite readiness attestation")
    result = {
        "status": "PASS_V3_D8A_D8B_READINESS",
        "schema_version": "phase-route-vla.v3.d8-readiness-attestation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "bound_artifacts": {
            "D8_contract_sha256": D8_CONTRACT_SHA256,
            "D8_schedule_sha256": D8_SCHEDULE_SHA256,
            "D8A_result_path": D8A_RESULT.as_posix(),
            "D8A_result_sha256": sha256(d8a_result_path),
            "D8A_payload_path": str(d8a_payload_path.relative_to(REPO_ROOT)),
            "D8A_payload_sha256": sha256(d8a_payload_path),
            "D8B_result_path": D8B_RESULT.as_posix(),
            "D8B_result_sha256": sha256(d8b_result_path),
            "D8B_payload_path": str(d8b_payload_path.relative_to(REPO_ROOT)),
            "D8B_payload_sha256": sha256(d8b_payload_path),
        },
        "readiness_checks": {
            "all_200_fresh_states_present": True,
            "two_generation_passes_byte_identical": True,
            "all_initial_success_predicates_false": True,
            "all_task_local_states_unique": True,
            "final_five_head_router_deserializes": True,
            "final_threshold_bound": True,
            "D8A_preceded_policy_rollout": True,
            "D8B_used_development_only": True,
            "fresh_policy_rollout_not_started": True,
            "official_episode_40_49_remain_sealed": True,
            "active_control_not_run": True,
        },
        "access_ledger": {
            "D8A_state_payload_opened_for_readiness_validation": True,
            "D8B_router_payload_opened_for_readiness_validation": True,
            "fresh_policy_rollouts": 0,
            "fresh_candidate_labels_opened": False,
            "calibration_or_test_payload_opened": False,
            "official_episode_40_49_opened": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": contract["authorization"]["on_D8A_D8B_readiness_pass"],
            "open_episode_40_49": False,
            "active_control": False,
            "deployment": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }
    if not all(result["readiness_checks"].values()):
        raise RuntimeError("V3-D8 readiness checks did not all pass")
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(incomplete), str(output))
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
