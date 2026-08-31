#!/usr/bin/env python3
"""Freeze Stage-11D states after validating two isolated deterministic passes."""

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

from a1.vla.dynamic_compute.route_first_reliability import (  # noqa: E402
    STAGE11D_PROTOCOL_SHA256,
    build_stage11d_schedule,
    validate_stage11d_protocol,
)
from a1.vla.dynamic_compute.route_first_reliability_artifacts import (  # noqa: E402
    STAGE11D_LIBERO_COMMIT,
    STAGE11D_STATE_ATTESTATION_SCHEMA,
    STAGE11D_STATE_PASSES,
    STAGE11D_STATE_PAYLOAD_SCHEMA,
    STAGE11D_STATE_RECORDS_RELATIVE_PATH,
    STAGE11D_STATE_RECORD_SCHEMA,
    STAGE11D_STATE_RUN_SCHEMA,
    STAGE11D_STATES_RELATIVE_PATH,
    Stage11DStateEvidence,
    sha256_file,
    validate_state_runner_readiness,
    validate_two_pass_states,
)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"expected JSON object: {path}")
    return dict(value)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("Stage-11D state aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("Stage-11D aggregation requires clean worktree")
    validate_stage11d_protocol(REPO_ROOT)
    readiness = validate_state_runner_readiness(REPO_ROOT)
    readiness_path = (
        REPO_ROOT
        / "results/route_first/route_first_stage11d_state_runner_readiness.json"
    )
    schedule = build_stage11d_schedule()
    input_root = (REPO_ROOT / STAGE11D_STATE_RECORDS_RELATIVE_PATH).resolve(strict=True)
    generation_path = input_root / "result.json"
    generation = json_object(generation_path)
    current_commit = git_output("rev-parse", "HEAD")
    manifest = generation.get("record_result_sha256")
    if (
        generation.get("schema_version") != STAGE11D_STATE_RUN_SCHEMA
        or generation.get("status")
        != "PASS_ROUTE_FIRST_STAGE11D_STATE_RUN_PENDING_AGGREGATION"
        or generation.get("processes") != 2 * len(schedule)
        or generation.get("source_git_commit") != current_commit
        or generation.get("source_worktree_dirty") is not False
        or generation.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
        or generation.get("runner_readiness_sha256") != sha256_file(readiness_path)
        or not isinstance(manifest, Mapping)
        or len(manifest) != 2 * len(schedule)
    ):
        raise PermissionError("Stage-11D state run evidence differs")
    output = REPO_ROOT / STAGE11D_STATES_RELATIVE_PATH
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("Stage-11D refuses to overwrite state aggregate")

    evidence_by_pass: dict[int, list[Stage11DStateEvidence]] = {
        value: [] for value in STAGE11D_STATE_PASSES
    }
    pass1_raw = []
    audit_records = []
    for pass_id in STAGE11D_STATE_PASSES:
        for record in schedule:
            directory = (
                input_root
                / f"pass{pass_id}"
                / f"task{record.task_id:02d}_replicate{record.replicate_id:02d}"
            )
            result_path = directory / "result.json"
            result = json_object(result_path)
            key = f"pass{pass_id}:task{record.task_id}:replicate{record.replicate_id}"
            if sha256_file(result_path) != manifest.get(key):
                raise PermissionError("Stage-11D state record manifest differs")
            evidence = Stage11DStateEvidence.from_mapping(result)
            evidence.validate(record, pass_id)
            raw = (directory / str(result.get("state_payload"))).read_bytes()
            if (
                result.get("schema_version") != STAGE11D_STATE_RECORD_SCHEMA
                or result.get("status") != "PASS_ROUTE_FIRST_STAGE11D_STATE_RECORD"
                or result.get("source_git_commit") != current_commit
                or result.get("source_worktree_dirty") is not False
                or result.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
                or result.get("runner_readiness_sha256") != sha256_file(readiness_path)
                or result.get("libero_gitlink_commit") != STAGE11D_LIBERO_COMMIT
                or result.get("state_dtype") != "float64-le"
                or result.get("state_rank") != 1
                or result.get("model_checkpoint_loaded") is not False
                or result.get("policy_action_sampled") is not False
                or result.get("official_episode_identity_used") is not False
                or result.get("historical_generated_state_payload_opened") is not False
                or result.get("gpu_query_or_initialization") != 0
                or len(raw) != evidence.state_nbytes
                or hashlib.sha256(raw).hexdigest() != evidence.state_sha256
            ):
                raise PermissionError("Stage-11D state record evidence differs")
            value = np.frombuffer(raw, dtype="<f8")
            if value.shape != (evidence.state_dimension,) or not bool(
                np.isfinite(value).all()
            ):
                raise PermissionError("Stage-11D state bytes differ")
            evidence_by_pass[pass_id].append(evidence)
            if pass_id == 1:
                pass1_raw.append(raw)
                audit_records.append(
                    {
                        "task_id": record.task_id,
                        "replicate_id": record.replicate_id,
                        "split": record.split,
                        "cluster_key": record.cluster_key,
                        "state_seed": record.state_seed,
                        "policy_seed": record.policy_seed,
                        "state_dimension": evidence.state_dimension,
                        "state_sha256": evidence.state_sha256,
                        "pass1_result_sha256": sha256_file(result_path),
                    }
                )
            else:
                audit_records[len(evidence_by_pass[2]) - 1][
                    "pass2_result_sha256"
                ] = sha256_file(result_path)
    audit = validate_two_pass_states(
        schedule, evidence_by_pass[1], evidence_by_pass[2]
    )
    incomplete.mkdir(parents=True, exist_ok=False)
    payload_path = incomplete / "fresh_states.pt"
    torch.save(
        {
            "schema_version": STAGE11D_STATE_PAYLOAD_SCHEMA,
            "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
            "source_git_commit": current_commit,
            "task_id": torch.tensor([item.task_id for item in schedule]),
            "replicate_id": torch.tensor([item.replicate_id for item in schedule]),
            "state_seed": torch.tensor([item.state_seed for item in schedule]),
            "policy_seed": torch.tensor([item.policy_seed for item in schedule]),
            "splits": [item.split for item in schedule],
            "cluster_keys": [item.cluster_key for item in schedule],
            "state_sha256": [item.state_sha256 for item in evidence_by_pass[1]],
            "states": [
                torch.from_numpy(np.frombuffer(raw, dtype="<f8").copy())
                for raw in pass1_raw
            ],
            "determinism_passes": 2,
            "initial_task_success_all_false": True,
            "official_episode_identity_used": False,
            "policy_rollout_performed": False,
        },
        payload_path,
    )
    result = {
        "schema_version": STAGE11D_STATE_ATTESTATION_SCHEMA,
        "status": "PASS_ROUTE_FIRST_STAGE11D_STATES_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "suite": "libero_10",
        "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
        "runner_readiness_status": readiness["status"],
        "runner_readiness_sha256": sha256_file(readiness_path),
        "audit": audit,
        "records": audit_records,
        "payload": payload_path.name,
        "payload_bytes": payload_path.stat().st_size,
        "payload_sha256": sha256_file(payload_path),
        "input_sha256": {
            "generation_result": sha256_file(generation_path),
            "runner_readiness": sha256_file(readiness_path),
        },
        "access_ledger": {
            "fresh_states_generated": len(schedule),
            "determinism_audit_regenerations": len(schedule),
            "model_checkpoint_loaded": False,
            "policy_action_sampled": False,
            "official_states_0_to_49_opened": False,
            "V3_D8_or_route_first_Stage10_states_reused": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "BIND_STATE_ATTESTATION_SHA_BEFORE_ORIGINAL_A1_COLLECTION",
            "original_A1_collection_authorized_by_this_file_alone": False,
            "same_noise_replay_authorized": False,
            "active_control_authorized": False,
        },
        "claim_boundary": {
            "policy_or_task_success_evaluated": False,
            "reliability_model_trained": False,
            "speedup_or_superiority_claim_authorized": False,
        },
    }
    result_path = incomplete / "state_attestation.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "state_attestation.sha256").write_text(
        f"{sha256_file(result_path)}  state_attestation.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
