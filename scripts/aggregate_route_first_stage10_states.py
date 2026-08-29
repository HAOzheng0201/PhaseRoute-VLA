#!/usr/bin/env python3
"""Seal the Stage 10 generated-state payload after the two-pass audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONTRACT_PATH = REPO_ROOT / "a1/vla/dynamic_compute/route_first_stage10.py"
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_phase_route_stage10_contract", CONTRACT_PATH
)
if CONTRACT_SPEC is None or CONTRACT_SPEC.loader is None:
    raise ImportError(f"cannot load Stage 10 contract: {CONTRACT_PATH}")
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = CONTRACT
CONTRACT_SPEC.loader.exec_module(CONTRACT)

FreshStateEvidence = CONTRACT.FreshStateEvidence
PROTOCOL_SHA256 = CONTRACT.PROTOCOL_SHA256
SCHEDULE_SHA256 = CONTRACT.SCHEDULE_SHA256
STATE_ATTESTATION_SCHEMA = CONTRACT.STATE_ATTESTATION_SCHEMA
STATE_PASSES = CONTRACT.STATE_PASSES
STATE_PAYLOAD_SCHEMA = CONTRACT.STATE_PAYLOAD_SCHEMA
STATE_RECORD_SCHEMA = CONTRACT.STATE_RECORD_SCHEMA
load_schedule = CONTRACT.load_schedule
sha256_file = CONTRACT.sha256_file
validate_generation_record_manifest = CONTRACT.validate_generation_record_manifest
validate_two_pass_states = CONTRACT.validate_two_pass_states


INPUT = Path("runs/route_first_stage10_state_records")
OUTPUT = Path("runs/route_first_stage10_states")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"expected JSON object: {path}")
    return dict(value)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("Stage 10 state aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("Stage 10 state aggregation requires a clean worktree")
    schedule = load_schedule(REPO_ROOT)
    input_root = (REPO_ROOT / INPUT).resolve(strict=True)
    generation_path = input_root / "result.json"
    generation = json_object(generation_path)
    current_commit = git_output("rev-parse", "HEAD")
    record_manifest_source = generation.get("record_result_sha256")
    if (
        generation.get("status")
        != "PASS_ROUTE_FIRST_STAGE10_STATE_GENERATION_PENDING_AGGREGATION"
        or generation.get("processes") != 2 * len(schedule)
        or generation.get("source_git_commit") != current_commit
        or generation.get("source_worktree_dirty") is not False
        or generation.get("protocol_sha256") != PROTOCOL_SHA256
        or generation.get("schedule_sha256") != SCHEDULE_SHA256
        or not isinstance(record_manifest_source, Mapping)
    ):
        raise PermissionError("Stage 10 generation-run evidence differs")
    record_manifest = validate_generation_record_manifest(
        schedule, record_manifest_source
    )
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(
            "Stage 10 state aggregation refuses to overwrite evidence"
        )

    evidence_by_pass: dict[int, list[FreshStateEvidence]] = {
        pass_id: [] for pass_id in STATE_PASSES
    }
    pass1_raw = []
    records = []
    for pass_id in STATE_PASSES:
        for spec in schedule:
            directory = (
                input_root
                / f"pass{pass_id}"
                / f"task{spec.task_id:02d}_replicate{spec.replicate_id:02d}"
            )
            result_path = directory / "result.json"
            result = json_object(result_path)
            record_key = (
                f"pass{pass_id}:task{spec.task_id}:replicate{spec.replicate_id}"
            )
            if sha256_file(result_path) != record_manifest[record_key]:
                raise PermissionError("Stage 10 state record manifest hash differs")
            evidence = FreshStateEvidence.from_mapping(result)
            evidence.validate(spec, pass_id)
            state_path = directory / str(result.get("state_payload"))
            raw = state_path.read_bytes()
            if (
                result.get("schema_version") != STATE_RECORD_SCHEMA
                or result.get("status") != "PASS_ROUTE_FIRST_STAGE10_STATE_RECORD"
                or result.get("source_git_commit") != current_commit
                or result.get("source_worktree_dirty") is not False
                or result.get("protocol_sha256") != PROTOCOL_SHA256
                or result.get("schedule_sha256") != SCHEDULE_SHA256
                or result.get("libero_gitlink_commit")
                != "8f1084e3132a39270c3a13ebe37270a43ece2a01"
                or result.get("arm_order") != list(spec.arm_order)
                or result.get("state_dtype") != "float64-le"
                or result.get("state_rank") != 1
                or result.get("model_checkpoint_loaded") is not False
                or result.get("policy_action_sampled") is not False
                or result.get("official_episode_identity_used") is not False
                or result.get("gpu_query_or_initialization") != 0
                or len(raw) != evidence.state_nbytes
                or hashlib.sha256(raw).hexdigest() != evidence.state_sha256
            ):
                raise PermissionError("Stage 10 state record evidence differs")
            value = np.frombuffer(raw, dtype="<f8")
            if value.shape != (evidence.state_dimension,) or not bool(
                np.isfinite(value).all()
            ):
                raise PermissionError("Stage 10 state geometry differs")
            evidence_by_pass[pass_id].append(evidence)
            if pass_id == 1:
                pass1_raw.append(raw)
                records.append(
                    {
                        "task_id": spec.task_id,
                        "replicate_id": spec.replicate_id,
                        "cluster_key": spec.cluster_key,
                        "arm_order": list(spec.arm_order),
                        "state_seed": spec.state_seed,
                        "policy_seed": spec.policy_seed,
                        "state_dimension": evidence.state_dimension,
                        "state_sha256": evidence.state_sha256,
                        "pass1_result_sha256": sha256_file(result_path),
                    }
                )
            else:
                records[len(evidence_by_pass[2]) - 1]["pass2_result_sha256"] = (
                    sha256_file(result_path)
                )
    audit = validate_two_pass_states(
        schedule, evidence_by_pass[1], evidence_by_pass[2]
    )
    incomplete.mkdir(parents=True, exist_ok=False)
    payload_path = incomplete / "fresh_states.pt"
    torch.save(
        {
            "schema_version": STATE_PAYLOAD_SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "schedule_sha256": SCHEDULE_SHA256,
            "source_git_commit": current_commit,
            "task_id": torch.tensor([item.task_id for item in schedule]),
            "replicate_id": torch.tensor([item.replicate_id for item in schedule]),
            "state_seed": torch.tensor([item.state_seed for item in schedule]),
            "policy_seed": torch.tensor([item.policy_seed for item in schedule]),
            "cluster_keys": [item.cluster_key for item in schedule],
            "arm_orders": [list(item.arm_order) for item in schedule],
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
        "schema_version": STATE_ATTESTATION_SCHEMA,
        "status": "PASS_ROUTE_FIRST_STAGE10_FRESH_STATES_FROZEN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "suite": "libero_10",
        "audit": audit,
        "records": records,
        "payload": payload_path.name,
        "payload_bytes": payload_path.stat().st_size,
        "payload_sha256": sha256_file(payload_path),
        "input_sha256": {
            "protocol": PROTOCOL_SHA256,
            "schedule": SCHEDULE_SHA256,
            "generation_result": sha256_file(generation_path),
        },
        "access_ledger": {
            "fresh_states_generated": len(schedule),
            "determinism_audit_regenerations": len(schedule),
            "model_checkpoint_loaded": False,
            "policy_action_sampled": False,
            "official_states_0_to_49_opened": False,
            "V3_D8_or_D10_states_reused": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": (
                "BIND_STATE_PAYLOAD_SHA_IN_CLEAN_COMMIT_BEFORE_ACTIVE_CONTROL"
            ),
            "active_control_authorized_by_this_local_file_alone": False,
            "deployment_authorized": False,
        },
        "claim_boundary": {
            "generated_states_are_official_fixed_benchmark_states": False,
            "policy_or_task_success_evaluated": False,
            "fresh_active_confirmation_complete": False,
            "superiority_claim_authorized": False,
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
