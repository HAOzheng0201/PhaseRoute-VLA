#!/usr/bin/env python3
"""Freeze the deterministic 200-state D8A payload after a two-pass audit."""

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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.d8_artifacts import (  # noqa: E402
    D8A_PAYLOAD_SCHEMA_VERSION,
    D8A_RESULT_SCHEMA_VERSION,
    FreshStateEvidence,
    validate_two_pass_evidence,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CONTRACT_SHA256,
    D8_SCHEDULE_SHA256,
    load_d8_contract,
    load_fresh_confirmation_schedule,
)


INPUT = Path("reports/v3_d8_fresh_states_records")
OUTPUT = Path("reports/v3_d8_fresh_states")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8A JSON must be an object: {path}")
    return dict(value)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D8A aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D8A aggregation requires a clean worktree")
    load_d8_contract(REPO_ROOT)
    schedule = load_fresh_confirmation_schedule(REPO_ROOT)
    input_root = (REPO_ROOT / INPUT).resolve(strict=True)
    generation_result_path = input_root / "result.json"
    generation_result = json_object(generation_result_path)
    if (
        generation_result.get("status")
        != "PASS_V3_D8A_ISOLATED_GENERATION_COMPLETE_PENDING_AGGREGATION"
        or generation_result.get("processes") != 400
        or generation_result.get("source_git_commit") != git_output("rev-parse", "HEAD")
        or generation_result.get("source_worktree_dirty") is not False
    ):
        raise PermissionError("V3-D8A generation-run metadata differs")
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D8A refuses to overwrite aggregate evidence")

    started = time.perf_counter()
    pass_records: dict[int, list[FreshStateEvidence]] = {1: [], 2: []}
    raw_states: list[bytes] = []
    record_audit = []
    for pass_id in (1, 2):
        for expected in schedule:
            directory = (
                input_root
                / f"pass{pass_id}"
                / f"task{expected.task_id:02d}_replicate{expected.replicate_id:02d}"
            )
            result_path = directory / "result.json"
            result = json_object(result_path)
            evidence = FreshStateEvidence.from_mapping(result)
            evidence.validate_against(expected, pass_id)
            state_path = directory / str(result.get("state_payload"))
            raw = state_path.read_bytes()
            if (
                result.get("status") != "PASS_V3_D8A_FRESH_STATE_RECORD"
                or result.get("source_git_commit") != git_output("rev-parse", "HEAD")
                or result.get("source_worktree_dirty") is not False
                or result.get("D8_contract_sha256") != D8_CONTRACT_SHA256
                or result.get("D8_schedule_sha256") != D8_SCHEDULE_SHA256
                or result.get("model_checkpoint_loaded") is not False
                or result.get("policy_action_sampled") is not False
                or result.get("official_episode_identity_used") is not False
                or result.get("gpu_query_or_initialization") != 0
                or len(raw) != evidence.state_nbytes
                or hashlib.sha256(raw).hexdigest() != evidence.state_sha256
            ):
                raise PermissionError("V3-D8A state record or raw payload differs")
            value = np.frombuffer(raw, dtype="<f8")
            if value.shape != (evidence.state_dimension,) or not bool(
                np.isfinite(value).all()
            ):
                raise PermissionError("V3-D8A raw state geometry differs")
            pass_records[pass_id].append(evidence)
            if pass_id == 1:
                raw_states.append(raw)
                record_audit.append(
                    {
                        "task_id": expected.task_id,
                        "replicate_id": expected.replicate_id,
                        "cluster_key": expected.cluster_key,
                        "state_seed": expected.state_seed,
                        "policy_seed": expected.policy_seed,
                        "state_dimension": evidence.state_dimension,
                        "state_sha256": evidence.state_sha256,
                        "pass1_result_sha256": sha256(result_path),
                    }
                )
            else:
                record_audit[len(pass_records[2]) - 1]["pass2_result_sha256"] = sha256(
                    result_path
                )
    audit = validate_two_pass_evidence(
        schedule, pass_records[1], pass_records[2]
    )
    incomplete.mkdir(parents=True, exist_ok=False)
    payload_path = incomplete / "fresh_states.pt"
    torch.save(
        {
            "schema_version": D8A_PAYLOAD_SCHEMA_VERSION,
            "D8_contract_sha256": D8_CONTRACT_SHA256,
            "D8_schedule_sha256": D8_SCHEDULE_SHA256,
            "task_id": torch.tensor([item.task_id for item in schedule]),
            "replicate_id": torch.tensor([item.replicate_id for item in schedule]),
            "state_seed": torch.tensor([item.state_seed for item in schedule]),
            "policy_seed": torch.tensor([item.policy_seed for item in schedule]),
            "cluster_keys": [item.cluster_key for item in schedule],
            "state_sha256": [item.state_sha256 for item in pass_records[1]],
            "states": [
                torch.from_numpy(np.frombuffer(raw, dtype="<f8").copy())
                for raw in raw_states
            ],
            "determinism_passes": 2,
            "initial_task_success_all_false": True,
            "official_episode_identity_used": False,
            "policy_rollout_performed": False,
        },
        payload_path,
    )
    result = {
        "status": "PASS_V3_D8A_FRESH_STATES_FROZEN",
        "schema_version": D8A_RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "suite": "libero_10",
        "audit": audit,
        "records": record_audit,
        "payload": payload_path.name,
        "payload_sha256": sha256(payload_path),
        "input_sha256": {
            "D8_contract": D8_CONTRACT_SHA256,
            "D8_schedule": D8_SCHEDULE_SHA256,
            "generation_result": sha256(generation_result_path),
        },
        "access_ledger": {
            "fresh_states_generated": 200,
            "determinism_audit_regenerations": 200,
            "model_checkpoint_loaded": False,
            "policy_action_sampled": False,
            "fresh_policy_rollout": False,
            "calibration_or_test_payload_opened": False,
            "official_episode_40_49_opened": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "claim_boundary": {
            "generated_states_are_official_fixed_benchmark_states": False,
            "policy_or_task_success_evaluated": False,
            "fresh_confirmation_complete": False,
            "superiority_claim_authorized": False,
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
