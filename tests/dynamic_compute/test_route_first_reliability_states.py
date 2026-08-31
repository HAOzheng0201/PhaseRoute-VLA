from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from a1.vla.dynamic_compute import route_first_reliability as protocol
from a1.vla.dynamic_compute import route_first_reliability_artifacts as artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_ROOT = REPO_ROOT / "scripts/dynamic_compute/route_first_stage11d"


def _evidence(record: protocol.Stage11DRecord, pass_id: int):
    digest = hashlib.sha256(record.cluster_key.encode("utf-8")).hexdigest()
    return artifacts.Stage11DStateEvidence(
        pass_id=pass_id,
        task_id=record.task_id,
        replicate_id=record.replicate_id,
        split=record.split,
        cluster_key=record.cluster_key,
        state_seed=record.state_seed,
        policy_seed=record.policy_seed,
        state_dimension=3,
        state_nbytes=24,
        state_sha256=digest,
        initial_task_success=False,
        explicit_reset_attempts=1,
    )


def test_canonical_state_bytes_are_little_endian_float64() -> None:
    source = np.asarray([1.0, -2.0, 3.5], dtype=">f8")
    state, raw, digest = artifacts.canonical_state_bytes(source)
    assert state.dtype == np.dtype("<f8")
    assert state.flags.c_contiguous
    assert raw == np.asarray([1.0, -2.0, 3.5], dtype="<f8").tobytes()
    assert digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((1, 2), dtype=np.float64),
        np.zeros(2, dtype=np.float32),
        np.asarray([], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
    ],
)
def test_canonical_state_rejects_geometry_dtype_and_nonfinite(value) -> None:
    with pytest.raises(artifacts.Stage11DArtifactError):
        artifacts.canonical_state_bytes(value)


def test_state_evidence_mapping_and_schedule_binding() -> None:
    record = protocol.build_stage11d_schedule()[37]
    evidence = _evidence(record, 1)
    mapping = {
        "schema_version": artifacts.STAGE11D_STATE_RECORD_SCHEMA,
        **evidence.__dict__,
    }
    loaded = artifacts.Stage11DStateEvidence.from_mapping(mapping)
    loaded.validate(record, 1)
    with pytest.raises(artifacts.Stage11DArtifactError):
        replace(loaded, split="calibration").validate(record, 1)


def test_two_pass_audit_requires_byte_identity_and_task_local_uniqueness() -> None:
    schedule = protocol.build_stage11d_schedule()
    first = [_evidence(record, 1) for record in schedule]
    second = [_evidence(record, 2) for record in schedule]
    result = artifacts.validate_two_pass_states(schedule, first, second)
    assert result["records"] == 200
    assert result["byte_identical_records"] == 200
    assert result["unique_state_sha_per_task"] == {str(task): 20 for task in range(10)}
    broken = list(second)
    broken[10] = replace(broken[10], state_sha256="0" * 64)
    with pytest.raises(artifacts.Stage11DArtifactError):
        artifacts.validate_two_pass_states(schedule, first, broken)


def test_state_task_suite_returns_copies_only() -> None:
    class BaseSuite:
        marker = "base"

    states = {
        task: tuple(
            np.asarray([task, replicate], dtype=np.float64)
            for replicate in range(20)
        )
        for task in range(10)
    }
    suite = artifacts.Stage11DFreshStateTaskSuite(BaseSuite(), states)
    first = suite.get_task_init_states(3)
    first[0][0] = 999
    second = suite.get_task_init_states(3)
    assert second[0][0] == 3
    assert suite.marker == "base"


def test_state_payload_loader_authenticates_every_row(tmp_path: Path) -> None:
    schedule = protocol.build_stage11d_schedule()
    directory = tmp_path / artifacts.STAGE11D_STATES_RELATIVE_PATH
    directory.mkdir(parents=True)
    states = [
        torch.tensor(
            [record.task_id, record.replicate_id, record.state_seed],
            dtype=torch.float64,
        )
        for record in schedule
    ]
    hashes = [artifacts.canonical_state_bytes(state.numpy())[2] for state in states]
    payload = {
        "schema_version": artifacts.STAGE11D_STATE_PAYLOAD_SCHEMA,
        "protocol_sha256": protocol.STAGE11D_PROTOCOL_SHA256,
        "task_id": torch.tensor([record.task_id for record in schedule]),
        "replicate_id": torch.tensor([record.replicate_id for record in schedule]),
        "state_seed": torch.tensor([record.state_seed for record in schedule]),
        "policy_seed": torch.tensor([record.policy_seed for record in schedule]),
        "splits": [record.split for record in schedule],
        "cluster_keys": [record.cluster_key for record in schedule],
        "state_sha256": hashes,
        "states": states,
        "official_episode_identity_used": False,
        "policy_rollout_performed": False,
    }
    payload_path = directory / "fresh_states.pt"
    torch.save(payload, payload_path)
    attestation = {
        "schema_version": artifacts.STAGE11D_STATE_ATTESTATION_SCHEMA,
        "status": "PASS_ROUTE_FIRST_STAGE11D_STATES_FROZEN",
        "protocol_sha256": protocol.STAGE11D_PROTOCOL_SHA256,
        "payload": payload_path.name,
        "payload_sha256": artifacts.sha256_file(payload_path),
        "access_ledger": {"policy_action_sampled": False, "active_control": False},
    }
    (directory / "state_attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )
    loaded_schedule, by_task, loaded_attestation = artifacts.load_stage11d_states(
        tmp_path
    )
    assert loaded_schedule == schedule
    assert all(len(by_task[task]) == 20 for task in range(10))
    assert loaded_attestation["payload_sha256"] == artifacts.sha256_file(payload_path)
    payload["cluster_keys"][0] = "tampered"
    torch.save(payload, payload_path)
    attestation["payload_sha256"] = artifacts.sha256_file(payload_path)
    (directory / "state_attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8"
    )
    with pytest.raises(artifacts.Stage11DArtifactError):
        artifacts.load_stage11d_states(tmp_path)


def test_runner_readiness_binds_files_and_only_state_generation(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("print('frozen')\n", encoding="utf-8")
    readiness_path = tmp_path / artifacts.STAGE11D_RUNNER_READINESS_RELATIVE_PATH
    readiness_path.parent.mkdir(parents=True)
    readiness = {
        "schema_version": "phase-route-vla.route-first-stage11d-state-runner-readiness.v1",
        "status": "PASS_ROUTE_FIRST_STAGE11D_STATE_RUNNER_READINESS",
        "protocol_sha256": protocol.STAGE11D_PROTOCOL_SHA256,
        "runner_files": {"runner.py": artifacts.sha256_file(runner)},
        "authorization": {
            "state_generation": True,
            "original_A1_collection": False,
            "same_noise_replay": False,
            "active_control": False,
        },
    }
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    assert artifacts.validate_state_runner_readiness(tmp_path)["status"].startswith(
        "PASS_"
    )
    runner.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        artifacts.validate_state_runner_readiness(tmp_path)


def test_state_runners_have_preflight_and_do_not_open_official_states() -> None:
    worker = (RUNNER_ROOT / "generate_state_record.py").read_text(encoding="utf-8")
    orchestrator = (RUNNER_ROOT / "generate_states.py").read_text(encoding="utf-8")
    aggregate = (RUNNER_ROOT / "aggregate_states.py").read_text(encoding="utf-8")
    assert worker.count("environment.env.reset()") == 1
    assert "--preflight-only" in worker and "--preflight-only" in orchestrator
    assert "get_task_init_states" not in worker + orchestrator + aggregate
    assert "model.pt" not in worker + orchestrator + aggregate
    assert "CUDA_VISIBLE_DEVICES\"] = \"-1\"" in worker
    assert "CUDA_VISIBLE_DEVICES\"] = \"-1\"" in orchestrator
    assert "CUDA_VISIBLE_DEVICES\"] = \"-1\"" in aggregate
