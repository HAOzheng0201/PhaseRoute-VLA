from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from a1.vla.dynamic_compute.v3 import d8_artifacts as da
from a1.vla.dynamic_compute.v3 import fresh_confirmation as fc


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_canonical_state_bytes_are_little_endian_float64() -> None:
    state, raw, digest = da.canonical_state_bytes(
        np.array([1.0, -2.0, 3.5], dtype=np.float64)
    )
    assert state.dtype == np.dtype("<f8")
    assert np.array_equal(np.frombuffer(raw, dtype="<f8"), state)
    assert digest == hashlib.sha256(raw).hexdigest()
    with pytest.raises(da.D8ArtifactError, match="float64"):
        da.canonical_state_bytes(np.array([1.0], dtype=np.float32))
    with pytest.raises(da.D8ArtifactError, match="finite"):
        da.canonical_state_bytes(np.array([float("nan")], dtype=np.float64))


def _evidence(
    record: fc.FreshConfirmationRecord, pass_id: int
) -> da.FreshStateEvidence:
    return da.FreshStateEvidence(
        pass_id=pass_id,
        task_id=record.task_id,
        replicate_id=record.replicate_id,
        cluster_key=record.cluster_key,
        state_seed=record.state_seed,
        policy_seed=record.policy_seed,
        state_dimension=100 + record.task_id,
        state_nbytes=8 * (100 + record.task_id),
        state_sha256=hashlib.sha256(record.cluster_key.encode()).hexdigest(),
        initial_task_success=False,
        explicit_reset_attempts=1,
    )


def test_two_pass_evidence_requires_exact_determinism_and_task_uniqueness() -> None:
    schedule = fc.load_fresh_confirmation_schedule(REPO_ROOT)
    first = [_evidence(record, 1) for record in schedule]
    second = [_evidence(record, 2) for record in schedule]
    audit = da.validate_two_pass_evidence(schedule, first, second)
    assert audit["records"] == 200
    assert audit["byte_identical_records"] == 200
    assert audit["unique_state_sha_per_task"] == {str(task): 20 for task in range(10)}

    drifted = list(second)
    value = drifted[17]
    drifted[17] = da.FreshStateEvidence(
        **{**value.__dict__, "state_sha256": "0" * 64}
    )
    with pytest.raises(da.D8ArtifactError, match="byte-identical"):
        da.validate_two_pass_evidence(schedule, first, drifted)


def test_state_record_mapping_rejects_solved_or_retry_semantics() -> None:
    record = fc.load_fresh_confirmation_schedule(REPO_ROOT)[0]
    evidence = _evidence(record, 1)
    evidence.validate_against(record, 1)
    solved = da.FreshStateEvidence(**{**evidence.__dict__, "initial_task_success": True})
    with pytest.raises(da.D8ArtifactError, match="semantics"):
        solved.validate_against(record, 1)
    retried = da.FreshStateEvidence(**{**evidence.__dict__, "explicit_reset_attempts": 2})
    with pytest.raises(da.D8ArtifactError, match="semantics"):
        retried.validate_against(record, 1)
