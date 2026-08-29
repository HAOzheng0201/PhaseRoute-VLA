from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil

import numpy as np
import pytest

from a1.vla.dynamic_compute.route_first_stage10 import (
    ARM_ORDERS,
    FreshStateEvidence,
    METHODS,
    PROTOCOL_RELATIVE_PATH,
    REPLICATE_IDS,
    SCHEDULE_RELATIVE_PATH,
    STATE_BINDING_RELATIVE_PATH,
    STATE_RESULT_RELATIVE_PATH,
    STAGE9_RESULT_RELATIVE_PATH,
    TASK_IDS,
    TRIPLET_COUNT,
    Stage10ContractError,
    canonical_state_bytes,
    load_protocol,
    load_schedule,
    load_state_binding,
    validate_two_pass_states,
    validate_generation_record_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _evidence(spec, pass_id: int, digest: str | None = None) -> FreshStateEvidence:
    state_digest = digest or f"{spec.task_id * 10 + spec.replicate_id + 1:064x}"
    return FreshStateEvidence(
        pass_id=pass_id,
        task_id=spec.task_id,
        replicate_id=spec.replicate_id,
        cluster_key=spec.cluster_key,
        state_seed=spec.state_seed,
        policy_seed=spec.policy_seed,
        state_dimension=128 + spec.task_id,
        state_nbytes=8 * (128 + spec.task_id),
        state_sha256=state_digest,
        initial_task_success=False,
        explicit_reset_attempts=1,
    )


def test_stage10_protocol_and_schedule_are_exact() -> None:
    protocol = load_protocol(REPO_ROOT)
    schedule = load_schedule(REPO_ROOT)
    assert protocol["confirmation_gate"]["required_triplets"] == 60
    assert len(schedule) == TRIPLET_COUNT
    assert {item.task_id for item in schedule} == set(TASK_IDS)
    assert {item.replicate_id for item in schedule} == set(REPLICATE_IDS)
    assert len({item.cluster_key for item in schedule}) == TRIPLET_COUNT
    assert all(item.arm_order == ARM_ORDERS[item.replicate_id] for item in schedule)


def test_stage10_six_orders_are_fully_counterbalanced() -> None:
    assert len(set(ARM_ORDERS)) == 6
    assert all(set(order) == set(METHODS) for order in ARM_ORDERS)
    for position in range(3):
        counts = Counter(order[position] for order in ARM_ORDERS)
        assert counts == {method: 2 for method in METHODS}


def test_stage10_schedule_hash_drift_fails_closed(tmp_path: Path) -> None:
    for relative in (
        PROTOCOL_RELATIVE_PATH,
        SCHEDULE_RELATIVE_PATH,
        STAGE9_RESULT_RELATIVE_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    schedule_path = tmp_path / SCHEDULE_RELATIVE_PATH
    schedule_path.write_text(
        schedule_path.read_text(encoding="utf-8").replace(
            '"seed_base": 71260829', '"seed_base": 71260830'
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="schedule SHA-256"):
        load_protocol(tmp_path)


def test_stage10_two_pass_state_audit_passes_complete_grid() -> None:
    schedule = load_schedule(REPO_ROOT)
    first = tuple(_evidence(spec, 1) for spec in schedule)
    second = tuple(_evidence(spec, 2) for spec in schedule)
    result = validate_two_pass_states(schedule, first, second)
    assert result["records"] == 60
    assert result["byte_identical_records"] == 60
    assert set(result["unique_state_sha_per_task"].values()) == {6}


def test_stage10_two_pass_state_audit_rejects_drift_and_duplicates() -> None:
    schedule = load_schedule(REPO_ROOT)
    first = tuple(_evidence(spec, 1) for spec in schedule)
    second = list(_evidence(spec, 2) for spec in schedule)
    second[0] = _evidence(schedule[0], 2, "f" * 64)
    with pytest.raises(ValueError, match="nondeterministic"):
        validate_two_pass_states(schedule, first, tuple(second))

    duplicate_first = list(first)
    duplicate_second = list(_evidence(spec, 2) for spec in schedule)
    duplicate_first[1] = _evidence(schedule[1], 1, first[0].state_sha256)
    duplicate_second[1] = _evidence(schedule[1], 2, first[0].state_sha256)
    with pytest.raises(ValueError, match="not task-local unique"):
        validate_two_pass_states(
            schedule, tuple(duplicate_first), tuple(duplicate_second)
        )


def test_stage10_generation_record_manifest_requires_exact_grid() -> None:
    schedule = load_schedule(REPO_ROOT)
    manifest = {
        f"pass{pass_id}:task{spec.task_id}:replicate{spec.replicate_id}": "a" * 64
        for pass_id in (1, 2)
        for spec in schedule
    }
    assert len(validate_generation_record_manifest(schedule, manifest)) == 120

    missing = dict(manifest)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="coverage differs"):
        validate_generation_record_manifest(schedule, missing)

    malformed = dict(manifest)
    malformed[next(iter(malformed))] = "not-a-sha256"
    with pytest.raises(ValueError, match="digest differs"):
        validate_generation_record_manifest(schedule, malformed)


def test_stage10_state_bytes_are_canonical_and_fail_closed() -> None:
    state, raw, digest = canonical_state_bytes(
        np.array([1.0, -2.5, 3.25], dtype=np.float64)
    )
    assert state.dtype == np.dtype("<f8")
    assert raw == state.tobytes(order="C")
    assert len(digest) == 64

    with pytest.raises(Stage10ContractError, match="finite nonempty float64"):
        canonical_state_bytes(np.array([1.0], dtype=np.float32))
    with pytest.raises(Stage10ContractError, match="finite nonempty float64"):
        canonical_state_bytes(np.array([float("nan")], dtype=np.float64))


def test_stage10_tracked_state_binding_is_exact() -> None:
    binding = load_state_binding(REPO_ROOT)
    assert binding["local_state_payload"]["records"] == 60
    assert binding["local_state_payload"]["bytes"] == 60714
    assert binding["authorization"]["active_rollout_started"] is False


def test_stage10_state_binding_hash_drift_fails_closed(tmp_path: Path) -> None:
    for relative in (
        PROTOCOL_RELATIVE_PATH,
        SCHEDULE_RELATIVE_PATH,
        STAGE9_RESULT_RELATIVE_PATH,
        STATE_BINDING_RELATIVE_PATH,
        STATE_RESULT_RELATIVE_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    binding_path = tmp_path / STATE_BINDING_RELATIVE_PATH
    binding_path.write_text(
        binding_path.read_text(encoding="utf-8").replace(
            '"bytes": 60714', '"bytes": 60715'
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="state binding SHA-256"):
        load_state_binding(tmp_path)
