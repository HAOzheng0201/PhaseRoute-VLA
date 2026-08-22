from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from a1.vla.dynamic_compute.v3 import paired_active_collection as d9c
from a1.vla.dynamic_compute.v3.independent_test_protocol import D9TestRecord


REPO_ROOT = Path(__file__).resolve().parents[3]


def _telemetry(
    *, episode_id: str, task_id: int, step_id: int, layer: int, phase: bool
) -> dict:
    events = []
    if phase:
        events.append(
            {
                "event": "phase_route_decision",
                "selected_layer": layer,
                "candidate_gates": [],
                "fallback": layer == 27,
            }
        )
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "step_id": step_id,
        "exit_layer": layer,
        "fm_calls": 3,
        "fm_steps_total": 30,
        "latency_ms": 12.5,
        "extra": {"exit_events": events},
    }


def _cache_record(*, episode_id: str, task_id: int, step_id: int, layer: int) -> dict:
    return {
        "schema_version": "phase-route-vla.vision-teacher-call.v3",
        "episode_id": episode_id,
        "task_id": task_id,
        "step_id": step_id,
        "teacher_kind": d9c.PHASE_ROUTE_TEACHER_KIND,
        "teacher_exit_layer": layer,
        "fm_calls": 2,
        "fm_trace_count": 3,
        "candidate_trace_count": 2,
        "comparison_trace_count": 1,
        "candidate_layers": [3, layer],
        "shapes": {
            "fm_trace_layers": [3],
            "fm_trace_roles": [3],
            "fm_trace_steps": [3],
            "fm_trace_input_x": [3, 8, 7],
            "fm_trace_output_action": [3, 8, 7],
        },
    }


def test_task_schedule_is_exact_and_alternates_arm_order() -> None:
    records = d9c.task_schedule(REPO_ROOT, 3)
    assert len(records) == 10
    assert [record.episode_index for record in records] == list(range(40, 50))
    assert [record.seed for record in records] == list(range(20_290_851, 20_290_861))
    assert {record.physical_gpu_index for record in records} == {3}
    assert records[0].arm_order == (
        d9c.PHASE_ROUTE_ARM,
        d9c.ORIGINAL_A1_ARM,
    )
    assert records[1].arm_order == (
        d9c.ORIGINAL_A1_ARM,
        d9c.PHASE_ROUTE_ARM,
    )


def test_output_path_is_fixed_below_d9c_root(tmp_path: Path) -> None:
    expected = d9c.expected_task_output(REPO_ROOT, 2)
    assert d9c.validate_task_output(REPO_ROOT, 2, expected) == expected
    with pytest.raises(d9c.D9CCollectionError, match="output differs"):
        d9c.validate_task_output(REPO_ROOT, 2, tmp_path / "task2")


def test_gpu_contract_requires_task_mod_four_and_one_visible_gpu() -> None:
    d9c.validate_gpu_contract(
        task_id=6,
        physical_gpu_index=2,
        visible_devices="2",
        visible_gpu_count=1,
        expected_gpu_uuid="GPU-abc",
        observed_gpu_uuid="abc",
    )
    with pytest.raises(d9c.D9CCollectionError, match="GPU assignment"):
        d9c.validate_gpu_contract(
            task_id=6,
            physical_gpu_index=6,
            visible_devices="6",
            visible_gpu_count=1,
            expected_gpu_uuid="GPU-abc",
            observed_gpu_uuid="abc",
        )
    with pytest.raises(d9c.D9CCollectionError, match="GPU assignment"):
        d9c.validate_gpu_contract(
            task_id=6,
            physical_gpu_index=2,
            visible_devices="2",
            visible_gpu_count=2,
            expected_gpu_uuid="GPU-abc",
            observed_gpu_uuid="abc",
        )


def test_state_hash_binds_dtype_shape_and_bytes() -> None:
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert d9c.sha256_array(value) == d9c.sha256_array(value.copy())
    assert d9c.sha256_array(value) != d9c.sha256_array(value.astype(np.float64))
    assert d9c.sha256_array(value) != d9c.sha256_array(value.reshape(4, 3))


def test_original_a1_telemetry_accepts_full_frozen_exit_schedule() -> None:
    episode = "libero_10:task0:episode40"
    summary = d9c.summarize_policy_telemetry(
        [
            _telemetry(
                episode_id=episode, task_id=0, step_id=10, layer=3, phase=False
            ),
            _telemetry(
                episode_id=episode, task_id=0, step_id=18, layer=27, phase=False
            ),
        ],
        arm=d9c.ORIGINAL_A1_ARM,
        expected_episode_id=episode,
        expected_task_id=0,
    )
    assert summary["policy_calls"] == 2
    assert summary["fm_calls"] == 6
    assert summary["exit_layer_counts"] == {"L3": 1, "L27": 1}
    assert summary["phase_route_decisions"] == 0


def test_phase_route_telemetry_requires_one_aligned_decision_per_call() -> None:
    episode = "libero_10:task1:episode41"
    records = [
        _telemetry(
            episode_id=episode, task_id=1, step_id=10, layer=11, phase=True
        ),
        _telemetry(
            episode_id=episode, task_id=1, step_id=18, layer=27, phase=True
        ),
    ]
    summary = d9c.summarize_policy_telemetry(
        records,
        arm=d9c.PHASE_ROUTE_ARM,
        expected_episode_id=episode,
        expected_task_id=1,
    )
    assert summary["phase_route_decisions"] == 2
    assert summary["exit_layer_counts"] == {"L11": 1, "L27": 1}
    records[0]["extra"]["exit_events"][0]["selected_layer"] = 13
    with pytest.raises(d9c.D9CCollectionError, match="decision mismatch"):
        d9c.summarize_policy_telemetry(
            records,
            arm=d9c.PHASE_ROUTE_ARM,
            expected_episode_id=episode,
            expected_task_id=1,
        )


def test_phase_route_telemetry_fails_on_identity_order_or_layer_drift() -> None:
    episode = "libero_10:task2:episode42"
    wrong_layer = _telemetry(
        episode_id=episode, task_id=2, step_id=10, layer=9, phase=True
    )
    wrong_layer["extra"]["exit_events"][0]["selected_layer"] = 9
    with pytest.raises(d9c.D9CCollectionError, match="non-frozen layer"):
        d9c.summarize_policy_telemetry(
            [wrong_layer],
            arm=d9c.PHASE_ROUTE_ARM,
            expected_episode_id=episode,
            expected_task_id=2,
        )
    records = [
        _telemetry(
            episode_id=episode, task_id=2, step_id=18, layer=27, phase=True
        ),
        _telemetry(
            episode_id=episode, task_id=2, step_id=10, layer=27, phase=True
        ),
    ]
    with pytest.raises(d9c.D9CCollectionError, match="not increasing"):
        d9c.summarize_policy_telemetry(
            records,
            arm=d9c.PHASE_ROUTE_ARM,
            expected_episode_id=episode,
            expected_task_id=2,
        )


def test_runtime_records_require_exact_call_ordinals_and_commits() -> None:
    episode = "libero_10:task4:episode44"
    records = (
        {
            "context": {
                "episode_id": episode,
                "task_id": 4,
                "call_ordinal": 0,
            },
            "prepared": True,
            "committed": True,
            "selected_layer": 11,
            "errors": [],
        },
        {
            "context": {
                "episode_id": episode,
                "task_id": 4,
                "call_ordinal": 1,
            },
            "prepared": True,
            "committed": True,
            "selected_layer": 27,
            "errors": [{"stage": "prepare", "error": "fail closed"}],
        },
    )
    summary = d9c.validate_phase_route_runtime_records(
        records,
        expected_episode_id=episode,
        expected_task_id=4,
        expected_policy_calls=2,
    )
    assert summary["records"] == 2
    assert summary["selected_layer_counts"] == {"L11": 1, "L13": 0, "L27": 1}
    assert summary["fail_closed_error_events"] == 1


def test_phase_route_cache_covers_every_policy_call() -> None:
    episode = "libero_10:task5:episode45"
    records = [
        _cache_record(episode_id=episode, task_id=5, step_id=10, layer=11),
        _cache_record(episode_id=episode, task_id=5, step_id=18, layer=27),
    ]
    summary = d9c.validate_phase_route_cache(
        records,
        expected_episode_id=episode,
        expected_task_id=5,
        expected_policy_calls=2,
    )
    assert summary == {
        "cache_records": 2,
        "early_exit_cache_records": 1,
        "all_policy_calls_cached": True,
    }
    records[0]["teacher_kind"] = "a1_early_exit"
    with pytest.raises(d9c.D9CCollectionError, match="manifest differs"):
        d9c.validate_phase_route_cache(
            records,
            expected_episode_id=episode,
            expected_task_id=5,
            expected_policy_calls=2,
        )


def test_file_inventory_is_deterministic_and_rejects_escape(tmp_path: Path) -> None:
    (tmp_path / "arrays").mkdir()
    payload = tmp_path / "arrays" / "call_000000.npz"
    payload.write_bytes(b"frozen-payload")
    inventory = d9c.build_file_inventory(tmp_path, ["arrays/call_000000.npz"])
    assert len(inventory) == 1
    assert inventory[0].bytes == len(b"frozen-payload")
    assert inventory[0].sha256 == hashlib.sha256(b"frozen-payload").hexdigest()
    with pytest.raises(d9c.D9CCollectionError, match="stay below"):
        d9c.build_file_inventory(tmp_path, ["../outside"])


def test_pair_record_binds_order_state_and_commit_without_aggregate() -> None:
    record = D9TestRecord(task_id=0, episode_index=40, seed=20_260_851)
    arm = {
        "status": "COMPLETE_V3_D9C_ARM_ROLLOUT",
        "initial_state_sha256": "a" * 64,
        "source_git_commit": "b" * 40,
    }
    value = {
        "status": "COMPLETE_V3_D9C_PAIRED_ACTIVE_PAIR",
        "schema_version": d9c.D9C_PAIR_SCHEMA_VERSION,
        "canonical_key": record.canonical_key,
        "task_id": record.task_id,
        "episode_index": record.episode_index,
        "seed": record.seed,
        "arm_order": list(record.arm_order),
        "arms": {
            d9c.ORIGINAL_A1_ARM: dict(arm),
            d9c.PHASE_ROUTE_ARM: dict(arm),
        },
    }
    d9c.validate_pair_record(value, record=record)
    value["arms"][d9c.PHASE_ROUTE_ARM]["initial_state_sha256"] = "c" * 64
    with pytest.raises(d9c.D9CCollectionError, match="state or commit"):
        d9c.validate_pair_record(value, record=record)


def test_d9b_protected_code_is_still_exact() -> None:
    audit = d9c.validate_d9b_readiness(REPO_ROOT)
    assert audit["sha256"] == d9c.D9B_READINESS_SHA256
    assert audit["bound_code_files"] == 14


def test_runner_source_keeps_test_schedule_and_aggregate_boundary_explicit() -> None:
    source = (
        REPO_ROOT / "scripts/dynamic_compute/v3/run_v3_d9c_task.py"
    ).read_text(encoding="utf-8")
    assert "episode_id_override=record.canonical_key" in source
    assert "for arm in record.arm_order" in source
    assert "physical_gpu_index != assigned" in (
        REPO_ROOT
        / "a1/vla/dynamic_compute/v3/paired_active_collection.py"
    ).read_text(encoding="utf-8")
    assert '"success": False' in source
    assert '"efficiency": False' in source
    assert '"D9_primary_gate_evaluated": False' in source
