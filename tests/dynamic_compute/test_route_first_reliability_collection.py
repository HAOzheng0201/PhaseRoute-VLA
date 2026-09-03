from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from a1.vla.dynamic_compute import route_first_reliability_collection as collection
from a1.vla.dynamic_compute.route_first_reliability import build_stage11d_schedule


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "scripts/dynamic_compute/route_first_stage11d/collect_original_a1_task.py"
)
LAUNCHER = (
    REPO_ROOT
    / "scripts/dynamic_compute/route_first_stage11d/launch_original_a1_development.py"
)


class _Suite:
    n_tasks = 10


def _manifest_row(task_id: int, replicate_id: int) -> dict[str, object]:
    return {
        "schema_version": "phase-route-vla.vision-teacher-call.v3",
        "array_path": f"arrays/call_{replicate_id:02d}.npz",
        "episode_id": (
            f"libero_10:task{task_id}:route_first_reliability_v1:"
            f"replicate{replicate_id}"
        ),
        "step_id": 10,
        "task_id": task_id,
        "teacher_kind": "frozen_original_a1_observer",
        "checkpoint_sha256": collection.STAGE11D_A1_CHECKPOINT_SHA256,
        "teacher_exit_layer": 13,
        "fm_calls": 1,
        "fm_trace_count": 1,
        "candidate_trace_count": 1,
        "comparison_trace_count": 0,
        "candidate_layers": [13],
        "shapes": {
            "fm_trace_layers": [1],
            "fm_trace_roles": [1],
            "fm_trace_steps": [1],
            "fm_trace_input_x": [1, 8, 7],
            "fm_trace_output_action": [1, 8, 7],
        },
    }


def test_development_schedule_never_exposes_calibration_or_shadow() -> None:
    selected = collection.development_schedule(build_stage11d_schedule())
    assert len(selected) == 120
    assert {record.replicate_id for record in selected} == set(range(12))
    assert {record.split for record in selected} == {"development_train"}
    assert len({record.cluster_key for record in selected}) == 120
    for task_id in range(10):
        task = collection.task_development_schedule(selected, task_id)
        assert tuple(record.replicate_id for record in task) == tuple(range(12))


def test_development_task_suite_returns_copies_of_only_first_12_states() -> None:
    states = {
        task_id: tuple(
            np.array([task_id, replicate_id], dtype=np.float64)
            for replicate_id in range(12)
        )
        for task_id in range(10)
    }
    suite = collection.Stage11DDevelopmentTaskSuite(_Suite(), states)
    observed = suite.get_task_init_states(3)
    assert len(observed) == 12
    assert [int(state[1]) for state in observed] == list(range(12))
    observed[0][0] = -1
    assert suite.get_task_init_states(3)[0][0] == 3
    states[0] = (*states[0], np.array([0, 12], dtype=np.float64))
    with pytest.raises(collection.Stage11DCollectionError, match="exactly 12"):
        collection.Stage11DDevelopmentTaskSuite(_Suite(), states)


def test_development_identity_rejects_withheld_and_official_aliases() -> None:
    valid = "libero_10:task4:route_first_reliability_v1:replicate11"
    assert collection.parse_development_cluster_key(valid) == (4, 11)
    assert collection.validate_episode_id_override(
        valid, task_id=4, replicate_id=11
    ) == valid
    with pytest.raises(collection.Stage11DCollectionError, match="outside"):
        collection.parse_development_cluster_key(
            "libero_10:task4:route_first_reliability_v1:replicate12"
        )
    with pytest.raises(collection.Stage11DCollectionError, match="canonical"):
        collection.parse_development_cluster_key("libero_10:task4:episode12")


def test_gpu_contract_requires_one_exact_visible_device() -> None:
    collection.validate_gpu_contract(
        physical_gpu_index=7,
        visible_devices="7",
        visible_gpu_count=1,
        expected_gpu_uuid="GPU-abc",
        observed_gpu_uuid="abc",
    )
    with pytest.raises(PermissionError, match="one visible"):
        collection.validate_gpu_contract(
            physical_gpu_index=7,
            visible_devices="6,7",
            visible_gpu_count=2,
            expected_gpu_uuid="abc",
            observed_gpu_uuid="abc",
        )
    with pytest.raises(PermissionError, match="UUID"):
        collection.validate_gpu_contract(
            physical_gpu_index=7,
            visible_devices="7",
            visible_gpu_count=1,
            expected_gpu_uuid="abc",
            observed_gpu_uuid="def",
        )


def test_manifest_loader_requires_all_12_development_replicates(tmp_path: Path) -> None:
    manifest = tmp_path / "observation_calls/manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "".join(
            json.dumps(_manifest_row(2, replicate_id)) + "\n"
            for replicate_id in range(12)
        ),
        encoding="utf-8",
    )
    calls = collection.load_development_task_calls(tmp_path, task_id=2)
    assert len(calls) == 12
    assert [call.replicate_id for call in calls] == list(range(12))
    rows = [_manifest_row(2, replicate_id) for replicate_id in range(12)]
    rows[-1]["episode_id"] = (
        "libero_10:task2:route_first_reliability_v1:replicate12"
    )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(collection.Stage11DCollectionError, match="outside"):
        collection.load_development_task_calls(tmp_path, task_id=2)


def test_runner_is_original_a1_observation_only_and_has_three_gates() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "make_exit_controller" in source
    assert "vision_teacher_cache_writer=observer" in source
    assert "phase_depth_runtime=None" in source
    assert "phase_route_runtime=None" in source
    assert 'phase_route_v3_enabled=False' in source
    assert 'vision_aggregation_enabled=False' in source
    assert "--cpu-preflight-only" in source
    assert "--gpu-preflight-only" in source
    assert "--model-load-smoke" in source
    assert "load_development_states" in source
    assert "load_route_first_active_runtime" not in source
    assert "load_frozen_phase_route_runtime" not in source


def test_launcher_selects_idle_gpus_and_never_retries_workers() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "memory_used_mib" in source
    assert "utilization_percent" in source
    assert "minimum_free_memory_mib" in source
    assert '"CUDA_VISIBLE_DEVICES": str(gpu["index"])' in source
    assert '"MUJOCO_EGL_DEVICE_ID": str(gpu["index"])' in source
    assert 'environment.get("HF_HOME")' in source
    assert 'REPO_ROOT.parent / "hf_cache"' in source
    assert "/data3/haozheng" not in source
    assert '"--preflight-only"' in source
    assert "for batch_start in range(0, 10, len(gpus))" in source
    assert source.count("subprocess.Popen(") == 1
    assert "no retry allowed" in source
