from __future__ import annotations

import json
from pathlib import Path

import pytest

from a1.vla.dynamic_compute.stage11_compute_measurement import (
    STAGE11_COMPUTE_SCHEMA,
    summarize_stage11_compute_records,
)
from scripts.aggregate_route_first_stage11b_profile import (
    AGGREGATE_SCHEMA,
    PROFILE_SCHEMA,
    Stage11BAggregationError,
    aggregate,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = REPO_ROOT / "configs/research/route_first_stage11b_profile_protocol.json"
SHARDS = {
    "shard0": [0, 4, 8],
    "shard1": [1, 5, 9],
    "shard2": [2, 6],
    "shard3": [3, 7],
}


def _record(task_id: int, layer: int) -> dict[str, object]:
    decoder = 20.0 if layer == 13 else 50.0
    model = 10.0 + decoder + 15.0 + 5.0
    return {
        "schema_version": STAGE11_COMPUTE_SCHEMA,
        "measurement_is_control_input": False,
        "frozen_model_or_evaluator_source_modified": False,
        "context": {
            "episode_id": f"libero_10:task{task_id}:episode0",
            "task_id": task_id,
            "call_ordinal": 0,
        },
        "selected_layer": layer,
        "outer_policy_wall_ms": model + 2.0,
        "error": None,
        "spans": [],
        "decomposition": {
            "structure_valid": True,
            "cuda_events_complete": True,
            "component_sum_not_above_model_with_1ms_tolerance": True,
            "model_predict_cpu_ms": model,
            "host_and_wrapper_outside_model_cpu_ms": 2.0,
            "model_predict_cuda_ms": model,
            "vision_backbone_cuda_ms": 10.0,
            "decoder_blocks_cuda_sum_ms": decoder,
            "selected_action_fm_cuda_ms": 15.0,
            "model_other_cuda_ms": 5.0,
        },
    }


def _write_grid(tmp_path: Path) -> Path:
    input_root = tmp_path / "runs"
    protocol_sha = sha256_file(PROTOCOL)
    for shard_index, (shard_name, task_ids) in enumerate(SHARDS.items()):
        directory = input_root / f"full_{shard_name}"
        directory.mkdir(parents=True)
        records = [
            _record(task_id, 13 if task_id % 2 == 0 else 27)
            for task_id in task_ids
        ]
        compute_path = directory / "stage11_compute_measurement.jsonl"
        compute_path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in records),
            encoding="utf-8",
        )
        layer_counts = {
            "13": sum(row["selected_layer"] == 13 for row in records),
            "27": sum(row["selected_layer"] == 27 for row in records),
        }
        episodes = [
            {
                "task_id": task_id,
                "episode_index": 0,
                "seed": 91260830 + task_id * 10000,
                "success": task_id != 6,
                "policy_calls": 1,
                "selected_layer_counts": {
                    "L13": int(task_id % 2 == 0),
                    "L27": int(task_id % 2 == 1),
                },
                "wall_seconds": 1.0,
            }
            for task_id in task_ids
        ]
        calls = len(records)
        result = {
            "schema_version": PROFILE_SCHEMA,
            "status": "COMPLETE_STAGE11B_DEVELOPMENT_PROFILE",
            "profile_stage": shard_name,
            "task_ids": task_ids,
            "episode_indices": [0],
            "protocol_sha256": protocol_sha,
            "source": {
                "source_git_commit": "a" * 40,
                "source_worktree_dirty": False,
                "protected_code_sha256": {"protected.py": "b" * 64},
                "model_binding": {"sha256": "c" * 64, "bytes": 1},
                "libero_config_sha256": "d" * 64,
            },
            "gpu": {
                "physical_index": shard_index,
                "uuid": f"GPU-{shard_index}",
                "name": "test-gpu",
                "preflight_processes": [],
                "sampling_monitor": {
                    "samples": 2,
                    "clean": True,
                    "foreign_processes": [],
                    "query_errors": [],
                },
            },
            "episodes": episodes,
            "successes_descriptive": sum(row["success"] for row in episodes),
            "policy_calls": calls,
            "runtime": {
                "records": calls,
                "records_with_errors": 0,
                "prepared_calls": calls,
                "committed_calls": calls,
                "route_first_integrity": {
                    "records": calls,
                    "valid_calls_with_exactly_one_fm": calls,
                    "fm_invocations": calls,
                },
            },
            "stage11_compute": summarize_stage11_compute_records(records),
            "gates": {
                "runtime_complete": True,
                "exactly_one_authoritative_fm": True,
                "stage11_compute_complete": True,
                "gpu_sampling_monitor_clean": True,
            },
        }
        (directory / "result.json").write_text(
            json.dumps(result, allow_nan=False) + "\n", encoding="utf-8"
        )
    return input_root


def test_aggregate_recomputes_pooled_compute_metrics_and_usage(tmp_path: Path) -> None:
    result = aggregate(_write_grid(tmp_path), protocol_path=PROTOCOL)
    assert result["schema_version"] == AGGREGATE_SCHEMA
    assert result["status"] == "PASS"
    assert result["successes_descriptive"] == 9
    assert result["policy_calls"] == 10
    assert result["routing_usage"] == {
        "L13_calls": 5,
        "L27_calls": 5,
        "L13_fraction": 0.5,
        "executed_decoder_blocks": 210,
        "full_L27_decoder_blocks": 280,
        "decoder_block_reduction_fraction": 0.25,
    }
    assert result["compute"]["valid_records"] == 10
    assert result["gpu_sampling"]["total_samples"] == 8
    assert result["selected_layer_descriptive"][
        "decoder_cuda_p50_ratio_L13_to_L27"
    ] == pytest.approx(0.4)


def test_aggregate_rejects_dirty_source_binding(tmp_path: Path) -> None:
    input_root = _write_grid(tmp_path)
    path = input_root / "full_shard2/result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["source"]["source_worktree_dirty"] = True
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    with pytest.raises(Stage11BAggregationError, match="worktree was dirty"):
        aggregate(input_root, protocol_path=PROTOCOL)


def test_aggregate_rejects_compute_context_outside_shard(tmp_path: Path) -> None:
    input_root = _write_grid(tmp_path)
    path = input_root / "full_shard0/stage11_compute_measurement.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["context"]["task_id"] = 3
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(Stage11BAggregationError, match="escaped its shard"):
        aggregate(input_root, protocol_path=PROTOCOL)


def test_aggregate_requires_all_four_frozen_shards(tmp_path: Path) -> None:
    input_root = _write_grid(tmp_path)
    missing = input_root / "full_shard3/result.json"
    missing.unlink()
    with pytest.raises(Stage11BAggregationError, match="missing shard result"):
        aggregate(input_root, protocol_path=PROTOCOL)
