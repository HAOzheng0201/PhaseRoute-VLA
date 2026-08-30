#!/usr/bin/env python3
"""Aggregate the four frozen Route-first Stage-11B profiling shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.stage11_compute_measurement import (  # noqa: E402
    STAGE11_COMPUTE_SCHEMA,
    summarize_stage11_compute_records,
)


PROTOCOL_SCHEMA = "phase-route-vla.route-first-stage11b-profile-protocol.v1"
PROFILE_SCHEMA = "phase-route-vla.route-first-stage11b-profile-result.v1"
AGGREGATE_SCHEMA = "phase-route-vla.route-first-stage11b-profile-aggregate.v1"
DEFAULT_INPUT_ROOT = REPO_ROOT / "runs/route_first_stage11b_profile"
DEFAULT_PROTOCOL = (
    REPO_ROOT / "configs/research/route_first_stage11b_profile_protocol.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/route_first/route_first_stage11b_profile_aggregate.json"
)


class Stage11BAggregationError(RuntimeError):
    """Raised when immutable Stage-11B evidence is missing or inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage11BAggregationError(f"JSON object required: {path}")
    return dict(value)


def _records(path: Path) -> list[dict[str, Any]]:
    output = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise Stage11BAggregationError(
                f"JSONL object required: {path}:{line_number}"
            )
        output.append(dict(value))
    if not output:
        raise Stage11BAggregationError(f"JSONL evidence is empty: {path}")
    return output


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11BAggregationError(message)


def _same_json(values: Sequence[Any], name: str) -> Any:
    _require(bool(values), f"{name} is empty")
    encoded = {
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        for value in values
    }
    _require(len(encoded) == 1, f"{name} differs across shards")
    return values[0]


def _ratio(numerator: float, denominator: float, name: str) -> float:
    _require(
        math.isfinite(numerator)
        and math.isfinite(denominator)
        and denominator > 0.0,
        f"cannot compute {name}",
    )
    return numerator / denominator


def _validate_result(
    result: Mapping[str, Any],
    *,
    shard_name: str,
    expected_tasks: Sequence[int],
    protocol_sha256: str,
) -> None:
    _require(result.get("schema_version") == PROFILE_SCHEMA, "profile schema differs")
    _require(
        result.get("status") == "COMPLETE_STAGE11B_DEVELOPMENT_PROFILE",
        f"{shard_name} is not complete",
    )
    _require(result.get("profile_stage") == shard_name, "profile stage differs")
    _require(result.get("task_ids") == list(expected_tasks), "task schedule differs")
    _require(result.get("episode_indices") == [0], "episode schedule differs")
    _require(
        result.get("protocol_sha256") == protocol_sha256,
        "profile protocol SHA-256 differs",
    )
    source = result.get("source")
    _require(isinstance(source, Mapping), "profile source binding is missing")
    _require(source.get("source_worktree_dirty") is False, "source worktree was dirty")
    _require(bool(source.get("source_git_commit")), "source commit is missing")
    _require(
        isinstance(source.get("protected_code_sha256"), Mapping),
        "protected source bindings are missing",
    )
    gates = result.get("gates")
    _require(
        isinstance(gates, Mapping)
        and gates
        and all(value is True for value in gates.values()),
        f"{shard_name} has a failed integrity gate",
    )
    gpu = result.get("gpu")
    monitor = gpu.get("sampling_monitor") if isinstance(gpu, Mapping) else None
    _require(isinstance(monitor, Mapping), "GPU sampling monitor is missing")
    _require(monitor.get("clean") is True, "GPU sampling monitor is not clean")
    _require(not monitor.get("foreign_processes"), "foreign GPU process was observed")
    _require(not monitor.get("query_errors"), "GPU sampling query failed")
    _require(not gpu.get("preflight_processes"), "GPU preflight was not clean")

    episodes = result.get("episodes")
    _require(isinstance(episodes, list), "episode records are missing")
    _require(len(episodes) == len(expected_tasks), "episode count differs")
    seed_base = 91260830
    for episode, task_id in zip(episodes, expected_tasks):
        _require(isinstance(episode, Mapping), "episode record is invalid")
        _require(episode.get("task_id") == task_id, "episode task order differs")
        _require(episode.get("episode_index") == 0, "episode index differs")
        _require(episode.get("seed") == seed_base + task_id * 10000, "seed differs")
        _require(
            isinstance(episode.get("success"), bool), "episode outcome is invalid"
        )
        _require(
            isinstance(episode.get("policy_calls"), int)
            and episode["policy_calls"] > 0,
            "episode policy-call count is invalid",
        )

    policy_calls = sum(int(episode["policy_calls"]) for episode in episodes)
    _require(result.get("policy_calls") == policy_calls, "policy-call total differs")
    _require(
        result.get("successes_descriptive")
        == sum(bool(episode["success"]) for episode in episodes),
        "descriptive success total differs",
    )
    runtime = result.get("runtime")
    integrity = (
        runtime.get("route_first_integrity") if isinstance(runtime, Mapping) else None
    )
    _require(isinstance(integrity, Mapping), "route-first integrity is missing")
    _require(runtime.get("records") == policy_calls, "runtime record count differs")
    _require(runtime.get("records_with_errors") == 0, "runtime errors were recorded")
    _require(
        runtime.get("prepared_calls") == policy_calls, "prepared-call count differs"
    )
    _require(
        runtime.get("committed_calls") == policy_calls, "committed-call count differs"
    )
    _require(integrity.get("records") == policy_calls, "integrity count differs")
    _require(
        integrity.get("valid_calls_with_exactly_one_fm") == policy_calls,
        "a policy call did not use exactly one authoritative FM",
    )
    _require(integrity.get("fm_invocations") == policy_calls, "FM count differs")
    compute = result.get("stage11_compute")
    _require(isinstance(compute, Mapping), "compute summary is missing")
    _require(compute.get("records") == policy_calls, "compute record count differs")
    _require(compute.get("valid_records") == policy_calls, "compute record is invalid")
    _require(compute.get("invalid_records") == 0, "invalid compute record exists")


def aggregate(
    input_root: Path,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    input_root = input_root.resolve(strict=True)
    protocol_path = protocol_path.resolve(strict=True)
    protocol = _object(protocol_path)
    _require(
        protocol.get("schema_version") == PROTOCOL_SCHEMA, "protocol schema differs"
    )
    protocol_sha256 = sha256_file(protocol_path)
    schedule = protocol.get("schedule", {}).get("full_shards", {})
    _require(
        isinstance(schedule, Mapping) and len(schedule) == 4,
        "shard schedule differs",
    )

    all_records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    input_manifest: dict[str, Any] = {}
    sources = []
    gpu_rows = []
    for shard_name in sorted(schedule):
        expected_tasks = schedule[shard_name]
        _require(isinstance(expected_tasks, list), "shard task list is invalid")
        directory = input_root / f"full_{shard_name}"
        result_path = directory / "result.json"
        compute_path = directory / "stage11_compute_measurement.jsonl"
        _require(result_path.is_file(), f"missing shard result: {result_path}")
        _require(compute_path.is_file(), f"missing compute evidence: {compute_path}")
        result = _object(result_path)
        _validate_result(
            result,
            shard_name=shard_name,
            expected_tasks=expected_tasks,
            protocol_sha256=protocol_sha256,
        )
        records = _records(compute_path)
        _require(
            len(records) == result["policy_calls"],
            f"{shard_name} raw compute count differs",
        )
        task_ordinals = {int(task_id): [] for task_id in expected_tasks}
        for record in records:
            _require(
                record.get("schema_version") == STAGE11_COMPUTE_SCHEMA,
                f"{shard_name} raw compute schema differs",
            )
            context = record.get("context")
            _require(isinstance(context, Mapping), "compute context is missing")
            task_id = context.get("task_id")
            _require(task_id in task_ordinals, "compute record escaped its shard")
            task_ordinals[int(task_id)].append(context.get("call_ordinal"))
        for episode in result["episodes"]:
            task_id = int(episode["task_id"])
            _require(
                task_ordinals[task_id] == list(range(int(episode["policy_calls"]))),
                f"task {task_id} compute call ordinals differ",
            )

        summary = summarize_stage11_compute_records(records)
        _require(summary == result["stage11_compute"], f"{shard_name} summary differs")
        all_records.extend(records)
        episodes.extend(dict(episode) for episode in result["episodes"])
        sources.append(result["source"])
        monitor = result["gpu"]["sampling_monitor"]
        gpu_rows.append(
            {
                "shard": shard_name,
                "physical_index": result["gpu"]["physical_index"],
                "uuid": result["gpu"]["uuid"],
                "name": result["gpu"]["name"],
                "samples": monitor["samples"],
                "clean": monitor["clean"],
            }
        )
        input_manifest[shard_name] = {
            "result": {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
                "bytes": result_path.stat().st_size,
            },
            "stage11_compute_measurement": {
                "path": str(compute_path),
                "sha256": sha256_file(compute_path),
                "bytes": compute_path.stat().st_size,
            },
        }

    _require(
        sorted(int(episode["task_id"]) for episode in episodes) == list(range(10)),
        "full task grid is not exactly 0..9",
    )
    _require(len(episodes) == 10, "full episode grid differs")
    source_git_commit = _same_json(
        [source["source_git_commit"] for source in sources], "source commit"
    )
    protected_code = _same_json(
        [source["protected_code_sha256"] for source in sources], "protected code"
    )
    model_binding = _same_json(
        [source["model_binding"] for source in sources], "model binding"
    )
    libero_config_sha256 = _same_json(
        [source["libero_config_sha256"] for source in sources], "LIBERO config"
    )

    compute = summarize_stage11_compute_records(all_records)
    policy_calls = len(all_records)
    l13_calls = int(compute["selected_layer_counts"]["13"])
    l27_calls = int(compute["selected_layer_counts"]["27"])
    _require(l13_calls + l27_calls == policy_calls, "selected-layer total differs")
    executed_decoder_blocks = 14 * l13_calls + 28 * l27_calls
    full_decoder_blocks = 28 * policy_calls
    metric = compute["latency_ms"]
    by_layer = compute["by_selected_layer"]
    model_cuda_sum = float(metric["model_predict_cuda_ms"]["sum"])
    component_cuda_sums = {
        name: float(metric[name]["sum"])
        for name in (
            "vision_backbone_cuda_ms",
            "decoder_blocks_cuda_sum_ms",
            "selected_action_fm_cuda_ms",
            "model_other_cuda_ms",
        )
    }
    checks = {
        "complete_frozen_shard_grid": True,
        "single_clean_source_commit": True,
        "protected_sources_identical": True,
        "all_gpu_sampling_monitors_clean": True,
        "all_policy_calls_exactly_one_authoritative_fm": True,
        "all_compute_records_valid": compute["valid_records"] == policy_calls,
        "full_task_grid_0_to_9_state_0": True,
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS" if all(checks.values()) else "INVALID",
        "scope": "stage11b_previously_opened_development_state_timing_diagnosis",
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "source": {
            "profile_source_git_commit": source_git_commit,
            "protected_code_sha256": protected_code,
            "model_binding": model_binding,
            "libero_config_sha256": libero_config_sha256,
        },
        "task_ids": list(range(10)),
        "episode_indices": [0],
        "episodes": 10,
        "successes_descriptive": sum(bool(row["success"]) for row in episodes),
        "policy_calls": policy_calls,
        "routing_usage": {
            "L13_calls": l13_calls,
            "L27_calls": l27_calls,
            "L13_fraction": l13_calls / policy_calls,
            "executed_decoder_blocks": executed_decoder_blocks,
            "full_L27_decoder_blocks": full_decoder_blocks,
            "decoder_block_reduction_fraction": 1.0
            - executed_decoder_blocks / full_decoder_blocks,
        },
        "compute": compute,
        "component_cuda_time_fraction_of_model_sum": {
            name: value / model_cuda_sum for name, value in component_cuda_sums.items()
        },
        "selected_layer_descriptive": {
            "model_predict_cuda_p50_ratio_L13_to_L27": _ratio(
                float(by_layer["13"]["latency_ms"]["model_predict_cuda_ms"]["p50"]),
                float(by_layer["27"]["latency_ms"]["model_predict_cuda_ms"]["p50"]),
                "model CUDA p50 ratio",
            ),
            "decoder_cuda_p50_ratio_L13_to_L27": _ratio(
                float(
                    by_layer["13"]["latency_ms"]["decoder_blocks_cuda_sum_ms"][
                        "p50"
                    ]
                ),
                float(
                    by_layer["27"]["latency_ms"]["decoder_blocks_cuda_sum_ms"][
                        "p50"
                    ]
                ),
                "decoder CUDA p50 ratio",
            ),
        },
        "task_rows": sorted(episodes, key=lambda row: int(row["task_id"])),
        "gpu_sampling": {
            "total_samples": sum(int(row["samples"]) for row in gpu_rows),
            "shards": gpu_rows,
        },
        "inputs": input_manifest,
        "checks": checks,
        "claim_boundary": {
            "timing_diagnosis_only": True,
            "profiling_overhead_included": True,
            "success_is_descriptive_only": True,
            "selected_layer_groups_are_not_randomized": True,
            "not_a_control_comparison": True,
            "not_a_speedup_confirmation": True,
            "not_a_threshold_selection_experiment": True,
            "not_a_stage10_gate_reinterpretation": True,
            "not_deployment_authorized": True,
        },
    }


def main() -> None:
    args = parse_args()
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip():
        raise PermissionError("Stage 11B aggregation requires a clean worktree")
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".incomplete")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or temporary.exists() or sidecar.exists():
        raise FileExistsError("Stage 11B aggregate refuses to overwrite evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = aggregate(args.input_root, protocol_path=args.protocol)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
