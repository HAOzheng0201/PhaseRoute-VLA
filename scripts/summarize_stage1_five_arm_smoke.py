#!/usr/bin/env python3
"""Build an auditable five-arm Stage-1 engineering-smoke summary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "phase-route-vla.stage1.five-arm-paired-smoke.v1"
TELEMETRY_SCHEMA = "phase-route-vla.telemetry.v1"
MEASUREMENT_SCHEMA = "phase-route-vla.stage1.measurement.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-l11-dir", type=Path, required=True)
    parser.add_argument("--fixed-l13-dir", type=Path, required=True)
    parser.add_argument("--fixed-l27-dir", type=Path, required=True)
    parser.add_argument("--original-a1-dir", type=Path, required=True)
    parser.add_argument("--phase-route-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    finite = sorted(float(value) for value in values)
    if not finite or not all(math.isfinite(value) and value >= 0 for value in finite):
        raise ValueError("latencies must be finite and non-negative")

    def nearest_rank(q: float) -> float:
        return finite[max(0, math.ceil(q * len(finite)) - 1)]

    return {
        "count": len(finite),
        "sum": math.fsum(finite),
        "mean": statistics.fmean(finite),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": finite[-1],
    }


def _command_assignments(path: Path) -> dict[str, str]:
    tokens = shlex.split(path.read_text(encoding="utf-8"))
    assignments = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.isidentifier():
            assignments[key] = value
    return assignments


def summarize_telemetry(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if any(row.get("schema_version") != TELEMETRY_SCHEMA for row in rows):
        raise ValueError("unexpected policy telemetry schema")
    if any(row.get("action_shape") != [1, 8, 7] for row in rows):
        raise ValueError("policy telemetry action shape is not [1,8,7]")
    layers = [int(row["exit_layer"]) for row in rows]
    fm_calls = [int(row["fm_calls"]) for row in rows]
    if any(value <= 0 for value in fm_calls):
        raise ValueError("FM call count must be positive")
    layer_counts = Counter(layers)
    calls = len(rows)
    return {
        "policy_calls": calls,
        "selected_layers": {
            str(layer): layer_counts[layer] for layer in sorted(layer_counts)
        },
        "selected_layer_index_mean": statistics.fmean(layers),
        "executed_depth_ratio_to_l27": statistics.fmean(
            layer + 1 for layer in layers
        )
        / 28.0,
        "early_exit_calls": sum(layer < 27 for layer in layers),
        "early_exit_fraction": sum(layer < 27 for layer in layers) / calls,
        "fm_calls": sum(fm_calls),
        "fm_calls_per_policy_call": sum(fm_calls) / calls,
        "model_inference_latency_ms": latency_summary(
            float(row["latency_ms"]) for row in rows
        ),
        "instruction_hashes": sorted({str(row["instruction_hash"]) for row in rows}),
    }


def summarize_measurements(
    rows: Sequence[Mapping[str, Any]], expected_mode: str
) -> dict[str, Any]:
    if any(row.get("schema_version") != MEASUREMENT_SCHEMA for row in rows):
        raise ValueError("unexpected Stage-1 measurement schema")
    if any(row.get("mode") != expected_mode for row in rows):
        raise ValueError(f"measurement mode is not {expected_mode}")
    if any(row.get("error") is not None for row in rows):
        raise ValueError("measurement contains policy errors")
    if any(row.get("action_finite") is not True for row in rows):
        raise ValueError("measurement contains a non-finite or unaudited action")
    if any(row.get("action_shape") != [8, 7] for row in rows):
        raise ValueError("measurement action shape is not [8,7]")
    if any(row.get("measurement_is_control_input") is not False for row in rows):
        raise ValueError("measurement was marked as a control input")
    if any(row.get("d9_protected_source_modified") is not False for row in rows):
        raise ValueError("measurement reports modified D9 protected source")
    cuda_values = [row.get("policy_cuda_event_latency_ms") for row in rows]
    if any(value is None for value in cuda_values):
        raise ValueError("GPU run is missing CUDA event timing")
    return {
        "records": len(rows),
        "finite_action_records": len(rows),
        "selected_layers": dict(
            sorted(Counter(int(row["selected_layer"]) for row in rows).items())
        ),
        "policy_wall_latency_ms": latency_summary(
            float(row["policy_wall_latency_ms"]) for row in rows
        ),
        "policy_cuda_event_latency_ms": latency_summary(
            float(value) for value in cuda_values
        ),
    }


def _identity(task_id: int, episode_index: int, seed: int) -> dict[str, int]:
    return {"task_id": task_id, "episode_index": episode_index, "seed": seed}


def load_fixed_arm(
    root: Path, layer: int, *, task_id: int, episode_index: int, seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    evaluation_path = root / "evaluation_summary.json"
    telemetry_path = root / "policy_telemetry.jsonl"
    measurement_path = root / "stage1_measurement.jsonl"
    command_path = root / "command.sh"
    evaluation = load_json(evaluation_path)
    telemetry_rows = load_jsonl(telemetry_path)
    measurement_rows = load_jsonl(measurement_path)
    expected_method = f"fixed_l{layer}"
    if evaluation.get("method") != expected_method:
        raise ValueError(f"fixed arm method is not {expected_method}")
    if evaluation.get("task_ids") != [task_id] or evaluation.get("episode_indices") != [episode_index]:
        raise ValueError("fixed arm task/state identity differs")
    episodes = evaluation.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("fixed arm must contain exactly one episode")
    episode = episodes[0]
    if _identity(int(episode["task_id"]), int(episode["episode_index"]), int(episode["seed"])) != _identity(task_id, episode_index, seed):
        raise ValueError("fixed arm episode identity differs")
    telemetry = summarize_telemetry(telemetry_rows)
    measurements = summarize_measurements(measurement_rows, expected_method)
    calls = int(evaluation["policy_calls"])
    if calls != telemetry["policy_calls"] or calls != measurements["records"]:
        raise ValueError("fixed arm policy-call counts differ")
    if telemetry["selected_layers"] != {str(layer): calls}:
        raise ValueError("fixed arm selected a non-fixed layer")
    if measurements["selected_layers"] != {layer: calls}:
        raise ValueError("fixed arm measurement selected a non-fixed layer")
    if telemetry["fm_calls"] != calls:
        raise ValueError("fixed arm did not use exactly one FM call per policy call")
    command = _command_assignments(command_path)
    return (
        {
            "success": bool(episode["success"]),
            **telemetry,
            "policy_wall_latency_ms": measurements["policy_wall_latency_ms"],
            "policy_cuda_event_latency_ms": measurements[
                "policy_cuda_event_latency_ms"
            ],
            "finite_action_records": measurements["finite_action_records"],
            "episode_wall_seconds": float(episode["wall_seconds"]),
        },
        {
            "GPU_UUID": command.get("GPU_UUID", ""),
            "CHECKPOINT": str(Path(command.get("CHECKPOINT", "")).resolve()),
            "evaluation_summary.json": sha256_file(evaluation_path),
            "policy_telemetry.jsonl": sha256_file(telemetry_path),
            "stage1_measurement.jsonl": sha256_file(measurement_path),
        },
    )


def load_original_arm(
    root: Path, *, task_id: int, episode_index: int, seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    result_path = root / "result.json"
    telemetry_path = root / "policy_calls.jsonl"
    result = load_json(result_path)
    rows = load_jsonl(telemetry_path)
    if result.get("status") != "PASS" or result.get("policy") != "early_exit":
        raise ValueError("original A1 collector did not pass")
    if result.get("task_ids") != [task_id] or result.get("episode_indices") != [episode_index]:
        raise ValueError("original A1 task/state identity differs")
    episodes = result.get("episode_records")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("original A1 must contain exactly one episode")
    episode = episodes[0]
    if _identity(int(episode["task_id"]), int(episode["episode_idx"]), int(episode["episode_seed"])) != _identity(task_id, episode_index, seed):
        raise ValueError("original A1 episode identity differs")
    telemetry = summarize_telemetry(rows)
    calls = int(episode["policy_calls"])
    if calls != telemetry["policy_calls"] or calls != len(episode["latency_ms_by_call"]):
        raise ValueError("original A1 policy-call counts differ")
    if telemetry["fm_calls"] != int(episode["fm_calls_total"]):
        raise ValueError("original A1 FM-call totals differ")
    action_hashes = episode.get("action_chunk_sha256", [])
    if len(action_hashes) != calls or len(set(action_hashes)) == 0:
        raise ValueError("original A1 action hashes are incomplete")
    return (
        {
            "success": bool(episode["success"]),
            **telemetry,
            "policy_wall_latency_ms": latency_summary(
                float(value) for value in episode["latency_ms_by_call"]
            ),
            "policy_cuda_event_latency_ms": None,
            "finite_action_records": None,
            "hashed_action_records": len(action_hashes),
            "episode_wall_seconds": float(episode["wall_seconds"]),
        },
        {
            "GPU_UUID": str(result["physical_gpu_uuid_nvidia_smi"]),
            "CHECKPOINT": str(Path(result["checkpoint"]).resolve().parent),
            "CHECKPOINT_SHA256": str(result["checkpoint_sha256"]),
            "result.json": sha256_file(result_path),
            "policy_calls.jsonl": sha256_file(telemetry_path),
        },
    )


def load_phase_arm(
    root: Path, *, task_id: int, episode_index: int, seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    evaluation_path = root / "evaluation_summary.json"
    telemetry_path = root / "policy_telemetry.jsonl"
    measurement_path = root / "stage1_measurement.jsonl"
    preflight_path = root / "preflight.json"
    evaluation = load_json(evaluation_path)
    preflight = load_json(preflight_path)
    telemetry_rows = load_jsonl(telemetry_path)
    measurement_rows = load_jsonl(measurement_path)
    if evaluation.get("method") != "phase_route_v3" or preflight.get("status") != "PASS":
        raise ValueError("PhaseRoute evaluation or preflight did not pass")
    if evaluation.get("task_ids") != [task_id] or evaluation.get("episode_indices") != [episode_index]:
        raise ValueError("PhaseRoute task/state identity differs")
    episodes = evaluation.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("PhaseRoute must contain exactly one episode")
    episode = episodes[0]
    if _identity(int(episode["task_id"]), int(episode["episode_index"]), int(episode["seed"])) != _identity(task_id, episode_index, seed):
        raise ValueError("PhaseRoute episode identity differs")
    telemetry = summarize_telemetry(telemetry_rows)
    measurements = summarize_measurements(measurement_rows, "phase_route_v3")
    calls = int(episode["policy_calls"])
    runtime = evaluation.get("runtime", {})
    if calls != telemetry["policy_calls"] or calls != measurements["records"] or calls != int(runtime.get("records", -1)):
        raise ValueError("PhaseRoute policy-call counts differ")
    summary_layers = {
        str(layer): int(count)
        for layer, count in episode["selected_layers"].items()
        if int(count) != 0
    }
    if telemetry["selected_layers"] != summary_layers:
        raise ValueError("PhaseRoute selected-layer counts differ")
    if measurements["selected_layers"] != {
        int(layer): count for layer, count in summary_layers.items()
    }:
        raise ValueError("PhaseRoute measurement selected-layer counts differ")
    backbone = preflight.get("release", {}).get("backbone", {}).get("files", {}).get("model.pt", {})
    return (
        {
            "success": bool(episode["success"]),
            **telemetry,
            "policy_wall_latency_ms": measurements["policy_wall_latency_ms"],
            "policy_cuda_event_latency_ms": measurements[
                "policy_cuda_event_latency_ms"
            ],
            "finite_action_records": measurements["finite_action_records"],
            "episode_wall_seconds": float(episode["wall_seconds"]),
        },
        {
            "GPU_UUID": str(preflight["expected_gpu_uuid"]),
            "CHECKPOINT": str(Path(backbone["path"]).resolve().parent),
            "CHECKPOINT_SHA256": str(backbone["sha256"]),
            "evaluation_summary.json": sha256_file(evaluation_path),
            "policy_telemetry.jsonl": sha256_file(telemetry_path),
            "stage1_measurement.jsonl": sha256_file(measurement_path),
            "preflight.json": sha256_file(preflight_path),
        },
    )


def reduction(new_value: float, reference_value: float) -> float:
    if reference_value <= 0:
        raise ValueError("comparison reference must be positive")
    return 1.0 - new_value / reference_value


def build_summary(
    methods: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, str]],
    *,
    task_id: int,
    episode_index: int,
    seed: int,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    required = {"fixed_l11", "fixed_l13", "fixed_l27", "original_a1", "phase_route_v3"}
    if set(methods) != required or set(bindings) != required:
        raise ValueError("five-arm summary requires exactly five named methods")
    gpu_uuids = {str(value["GPU_UUID"]).removeprefix("GPU-").lower() for value in bindings.values()}
    checkpoint_paths = {str(Path(value["CHECKPOINT"]).resolve()) for value in bindings.values()}
    bound_hashes = {
        value.get("CHECKPOINT_SHA256") for value in bindings.values() if value.get("CHECKPOINT_SHA256")
    }
    instruction_hashes = {
        value
        for method in methods.values()
        for value in method["instruction_hashes"]
    }
    checks = {
        "all_five_methods_present": set(methods) == required,
        "same_physical_gpu_uuid": len(gpu_uuids) == 1,
        "same_checkpoint_path": len(checkpoint_paths) == 1,
        "checkpoint_sha256_matches": bound_hashes == {checkpoint_sha256},
        "same_instruction_hash": len(instruction_hashes) == 1,
        "all_methods_successful": all(bool(method["success"]) for method in methods.values()),
        "all_policy_call_counts_positive": all(int(method["policy_calls"]) > 0 for method in methods.values()),
    }
    original = methods["original_a1"]
    phase = methods["phase_route_v3"]
    full = methods["fixed_l27"]
    comparisons = {
        "phase_vs_original_a1": {
            "success_difference": float(phase["success"]) - float(original["success"]),
            "fm_calls_per_policy_call_reduction_fraction": reduction(
                float(phase["fm_calls_per_policy_call"]),
                float(original["fm_calls_per_policy_call"]),
            ),
            "policy_wall_mean_reduction_fraction": reduction(
                float(phase["policy_wall_latency_ms"]["mean"]),
                float(original["policy_wall_latency_ms"]["mean"]),
            ),
            "episode_wall_reduction_fraction": reduction(
                float(phase["episode_wall_seconds"]),
                float(original["episode_wall_seconds"]),
            ),
        },
        "phase_vs_single_head_fixed_l27": {
            "fm_calls_per_policy_call_reduction_fraction": reduction(
                float(phase["fm_calls_per_policy_call"]),
                float(full["fm_calls_per_policy_call"]),
            ),
            "policy_wall_mean_reduction_fraction": reduction(
                float(phase["policy_wall_latency_ms"]["mean"]),
                float(full["policy_wall_latency_ms"]["mean"]),
            ),
            "episode_wall_reduction_fraction": reduction(
                float(phase["episode_wall_seconds"]),
                float(full["episode_wall_seconds"]),
            ),
        },
    }
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "ordinary_engineering_smoke_not_D9_retest",
        "identity": {
            "suite": "libero_10",
            "task_id": task_id,
            "episode_index": episode_index,
            "seed": seed,
            "checkpoint_sha256": checkpoint_sha256,
            "physical_gpu_uuid": next(iter(gpu_uuids)) if len(gpu_uuids) == 1 else None,
        },
        "methods": dict(methods),
        "descriptive_comparisons": comparisons,
        "checks": checks,
        "input_artifacts": dict(bindings),
        "claim_boundary": {
            "new_independent_test": False,
            "D9_retest": False,
            "formal_speedup_claim": False,
            "success_rate_improvement_established": False,
            "single_episode_descriptive_only": True,
            "fixed_arms_are_single_FM_head_depth_bounds": True,
            "deployment_authorized": False,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint SHA-256 must contain 64 characters")
    int(args.checkpoint_sha256, 16)
    methods: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for name, root, layer in (
        ("fixed_l11", args.fixed_l11_dir, 11),
        ("fixed_l13", args.fixed_l13_dir, 13),
        ("fixed_l27", args.fixed_l27_dir, 27),
    ):
        methods[name], bindings[name] = load_fixed_arm(
            root.resolve(),
            layer,
            task_id=args.task_id,
            episode_index=args.episode_index,
            seed=args.seed,
        )
    methods["original_a1"], bindings["original_a1"] = load_original_arm(
        args.original_a1_dir.resolve(),
        task_id=args.task_id,
        episode_index=args.episode_index,
        seed=args.seed,
    )
    methods["phase_route_v3"], bindings["phase_route_v3"] = load_phase_arm(
        args.phase_route_dir.resolve(),
        task_id=args.task_id,
        episode_index=args.episode_index,
        seed=args.seed,
    )
    summary = build_summary(
        methods,
        bindings,
        task_id=args.task_id,
        episode_index=args.episode_index,
        seed=args.seed,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
