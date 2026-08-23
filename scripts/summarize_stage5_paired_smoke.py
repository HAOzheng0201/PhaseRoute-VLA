#!/usr/bin/env python3
"""Summarize one original-A1/PhaseRoute-V3 state-matched engineering smoke.

This utility is intentionally separate from the preregistered D9 analysis.  It
validates the run identities and telemetry before reporting descriptive paired
outcomes and compute measurements.  It does not turn a small engineering smoke
into a statistical or wall-clock speedup claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


EPISODE_ID_RE = re.compile(r"^libero_10:task(?P<task>\d+):episode(?P<episode>\d+)$")
TASK_RE = re.compile(r"^Task (?P<task>\d+): (?P<instruction>.+)$")
SEED_RE = re.compile(r"^Episode seed: (?P<seed>\d+)$")
EXIT_RE = re.compile(r"^Exit layers this episode: \[(?P<layers>.*)\]$")
SUCCESS_RE = re.compile(r"^Success: (?P<success>True|False)$")
WALL_RE = re.compile(r"^Episode duration: (?P<seconds>\d+(?:\.\d+)?)s$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--baseline-telemetry", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--phase-telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=20260823)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            records.append(value)
    return records


def parse_baseline_log(text: str) -> list[dict[str, Any]]:
    """Parse the one-episode-per-task records emitted by the legacy evaluator."""

    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        task_match = TASK_RE.fullmatch(line)
        if task_match:
            if current is not None:
                raise ValueError("new task started before the previous log row completed")
            current = {
                "task_id": int(task_match.group("task")),
                "episode_index": 0,
                "instruction": task_match.group("instruction"),
            }
            continue
        if current is None:
            continue
        seed_match = SEED_RE.fullmatch(line)
        exit_match = EXIT_RE.fullmatch(line)
        success_match = SUCCESS_RE.fullmatch(line)
        wall_match = WALL_RE.fullmatch(line)
        if seed_match:
            current["seed"] = int(seed_match.group("seed"))
        elif exit_match:
            body = exit_match.group("layers").strip()
            current["exit_layers"] = (
                [int(value.strip()) for value in body.split(",")] if body else []
            )
        elif success_match:
            current["success"] = success_match.group("success") == "True"
        elif wall_match:
            current["wall_seconds"] = float(wall_match.group("seconds"))
            required = {
                "task_id",
                "episode_index",
                "instruction",
                "seed",
                "exit_layers",
                "success",
                "wall_seconds",
            }
            missing = sorted(required - current.keys())
            if missing:
                raise ValueError(f"incomplete baseline task row; missing {missing}")
            rows.append(current)
            current = None
    if current is not None:
        raise ValueError("baseline log ended before its final task row completed")
    if not rows:
        raise ValueError("baseline log contains no completed episode")
    return rows


def _episode_key(record: Mapping[str, Any]) -> tuple[int, int]:
    episode_id = str(record.get("episode_id", ""))
    match = EPISODE_ID_RE.fullmatch(episode_id)
    if match is None:
        raise ValueError(f"invalid telemetry episode_id: {episode_id!r}")
    task_id = int(match.group("task"))
    episode_index = int(match.group("episode"))
    if int(record.get("task_id", -1)) != task_id:
        raise ValueError(f"task_id disagrees with episode_id: {episode_id}")
    return task_id, episode_index


def group_telemetry(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("schema_version") != "phase-route-vla.telemetry.v1":
            raise ValueError("unexpected telemetry schema")
        grouped[_episode_key(record)].append(record)
    if not grouped:
        raise ValueError("telemetry is empty")
    return dict(grouped)


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("latencies must be finite and non-negative")
    return {
        "records": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
    }


def summarize_calls(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layers = [int(record["exit_layer"]) for record in records]
    fm_calls = [int(record["fm_calls"]) for record in records]
    latencies = [float(record["latency_ms"]) for record in records]
    if any(value <= 0 for value in fm_calls):
        raise ValueError("FM call counts must be positive")
    instruction_hashes = {str(record["instruction_hash"]) for record in records}
    if len(instruction_hashes) != 1:
        raise ValueError("one episode contains multiple instruction hashes")
    layer_counts = Counter(layers)
    return {
        "policy_calls": len(records),
        "selected_layers": {
            str(layer): layer_counts[layer] for layer in sorted(layer_counts)
        },
        "early_exit_calls": sum(layer < 27 for layer in layers),
        "early_exit_fraction": sum(layer < 27 for layer in layers) / len(layers),
        "fm_calls": sum(fm_calls),
        "fm_calls_per_policy_call": sum(fm_calls) / len(records),
        "policy_latency": _latency_summary(latencies),
        "instruction_hash": instruction_hashes.pop(),
        "exit_layer_sequence": layers,
    }


def _unique_rows(
    rows: Iterable[Mapping[str, Any]], label: str
) -> dict[tuple[int, int], Mapping[str, Any]]:
    output: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["task_id"]), int(row.get("episode_index", 0)))
        if key in output:
            raise ValueError(f"duplicate {label} episode: {key}")
        output[key] = row
    return output


def _aggregate_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = sum(int(row["policy_calls"]) for row in rows)
    fm_calls = sum(int(row["fm_calls"]) for row in rows)
    layer_counts: Counter[int] = Counter()
    latencies = []
    for row in rows:
        layer_counts.update(
            {int(layer): int(count) for layer, count in row["selected_layers"].items()}
        )
        latencies.extend(float(value) for value in row["latency_ms_by_call"])
    early = sum(count for layer, count in layer_counts.items() if layer < 27)
    return {
        "successes": sum(bool(row["success"]) for row in rows),
        "episodes": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "policy_calls": calls,
        "selected_layers": {
            str(layer): layer_counts[layer] for layer in sorted(layer_counts)
        },
        "early_exit_calls": early,
        "early_exit_fraction": early / calls,
        "fm_calls": fm_calls,
        "fm_calls_per_policy_call": fm_calls / calls,
        "policy_latency": _latency_summary(latencies),
        "summed_episode_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
    }


def build_summary(
    baseline_log_rows: Sequence[Mapping[str, Any]],
    baseline_telemetry: Sequence[Mapping[str, Any]],
    phase_summary: Mapping[str, Any],
    phase_telemetry: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Iterable[int] = range(10),
    expected_episode_indices: Iterable[int] = (0,),
    seed_base: int = 20260823,
) -> dict[str, Any]:
    baseline_logs = _unique_rows(baseline_log_rows, "baseline log")
    baseline_calls = group_telemetry(baseline_telemetry)
    phase_calls = group_telemetry(phase_telemetry)
    phase_rows = _unique_rows(phase_summary.get("episodes", []), "phase summary")
    expected_grid = {
        (int(task_id), int(episode_index))
        for task_id in expected_task_ids
        for episode_index in expected_episode_indices
    }
    observed_grids = (
        baseline_logs.keys(),
        baseline_calls.keys(),
        phase_rows.keys(),
        phase_calls.keys(),
    )
    if any(set(grid) != expected_grid for grid in observed_grids):
        raise ValueError("baseline/PhaseRoute episode grids are incomplete or differ")
    if phase_summary.get("schema_version") != "phase-route-vla.libero-evaluation-summary.v1":
        raise ValueError("unexpected PhaseRoute evaluation schema")
    if phase_summary.get("method") != "phase_route_v3":
        raise ValueError("expected a PhaseRoute-V3 evaluation summary")

    paired_rows = []
    seed_matches = True
    instruction_matches = True
    for key in sorted(expected_grid):
        task_id, episode_index = key
        log_row = baseline_logs[key]
        phase_row = phase_rows[key]
        baseline_call_summary = summarize_calls(baseline_calls[key])
        phase_call_summary = summarize_calls(phase_calls[key])
        expected_seed = seed_base + task_id * 10_000 + episode_index
        seed_match = (
            int(log_row["seed"]) == int(phase_row["seed"]) == expected_seed
        )
        instruction_match = (
            baseline_call_summary["instruction_hash"]
            == phase_call_summary["instruction_hash"]
        )
        seed_matches &= seed_match
        instruction_matches &= instruction_match
        if list(log_row["exit_layers"]) != baseline_call_summary["exit_layer_sequence"]:
            raise ValueError(f"baseline log/telemetry exit layers differ for {key}")
        if int(phase_row["policy_calls"]) != phase_call_summary["policy_calls"]:
            raise ValueError(f"PhaseRoute call count differs for {key}")
        phase_summary_layers = {
            str(layer): int(count)
            for layer, count in phase_row["selected_layers"].items()
            if int(count) != 0
        }
        if phase_summary_layers != phase_call_summary["selected_layers"]:
            raise ValueError(f"PhaseRoute route counts differ for {key}")

        baseline = {
            "success": bool(log_row["success"]),
            "policy_calls": baseline_call_summary["policy_calls"],
            "selected_layers": baseline_call_summary["selected_layers"],
            "early_exit_calls": baseline_call_summary["early_exit_calls"],
            "fm_calls": baseline_call_summary["fm_calls"],
            "fm_calls_per_policy_call": baseline_call_summary[
                "fm_calls_per_policy_call"
            ],
            "latency_ms_by_call": [
                float(record["latency_ms"]) for record in baseline_calls[key]
            ],
            "policy_latency": baseline_call_summary["policy_latency"],
            "wall_seconds": float(log_row["wall_seconds"]),
        }
        phase = {
            "success": bool(phase_row["success"]),
            "policy_calls": phase_call_summary["policy_calls"],
            "selected_layers": phase_call_summary["selected_layers"],
            "early_exit_calls": phase_call_summary["early_exit_calls"],
            "fm_calls": phase_call_summary["fm_calls"],
            "fm_calls_per_policy_call": phase_call_summary["fm_calls_per_policy_call"],
            "latency_ms_by_call": [
                float(record["latency_ms"]) for record in phase_calls[key]
            ],
            "policy_latency": phase_call_summary["policy_latency"],
            "wall_seconds": float(phase_row["wall_seconds"]),
        }
        if baseline["success"] and phase["success"]:
            outcome = "both_success"
        elif baseline["success"]:
            outcome = "baseline_only_success"
        elif phase["success"]:
            outcome = "phase_only_success"
        else:
            outcome = "both_failure"
        paired_rows.append(
            {
                "task_id": task_id,
                "episode_index": episode_index,
                "seed": expected_seed,
                "seed_match": seed_match,
                "instruction_hash_match": instruction_match,
                "paired_outcome": outcome,
                "baseline": baseline,
                "phase_route_v3": phase,
            }
        )

    baseline_overall = _aggregate_method([row["baseline"] for row in paired_rows])
    phase_overall = _aggregate_method([row["phase_route_v3"] for row in paired_rows])
    outcomes = Counter(row["paired_outcome"] for row in paired_rows)
    comparisons = {
        "success_rate_difference_phase_minus_baseline": (
            phase_overall["success_rate"] - baseline_overall["success_rate"]
        ),
        "fm_calls_per_policy_call_reduction_fraction": 1.0
        - phase_overall["fm_calls_per_policy_call"]
        / baseline_overall["fm_calls_per_policy_call"],
        "policy_latency_mean_reduction_fraction": 1.0
        - phase_overall["policy_latency"]["mean_ms"]
        / baseline_overall["policy_latency"]["mean_ms"],
        "policy_latency_median_reduction_fraction": 1.0
        - phase_overall["policy_latency"]["median_ms"]
        / baseline_overall["policy_latency"]["median_ms"],
        "summed_episode_wall_reduction_fraction": 1.0
        - phase_overall["summed_episode_wall_seconds"]
        / baseline_overall["summed_episode_wall_seconds"],
    }
    checks = {
        "complete_expected_grid": len(paired_rows) == len(expected_grid),
        "seed_alignment": seed_matches,
        "instruction_alignment": instruction_matches,
        "baseline_log_grid_complete": baseline_overall["episodes"]
        == len(expected_grid),
        "phase_success_total_matches_summary": phase_overall["successes"]
        == int(phase_summary["total_successes"]),
        "phase_telemetry_error_free": int(phase_summary["telemetry_errors"]) == 0,
    }
    return {
        "schema_version": "phase-route-vla.stage5-paired-engineering-smoke.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "general_simulator_engineering_smoke_not_D9_retest",
        "suite": "libero_10",
        "task_ids": sorted({task_id for task_id, _ in expected_grid}),
        "episode_indices": sorted({index for _, index in expected_grid}),
        "seed_base": seed_base,
        "paired_episodes": len(paired_rows),
        "paired_outcomes": {
            name: outcomes[name]
            for name in (
                "both_success",
                "baseline_only_success",
                "phase_only_success",
                "both_failure",
            )
        },
        "original_a1": baseline_overall,
        "phase_route_v3": phase_overall,
        "descriptive_comparisons": comparisons,
        "checks": checks,
        "claim_boundary": {
            "new_independent_test": False,
            "D9_retest": False,
            "formal_speedup_claim": False,
            "causal_early_exit_failure_attribution": False,
            "deployment_authorized": False,
            "wall_clock_is_descriptive_only": True,
        },
        "rows": paired_rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    input_paths = (
        args.baseline_log,
        args.baseline_telemetry,
        args.phase_summary,
        args.phase_telemetry,
    )
    result = build_summary(
        parse_baseline_log(args.baseline_log.read_text(encoding="utf-8")),
        read_jsonl(args.baseline_telemetry),
        json.loads(args.phase_summary.read_text(encoding="utf-8")),
        read_jsonl(args.phase_telemetry),
        seed_base=args.seed_base,
    )
    result["input_sha256"] = {
        str(path.resolve()): sha256_file(path) for path in input_paths
    }
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
