#!/usr/bin/env python3
"""Aggregate all ten sealed state-13 task pairs and evaluate pilot gates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


SCHEMA = "phase-route-vla.route-first-stage9-pilot-result.v1"
PAIR_SCHEMA = "phase-route-vla.route-first-stage9-pilot-task-pair.v1"
PROTOCOL_SHA = "fcb1c2a1fdf7ea3f79343f72d25240449500a5eac3fad1372f0808023888db4d"
ARM_SCHEMA = "phase-route-vla.route-first-stage9-pilot-arm.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-pair", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_policy_latencies(path: Path) -> list[float]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("policy_wall_latency_ms") if isinstance(row, dict) else None
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"invalid policy latency: {path}")
        values.append(float(value))
    if not values:
        raise ValueError(f"empty policy latency file: {path}")
    return values


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize empty latency values")
    return {
        "count": len(values),
        "sum": math.fsum(values),
        "mean": statistics.fmean(values),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": max(values),
    }


def verify_arm_evidence(
    pair: Mapping[str, Any], role: str, expected_method: str, run_dir: Path
) -> Mapping[str, Any]:
    attestations = pair.get("input_attestations")
    expected = attestations.get(role) if isinstance(attestations, Mapping) else None
    attestation_path = run_dir / "stage9_pilot_arm_attestation.json"
    if (
        not isinstance(expected, Mapping)
        or expected.get("path") != str(attestation_path)
        or not attestation_path.is_file()
        or sha256_file(attestation_path) != expected.get("sha256")
    ):
        raise ValueError(f"pilot arm attestation drift: {run_dir}")
    attestation = load_object(attestation_path)
    if (
        attestation.get("schema_version") != ARM_SCHEMA
        or attestation.get("status") != "PASS"
        or attestation.get("protocol_sha256") != PROTOCOL_SHA
        or attestation.get("method") != expected_method
    ):
        raise ValueError(f"pilot arm identity drift: {run_dir}")
    artifacts = attestation.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"pilot arm artifact inventory missing: {run_dir}")
    for filename, metadata in artifacts.items():
        path = run_dir / str(filename)
        if (
            not isinstance(metadata, Mapping)
            or not path.is_file()
            or path.stat().st_size != metadata.get("bytes")
            or sha256_file(path) != metadata.get("sha256")
        ):
            raise ValueError(f"pilot raw artifact drift: {path}")
    return attestation


def aggregate_pairs(pair_paths: Sequence[Path]) -> dict[str, Any]:
    if len(pair_paths) != 10:
        raise ValueError("pilot aggregation requires exactly 10 task pairs")
    pairs: dict[int, Mapping[str, Any]] = {}
    inputs: dict[str, Any] = {}
    candidate_latencies: list[float] = []
    route_latencies: list[float] = []
    candidate_successes = 0
    route_successes = 0
    route_fm_invocations = 0
    route_decoder_blocks = 0
    outcomes: Counter[str] = Counter()
    rows = []
    for raw_path in pair_paths:
        path = raw_path.resolve(strict=True)
        pair = load_object(path)
        if (
            pair.get("schema_version") != PAIR_SCHEMA
            or pair.get("status") != "PASS"
            or pair.get("protocol_sha256") != PROTOCOL_SHA
            or pair.get("episode_index") != 13
            or not all(value is True for value in pair.get("checks", {}).values())
        ):
            raise ValueError(f"task pair did not pass: {path}")
        task_id = int(pair.get("task_id", -1))
        if task_id in pairs:
            raise ValueError(f"duplicate pilot task pair: {task_id}")
        pairs[task_id] = pair
        inputs[str(task_id)] = {"path": str(path), "sha256": sha256_file(path)}
        run_dirs = pair.get("run_dirs")
        candidate = pair.get("candidate_first")
        route = pair.get("route_first")
        if not all(isinstance(item, Mapping) for item in (run_dirs, candidate, route)):
            raise ValueError(f"task pair evidence incomplete: {path}")
        candidate_dir = Path(str(run_dirs["candidate_first"])).resolve(strict=True)
        route_dir = Path(str(run_dirs["route_first"])).resolve(strict=True)
        verify_arm_evidence(
            pair, "candidate_first", "candidate_first_v3", candidate_dir
        )
        verify_arm_evidence(
            pair, "route_first", "route_first_stage8", route_dir
        )
        candidate_values = load_policy_latencies(
            candidate_dir / "stage1_measurement.jsonl"
        )
        route_values = load_policy_latencies(route_dir / "stage1_measurement.jsonl")
        if len(candidate_values) != candidate.get("policy_calls"):
            raise ValueError(f"candidate measurement count drift: task {task_id}")
        if len(route_values) != route.get("policy_calls"):
            raise ValueError(f"route measurement count drift: task {task_id}")
        candidate_latencies.extend(candidate_values)
        route_latencies.extend(route_values)
        candidate_successes += int(bool(candidate.get("success")))
        route_successes += int(bool(route.get("success")))
        route_fm_invocations += int(route.get("route_exact_fm_invocations", 0))
        route_decoder_blocks += int(route.get("route_decoder_blocks", 0))
        outcomes[str(pair.get("paired_outcome"))] += 1
        rows.append(
            {
                "task_id": task_id,
                "arm_order": pair.get("arm_order"),
                "paired_outcome": pair.get("paired_outcome"),
                "candidate_success": bool(candidate.get("success")),
                "route_success": bool(route.get("success")),
                "candidate_policy_calls": len(candidate_values),
                "route_policy_calls": len(route_values),
                "candidate_p50_ms": pair["candidate_first"][
                    "policy_wall_latency_ms"
                ]["p50"],
                "route_p50_ms": pair["route_first"]["policy_wall_latency_ms"]["p50"],
            }
        )
    if set(pairs) != set(range(10)):
        raise ValueError("pilot task grid must be exactly 0..9")
    candidate_latency = latency_summary(candidate_latencies)
    route_latency = latency_summary(route_latencies)
    median_ratio = route_latency["p50"] / candidate_latency["p50"]
    checks = {
        "complete_task_grid": True,
        "all_task_pairs_pass": True,
        "route_success_guardrail": route_successes >= candidate_successes - 2,
        "route_median_latency_ratio_at_most_0_90": median_ratio <= 0.90,
        "route_one_fm_per_policy_call": route_fm_invocations
        == route_latency["count"],
    }
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "stage9_state13_preregistered_engineering_pilot",
        "protocol_sha256": PROTOCOL_SHA,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "task_ids": list(range(10)),
        "episode_indices": [13],
        "paired_episodes": 10,
        "candidate_first": {
            "successes": candidate_successes,
            "policy_wall_latency_ms": candidate_latency,
        },
        "route_first": {
            "successes": route_successes,
            "policy_wall_latency_ms": route_latency,
            "fm_invocations": route_fm_invocations,
            "decoder_blocks": route_decoder_blocks,
        },
        "paired_outcomes": dict(sorted(outcomes.items())),
        "descriptive_comparison": {
            "success_difference_route_minus_candidate": route_successes
            - candidate_successes,
            "policy_wall_median_ratio_route_to_candidate": median_ratio,
            "policy_wall_median_reduction_fraction": 1.0 - median_ratio,
            "policy_wall_mean_ratio_route_to_candidate": route_latency["mean"]
            / candidate_latency["mean"],
            "policy_wall_mean_reduction_fraction": 1.0
            - route_latency["mean"] / candidate_latency["mean"],
        },
        "checks": checks,
        "access_ledger": {
            "state12_smoke_opened": True,
            "state13_pilot_opened": True,
            "historical_D9_states40_to49_opened_for_this_stage": False,
        },
        "next_gate": {
            "fresh_state_confirmation_authorized": all(checks.values()),
            "deployment_authorized": False,
        },
        "claim_boundary": {
            "engineering_pilot_only": True,
            "powered_noninferiority_claim": False,
            "formal_wall_clock_speedup_claim": False,
            "final_closed_loop_improvement_claim": False,
            "deployment_authorized": False,
        },
        "rows": sorted(rows, key=lambda row: row["task_id"]),
        "inputs": inputs,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_name(
        args.output.name + ".incomplete"
    ).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = aggregate_pairs(args.task_pair)
    temporary = args.output.with_name(args.output.name + ".incomplete")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
