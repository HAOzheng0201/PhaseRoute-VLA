#!/usr/bin/env python3
"""Aggregate all 60 immutable Stage 10 triplets and evaluate frozen gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402


METHODS = CONTRACT.METHODS
TRIPLET_COUNT = CONTRACT.TRIPLET_COUNT
load_schedule = CONTRACT.load_schedule
sha256_file = CONTRACT.sha256_file
ACTIVE_AGGREGATE_SCHEMA = ACTIVE.ACTIVE_AGGREGATE_SCHEMA
ACTIVE_OUTPUT_RELATIVE_PATH = ACTIVE.ACTIVE_OUTPUT_RELATIVE_PATH
Stage10ActiveError = ACTIVE.Stage10ActiveError
aggregate_triplets = ACTIVE.aggregate_triplets
read_jsonl = ACTIVE.read_jsonl
validate_triplet_record = ACTIVE.validate_triplet_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / ACTIVE_OUTPUT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / ACTIVE_OUTPUT_RELATIVE_PATH
        / "stage10_active_aggregate.json",
    )
    return parser.parse_args()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ActiveError(f"JSON object required: {path}")
    return dict(value)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _exact_mcnemar(discordant_a: int, discordant_b: int) -> dict[str, Any]:
    total = discordant_a + discordant_b
    if total == 0:
        p_value = 1.0
    else:
        lower = min(discordant_a, discordant_b)
        tail = sum(math.comb(total, k) for k in range(lower + 1)) / (2**total)
        p_value = min(1.0, 2.0 * tail)
    return {
        "route_only_success": discordant_a,
        "comparator_only_success": discordant_b,
        "discordant_pairs": total,
        "two_sided_exact_p_value_descriptive_only": p_value,
    }


def aggregate(input_root: Path) -> dict[str, Any]:
    schedule = load_schedule(REPO_ROOT)
    if len(schedule) != TRIPLET_COUNT:
        raise Stage10ActiveError("frozen schedule coverage differs")
    triplets = []
    manifest = {}
    pooled_latency: dict[str, list[float]] = {method: [] for method in METHODS}
    telemetry_fm_calls = Counter({method: 0 for method in METHODS})
    environment_steps = Counter({method: 0 for method in METHODS})
    rollout_wall_seconds = defaultdict(float)
    arm_order_success = {
        method: {"position1": [0, 0], "position2": [0, 0], "position3": [0, 0]}
        for method in METHODS
    }
    commits = set()
    gpu_uuids = Counter()
    for spec in schedule:
        directory = input_root / f"task{spec.task_id:02d}_replicate{spec.replicate_id:02d}"
        path = directory / "triplet_record.json"
        sidecar = directory / "triplet_record.sha256"
        if not path.is_file() or not sidecar.is_file():
            raise Stage10ActiveError(
                "all 60 triplets are required before any aggregate is computed"
            )
        digest = sha256_file(path)
        if sidecar.read_text(encoding="utf-8").split()[0] != digest:
            raise Stage10ActiveError("triplet SHA-256 sidecar differs")
        record = _object(path)
        validate_triplet_record(record, spec=spec)
        triplets.append(record)
        manifest[spec.cluster_key] = digest
        commits.add(record["source_git_commit"])
        gpu_uuids[record["gpu_uuid"]] += 1
        for position, method in enumerate(spec.arm_order, start=1):
            arm_dir = directory / f"arm{position}_{method}"
            result = _object(arm_dir / "result.json")
            measurements = read_jsonl(arm_dir / "stage1_measurement.jsonl")
            values = [float(item["policy_wall_latency_ms"]) for item in measurements]
            if not values:
                raise Stage10ActiveError("completed arm has no latency records")
            pooled_latency[method].extend(values)
            policy = result["policy_accounting"]
            telemetry_fm_calls[method] += int(policy["telemetry_fm_calls"])
            environment_steps[method] += int(result["environment_steps"])
            rollout_wall_seconds[method] += float(result["rollout_wall_seconds"])
            bucket = arm_order_success[method][f"position{position}"]
            bucket[0] += int(bool(result["success"]))
            bucket[1] += 1
    primary = aggregate_triplets(schedule, triplets)
    if primary.get("schema_version") != ACTIVE_AGGREGATE_SCHEMA:
        raise Stage10ActiveError("primary Stage 10 aggregate schema differs")
    pooled = {}
    for method, values in pooled_latency.items():
        pooled[method] = {
            "policy_calls": len(values),
            "mean_ms": math.fsum(values) / len(values),
            "p50_ms": _percentile(values, 0.50),
            "p90_ms": _percentile(values, 0.90),
            "p95_ms": _percentile(values, 0.95),
            "max_ms": max(values),
        }
    discordance = primary["paired_success_discordance"]
    primary.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "source_git_commits": sorted(commits),
            "gpu_uuid_triplet_counts": dict(gpu_uuids),
            "triplet_record_sha256": manifest,
            "pooled_policy_latency_ms": pooled,
            "telemetry_fm_calls": dict(telemetry_fm_calls),
            "environment_steps": dict(environment_steps),
            "rollout_wall_seconds": dict(rollout_wall_seconds),
            "arm_order_stratified_raw_success": arm_order_success,
            "exact_mcnemar_descriptive": {
                "route_vs_candidate": _exact_mcnemar(
                    discordance["route_success_candidate_failure"],
                    discordance["route_failure_candidate_success"],
                ),
                "route_vs_original_a1": _exact_mcnemar(
                    discordance["route_success_original_failure"],
                    discordance["route_failure_original_success"],
                ),
            },
            "reporting_complete": {
                "three_arm_success_and_paired_differences": True,
                "within_triplet_policy_p50_ratios": True,
                "route_first_exact_fm_integrity": True,
                "per_task_success_counts": True,
                "pooled_policy_latency": True,
                "selected_layer_counts": True,
                "arm_order_stratification": True,
                "all_infrastructure_incidents_in_raw_tree": True,
            },
        }
    )
    return primary


def main() -> None:
    args = parse_args()
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip():
        raise PermissionError("Stage 10 aggregation requires a clean worktree")
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".incomplete")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or temporary.exists() or sidecar.exists():
        raise FileExistsError("Stage 10 aggregate refuses to overwrite evidence")
    value = aggregate(args.input_root.resolve(strict=True))
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
