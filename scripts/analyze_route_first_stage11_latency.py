#!/usr/bin/env python3
"""Post-hoc latency diagnosis for the sealed Route-first Stage-10 evidence.

This analysis is deliberately read-only and descriptive.  It verifies the
Stage-10 evidence chain, aligns policy telemetry with the external timing
overlay, and separates non-overlapping route-runtime calls from the remaining
policy wall time.  It never changes a router, threshold, action, or gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "phase-route-vla.route-first-stage11-latency-diagnosis.v1"
AGGREGATE_SCHEMA = "phase-route-vla.route-first-stage10-active-result.v1"
TRIPLET_SCHEMA = "phase-route-vla.route-first-stage10-active-triplet.v1"
ATTESTATION_SCHEMA = "phase-route-vla.route-first-stage10-arm-attestation.v1"
RESULT_SCHEMA = "phase-route-vla.route-first-stage10-active-arm.v1"
MEASUREMENT_SCHEMA = "phase-route-vla.stage1.measurement.v1"
TELEMETRY_SCHEMA = "phase-route-vla.telemetry.v1"

METHODS = ("original_a1", "candidate_first_v3", "route_first_stage8")
EXPECTED_MODES = {
    "original_a1": "original_a1",
    "candidate_first_v3": "phase_route_v3",
    "route_first_stage8": "route_first_stage8",
}
ROUTE_TOP_LEVEL_COMPONENTS = (
    "runtime_begin",
    "visual_capture",
    "runtime_prepare",
    "selected_action_route",
    "runtime_commit",
)
ROUTE_NESTED_COMPONENTS = ("phase_estimator", "router_predict")


class Stage11LatencyError(ValueError):
    """Raised when the sealed Stage-10 timing evidence is inconsistent."""


@dataclass(frozen=True)
class CallRecord:
    method: str
    task_id: int
    replicate_id: int
    arm_position: int
    call_ordinal: int
    layer: int
    telemetry_fm_event_sum: int
    authoritative_fm_calls: int
    policy_wall_ms: float
    policy_cuda_ms: float
    components: Mapping[str, tuple[float, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage10-root",
        type=Path,
        default=Path("runs/route_first_stage10_active"),
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path(
            "runs/route_first_stage10_active/stage10_active_aggregate.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage11LatencyError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Stage11LatencyError(f"JSON root is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Stage11LatencyError(f"cannot load JSONL {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise Stage11LatencyError(
                f"invalid JSONL row {line_number} in {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise Stage11LatencyError(
                f"JSONL row {line_number} is not an object: {path}"
            )
        rows.append(value)
    if not rows:
        raise Stage11LatencyError(f"JSONL file is empty: {path}")
    return rows


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage11LatencyError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Stage11LatencyError(f"{name} is not finite and non-negative")
    return result


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(_finite_nonnegative(value, "latency") for value in values)
    if not ordered:
        raise Stage11LatencyError("latency collection is empty")

    def nearest_rank(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    total = math.fsum(ordered)
    return {
        "count": len(ordered),
        "sum": total,
        "mean": total / len(ordered),
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "max": ordered[-1],
    }


def _inventory_sha(attestation: Mapping[str, Any], relative_path: str) -> str:
    inventory = attestation.get("artifact_inventory")
    if not isinstance(inventory, list):
        raise Stage11LatencyError("arm attestation artifact inventory is missing")
    matches = [
        item
        for item in inventory
        if isinstance(item, Mapping) and item.get("relative_path") == relative_path
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise Stage11LatencyError(
            f"arm attestation does not bind exactly one {relative_path}"
        )
    return str(matches[0]["sha256"])


def _component_values(row: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    components = row.get("components")
    if not isinstance(components, Mapping):
        raise Stage11LatencyError("measurement components are not an object")
    result: dict[str, tuple[float, ...]] = {}
    for name, events in components.items():
        if not isinstance(name, str) or not isinstance(events, list):
            raise Stage11LatencyError("measurement component structure differs")
        values = []
        for event in events:
            if not isinstance(event, Mapping):
                raise Stage11LatencyError("component event is not an object")
            values.append(
                _finite_nonnegative(event.get("latency_ms"), f"component {name}")
            )
        if not values:
            raise Stage11LatencyError(f"component {name} has no events")
        result[name] = tuple(values)
    return result


def _validate_arm_files(
    *,
    stage10_root: Path,
    triplet: Mapping[str, Any],
    method: str,
    manifest: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    arms = triplet.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(method), Mapping):
        raise Stage11LatencyError(f"triplet arm is missing: {method}")
    arm = arms[method]
    attestation_relative = arm.get("attestation_path")
    attestation_sha = arm.get("attestation_sha256")
    if not isinstance(attestation_relative, str) or not isinstance(attestation_sha, str):
        raise Stage11LatencyError(f"triplet does not bind {method} attestation")
    attestation_path = stage10_root / attestation_relative
    if sha256_file(attestation_path) != attestation_sha:
        raise Stage11LatencyError(f"{method} attestation SHA differs")
    attestation = _load_json(attestation_path)
    if (
        attestation.get("schema_version") != ATTESTATION_SCHEMA
        or attestation.get("status") != "PASS"
        or attestation.get("method") != method
    ):
        raise Stage11LatencyError(f"{method} attestation contract differs")

    arm_root = attestation_path.parent
    paths = {
        "result.json": arm_root / "result.json",
        "stage1_measurement.jsonl": arm_root / "stage1_measurement.jsonl",
        "policy_telemetry.jsonl": arm_root / "policy_telemetry.jsonl",
    }
    for relative, path in paths.items():
        observed = sha256_file(path)
        if observed != _inventory_sha(attestation, relative):
            raise Stage11LatencyError(f"{method} {relative} SHA differs")
        manifest[str(path.relative_to(stage10_root))] = observed

    result = _load_json(paths["result.json"])
    measurements = _load_jsonl(paths["stage1_measurement.jsonl"])
    telemetry = _load_jsonl(paths["policy_telemetry.jsonl"])
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "COMPLETE_ROUTE_FIRST_STAGE10_ACTIVE_ARM"
        or result.get("method") != method
    ):
        raise Stage11LatencyError(f"{method} result contract differs")
    return result, measurements, telemetry


def _aligned_calls(
    result: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
) -> list[CallRecord]:
    method = str(result["method"])
    expected_calls = result.get("policy_accounting", {}).get("policy_calls")
    if (
        type(expected_calls) is not int
        or expected_calls <= 0
        or len(measurements) != expected_calls
        or len(telemetry) != expected_calls
    ):
        raise Stage11LatencyError(f"{method} policy-call counts differ")
    records = []
    for ordinal, (measurement, event) in enumerate(zip(measurements, telemetry)):
        if measurement.get("schema_version") != MEASUREMENT_SCHEMA:
            raise Stage11LatencyError(f"{method} measurement schema differs")
        if event.get("schema_version") != TELEMETRY_SCHEMA:
            raise Stage11LatencyError(f"{method} telemetry schema differs")
        if measurement.get("mode") != EXPECTED_MODES[method]:
            raise Stage11LatencyError(f"{method} measurement mode differs")
        if measurement.get("measurement_is_control_input") is not False:
            raise Stage11LatencyError("Stage-10 timing was marked as a control input")
        if measurement.get("d9_protected_source_modified") is not False:
            raise Stage11LatencyError("Stage-10 timing reports modified D9 source")
        if measurement.get("error") is not None:
            raise Stage11LatencyError(f"{method} measurement contains an error")
        if measurement.get("action_finite") is not True:
            raise Stage11LatencyError(f"{method} action audit is not finite")
        context = measurement.get("context")
        if not isinstance(context, Mapping):
            raise Stage11LatencyError(f"{method} measurement context is missing")
        if (
            context.get("episode_id") != event.get("episode_id")
            or context.get("step_id") != event.get("step_id")
            or context.get("task_id") != event.get("task_id")
        ):
            raise Stage11LatencyError(f"{method} measurement/telemetry alignment differs")
        layer = event.get("exit_layer")
        fm_calls = event.get("fm_calls")
        if type(layer) is not int or layer < 0 or layer > 27:
            raise Stage11LatencyError(f"{method} exit layer is invalid")
        if type(fm_calls) is not int or fm_calls <= 0:
            raise Stage11LatencyError(f"{method} FM call count is invalid")
        authoritative_fm_calls = fm_calls
        if method == "route_first_stage8":
            extra = event.get("extra")
            events = extra.get("exit_events") if isinstance(extra, Mapping) else None
            if not isinstance(events, list):
                raise Stage11LatencyError("route-first runtime events are missing")
            selected_events = [
                item
                for item in events
                if isinstance(item, Mapping)
                and item.get("event") == "route_first_selected_action"
                and item.get("should_exit") is True
                and item.get("fm_calls") == 1
            ]
            if len(selected_events) != 1:
                raise Stage11LatencyError(
                    "route-first call does not contain exactly one authoritative FM event"
                )
            # The same selected FM is referenced by route-first-selected,
            # phase-route-decision, and exit-candidate telemetry.  The top-level
            # telemetry sum is therefore not an execution count for this arm.
            authoritative_fm_calls = 1
        measured_layer = measurement.get("selected_layer")
        if method == "original_a1":
            if measured_layer is not None:
                raise Stage11LatencyError("Original A1 measurement unexpectedly has a layer")
        elif measured_layer != layer:
            raise Stage11LatencyError(f"{method} selected layers differ")
        records.append(
            CallRecord(
                method=method,
                task_id=int(result["task_id"]),
                replicate_id=int(result["replicate_id"]),
                arm_position=int(result["arm_position"]),
                call_ordinal=ordinal,
                layer=layer,
                telemetry_fm_event_sum=fm_calls,
                authoritative_fm_calls=authoritative_fm_calls,
                policy_wall_ms=_finite_nonnegative(
                    measurement.get("policy_wall_latency_ms"), "policy wall latency"
                ),
                policy_cuda_ms=_finite_nonnegative(
                    measurement.get("policy_cuda_event_latency_ms"),
                    "policy CUDA latency",
                ),
                components=_component_values(measurement),
            )
        )
    return records


def _method_summary(records: Sequence[CallRecord]) -> dict[str, Any]:
    if not records:
        raise Stage11LatencyError("method has no call records")
    method = records[0].method
    if any(record.method != method for record in records):
        raise Stage11LatencyError("method summary mixes different methods")
    layers = Counter(record.layer for record in records)
    layer_groups: dict[int, list[CallRecord]] = defaultdict(list)
    position_groups: dict[int, list[CallRecord]] = defaultdict(list)
    cold_groups: dict[str, list[CallRecord]] = defaultdict(list)
    for record in records:
        layer_groups[record.layer].append(record)
        position_groups[record.arm_position].append(record)
        cold_groups["first_call" if record.call_ordinal == 0 else "steady_calls"].append(
            record
        )

    def group_summary(group: Sequence[CallRecord]) -> dict[str, Any]:
        return {
            "calls": len(group),
            "policy_wall_ms": latency_summary(r.policy_wall_ms for r in group),
            "policy_cuda_ms": latency_summary(r.policy_cuda_ms for r in group),
            "telemetry_fm_event_sum": {
                "total": sum(r.telemetry_fm_event_sum for r in group),
                "per_policy_call_mean": statistics.fmean(
                    r.telemetry_fm_event_sum for r in group
                ),
                "distribution": latency_summary(
                    float(r.telemetry_fm_event_sum) for r in group
                ),
                "is_authoritative_execution_count_for_this_method": (
                    method != "route_first_stage8"
                ),
                "route_first_repeats_one_FM_across_three_events": (
                    method == "route_first_stage8"
                ),
            },
            "authoritative_fm_calls": {
                "total": sum(r.authoritative_fm_calls for r in group),
                "per_policy_call_mean": statistics.fmean(
                    r.authoritative_fm_calls for r in group
                ),
                "distribution": latency_summary(
                    float(r.authoritative_fm_calls) for r in group
                ),
            },
        }

    return {
        **group_summary(records),
        "selected_layer_counts": {f"L{layer}": layers[layer] for layer in sorted(layers)},
        "selected_layer_share": {
            f"L{layer}": layers[layer] / len(records) for layer in sorted(layers)
        },
        "by_selected_layer": {
            f"L{layer}": group_summary(group)
            for layer, group in sorted(layer_groups.items())
        },
        "by_arm_position": {
            str(position): group_summary(group)
            for position, group in sorted(position_groups.items())
        },
        "cold_start_stratification": {
            name: group_summary(group) for name, group in sorted(cold_groups.items())
        },
    }


def route_overlay_for_call(record: CallRecord) -> dict[str, float]:
    if record.method != "route_first_stage8":
        raise Stage11LatencyError("route overlay requested for a non-route call")
    unexpected = set(ROUTE_TOP_LEVEL_COMPONENTS).difference(record.components)
    if unexpected:
        raise Stage11LatencyError(
            "route call is missing top-level components: " + ", ".join(sorted(unexpected))
        )
    top_level: dict[str, float] = {}
    for name in ROUTE_TOP_LEVEL_COMPONENTS:
        values = record.components[name]
        if len(values) != 1:
            raise Stage11LatencyError(f"route top-level component {name} is not unique")
        top_level[name] = values[0]
    overlay = math.fsum(top_level.values())
    residual = record.policy_wall_ms - overlay
    if residual < -1e-6:
        raise Stage11LatencyError("instrumented route overlay exceeds policy wall time")
    prepare = top_level["runtime_prepare"]
    nested: dict[str, float] = {}
    for name in ROUTE_NESTED_COMPONENTS:
        values = record.components.get(name)
        if values is None or len(values) != 1:
            raise Stage11LatencyError(f"route nested component {name} is not unique")
        nested[name] = values[0]
    prepare_other = prepare - math.fsum(nested.values())
    if prepare_other < -1e-6:
        raise Stage11LatencyError("nested prepare timings exceed runtime_prepare")
    return {
        **top_level,
        **{f"nested_{name}": value for name, value in nested.items()},
        "runtime_prepare_other": max(0.0, prepare_other),
        "instrumented_route_overlay": overlay,
        "uninstrumented_policy_residual": max(0.0, residual),
        "instrumented_route_overlay_fraction": overlay / record.policy_wall_ms,
    }


def _route_diagnosis(records: Sequence[CallRecord]) -> dict[str, Any]:
    routes = [record for record in records if record.method == "route_first_stage8"]
    if not routes:
        raise Stage11LatencyError("route-first calls are missing")
    rows = [(record, route_overlay_for_call(record)) for record in routes]
    metric_names = tuple(rows[0][1])
    by_layer: dict[int, list[tuple[CallRecord, Mapping[str, float]]]] = defaultdict(list)
    by_task: dict[int, list[CallRecord]] = defaultdict(list)
    for record, values in rows:
        if tuple(values) != metric_names:
            raise Stage11LatencyError("route overlay metrics differ across calls")
        by_layer[record.layer].append((record, values))
        by_task[record.task_id].append(record)
    return {
        "timing_hierarchy": {
            "non_overlapping_top_level_components": list(ROUTE_TOP_LEVEL_COMPONENTS),
            "runtime_prepare_contains_nested_components": list(ROUTE_NESTED_COMPONENTS),
            "nested_components_are_not_added_to_overlay_again": True,
            "uninstrumented_residual_contains": [
                "image_preprocessing_and_vision_backbone",
                "VLM_decoder_to_selected_layer",
                "single_selected_action_flow_matching",
                "action_conversion_and_uninstrumented_wrapper_cost",
            ],
            "VLM_and_FM_are_not_separately_identifiable_from_stage10_evidence": True,
        },
        "all_route_calls": {
            name: latency_summary(values[name] for _, values in rows)
            for name in metric_names
        },
        "by_selected_layer": {
            f"L{layer}": {
                "calls": len(group),
                **{
                    name: latency_summary(values[name] for _, values in group)
                    for name in metric_names
                },
            }
            for layer, group in sorted(by_layer.items())
        },
        "safe_l13_coverage_by_task": {
            str(task_id): {
                "calls": len(group),
                "L13_calls": sum(record.layer == 13 for record in group),
                "L13_share": sum(record.layer == 13 for record in group) / len(group),
            }
            for task_id, group in sorted(by_task.items())
        },
    }


def _comparison(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def layer(method: str, name: str) -> Mapping[str, Any]:
        value = methods[method]["by_selected_layer"].get(name)
        if not isinstance(value, Mapping):
            raise Stage11LatencyError(f"{method} does not contain {name}")
        return value

    a1_l11 = layer("original_a1", "L11")
    a1_l27 = layer("original_a1", "L27")
    route_l13 = layer("route_first_stage8", "L13")
    route_l27 = layer("route_first_stage8", "L27")

    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            raise Stage11LatencyError("latency ratio denominator is not positive")
        return numerator / denominator

    return {
        "route_L13_vs_A1_L11_policy_wall_p50_ratio_descriptive": ratio(
            float(route_l13["policy_wall_ms"]["p50"]),
            float(a1_l11["policy_wall_ms"]["p50"]),
        ),
        "route_L27_vs_A1_L27_policy_wall_p50_ratio_descriptive": ratio(
            float(route_l27["policy_wall_ms"]["p50"]),
            float(a1_l27["policy_wall_ms"]["p50"]),
        ),
        "A1_L11_share": methods["original_a1"]["selected_layer_share"]["L11"],
        "route_L13_share": methods["route_first_stage8"]["selected_layer_share"]["L13"],
        "route_L27_share": methods["route_first_stage8"]["selected_layer_share"]["L27"],
        "interpretation": {
            "median_gap_is_a_mixture_and_path_coverage_problem": True,
            "route_deep_path_avoids_repeated_candidate_FM": True,
            "stage10_does_not_identify_a_safe_new_threshold": True,
        },
    }


def analyze_stage10_latency(stage10_root: Path, aggregate_path: Path) -> dict[str, Any]:
    root = stage10_root.resolve(strict=True)
    aggregate_target = aggregate_path.resolve(strict=True)
    aggregate = _load_json(aggregate_target)
    if aggregate.get("schema_version") != AGGREGATE_SCHEMA:
        raise Stage11LatencyError("Stage-10 aggregate schema differs")
    if aggregate.get("triplets") != 60 or aggregate.get("active_rollouts") != 180:
        raise Stage11LatencyError("Stage-10 aggregate coverage is incomplete")
    bindings = aggregate.get("triplet_record_sha256")
    if not isinstance(bindings, Mapping) or len(bindings) != 60:
        raise Stage11LatencyError("Stage-10 triplet bindings are incomplete")

    calls: list[CallRecord] = []
    manifest: dict[str, str] = {}
    for task_id in range(10):
        for replicate_id in range(6):
            cluster = (
                f"libero_10:task{task_id}:route_first_fresh_v1:replicate{replicate_id}"
            )
            expected_sha = bindings.get(cluster)
            if not isinstance(expected_sha, str):
                raise Stage11LatencyError(f"aggregate does not bind {cluster}")
            triplet_path = (
                root
                / f"task{task_id:02d}_replicate{replicate_id:02d}"
                / "triplet_record.json"
            )
            observed_sha = sha256_file(triplet_path)
            if observed_sha != expected_sha:
                raise Stage11LatencyError(f"triplet SHA differs: {cluster}")
            manifest[str(triplet_path.relative_to(root))] = observed_sha
            triplet = _load_json(triplet_path)
            if (
                triplet.get("schema_version") != TRIPLET_SCHEMA
                or triplet.get("status") != "COMPLETE_ROUTE_FIRST_STAGE10_TRIPLET"
                or triplet.get("cluster_key") != cluster
                or triplet.get("task_id") != task_id
                or triplet.get("replicate_id") != replicate_id
            ):
                raise Stage11LatencyError(f"triplet identity differs: {cluster}")
            for method in METHODS:
                result, measurements, telemetry = _validate_arm_files(
                    stage10_root=root,
                    triplet=triplet,
                    method=method,
                    manifest=manifest,
                )
                calls.extend(_aligned_calls(result, measurements, telemetry))

    by_method = {
        method: _method_summary([record for record in calls if record.method == method])
        for method in METHODS
    }
    manifest_lines = "".join(
        f"{path}\0{manifest[path]}\n" for path in sorted(manifest)
    ).encode("utf-8")
    aggregate_sha = sha256_file(aggregate_target)
    return {
        "schema_version": SCHEMA,
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC_NOT_A_CONFIRMATION",
        "source": {
            "stage10_aggregate": str(aggregate_target.relative_to(root.parent.parent)),
            "stage10_aggregate_sha256": aggregate_sha,
            "stage10_status": aggregate.get("status"),
            "stage10_source_git_commits": aggregate.get("source_git_commits"),
            "evidence_files": len(manifest),
            "evidence_manifest_sha256": hashlib.sha256(manifest_lines).hexdigest(),
        },
        "coverage": {
            "triplets": 60,
            "arms": 180,
            "policy_calls": len(calls),
            "policy_calls_by_method": Counter(record.method for record in calls),
            "all_measurement_telemetry_pairs_aligned": True,
            "all_bound_artifact_sha256_verified": True,
        },
        "methods": by_method,
        "route_runtime_decomposition": _route_diagnosis(calls),
        "descriptive_comparison": _comparison(by_method),
        "claim_boundary": {
            "posthoc_diagnostic_only": True,
            "stage10_gate_reinterpreted": False,
            "stage10_threshold_tuning_allowed": False,
            "VLM_FM_separate_latency_claim_allowed": False,
            "causal_component_speedup_claim_allowed": False,
            "new_confirmation_claim_allowed": False,
            "independent_development_data_required_for_changes": True,
        },
    }


def main() -> int:
    args = parse_args()
    result = analyze_stage10_latency(args.stage10_root, args.aggregate)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.write_text(payload, encoding="utf-8")
        print(f"wrote {output}")
        print(f"SHA-256: {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
