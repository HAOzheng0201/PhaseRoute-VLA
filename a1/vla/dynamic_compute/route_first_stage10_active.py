"""Fail-closed contracts for the Stage 10 fresh-state active confirmation.

This module intentionally has no simulator or CUDA imports.  It validates the
frozen schedule, the local state payload, per-policy-call evidence, immutable
arm/triplet records, and the final three-arm aggregate before a scientific
claim is emitted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from .route_first_stage10 import (
        ACTIVE_ROLLOUT_COUNT,
        METHODS,
        PROTOCOL_SHA256,
        SCHEDULE_SHA256,
        STATE_PAYLOAD_SCHEMA,
        TRIPLET_COUNT,
        FreshTripletSpec,
        canonical_state_bytes,
        load_schedule,
        sha256_file,
        validate_local_state_artifacts,
    )
else:  # Loaded by the no-CUDA script bootstrap without importing ``a1``.
    _contract = sys.modules.get("_phase_route_stage10_contract")
    if _contract is None:
        raise ImportError("Stage 10 direct contract bootstrap is missing")
    ACTIVE_ROLLOUT_COUNT = _contract.ACTIVE_ROLLOUT_COUNT
    METHODS = _contract.METHODS
    PROTOCOL_SHA256 = _contract.PROTOCOL_SHA256
    SCHEDULE_SHA256 = _contract.SCHEDULE_SHA256
    STATE_PAYLOAD_SCHEMA = _contract.STATE_PAYLOAD_SCHEMA
    TRIPLET_COUNT = _contract.TRIPLET_COUNT
    FreshTripletSpec = _contract.FreshTripletSpec
    canonical_state_bytes = _contract.canonical_state_bytes
    load_schedule = _contract.load_schedule
    sha256_file = _contract.sha256_file
    validate_local_state_artifacts = _contract.validate_local_state_artifacts


ACTIVE_ARM_SCHEMA = "phase-route-vla.route-first-stage10-active-arm.v1"
ACTIVE_TRIPLET_SCHEMA = "phase-route-vla.route-first-stage10-active-triplet.v1"
ACTIVE_AGGREGATE_SCHEMA = "phase-route-vla.route-first-stage10-active-result.v1"
ARM_ATTESTATION_SCHEMA = (
    "phase-route-vla.route-first-stage10-arm-attestation.v1"
)
PREFLIGHT_SCHEMA = "phase-route-vla.route-first-stage10-arm-preflight.v1"
POSTFLIGHT_SCHEMA = "phase-route-vla.route-first-stage10-arm-postflight.v1"
RUNNER_READINESS_SCHEMA = (
    "phase-route-vla.route-first-stage10-runner-readiness.v1"
)
RUNNER_READINESS_STATUS = "PASS_ROUTE_FIRST_STAGE10_RUNNER_READINESS"
RUNNER_READINESS_RELATIVE_PATH = Path(
    "results/route_first/route_first_stage10_runner_readiness.json"
)
ACTIVE_OUTPUT_RELATIVE_PATH = Path("runs/route_first_stage10_active")
MINIMUM_FREE_MEMORY_MIB = 40_000
PROTECTED_CODE_SHA256 = {
    "a1/vla/value_net.py": (
        "ec3a860427f32d5837e279eb17eeb28befaee9dd7944d46482173c85e8847dc1"
    ),
    "robot_experiments/libero/exit_vla_utils.py": (
        "e5c88b72199c1354fc7b3f2fa22e056b593ee5cdadf7185cc7d1c09fe768051a"
    ),
    "robot_experiments/libero/eval_libero_early_exit.py": (
        "a4e3b1b49cdaf2021b3cd370d8a1e89c927906e7cbd5f8afdccd5ceb5b1826cd"
    ),
}
METHOD_LAYERS = {
    "original_a1": tuple(range(1, 28, 2)),
    "candidate_first_v3": (11, 13, 27),
    "route_first_stage8": (13, 27),
}
MEASUREMENT_MODES = {
    "original_a1": "original_a1",
    "candidate_first_v3": "phase_route_v3",
    "route_first_stage8": "route_first_stage8",
}


class Stage10ActiveError(ValueError):
    """Raised when active-control identity or evidence differs from freeze."""


@dataclass(frozen=True)
class ActiveArmSpec:
    task_id: int
    replicate_id: int
    cluster_key: str
    state_seed: int
    policy_seed: int
    arm_order: tuple[str, str, str]
    method: str
    arm_position: int


def normalize_gpu_uuid(value: Any) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def git_output(repo_root: str | Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=Path(repo_root), text=True
    ).strip()


def select_triplet(
    repo_root: str | Path, task_id: int, replicate_id: int
) -> FreshTripletSpec:
    if type(task_id) is not int or type(replicate_id) is not int:
        raise Stage10ActiveError("task and replicate ids must be integers")
    matches = tuple(
        item
        for item in load_schedule(repo_root)
        if item.task_id == task_id and item.replicate_id == replicate_id
    )
    if len(matches) != 1:
        raise Stage10ActiveError("selection is not one frozen Stage 10 triplet")
    return matches[0]


def select_arm(
    repo_root: str | Path,
    *,
    task_id: int,
    replicate_id: int,
    method: str,
    arm_position: int,
) -> ActiveArmSpec:
    triplet = select_triplet(repo_root, task_id, replicate_id)
    if method not in METHODS:
        raise Stage10ActiveError("unknown Stage 10 method")
    if type(arm_position) is not int or arm_position not in (1, 2, 3):
        raise Stage10ActiveError("arm position must be 1, 2, or 3")
    if triplet.arm_order[arm_position - 1] != method:
        raise Stage10ActiveError("method differs from frozen arm order")
    return ActiveArmSpec(
        task_id=triplet.task_id,
        replicate_id=triplet.replicate_id,
        cluster_key=triplet.cluster_key,
        state_seed=triplet.state_seed,
        policy_seed=triplet.policy_seed,
        arm_order=triplet.arm_order,
        method=method,
        arm_position=arm_position,
    )


def expected_triplet_directory(
    repo_root: str | Path, task_id: int, replicate_id: int
) -> Path:
    select_triplet(repo_root, task_id, replicate_id)
    return (
        Path(repo_root).resolve()
        / ACTIVE_OUTPUT_RELATIVE_PATH
        / f"task{task_id:02d}_replicate{replicate_id:02d}"
    )


def _strict_int_sequence(value: Any, name: str) -> tuple[int, ...]:
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Stage10ActiveError(f"{name} must be an integer sequence")
    result = tuple(value)
    if any(type(item) is not int for item in result):
        raise Stage10ActiveError(f"{name} contains non-integers")
    return result


def validate_state_payload_mapping(
    schedule: Sequence[FreshTripletSpec],
    payload: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> tuple[np.ndarray, ...]:
    """Validate all payload metadata and canonical state bytes in memory."""

    if len(schedule) != TRIPLET_COUNT:
        raise Stage10ActiveError("state payload schedule coverage differs")
    expected_source = attestation.get("source_git_commit")
    if (
        payload.get("schema_version") != STATE_PAYLOAD_SCHEMA
        or payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("schedule_sha256") != SCHEDULE_SHA256
        or payload.get("source_git_commit") != expected_source
        or payload.get("determinism_passes") != 2
        or payload.get("initial_task_success_all_false") is not True
        or payload.get("official_episode_identity_used") is not False
        or payload.get("policy_rollout_performed") is not False
    ):
        raise Stage10ActiveError("state payload header differs")
    expected = {
        "task_id": tuple(item.task_id for item in schedule),
        "replicate_id": tuple(item.replicate_id for item in schedule),
        "state_seed": tuple(item.state_seed for item in schedule),
        "policy_seed": tuple(item.policy_seed for item in schedule),
    }
    for name, values in expected.items():
        if _strict_int_sequence(payload.get(name), name) != values:
            raise Stage10ActiveError(f"state payload {name} differs")
    if tuple(payload.get("cluster_keys", ())) != tuple(
        item.cluster_key for item in schedule
    ):
        raise Stage10ActiveError("state payload cluster keys differ")
    observed_orders = tuple(
        tuple(item) if isinstance(item, Sequence) else ()
        for item in payload.get("arm_orders", ())
    )
    if observed_orders != tuple(item.arm_order for item in schedule):
        raise Stage10ActiveError("state payload arm orders differ")

    records = attestation.get("records")
    digests = payload.get("state_sha256")
    states = payload.get("states")
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in (records, digests, states)
    ) or not (len(records) == len(digests) == len(states) == TRIPLET_COUNT):
        raise Stage10ActiveError("state payload record coverage differs")

    result: list[np.ndarray] = []
    for ordinal, (spec, record, expected_digest, state) in enumerate(
        zip(schedule, records, digests, states, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise Stage10ActiveError("state attestation record is not an object")
        array = (
            state.detach().cpu().numpy()
            if hasattr(state, "detach") and hasattr(state, "numpy")
            else np.asarray(state)
        )
        canonical, _, observed_digest = canonical_state_bytes(array)
        required = {
            "task_id": spec.task_id,
            "replicate_id": spec.replicate_id,
            "cluster_key": spec.cluster_key,
            "arm_order": list(spec.arm_order),
            "state_seed": spec.state_seed,
            "policy_seed": spec.policy_seed,
            "state_dimension": int(canonical.size),
            "state_sha256": observed_digest,
        }
        if any(record.get(name) != value for name, value in required.items()):
            raise Stage10ActiveError(
                f"state payload/attestation mismatch at ordinal {ordinal}"
            )
        if expected_digest != observed_digest:
            raise Stage10ActiveError(
                f"state payload digest mismatch at ordinal {ordinal}"
            )
        result.append(canonical.copy())
    return tuple(result)


def load_bound_state(
    repo_root: str | Path, *, task_id: int, replicate_id: int
) -> tuple[FreshTripletSpec, np.ndarray, dict[str, Any]]:
    """Open the exact bound payload and return one fully revalidated state."""

    root = Path(repo_root).resolve(strict=True)
    local = validate_local_state_artifacts(root)
    import torch

    payload = torch.load(
        local["payload_path"], map_location="cpu", weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise Stage10ActiveError("state payload must deserialize to an object")
    schedule = load_schedule(root)
    states = validate_state_payload_mapping(
        schedule, payload, local["attestation"]
    )
    selected = select_triplet(root, task_id, replicate_id)
    ordinal = schedule.index(selected)
    _, _, digest = canonical_state_bytes(states[ordinal])
    attestation_record = local["attestation"]["records"][ordinal]
    binding = local["binding"]["local_state_payload"]
    if digest != attestation_record["state_sha256"]:
        raise Stage10ActiveError("selected state SHA differs after payload load")
    audit = {
        "payload_path": str(local["payload_path"].relative_to(root)),
        "payload_sha256": binding["sha256"],
        "payload_bytes": binding["bytes"],
        "attestation_path": str(local["attestation_path"].relative_to(root)),
        "attestation_sha256": local["binding"]["local_state_attestation"][
            "sha256"
        ],
        "source_generation_commit": local["binding"][
            "source_generation_commit"
        ],
        "state_sha256": digest,
        "state_dimension": int(states[ordinal].size),
        "all_payload_records_validated": len(states),
    }
    return selected, states[ordinal], audit


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ActiveError(f"JSON object required: {path}")
    return dict(value)


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    records = []
    for ordinal, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise Stage10ActiveError(f"JSONL line {ordinal} is not an object")
        records.append(dict(value))
    return tuple(records)


def load_runner_readiness(repo_root: str | Path) -> dict[str, Any]:
    """Validate the tracked readiness and every code/artifact binding."""

    root = Path(repo_root).resolve(strict=True)
    path = root / RUNNER_READINESS_RELATIVE_PATH
    readiness = _read_object(path)
    try:
        tracked = subprocess.check_output(
            ["git", "show", f"HEAD:{RUNNER_READINESS_RELATIVE_PATH.as_posix()}"],
            cwd=root,
        )
    except subprocess.CalledProcessError as error:
        raise Stage10ActiveError("runner readiness is not tracked at HEAD") from error
    if tracked != path.read_bytes():
        raise Stage10ActiveError("runner readiness differs from tracked HEAD")
    if (
        readiness.get("schema_version") != RUNNER_READINESS_SCHEMA
        or readiness.get("status") != RUNNER_READINESS_STATUS
        or readiness.get("source_worktree_dirty") is not False
        or readiness.get("protocol_sha256") != PROTOCOL_SHA256
        or readiness.get("schedule_sha256") != SCHEDULE_SHA256
        or readiness.get("access_ledger", {}).get("fresh_state_payload_opened")
        is not False
        or readiness.get("access_ledger", {}).get("active_rollouts") != 0
        or not all(readiness.get("checks", {}).values())
    ):
        raise Stage10ActiveError("runner readiness semantics differ")
    bound_code = readiness.get("bound_code_sha256")
    bound_artifacts = readiness.get("bound_artifacts")
    if not isinstance(bound_code, Mapping) or not isinstance(
        bound_artifacts, Mapping
    ):
        raise Stage10ActiveError("runner readiness bindings are missing")
    for relative, digest in bound_code.items():
        if sha256_file(root / str(relative)) != digest:
            raise Stage10ActiveError(f"runner code changed: {relative}")
    for relative, evidence in bound_artifacts.items():
        target = root / str(relative)
        if not isinstance(evidence, Mapping) or not target.is_file():
            raise Stage10ActiveError(f"runner artifact is missing: {relative}")
        stat = target.stat()
        if (
            stat.st_size != evidence.get("bytes")
            or stat.st_mtime_ns != evidence.get("mtime_ns")
            or stat.st_ino != evidence.get("inode")
        ):
            raise Stage10ActiveError(f"runner artifact identity changed: {relative}")
        if evidence.get("rehash_each_preflight") is True and sha256_file(
            target
        ) != evidence.get("sha256"):
            raise Stage10ActiveError(f"runner artifact SHA changed: {relative}")
    for relative, digest in PROTECTED_CODE_SHA256.items():
        if sha256_file(root / relative) != digest:
            raise Stage10ActiveError(f"protected code changed: {relative}")
    return readiness


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise Stage10ActiveError(f"{name} must be a non-negative integer")
    return value


def summarize_policy_records(
    records: Sequence[Mapping[str, Any]],
    *,
    spec: ActiveArmSpec,
) -> dict[str, Any]:
    if not records:
        raise Stage10ActiveError("completed arm has no policy telemetry")
    layers: Counter[int] = Counter()
    fm_calls = 0
    fm_steps = 0
    route_exact = 0
    previous_step = -1
    for record in records:
        if (
            record.get("episode_id") != spec.cluster_key
            or record.get("task_id") != spec.task_id
            or record.get("schema_version") != "phase-route-vla.telemetry.v1"
        ):
            raise Stage10ActiveError("policy telemetry identity differs")
        step = _nonnegative_int(record.get("step_id"), "step_id")
        if step <= previous_step:
            raise Stage10ActiveError("policy telemetry steps are not increasing")
        previous_step = step
        layer = _nonnegative_int(record.get("exit_layer"), "exit_layer")
        if layer not in METHOD_LAYERS[spec.method]:
            raise Stage10ActiveError("selected layer differs from frozen method")
        layers[layer] += 1
        fm_calls += _nonnegative_int(record.get("fm_calls"), "fm_calls")
        fm_steps += _nonnegative_int(record.get("fm_steps_total"), "fm_steps")
        if spec.method == "route_first_stage8":
            events = record.get("extra", {}).get("exit_events", [])
            if not isinstance(events, Sequence):
                raise Stage10ActiveError("route-first events are missing")
            evaluated = [
                item
                for item in events
                if isinstance(item, Mapping)
                and item.get("event") == "exit_candidate"
                and item.get("evaluated") is True
            ]
            selected = [
                item
                for item in events
                if isinstance(item, Mapping)
                and item.get("event") == "route_first_selected_action"
            ]
            decisions = [
                item
                for item in events
                if isinstance(item, Mapping)
                and item.get("event") == "phase_route_decision"
            ]
            errors = [
                item
                for item in events
                if isinstance(item, Mapping)
                and item.get("event")
                in ("route_first_action_error", "route_first_action_rejected")
            ]
            exact = bool(
                len(evaluated) == len(selected) == len(decisions) == 1
                and evaluated[0].get("layer_idx") == layer
                and evaluated[0].get("should_exit") is True
                and evaluated[0].get("fm_calls") == 1
                and selected[0].get("layer_idx") == layer
                and selected[0].get("fm_calls") == 1
                and selected[0].get("fail_reason") is None
                and decisions[0].get("selected_layer") == layer
                and decisions[0].get("fm_calls") == 1
                and not errors
            )
            if not exact:
                raise Stage10ActiveError(
                    "route-first valid call does not contain exactly one FM"
                )
            route_exact += 1
    return {
        "policy_calls": len(records),
        "telemetry_fm_calls": fm_calls,
        "telemetry_fm_steps": fm_steps,
        "selected_layer_counts": {
            f"L{layer}": layers[layer] for layer in METHOD_LAYERS[spec.method]
        },
        "route_exactly_one_fm_calls": route_exact,
        "route_exactly_one_fm_fraction": (
            route_exact / len(records)
            if spec.method == "route_first_stage8"
            else None
        ),
    }


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize_measurement_records(
    records: Sequence[Mapping[str, Any]],
    *,
    spec: ActiveArmSpec,
    expected_policy_calls: int,
) -> dict[str, Any]:
    if len(records) != expected_policy_calls or not records:
        raise Stage10ActiveError("measurement count differs from policy calls")
    latency = []
    for record in records:
        context = record.get("context", {})
        value = record.get("policy_wall_latency_ms")
        if (
            record.get("schema_version")
            != "phase-route-vla.stage1.measurement.v1"
            or record.get("measurement_is_control_input") is not False
            or record.get("d9_protected_source_modified") is not False
            or record.get("mode") != MEASUREMENT_MODES[spec.method]
            or context.get("episode_id") != spec.cluster_key
            or context.get("task_id") != spec.task_id
            or record.get("error") is not None
            or record.get("action_finite") is not True
            or record.get("action_shape") != [8, 7]
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise Stage10ActiveError("Stage 10 measurement record differs")
        selected = record.get("selected_layer")
        if spec.method != "original_a1" and selected not in METHOD_LAYERS[spec.method]:
            raise Stage10ActiveError("measurement selected layer differs")
        latency.append(float(value))
    latency.sort()
    return {
        "records": len(latency),
        "mean_ms": math.fsum(latency) / len(latency),
        "p50_ms": _nearest_rank(latency, 0.50),
        "p90_ms": _nearest_rank(latency, 0.90),
        "p95_ms": _nearest_rank(latency, 0.95),
        "max_ms": latency[-1],
    }


def validate_triplet_record(
    record: Mapping[str, Any], *, spec: FreshTripletSpec
) -> None:
    if (
        record.get("schema_version") != ACTIVE_TRIPLET_SCHEMA
        or record.get("status") != "COMPLETE_ROUTE_FIRST_STAGE10_TRIPLET"
        or record.get("task_id") != spec.task_id
        or record.get("replicate_id") != spec.replicate_id
        or record.get("cluster_key") != spec.cluster_key
        or record.get("state_seed") != spec.state_seed
        or record.get("policy_seed") != spec.policy_seed
        or record.get("arm_order") != list(spec.arm_order)
    ):
        raise Stage10ActiveError("triplet identity differs")
    arms = record.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(METHODS):
        raise Stage10ActiveError("triplet arm coverage differs")
    if any(not isinstance(item, Mapping) for item in arms.values()):
        raise Stage10ActiveError("triplet arm evidence must be an object")
    commits = {item.get("source_git_commit") for item in arms.values()}
    gpu_uuids = {
        normalize_gpu_uuid(item.get("gpu_uuid")) for item in arms.values()
    }
    state_sha = {item.get("state_sha256") for item in arms.values()}
    seeds = {item.get("policy_seed") for item in arms.values()}
    if any(
        not isinstance(item, Mapping)
        or item.get("method") != method
        or item.get("arm_position") != spec.arm_order.index(method) + 1
        or type(item.get("success")) is not bool
        or item.get("evidence_valid") is not True
        for method, item in arms.items()
    ) or not (
        len(commits) == len(gpu_uuids) == len(state_sha) == len(seeds) == 1
        and next(iter(seeds)) == spec.policy_seed
    ):
        raise Stage10ActiveError("within-triplet pairing differs")


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def aggregate_triplets(
    schedule: Sequence[FreshTripletSpec],
    triplets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the preregistered Stage 10 gate only on complete evidence."""

    if len(schedule) != TRIPLET_COUNT or len(triplets) != TRIPLET_COUNT:
        raise Stage10ActiveError("all 60 Stage 10 triplets are required")
    by_key = {str(item.get("cluster_key")): item for item in triplets}
    if set(by_key) != {item.cluster_key for item in schedule}:
        raise Stage10ActiveError("aggregate triplet coverage differs")
    successes = Counter({method: 0 for method in METHODS})
    layers = {method: Counter() for method in METHODS}
    policy_calls = Counter({method: 0 for method in METHODS})
    route_candidate_ratios = []
    route_original_ratios = []
    discordance = {
        "route_success_candidate_failure": 0,
        "route_failure_candidate_success": 0,
        "route_success_original_failure": 0,
        "route_failure_original_success": 0,
    }
    per_task = {
        str(task): {method: 0 for method in METHODS} for task in range(10)
    }
    all_arms = 0
    route_exact_calls = 0
    for spec in schedule:
        record = by_key[spec.cluster_key]
        validate_triplet_record(record, spec=spec)
        arms = record["arms"]
        all_arms += len(arms)
        for method in METHODS:
            arm = arms[method]
            successes[method] += int(arm["success"])
            per_task[str(spec.task_id)][method] += int(arm["success"])
            policy_calls[method] += int(arm["policy_calls"])
            layers[method].update(arm["selected_layer_counts"])
        route = arms["route_first_stage8"]
        candidate = arms["candidate_first_v3"]
        original = arms["original_a1"]
        for denominator in (candidate, original):
            if float(denominator["policy_p50_ms"]) <= 0.0:
                raise Stage10ActiveError("latency ratio denominator is not positive")
        route_candidate_ratios.append(
            float(route["policy_p50_ms"]) / float(candidate["policy_p50_ms"])
        )
        route_original_ratios.append(
            float(route["policy_p50_ms"]) / float(original["policy_p50_ms"])
        )
        discordance["route_success_candidate_failure"] += int(
            route["success"] and not candidate["success"]
        )
        discordance["route_failure_candidate_success"] += int(
            not route["success"] and candidate["success"]
        )
        discordance["route_success_original_failure"] += int(
            route["success"] and not original["success"]
        )
        discordance["route_failure_original_success"] += int(
            not route["success"] and original["success"]
        )
        route_exact_calls += int(route["route_exactly_one_fm_calls"])
    route_candidate_median = _median(route_candidate_ratios)
    route_original_median = _median(route_original_ratios)
    gates = {
        "complete_60_triplets": len(triplets) == TRIPLET_COUNT,
        "complete_180_active_rollouts": all_arms == ACTIVE_ROLLOUT_COUNT,
        "route_success_at_least_candidate_minus_6": (
            successes["route_first_stage8"]
            >= successes["candidate_first_v3"] - 6
        ),
        "route_success_at_least_original_a1_minus_6": (
            successes["route_first_stage8"] >= successes["original_a1"] - 6
        ),
        "route_candidate_episode_p50_ratio_median_at_most_0_80": (
            route_candidate_median <= 0.80
        ),
        "route_original_episode_p50_ratio_median_at_most_0_90": (
            route_original_median <= 0.90
        ),
        "route_every_valid_policy_call_exactly_one_fm": (
            route_exact_calls == policy_calls["route_first_stage8"]
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": ACTIVE_AGGREGATE_SCHEMA,
        "status": (
            "PASS_ROUTE_FIRST_STAGE10_FRESH_ACTIVE_CONFIRMATION"
            if passed
            else "INCOMPLETE_ROUTE_FIRST_STAGE10_FRESH_ACTIVE_CONFIRMATION"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "triplets": len(triplets),
        "active_rollouts": all_arms,
        "success_counts": dict(successes),
        "per_task_success_counts": per_task,
        "policy_calls": dict(policy_calls),
        "selected_layer_counts": {
            method: dict(layers[method]) for method in METHODS
        },
        "within_triplet_latency_ratios": {
            "route_to_candidate_episode_p50_median": route_candidate_median,
            "route_to_original_a1_episode_p50_median": route_original_median,
        },
        "paired_success_discordance": discordance,
        "route_exactly_one_fm_calls": route_exact_calls,
        "gates": gates,
        "claim_boundary": {
            "powered_noninferiority": False,
            "statistical_superiority": False,
            "system_wide_speedup": False,
            "cross_suite_generalization": False,
            "deployment_authorized": False,
        },
    }


__all__ = [
    "ACTIVE_AGGREGATE_SCHEMA",
    "ACTIVE_ARM_SCHEMA",
    "ACTIVE_OUTPUT_RELATIVE_PATH",
    "ACTIVE_TRIPLET_SCHEMA",
    "ARM_ATTESTATION_SCHEMA",
    "ActiveArmSpec",
    "METHOD_LAYERS",
    "MINIMUM_FREE_MEMORY_MIB",
    "POSTFLIGHT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "PROTECTED_CODE_SHA256",
    "RUNNER_READINESS_RELATIVE_PATH",
    "RUNNER_READINESS_SCHEMA",
    "RUNNER_READINESS_STATUS",
    "Stage10ActiveError",
    "aggregate_triplets",
    "expected_triplet_directory",
    "git_output",
    "load_bound_state",
    "load_runner_readiness",
    "normalize_gpu_uuid",
    "read_jsonl",
    "select_arm",
    "select_triplet",
    "summarize_measurement_records",
    "summarize_policy_records",
    "validate_state_payload_mapping",
    "validate_triplet_record",
]
