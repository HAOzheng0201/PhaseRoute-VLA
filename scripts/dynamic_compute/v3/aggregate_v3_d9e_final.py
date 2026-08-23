#!/usr/bin/env python3
"""Run the single authorized D9E success/efficiency/safety aggregate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.independent_test_aggregate import (  # noqa: E402
    ArmEvidence,
    PairEvidence,
    PhaseCallEvidence,
    TruthEvidence,
    aggregate_independent_test,
)
from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    D9_ARMS,
    D9_CONTRACT_SHA256,
    D9_RECORD_COUNT,
    D9_TASK_IDS,
    load_d9_contract,
    load_d9_selection_metadata,
)
from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    D9C_ARM_SCHEMA_VERSION,
    D9C_COLLECTION_SCHEMA_VERSION,
    D9C_COLLECTION_STATUS,
    D9C_OUTPUT_RELATIVE_PATH,
    ORIGINAL_A1_ARM,
    PHASE_ROUTE_ARM,
    read_json_object,
    read_jsonl,
    sha256_file,
    summarize_policy_telemetry,
    validate_pair_record,
)
from a1.vla.dynamic_compute.v3.same_noise_replay import (  # noqa: E402
    D9C_COLLECTION_SHA256,
    D9D_ACTION_THRESHOLD,
    D9D_COLLECTION_SCHEMA_VERSION,
    D9D_COLLECTION_STATUS,
    D9D_EXPECTED_ROWS,
    D9D_REPLAY_LAYERS,
    D9D_SEVERE_RATIO,
    D9D_SHARD_COUNT,
    D9D_SHARD_SCHEMA_VERSION,
)


D9C_ATTESTATION = Path("results/v3/v3_d9c_collection_attestation.json")
D9D_ATTESTATION = Path("results/v3/v3_d9d_collection_attestation.json")
D9D_ATTESTATION_SHA256 = (
    "f8b3421948ca6c8ccfda6837afde9cfec0a7dbd6cee61987eb03e2dee2f6ea65"
)
D9E_READINESS = Path("results/v3/v3_d9e_runner_readiness.json")
D9E_READINESS_STATUS = "PASS_V3_D9E_FROZEN_AGGREGATE_RUNNER_READINESS"
REPORT_OUTPUT = Path("reports/v3_d9e_final")
FORMAL_OUTPUT = Path("results/v3/v3_d9_final_result.json")
FORMAL_SCHEMA_VERSION = "phase-route-vla.v3.d9-final-result.v1"
REPORT_SCHEMA_VERSION = "phase-route-vla.v3.d9e-aggregate-report.v1"
PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d9e-aggregate-payload.v1"


class D9EInputError(PermissionError):
    """Raised when authenticated D9 inputs or one-shot constraints differ."""


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sidecar_digest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    try:
        expected = sidecar.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError) as error:
        raise D9EInputError(f"missing SHA-256 sidecar: {path}") from error
    observed = sha256_file(path)
    if expected != observed:
        raise D9EInputError(f"SHA-256 sidecar differs: {path}")
    return observed


def _safe_child(root: Path, relative: Any, *, context: str) -> Path:
    if type(relative) is not str:
        raise D9EInputError(f"{context} path must be a string")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or value.as_posix() != relative:
        raise D9EInputError(f"{context} path is unsafe")
    target = (root / value).resolve(strict=True)
    if root not in target.parents or not target.is_file() or target.is_symlink():
        raise D9EInputError(f"{context} is not a regular child file")
    return target


def _readiness() -> dict[str, Any]:
    path = REPO_ROOT / D9E_READINESS
    value = read_json_object(path)
    tracked = subprocess.check_output(
        ["git", "show", f"HEAD:{D9E_READINESS.as_posix()}"], cwd=REPO_ROOT
    )
    if tracked != path.read_bytes():
        raise D9EInputError("D9E readiness must be tracked exactly at HEAD")
    if (
        value.get("status") != D9E_READINESS_STATUS
        or not all(value.get("checks", {}).values())
        or value.get("access_ledger", {}).get("D9C_success_values_opened")
        is not False
        or value.get("access_ledger", {}).get("D9D_truth_payloads_opened") != 0
        or value.get("authorization", {}).get("next_stage")
        != "D9E_ONE_SHOT_FINAL_AGGREGATE"
    ):
        raise D9EInputError("D9E runner readiness semantics differ")
    bound = value.get("bound_code_sha256")
    if not isinstance(bound, Mapping) or not bound:
        raise D9EInputError("D9E readiness code binding is missing")
    for relative, expected in bound.items():
        if sha256_file(REPO_ROOT / str(relative)) != expected:
            raise D9EInputError(f"D9E frozen code changed: {relative}")
    return {
        "path": D9E_READINESS.as_posix(),
        "sha256": _sidecar_digest(path),
        "implementation_commit": value["source_git_commit"],
        "bound_code_files": len(bound),
    }


def _authenticate_attestations() -> tuple[dict[str, Any], dict[str, Any]]:
    load_d9_contract(REPO_ROOT)
    d9c_path = REPO_ROOT / D9C_ATTESTATION
    d9d_path = REPO_ROOT / D9D_ATTESTATION
    d9c_sha = _sidecar_digest(d9c_path)
    d9d_sha = _sidecar_digest(d9d_path)
    d9c = read_json_object(d9c_path)
    d9d = read_json_object(d9d_path)
    if (
        d9c_sha != D9C_COLLECTION_SHA256
        or d9c.get("status") != D9C_COLLECTION_STATUS
        or d9c.get("schema_version") != D9C_COLLECTION_SCHEMA_VERSION
        or d9c.get("completeness", {}).get("pairs") != D9_RECORD_COUNT
        or d9c.get("completeness", {}).get("rollouts") != 2 * D9_RECORD_COUNT
        or d9c.get("completeness", {}).get("PhaseRoute_same_noise_cache_shards")
        != D9D_EXPECTED_ROWS
        or d9c.get("claim_boundary", {}).get("D9_primary_gate_evaluated")
        is not None
        or d9c.get("claim_boundary", {}).get("D9C_is_D9_pass_or_negative")
        is not False
    ):
        raise D9EInputError("D9C collection attestation differs")
    # D9C uses the explicit absence of a D9 gate field plus its frozen claim
    # boundary; D9D carries the one-shot D9E authorization.
    if (
        d9d_sha != D9D_ATTESTATION_SHA256
        or d9d.get("status") != D9D_COLLECTION_STATUS
        or d9d.get("schema_version") != D9D_COLLECTION_SCHEMA_VERSION
        or d9d.get("D9C_collection", {}).get("sha256") != D9C_COLLECTION_SHA256
        or d9d.get("completeness", {}).get("shards") != D9D_SHARD_COUNT
        or d9d.get("completeness", {}).get("policy_call_truth_rows")
        != D9D_EXPECTED_ROWS
        or d9d.get("completeness", {}).get("candidate_layers")
        != list(D9D_REPLAY_LAYERS)
        or not all(d9d.get("checks", {}).values())
        or d9d.get("authorization", {}).get(
            "D9E_one_shot_aggregate_authorized"
        )
        is not True
        or d9d.get("authorization", {}).get(
            "additional_test_tuning_or_second_independent_test"
        )
        is not False
        or d9d.get("claim_boundary", {}).get("D9_primary_gate_evaluated")
        is not False
    ):
        raise D9EInputError("D9D collection attestation differs")
    return d9c, d9d


def _exit_counts(value: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for name, count in value.items():
        if type(name) is not str or not name.startswith("L"):
            raise D9EInputError("invalid exit-layer name")
        try:
            layer = int(name[1:])
        except ValueError as error:
            raise D9EInputError("invalid exit-layer name") from error
        if type(count) is not int or count < 0:
            raise D9EInputError("invalid exit-layer count")
        result.append((layer, count))
    return tuple(sorted(result))


def _load_arm(
    arm_dir: Path,
    *,
    arm: str,
    record: Any,
    expected_binding: Mapping[str, Any],
) -> tuple[ArmEvidence, tuple[PhaseCallEvidence, ...], dict[str, Any]]:
    result_path = arm_dir / "result.json"
    result_sha = _sidecar_digest(result_path)
    result = read_json_object(result_path)
    if (
        result_sha != expected_binding.get("result_sha256")
        or result.get("status") != "COMPLETE_V3_D9C_ARM_ROLLOUT"
        or result.get("schema_version") != D9C_ARM_SCHEMA_VERSION
        or result.get("arm") != arm
        or result.get("canonical_key") != record.canonical_key
        or result.get("task_id") != record.task_id
        or result.get("episode_index") != record.episode_index
        or result.get("seed") != record.seed
        or result.get("source_worktree_dirty") is not False
        or result.get("gpu", {}).get("physical_index") != record.task_id % 4
        or result.get("gpu", {}).get("visible_count") != 1
        or result.get("claim_boundary", {}).get("cross_pair_aggregate_computed")
        is not False
        or result.get("claim_boundary", {}).get("D9_primary_gate_evaluated")
        is not False
        or type(result.get("success")) is not bool
    ):
        raise D9EInputError(f"D9C arm result differs: {arm_dir}")
    telemetry_info = result.get("telemetry", {})
    telemetry_path = _safe_child(
        arm_dir, telemetry_info.get("path"), context="policy telemetry"
    )
    telemetry = read_jsonl(telemetry_path)
    if (
        sha256_file(telemetry_path) != telemetry_info.get("sha256")
        or len(telemetry) != telemetry_info.get("records")
    ):
        raise D9EInputError("D9C policy telemetry binding differs")
    policy = summarize_policy_telemetry(
        telemetry,
        arm=arm,
        expected_episode_id=record.canonical_key,
        expected_task_id=record.task_id,
    )
    if policy != result.get("policy_accounting"):
        raise D9EInputError("D9C policy accounting recomputation differs")
    evidence = ArmEvidence(
        success=result["success"],
        environment_steps=int(result["environment_steps"]),
        policy_calls=int(policy["policy_calls"]),
        fm_calls=int(policy["fm_calls"]),
        fm_steps=int(policy["fm_steps"]),
        exit_layer_counts=_exit_counts(policy["exit_layer_counts"]),
        policy_wall_seconds=float(result["policy_wall_seconds"]),
        rollout_wall_seconds=float(result["rollout_wall_seconds"]),
        policy_latency_ms=tuple(float(item["latency_ms"]) for item in telemetry),
    )
    phase_calls: list[PhaseCallEvidence] = []
    runtime_sha = None
    if arm == PHASE_ROUTE_ARM:
        runtime_info = result.get("phase_route_runtime", {})
        runtime_path = _safe_child(
            arm_dir, runtime_info.get("path"), context="PhaseRoute runtime"
        )
        runtime = read_jsonl(runtime_path)
        runtime_sha = sha256_file(runtime_path)
        if (
            runtime_sha != runtime_info.get("sha256")
            or len(runtime) != evidence.policy_calls
            or runtime_info.get("records") != evidence.policy_calls
        ):
            raise D9EInputError("D9C PhaseRoute runtime binding differs")
        for ordinal, item in enumerate(runtime):
            context = item.get("context", {})
            events = item.get("events")
            errors = item.get("errors")
            if (
                context.get("episode_id") != record.canonical_key
                or context.get("task_id") != record.task_id
                or context.get("call_ordinal") != ordinal
                or type(context.get("step_id")) is not int
                or item.get("prepared") is not True
                or item.get("committed") is not True
                or item.get("selected_layer") not in D9D_REPLAY_LAYERS
                or not isinstance(events, Sequence)
                or isinstance(events, (str, bytes))
                or not isinstance(errors, Sequence)
                or isinstance(errors, (str, bytes))
            ):
                raise D9EInputError("D9C PhaseRoute runtime semantics differ")
            candidates = [
                event
                for event in events
                if isinstance(event, Mapping)
                and event.get("event") == "phase_route_candidate"
            ]
            decisions = [
                event
                for event in events
                if isinstance(event, Mapping)
                and event.get("event") == "phase_route_decision"
            ]
            if len(decisions) != 1 or decisions[0].get("selected_layer") != item.get(
                "selected_layer"
            ):
                raise D9EInputError("D9C PhaseRoute decision event differs")
            head_ranges: list[float | None] = []
            for candidate in candidates:
                raw = candidate.get("full_action_head_range")
                if raw is None:
                    head_ranges.append(None)
                else:
                    value = float(raw)
                    if not math.isfinite(value) or value < 0:
                        raise D9EInputError("D9C head range is invalid")
                    head_ranges.append(value)
            phase_calls.append(
                PhaseCallEvidence(
                    call_ordinal=ordinal,
                    step_id=int(context["step_id"]),
                    selected_layer=int(item["selected_layer"]),
                    head_ranges=tuple(head_ranges),
                    prepare_latency_ms=float(item["prepare_latency_ms"]),
                    fail_closed_errors=len(errors),
                )
            )
    elif result.get("phase_route_runtime") is not None:
        raise D9EInputError("A1 arm unexpectedly has PhaseRoute runtime")
    evidence.validate(phase_route=arm == PHASE_ROUTE_ARM)
    return evidence, tuple(phase_calls), {
        "result_sha256": result_sha,
        "telemetry_sha256": telemetry_info["sha256"],
        "runtime_sha256": runtime_sha,
        "gpu_uuid": result["gpu"]["uuid"],
        "source_git_commit": result["source_git_commit"],
    }


def _load_pairs(d9c: Mapping[str, Any]) -> tuple[list[PairEvidence], list[dict[str, Any]]]:
    schedule = load_d9_selection_metadata(REPO_ROOT)
    pair_bindings = d9c.get("pair_record_sha256")
    arm_bindings = d9c.get("arm_payload_binding")
    if (
        not isinstance(pair_bindings, Mapping)
        or len(pair_bindings) != D9_RECORD_COUNT
        or not isinstance(arm_bindings, Mapping)
        or len(arm_bindings) != D9_RECORD_COUNT
    ):
        raise D9EInputError("D9C raw binding inventory differs")
    pairs: list[PairEvidence] = []
    records: list[dict[str, Any]] = []
    for record in schedule:
        pair_dir = (
            REPO_ROOT
            / D9C_OUTPUT_RELATIVE_PATH
            / f"task{record.task_id}"
            / f"pair_episode{record.episode_index}"
        ).resolve(strict=True)
        pair_path = pair_dir / "pair_record.json"
        pair_sha = _sidecar_digest(pair_path)
        pair_value = read_json_object(pair_path)
        validate_pair_record(pair_value, record=record)
        if pair_sha != pair_bindings.get(record.canonical_key):
            raise D9EInputError("D9C pair attestation binding differs")
        expected_arms = arm_bindings.get(record.canonical_key)
        if not isinstance(expected_arms, Mapping) or set(expected_arms) != set(D9_ARMS):
            raise D9EInputError("D9C arm attestation binding differs")
        a1, a1_calls, a1_audit = _load_arm(
            pair_dir / ORIGINAL_A1_ARM,
            arm=ORIGINAL_A1_ARM,
            record=record,
            expected_binding=expected_arms[ORIGINAL_A1_ARM],
        )
        if a1_calls:
            raise D9EInputError("original A1 produced PhaseRoute call evidence")
        phase, phase_calls, phase_audit = _load_arm(
            pair_dir / PHASE_ROUTE_ARM,
            arm=PHASE_ROUTE_ARM,
            record=record,
            expected_binding=expected_arms[PHASE_ROUTE_ARM],
        )
        if pair_value["arms"][ORIGINAL_A1_ARM]["result_sha256"] != a1_audit[
            "result_sha256"
        ] or pair_value["arms"][PHASE_ROUTE_ARM]["result_sha256"] != phase_audit[
            "result_sha256"
        ]:
            raise D9EInputError("pair-to-arm result binding differs")
        pair = PairEvidence(
            canonical_key=record.canonical_key,
            task_id=record.task_id,
            episode_index=record.episode_index,
            seed=record.seed,
            a1=a1,
            phase_route=phase,
            phase_calls=phase_calls,
        )
        pair.validate()
        pairs.append(pair)
        records.append(
            {
                "canonical_key": record.canonical_key,
                "task_id": record.task_id,
                "episode_index": record.episode_index,
                "seed": record.seed,
                "arm_order": list(record.arm_order),
                "A1": {
                    "success": a1.success,
                    "environment_steps": a1.environment_steps,
                    "policy_calls": a1.policy_calls,
                    "FM_calls": a1.fm_calls,
                    "FM_steps": a1.fm_steps,
                    "exit_layer_counts": dict(a1.exit_layer_counts),
                    "policy_wall_seconds": a1.policy_wall_seconds,
                    "rollout_wall_seconds": a1.rollout_wall_seconds,
                    **a1_audit,
                },
                "PhaseRoute": {
                    "success": phase.success,
                    "environment_steps": phase.environment_steps,
                    "policy_calls": phase.policy_calls,
                    "FM_calls": phase.fm_calls,
                    "FM_steps": phase.fm_steps,
                    "exit_layer_counts": dict(phase.exit_layer_counts),
                    "early_exit_calls": sum(
                        call.selected_layer in (11, 13) for call in phase_calls
                    ),
                    "policy_wall_seconds": phase.policy_wall_seconds,
                    "rollout_wall_seconds": phase.rollout_wall_seconds,
                    **phase_audit,
                },
            }
        )
    return pairs, records


def _tensor(
    payload: Mapping[str, Any], name: str, rows: int, dtype: torch.dtype
) -> torch.Tensor:
    value = payload.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.shape != (rows,)
        or value.dtype != dtype
    ):
        raise D9EInputError(f"D9D tensor differs: {name}")
    return value.contiguous()


def _load_truths(d9d: Mapping[str, Any]) -> list[TruthEvidence]:
    bindings = d9d.get("shard_binding")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        str(index) for index in range(D9D_SHARD_COUNT)
    }:
        raise D9EInputError("D9D shard binding differs")
    indexed: dict[int, TruthEvidence] = {}
    for shard in range(D9D_SHARD_COUNT):
        binding = bindings[str(shard)]
        if not isinstance(binding, Mapping) or binding.get("rows") != 925:
            raise D9EInputError("D9D shard metadata differs")
        relative = binding.get("payload_path")
        if type(relative) is not str:
            raise D9EInputError("D9D payload path differs")
        path = (REPO_ROOT / relative).resolve(strict=True)
        if REPO_ROOT not in path.parents or path.is_symlink() or not path.is_file():
            raise D9EInputError("D9D payload path is unsafe")
        if sha256_file(path) != binding.get("payload_sha256"):
            raise D9EInputError("D9D payload SHA-256 differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != D9D_SHARD_SCHEMA_VERSION
            or payload.get("role") != "D9D_same_noise_truth_only"
            or payload.get("D9C_collection_sha256") != D9C_COLLECTION_SHA256
            or payload.get("shard_index") != shard
            or payload.get("shard_count") != D9D_SHARD_COUNT
            or payload.get("full_action_threshold") != D9D_ACTION_THRESHOLD
            or payload.get("severe_ratio") != D9D_SEVERE_RATIO
            or payload.get("layer27_is_consistency_teacher_only") is not True
            or payload.get("active_control") is not False
            or payload.get("D9_gate_evaluated") is not False
        ):
            raise D9EInputError("D9D payload semantics differ")
        rows = int(binding["rows"])
        indices = _tensor(payload, "dataset_index", rows, torch.int64)
        tasks = _tensor(payload, "task_id", rows, torch.int64)
        episodes = _tensor(payload, "episode_index", rows, torch.int64)
        ordinals = _tensor(payload, "call_ordinal", rows, torch.int64)
        steps = _tensor(payload, "step_id", rows, torch.int64)
        layers = _tensor(payload, "selected_layer", rows, torch.int64)
        distance = _tensor(payload, "full_action_distance", rows, torch.float64)
        full = _tensor(payload, "full_action_unsafe", rows, torch.bool)
        gripper = _tensor(payload, "gripper_unsafe", rows, torch.bool)
        severe = _tensor(payload, "severe_full_action", rows, torch.bool)
        errors = _tensor(
            payload, "selected_replay_max_abs_error", rows, torch.float64
        )
        keys = payload.get("canonical_keys")
        if not isinstance(keys, list) or len(keys) != rows:
            raise D9EInputError("D9D canonical-key geometry differs")
        if (
            not bool(torch.isfinite(distance).all())
            or not bool(torch.isfinite(errors).all())
            or not bool((indices.remainder(D9D_SHARD_COUNT) == shard).all())
        ):
            raise D9EInputError("D9D truth numeric integrity differs")
        for row in range(rows):
            index = int(indices[row])
            if index in indexed:
                raise D9EInputError("D9D dataset index is duplicated")
            truth = TruthEvidence(
                canonical_key=str(keys[row]),
                task_id=int(tasks[row]),
                episode_index=int(episodes[row]),
                call_ordinal=int(ordinals[row]),
                step_id=int(steps[row]),
                selected_layer=int(layers[row]),
                full_action_distance=float(distance[row]),
                full_action_unsafe=bool(full[row]),
                gripper_unsafe=bool(gripper[row]),
                severe_full_action=bool(severe[row]),
                selected_replay_max_abs_error=float(errors[row]),
            )
            truth.validate()
            indexed[index] = truth
    if sorted(indexed) != list(range(D9D_EXPECTED_ROWS)):
        raise D9EInputError("D9D global dataset-index coverage differs")
    return [indexed[index] for index in range(D9D_EXPECTED_ROWS)]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for value in values:
            output.write(
                json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise D9EInputError("D9E is CPU-only")
    if torch.cuda.is_initialized():
        raise D9EInputError("D9E must not initialize CUDA")
    if git_output("status", "--porcelain=v1"):
        raise D9EInputError("D9E requires a clean frozen-runner worktree")
    report = REPO_ROOT / REPORT_OUTPUT
    report_incomplete = report.with_name(report.name + ".incomplete")
    formal = REPO_ROOT / FORMAL_OUTPUT
    formal_incomplete = formal.with_suffix(".json.incomplete")
    if any(
        path.exists()
        for path in (
            report,
            report_incomplete,
            formal,
            formal.with_suffix(".sha256"),
            formal_incomplete,
        )
    ):
        raise FileExistsError("D9E refuses to overwrite one-shot evidence")

    source_commit = git_output("rev-parse", "HEAD")
    readiness = _readiness()
    d9c, d9d = _authenticate_attestations()
    pairs, pair_records = _load_pairs(d9c)
    truths = _load_truths(d9d)
    # This is the sole call that computes any cross-pair D9 primary metric.
    aggregate = aggregate_independent_test(pairs, truths)

    early_by_key: dict[str, list[TruthEvidence]] = {}
    for truth in truths:
        if truth.selected_layer in (11, 13):
            early_by_key.setdefault(truth.canonical_key, []).append(truth)
    pair_lookup = {pair.canonical_key: pair for pair in pairs}
    false_records = []
    for key in aggregate["safety"]["false_safe_cluster_keys"]:
        unsafe = [
            truth
            for truth in early_by_key[key]
            if truth.full_action_unsafe or truth.gripper_unsafe
        ]
        pair = pair_lookup[key]
        false_records.append(
            {
                "canonical_key": key,
                "task_id": pair.task_id,
                "episode_index": pair.episode_index,
                "A1_success": pair.a1.success,
                "PhaseRoute_success": pair.phase_route.success,
                "unsafe_early_calls": len(unsafe),
                "full_action_unsafe_calls": sum(t.full_action_unsafe for t in unsafe),
                "gripper_unsafe_calls": sum(t.gripper_unsafe for t in unsafe),
                "severe_full_action_calls": sum(
                    t.severe_full_action for t in unsafe
                ),
                "calls": [
                    {
                        "call_ordinal": truth.call_ordinal,
                        "step_id": truth.step_id,
                        "selected_layer": truth.selected_layer,
                        "full_action_distance": truth.full_action_distance,
                        "full_action_unsafe": truth.full_action_unsafe,
                        "gripper_unsafe": truth.gripper_unsafe,
                        "severe_full_action": truth.severe_full_action,
                    }
                    for truth in unsafe
                ],
            }
        )

    report_incomplete.mkdir(parents=False, exist_ok=False)
    pair_path = report_incomplete / "pair_records.jsonl"
    false_path = report_incomplete / "false_safe_records.jsonl"
    payload_path = report_incomplete / "result_payload.pt"
    _write_jsonl(pair_path, pair_records)
    _write_jsonl(false_path, false_records)
    torch.save(
        {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "role": "frozen_D9E_secondary_analysis_payload",
            "task_id": torch.tensor([pair.task_id for pair in pairs], dtype=torch.long),
            "episode_index": torch.tensor(
                [pair.episode_index for pair in pairs], dtype=torch.long
            ),
            "A1_success": torch.tensor(
                [pair.a1.success for pair in pairs], dtype=torch.bool
            ),
            "PhaseRoute_success": torch.tensor(
                [pair.phase_route.success for pair in pairs], dtype=torch.bool
            ),
            "A1_policy_calls": torch.tensor(
                [pair.a1.policy_calls for pair in pairs], dtype=torch.long
            ),
            "PhaseRoute_policy_calls": torch.tensor(
                [pair.phase_route.policy_calls for pair in pairs], dtype=torch.long
            ),
            "A1_FM_calls": torch.tensor(
                [pair.a1.fm_calls for pair in pairs], dtype=torch.long
            ),
            "PhaseRoute_FM_calls": torch.tensor(
                [pair.phase_route.fm_calls for pair in pairs], dtype=torch.long
            ),
            "truth_task_id": torch.tensor(
                [truth.task_id for truth in truths], dtype=torch.long
            ),
            "truth_episode_index": torch.tensor(
                [truth.episode_index for truth in truths], dtype=torch.long
            ),
            "truth_selected_layer": torch.tensor(
                [truth.selected_layer for truth in truths], dtype=torch.long
            ),
            "truth_full_action_distance": torch.tensor(
                [truth.full_action_distance for truth in truths], dtype=torch.float64
            ),
            "truth_full_action_unsafe": torch.tensor(
                [truth.full_action_unsafe for truth in truths], dtype=torch.bool
            ),
            "truth_gripper_unsafe": torch.tensor(
                [truth.gripper_unsafe for truth in truths], dtype=torch.bool
            ),
            "truth_severe_full_action": torch.tensor(
                [truth.severe_full_action for truth in truths], dtype=torch.bool
            ),
            "canonical_keys": [pair.canonical_key for pair in pairs],
            "truth_canonical_keys": [truth.canonical_key for truth in truths],
            "gate_checks": aggregate["gate_checks"],
            "D9C_collection_sha256": D9C_COLLECTION_SHA256,
            "D9D_collection_sha256": D9D_ATTESTATION_SHA256,
        },
        payload_path,
    )
    report_result = {
        "status": aggregate["status"],
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "source_git_commit": source_commit,
        "source_worktree_dirty_at_entry": False,
        "suite": "libero_10",
        "role": "one_shot_paired_active_independent_test_aggregate",
        **aggregate,
        "input_binding": {
            "D9_contract_sha256": D9_CONTRACT_SHA256,
            "D9C_collection_sha256": D9C_COLLECTION_SHA256,
            "D9D_collection_sha256": D9D_ATTESTATION_SHA256,
            "D9E_readiness_sha256": readiness["sha256"],
        },
        "artifacts": {
            "pair_records": "pair_records.jsonl",
            "pair_records_sha256": sha256_file(pair_path),
            "false_safe_records": "false_safe_records.jsonl",
            "false_safe_records_sha256": sha256_file(false_path),
            "result_payload": "result_payload.pt",
            "result_payload_sha256": sha256_file(payload_path),
        },
        "access_ledger": {
            "formal_D9E_aggregate_calls": 1,
            "D9C_pairs_opened": len(pairs),
            "D9C_rollout_success_values_opened": 2 * len(pairs),
            "D9C_policy_call_records_opened": sum(
                pair.a1.policy_calls + pair.phase_route.policy_calls for pair in pairs
            ),
            "D9D_truth_rows_opened": len(truths),
            "LIBERO_environments_created": 0,
            "environment_actions_executed": 0,
            "model_or_router_loaded": False,
            "CUDA_initialized": torch.cuda.is_initialized(),
            "threshold_model_feature_or_episode_tuning": 0,
            "replacement_or_second_independent_test": False,
        },
        "claim_boundary": {
            "same_noise_L27_is_expert_or_task_success_certificate": False,
            "early_exit_and_failure_cooccurrence_proves_causation": False,
            "McNemar_equality_is_noninferiority": False,
            "measured_FM_reduction_is_wall_clock_speedup": False,
            "five_head_router_predict_latency_was_recorded_online": False,
            "independent_test_authorizes_deployment": False,
        },
    }
    report_path = report_incomplete / "result.json"
    _write_json(report_path, report_result)
    (report_incomplete / "result.sha256").write_text(
        f"{sha256_file(report_path)}  result.json\n", encoding="utf-8"
    )
    report_incomplete.replace(report)

    formal_value = {
        "status": aggregate["status"],
        "schema_version": FORMAL_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "source_git_commit": source_commit,
        "source_worktree_dirty_at_entry": False,
        "suite": "libero_10",
        "formal_report": {
            "path": (REPORT_OUTPUT / "result.json").as_posix(),
            "sha256": sha256_file(report / "result.json"),
        },
        "artifacts": report_result["artifacts"],
        "success": aggregate["success"],
        "efficiency": aggregate["efficiency"],
        "safety": aggregate["safety"],
        "early_exit_failure_association": aggregate[
            "early_exit_failure_association"
        ],
        "per_task": aggregate["per_task"],
        "gate_checks": aggregate["gate_checks"],
        "all_primary_gates_pass": aggregate["all_primary_gates_pass"],
        "input_binding": report_result["input_binding"],
        "access_ledger": report_result["access_ledger"],
        "authorization": {
            "next_stage": (
                "FINAL_PAPER_ANALYSIS_AND_ABLATION_PROTOCOL_DESIGN_ONLY"
                if aggregate["all_primary_gates_pass"]
                else "D9_NEGATIVE_RESULT_ANALYSIS_ONLY"
            ),
            "additional_test_tuning_or_second_independent_test": False,
            "deployment_authorized": False,
        },
        "claim_boundary": report_result["claim_boundary"],
    }
    _write_json(formal_incomplete, formal_value)
    formal_incomplete.replace(formal)
    formal.with_suffix(".sha256").write_text(
        f"{sha256_file(formal)}  {formal.name}\n", encoding="utf-8"
    )
    print(aggregate["status"])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Never remove partial one-shot evidence automatically; a failure must
        # remain visible for audit rather than silently becoming a clean rerun.
        raise
