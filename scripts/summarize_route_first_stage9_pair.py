#!/usr/bin/env python3
"""Build the preregistered Stage-9 state-12 paired engineering summary.

The utility consumes only sealed candidate-first and route-first run
directories.  It rechecks the stored artifact hashes and experiment identity
before reporting descriptive latency deltas.  A one-episode engineering pair
is not promoted into a formal speedup or closed-loop improvement claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "phase-route-vla.route-first-stage9-paired-smoke.v1"
CANDIDATE_ATTESTATION_SCHEMA = "phase-route-vla.route-first-stage9-candidate-arm.v1"
CANDIDATE_EVALUATION_SCHEMA = "phase-route-vla.libero-evaluation-summary.v1"
ROUTE_ATTESTATION_SCHEMA = "phase-route-vla.route-first-active-attestation.v1"
ROUTE_EVALUATION_SCHEMA = "phase-route-vla.route-first-active-evaluation.v1"
FROZEN_TASK_IDS = [0]
FROZEN_EPISODE_INDICES = [12]
FROZEN_SEED_BASE = 20260826
FROZEN_EPISODE_SEED = 20260838


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    output = float(value)
    _require(math.isfinite(output), f"{label} must be finite")
    _require(output > 0.0 if positive else output >= 0.0, f"invalid {label}")
    return output


def _int(value: Any, label: str, *, positive: bool = False) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be int")
    _require(value > 0 if positive else value >= 0, f"invalid {label}")
    return value


def _policy_latency(evaluation: Mapping[str, Any], label: str) -> dict[str, float | int]:
    measurement = evaluation.get("stage1_measurement")
    _require(isinstance(measurement, Mapping), f"{label} measurement missing")
    latency_ms = measurement.get("latency_ms")
    _require(isinstance(latency_ms, Mapping), f"{label} latency summary missing")
    policy = latency_ms.get("policy_wall")
    _require(isinstance(policy, Mapping), f"{label} policy_wall summary missing")
    count = _int(policy.get("count"), f"{label} policy count", positive=True)
    return {
        "count": count,
        "sum": _number(policy.get("sum"), f"{label} policy sum", positive=True),
        "mean": _number(policy.get("mean"), f"{label} policy mean", positive=True),
        "p50": _number(policy.get("p50"), f"{label} policy p50", positive=True),
        "p95": _number(policy.get("p95"), f"{label} policy p95", positive=True),
        "max": _number(policy.get("max"), f"{label} policy max", positive=True),
    }


def _runtime(evaluation: Mapping[str, Any], label: str) -> dict[str, Any]:
    runtime = evaluation.get("runtime")
    _require(isinstance(runtime, Mapping), f"{label} runtime missing")
    selected = runtime.get("selected_layers")
    _require(isinstance(selected, Mapping), f"{label} selected_layers missing")
    selected_layers = {
        str(layer): _int(selected.get(str(layer), 0), f"{label} layer {layer}")
        for layer in (11, 13, 27)
    }
    calls = _int(runtime.get("policy_calls"), f"{label} policy calls", positive=True)
    _require(sum(selected_layers.values()) == calls, f"{label} route counts do not sum to calls")
    return {
        "policy_calls": calls,
        "prepared_calls": _int(runtime.get("prepared_calls"), f"{label} prepared calls"),
        "committed_calls": _int(runtime.get("committed_calls"), f"{label} committed calls"),
        "error_count": _int(runtime.get("error_count"), f"{label} error count"),
        "selected_layers": selected_layers,
    }


def _episode(evaluation: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    episodes = evaluation.get("episodes")
    _require(isinstance(episodes, list) and len(episodes) == 1, f"{label} must contain one episode")
    episode = episodes[0]
    _require(isinstance(episode, Mapping), f"{label} episode must be an object")
    return episode


def _reduction(candidate: float, route: float) -> float:
    _require(candidate > 0.0 and route > 0.0, "latency values must be positive")
    return 1.0 - route / candidate


def _speedup(candidate: float, route: float) -> float:
    _require(candidate > 0.0 and route > 0.0, "latency values must be positive")
    return candidate / route


def build_summary(
    candidate_attestation: Mapping[str, Any],
    candidate_evaluation: Mapping[str, Any],
    route_attestation: Mapping[str, Any],
    route_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate experiment identity and build a descriptive paired result."""

    _require(
        candidate_attestation.get("schema_version") == CANDIDATE_ATTESTATION_SCHEMA,
        "unexpected candidate attestation schema",
    )
    _require(candidate_attestation.get("status") == "PASS", "candidate arm did not pass")
    _require(candidate_attestation.get("method") == "candidate_first_v3", "candidate method drift")
    _require(candidate_attestation.get("arm_position") == 1, "candidate must be arm 1")
    _require(
        route_attestation.get("schema_version") == ROUTE_ATTESTATION_SCHEMA,
        "unexpected route attestation schema",
    )
    _require(route_attestation.get("status") == "PASS", "route arm did not pass")
    _require(route_attestation.get("selection_error") is None, "route selection error is not null")
    protocol_sha = str(candidate_attestation.get("protocol_sha256", ""))
    _require(len(protocol_sha) == 64, "candidate protocol SHA is invalid")
    _require(route_attestation.get("protocol_sha256") == protocol_sha, "protocol SHA mismatch")

    _require(
        candidate_evaluation.get("schema_version") == CANDIDATE_EVALUATION_SCHEMA,
        "unexpected candidate evaluation schema",
    )
    _require(candidate_evaluation.get("method") == "phase_route_v3", "candidate evaluation method drift")
    _require(
        route_evaluation.get("schema_version") == ROUTE_EVALUATION_SCHEMA,
        "unexpected route evaluation schema",
    )
    _require(route_evaluation.get("method") == "route_first_stage8", "route evaluation method drift")
    _require(route_evaluation.get("arm_position") == 2, "route-first must be arm 2")
    _require(route_evaluation.get("protocol_sha256") == protocol_sha, "route evaluation protocol drift")

    for label, evaluation in (("candidate", candidate_evaluation), ("route", route_evaluation)):
        _require(evaluation.get("suite") == "libero_10", f"{label} suite drift")
        _require(evaluation.get("task_ids") == FROZEN_TASK_IDS, f"{label} task drift")
        _require(
            evaluation.get("episode_indices") == FROZEN_EPISODE_INDICES,
            f"{label} state drift",
        )
        _require(evaluation.get("seed_base") == FROZEN_SEED_BASE, f"{label} seed base drift")
        _require(evaluation.get("total_episodes") == 1, f"{label} episode count drift")
        episode = _episode(evaluation, label)
        _require(episode.get("task_id") == 0, f"{label} episode task drift")
        _require(episode.get("episode_index") == 12, f"{label} episode state drift")
        _require(episode.get("seed") == FROZEN_EPISODE_SEED, f"{label} episode seed drift")

    candidate_gpu = str(candidate_attestation.get("gpu_uuid", ""))
    route_gpu = route_evaluation.get("gpu")
    _require(isinstance(route_gpu, Mapping), "route GPU identity missing")
    _require(route_gpu.get("expected_uuid") == candidate_gpu, "paired arms used different GPU UUIDs")

    candidate_runtime = _runtime(candidate_evaluation, "candidate")
    route_runtime = _runtime(route_evaluation, "route")
    candidate_latency = _policy_latency(candidate_evaluation, "candidate")
    route_latency = _policy_latency(route_evaluation, "route")
    _require(candidate_latency["count"] == candidate_runtime["policy_calls"], "candidate timing count drift")
    _require(route_latency["count"] == route_runtime["policy_calls"], "route timing count drift")

    route_integrity = route_evaluation["runtime"].get("route_first_integrity")
    _require(isinstance(route_integrity, Mapping), "route-first integrity missing")
    exact_fm = _int(
        route_integrity.get("valid_calls_with_exactly_one_fm"),
        "route exact-FM calls",
    )
    fm_invocations = _int(route_integrity.get("fm_invocations"), "route FM invocations")
    route_calls = route_runtime["policy_calls"]

    candidate_episode = _episode(candidate_evaluation, "candidate")
    route_episode = _episode(route_evaluation, "route")
    candidate_successes = _int(candidate_evaluation.get("total_successes"), "candidate successes")
    route_successes = _int(route_evaluation.get("total_successes"), "route successes")
    candidate_episode_wall = _number(
        candidate_episode.get("wall_seconds"), "candidate episode wall", positive=True
    )
    route_episode_wall = _number(route_episode.get("wall_seconds"), "route episode wall", positive=True)

    checks = {
        "candidate_attestation_pass": True,
        "route_attestation_pass": True,
        "protocol_identity": True,
        "candidate_is_arm_1": True,
        "route_first_is_arm_2": True,
        "frozen_task_state_seed": True,
        "same_physical_gpu_uuid": True,
        "candidate_runtime_error_free": candidate_runtime["error_count"] == 0,
        "route_runtime_error_free": route_runtime["error_count"] == 0,
        "candidate_calls_all_committed": (
            candidate_runtime["policy_calls"]
            == candidate_runtime["prepared_calls"]
            == candidate_runtime["committed_calls"]
        ),
        "route_calls_all_committed": (
            route_runtime["policy_calls"]
            == route_runtime["prepared_calls"]
            == route_runtime["committed_calls"]
        ),
        "route_l11_disabled": route_runtime["selected_layers"]["11"] == 0,
        "route_every_call_exactly_one_fm": exact_fm == route_calls == fm_invocations,
        "candidate_measurement_complete": candidate_latency["count"]
        == candidate_runtime["policy_calls"],
        "route_measurement_complete": route_latency["count"] == route_calls,
        "candidate_episode_success": candidate_successes == 1
        and bool(candidate_episode.get("success")),
        "route_episode_success": route_successes == 1
        and bool(route_episode.get("success")),
    }

    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "stage9_preregistered_state12_paired_engineering_smoke",
        "protocol_sha256": protocol_sha,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "identity": {
            "suite": "libero_10",
            "task_ids": FROZEN_TASK_IDS,
            "episode_indices": FROZEN_EPISODE_INDICES,
            "seed_base": FROZEN_SEED_BASE,
            "episode_seed": FROZEN_EPISODE_SEED,
            "gpu_uuid": candidate_gpu,
        },
        "candidate_first": {
            "arm_position": 1,
            "successes": candidate_successes,
            "policy_calls": candidate_runtime["policy_calls"],
            "selected_layers": candidate_runtime["selected_layers"],
            "policy_wall_latency_ms": candidate_latency,
            "episode_wall_seconds": candidate_episode_wall,
        },
        "route_first": {
            "arm_position": 2,
            "successes": route_successes,
            "policy_calls": route_calls,
            "selected_layers": route_runtime["selected_layers"],
            "policy_wall_latency_ms": route_latency,
            "episode_wall_seconds": route_episode_wall,
            "fm_invocations": fm_invocations,
            "calls_with_exactly_one_fm": exact_fm,
        },
        "descriptive_comparison": {
            "success_difference_route_minus_candidate": route_successes - candidate_successes,
            "same_policy_call_count": route_calls == candidate_runtime["policy_calls"],
            "same_selected_layer_counts": (
                route_runtime["selected_layers"] == candidate_runtime["selected_layers"]
            ),
            "policy_wall_mean_reduction_fraction": _reduction(
                float(candidate_latency["mean"]), float(route_latency["mean"])
            ),
            "policy_wall_mean_speedup": _speedup(
                float(candidate_latency["mean"]), float(route_latency["mean"])
            ),
            "policy_wall_p50_reduction_fraction": _reduction(
                float(candidate_latency["p50"]), float(route_latency["p50"])
            ),
            "policy_wall_p95_reduction_fraction": _reduction(
                float(candidate_latency["p95"]), float(route_latency["p95"])
            ),
            "summed_policy_wall_reduction_fraction": _reduction(
                float(candidate_latency["sum"]), float(route_latency["sum"])
            ),
            "episode_wall_reduction_fraction": _reduction(
                candidate_episode_wall, route_episode_wall
            ),
        },
        "checks": checks,
        "next_gate": {
            "state13_pilot_protocol_gate_unlocked": all(checks.values()),
            "state13_opened": False,
        },
        "claim_boundary": {
            "paired_episodes": 1,
            "engineering_smoke_only": True,
            "formal_closed_loop_improvement_claim": False,
            "formal_wall_clock_speedup_claim": False,
            "statistical_significance_claim": False,
            "latency_is_descriptive_same_gpu_measurement": True,
            "deployment_authorized": False,
        },
    }


def _verify_attested_artifacts(run_dir: Path, attestation: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = attestation.get("artifacts")
    _require(isinstance(artifacts, Mapping) and artifacts, f"attested artifacts missing: {run_dir}")
    verified: dict[str, Any] = {}
    for filename, expected in artifacts.items():
        _require(isinstance(filename, str), "artifact filename must be a string")
        _require(isinstance(expected, Mapping), f"invalid artifact metadata: {filename}")
        path = run_dir / filename
        _require(path.is_file(), f"attested artifact missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        _require(size == expected.get("bytes"), f"artifact size mismatch: {path}")
        _require(digest == expected.get("sha256"), f"artifact SHA mismatch: {path}")
        verified[filename] = {"bytes": size, "sha256": digest}
    return verified


def summarize_dirs(candidate_dir: Path, route_dir: Path) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve(strict=True)
    route_dir = route_dir.resolve(strict=True)
    _require(candidate_dir.is_dir() and route_dir.is_dir(), "run paths must be directories")
    _require(candidate_dir != route_dir, "candidate and route directories must differ")

    candidate_attestation_path = candidate_dir / "stage9_arm_attestation.json"
    candidate_evaluation_path = candidate_dir / "evaluation_summary.json"
    route_attestation_path = route_dir / "run_attestation.json"
    route_evaluation_path = route_dir / "evaluation_summary.json"
    candidate_attestation = _read_json(candidate_attestation_path)
    route_attestation = _read_json(route_attestation_path)
    result = build_summary(
        candidate_attestation,
        _read_json(candidate_evaluation_path),
        route_attestation,
        _read_json(route_evaluation_path),
    )
    result["run_dirs"] = {
        "candidate_first": str(candidate_dir),
        "route_first": str(route_dir),
    }
    result["verified_artifacts"] = {
        "candidate_first": _verify_attested_artifacts(candidate_dir, candidate_attestation),
        "route_first": _verify_attested_artifacts(route_dir, route_attestation),
    }
    result["input_attestations"] = {
        "candidate_first": {
            "path": str(candidate_attestation_path),
            "sha256": sha256_file(candidate_attestation_path),
        },
        "route_first": {
            "path": str(route_attestation_path),
            "sha256": sha256_file(route_attestation_path),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = summarize_dirs(args.candidate_dir, args.route_dir)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
