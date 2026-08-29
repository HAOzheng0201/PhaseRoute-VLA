#!/usr/bin/env python3
"""Seal one state-13 task pair after both preregistered arms finish."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_active_protocol import (  # noqa: E402
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
)
from a1.vla.dynamic_compute.route_first_stage9_pilot_protocol import (  # noqa: E402
    STAGE9_CANDIDATE_METHOD,
    STAGE9_ROUTE_FIRST_METHOD,
    expected_stage9_pilot_arm_order,
)


SCHEMA = "phase-route-vla.route-first-stage9-pilot-task-pair.v1"
ARM_SCHEMA = "phase-route-vla.route-first-stage9-pilot-arm.v1"


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


def load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalize_uuid(value: Any) -> str:
    result = str(value).strip().lower()
    return result[4:] if result.startswith("gpu-") else result


def _verify_arm(run_dir: Path) -> tuple[Mapping[str, Any], str]:
    attestation_path = run_dir / "stage9_pilot_arm_attestation.json"
    attestation = load_object(attestation_path)
    if (
        attestation.get("schema_version") != ARM_SCHEMA
        or attestation.get("status") != "PASS"
        or attestation.get("protocol_sha256") != ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256
        or not isinstance(attestation.get("checks"), Mapping)
        or not attestation.get("checks")
        or not all(value is True for value in attestation["checks"].values())
    ):
        raise ValueError(f"pilot arm attestation did not pass: {run_dir}")
    artifacts = attestation.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"pilot arm artifacts missing: {run_dir}")
    for filename, expected in artifacts.items():
        if not isinstance(filename, str) or not isinstance(expected, Mapping):
            raise ValueError("invalid pilot arm artifact metadata")
        path = run_dir / filename
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise ValueError(f"pilot arm artifact mismatch: {path}")
    return attestation, sha256_file(attestation_path)


def build_task_pair(
    candidate: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate.get("method") != STAGE9_CANDIDATE_METHOD:
        raise ValueError("candidate arm method differs")
    if route.get("method") != STAGE9_ROUTE_FIRST_METHOD:
        raise ValueError("route arm method differs")
    candidate_selection = candidate.get("selection")
    route_selection = route.get("selection")
    candidate_result = candidate.get("result")
    route_result = route.get("result")
    if not all(
        isinstance(item, Mapping)
        for item in (
            candidate_selection,
            route_selection,
            candidate_result,
            route_result,
        )
    ):
        raise ValueError("pilot pair evidence is incomplete")
    task_id = int(candidate_selection.get("task_id", -1))
    expected_order = expected_stage9_pilot_arm_order(task_id)
    methods_by_position = {
        int(candidate.get("arm_position", -1)): STAGE9_CANDIDATE_METHOD,
        int(route.get("arm_position", -1)): STAGE9_ROUTE_FIRST_METHOD,
    }
    candidate_gpu = normalize_uuid(candidate_selection.get("gpu_uuid"))
    route_gpu = normalize_uuid(route_selection.get("gpu_uuid"))
    same_identity = bool(
        route_selection.get("task_id") == task_id
        and candidate_selection.get("episode_index")
        == route_selection.get("episode_index")
        == 13
        and candidate_selection.get("base_seed")
        == route_selection.get("base_seed")
        == 20260826
        and candidate_selection.get("episode_seed")
        == route_selection.get("episode_seed")
        == 20260839 + task_id * 10_000
        and candidate_gpu not in ("", "none")
        and candidate_gpu == route_gpu
    )
    candidate_latency = candidate_result.get("policy_wall_latency_ms")
    route_latency = route_result.get("policy_wall_latency_ms")
    if not isinstance(candidate_latency, Mapping) or not isinstance(
        route_latency, Mapping
    ):
        raise ValueError("pilot pair latency evidence is missing")
    candidate_p50 = float(candidate_latency["p50"])
    route_p50 = float(route_latency["p50"])
    checks = {
        "candidate_arm_pass": candidate.get("status") == "PASS",
        "route_arm_pass": route.get("status") == "PASS",
        "same_task_state_seed_gpu": same_identity,
        "alternating_arm_order": tuple(
            methods_by_position.get(index) for index in (1, 2)
        )
        == expected_order,
        "route_one_fm_per_call": route_result.get("route_exact_fm_invocations")
        == route_result.get("policy_calls"),
        "positive_latency": candidate_p50 > 0.0 and route_p50 > 0.0,
    }
    candidate_success = bool(candidate_result.get("success"))
    route_success = bool(route_result.get("success"))
    if candidate_success and route_success:
        outcome = "both_success"
    elif candidate_success:
        outcome = "candidate_only_success"
    elif route_success:
        outcome = "route_only_success"
    else:
        outcome = "both_failure"
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "stage9_state13_preregistered_task_pair",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "research_simulation_only": True,
        "deployment_authorized": False,
        "task_id": task_id,
        "episode_index": 13,
        "base_seed": 20260826,
        "episode_seed": 20260839 + task_id * 10_000,
        "gpu_uuid": candidate_selection.get("gpu_uuid"),
        "arm_order": list(expected_order),
        "paired_outcome": outcome,
        "candidate_first": dict(candidate_result),
        "route_first": dict(route_result),
        "descriptive_comparison": {
            "success_route_minus_candidate": int(route_success)
            - int(candidate_success),
            "policy_wall_p50_ratio_route_to_candidate": route_p50 / candidate_p50,
            "policy_wall_p50_reduction_fraction": 1.0 - route_p50 / candidate_p50,
            "policy_wall_mean_ratio_route_to_candidate": float(route_latency["mean"])
            / float(candidate_latency["mean"]),
        },
        "checks": checks,
        "claim_boundary": {
            "single_task_pair_only": True,
            "task_failure_retained": True,
            "global_pilot_gate_evaluated": False,
            "formal_speedup_claim": False,
            "deployment_authorized": False,
        },
    }


def summarize_dirs(candidate_dir: Path, route_dir: Path) -> dict[str, Any]:
    candidate_root = candidate_dir.resolve(strict=True)
    route_root = route_dir.resolve(strict=True)
    candidate, candidate_sha = _verify_arm(candidate_root)
    route, route_sha = _verify_arm(route_root)
    result = build_task_pair(candidate, route)
    result["run_dirs"] = {
        "candidate_first": str(candidate_root),
        "route_first": str(route_root),
    }
    result["input_attestations"] = {
        "candidate_first": {
            "path": str(candidate_root / "stage9_pilot_arm_attestation.json"),
            "sha256": candidate_sha,
        },
        "route_first": {
            "path": str(route_root / "stage9_pilot_arm_attestation.json"),
            "sha256": route_sha,
        },
    }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_name(
        args.output.name + ".incomplete"
    ).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = summarize_dirs(args.candidate_dir, args.route_dir)
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
