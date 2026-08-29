#!/usr/bin/env python3
"""Authorize one Stage-9 state-13 arm without opening a simulator episode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
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
    STAGE9_STATE12_GATE_RELATIVE_PATH,
    STAGE9_STATE12_GATE_SHA256,
    authorize_stage9_pilot_arm,
)


SCHEMA = "phase-route-vla.route-first-stage9-pilot-prelaunch.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--arm-position", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--stage9-preflight", type=Path, required=True)
    parser.add_argument("--v3-preflight", type=Path)
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


def git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def all_checks_true(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(item is True for item in value.values())
    )


def validate_prelaunch(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repo_root.resolve(strict=True)
    selection, _, _ = authorize_stage9_pilot_arm(
        repo_root=repository,
        protocol_path=args.protocol.resolve(strict=True),
        method=args.method,
        task_id=args.task_id,
        episode_index=args.episode_index,
        arm_position=args.arm_position,
        seed=args.seed,
    )
    stage9_path = args.stage9_preflight.resolve(strict=True)
    stage9 = load_object(stage9_path)
    v3_path = args.v3_preflight.resolve(strict=True) if args.v3_preflight else None
    v3 = load_object(v3_path) if v3_path else None
    candidate = args.method == STAGE9_CANDIDATE_METHOD
    current_commit = git_head(repository)
    physical_before = stage9.get("physical_gpu_before_cuda_smoke")
    physical_after = stage9.get("physical_gpu_after_cuda_smoke")
    checks = {
        "state12_unlock_exact": True,
        "selection_preregistered": True,
        "stage9_preflight_pass": stage9.get("status") == "PASS",
        "stage9_preflight_schema": stage9.get("schema_version")
        == "phase-route-vla.route-first-active-preflight.v1",
        "stage9_preflight_scope": stage9.get("scope")
        == "route_first_stage9_active_preflight",
        "stage9_preflight_no_episode": stage9.get("simulator_episode_opened") is False,
        "stage9_preflight_protocol": stage9.get("protocol_sha256")
        == ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "stage9_gpu_index": stage9.get("physical_gpu_index")
        == args.physical_gpu_index,
        "stage9_gpu_uuid": normalize_uuid(stage9.get("expected_gpu_uuid"))
        == normalize_uuid(args.expected_gpu_uuid),
        "stage9_physical_gpu_uuid": bool(
            isinstance(physical_before, Mapping)
            and isinstance(physical_after, Mapping)
            and normalize_uuid(physical_before.get("uuid"))
            == normalize_uuid(args.expected_gpu_uuid)
            == normalize_uuid(physical_after.get("uuid"))
        ),
        "stage9_repository": stage9.get("repo_root") == str(repository),
        "stage9_clean_exact_commit": stage9.get("worktree_dirty") is False
        and stage9.get("git_commit") == current_commit,
        "stage9_all_checks_true": all_checks_true(stage9.get("checks")),
        "candidate_v3_preflight_present": (v3 is not None) if candidate else True,
        "candidate_v3_preflight_pass": (
            bool(
                v3
                and v3.get("schema_version") == "phase-route-vla.v3.preflight.v1"
                and v3.get("status") == "PASS"
                and v3.get("scope") == "phase_route_v3_release_preflight"
            )
            if candidate
            else True
        ),
        "candidate_v3_gpu_uuid": (
            bool(
                v3
                and v3.get("physical_gpu_index") == args.physical_gpu_index
                and normalize_uuid(v3.get("expected_gpu_uuid"))
                == normalize_uuid(args.expected_gpu_uuid)
            )
            if candidate
            else True
        ),
        "candidate_v3_repository_commit": (
            bool(
                v3
                and v3.get("repo_root") == str(repository)
                and v3.get("git_commit") == current_commit
                and v3.get("worktree_dirty") is False
            )
            if candidate
            else True
        ),
        "candidate_v3_all_checks_true": (
            bool(v3 and all_checks_true(v3.get("checks"))) if candidate else True
        ),
        "state13_not_opened_by_prelaunch": True,
    }
    artifacts: dict[str, Any] = {
        "state12_gate": {
            "path": STAGE9_STATE12_GATE_RELATIVE_PATH.as_posix(),
            "sha256": STAGE9_STATE12_GATE_SHA256,
        },
        "stage9_preflight": {
            "path": str(stage9_path),
            "sha256": sha256_file(stage9_path),
        },
    }
    if v3_path is not None:
        artifacts["v3_preflight"] = {
            "path": str(v3_path),
            "sha256": sha256_file(v3_path),
        }
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "stage9_state13_pilot_prelaunch_no_episode",
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "simulator_episode_opened": False,
        "state13_open_authorized_for_this_arm": all(checks.values()),
        "research_simulation_only": True,
        "deployment_authorized": False,
        "repo_root": str(repository),
        "git_commit": stage9.get("git_commit"),
        "physical_gpu_index": args.physical_gpu_index,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "selection": asdict(selection),
        "checks": checks,
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_name(
        args.output.name + ".incomplete"
    ).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = validate_prelaunch(args)
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
