"""Fail-closed machine summary for the frozen M4.25b 60-episode cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.temporal_route_features import (  # noqa: E402
    canonical_teacher_route,
    parse_episode_index,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (  # noqa: E402
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)


ROLE_BY_EPISODE = {
    0: "development",
    1: "development",
    2: "development",
    3: "calibration",
    4: "test",
    5: "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cache_root = args.cache_root.resolve()
    all_rows = []
    task_summaries = {}
    identities = set()
    checks = {
        "ten_tasks": True,
        "all_results_pass": True,
        "all_six_episodes": True,
        "all_cache_rows_aligned": True,
        "all_trace_rows_complete": True,
        "all_gpu_mappings_verified": True,
        "only_physical_gpu0_3": True,
        "all_rows_unique": True,
    }
    for task_id in range(10):
        task_dir = cache_root / f"task{task_id}"
        result_path = task_dir / "result.json"
        manifest_path = task_dir / "teacher_calls" / "manifest.jsonl"
        if not result_path.is_file() or not manifest_path.is_file():
            checks["ten_tasks"] = False
            raise FileNotFoundError(f"missing task{task_id} cache result or manifest")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        episode_grid = {parse_episode_index(row["episode_id"]) for row in rows}
        gpu_audit = result.get("gpu_audit") or {}
        result_pass = (
            result.get("status") == "PASS"
            and result.get("checkpoint_sha256") == args.checkpoint_sha256
            and int(result.get("seed", -1)) == args.expected_seed
            and int(result.get("requested_episodes", -1)) == 6
            and int(result.get("completed_episodes", -1)) == 6
        )
        aligned = (
            int(result.get("teacher_cache_calls", -1)) == len(rows)
            and int(result.get("telemetry_calls", -1)) == len(rows)
            and bool(result.get("aligned_call_keys"))
            and bool(result.get("aligned_exit_layers"))
            and bool(result.get("aligned_fm_counts"))
            and int(result.get("missing_shards", -1)) == 0
            and int(result.get("teacher_cache_errors", -1)) == 0
            and int(result.get("telemetry_errors", -1)) == 0
        )
        trace_complete = all(has_complete_candidate_fm_traces(row) for row in rows)
        mapping_verified = bool(gpu_audit.get("mapping_verified"))
        physical_gpu = int(gpu_audit.get("physical_gpu_index", -1))
        checks["all_results_pass"] &= result_pass
        checks["all_six_episodes"] &= episode_grid == set(range(6))
        checks["all_cache_rows_aligned"] &= aligned
        checks["all_trace_rows_complete"] &= trace_complete
        checks["all_gpu_mappings_verified"] &= mapping_verified
        checks["only_physical_gpu0_3"] &= physical_gpu in (0, 1, 2, 3)
        for row in rows:
            identity = (
                str(row["episode_id"]),
                int(row["step_id"]),
                int(row["task_id"]),
                str(row["array_path"]),
            )
            if identity in identities:
                checks["all_rows_unique"] = False
            identities.add(identity)
        task_summaries[str(task_id)] = {
            "status": result.get("status"),
            "episodes": int(result.get("completed_episodes", -1)),
            "successes": int(result.get("successes", -1)),
            "records": len(rows),
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "gpu_audit": gpu_audit,
            "route_distribution": {
                str(route): sum(
                    canonical_teacher_route(int(row["teacher_exit_layer"])) == route
                    for row in rows
                )
                for route in (11, 13, 27)
            },
            "raw_exit_distribution": {
                str(raw): sum(int(row["teacher_exit_layer"]) == raw for row in rows)
                for raw in sorted({int(row["teacher_exit_layer"]) for row in rows})
            },
        }
        all_rows.extend(rows)

    role_summaries = {}
    for role in ("development", "calibration", "test"):
        episode_indices = [
            index for index, value in ROLE_BY_EPISODE.items() if value == role
        ]
        rows = [
            row
            for row in all_rows
            if parse_episode_index(row["episode_id"]) in episode_indices
        ]
        role_summaries[role] = {
            "episode_indices": episode_indices,
            "episodes": len(episode_indices) * 10,
            "records": len(rows),
            "route_distribution": {
                str(route): sum(
                    canonical_teacher_route(int(row["teacher_exit_layer"])) == route
                    for row in rows
                )
                for route in (11, 13, 27)
            },
            "raw_exit_distribution": {
                str(raw): sum(int(row["teacher_exit_layer"]) == raw for row in rows)
                for raw in sorted({int(row["teacher_exit_layer"]) for row in rows})
            },
        }
    calibration_test_27 = (
        role_summaries["calibration"]["route_distribution"]["27"]
        + role_summaries["test"]["route_distribution"]["27"]
    )
    checks["calibration_plus_test_route27_at_least_10"] = calibration_test_27 >= 10
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "status": status,
        "scope": "m425b_teacher_cache_summary",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "cache_root": str(cache_root),
        "checkpoint_sha256": args.checkpoint_sha256,
        "expected_seed": args.expected_seed,
        "teacher_route_mapping": {
            "kind": "minimum_non_shallower_m425b_route",
            "raw_a1_exit_layers": list(range(1, 28, 2)),
            "m425b_route_layers": [11, 13, 27],
            "rule": "raw<=11 -> 11; raw<=13 -> 13; raw>=15 -> 27",
        },
        "tasks": 10,
        "episodes": 60,
        "successes": sum(value["successes"] for value in task_summaries.values()),
        "records": len(all_rows),
        "checks": checks,
        "role_summaries": role_summaries,
        "calibration_plus_test_route27": calibration_test_27,
        "task_summaries": task_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
