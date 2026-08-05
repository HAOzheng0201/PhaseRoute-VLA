"""Summarize paired early-exit and full-depth LIBERO outcomes for M4.17."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-result", type=Path, action="append", required=True)
    parser.add_argument("--full-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_name(early_success: bool, full_success: bool) -> str:
    if early_success and full_success:
        return "both_succeed"
    if not early_success and not full_success:
        return "both_fail"
    if not early_success and full_success:
        return "early_exit_failure_suspected"
    return "full_depth_regression_or_trajectory_difference"


def build_summary(
    early_items: list[tuple[str, dict[str, Any]]],
    full_items: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    early_by_task = {int(item[1]["task_id"]): item for item in early_items}
    full_by_task = {int(item[1]["task_id"]): item for item in full_items}
    if len(early_by_task) != len(early_items) or len(full_by_task) != len(full_items):
        raise ValueError("duplicate task result")
    if early_by_task.keys() != full_by_task.keys():
        raise ValueError("early/full task sets differ")

    rows = []
    for task_id in sorted(early_by_task):
        early_path, early = early_by_task[task_id]
        full_path, full = full_by_task[task_id]
        for name, result in (("early", early), ("full", full)):
            if result.get("status") != "PASS":
                raise ValueError(f"{name} task{task_id} engineering status is not PASS")
            if int(result["completed_episodes"]) != 1:
                raise ValueError("M4.17 attribution requires exactly one paired episode")
        if full.get("scope") != "m417_full_depth_no_early_exit_control":
            raise ValueError("unexpected full-depth result scope")
        if full.get("model_class") != "a1.vla.affordvla.AffordVLA":
            raise ValueError("full-depth result used the wrong model class")
        paired_fields = ("checkpoint_sha256", "task_suite", "task_id", "seed")
        if any(early.get(field) != full.get(field) for field in paired_fields):
            raise ValueError(f"task{task_id} paired metadata differs")

        early_success = int(early["successes"]) == 1
        full_success = int(full["successes"]) == 1
        early_latency = float(early["latency_ms_mean"])
        full_latency = float(full["latency_ms_mean"])
        rows.append(
            {
                "task_id": task_id,
                "outcome": outcome_name(early_success, full_success),
                "paired_metadata": {
                    "checkpoint_sha256": full["checkpoint_sha256"],
                    "task_suite": full["task_suite"],
                    "seed": int(full["seed"]),
                    "episode_index": 0,
                    "episode_seed": int(full["episode_seeds"][0]),
                    "initial_state_sha256": full["initial_state_sha256"][0],
                    "fm_steps": int(full["fm_steps"]),
                },
                "early_exit": {
                    "result_path": early_path,
                    "success": early_success,
                    "policy_calls": int(early["telemetry_calls"]),
                    "latency_ms_mean": early_latency,
                    "mean_exit_ratio": float(early["mean_exit_ratio"]),
                },
                "full_depth": {
                    "result_path": full_path,
                    "success": full_success,
                    "policy_calls": int(full["policy_calls"]),
                    "latency_ms_mean": full_latency,
                    "peak_cuda_memory_bytes": int(full["peak_cuda_memory_bytes"]),
                },
                "policy_calls_delta_full_minus_early": (
                    int(full["policy_calls"]) - int(early["telemetry_calls"])
                ),
                "latency_ratio_full_over_early": full_latency / early_latency,
            }
        )

    early_failures = [row for row in rows if not row["early_exit"]["success"]]
    fixed = [row for row in early_failures if row["full_depth"]["success"]]
    positive_controls = [row for row in rows if row["outcome"] == "both_succeed"]
    outcome_counts = {
        name: sum(row["outcome"] == name for row in rows)
        for name in (
            "both_succeed",
            "both_fail",
            "early_exit_failure_suspected",
            "full_depth_regression_or_trajectory_difference",
        )
    }
    return {
        "status": "PASS",
        "scope": "m417_full_depth_failure_attribution",
        "paired_tasks": [row["task_id"] for row in rows],
        "paired_episodes": len(rows),
        "outcome_counts": outcome_counts,
        "observed_early_exit_failures": len(early_failures),
        "failures_fixed_by_full_depth": len(fixed),
        "early_exit_attributable_failure_fraction": (
            len(fixed) / len(early_failures) if early_failures else None
        ),
        "full_depth_positive_controls": len(positive_controls),
        "full_depth_path_validated": bool(positive_controls),
        "interpretation": (
            "No observed early-exit failure was fixed by full depth in this paired "
            "single-episode set; this does not replace multi-seed evaluation."
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    early_items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.early_result
    ]
    full_items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.full_result
    ]
    result = build_summary(early_items, full_items)
    result["input_sha256"] = {
        str(path.resolve()): sha256_file(path)
        for path in (*args.early_result, *args.full_result)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
