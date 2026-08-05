"""Summarize paired task0-3 baseline/learned/joint EFA rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_TASKS = tuple(range(4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--learned-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--risk-root", type=Path)
    parser.add_argument(
        "--risk-mode",
        choices=(
            "joint_risk_full_token_efa144",
            "joint_contact_full_token_efa144",
            "phase_width_contact_full_token_efa144",
            "phase_width_hysteresis_full_token_efa144",
            "phase_width_uncertainty_hysteresis_full_token_efa144",
        ),
        default="joint_risk_full_token_efa144",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _weighted_mean(rows: list[dict[str, Any]], name: str, weight: str):
    pairs = [
        (float(row[name]), int(row[weight]))
        for row in rows
        if row.get(name) is not None and int(row[weight]) > 0
    ]
    denominator = sum(item_weight for _, item_weight in pairs)
    if denominator == 0:
        return None
    return sum(value * item_weight for value, item_weight in pairs) / denominator


def _load_mode(root: Path, expected_mode: str) -> dict[str, Any]:
    rows = []
    for task_id in EXPECTED_TASKS:
        result_path = root / f"task{task_id}" / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if row.get("task_id") != task_id:
            raise ValueError(f"task id mismatch in {result_path}")
        if row.get("mode") != expected_mode:
            raise ValueError(f"mode mismatch in {result_path}")
        rows.append(row)

    paired_fields = ("task_suite", "seed", "requested_episodes", "fm_steps")
    for name in paired_fields:
        values = {json.dumps(row.get(name), sort_keys=True) for row in rows}
        if len(values) != 1:
            raise ValueError(f"{expected_mode} has inconsistent {name}")

    exit_counts = Counter()
    for row in rows:
        exit_counts.update(
            {int(layer): int(count) for layer, count in row["exit_layer_counts"].items()}
        )
    calls = sum(int(row["policy_calls"]) for row in rows)
    latency_total = sum(float(row["inference_latency_ms_total"]) for row in rows)
    episode_outcomes = [
        {
            "task_id": int(row["task_id"]),
            "episode_successes": [bool(value) for value in row["episode_successes"]],
        }
        for row in rows
    ]
    return {
        "status": (
            "PASS"
            if all(row.get("status") == "PASS" for row in rows)
            else "FAIL"
        ),
        "mode": expected_mode,
        "root": str(root.resolve()),
        "task_suite": rows[0]["task_suite"],
        "seed": rows[0]["seed"],
        "tasks": list(EXPECTED_TASKS),
        "requested_episodes": sum(int(row["requested_episodes"]) for row in rows),
        "completed_episodes": sum(int(row["completed_episodes"]) for row in rows),
        "successes": sum(int(row["successes"]) for row in rows),
        "episode_outcomes": episode_outcomes,
        "policy_calls": calls,
        "mean_active_tokens": _weighted_mean(rows, "mean_active_tokens", "policy_calls"),
        "mean_llm_sequence_length": _weighted_mean(
            rows, "mean_llm_sequence_length", "vision_events"
        ),
        "mean_original_llm_sequence_length": _weighted_mean(
            rows, "mean_original_llm_sequence_length", "vision_events"
        ),
        "mean_exit_layer": _weighted_mean(rows, "mean_exit_layer", "policy_calls"),
        "exit_layer_counts": dict(sorted(exit_counts.items())),
        "fm_calls_total": sum(int(row["fm_calls_total"]) for row in rows),
        "fm_steps_total": sum(int(row["fm_steps_total"]) for row in rows),
        "compressed_vision_calls": sum(
            int(row.get("compressed_vision_calls", 0)) for row in rows
        ),
        "full_token_fallback_calls": sum(
            int(row.get("full_token_fallback_calls", 0)) for row in rows
        ),
        "uncertainty_trigger_calls": sum(
            int(row.get("uncertainty_trigger_calls", 0)) for row in rows
        ),
        "mean_kept_visual_tokens": _weighted_mean(
            rows, "mean_kept_visual_tokens", "vision_events"
        ),
        "latency_ms_mean_weighted": latency_total / calls if calls else None,
        "inference_latency_ms_total": latency_total,
        "telemetry_errors": sum(int(row["telemetry_errors"]) for row in rows),
        "aggregation_error_count": sum(len(row["aggregation_errors"]) for row in rows),
        "phase_runtime_errors": sum(int(row["phase_runtime_errors"]) for row in rows),
        "task_results": rows,
    }


def _relative(candidate: float, baseline: float):
    return candidate / baseline - 1.0 if baseline else None


def _compare(candidate: dict[str, Any], baseline: dict[str, Any]):
    same_outcomes = candidate["episode_outcomes"] == baseline["episode_outcomes"]
    gates = {
        "paired_success_preserved": same_outcomes,
        "policy_calls_not_increased": (
            candidate["policy_calls"] <= baseline["policy_calls"]
        ),
        "active_tokens_reduced": (
            candidate["mean_active_tokens"] < baseline["mean_active_tokens"]
        ),
        "mean_exit_layer_not_increased": (
            candidate["mean_exit_layer"] <= baseline["mean_exit_layer"]
        ),
        "fm_calls_reduced": (
            candidate["fm_calls_total"] < baseline["fm_calls_total"]
        ),
        "total_inference_latency_reduced": (
            candidate["inference_latency_ms_total"]
            < baseline["inference_latency_ms_total"]
        ),
    }
    return {
        "candidate": candidate["mode"],
        "reference": baseline["mode"],
        "success_delta": candidate["successes"] - baseline["successes"],
        "policy_call_delta": candidate["policy_calls"] - baseline["policy_calls"],
        "active_token_relative_change": _relative(
            candidate["mean_active_tokens"], baseline["mean_active_tokens"]
        ),
        "mean_exit_layer_delta": (
            candidate["mean_exit_layer"] - baseline["mean_exit_layer"]
        ),
        "fm_call_delta": candidate["fm_calls_total"] - baseline["fm_calls_total"],
        "fm_call_relative_change": _relative(
            candidate["fm_calls_total"], baseline["fm_calls_total"]
        ),
        "total_latency_ms_delta": (
            candidate["inference_latency_ms_total"]
            - baseline["inference_latency_ms_total"]
        ),
        "total_latency_relative_change": _relative(
            candidate["inference_latency_ms_total"],
            baseline["inference_latency_ms_total"],
        ),
        "gates": gates,
        "pareto_pass": all(gates.values()),
    }


def main() -> None:
    args = parse_args()
    modes = {
        "baseline": _load_mode(args.baseline_root, "baseline"),
        "learned_efa144": _load_mode(args.learned_root, "learned_efa144"),
        "joint_learned_efa144": _load_mode(
            args.joint_root, "joint_learned_efa144"
        ),
    }
    if args.risk_root is not None:
        modes[args.risk_mode] = _load_mode(
            args.risk_root,
            args.risk_mode,
        )
    paired_fields = ("task_suite", "seed", "requested_episodes")
    for name in paired_fields:
        values = {json.dumps(mode[name], sort_keys=True) for mode in modes.values()}
        if len(values) != 1:
            raise ValueError(f"modes are not paired on {name}")
    comparisons = {
        name: _compare(mode, modes["baseline"])
        for name, mode in modes.items()
        if name != "baseline"
    }
    finite = all(
        math.isfinite(float(mode[name]))
        for mode in modes.values()
        for name in (
            "mean_active_tokens",
            "mean_exit_layer",
            "inference_latency_ms_total",
        )
    )
    result = {
        "status": (
            "PASS"
            if finite and all(mode["status"] == "PASS" for mode in modes.values())
            else "FAIL"
        ),
        "scope": (
            "m415_task0_3_width_uncertainty_hysteresis_rollout_summary"
            if (
                args.risk_root is not None
                and args.risk_mode
                == "phase_width_uncertainty_hysteresis_full_token_efa144"
            )
            else "m414_task0_3_width_hysteresis_rollout_summary"
            if (
                args.risk_root is not None
                and args.risk_mode == "phase_width_hysteresis_full_token_efa144"
            )
            else "m413_task0_3_width_only_contact_rollout_summary"
            if (
                args.risk_root is not None
                and args.risk_mode == "phase_width_contact_full_token_efa144"
            )
            else "m412_task0_3_contact_width_rollout_summary"
            if (
                args.risk_root is not None
                and args.risk_mode == "joint_contact_full_token_efa144"
            )
            else "m411_task0_3_risk_width_rollout_summary"
            if args.risk_root is not None
            else "m48_task0_3_paired_rollout_summary"
        ),
        "modes": modes,
        "comparisons": comparisons,
        "any_candidate_pareto_pass": any(
            comparison["pareto_pass"] for comparison in comparisons.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
