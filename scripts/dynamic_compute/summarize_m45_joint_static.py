"""Validate and summarize the four M4.5 paired experiment modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODES = ("baseline", "joint", "pool144", "joint_pool144")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection_dir", type=Path)
    parser.add_argument("--output-name", default="summary.json")
    return parser.parse_args()


def _relative_change(candidate: float, reference: float) -> float | None:
    return (candidate / reference - 1.0) if reference else None


def _comparison(
    results: dict[str, dict[str, Any]], candidate_name: str, reference_name: str
) -> dict[str, Any]:
    candidate = results[candidate_name]
    reference = results[reference_name]
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "success_delta": candidate["successes"] - reference["successes"],
        "policy_call_delta": candidate["policy_calls"] - reference["policy_calls"],
        "active_token_relative_change": _relative_change(
            candidate["mean_active_tokens"], reference["mean_active_tokens"]
        ),
        "mean_exit_layer_delta": (
            candidate["mean_exit_layer"] - reference["mean_exit_layer"]
        ),
        "fm_calls_total_relative_change": _relative_change(
            candidate["fm_calls_total"], reference["fm_calls_total"]
        ),
        "fm_calls_per_policy_call_relative_change": _relative_change(
            candidate["fm_calls_per_policy_call"],
            reference["fm_calls_per_policy_call"],
        ),
        "latency_mean_relative_change": _relative_change(
            candidate["latency_ms_mean"], reference["latency_ms_mean"]
        ),
        "latency_median_relative_change": _relative_change(
            candidate["latency_ms_median"], reference["latency_ms_median"]
        ),
        "inference_latency_total_relative_change": _relative_change(
            candidate["latency_ms_mean"] * candidate["policy_calls"],
            reference["latency_ms_mean"] * reference["policy_calls"],
        ),
        "paired_episode_outcomes_equal": (
            candidate["episode_successes"] == reference["episode_successes"]
        ),
    }


def main() -> None:
    args = parse_args()
    results = {}
    for mode in MODES:
        result_path = args.collection_dir / mode / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("mode") != mode:
            raise ValueError(f"Mode mismatch in {result_path}")
        if result.get("status") != "PASS":
            raise ValueError(f"Run did not pass: {result_path}")
        results[mode] = result

    paired_keys = ("task_suite", "task_id", "seed", "requested_episodes")
    reference = results["baseline"]
    for mode, result in results.items():
        for key in paired_keys:
            if result[key] != reference[key]:
                raise ValueError(f"Unpaired {key}: baseline vs {mode}")

    comparisons = {
        "joint_vs_baseline": _comparison(results, "joint", "baseline"),
        "pool144_vs_baseline": _comparison(results, "pool144", "baseline"),
        "joint_pool144_vs_pool144": _comparison(
            results, "joint_pool144", "pool144"
        ),
        "joint_pool144_vs_baseline": _comparison(
            results, "joint_pool144", "baseline"
        ),
    }
    joint_pool = results["joint_pool144"]
    baseline = results["baseline"]
    pareto_screen = {
        "success_not_below_baseline": (
            joint_pool["successes"] >= baseline["successes"]
        ),
        "active_tokens_below_baseline": (
            joint_pool["mean_active_tokens"] < baseline["mean_active_tokens"]
        ),
        "fm_calls_total_below_baseline": (
            joint_pool["fm_calls_total"] < baseline["fm_calls_total"]
        ),
        "inference_latency_total_below_baseline": (
            joint_pool["latency_ms_mean"] * joint_pool["policy_calls"]
            < baseline["latency_ms_mean"] * baseline["policy_calls"]
        ),
    }
    pareto_screen["all_pass"] = all(pareto_screen.values())
    summary = {
        "status": "PASS",
        "collection_dir": str(args.collection_dir.resolve()),
        "paired_config": {key: reference[key] for key in paired_keys},
        "modes": {
            mode: {
                key: results[mode][key]
                for key in (
                    "successes",
                    "completed_episodes",
                    "episode_successes",
                    "policy_calls",
                    "mean_active_tokens",
                    "mean_llm_sequence_length",
                    "mean_exit_layer",
                    "fm_calls_total",
                    "fm_calls_per_policy_call",
                    "latency_ms_mean",
                    "latency_ms_median",
                    "inference_latency_ms_total",
                    "runtime_errors",
                    "telemetry_errors",
                )
                if key in results[mode]
            }
            for mode in MODES
        },
        "comparisons": comparisons,
        "joint_pool144_pareto_screen": pareto_screen,
    }
    for mode in MODES:
        summary["modes"][mode]["inference_latency_ms_total"] = (
            results[mode]["latency_ms_mean"] * results[mode]["policy_calls"]
        )
    summary_path = args.collection_dir / args.output_name
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
