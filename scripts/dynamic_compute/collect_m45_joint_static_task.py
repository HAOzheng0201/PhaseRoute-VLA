"""Evaluate one M4.5 depth-routing/static-vision mode on paired LIBERO states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark

from a1.vla.dynamic_compute.exit_policy import (
    phase_exit_policy_config_for_ablation,
)
from a1.vla.dynamic_compute.phase_depth_runtime import SafePhaseDepthRuntime
from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
from robot_experiments.libero.eval_libero_early_exit import (
    GenerateConfig,
    get_image_resize_size,
    initialize_and_load_model,
    run_task,
    setup_logging,
)
from robot_experiments.robot_utils import set_seed_everywhere
from smoke_m1_telemetry import make_exit_controller


MODES = ("baseline", "joint", "pool144", "joint_pool144")
PHASE_MODES = frozenset(("joint", "joint_pool144"))
VISION_MODES = frozenset(("pool144", "joint_pool144"))
SUCCESS_PATTERN = re.compile(r"^Success: (True|False)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--phase-checkpoint", type=Path)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--keep-tokens", type=int, default=144)
    parser.add_argument("--bank-tokens", type=int, default=144)
    parser.add_argument("--min-tokens-per-crop", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def _episode_summaries(
    records: list[dict[str, Any]], episode_successes: list[bool]
) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[str(record.get("episode_id"))].append(record)

    summaries = []
    for episode_index, episode_id in enumerate(sorted(by_episode)):
        episode_records = by_episode[episode_id]
        exit_layers = [int(record["exit_layer"]) for record in episode_records]
        latencies = [float(record["latency_ms"]) for record in episode_records]
        summaries.append(
            {
                "episode_id": episode_id,
                "success": (
                    episode_successes[episode_index]
                    if episode_index < len(episode_successes)
                    else None
                ),
                "policy_calls": len(episode_records),
                "mean_exit_layer": mean_or_none(exit_layers),
                "fm_calls_total": sum(
                    int(record["fm_calls"]) for record in episode_records
                ),
                "fm_steps_total": sum(
                    int(record["fm_steps_total"]) for record in episode_records
                ),
                "latency_ms_mean": mean_or_none(latencies),
                "latency_ms_median": (
                    statistics.median(latencies) if latencies else None
                ),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    phase_enabled = args.mode in PHASE_MODES
    vision_enabled = args.mode in VISION_MODES
    if phase_enabled and args.phase_checkpoint is None:
        raise ValueError(f"--phase-checkpoint is required for mode={args.mode}")
    if args.keep_tokens <= 0:
        raise ValueError("--keep-tokens must be positive")
    if args.bank_tokens < args.keep_tokens:
        raise ValueError("--bank-tokens must be >= --keep-tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"Refusing to overwrite run in {args.output_dir}")

    checkpoint = args.checkpoint.resolve()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_episodes,
        action_head_flow_matching_inference_steps=args.fm_steps,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        local_log_dir=str(args.output_dir / "eval_logs"),
        save_rollout_video=False,
        save_rollout_video_path=str(args.output_dir),
        use_wandb=False,
        reseed_each_episode=True,
        seed=args.seed,
        run_id_note=f"m45-{args.mode}-task{args.task_id}",
        vision_aggregation_enabled=vision_enabled,
        vision_keep_tokens=args.keep_tokens,
        vision_bank_tokens=args.bank_tokens,
        vision_min_tokens_per_crop=args.min_tokens_per_crop,
        # A candidate must fail visibly instead of silently becoming baseline.
        vision_aggregation_fail_open=False,
    )

    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
    runtime = None
    if phase_enabled:
        runtime = SafePhaseDepthRuntime(
            args.phase_checkpoint.resolve(),
            device=device,
            eligible_exit_layers=tuple(exit_controller.exit_id_list),
            history_len=8,
            fm_steps_per_exit=args.fm_steps,
            exit_policy_config=phase_exit_policy_config_for_ablation("joint"),
        )

    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} outside suite")

    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    logger = SafeJSONLTelemetryLogger(telemetry_path, flush_every=25)
    try:
        episodes, successes, exit_sum, exit_count = run_task(
            cfg=cfg,
            task_suite=task_suite,
            task_id=args.task_id,
            model=model,
            exit_controller=exit_controller,
            device=device,
            num_tasks=1,
            resize_size=resize_size,
            total_episodes=0,
            total_successes=0,
            log_file=log_file,
            total_exit_mean_sum=0.0,
            total_exit_mean_count=0,
            telemetry_logger=logger,
            phase_depth_runtime=runtime,
        )
    finally:
        logger.close()
        log_file.close()

    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exit_layers = [int(record["exit_layer"]) for record in records]
    latencies = [float(record["latency_ms"]) for record in records]
    plans = [record.get("extra", {}).get("phase_plan") for record in records]
    plans = [plan for plan in plans if plan]
    vision_events = []
    aggregation_errors = []
    exit_reasons = Counter()
    adjusted_candidates = 0
    below_min_depth_candidates = 0
    risk_guard_candidates = 0
    for record in records:
        extra = record.get("extra", {})
        vision = extra.get("vision_aggregation")
        if vision:
            vision_events.append(vision)
        for event in extra.get("exit_events", []):
            if event.get("event") == "vision_aggregation_error":
                aggregation_errors.append(event)
            if event.get("event") != "exit_candidate":
                continue
            reason = event.get("phase_reason")
            if reason:
                exit_reasons[reason] += 1
            if (
                event.get("evaluated")
                and event.get("threshold") is not None
                and event.get("base_threshold") is not None
                and event.get("threshold") != event.get("base_threshold")
            ):
                adjusted_candidates += 1
            below_min_depth_candidates += int(reason == "below_min_depth")
            risk_guard_candidates += int(reason == "phase_risk_guard")

    eval_text = Path(eval_log_path).read_text(encoding="utf-8")
    episode_successes = [
        match == "True" for match in SUCCESS_PATTERN.findall(eval_text)
    ]
    profile_counts = Counter(plan["profile_name"] for plan in plans)
    plan_reasons = Counter(plan["profile_reason"] for plan in plans)
    expected_plans = len(records) if phase_enabled else 0
    expected_vision_events = len(records) if vision_enabled else 0
    run_ok = (
        episodes == args.num_episodes
        and len(episode_successes) == episodes
        and sum(episode_successes) == successes
        and logger.error_count == 0
        and len(plans) == expected_plans
        and (runtime is None or runtime.error_count == 0)
        and not aggregation_errors
        and len(vision_events) == expected_vision_events
        and all(int(event["kept_tokens"]) == args.keep_tokens for event in vision_events)
    )
    result = {
        "status": "PASS" if run_ok else "FAIL",
        "mode": args.mode,
        "phase_enabled": phase_enabled,
        "vision_enabled": vision_enabled,
        "checkpoint": str(checkpoint),
        "phase_checkpoint": (
            str(args.phase_checkpoint.resolve()) if phase_enabled else None
        ),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "episode_successes": episode_successes,
        "episode_summaries": _episode_summaries(records, episode_successes),
        "policy_calls": len(records),
        "keep_tokens": args.keep_tokens if vision_enabled else None,
        "bank_tokens": args.bank_tokens if vision_enabled else None,
        "mean_active_tokens": mean_or_none(
            [record["active_tokens_by_layer"][0] for record in records]
        ),
        "mean_llm_sequence_length": mean_or_none(
            [event["llm_sequence_length"] for event in vision_events]
        ),
        "mean_original_llm_sequence_length": mean_or_none(
            [event["original_llm_sequence_length"] for event in vision_events]
        ),
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "mean_exit_layer": mean_or_none(exit_layers),
        "exit_layer_counts": dict(sorted(Counter(exit_layers).items())),
        "fm_calls_total": sum(int(record["fm_calls"]) for record in records),
        "fm_calls_per_policy_call": (
            sum(int(record["fm_calls"]) for record in records) / len(records)
            if records
            else None
        ),
        "fm_steps_total": sum(int(record["fm_steps_total"]) for record in records),
        "latency_ms_mean": mean_or_none(latencies),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "inference_latency_ms_total": sum(latencies),
        "source_projected_tokens": (
            vision_events[0]["source_projected_tokens"] if vision_events else None
        ),
        "original_visual_slots": (
            vision_events[0]["original_tokens"] if vision_events else None
        ),
        "vision_events": len(vision_events),
        "aggregation_errors": aggregation_errors,
        "profile_counts": dict(sorted(profile_counts.items())),
        "profile_reason_counts": dict(sorted(plan_reasons.items())),
        "exit_reason_counts": dict(sorted(exit_reasons.items())),
        "adjusted_threshold_candidates": adjusted_candidates,
        "below_min_depth_candidates": below_min_depth_candidates,
        "risk_guard_candidates": risk_guard_candidates,
        "phase_latency_ms_mean": mean_or_none(
            [float(plan["phase_latency_ms"]) for plan in plans]
        ),
        "phase_fallbacks": sum(int(bool(plan.get("fallback"))) for plan in plans),
        "runtime_plans": runtime.records_prepared if runtime is not None else 0,
        "runtime_errors": runtime.error_count if runtime is not None else 0,
        "runtime_last_error": runtime.last_error if runtime is not None else None,
        "telemetry_errors": logger.error_count,
        "telemetry_last_error": logger.last_error,
        "eval_log": str(eval_log_path),
        "telemetry_path": str(telemetry_path),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not run_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
