"""Evaluate one static M4 vision budget on paired LIBERO initial states."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument(
        "--keep-tokens",
        type=int,
        default=0,
        help="0 selects the unmodified baseline",
    )
    parser.add_argument("--bank-tokens", type=int, default=144)
    parser.add_argument("--min-tokens-per-crop", type=int, default=4)
    parser.add_argument("--variant-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def main() -> None:
    args = parse_args()
    if args.keep_tokens < 0:
        raise ValueError("--keep-tokens must be non-negative")
    if args.keep_tokens and args.bank_tokens < args.keep_tokens:
        raise ValueError("--bank-tokens must be >= --keep-tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    checkpoint = args.checkpoint.resolve()
    mode = args.variant_name or (
        "baseline" if args.keep_tokens == 0 else f"keep_{args.keep_tokens}"
    )
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
        run_id_note=f"m4-{mode}-task{args.task_id}",
        vision_aggregation_enabled=args.keep_tokens > 0,
        vision_keep_tokens=args.keep_tokens or 64,
        vision_bank_tokens=args.bank_tokens,
        vision_min_tokens_per_crop=args.min_tokens_per_crop,
        # Evaluation must expose an aggregation bug instead of silently
        # turning a candidate into the baseline.
        vision_aggregation_fail_open=False,
    )

    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
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
    vision_events = []
    aggregation_errors = []
    for record in records:
        extra = record.get("extra", {})
        vision = extra.get("vision_aggregation")
        if vision:
            vision_events.append(vision)
        aggregation_errors.extend(
            event
            for event in extra.get("exit_events", [])
            if event.get("event") == "vision_aggregation_error"
        )

    candidate = args.keep_tokens > 0
    run_ok = (
        episodes == args.num_episodes
        and logger.error_count == 0
        and not aggregation_errors
        and (len(vision_events) == len(records) if candidate else not vision_events)
        and all(
            int(event["kept_tokens"]) == args.keep_tokens
            for event in vision_events
        )
    )
    result = {
        "status": "PASS" if run_ok else "FAIL",
        "mode": mode,
        "checkpoint": str(checkpoint),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "policy_calls": len(records),
        "keep_tokens": args.keep_tokens or None,
        "bank_tokens": args.bank_tokens if candidate else None,
        "mean_active_tokens": mean_or_none(
            [record["active_tokens_by_layer"][0] for record in records]
        ),
        "mean_llm_sequence_length": mean_or_none(
            [event["llm_sequence_length"] for event in vision_events]
        ),
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "mean_exit_layer": mean_or_none(exit_layers),
        "exit_layer_counts": dict(sorted(Counter(exit_layers).items())),
        "fm_calls_total": sum(int(record["fm_calls"]) for record in records),
        "fm_steps_total": sum(int(record["fm_steps_total"]) for record in records),
        "latency_ms_mean": mean_or_none(latencies),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "source_projected_tokens": (
            vision_events[0]["source_projected_tokens"] if vision_events else None
        ),
        "original_visual_slots": (
            vision_events[0]["original_tokens"] if vision_events else None
        ),
        "vision_events": len(vision_events),
        "aggregation_errors": aggregation_errors,
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
