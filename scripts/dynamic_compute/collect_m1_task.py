"""Collect M1 policy-call telemetry for selected LIBERO task episodes."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        run_id_note=f"m1-telemetry-task{args.task_id}",
    )

    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} is outside [0, {task_suite.n_tasks})")

    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    telemetry_logger = SafeJSONLTelemetryLogger(telemetry_path, flush_every=100)
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
            telemetry_logger=telemetry_logger,
        )
    finally:
        telemetry_logger.close()
        log_file.close()

    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_ids = sorted({record["episode_id"] for record in records})
    latencies = [float(record["latency_ms"]) for record in records]
    result = {
        "status": "PASS" if episodes == args.num_episodes and telemetry_logger.error_count == 0 else "FAIL",
        "checkpoint": str(checkpoint),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "policy_calls": len(records),
        "unique_episode_ids": len(episode_ids),
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "telemetry_errors": telemetry_logger.error_count,
        "telemetry_last_error": telemetry_logger.last_error,
        "eval_log": str(eval_log_path),
        "telemetry_path": str(telemetry_path),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

