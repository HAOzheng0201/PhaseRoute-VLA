"""Collect aligned M1 telemetry and enriched M2 phase inputs on LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark

from a1.vla.dynamic_compute.phase_cache import SafePhaseCacheWriter
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
    parser.add_argument(
        "--checkpoint-sha256",
        default="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f",
    )
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--summary-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint-sha256 must have 64 hexadecimal characters")
    int(args.checkpoint_sha256, 16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    phase_cache_dir = args.output_dir / "phase_calls"
    if result_path.exists() or telemetry_path.exists() or (phase_cache_dir / "manifest.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run in {args.output_dir}")

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
        run_id_note=f"m2-phase-cache-task{args.task_id}",
    )

    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} is outside [0, {task_suite.n_tasks})")

    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    telemetry_logger = SafeJSONLTelemetryLogger(telemetry_path, flush_every=100)
    phase_cache_writer = SafePhaseCacheWriter(
        phase_cache_dir,
        summary_dtype=args.summary_dtype,
    )
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
            phase_cache_writer=phase_cache_writer,
        )
    finally:
        telemetry_logger.close()
        phase_cache_writer.close()
        log_file.close()

    telemetry_records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_path = phase_cache_dir / "manifest.jsonl"
    manifest_records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry_keys = {
        (str(record["episode_id"]), int(record["step_id"]))
        for record in telemetry_records
    }
    phase_keys = {
        (str(record["episode_id"]), int(record["step_id"]))
        for record in manifest_records
    }
    shard_paths = [phase_cache_dir / record["array_path"] for record in manifest_records]
    first_shapes = {}
    if shard_paths:
        with np.load(shard_paths[0]) as first_shard:
            first_shapes = {name: list(first_shard[name].shape) for name in first_shard.files}
    latencies = [float(record["latency_ms"]) for record in telemetry_records]
    source_status = git_output("status", "--porcelain=v1")
    status_ok = (
        episodes == args.num_episodes
        and telemetry_logger.error_count == 0
        and phase_cache_writer.error_count == 0
        and len(telemetry_records) == len(manifest_records)
        and telemetry_keys == phase_keys
        and all(path.is_file() for path in shard_paths)
    )
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "checkpoint": str(checkpoint / "model.pt"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "telemetry_calls": len(telemetry_records),
        "phase_cache_calls": len(manifest_records),
        "aligned_call_keys": telemetry_keys == phase_keys,
        "unique_episode_ids": len({record["episode_id"] for record in manifest_records}),
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "summary_dtype": args.summary_dtype,
        "first_shard_shapes": first_shapes,
        "missing_shards": sum(not path.is_file() for path in shard_paths),
        "telemetry_errors": telemetry_logger.error_count,
        "telemetry_last_error": telemetry_logger.last_error,
        "phase_cache_errors": phase_cache_writer.error_count,
        "phase_cache_last_error": phase_cache_writer.last_error,
        "eval_log": str(eval_log_path),
        "telemetry_path": str(telemetry_path),
        "telemetry_sha256": sha256_file(telemetry_path),
        "phase_manifest_path": str(manifest_path),
        "phase_manifest_sha256": sha256_file(manifest_path),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
