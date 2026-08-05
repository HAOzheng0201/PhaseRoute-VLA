"""Run a real LIBERO episode with the M2 phase estimator in observer-only mode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark

from a1.vla.dynamic_compute.phase_observer import SafePhaseObserver
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
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("num-episodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    observer_path = args.output_dir / "phase_observer_calls.jsonl"
    for path in (result_path, telemetry_path, observer_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite observer rollout: {path}")

    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
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
        run_id_note=f"m2-phase-observer-task{args.task_id}",
    )
    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} is outside the benchmark")
    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=100)
    observer = SafePhaseObserver(
        args.phase_checkpoint,
        observer_path,
        device=device,
        history_len=8,
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
            telemetry_logger=telemetry,
            phase_cache_writer=observer,
        )
    finally:
        telemetry.close()
        observer.close()
        log_file.close()

    telemetry_records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observer_records = [
        json.loads(line)
        for line in observer_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry_keys = {
        (str(record["episode_id"]), int(record["step_id"]))
        for record in telemetry_records
    }
    observer_keys = {
        (str(record["episode_id"]), int(record["step_id"]))
        for record in observer_records
    }
    policy_latencies = [float(record["latency_ms"]) for record in telemetry_records]
    observer_latencies = [float(record["latency_ms"]) for record in observer_records]
    status_ok = (
        episodes == args.num_episodes
        and successes == args.num_episodes
        and telemetry.error_count == 0
        and observer.error_count == 0
        and len(telemetry_records) == len(observer_records)
        and telemetry_keys == observer_keys
        and all(record["observer_only"] for record in observer_records)
        and not any(record["controls_early_exit"] for record in observer_records)
    )
    source_status = git_output("status", "--porcelain=v1")
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "observer_only": True,
        "controls_early_exit": False,
        "a1_checkpoint": str(args.checkpoint.resolve()),
        "phase_checkpoint": str(args.phase_checkpoint),
        "phase_checkpoint_sha256": observer.checkpoint_sha256,
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
        "observer_calls": len(observer_records),
        "aligned_call_keys": telemetry_keys == observer_keys,
        "telemetry_errors": telemetry.error_count,
        "telemetry_last_error": telemetry.last_error,
        "observer_errors": observer.error_count,
        "observer_last_error": observer.last_error,
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "policy_latency_ms_mean": (
            statistics.fmean(policy_latencies) if policy_latencies else None
        ),
        "observer_latency_ms_mean": (
            statistics.fmean(observer_latencies) if observer_latencies else None
        ),
        "observer_to_policy_latency_percent": (
            statistics.fmean(observer_latencies)
            / statistics.fmean(policy_latencies)
            * 100.0
            if policy_latencies and observer_latencies
            else None
        ),
        "eval_log": str(eval_log_path),
        "telemetry_path": str(telemetry_path),
        "telemetry_sha256": file_sha256(telemetry_path),
        "observer_path": str(observer_path),
        "observer_sha256": file_sha256(observer_path),
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
