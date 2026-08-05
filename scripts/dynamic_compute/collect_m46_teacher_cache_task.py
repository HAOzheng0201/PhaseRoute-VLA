"""Collect projected visual features aligned with A1 early-exit teacher actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark

from a1.vla.dynamic_compute.device_guard import normalize_gpu_uuid
from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
from a1.vla.dynamic_compute.vision_teacher_cache import (
    SafeVisionTeacherCacheWriter,
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)
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
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument(
        "--feature-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--physical-gpu-index", type=int)
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
    teacher_cache_dir = args.output_dir / "teacher_calls"
    if (
        result_path.exists()
        or telemetry_path.exists()
        or (teacher_cache_dir / "manifest.jsonl").exists()
    ):
        raise FileExistsError(f"Refusing to overwrite run in {args.output_dir}")

    gpu_audit = None
    if args.expected_gpu_uuid is not None or args.physical_gpu_index is not None:
        if args.expected_gpu_uuid is None or args.physical_gpu_index is None:
            raise ValueError(
                "expected-gpu-uuid and physical-gpu-index must be provided together"
            )
        if args.physical_gpu_index not in (0, 1, 2, 3):
            raise ValueError("teacher-cache collection only permits physical GPUs 0-3")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("teacher-cache collection requires one visible CUDA device")
        visible_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if normalize_gpu_uuid(visible_uuid) != normalize_gpu_uuid(
            args.expected_gpu_uuid
        ):
            raise RuntimeError(
                "GPU UUID mismatch: "
                f"host={args.expected_gpu_uuid} visible={visible_uuid}"
            )
        gpu_audit = {
            "physical_gpu_index": int(args.physical_gpu_index),
            "expected_host_uuid": str(args.expected_gpu_uuid),
            "visible_device_index": 0,
            "visible_uuid": visible_uuid,
            "visible_device_count": int(torch.cuda.device_count()),
            "mapping_verified": True,
        }

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
        run_id_note=f"m46-teacher-cache-task{args.task_id}",
        vision_aggregation_enabled=False,
    )

    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} outside suite")

    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    telemetry_logger = SafeJSONLTelemetryLogger(telemetry_path, flush_every=25)
    teacher_writer = SafeVisionTeacherCacheWriter(
        teacher_cache_dir,
        feature_dtype=args.feature_dtype,
        teacher_kind="a1_early_exit",
        checkpoint_sha256=args.checkpoint_sha256,
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
            vision_teacher_cache_writer=teacher_writer,
        )
    finally:
        telemetry_logger.close()
        teacher_writer.close()
        log_file.close()

    telemetry_records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_path = teacher_cache_dir / "manifest.jsonl"
    manifest_records = (
        [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if manifest_path.is_file()
        else []
    )
    telemetry_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in telemetry_records
    }
    teacher_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in manifest_records
    }
    shard_paths = [teacher_cache_dir / record["array_path"] for record in manifest_records]
    first_shapes = {}
    first_dtypes = {}
    if shard_paths:
        with np.load(shard_paths[0]) as first_shard:
            first_shapes = {name: list(first_shard[name].shape) for name in first_shard.files}
            first_dtypes = {name: str(first_shard[name].dtype) for name in first_shard.files}

    aligned_keys = telemetry_by_key.keys() == teacher_by_key.keys()
    aligned_exit_layers = aligned_keys and all(
        int(teacher_by_key[key]["teacher_exit_layer"])
        == int(telemetry_by_key[key]["exit_layer"])
        for key in teacher_by_key
    )
    aligned_fm = aligned_keys and all(
        int(teacher_by_key[key]["fm_calls"])
        == int(telemetry_by_key[key]["fm_calls"])
        and int(teacher_by_key[key]["fm_steps_total"])
        == int(telemetry_by_key[key]["fm_steps_total"])
        for key in teacher_by_key
    )
    expected_geometry = bool(manifest_records) and all(
        record["schema_version"] == VISION_TEACHER_CACHE_SCHEMA_VERSION
        and int(record["source_projected_tokens"]) == 576
        and int(record["unique_visual_slots"]) == 288
        and int(record["valid_crop_count"]) == 4
        for record in manifest_records
    )
    exact_exit_trace = bool(manifest_records) and all(
        int(record.get("fm_trace_count", 0)) >= int(record["fm_calls"])
        and float(record.get("teacher_trace_max_abs_error", float("inf"))) <= 1e-5
        and record.get("shapes", {}).get("teacher_exit_input_x") == [8, 7]
        and record.get("shapes", {}).get("teacher_exit_trace_action") == [8, 7]
        and record.get("shapes", {}).get("input_ids")
        == [int(record.get("sequence_length", -1))]
        for record in manifest_records
    )
    complete_candidate_traces = bool(manifest_records) and all(
        has_complete_candidate_fm_traces(record) for record in manifest_records
    )
    source_status = git_output("status", "--porcelain=v1")
    latencies = [float(record["latency_ms"]) for record in telemetry_records]
    status_ok = (
        episodes == args.num_episodes
        and telemetry_logger.error_count == 0
        and teacher_writer.error_count == 0
        and len(telemetry_records) == len(manifest_records)
        and aligned_keys
        and aligned_exit_layers
        and aligned_fm
        and expected_geometry
        and exact_exit_trace
        and complete_candidate_traces
        and all(path.is_file() for path in shard_paths)
    )
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "teacher_kind": "a1_early_exit",
        "checkpoint": str(checkpoint / "model.pt"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "gpu_audit": gpu_audit,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "telemetry_calls": len(telemetry_records),
        "teacher_cache_calls": len(manifest_records),
        "aligned_call_keys": aligned_keys,
        "aligned_exit_layers": aligned_exit_layers,
        "aligned_fm_counts": aligned_fm,
        "expected_projected_geometry": expected_geometry,
        "exact_exit_fm_trace": exact_exit_trace,
        "complete_candidate_fm_traces": complete_candidate_traces,
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "feature_dtype": args.feature_dtype,
        "first_shard_shapes": first_shapes,
        "first_shard_dtypes": first_dtypes,
        "cache_bytes": sum(path.stat().st_size for path in shard_paths if path.is_file()),
        "missing_shards": sum(not path.is_file() for path in shard_paths),
        "telemetry_errors": telemetry_logger.error_count,
        "telemetry_last_error": telemetry_logger.last_error,
        "teacher_cache_errors": teacher_writer.error_count,
        "teacher_cache_last_error": teacher_writer.last_error,
        "eval_log": str(eval_log_path),
        "telemetry_path": str(telemetry_path),
        "telemetry_sha256": sha256_file(telemetry_path),
        "teacher_manifest_path": str(manifest_path),
        "teacher_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
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
