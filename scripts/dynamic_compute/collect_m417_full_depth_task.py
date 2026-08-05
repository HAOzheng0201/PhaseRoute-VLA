"""Run a paired LIBERO control with full-depth A1 and no early exit.

This collector intentionally reuses the original full-depth evaluation path in
``robot_experiments.libero.eval_libero``.  It changes neither the A1 checkpoint
nor the observation/action preprocessing; it only restricts the evaluation to
one task and records enough metadata to compare with an existing early-exit
run that used the same task, seed, episode index, and initial state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def control_status_ok(
    *,
    requested_episodes: int,
    completed_episodes: int,
    successes: int,
    policy_calls: int,
    action_chunk_lengths: list[int],
    model_class: str,
) -> bool:
    """Validate engineering completion without requiring task success."""

    return (
        requested_episodes > 0
        and completed_episodes == requested_episodes
        and 0 <= successes <= completed_episodes
        and policy_calls > 0
        and len(action_chunk_lengths) == policy_calls
        and all(length > 0 for length in action_chunk_lengths)
        and model_class == "a1.vla.affordvla.AffordVLA"
    )


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("num-episodes must be positive")
    if args.fm_steps <= 0:
        raise ValueError("fm-steps must be positive")
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint-sha256 must contain 64 hexadecimal characters")
    int(args.checkpoint_sha256, 16)

    checkpoint = args.checkpoint.resolve()
    checkpoint_model = checkpoint / "model.pt"
    if not checkpoint_model.is_file():
        raise FileNotFoundError(checkpoint_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists() or (args.output_dir / "eval_logs").exists():
        raise FileExistsError(f"Refusing to overwrite run in {args.output_dir}")

    # Heavy imports remain inside main so the pure audit helpers stay cheap to test.
    import torch
    from libero.libero import benchmark
    import robot_experiments.libero.eval_libero as full_eval
    from robot_experiments.robot_utils import set_seed_everywhere

    cfg = full_eval.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_episodes,
        action_head_flow_matching_inference_steps=args.fm_steps,
        local_log_dir=str(args.output_dir / "eval_logs"),
        save_rollout_video=False,
        save_rollout_video_path=str(args.output_dir),
        use_wandb=False,
        reseed_each_episode=True,
        seed=args.seed,
        run_id_note=f"m417-full-depth-task{args.task_id}",
    )

    set_seed_everywhere(cfg.seed)
    model, device = full_eval.initialize_and_load_model(cfg)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    resize_size = full_eval.get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} outside suite")

    initial_states = task_suite.get_task_init_states(args.task_id)
    if len(initial_states) < args.num_episodes:
        raise ValueError("not enough default initial states for requested episodes")
    initial_state_sha256 = [
        sha256_array(np.asarray(initial_states[index]))
        for index in range(args.num_episodes)
    ]
    episode_seeds = [
        args.seed + args.task_id * 10_000 + index
        for index in range(args.num_episodes)
    ]

    log_file, eval_log_path, _ = full_eval.setup_logging(cfg)
    original_get_vla_action = full_eval.get_vla_action
    call_latencies_ms: list[float] = []
    action_chunk_lengths: list[int] = []

    def timed_get_vla_action(*call_args: Any, **call_kwargs: Any):
        start = time.perf_counter()
        actions = original_get_vla_action(*call_args, **call_kwargs)
        call_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        action_chunk_lengths.append(len(actions))
        return actions

    full_eval.get_vla_action = timed_get_vla_action
    wall_start = time.perf_counter()
    try:
        episodes, successes = full_eval.run_task(
            cfg=cfg,
            task_suite=task_suite,
            task_id=args.task_id,
            model=model,
            device=device,
            num_tasks=1,
            resize_size=resize_size,
            total_episodes=0,
            total_successes=0,
            log_file=log_file,
        )
    finally:
        full_eval.get_vla_action = original_get_vla_action
        log_file.close()
    wall_seconds = time.perf_counter() - wall_start

    source_status = git_output("status", "--porcelain=v1")
    engineering_ok = control_status_ok(
        requested_episodes=args.num_episodes,
        completed_episodes=episodes,
        successes=successes,
        policy_calls=len(call_latencies_ms),
        action_chunk_lengths=action_chunk_lengths,
        model_class=model_class,
    )
    result = {
        "status": "PASS" if engineering_ok else "FAIL",
        "scope": "m417_full_depth_no_early_exit_control",
        "policy_kind": "a1_full_depth",
        "early_exit_enabled": False,
        "exit_controller_constructed": False,
        "vision_aggregation_enabled": False,
        "checkpoint": str(checkpoint_model),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_bytes": checkpoint_model.stat().st_size,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "model_class": model_class,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "episode_seeds": episode_seeds,
        "initial_state_sha256": initial_state_sha256,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "fm_steps": args.fm_steps,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "policy_calls": len(call_latencies_ms),
        "action_chunk_lengths": action_chunk_lengths,
        "latency_ms_mean": (
            statistics.fmean(call_latencies_ms) if call_latencies_ms else None
        ),
        "latency_ms_median": (
            statistics.median(call_latencies_ms) if call_latencies_ms else None
        ),
        "latency_ms_total": sum(call_latencies_ms),
        "wall_seconds": wall_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "eval_log": str(eval_log_path),
        "eval_log_sha256": sha256_file(Path(eval_log_path)),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not engineering_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
