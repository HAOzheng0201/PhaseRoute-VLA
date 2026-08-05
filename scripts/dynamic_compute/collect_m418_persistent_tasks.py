"""Collect multiple LIBERO tasks while loading one A1 policy only once.

The collector supports either the original A1 early-exit policy or the
full-depth policy.  Each process evaluates a disjoint task shard and appends a
durable JSONL episode record immediately after every completed rollout.  The
same task ids, episode indices, checkpoint, seed formula, and initial-state
selection can therefore be paired across two independent policy processes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from scripts.dynamic_compute.collect_m417_full_depth_task import (  # noqa: E402
    sha256_array,
    sha256_file,
)


POLICY_MODEL_CLASSES = {
    "early_exit": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
    "rp_pep": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
    "full_depth": "a1.vla.affordvla.AffordVLA",
}
EARLY_EXIT_POLICIES = {"early_exit", "rp_pep"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=tuple(POLICY_MODEL_CLASSES), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, action="append", required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--episode-start-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, action="append")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def engineering_status_ok(
    *,
    policy: str,
    model_class: str,
    task_ids: list[int],
    episodes_per_task: int,
    episode_start_index: int = 0,
    episode_indices: list[int] | None = None,
    episode_records: list[dict[str, Any]],
    telemetry_errors: int,
) -> bool:
    selected_indices = (
        list(episode_indices)
        if episode_indices is not None
        else list(
            range(
                episode_start_index,
                episode_start_index + episodes_per_task,
            )
        )
    )
    expected_pairs = {
        (task_id, episode_idx)
        for task_id in task_ids
        for episode_idx in selected_indices
    }
    actual_pairs = {
        (int(row["task_id"]), int(row["episode_idx"])) for row in episode_records
    }
    return (
        policy in POLICY_MODEL_CLASSES
        and model_class == POLICY_MODEL_CLASSES[policy]
        and len(task_ids) == len(set(task_ids))
        and len(selected_indices) == episodes_per_task
        and len(selected_indices) == len(set(selected_indices))
        and actual_pairs == expected_pairs
        and len(episode_records) == len(expected_pairs)
        and telemetry_errors == 0
        and all(row.get("status") == "PASS" for row in episode_records)
        and all(int(row["policy_calls"]) > 0 for row in episode_records)
        and all(
            len(row["action_chunk_lengths"]) == int(row["policy_calls"])
            for row in episode_records
        )
        and all(
            len(row.get("action_chunk_sha256", [])) == int(row["policy_calls"])
            for row in episode_records
        )
        and all(
            len(row.get("latency_ms_by_call", [])) == int(row["policy_calls"])
            for row in episode_records
        )
    )


class InitialStateWindowTaskSuite:
    """Expose a contiguous default-initial-state window as local episodes.

    LIBERO's ``run_task`` always indexes initial states from zero.  This proxy
    lets independent persistent processes evaluate disjoint global episode
    ranges without modifying LIBERO itself.  All other suite attributes and
    methods are forwarded unchanged.
    """

    def __init__(self, task_suite: Any, start_index: int, episode_count: int):
        if start_index < 0 or episode_count <= 0:
            raise ValueError("invalid initial-state window")
        self._task_suite = task_suite
        self.start_index = start_index
        self.stop_index = start_index + episode_count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task_suite, name)

    def get_task_init_states(self, task_id: int):
        initial_states = self._task_suite.get_task_init_states(task_id)
        if self.stop_index > len(initial_states):
            raise ValueError(
                f"task {task_id} has {len(initial_states)} initial states, "
                f"cannot select [{self.start_index}:{self.stop_index}]"
            )
        return initial_states[self.start_index : self.stop_index]


class SelectedInitialStateTaskSuite:
    """Expose arbitrary global default initial-state indices as local episodes."""

    def __init__(self, task_suite: Any, episode_indices: list[int]):
        if not episode_indices:
            raise ValueError("episode index selection must not be empty")
        if any(index < 0 for index in episode_indices):
            raise ValueError("episode indices must be nonnegative")
        if len(episode_indices) != len(set(episode_indices)):
            raise ValueError("episode indices must be unique")
        self._task_suite = task_suite
        self.episode_indices = list(episode_indices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task_suite, name)

    def get_task_init_states(self, task_id: int):
        initial_states = self._task_suite.get_task_init_states(task_id)
        if max(self.episode_indices) >= len(initial_states):
            raise ValueError(
                f"task {task_id} has {len(initial_states)} initial states, cannot "
                f"select indices {self.episode_indices}"
            )
        return [initial_states[index] for index in self.episode_indices]


def resolve_episode_indices(
    *,
    episode_indices: list[int] | None,
    episode_start_index: int,
    episodes_per_task: int,
) -> list[int]:
    if episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")
    if episode_start_index < 0:
        raise ValueError("episode-start-index must be nonnegative")
    if episode_indices is None:
        return list(
            range(
                episode_start_index,
                episode_start_index + episodes_per_task,
            )
        )
    selected = [int(index) for index in episode_indices]
    if episode_start_index != 0:
        raise ValueError(
            "episode-start-index cannot be combined with explicit episode-index values"
        )
    if len(selected) != episodes_per_task:
        raise ValueError("num-episodes must equal the number of episode-index values")
    if any(index < 0 for index in selected):
        raise ValueError("episode-index values must be nonnegative")
    if len(selected) != len(set(selected)):
        raise ValueError("episode-index values must be unique")
    return selected


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _task_summaries(
    task_ids: list[int], episode_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result = {}
    for task_id in task_ids:
        selected = [row for row in episode_records if int(row["task_id"]) == task_id]
        result[str(task_id)] = {
            "episodes": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
            "success_rate": (
                sum(bool(row["success"]) for row in selected) / len(selected)
                if selected
                else 0.0
            ),
            "policy_calls": sum(int(row["policy_calls"]) for row in selected),
            "latency_ms_total": sum(float(row["latency_ms_total"]) for row in selected),
        }
    return result


def main() -> None:
    args = parse_args()
    task_ids = [int(task_id) for task_id in args.task_id]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task-id values must be unique")
    if args.num_episodes <= 0 or args.fm_steps <= 0:
        raise ValueError("num-episodes and fm-steps must be positive")
    selected_episode_indices = resolve_episode_indices(
        episode_indices=args.episode_index,
        episode_start_index=args.episode_start_index,
        episodes_per_task=args.num_episodes,
    )
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint-sha256 must contain 64 hexadecimal characters")
    int(args.checkpoint_sha256, 16)

    checkpoint = args.checkpoint.resolve()
    checkpoint_model = checkpoint / "model.pt"
    if not checkpoint_model.is_file():
        raise FileNotFoundError(checkpoint_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    episodes_path = args.output_dir / "episodes.jsonl"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    if (
        result_path.exists()
        or episodes_path.exists()
        or telemetry_path.exists()
        or (args.output_dir / "eval_logs").exists()
    ):
        raise FileExistsError(f"Refusing to overwrite run in {args.output_dir}")

    import torch
    from libero.libero import benchmark
    from robot_experiments.robot_utils import set_seed_everywhere

    telemetry_logger = None
    if args.expected_gpu_uuid:
        from scripts.dynamic_compute.replay_m420b_rp_pep import normalize_gpu_uuid

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("collector requires exactly one visible CUDA device")
        visible_gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if normalize_gpu_uuid(visible_gpu_uuid) != normalize_gpu_uuid(
            args.expected_gpu_uuid
        ):
            raise RuntimeError(
                f"GPU UUID mismatch: expected {args.expected_gpu_uuid}, "
                f"visible {visible_gpu_uuid}"
            )
    else:
        visible_gpu_uuid = None

    if args.policy in EARLY_EXIT_POLICIES:
        import robot_experiments.libero.eval_libero_early_exit as eval_module
        from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
        from scripts.dynamic_compute.smoke_m1_telemetry import make_exit_controller

        cfg = eval_module.GenerateConfig(
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
            reseed_each_episode=False,
            seed=args.seed,
            run_id_note="m418-persistent-early-" + "_".join(map(str, task_ids)),
            vision_aggregation_enabled=False,
            rp_pep_enabled=args.policy == "rp_pep",
        )
        set_seed_everywhere(cfg.seed)
        model, device, _ = eval_module.initialize_and_load_model(cfg)
        exit_controller = (
            eval_module.initialize_exit_controller(cfg, model, None, device)
            if args.policy == "rp_pep"
            else make_exit_controller(cfg, model, device)
        )
        log_file, eval_log_path, _ = eval_module.setup_logging(
            cfg, model.config.action_head
        )
        telemetry_logger = SafeJSONLTelemetryLogger(telemetry_path, flush_every=10)
    else:
        import robot_experiments.libero.eval_libero as eval_module

        cfg = eval_module.GenerateConfig(
            pretrained_checkpoint=str(checkpoint),
            task_suite_name=args.task_suite,
            num_trials_per_task=args.num_episodes,
            action_head_flow_matching_inference_steps=args.fm_steps,
            local_log_dir=str(args.output_dir / "eval_logs"),
            save_rollout_video=False,
            save_rollout_video_path=str(args.output_dir),
            use_wandb=False,
            reseed_each_episode=False,
            seed=args.seed,
            run_id_note="m418-persistent-full-" + "_".join(map(str, task_ids)),
        )
        set_seed_everywhere(cfg.seed)
        model, device = eval_module.initialize_and_load_model(cfg)
        exit_controller = None
        log_file, eval_log_path, _ = eval_module.setup_logging(cfg)

    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_class != POLICY_MODEL_CLASSES[args.policy]:
        raise TypeError(
            f"{args.policy} expected {POLICY_MODEL_CLASSES[args.policy]}, got {model_class}"
        )
    base_task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if any(not 0 <= task_id < base_task_suite.n_tasks for task_id in task_ids):
        raise ValueError("one or more task ids are outside the suite")
    for task_id in task_ids:
        available_states = len(base_task_suite.get_task_init_states(task_id))
        if max(selected_episode_indices) >= available_states:
            raise ValueError(
                f"task {task_id} has {available_states} initial states, cannot "
                f"select indices {selected_episode_indices}"
            )
    task_suite = SelectedInitialStateTaskSuite(
        base_task_suite, selected_episode_indices
    )
    resize_size = eval_module.get_image_resize_size(cfg)

    expected_initial_hashes = {
        (task_id, episode_idx): sha256_array(
            np.asarray(base_task_suite.get_task_init_states(task_id)[episode_idx])
        )
        for task_id in task_ids
        for episode_idx in selected_episode_indices
    }
    episode_positions = defaultdict(int)
    active_task_id: int | None = None
    active_call_latencies: list[float] | None = None
    active_chunk_lengths: list[int] | None = None
    active_chunk_hashes: list[str] | None = None
    episode_records: list[dict[str, Any]] = []
    original_get_vla_action = eval_module.get_vla_action
    original_run_episode = eval_module.run_episode

    def timed_get_vla_action(*call_args: Any, **call_kwargs: Any):
        if (
            active_call_latencies is None
            or active_chunk_lengths is None
            or active_chunk_hashes is None
        ):
            raise RuntimeError("policy call occurred outside an active episode")
        start = time.perf_counter()
        actions = original_get_vla_action(*call_args, **call_kwargs)
        active_call_latencies.append((time.perf_counter() - start) * 1000.0)
        active_chunk_lengths.append(len(actions))
        active_chunk_hashes.append(sha256_array(np.asarray(actions)))
        return actions

    def recorded_run_episode(*call_args: Any, **call_kwargs: Any):
        nonlocal active_call_latencies, active_chunk_lengths, active_chunk_hashes
        if active_task_id is None:
            raise RuntimeError("episode started without an active task")
        episode_position = episode_positions[active_task_id]
        episode_idx = selected_episode_indices[episode_position]
        initial_state_index = 7 if args.policy in EARLY_EXIT_POLICIES else 6
        initial_state = (
            call_args[initial_state_index]
            if len(call_args) > initial_state_index
            else call_kwargs.get("initial_state")
        )
        initial_hash = sha256_array(np.asarray(initial_state))
        expected_hash = expected_initial_hashes[(active_task_id, episode_idx)]
        if initial_hash != expected_hash:
            raise ValueError("run_task selected an unexpected initial state")
        call_latencies: list[float] = []
        chunk_lengths: list[int] = []
        chunk_hashes: list[str] = []
        active_call_latencies = call_latencies
        active_chunk_lengths = chunk_lengths
        active_chunk_hashes = chunk_hashes
        wall_start = time.perf_counter()
        original_call_args = call_args
        original_call_kwargs = call_kwargs
        if args.policy in EARLY_EXIT_POLICIES:
            if len(call_args) > 10:
                mutable_call_args = list(call_args)
                mutable_call_args[10] = episode_idx
                original_call_args = tuple(mutable_call_args)
            else:
                original_call_kwargs = dict(call_kwargs)
                original_call_kwargs["episode_idx"] = episode_idx
        episode_seed = args.seed + active_task_id * 10_000 + episode_idx
        set_seed_everywhere(episode_seed)
        try:
            output = original_run_episode(*original_call_args, **original_call_kwargs)
        finally:
            active_call_latencies = None
            active_chunk_lengths = None
            active_chunk_hashes = None
        wall_seconds = time.perf_counter() - wall_start
        success = bool(output[0])
        row = {
            "status": "PASS",
            "policy": args.policy,
            "task_id": active_task_id,
            "episode_idx": episode_idx,
            "episode_seed": episode_seed,
            "initial_state_sha256": initial_hash,
            "success": success,
            "policy_calls": len(call_latencies),
            "action_chunk_lengths": chunk_lengths,
            "action_chunk_sha256": chunk_hashes,
            "latency_ms_by_call": call_latencies,
            "latency_ms_mean": _mean(call_latencies),
            "latency_ms_median": (
                statistics.median(call_latencies) if call_latencies else None
            ),
            "latency_ms_total": sum(call_latencies),
            "wall_seconds": wall_seconds,
            "exit_mean_ratio": (
                float(output[2])
                if args.policy in EARLY_EXIT_POLICIES and output[2] is not None
                else None
            ),
        }
        if not call_latencies or any(length <= 0 for length in chunk_lengths):
            row["status"] = "FAIL"
        with episodes_path.open("a", encoding="utf-8") as episode_file:
            episode_file.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
            episode_file.flush()
        episode_records.append(row)
        episode_positions[active_task_id] += 1
        return output

    eval_module.get_vla_action = timed_get_vla_action
    eval_module.run_episode = recorded_run_episode
    total_episodes = 0
    total_successes = 0
    total_exit_sum = 0.0
    total_exit_count = 0
    wall_start = time.perf_counter()
    try:
        for task_id in task_ids:
            active_task_id = task_id
            if args.policy in EARLY_EXIT_POLICIES:
                (
                    total_episodes,
                    total_successes,
                    total_exit_sum,
                    total_exit_count,
                ) = eval_module.run_task(
                    cfg=cfg,
                    task_suite=task_suite,
                    task_id=task_id,
                    model=model,
                    exit_controller=exit_controller,
                    device=device,
                    num_tasks=len(task_ids),
                    resize_size=resize_size,
                    total_episodes=total_episodes,
                    total_successes=total_successes,
                    log_file=log_file,
                    total_exit_mean_sum=total_exit_sum,
                    total_exit_mean_count=total_exit_count,
                    telemetry_logger=telemetry_logger,
                )
            else:
                total_episodes, total_successes = eval_module.run_task(
                    cfg=cfg,
                    task_suite=task_suite,
                    task_id=task_id,
                    model=model,
                    device=device,
                    num_tasks=len(task_ids),
                    resize_size=resize_size,
                    total_episodes=total_episodes,
                    total_successes=total_successes,
                    log_file=log_file,
                )
    finally:
        eval_module.get_vla_action = original_get_vla_action
        eval_module.run_episode = original_run_episode
        if telemetry_logger is not None:
            telemetry_logger.close()
        log_file.close()
    wall_seconds = time.perf_counter() - wall_start

    telemetry_records: list[dict[str, Any]] = []
    telemetry_errors = 0
    telemetry_last_error = None
    if telemetry_logger is not None:
        telemetry_errors = telemetry_logger.error_count
        telemetry_last_error = telemetry_logger.last_error
        telemetry_records = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in telemetry_records:
            by_episode[str(record["episode_id"])].append(record)
        for row in episode_records:
            episode_id = (
                f"{args.task_suite}:task{row['task_id']}:episode{row['episode_idx']}"
            )
            selected = by_episode.get(episode_id, [])
            row["telemetry_calls"] = len(selected)
            row["exit_layer_counts"] = dict(
                sorted(Counter(int(record["exit_layer"]) for record in selected).items())
            )
            row["exit_layer_sequence"] = [
                int(record["exit_layer"]) for record in selected
            ]
            row["fm_calls_by_policy_call"] = [
                int(record["fm_calls"]) for record in selected
            ]
            row["fm_calls_total"] = sum(int(record["fm_calls"]) for record in selected)
            row["fm_steps_total"] = sum(
                int(record["fm_steps_total"]) for record in selected
            )
            if len(selected) != int(row["policy_calls"]):
                row["status"] = "FAIL"

    engineering_ok = engineering_status_ok(
        policy=args.policy,
        model_class=model_class,
        task_ids=task_ids,
        episodes_per_task=args.num_episodes,
        episode_start_index=args.episode_start_index,
        episode_indices=selected_episode_indices,
        episode_records=episode_records,
        telemetry_errors=telemetry_errors,
    )
    source_status = git_output("status", "--porcelain=v1")
    result = {
        "status": "PASS" if engineering_ok else "FAIL",
        "scope": (
            "m420b_rp_pep_closed_loop_shard"
            if args.policy == "rp_pep"
            else "m418_persistent_closed_loop_counterfactual_shard"
        ),
        "policy": args.policy,
        "model_class": model_class,
        "early_exit_enabled": args.policy in EARLY_EXIT_POLICIES,
        "productive_exit_enabled": args.policy == "rp_pep",
        "vision_aggregation_enabled": False,
        "checkpoint": str(checkpoint_model),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_bytes": checkpoint_model.stat().st_size,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "seed": args.seed,
        "episodes_per_task": args.num_episodes,
        "episode_indices": selected_episode_indices,
        "episode_start_index": (
            args.episode_start_index if args.episode_index is None else None
        ),
        "episode_stop_index_exclusive": (
            args.episode_start_index + args.num_episodes
            if args.episode_index is None
            else None
        ),
        "fm_steps": args.fm_steps,
        "completed_episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "policy_calls": sum(int(row["policy_calls"]) for row in episode_records),
        "wall_seconds": wall_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "physical_gpu_uuid_visible": visible_gpu_uuid,
        "physical_gpu_uuid_nvidia_smi": args.expected_gpu_uuid,
        "task_summaries": _task_summaries(task_ids, episode_records),
        "episode_records": episode_records,
        "telemetry_records": len(telemetry_records),
        "telemetry_errors": telemetry_errors,
        "telemetry_last_error": telemetry_last_error,
        "eval_log": str(eval_log_path),
        "eval_log_sha256": sha256_file(Path(eval_log_path)),
        "episodes_path": str(episodes_path),
        "episodes_sha256": sha256_file(episodes_path),
        "telemetry_path": str(telemetry_path) if telemetry_logger is not None else None,
        "telemetry_sha256": (
            sha256_file(telemetry_path) if telemetry_logger is not None else None
        ),
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
