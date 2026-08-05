"""One-GPU regression: M2 phase-cache off vs on for one policy call."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from libero.libero import benchmark

from a1.vla.dynamic_compute.phase_cache import SafePhaseCacheWriter
from robot_experiments.libero.eval_libero_early_exit import (
    GenerateConfig,
    initialize_and_load_model,
    prepare_observation,
)
from robot_experiments.libero.exit_vla_utils import get_vla_action
from robot_experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from robot_experiments.robot_utils import set_seed_everywhere
from smoke_m1_telemetry import make_exit_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def action_digest(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "m2_gpu_smoke_result.json"
    phase_cache_dir = args.output_dir / "phase_calls"
    if result_path.exists() or (phase_cache_dir / "manifest.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite an existing smoke run in {args.output_dir}")

    checkpoint = args.checkpoint.resolve()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=args.task_suite,
        action_head_flow_matching_inference_steps=args.fm_steps,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        use_wandb=False,
        save_rollout_video=False,
    )

    set_seed_everywhere(args.seed)
    model, device, _ = initialize_and_load_model(cfg)
    baseline_controller = make_exit_controller(cfg, model, device)
    cache_controller = make_exit_controller(cfg, model, device)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id {args.task_id} is outside [0, {task_suite.n_tasks})")
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_id < len(initial_states):
        raise ValueError(f"episode-id {args.episode_id} is outside [0, {len(initial_states)})")
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    try:
        env.reset()
        observation = env.set_init_state(initial_states[args.episode_id])
        for _ in range(cfg.num_steps_wait):
            observation, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        policy_observation, _ = prepare_observation(
            observation,
            model.config.vision_backbone.image_default_input_size,
        )

        set_seed_everywhere(args.seed)
        baseline_controller.set_timestep(cfg.num_steps_wait)
        baseline_actions = get_vla_action(
            cfg,
            model,
            device,
            copy.deepcopy(policy_observation),
            task_description,
            baseline_controller,
            output_hidden_states=True,
        )

        set_seed_everywhere(args.seed)
        cache_controller.set_timestep(cfg.num_steps_wait)
        phase_writer = SafePhaseCacheWriter(phase_cache_dir, summary_dtype="float16")
        try:
            cache_actions = get_vla_action(
                cfg,
                model,
                device,
                copy.deepcopy(policy_observation),
                task_description,
                cache_controller,
                output_hidden_states=True,
                phase_cache_writer=phase_writer,
                phase_cache_context={
                    "episode_id": (
                        f"{args.task_suite}:task{args.task_id}:episode{args.episode_id}"
                    ),
                    "step_id": cfg.num_steps_wait,
                    "task_id": args.task_id,
                    "previous_action": None,
                },
            )
        finally:
            phase_writer.close()

        baseline_array = np.stack(baseline_actions)
        cache_array = np.stack(cache_actions)
        exact_equal = bool(np.array_equal(baseline_array, cache_array))
        max_abs_diff = float(np.max(np.abs(baseline_array - cache_array)))
        if not exact_equal:
            raise AssertionError(
                f"Phase-cache callback changed the action: max_abs_diff={max_abs_diff:.9g}"
            )
        if phase_writer.error_count:
            raise AssertionError(f"Phase cache writer failed: {phase_writer.last_error}")
        manifest_records = [
            json.loads(line)
            for line in (phase_cache_dir / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(manifest_records) != 1:
            raise AssertionError(
                f"Expected one phase-cache record, got {len(manifest_records)}"
            )
        manifest = manifest_records[0]
        with np.load(phase_cache_dir / manifest["array_path"]) as shard:
            shard_shapes = {name: list(shard[name].shape) for name in shard.files}
            all_finite = all(np.isfinite(shard[name]).all() for name in shard.files)
        if not all_finite:
            raise AssertionError("Phase cache contains a non-finite value")

        result = {
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "episode_id": args.episode_id,
            "seed": args.seed,
            "fm_steps": args.fm_steps,
            "gpu": torch.cuda.get_device_name(0),
            "action_shape": list(baseline_array.shape),
            "baseline_sha256": action_digest(baseline_array),
            "phase_cache_sha256": action_digest(cache_array),
            "exact_equal": exact_equal,
            "max_abs_diff": max_abs_diff,
            "phase_cache_records": len(manifest_records),
            "phase_cache_errors": phase_writer.error_count,
            "phase_cache_last_error": phase_writer.last_error,
            "summary_counts": manifest["summary_counts"],
            "shard_shapes": shard_shapes,
            "all_cache_values_finite": all_finite,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"result={result_path}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
