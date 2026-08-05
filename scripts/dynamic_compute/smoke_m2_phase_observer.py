"""One-GPU paired regression for the observer-only M2 PhaseEstimator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from libero.libero import benchmark

from a1.vla.dynamic_compute.phase_observer import SafePhaseObserver
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
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
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
    result_path = args.output_dir / "result.json"
    observer_path = args.output_dir / "phase_observer_calls.jsonl"
    if result_path.exists() or observer_path.exists():
        raise FileExistsError(f"Refusing to overwrite smoke run in {args.output_dir}")

    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
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
    observer_controller = make_exit_controller(cfg, model, device)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    initial_states = task_suite.get_task_init_states(args.task_id)
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
    )
    observer = SafePhaseObserver(
        args.phase_checkpoint,
        observer_path,
        device=device,
        history_len=8,
    )
    try:
        env.reset()
        observation = env.set_init_state(initial_states[args.episode_id])
        for _ in range(cfg.num_steps_wait):
            observation, _, _, _ = env.step(
                get_libero_dummy_action(cfg.model_family)
            )
        policy_observation, _ = prepare_observation(
            observation,
            model.config.vision_backbone.image_default_input_size,
        )

        set_seed_everywhere(args.seed)
        baseline_controller.set_timestep(cfg.num_steps_wait)
        torch.cuda.synchronize(device)
        baseline_start = time.perf_counter_ns()
        baseline_actions = get_vla_action(
            cfg,
            model,
            device,
            copy.deepcopy(policy_observation),
            task_description,
            baseline_controller,
            output_hidden_states=True,
        )
        torch.cuda.synchronize(device)
        baseline_wall_ms = (time.perf_counter_ns() - baseline_start) / 1e6

        set_seed_everywhere(args.seed)
        observer_controller.set_timestep(cfg.num_steps_wait)
        torch.cuda.synchronize(device)
        observer_start = time.perf_counter_ns()
        observer_actions = get_vla_action(
            cfg,
            model,
            device,
            copy.deepcopy(policy_observation),
            task_description,
            observer_controller,
            output_hidden_states=True,
            phase_cache_writer=observer,
            phase_cache_context={
                "episode_id": (
                    f"{args.task_suite}:task{args.task_id}:episode{args.episode_id}"
                ),
                "step_id": cfg.num_steps_wait,
                "task_id": args.task_id,
                "previous_action": None,
            },
        )
        torch.cuda.synchronize(device)
        observer_wall_ms = (time.perf_counter_ns() - observer_start) / 1e6
    finally:
        observer.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    baseline_array = np.stack(baseline_actions)
    observer_array = np.stack(observer_actions)
    exact_equal = bool(np.array_equal(baseline_array, observer_array))
    max_abs_diff = float(np.max(np.abs(baseline_array - observer_array)))
    records = [
        json.loads(line)
        for line in observer_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    status_ok = exact_equal and observer.error_count == 0 and len(records) == 1
    record = records[0] if records else {}
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "observer_only": True,
        "controls_early_exit": False,
        "a1_checkpoint": str(args.checkpoint.resolve()),
        "phase_checkpoint": str(args.phase_checkpoint),
        "phase_checkpoint_sha256": observer.checkpoint_sha256,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "seed": args.seed,
        "fm_steps": args.fm_steps,
        "gpu": torch.cuda.get_device_name(device),
        "action_shape": list(baseline_array.shape),
        "baseline_action_sha256": action_digest(baseline_array),
        "observer_action_sha256": action_digest(observer_array),
        "exact_equal": exact_equal,
        "max_abs_diff": max_abs_diff,
        "observer_records": len(records),
        "observer_errors": observer.error_count,
        "observer_last_error": observer.last_error,
        "baseline_wall_ms": baseline_wall_ms,
        "observer_wall_ms": observer_wall_ms,
        "paired_wall_overhead_percent": (
            (observer_wall_ms - baseline_wall_ms) / baseline_wall_ms * 100.0
        ),
        "phase_estimator_latency_ms": record.get("latency_ms"),
        "phase_estimator_to_baseline_wall_percent": (
            float(record["latency_ms"]) / baseline_wall_ms * 100.0
            if record
            else None
        ),
        "phase_prediction": {
            name: record.get(name)
            for name in (
                "history_count",
                "progress",
                "boundary_prob",
                "boundary_threshold",
                "boundary_pred",
                "uncertainty",
            )
        },
        "latency_note": (
            "Single paired call is a smoke measurement; use repeated observer "
            "replay for stable latency statistics."
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
