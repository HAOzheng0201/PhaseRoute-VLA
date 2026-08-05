"""One-GPU paired smoke test for M4 static visual-token aggregation."""

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

from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
from a1.vla.dynamic_compute.vision_aggregation import StaticVisionAggregationConfig
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
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--keep-tokens", default="288,128,96,64")
    parser.add_argument("--bank-tokens", type=int, default=288)
    parser.add_argument("--min-tokens-per-crop", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def action_digest(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()


def timed_policy_call(device, call):
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    action = call()
    torch.cuda.synchronize(device)
    return action, (time.perf_counter_ns() - start_ns) / 1e6


def main() -> None:
    args = parse_args()
    keep_budgets = tuple(int(value) for value in args.keep_tokens.split(",") if value)
    if not keep_budgets or any(value < 1 for value in keep_budgets):
        raise ValueError("--keep-tokens must contain positive comma-separated integers")
    if any(value > args.bank_tokens for value in keep_budgets):
        raise ValueError("every keep budget must be <= --bank-tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

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

    variants: dict[str, StaticVisionAggregationConfig | None] = {
        "baseline": None,
        "disabled": StaticVisionAggregationConfig(enabled=False),
        "keep_all": StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=10_000,
            min_tokens_per_crop=args.min_tokens_per_crop,
        ),
    }
    for keep_tokens in keep_budgets:
        variants[f"keep_{keep_tokens}"] = StaticVisionAggregationConfig(
            enabled=True,
            keep_tokens=keep_tokens,
            bank_tokens=args.bank_tokens,
            min_tokens_per_crop=args.min_tokens_per_crop,
        )
    controllers = {
        variant: make_exit_controller(cfg, model, device) for variant in variants
    }

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    initial_states = task_suite.get_task_init_states(args.task_id)
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
    )
    actions: dict[str, np.ndarray] = {}
    latency_ms: dict[str, float] = {}
    exit_layers: dict[str, list[int]] = {variant: [] for variant in variants}

    def make_log_collector(variant: str):
        def collect(message: str) -> None:
            key = "Exit by exit_controller, block_idx:"
            if key in message:
                exit_layers[variant].append(
                    int(message.split(key)[-1].strip().split()[0].strip(","))
                )

        return collect

    try:
        env.reset()
        observation = env.set_init_state(initial_states[args.episode_id])
        for _ in range(cfg.num_steps_wait):
            observation, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        policy_observation, _ = prepare_observation(
            observation,
            model.config.vision_backbone.image_default_input_size,
        )

        with SafeJSONLTelemetryLogger(telemetry_path, flush_every=1) as logger:
            for variant, aggregation_config in variants.items():
                set_seed_everywhere(args.seed)
                controllers[variant].set_timestep(cfg.num_steps_wait)
                action_list, elapsed_ms = timed_policy_call(
                    device,
                    lambda variant=variant, aggregation_config=aggregation_config: get_vla_action(
                        cfg,
                        model,
                        device,
                        copy.deepcopy(policy_observation),
                        task_description,
                        controllers[variant],
                        output_hidden_states=True,
                        log_fn=make_log_collector(variant),
                        telemetry_logger=logger,
                        telemetry_context={
                            "episode_id": (
                                f"{args.task_suite}:task{args.task_id}:"
                                f"episode{args.episode_id}:{variant}"
                            ),
                            "step_id": cfg.num_steps_wait,
                            "task_id": args.task_id,
                            "previous_action": None,
                        },
                        vision_aggregation_config=aggregation_config,
                    ),
                )
                actions[variant] = np.stack(action_list)
                latency_ms[variant] = elapsed_ms
            telemetry_errors = logger.error_count
            telemetry_last_error = logger.last_error
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records_by_variant = {
        record["episode_id"].rsplit(":", 1)[-1]: record for record in records
    }
    baseline = actions["baseline"]
    comparisons = {}
    for variant, action in actions.items():
        comparisons[variant] = {
            "action_sha256": action_digest(action),
            "exact_equal_to_baseline": bool(np.array_equal(action, baseline)),
            "max_abs_diff_to_baseline": float(np.max(np.abs(action - baseline))),
            "finite": bool(np.isfinite(action).all()),
            "latency_ms": latency_ms[variant],
            "exit_layers": exit_layers[variant],
            "active_tokens": records_by_variant[variant]["active_tokens_by_layer"][0],
            "vision_aggregation": records_by_variant[variant]
            .get("extra", {})
            .get("vision_aggregation"),
        }

    status_ok = (
        len(records) == len(variants)
        and telemetry_errors == 0
        and comparisons["disabled"]["exact_equal_to_baseline"]
        and comparisons["keep_all"]["exact_equal_to_baseline"]
        and all(item["finite"] for item in comparisons.values())
        and all(len(item["exit_layers"]) == 1 for item in comparisons.values())
        and all(
            comparisons[f"keep_{keep}"]["vision_aggregation"] is not None
            for keep in keep_budgets
        )
    )
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "checkpoint": str(checkpoint),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "seed": args.seed,
        "fm_steps": args.fm_steps,
        "gpu": torch.cuda.get_device_name(device),
        "bank_tokens": args.bank_tokens,
        "keep_budgets": list(keep_budgets),
        "min_tokens_per_crop": args.min_tokens_per_crop,
        "action_shape": list(baseline.shape),
        "telemetry_records": len(records),
        "telemetry_errors": telemetry_errors,
        "telemetry_last_error": telemetry_last_error,
        "comparisons": comparisons,
        "note": (
            "Single-observation smoke validates real sequence compaction and regression "
            "contracts; it does not establish rollout success or Pareto quality."
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
