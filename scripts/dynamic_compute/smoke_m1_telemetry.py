"""One-GPU M1 regression: telemetry off vs on for the same LIBERO policy call."""

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

from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
from a1.vla.value_net import ActionValueNet, ExitController
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


def make_exit_controller(cfg: GenerateConfig, model, device) -> ExitController:
    exit_layers = model.get_all_exit_idx(cfg.exit_interval)
    value_net = ActionValueNet(
        exit_list=exit_layers,
        exit_head=model.action_head,
        model=model,
        interval=cfg.exit_interval,
        threshold_type=cfg.threshold_type,
        anchor=False,
    )
    controller = ExitController(
        value_net,
        exit_id_list=exit_layers,
        steps_per_stage=cfg.steps_per_stage,
        leq=True,
        exit_dist=cfg.exit_dist,
        max_layer=model.config.n_layers,
    )
    thresholds_path = Path(cfg.pretrained_checkpoint) / (
        f"exit_thresholds_{cfg.task_suite_name}_{cfg.exit_dist}_{cfg.exit_ratio}.json"
    )
    with thresholds_path.open("r", encoding="utf-8") as input_file:
        controller.thresholds = {
            int(layer): float(value)
            for layer, value in json.load(input_file).items()
        }
    missing = set(exit_layers) - set(controller.thresholds)
    if missing:
        raise ValueError(f"Threshold file is missing exit layers: {sorted(missing)}")
    controller.to(device)
    controller.eval()
    return controller


def action_digest(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    telemetry_controller = make_exit_controller(cfg, model, device)

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

        telemetry_path = args.output_dir / "m1_policy_calls.jsonl"
        set_seed_everywhere(args.seed)
        telemetry_controller.set_timestep(cfg.num_steps_wait)
        with SafeJSONLTelemetryLogger(telemetry_path, flush_every=1) as telemetry_logger:
            telemetry_actions = get_vla_action(
                cfg,
                model,
                device,
                copy.deepcopy(policy_observation),
                task_description,
                telemetry_controller,
                output_hidden_states=True,
                telemetry_logger=telemetry_logger,
                telemetry_context={
                    "episode_id": (
                        f"{args.task_suite}:task{args.task_id}:episode{args.episode_id}"
                    ),
                    "step_id": cfg.num_steps_wait,
                    "task_id": args.task_id,
                    "previous_action": None,
                },
            )
            telemetry_errors = telemetry_logger.error_count
            telemetry_last_error = telemetry_logger.last_error

        baseline_array = np.stack(baseline_actions)
        telemetry_array = np.stack(telemetry_actions)
        exact_equal = bool(np.array_equal(baseline_array, telemetry_array))
        max_abs_diff = float(np.max(np.abs(baseline_array - telemetry_array)))
        if not exact_equal:
            raise AssertionError(
                f"Telemetry changed the action: max_abs_diff={max_abs_diff:.9g}"
            )
        telemetry_records = telemetry_path.read_text(encoding="utf-8").splitlines()
        if len(telemetry_records) != 1:
            raise AssertionError(f"Expected one telemetry record, got {len(telemetry_records)}")
        telemetry_payload = json.loads(telemetry_records[0])

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
            "telemetry_sha256": action_digest(telemetry_array),
            "exact_equal": exact_equal,
            "max_abs_diff": max_abs_diff,
            "telemetry_records": len(telemetry_records),
            "telemetry_errors": telemetry_errors,
            "telemetry_last_error": telemetry_last_error,
            "exit_layer": telemetry_payload["exit_layer"],
            "fm_calls": telemetry_payload["fm_calls"],
            "fm_steps_total": telemetry_payload["fm_steps_total"],
            "active_tokens": telemetry_payload["active_tokens_by_layer"][0],
            "visual_tokens": telemetry_payload["extra"]["visual_tokens"],
        }
        result_path = args.output_dir / "m1_gpu_smoke_result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"result={result_path}")
        print(f"telemetry={telemetry_path}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
