"""One-GPU paired smoke test for opt-in M3 phase-aware depth routing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from libero.libero import benchmark

from a1.vla.dynamic_compute.exit_policy import (
    phase_exit_policy_config_for_ablation,
)
from a1.vla.dynamic_compute.phase_depth_runtime import SafePhaseDepthRuntime
from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
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
    parser.add_argument(
        "--ablation",
        choices=("min_depth", "threshold", "joint"),
        default="joint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def action_digest(action: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()


def exit_log_collector(target: list[int]):
    def collect(message: str) -> None:
        key = "Exit by exit_controller, block_idx:"
        if key in message:
            target.append(int(message.split(key)[-1].strip().split()[0].strip(",")))

    return collect


def timed_policy_call(*, device, call):
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    actions = call()
    torch.cuda.synchronize(device)
    return actions, (time.perf_counter_ns() - start_ns) / 1e6


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "phase_depth_calls.jsonl"
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"Refusing to overwrite smoke run in {args.output_dir}")

    checkpoint = args.checkpoint.resolve()
    phase_checkpoint = args.phase_checkpoint.resolve()
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
    disabled_controller = make_exit_controller(cfg, model, device)
    phase_controller = make_exit_controller(cfg, model, device)
    runtime = SafePhaseDepthRuntime(
        phase_checkpoint,
        device=device,
        eligible_exit_layers=tuple(phase_controller.exit_id_list),
        history_len=8,
        fm_steps_per_exit=args.fm_steps,
        exit_policy_config=phase_exit_policy_config_for_ablation(args.ablation),
    )

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    initial_states = task_suite.get_task_init_states(args.task_id)
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
    )
    context = {
        "episode_id": f"{args.task_suite}:task{args.task_id}:episode{args.episode_id}",
        "step_id": cfg.num_steps_wait,
        "task_id": args.task_id,
        "previous_action": None,
    }
    exit_layers = {"baseline": [], "disabled": [], "phase_depth": []}

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
        baseline_actions, baseline_ms = timed_policy_call(
            device=device,
            call=lambda: get_vla_action(
                cfg,
                model,
                device,
                copy.deepcopy(policy_observation),
                task_description,
                baseline_controller,
                output_hidden_states=True,
                log_fn=exit_log_collector(exit_layers["baseline"]),
            ),
        )

        set_seed_everywhere(args.seed)
        disabled_controller.set_timestep(cfg.num_steps_wait)
        disabled_actions, disabled_ms = timed_policy_call(
            device=device,
            call=lambda: get_vla_action(
                cfg,
                model,
                device,
                copy.deepcopy(policy_observation),
                task_description,
                disabled_controller,
                output_hidden_states=True,
                log_fn=exit_log_collector(exit_layers["disabled"]),
                phase_depth_runtime=SimpleNamespace(enabled=False),
                phase_depth_context=context,
            ),
        )

        set_seed_everywhere(args.seed)
        phase_controller.set_timestep(cfg.num_steps_wait)
        with SafeJSONLTelemetryLogger(telemetry_path, flush_every=1) as logger:
            phase_actions, phase_ms = timed_policy_call(
                device=device,
                call=lambda: get_vla_action(
                    cfg,
                    model,
                    device,
                    copy.deepcopy(policy_observation),
                    task_description,
                    phase_controller,
                    output_hidden_states=True,
                    log_fn=exit_log_collector(exit_layers["phase_depth"]),
                    telemetry_logger=logger,
                    telemetry_context=context,
                    phase_depth_runtime=runtime,
                    phase_depth_context=context,
                ),
            )
            telemetry_errors = logger.error_count
            telemetry_last_error = logger.last_error
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    baseline = np.stack(baseline_actions)
    disabled = np.stack(disabled_actions)
    phase = np.stack(phase_actions)
    disabled_exact = bool(np.array_equal(baseline, disabled))
    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[0] if records else {}
    phase_plan = record.get("extra", {}).get("phase_plan") or {}
    status_ok = (
        disabled_exact
        and len(records) == 1
        and telemetry_errors == 0
        and runtime.records_prepared == 1
        and runtime.error_count == 0
        and bool(phase_plan)
        and len(exit_layers["baseline"]) == 1
        and len(exit_layers["disabled"]) == 1
        and len(exit_layers["phase_depth"]) == 1
    )
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "checkpoint": str(checkpoint),
        "phase_checkpoint": str(phase_checkpoint),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "seed": args.seed,
        "fm_steps": args.fm_steps,
        "ablation": args.ablation,
        "gpu": torch.cuda.get_device_name(device),
        "action_shape": list(baseline.shape),
        "baseline_action_sha256": action_digest(baseline),
        "disabled_action_sha256": action_digest(disabled),
        "phase_action_sha256": action_digest(phase),
        "disabled_exact_equal": disabled_exact,
        "disabled_max_abs_diff": float(np.max(np.abs(baseline - disabled))),
        "phase_exact_equal_to_baseline": bool(np.array_equal(baseline, phase)),
        "phase_max_abs_diff_to_baseline": float(np.max(np.abs(baseline - phase))),
        "exit_layers": exit_layers,
        "latency_ms": {
            "baseline": baseline_ms,
            "disabled": disabled_ms,
            "phase_depth": phase_ms,
        },
        "phase_plan": phase_plan,
        "phase_profile_id": record.get("profile_id"),
        "phase_progress": record.get("progress"),
        "phase_boundary_prob": record.get("boundary_prob"),
        "phase_uncertainty": record.get("uncertainty"),
        "phase_fm_calls": record.get("fm_calls"),
        "phase_fm_steps_total": record.get("fm_steps_total"),
        "telemetry_records": len(records),
        "telemetry_errors": telemetry_errors,
        "telemetry_last_error": telemetry_last_error,
        "runtime_plans": runtime.records_prepared,
        "runtime_errors": runtime.error_count,
        "runtime_last_error": runtime.last_error,
        "note": "Single-call smoke validates wiring, not task success or Pareto quality.",
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
