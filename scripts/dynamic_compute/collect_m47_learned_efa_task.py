"""Evaluate baseline or distilled EFA144 with paired LIBERO settings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark

from a1.vla.dynamic_compute.exit_policy import (
    phase_exit_policy_config_for_ablation,
)
from a1.vla.dynamic_compute.learnable_vision_runtime import (
    load_distilled_vision_aggregator,
)
from a1.vla.dynamic_compute.phase_depth_runtime import SafePhaseDepthRuntime
from a1.vla.dynamic_compute.phase_vision_runtime import (
    PhaseProfileVisionRouter,
    make_exit_controller_profile_provider,
    make_phase_runtime_profile_provider,
)
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


MODES = (
    "baseline",
    "learned_efa144",
    "joint_learned_efa144",
    "joint_risk_full_token_efa144",
    "joint_contact_full_token_efa144",
    "phase_width_contact_full_token_efa144",
    "phase_width_hysteresis_full_token_efa144",
    "phase_width_uncertainty_hysteresis_full_token_efa144",
)
RISK_FULL_TOKEN_PROFILES = {
    "joint_risk_full_token_efa144": ("B3",),
    "joint_contact_full_token_efa144": ("B1", "B3"),
    "phase_width_contact_full_token_efa144": ("B1", "B3"),
    "phase_width_hysteresis_full_token_efa144": ("B3",),
    "phase_width_uncertainty_hysteresis_full_token_efa144": ("B3",),
}
FULL_TOKEN_HOLD_CALLS = {
    "phase_width_hysteresis_full_token_efa144": 64,
    "phase_width_uncertainty_hysteresis_full_token_efa144": 64,
}
FULL_TOKEN_UNCERTAINTY_THRESHOLDS = {
    "phase_width_uncertainty_hysteresis_full_token_efa144": 0.06,
}
SUCCESS_PATTERN = re.compile(r"^Success: (True|False)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="learned_efa144")
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--efa-checkpoint", type=Path)
    parser.add_argument("--phase-checkpoint", type=Path)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def main() -> None:
    args = parse_args()
    learned = args.mode != "baseline"
    joint = args.mode in {
        "joint_learned_efa144",
        "joint_risk_full_token_efa144",
        "joint_contact_full_token_efa144",
        "phase_width_contact_full_token_efa144",
        "phase_width_hysteresis_full_token_efa144",
        "phase_width_uncertainty_hysteresis_full_token_efa144",
    }
    width_only = args.mode in {
        "phase_width_contact_full_token_efa144",
        "phase_width_hysteresis_full_token_efa144",
        "phase_width_uncertainty_hysteresis_full_token_efa144",
    }
    risk_full_token_profiles = RISK_FULL_TOKEN_PROFILES.get(args.mode)
    full_token_hold_calls = FULL_TOKEN_HOLD_CALLS.get(args.mode, 0)
    full_token_uncertainty_threshold = FULL_TOKEN_UNCERTAINTY_THRESHOLDS.get(
        args.mode
    )
    risk_full_token = risk_full_token_profiles is not None
    if learned and args.efa_checkpoint is None:
        raise ValueError(f"mode={args.mode} requires --efa-checkpoint")
    if joint and args.phase_checkpoint is None:
        raise ValueError("joint mode requires --phase-checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    telemetry_path = args.output_dir / "policy_calls.jsonl"
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

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
        run_id_note=f"m47-{args.mode}-task{args.task_id}",
        vision_aggregation_enabled=False,
    )
    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    exit_controller = make_exit_controller(cfg, model, device)
    loaded_efa = None
    if learned:
        loaded_efa = load_distilled_vision_aggregator(
            args.efa_checkpoint.resolve(),
            device=device,
            expected_hidden_dim=model.config.d_model,
            expected_teacher_checkpoint_sha256=args.checkpoint_sha256,
        )
    phase_runtime = None
    if joint:
        phase_runtime = SafePhaseDepthRuntime(
            args.phase_checkpoint.resolve(),
            device=device,
            eligible_exit_layers=tuple(exit_controller.exit_id_list),
            history_len=8,
            fm_steps_per_exit=args.fm_steps,
            exit_policy_config=phase_exit_policy_config_for_ablation("joint"),
        )
    runtime_aggregator = loaded_efa.model if loaded_efa is not None else None
    if risk_full_token:
        profile_provider = (
            make_phase_runtime_profile_provider(phase_runtime)
            if width_only
            else make_exit_controller_profile_provider(exit_controller)
        )
        runtime_aggregator = PhaseProfileVisionRouter(
            loaded_efa.model,
            profile_provider=profile_provider,
            full_token_profiles=risk_full_token_profiles,
            full_token_hold_calls=full_token_hold_calls,
            full_token_uncertainty_threshold=(
                full_token_uncertainty_threshold
            ),
        ).to(device)
        runtime_aggregator.eval()

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError("task-id is outside the selected suite")
    resize_size = get_image_resize_size(cfg)
    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    logger = SafeJSONLTelemetryLogger(telemetry_path, flush_every=25)
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
            log_file=log_file,
            telemetry_logger=logger,
            phase_depth_runtime=phase_runtime,
            learnable_vision_aggregator=runtime_aggregator,
            phase_depth_control_enabled=not width_only,
        )
    finally:
        logger.close()
        log_file.close()

    records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vision_events = [
        record.get("extra", {}).get("vision_aggregation") for record in records
    ]
    vision_events = [event for event in vision_events if event]
    errors = [
        event
        for record in records
        for event in record.get("extra", {}).get("exit_events", [])
        if event.get("event") == "vision_aggregation_error"
    ]
    plans = [record.get("extra", {}).get("phase_plan") for record in records]
    plans = [plan for plan in plans if plan]
    eval_text = Path(eval_log_path).read_text(encoding="utf-8")
    episode_successes = [
        match == "True" for match in SUCCESS_PATTERN.findall(eval_text)
    ]
    latencies = [float(record["latency_ms"]) for record in records]
    exit_layers = [int(record["exit_layer"]) for record in records]
    expected_tokens = (
        loaded_efa.config.output_tokens if loaded_efa is not None else None
    )
    expected_vision_events = len(records) if learned else 0
    paired_vision_plans = [
        (
            record.get("extra", {}).get("vision_aggregation"),
            record.get("extra", {}).get("phase_plan"),
        )
        for record in records
    ]
    if risk_full_token:
        risk_route_ok = all(
            event
            and plan
            and event.get("aggregation_kind") == "phase_profile_learned"
            and event.get("phase_profile_name") == plan.get("profile_name")
            and (
                (
                    plan.get("profile_name") not in risk_full_token_profiles
                    and not (
                        full_token_uncertainty_threshold is not None
                        and float(plan.get("uncertainty", 0.0))
                        >= full_token_uncertainty_threshold
                    )
                )
                or event.get("full_token_fallback") is True
            )
            and (
                (
                    event.get("full_token_fallback") is True
                    and event.get("vision_route") in {
                        "full_token",
                        "full_token_hold",
                    }
                    and event.get("status") == "keep_all"
                    and int(event["kept_tokens"]) == int(event["original_tokens"])
                )
                or (
                    event.get("vision_route") == "learned_efa"
                    and event.get("full_token_fallback") is False
                    and event.get("status") == "compressed"
                    and int(event["kept_tokens"]) == expected_tokens
                )
            )
            for event, plan in paired_vision_plans
        )
    else:
        risk_route_ok = True
    learned_route_ok = (
        risk_route_ok
        if risk_full_token
        else all(
            event.get("aggregation_kind") == "learned"
            and int(event["kept_tokens"]) == expected_tokens
            for event in vision_events
        )
    )
    run_ok = (
        episodes == args.num_episodes
        and len(episode_successes) == episodes
        and sum(episode_successes) == successes
        and logger.error_count == 0
        and not errors
        and len(vision_events) == expected_vision_events
        and learned_route_ok
        and len(plans) == (len(records) if joint else 0)
        and (phase_runtime is None or phase_runtime.error_count == 0)
    )
    result = {
        "status": "PASS" if run_ok else "FAIL",
        "scope": (
            "m415_phase_width_uncertainty_hysteresis_rollout"
            if args.mode
            == "phase_width_uncertainty_hysteresis_full_token_efa144"
            else (
                "m414_phase_width_hysteresis_rollout"
                if args.mode == "phase_width_hysteresis_full_token_efa144"
            else (
                "m413_phase_width_only_contact_rollout"
                if width_only
                else (
                    "m412_phase_contact_full_token_rollout"
                    if args.mode == "joint_contact_full_token_efa144"
                    else (
                        "m411_phase_risk_full_token_rollout"
                        if risk_full_token
                        else "m48_paired_efa_rollout"
                    )
                )
            )
            )
        ),
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "efa_checkpoint": (
            str(loaded_efa.checkpoint_path) if loaded_efa is not None else None
        ),
        "efa_teacher_checkpoint_sha256": (
            loaded_efa.teacher_checkpoint_sha256
            if loaded_efa is not None
            else None
        ),
        "output_tokens": expected_tokens,
        "full_token_profiles": (
            list(risk_full_token_profiles)
            if risk_full_token_profiles is not None
            else []
        ),
        "phase_depth_control_enabled": not width_only,
        "full_token_hold_calls": full_token_hold_calls,
        "full_token_uncertainty_threshold": (
            full_token_uncertainty_threshold
        ),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "fm_steps": args.fm_steps,
        "requested_episodes": args.num_episodes,
        "completed_episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "episode_successes": episode_successes,
        "policy_calls": len(records),
        "vision_events": len(vision_events),
        "compressed_vision_calls": sum(
            event.get("status") == "compressed" for event in vision_events
        ),
        "full_token_fallback_calls": sum(
            event.get("full_token_fallback") is True for event in vision_events
        ),
        "uncertainty_trigger_calls": sum(
            event.get("full_token_trigger") == "uncertainty"
            for event in vision_events
        ),
        "vision_route_counts": dict(
            sorted(
                Counter(
                    str(event.get("vision_route") or "legacy")
                    for event in vision_events
                ).items()
            )
        ),
        "vision_profile_counts": dict(
            sorted(
                Counter(
                    str(event.get("phase_profile_name") or "none")
                    for event in vision_events
                ).items()
            )
        ),
        "aggregation_errors": errors,
        "mean_kept_visual_tokens": mean_or_none(
            [int(event["kept_tokens"]) for event in vision_events]
        ),
        "mean_active_tokens": mean_or_none(
            [record["active_tokens_by_layer"][0] for record in records]
        ),
        "mean_llm_sequence_length": mean_or_none(
            [int(event["llm_sequence_length"]) for event in vision_events]
        ),
        "mean_original_llm_sequence_length": mean_or_none(
            [int(event["original_llm_sequence_length"]) for event in vision_events]
        ),
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "mean_exit_layer": mean_or_none(exit_layers),
        "exit_layer_counts": dict(sorted(Counter(exit_layers).items())),
        "fm_calls_total": sum(int(record["fm_calls"]) for record in records),
        "fm_steps_total": sum(int(record["fm_steps_total"]) for record in records),
        "latency_ms_mean": mean_or_none(latencies),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "inference_latency_ms_total": sum(latencies),
        "phase_plans": len(plans),
        "phase_runtime_errors": phase_runtime.error_count if phase_runtime else 0,
        "telemetry_errors": logger.error_count,
        "telemetry_last_error": logger.last_error,
        "eval_log": str(eval_log_path),
        "telemetry_path": str(telemetry_path),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not run_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
