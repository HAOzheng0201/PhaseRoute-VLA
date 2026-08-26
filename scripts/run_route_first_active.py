#!/usr/bin/env python3
"""Execute one preregistered Stage-9 route-first LIBERO arm.

The historical D9 evaluator remains byte-for-byte frozen.  This entrypoint
loads its model and episode helpers, clones the frozen sparse controller into
the isolated :class:`RouteFirstExitController`, and executes exactly one
flow-matching head at the context-selected L13 or L27 depth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark  # noqa: E402

from a1.vla.dynamic_compute.route_first_active_protocol import (  # noqa: E402
    ROUTE_FIRST_ACTIVE_METHOD,
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
    load_route_first_active_protocol,
    validate_route_first_active_selection,
)
from a1.vla.dynamic_compute.route_first_controller import (  # noqa: E402
    RouteFirstExitController,
)
from a1.vla.dynamic_compute.route_first_runtime import (  # noqa: E402
    load_route_first_active_runtime,
)
from a1.vla.dynamic_compute.stage1_measurement import (  # noqa: E402
    summarize_stage1_records,
)
from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger  # noqa: E402
from a1.vla.dynamic_compute.v3.release import summarize_runtime_records  # noqa: E402
import robot_experiments.libero.eval_libero_early_exit as frozen_evaluator  # noqa: E402
from robot_experiments.libero.eval_libero_early_exit import (  # noqa: E402
    GenerateConfig,
    initialize_and_load_model,
    initialize_exit_controller,
    run_episode,
    validate_config,
)
from robot_experiments.libero.libero_utils import get_libero_env  # noqa: E402
from robot_experiments.libero.stage1_vla_utils import (  # noqa: E402
    STAGE1_TIMING_ENV,
    get_vla_action as get_stage1_vla_action,
)
from robot_experiments.robot_utils import set_seed_everywhere  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibrated-router", type=Path, required=True)
    parser.add_argument("--stage7-holdout", type=Path, required=True)
    parser.add_argument("--context-router", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--experiment-stage",
        choices=("engineering_smoke", "paired_pilot"),
        required=True,
    )
    parser.add_argument("--arm-position", type=int, choices=(1, 2), required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: tuple[Mapping[str, Any], ...]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSONL objects: {path}")
        result.append(value)
    return tuple(result)


def _normalize_uuid(value: Any) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def _visible_gpu(args: argparse.Namespace) -> Mapping[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("active arm requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    visible_uuid = str(properties.uuid)
    if _normalize_uuid(visible_uuid) != _normalize_uuid(args.expected_gpu_uuid):
        raise RuntimeError("visible CUDA GPU UUID differs from the preflight binding")
    return {
        "physical_index": args.physical_gpu_index,
        "expected_uuid": args.expected_gpu_uuid,
        "visible_uuid": visible_uuid,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def _config(args: argparse.Namespace) -> GenerateConfig:
    return GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
        task_suite_name="libero_10",
        num_trials_per_task=1,
        action_head_flow_matching_inference_steps=10,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        local_log_dir=str(args.output_dir.resolve() / "eval_logs"),
        save_rollout_video=False,
        save_rollout_video_path=str(args.output_dir.resolve()),
        use_wandb=False,
        reseed_each_episode=True,
        seed=args.seed,
        run_id_note="route-first-stage9-active",
        vision_aggregation_enabled=False,
        learned_vision_aggregation_checkpoint=None,
        phase_depth_enabled=False,
        rp_pep_enabled=True,
        phase_route_v3_enabled=True,
        phase_route_router_checkpoint=str(args.context_router.resolve()),
        phase_route_phase_checkpoint=str(args.phase_checkpoint.resolve()),
    )


def _latency_summary(records: tuple[Mapping[str, Any], ...]) -> Mapping[str, Any]:
    values = sorted(
        float(record["policy_wall_latency_ms"])
        for record in records
        if isinstance(record.get("policy_wall_latency_ms"), (int, float))
        and not isinstance(record.get("policy_wall_latency_ms"), bool)
        and math.isfinite(float(record["policy_wall_latency_ms"]))
        and float(record["policy_wall_latency_ms"]) >= 0.0
    )
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}

    def nearest_rank(percentile: float) -> float:
        return values[max(0, math.ceil(percentile * len(values)) - 1)]

    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "max": values[-1],
    }


def summarize_route_first_integrity(
    records: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    """Summarize the one-FM invariant without trusting aggregate counters."""

    valid_one_fm = 0
    fm_invocations = 0
    decoder_blocks = 0
    calls_with_route_errors = 0
    for record in records:
        events = record.get("events")
        if not isinstance(events, list):
            events = []
        evaluated = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event") == "exit_candidate"
            and event.get("evaluated") is True
        ]
        selected_events = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event") == "route_first_selected_action"
        ]
        decisions = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event") == "phase_route_decision"
        ]
        error_events = [
            event
            for event in events
            if isinstance(event, Mapping)
            and event.get("event")
            in ("route_first_action_error", "route_first_action_rejected")
        ]
        layer = record.get("selected_layer")
        if layer in (13, 27):
            decoder_blocks += int(layer) + 1
        fm_invocations += sum(
            int(event.get("fm_calls", 0))
            for event in evaluated
            if type(event.get("fm_calls")) is int
        )
        valid = bool(
            record.get("prepared") is True
            and record.get("committed") is True
            and not record.get("errors")
            and layer in (13, 27)
            and len(evaluated) == 1
            and evaluated[0].get("layer_idx") == layer
            and evaluated[0].get("should_exit") is True
            and evaluated[0].get("fm_calls") == 1
            and len(selected_events) == 1
            and selected_events[0].get("layer_idx") == layer
            and selected_events[0].get("fm_calls") == 1
            and selected_events[0].get("fail_reason") is None
            and len(decisions) == 1
            and decisions[0].get("selected_layer") == layer
            and decisions[0].get("fm_calls") == 1
            and not error_events
        )
        valid_one_fm += int(valid)
        calls_with_route_errors += int(bool(error_events or record.get("errors")))
    return {
        "records": len(records),
        "valid_calls_with_exactly_one_fm": valid_one_fm,
        "valid_calls_with_fm_calls_equal_one_fraction": (
            valid_one_fm / len(records) if records else 0.0
        ),
        "fm_invocations": fm_invocations,
        "decoder_blocks": decoder_blocks,
        "calls_with_route_errors": calls_with_route_errors,
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output}")
    protocol = load_route_first_active_protocol(args.protocol, REPO_ROOT)
    selection = validate_route_first_active_selection(
        protocol,
        experiment_stage=args.experiment_stage,
        task_spec=args.task_ids,
        episode_spec=args.episode_indices,
        arm_position=args.arm_position,
        seed=args.seed,
    )
    preflight = _load_json_object(args.preflight.resolve(strict=True))
    if (
        preflight.get("status") != "PASS"
        or preflight.get("scope") != "route_first_stage9_active_preflight"
        or preflight.get("protocol_sha256") != ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256
        or _normalize_uuid(preflight.get("expected_gpu_uuid"))
        != _normalize_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError("preflight does not authorize this active arm")
    gpu = _visible_gpu(args)

    telemetry_path = output / "policy_telemetry.jsonl"
    runtime_path = output / "phase_route_runtime.jsonl"
    evaluation_path = output / "evaluation_summary.json"
    measurement_path = args.measurement_output.resolve()
    episode_log_dir = output / "episode_logs"
    for path in (telemetry_path, runtime_path, evaluation_path, measurement_path):
        if path.parent != output:
            raise ValueError("all active artifacts must be direct children of output-dir")
        if path.exists() or path.with_name(path.name + ".incomplete").exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    episode_log_dir.mkdir(exist_ok=False)
    os.environ[STAGE1_TIMING_ENV] = str(measurement_path)
    frozen_evaluator.get_vla_action = get_stage1_vla_action

    cfg = _config(args)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    print("[RouteFirst-Stage9] loading frozen A1 backbone", flush=True)
    model, device, _ = initialize_and_load_model(cfg)
    runtime = load_route_first_active_runtime(
        args.calibrated_router,
        args.stage7_holdout,
        args.context_router,
        args.phase_checkpoint,
    )
    base_controller = initialize_exit_controller(cfg, model, None, device)
    controller = RouteFirstExitController.from_frozen_sparse_controller(base_controller)
    controller.install_route_first_adapter(runtime.adapter)
    controller.eval()
    print(
        "[RouteFirst-Stage9] route-first runtime loaded; L11 disabled, "
        "one selected L13/L27 FM head per call",
        flush=True,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=1)
    episode_results = []
    total_successes = 0
    started = time.time()
    try:
        for task_id in selection.task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if any(index >= len(initial_states) for index in selection.episode_indices):
                raise ValueError(
                    f"task {task_id} exposes only {len(initial_states)} initial states"
                )
            environment, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                for episode_index in selection.episode_indices:
                    episode_seed = cfg.seed + task_id * 10_000 + episode_index
                    set_seed_everywhere(episode_seed)
                    calls_before = runtime.policy_calls
                    wall_started = time.time()
                    print(
                        "[RouteFirst-Stage9] start "
                        f"task={task_id} state={episode_index} seed={episode_seed}",
                        flush=True,
                    )
                    episode_log_path = (
                        episode_log_dir / f"task{task_id}_episode{episode_index}.log"
                    )
                    with episode_log_path.open("x", encoding="utf-8") as episode_log:
                        success, replay_images, _ = run_episode(
                            cfg=cfg,
                            env=environment,
                            task_description=task_description,
                            model=model,
                            exit_controller=controller,
                            device=device,
                            resize_size=model.config.vision_backbone.image_default_input_size,
                            initial_state=np.array(initial_states[episode_index], copy=True),
                            log_file=episode_log,
                            task_id=task_id,
                            episode_idx=episode_index,
                            telemetry_logger=telemetry,
                            phase_cache_writer=None,
                            phase_depth_runtime=None,
                            vision_teacher_cache_writer=None,
                            learnable_vision_aggregator=None,
                            phase_depth_control_enabled=False,
                            phase_route_runtime=runtime,
                        )
                    del replay_images
                    torch.cuda.synchronize()
                    calls = runtime.policy_calls - calls_before
                    episode_runtime = runtime.records[calls_before:]
                    route_counts = {
                        str(layer): sum(
                            record.get("selected_layer") == layer
                            for record in episode_runtime
                        )
                        for layer in (11, 13, 27)
                    }
                    total_successes += int(bool(success))
                    episode_results.append(
                        {
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "episode_id": (
                                f"libero_10:task{task_id}:episode{episode_index}"
                            ),
                            "seed": episode_seed,
                            "success": bool(success),
                            "policy_calls": calls,
                            "selected_layers": route_counts,
                            "episode_log": episode_log_path.relative_to(output).as_posix(),
                            "wall_seconds": time.time() - wall_started,
                        }
                    )
                    print(
                        "[RouteFirst-Stage9] complete "
                        f"success={success} calls={calls} routes={route_counts}",
                        flush=True,
                    )
            finally:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()
    finally:
        telemetry.close()

    if telemetry.error_count:
        raise RuntimeError(
            f"telemetry writer failed {telemetry.error_count} times: {telemetry.last_error}"
        )
    if not measurement_path.is_file():
        raise RuntimeError("Stage-1 measurement output was not created")
    records = runtime.records
    measurement_records = _load_jsonl(measurement_path)
    measurement_summary = summarize_stage1_records(measurement_records)
    runtime_summary = summarize_runtime_records(records)
    route_integrity = summarize_route_first_integrity(records)
    runtime_summary.update(
        {
            "policy_calls": runtime.policy_calls,
            "prepared_calls": runtime.prepared_calls,
            "committed_calls": runtime.committed_calls,
            "error_count": runtime.error_count,
            "last_error": runtime.last_error,
            "route_first_integrity": route_integrity,
            "artifacts": {
                "v3_context": asdict(runtime.artifacts),
                "route_first": asdict(runtime.route_first_artifacts),
            },
        }
    )
    measurement_complete = bool(
        measurement_summary["records"] == runtime.policy_calls
        and measurement_summary["records_with_errors"] == 0
        and measurement_summary["records_with_nonfinite_actions"] == 0
        and measurement_summary["records_without_action_audit"] == 0
        and all(
            record.get("mode") == ROUTE_FIRST_ACTIVE_METHOD
            for record in measurement_records
        )
    )
    runtime_complete = bool(
        runtime.error_count == 0
        and runtime.policy_calls == runtime.prepared_calls == runtime.committed_calls
        and route_integrity["valid_calls_with_exactly_one_fm"] == runtime.policy_calls
        and runtime_summary["selected_layers"]["11"] == 0
    )
    _write_jsonl(runtime_path, records)
    total_episodes = len(episode_results)
    summary = {
        "schema_version": "phase-route-vla.route-first-active-evaluation.v1",
        "method": ROUTE_FIRST_ACTIVE_METHOD,
        "experiment_stage": selection.experiment_stage,
        "arm_position": selection.arm_position,
        "protocol_sha256": ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
        "scope": "stage9_preregistered_engineering_active_control",
        "suite": "libero_10",
        "task_ids": list(selection.task_ids),
        "episode_indices": list(selection.episode_indices),
        "seed_base": cfg.seed,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "wall_seconds": time.time() - started,
        "gpu": gpu,
        "episodes": episode_results,
        "telemetry_errors": telemetry.error_count,
        "runtime": runtime_summary,
        "stage1_measurement": measurement_summary,
        "active_latency_ms": _latency_summary(measurement_records),
        "gates": {
            "runtime_integrity": runtime_complete,
            "measurement_integrity": measurement_complete,
        },
        "claim_boundary": {
            "engineering_only": True,
            "final_closed_loop_improvement": False,
            "final_wall_clock_speedup": False,
            "deployment_authorized": False,
        },
    }
    _write_json(evaluation_path, summary)
    if not runtime_complete or not measurement_complete:
        raise RuntimeError("route-first active arm failed its integrity gate")
    print(
        "[RouteFirst-Stage9] run complete: "
        f"success={total_successes}/{total_episodes}, calls={runtime.policy_calls}, "
        f"FM={route_integrity['fm_invocations']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
