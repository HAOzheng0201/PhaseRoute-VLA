#!/usr/bin/env python3
"""General PhaseRoute-V3 LIBERO-10 simulator runner.

This entrypoint deliberately lives outside the SHA-protected D9 evaluator.  It
reuses the frozen model/controller/episode functions without changing the
historical code bytes or the consumed D9 schedule.
"""

from __future__ import annotations

import argparse
import json
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

from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger  # noqa: E402
from a1.vla.dynamic_compute.v3.active_runtime import (  # noqa: E402
    load_frozen_phase_route_runtime,
)
from a1.vla.dynamic_compute.v3.release import (  # noqa: E402
    summarize_runtime_records,
    validate_general_release_selection,
)
from a1.vla.dynamic_compute.stage1_measurement import (  # noqa: E402
    summarize_stage1_records,
)
import robot_experiments.libero.eval_libero_early_exit as frozen_evaluator  # noqa: E402
from robot_experiments.libero.eval_libero_early_exit import (  # noqa: E402
    GenerateConfig,
    initialize_and_load_model,
    initialize_exit_controller,
    run_episode,
    validate_config,
)
from robot_experiments.libero.stage1_vla_utils import (  # noqa: E402
    STAGE1_TIMING_ENV,
    get_vla_action as get_stage1_vla_action,
)
from robot_experiments.libero.libero_utils import get_libero_env  # noqa: E402
from robot_experiments.robot_utils import set_seed_everywhere  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-output", type=Path)
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


def _validated_indices(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return validate_general_release_selection(args.task_ids, args.episode_indices)


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
        run_id_note="phase-route-v3-general-release",
        vision_aggregation_enabled=False,
        learned_vision_aggregation_checkpoint=None,
        phase_depth_enabled=False,
        rp_pep_enabled=True,
        phase_route_v3_enabled=True,
        phase_route_router_checkpoint=str(args.router.resolve()),
        phase_route_phase_checkpoint=str(args.phase_checkpoint.resolve()),
    )


def main() -> None:
    args = parse_args()
    task_ids, episode_indices = _validated_indices(args)
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output}")
    telemetry_path = output / "policy_telemetry.jsonl"
    runtime_path = output / "phase_route_runtime.jsonl"
    evaluation_path = output / "evaluation_summary.json"
    measurement_path = (
        args.measurement_output.resolve()
        if args.measurement_output is not None
        else None
    )
    episode_log_dir = output / "episode_logs"
    for path in (telemetry_path, runtime_path, evaluation_path):
        if path.exists() or path.with_name(path.name + ".incomplete").exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if measurement_path is not None:
        if measurement_path.parent != output:
            raise ValueError("measurement output must be a direct child of output-dir")
        if measurement_path.exists():
            raise FileExistsError(f"refusing to overwrite {measurement_path}")
        os.environ[STAGE1_TIMING_ENV] = str(measurement_path)
        frozen_evaluator.get_vla_action = get_stage1_vla_action
    episode_log_dir.mkdir(exist_ok=False)

    cfg = _config(args)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    print("[PhaseRoute-V3] loading frozen A1 backbone", flush=True)
    model, device, _ = initialize_and_load_model(cfg)
    runtime = load_frozen_phase_route_runtime(args.router, args.phase_checkpoint)
    controller = initialize_exit_controller(cfg, model, None, device)
    controller.set_phase_route_runtime_adapter(runtime.adapter)
    controller.eval()
    print(
        "[PhaseRoute-V3] loaded five-head router and phase estimator: "
        f"{runtime.artifacts}",
        flush=True,
    )

    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=1)
    episode_results = []
    total_successes = 0
    started = time.time()
    try:
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if any(index >= len(initial_states) for index in episode_indices):
                raise ValueError(
                    f"task {task_id} exposes only {len(initial_states)} initial states"
                )
            environment, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                for episode_index in episode_indices:
                    episode_seed = cfg.seed + task_id * 10_000 + episode_index
                    set_seed_everywhere(episode_seed)
                    policy_calls_before = runtime.policy_calls
                    wall_started = time.time()
                    print(
                        "[PhaseRoute-V3] start "
                        f"task={task_id} state={episode_index} seed={episode_seed}",
                        flush=True,
                    )
                    episode_log_path = (
                        episode_log_dir
                        / f"task{task_id}_episode{episode_index}.log"
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
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    calls = runtime.policy_calls - policy_calls_before
                    episode_runtime = runtime.records[policy_calls_before:]
                    route_counts = {
                        str(layer): sum(
                            record.get("selected_layer") == layer
                            for record in episode_runtime
                        )
                        for layer in (11, 13, 27)
                    }
                    total_successes += int(bool(success))
                    record = {
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "episode_id": f"libero_10:task{task_id}:episode{episode_index}",
                        "seed": episode_seed,
                        "success": bool(success),
                        "policy_calls": calls,
                        "selected_layers": route_counts,
                        "episode_log": episode_log_path.relative_to(output).as_posix(),
                        "wall_seconds": time.time() - wall_started,
                    }
                    episode_results.append(record)
                    print(
                        "[PhaseRoute-V3] complete "
                        f"success={success} policy_calls={calls} routes={route_counts}",
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
    records = runtime.records
    runtime_summary = summarize_runtime_records(records)
    runtime_summary.update(
        {
            "policy_calls": runtime.policy_calls,
            "prepared_calls": runtime.prepared_calls,
            "committed_calls": runtime.committed_calls,
            "error_count": runtime.error_count,
            "last_error": runtime.last_error,
            "artifacts": vars(runtime.artifacts) if runtime.artifacts is not None else None,
        }
    )
    measurement_summary = None
    if measurement_path is not None:
        if not measurement_path.is_file():
            raise RuntimeError("Stage-1 measurement output was not created")
        measurement_records = tuple(
            json.loads(line)
            for line in measurement_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        measurement_summary = summarize_stage1_records(measurement_records)
        if (
            measurement_summary["records"] != runtime.policy_calls
            or measurement_summary["records_with_errors"] != 0
            or measurement_summary["records_with_nonfinite_actions"] != 0
            or measurement_summary["records_without_action_audit"] != 0
        ):
            raise RuntimeError("Stage-1 measurement records are incomplete")
    _write_jsonl(runtime_path, records)
    total_episodes = len(episode_results)
    summary = {
        "schema_version": "phase-route-vla.libero-evaluation-summary.v1",
        "method": "phase_route_v3",
        "scope": "general_simulator_run_not_D9_retest",
        "suite": "libero_10",
        "task_ids": list(task_ids),
        "episode_indices": list(episode_indices),
        "seed_base": cfg.seed,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "wall_seconds": time.time() - started,
        "episodes": episode_results,
        "telemetry_errors": telemetry.error_count,
        "runtime": runtime_summary,
        "stage1_measurement": measurement_summary,
        "claim_boundary": {
            "D9_retest": False,
            "deployment_authorized": False,
            "new_independent_test": False,
        },
    }
    _write_json(evaluation_path, summary)
    if (
        runtime.error_count
        or runtime.policy_calls != runtime.prepared_calls
        or runtime.policy_calls != runtime.committed_calls
    ):
        raise RuntimeError("PhaseRoute runtime completed with incomplete or failed calls")
    print(
        "[PhaseRoute-V3] run complete: "
        f"success={total_successes}/{total_episodes}, calls={runtime.policy_calls}",
        flush=True,
    )


if __name__ == "__main__":
    main()
