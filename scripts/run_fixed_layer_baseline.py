#!/usr/bin/env python3
"""Run an auditable fixed L11/L13/L27 LIBERO-10 engineering baseline."""

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

from a1.vla.dynamic_compute.stage1_measurement import (  # noqa: E402
    summarize_stage1_records,
)
from a1.vla.dynamic_compute.fixed_layer_controller import (  # noqa: E402
    FixedLayerFlowMatchingController,
)
from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger  # noqa: E402
from a1.vla.dynamic_compute.v3.release import (  # noqa: E402
    validate_general_release_selection,
)
import robot_experiments.libero.eval_libero_early_exit as frozen_evaluator  # noqa: E402
from robot_experiments.libero.eval_libero_early_exit import (  # noqa: E402
    GenerateConfig,
    initialize_and_load_model,
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
    parser.add_argument("--exit-layer", type=int, choices=(11, 13, 27), required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _config(args: argparse.Namespace) -> GenerateConfig:
    return GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
        task_suite_name="libero_10",
        num_trials_per_task=1,
        action_head_flow_matching_inference_steps=10,
        exit_layer_id=args.exit_layer,
        local_log_dir=str(args.output_dir.resolve() / "eval_logs"),
        save_rollout_video=False,
        save_rollout_video_path=str(args.output_dir.resolve()),
        use_wandb=False,
        reseed_each_episode=True,
        seed=args.seed,
        run_id_note=f"fixed-l{args.exit_layer}-stage1",
        telemetry_enabled=True,
        telemetry_output_path=str(args.output_dir.resolve() / "policy_telemetry.jsonl"),
        vision_aggregation_enabled=False,
        learned_vision_aggregation_checkpoint=None,
        phase_depth_enabled=False,
        rp_pep_enabled=False,
        phase_route_v3_enabled=False,
    )


def main() -> None:
    args = parse_args()
    task_ids, episode_indices = validate_general_release_selection(
        args.task_ids, args.episode_indices
    )
    output = args.output_dir.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output}")
    telemetry_path = output / "policy_telemetry.jsonl"
    measurement_path = output / "stage1_measurement.jsonl"
    summary_path = output / "evaluation_summary.json"
    episode_log_dir = output / "episode_logs"
    for path in (telemetry_path, measurement_path, summary_path):
        if path.exists() or path.with_name(path.name + ".incomplete").exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    episode_log_dir.mkdir(exist_ok=False)

    os.environ[STAGE1_TIMING_ENV] = str(measurement_path)
    frozen_evaluator.get_vla_action = get_stage1_vla_action
    cfg = _config(args)
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    print(f"[fixed-L{args.exit_layer}] loading frozen A1 backbone", flush=True)
    model, device, _ = initialize_and_load_model(cfg)
    controller = FixedLayerFlowMatchingController(model, args.exit_layer).to(device)
    controller.eval()

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
                    calls_before = telemetry.records_written
                    episode_started = time.time()
                    episode_log_path = (
                        episode_log_dir / f"task{task_id}_episode{episode_index}.log"
                    )
                    print(
                        f"[fixed-L{args.exit_layer}] start task={task_id} "
                        f"state={episode_index} seed={episode_seed}",
                        flush=True,
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
                            initial_state=np.array(
                                initial_states[episode_index], copy=True
                            ),
                            log_file=episode_log,
                            task_id=task_id,
                            episode_idx=episode_index,
                            telemetry_logger=telemetry,
                            phase_cache_writer=None,
                            phase_depth_runtime=None,
                            vision_teacher_cache_writer=None,
                            learnable_vision_aggregator=None,
                            phase_depth_control_enabled=False,
                            phase_route_runtime=None,
                        )
                    del replay_images
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    calls = telemetry.records_written - calls_before
                    total_successes += int(bool(success))
                    episode_results.append(
                        {
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "seed": episode_seed,
                            "success": bool(success),
                            "policy_calls": calls,
                            "selected_layer": args.exit_layer,
                            "wall_seconds": time.time() - episode_started,
                            "episode_log": episode_log_path.relative_to(output).as_posix(),
                        }
                    )
                    print(
                        f"[fixed-L{args.exit_layer}] complete success={success} "
                        f"policy_calls={calls}",
                        flush=True,
                    )
            finally:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()
    finally:
        telemetry.close()

    if telemetry.error_count:
        raise RuntimeError(f"telemetry failed: {telemetry.last_error}")
    measurement_records = tuple(
        json.loads(line)
        for line in measurement_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    measurement_summary = summarize_stage1_records(measurement_records)
    telemetry_records = tuple(
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (
        measurement_summary["records"] != telemetry.records_written
        or measurement_summary["records_with_errors"] != 0
        or measurement_summary["records_with_nonfinite_actions"] != 0
        or measurement_summary["records_without_action_audit"] != 0
        or measurement_summary["selected_layers"][str(args.exit_layer)]
        != telemetry.records_written
    ):
        raise RuntimeError("fixed-layer measurement records are incomplete")
    if len(telemetry_records) != telemetry.records_written or any(
        record.get("candidate_exit_layers") != [args.exit_layer]
        or record.get("exit_layer") != args.exit_layer
        or record.get("fm_calls") != 1
        or record.get("action_shape") != [1, 8, 7]
        for record in telemetry_records
    ):
        raise RuntimeError("fixed-layer policy telemetry failed its invariant audit")
    total_episodes = len(episode_results)
    _write_json(
        summary_path,
        {
            "schema_version": "phase-route-vla.stage1.fixed-baseline-result.v1",
            "scope": "engineering_measurement_not_D9_retest",
            "method": f"fixed_l{args.exit_layer}",
            "suite": "libero_10",
            "task_ids": list(task_ids),
            "episode_indices": list(episode_indices),
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": total_successes / total_episodes,
            "wall_seconds": time.time() - started,
            "policy_calls": telemetry.records_written,
            "episodes": episode_results,
            "stage1_measurement": measurement_summary,
            "claim_boundary": {
                "D9_retest": False,
                "new_independent_test": False,
                "measurement_is_control_input": False,
            },
        },
    )
    print(
        f"[fixed-L{args.exit_layer}] run complete: "
        f"success={total_successes}/{total_episodes}, "
        f"calls={telemetry.records_written}",
        flush=True,
    )


if __name__ == "__main__":
    main()
