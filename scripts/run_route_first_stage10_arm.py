#!/usr/bin/env python3
"""Run exactly one frozen Stage 10 arm from one bound fresh state."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


_REQUESTED_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
# Importing the top-level ``a1`` package pulls optional ML dependencies.  Hide
# CUDA until the bound state has passed its CPU deserialization audit.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE, CONTRACT  # noqa: E402


sha256_file = CONTRACT.sha256_file
ACTIVE_ARM_SCHEMA = ACTIVE.ACTIVE_ARM_SCHEMA
METHOD_LAYERS = ACTIVE.METHOD_LAYERS
PREFLIGHT_SCHEMA = ACTIVE.PREFLIGHT_SCHEMA
Stage10ActiveError = ACTIVE.Stage10ActiveError
load_bound_state = ACTIVE.load_bound_state
load_runner_readiness = ACTIVE.load_runner_readiness
normalize_gpu_uuid = ACTIVE.normalize_gpu_uuid
read_jsonl = ACTIVE.read_jsonl
select_arm = ACTIVE.select_arm
summarize_measurement_records = ACTIVE.summarize_measurement_records
summarize_policy_records = ACTIVE.summarize_policy_records


THRESHOLD_SHA256 = (
    "a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=("original_a1", "candidate_first_v3", "route_first_stage8"),
        required=True,
    )
    parser.add_argument("--arm-position", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "model/libero_exit"
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=REPO_ROOT / "artifacts/phase_route_v3/final_router.pt",
    )
    parser.add_argument(
        "--phase-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts/phase_route_v3/phase_estimator.pt",
    )
    parser.add_argument(
        "--calibrated-router",
        type=Path,
        default=REPO_ROOT
        / "runs/route_first_calibration_stage6/router_calibrated.npz",
    )
    parser.add_argument(
        "--stage7-holdout",
        type=Path,
        default=REPO_ROOT
        / "results/route_first/route_first_stage7_holdout.json",
    )
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ActiveError(f"JSON object required: {path}")
    return dict(value)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


class CountingEnvironment:
    def __init__(self, environment: Any):
        self.environment = environment
        self.steps = 0

    def step(self, *args: Any, **kwargs: Any):
        self.steps += 1
        return self.environment.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)


def _thresholds(checkpoint: Path) -> dict[int, float]:
    path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    if sha256_file(path) != THRESHOLD_SHA256:
        raise Stage10ActiveError("frozen A1 threshold SHA-256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ActiveError("frozen thresholds must be an object")
    return {int(layer): float(threshold) for layer, threshold in value.items()}


def _original_controller(cfg: Any, model: Any, device: Any) -> Any:
    import torch

    from a1.vla.value_net import ActionValueNet, ExitController

    layers = tuple(model.get_all_exit_idx(cfg.exit_interval))
    if layers != tuple(range(1, 28, 2)):
        raise Stage10ActiveError("original A1 exit schedule differs")
    value_net = ActionValueNet(
        exit_list=list(layers),
        exit_head=model.action_head,
        model=model,
        interval=cfg.exit_interval,
        threshold_type=cfg.threshold_type,
        anchor=False,
    )
    controller = ExitController(
        value_net,
        exit_id_list=list(layers),
        steps_per_stage=cfg.steps_per_stage,
        leq=True,
        exit_dist=cfg.exit_dist,
        max_layer=model.config.n_layers,
    )
    thresholds = _thresholds(Path(cfg.pretrained_checkpoint))
    if set(thresholds) != set(layers):
        raise Stage10ActiveError("original A1 threshold keys differ")
    controller.thresholds = thresholds
    controller.to(device)
    controller.eval()
    if not isinstance(controller, torch.nn.Module):
        raise Stage10ActiveError("original controller is not a torch module")
    return controller


def _sparse_controller(cfg: Any, model: Any, device: Any) -> Any:
    from a1.vla.dynamic_compute.productive_exit import a1_fm10_rp_pep_plan
    from a1.vla.value_net import ActionValueNet, ExitController

    original = tuple(model.get_all_exit_idx(cfg.exit_interval))
    plan = a1_fm10_rp_pep_plan(original)
    value_net = ActionValueNet(
        exit_list=list(plan.eligible_exit_layers),
        exit_head=model.action_head,
        model=model,
        interval=cfg.exit_interval,
        threshold_type=cfg.threshold_type,
        anchor=False,
        productive_exit_plan=plan,
    )
    controller = ExitController(
        value_net,
        exit_id_list=list(plan.eligible_exit_layers),
        steps_per_stage=cfg.steps_per_stage,
        leq=True,
        exit_dist=cfg.exit_dist,
        max_layer=model.config.n_layers,
    )
    selected = plan.select_eligible_thresholds(
        _thresholds(Path(cfg.pretrained_checkpoint)), lower_is_easier=True
    )
    controller.thresholds = dict(zip(plan.eligible_exit_layers, selected))
    controller.to(device)
    controller.eval()
    return controller


def _runtime_summary(records: tuple[Mapping[str, Any], ...], spec: Any) -> dict[str, Any]:
    selected = Counter()
    if len(records) == 0:
        raise Stage10ActiveError("dynamic arm has no runtime records")
    for ordinal, record in enumerate(records):
        context = record.get("context", {})
        layer = record.get("selected_layer")
        if (
            context.get("episode_id") != spec.cluster_key
            or context.get("task_id") != spec.task_id
            or context.get("call_ordinal") != ordinal
            or record.get("prepared") is not True
            or record.get("committed") is not True
            or layer not in METHOD_LAYERS[spec.method]
            or record.get("errors")
        ):
            raise Stage10ActiveError("dynamic runtime record differs")
        selected[int(layer)] += 1
    return {
        "records": len(records),
        "selected_layer_counts": {
            f"L{layer}": selected[layer] for layer in METHOD_LAYERS[spec.method]
        },
        "errors": 0,
    }


def _artifact_inventory(output: Path, names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name in names:
        path = output / name
        if not path.is_file():
            raise Stage10ActiveError(f"required arm artifact is missing: {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve(strict=True)
    preflight_path = args.preflight.resolve(strict=True)
    if preflight_path.parent != output:
        raise Stage10ActiveError("preflight must be a direct arm artifact")
    if _git("status", "--porcelain=v1"):
        raise PermissionError("Stage 10 active arm requires a clean worktree")
    source_commit = _git("rev-parse", "HEAD")
    readiness = load_runner_readiness(REPO_ROOT)
    readiness_path = REPO_ROOT / (
        "results/route_first/route_first_stage10_runner_readiness.json"
    )
    spec = select_arm(
        REPO_ROOT,
        task_id=args.task_id,
        replicate_id=args.replicate_id,
        method=args.method,
        arm_position=args.arm_position,
    )
    preflight = _read_object(preflight_path)
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "PASS"
        or preflight.get("source_git_commit") != source_commit
        or preflight.get("task_id") != spec.task_id
        or preflight.get("replicate_id") != spec.replicate_id
        or preflight.get("method") != spec.method
        or preflight.get("arm_position") != spec.arm_position
        or preflight.get("cluster_key") != spec.cluster_key
        or preflight.get("policy_seed") != spec.policy_seed
        or normalize_gpu_uuid(preflight.get("expected_gpu_uuid"))
        != normalize_gpu_uuid(args.expected_gpu_uuid)
        or preflight.get("physical_gpu_index") != args.physical_gpu_index
        or not all(preflight.get("checks", {}).values())
    ):
        raise PermissionError("preflight does not authorize this exact arm")

    # This is the first payload deserialization in the arm process.  It occurs
    # before LIBERO construction, model loading, or a CUDA query.
    triplet, initial_state, state_audit = load_bound_state(
        REPO_ROOT, task_id=spec.task_id, replicate_id=spec.replicate_id
    )
    if triplet.arm_order != spec.arm_order:
        raise Stage10ActiveError("loaded state arm order differs")

    import torch

    if torch.cuda.is_initialized():
        raise PermissionError("CUDA initialized before fresh-state validation")
    if _REQUESTED_VISIBLE_DEVICES is None:
        raise PermissionError("arm process has no requested CUDA binding")
    os.environ["CUDA_VISIBLE_DEVICES"] = _REQUESTED_VISIBLE_DEVICES

    import numpy as np
    from libero.libero import benchmark

    from a1.vla.dynamic_compute.route_first_controller import RouteFirstExitController
    from a1.vla.dynamic_compute.route_first_runtime import load_route_first_active_runtime
    from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
    from a1.vla.dynamic_compute.v3.active_runtime import load_frozen_phase_route_runtime
    import robot_experiments.libero.eval_libero_early_exit as frozen_evaluator
    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
        run_episode,
        validate_config,
    )
    from robot_experiments.libero.libero_utils import get_libero_env
    from robot_experiments.libero.stage1_vla_utils import (
        STAGE1_TIMING_ENV,
        get_vla_action as get_stage1_vla_action,
    )
    from robot_experiments.robot_utils import set_seed_everywhere

    if _REQUESTED_VISIBLE_DEVICES not in (
        str(args.physical_gpu_index),
        args.expected_gpu_uuid,
    ):
        raise PermissionError("CUDA_VISIBLE_DEVICES differs from GPU binding")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage 10 arm requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if normalize_gpu_uuid(properties.uuid) != normalize_gpu_uuid(
        args.expected_gpu_uuid
    ):
        raise RuntimeError("visible CUDA UUID differs from triplet binding")

    checkpoint = args.checkpoint.resolve(strict=True)
    threshold_path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    threshold_before = threshold_path.read_bytes()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_10",
        num_trials_per_task=1,
        action_head_flow_matching_inference_steps=10,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        local_log_dir=str(output / "eval_logs"),
        save_rollout_video=False,
        save_rollout_video_path=str(output),
        use_wandb=False,
        reseed_each_episode=False,
        seed=spec.policy_seed,
        run_id_note=f"route-first-stage10-{spec.method}",
        vision_aggregation_enabled=False,
        learned_vision_aggregation_checkpoint=None,
        phase_depth_enabled=False,
        rp_pep_enabled=False,
        phase_route_v3_enabled=False,
    )
    validate_config(cfg)
    set_seed_everywhere(spec.policy_seed)
    print(
        f"[Stage10] loading A1 arm={spec.method} task={spec.task_id} "
        f"replicate={spec.replicate_id}",
        flush=True,
    )
    model, device, _ = initialize_and_load_model(cfg)
    runtime = None
    if spec.method == "original_a1":
        controller = _original_controller(cfg, model, device)
    elif spec.method == "candidate_first_v3":
        runtime = load_frozen_phase_route_runtime(
            args.router.resolve(strict=True),
            args.phase_checkpoint.resolve(strict=True),
        )
        controller = _sparse_controller(cfg, model, device)
        controller.set_phase_route_runtime_adapter(runtime.adapter)
    else:
        runtime = load_route_first_active_runtime(
            args.calibrated_router.resolve(strict=True),
            args.stage7_holdout.resolve(strict=True),
            args.router.resolve(strict=True),
            args.phase_checkpoint.resolve(strict=True),
        )
        base = _sparse_controller(cfg, model, device)
        controller = RouteFirstExitController.from_frozen_sparse_controller(base)
        controller.install_route_first_adapter(runtime.adapter)
        controller.to(device)
        controller.eval()
    if threshold_path.read_bytes() != threshold_before:
        raise Stage10ActiveError("controller initialization changed thresholds")

    telemetry_path = output / "policy_telemetry.jsonl"
    measurement_path = output / "stage1_measurement.jsonl"
    runtime_path = output / "phase_route_runtime.jsonl"
    episode_log_path = output / "episode.log"
    for path in (telemetry_path, measurement_path, runtime_path, episode_log_path):
        if path.exists() or path.with_name(path.name + ".incomplete").exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    os.environ[STAGE1_TIMING_ENV] = str(measurement_path)
    frozen_evaluator.get_vla_action = get_stage1_vla_action
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    if suite.n_tasks != 10:
        raise Stage10ActiveError("LIBERO-10 task count differs")
    task = suite.get_task(spec.task_id)
    environment, task_description = get_libero_env(
        task, cfg.model_family, resolution=cfg.env_img_res
    )
    counted = CountingEnvironment(environment)
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=1)
    success = False
    started = time.perf_counter()
    try:
        set_seed_everywhere(spec.policy_seed)
        with episode_log_path.open("x", encoding="utf-8") as episode_log:
            success, replay_images, _ = run_episode(
                cfg=cfg,
                env=counted,
                task_description=task_description,
                model=model,
                exit_controller=controller,
                device=device,
                resize_size=model.config.vision_backbone.image_default_input_size,
                initial_state=np.array(initial_state, copy=True),
                log_file=episode_log,
                task_id=spec.task_id,
                episode_idx=spec.replicate_id,
                telemetry_logger=telemetry,
                phase_cache_writer=None,
                phase_depth_runtime=None,
                vision_teacher_cache_writer=None,
                learnable_vision_aggregator=None,
                phase_depth_control_enabled=False,
                episode_id_override=spec.cluster_key,
                phase_route_runtime=runtime,
            )
            del replay_images
        torch.cuda.synchronize()
    finally:
        telemetry.close()
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    rollout_wall_seconds = time.perf_counter() - started
    if telemetry.error_count:
        raise Stage10ActiveError(f"telemetry write failed: {telemetry.last_error}")
    telemetry_records = read_jsonl(telemetry_path)
    measurement_records = read_jsonl(measurement_path)
    policy = summarize_policy_records(telemetry_records, spec=spec)
    measurement = summarize_measurement_records(
        measurement_records,
        spec=spec,
        expected_policy_calls=policy["policy_calls"],
    )
    runtime_summary = None
    if runtime is not None:
        runtime_records = tuple(runtime.records)
        runtime_summary = _runtime_summary(runtime_records, spec)
        if len(runtime_records) != policy["policy_calls"]:
            raise Stage10ActiveError("runtime count differs from policy calls")
        _write_jsonl(runtime_path, runtime_records)
    required_artifacts = (
        ("preflight.json", "policy_telemetry.jsonl", "stage1_measurement.jsonl", "episode.log")
        if runtime is None
        else (
            "preflight.json",
            "policy_telemetry.jsonl",
            "stage1_measurement.jsonl",
            "phase_route_runtime.jsonl",
            "episode.log",
        )
    )
    result = {
        "schema_version": ACTIVE_ARM_SCHEMA,
        "status": "COMPLETE_ROUTE_FIRST_STAGE10_ACTIVE_ARM",
        "timestamp_utc": utc_now(),
        "method": spec.method,
        "task_id": spec.task_id,
        "replicate_id": spec.replicate_id,
        "cluster_key": spec.cluster_key,
        "state_seed": spec.state_seed,
        "policy_seed": spec.policy_seed,
        "arm_order": list(spec.arm_order),
        "arm_position": spec.arm_position,
        "success": bool(success),
        "environment_steps": counted.steps,
        "control_steps": max(0, counted.steps - cfg.num_steps_wait),
        "policy_accounting": policy,
        "policy_latency_ms": measurement,
        "runtime_integrity": runtime_summary,
        "rollout_wall_seconds": rollout_wall_seconds,
        "source_git_commit": source_commit,
        "source_worktree_dirty": False,
        "state_evidence": state_audit,
        "gpu": {
            "physical_index": args.physical_gpu_index,
            "uuid": "GPU-" + str(properties.uuid).removeprefix("GPU-"),
            "visible_count": torch.cuda.device_count(),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
        },
        "runner_readiness": {
            "path": str(readiness_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(readiness_path),
            "source_runner_commit": readiness["source_git_commit"],
        },
        "preflight": {
            "path": "preflight.json",
            "sha256": sha256_file(preflight_path),
        },
        "artifact_inventory_before_result": _artifact_inventory(
            output, required_artifacts
        ),
        "retry_policy": {
            "same_task_replicate_state_policy_seed_order_gpu_uuid_commit": True,
            "valid_task_failure_retained": True,
            "outcome_based_retry": False,
            "replacement_state_or_seed": False,
        },
        "claim_boundary": {
            "raw_arm_evidence_only": True,
            "cross_triplet_aggregate_computed": False,
            "stage10_gate_evaluated": False,
            "deployment_authorized": False,
        },
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    result_sha = sha256_file(result_path)
    (output / "result.sha256").write_text(
        f"{result_sha}  result.json\n", encoding="utf-8"
    )
    print(
        f"[Stage10] arm complete success={bool(success)} "
        f"calls={policy['policy_calls']} layers={policy['selected_layer_counts']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "ABORT_ROUTE_FIRST_STAGE10_ARM",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "valid_task_failure": False,
                    "infrastructure_retry_requires_exact_same_tuple": True,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
