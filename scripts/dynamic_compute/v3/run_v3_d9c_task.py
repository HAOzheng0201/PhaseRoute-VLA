#!/usr/bin/env python3
"""Run one frozen D9C LIBERO-10 task as ten immutable paired rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark  # noqa: E402

from a1.vla.dynamic_compute.productive_exit import (  # noqa: E402
    a1_fm10_rp_pep_plan,
)
from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger  # noqa: E402
from a1.vla.dynamic_compute.v3.active_runtime import (  # noqa: E402
    load_frozen_phase_route_runtime,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_CHECKPOINT_SHA256,
    D2_EXIT_THRESHOLDS_SHA256,
    stream_sha256,
    validate_runtime_model_directory,
)
from a1.vla.dynamic_compute.v3.paired_active_collection import (  # noqa: E402
    D9C_ARM_SCHEMA_VERSION,
    D9C_PAIR_SCHEMA_VERSION,
    D9C_TASK_SCHEMA_VERSION,
    D9C_TASK_STATUS,
    D9CCollectionError,
    ORIGINAL_A1_ARM,
    PHASE_ROUTE_ARM,
    PHASE_ROUTE_TEACHER_KIND,
    build_file_inventory,
    read_json_object,
    read_jsonl,
    sha256_array,
    sha256_file,
    summarize_policy_telemetry,
    task_schedule,
    validate_d9b_readiness,
    validate_gpu_contract,
    validate_pair_record,
    validate_phase_route_cache,
    validate_phase_route_runtime_records,
    validate_runner_readiness,
    validate_task_output,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (  # noqa: E402
    SafeVisionTeacherCacheWriter,
)
from a1.vla.value_net import ActionValueNet, ExitController  # noqa: E402
from robot_experiments.libero.eval_libero_early_exit import (  # noqa: E402
    GenerateConfig,
    initialize_and_load_model,
    run_episode,
)
from robot_experiments.libero.libero_utils import get_libero_env  # noqa: E402
from robot_experiments.robot_utils import set_seed_everywhere  # noqa: E402


FM_STEPS = 10
SUITE = "libero_10"
ROUTER_RELATIVE_PATH = Path("reports/v3_d8_final_router/final_router.pt")
EXPECTED_ROUTER_SHA256 = (
    "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830"
)
EXPECTED_PHASE_SHA256 = (
    "b601f8221d47136818d7a008eaf7cee06e1201bf514f371ae33f42cfb39515a1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "model" / "v3_d2" / "libero_exit",
    )
    parser.add_argument("--model-attestation", type=Path, required=True)
    parser.add_argument(
        "--router", type=Path, default=REPO_ROOT / ROUTER_RELATIVE_PATH
    )
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: tuple[Mapping[str, Any], ...]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _command_text() -> str:
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_EGL_DEVICE_ID",
        "DATA_DIR",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLA_CONFIG_YAML",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
    )
    environment = [
        f"{name}={os.environ[name]}"
        for name in environment_names
        if name in os.environ
    ]
    return (
        "cd "
        + shlex.quote(str(REPO_ROOT))
        + "\n\n"
        + shlex.join(["env", *environment, sys.executable, *sys.argv])
        + "\n"
    )


def _tracked_readiness_is_current() -> bool:
    relative = "results/v3/v3_d9c_runner_readiness.json"
    try:
        tracked = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT
        )
    except subprocess.CalledProcessError:
        return False
    return tracked == (REPO_ROOT / relative).read_bytes()


class CountingEnvironment:
    """Transparent wrapper that counts calls to the frozen evaluator's step."""

    def __init__(self, environment: Any):
        self.environment = environment
        self.steps = 0

    def reset(self, *args: Any, **kwargs: Any):
        return self.environment.reset(*args, **kwargs)

    def set_init_state(self, *args: Any, **kwargs: Any):
        return self.environment.set_init_state(*args, **kwargs)

    def get_observation(self, *args: Any, **kwargs: Any):
        return self.environment.get_observation(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any):
        self.steps += 1
        return self.environment.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)


def _frozen_thresholds(checkpoint: Path) -> dict[int, float]:
    path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    if stream_sha256(path) != D2_EXIT_THRESHOLDS_SHA256:
        raise D9CCollectionError("frozen A1 thresholds changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    return {int(layer): float(threshold) for layer, threshold in value.items()}


def _original_a1_controller(
    cfg: GenerateConfig, model: torch.nn.Module, device: torch.device
) -> ExitController:
    exit_layers = tuple(model.get_all_exit_idx(cfg.exit_interval))
    if exit_layers != tuple(range(1, 28, 2)):
        raise D9CCollectionError("original A1 exit schedule differs")
    value_net = ActionValueNet(
        exit_list=list(exit_layers),
        exit_head=model.action_head,
        model=model,
        interval=cfg.exit_interval,
        threshold_type=cfg.threshold_type,
        anchor=False,
    )
    controller = ExitController(
        value_net,
        exit_id_list=list(exit_layers),
        steps_per_stage=cfg.steps_per_stage,
        leq=True,
        exit_dist=cfg.exit_dist,
        max_layer=model.config.n_layers,
    )
    thresholds = _frozen_thresholds(Path(cfg.pretrained_checkpoint))
    if set(thresholds) != set(exit_layers):
        raise D9CCollectionError("original A1 threshold keys differ")
    controller.thresholds = thresholds
    controller.to(device)
    controller.eval()
    return controller


def _phase_route_controller(
    cfg: GenerateConfig,
    model: torch.nn.Module,
    device: torch.device,
    runtime: Any,
) -> ExitController:
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
        _frozen_thresholds(Path(cfg.pretrained_checkpoint)), lower_is_easier=True
    )
    controller.thresholds = dict(zip(plan.eligible_exit_layers, selected))
    controller.set_phase_route_runtime_adapter(runtime.adapter)
    controller.to(device)
    controller.eval()
    return controller


def _next_attempt(pair_dir: Path, arm: str) -> Path:
    root = pair_dir / ".attempts" / arm
    root.mkdir(parents=True, exist_ok=True)
    existing = list(root.glob("attempt_*.incomplete")) + list(root.glob("abort_*.json"))
    ordinal = len(existing) + 1
    path = root / f"attempt_{ordinal:03d}.incomplete"
    path.mkdir(parents=False, exist_ok=False)
    return path


def _completed_arm_summary(
    arm_dir: Path,
    *,
    arm: str,
    canonical_key: str,
    task_id: int,
    episode_index: int,
    seed: int,
    state_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    result_path = arm_dir / "result.json"
    result = read_json_object(result_path)
    required = {
        "status": "COMPLETE_V3_D9C_ARM_ROLLOUT",
        "schema_version": D9C_ARM_SCHEMA_VERSION,
        "arm": arm,
        "canonical_key": canonical_key,
        "task_id": task_id,
        "episode_index": episode_index,
        "seed": seed,
        "initial_state_sha256": state_sha256,
        "source_git_commit": source_commit,
    }
    if any(result.get(name) != value for name, value in required.items()):
        raise D9CCollectionError(f"completed arm evidence differs: {arm_dir}")
    sidecar = (arm_dir / "result.sha256").read_text(encoding="utf-8").split()[0]
    observed = sha256_file(result_path)
    if sidecar != observed:
        raise D9CCollectionError("completed arm result SHA-256 differs")
    return {
        "status": result["status"],
        "success": result["success"],
        "environment_steps": result["environment_steps"],
        "policy_calls": result["policy_accounting"]["policy_calls"],
        "fm_calls": result["policy_accounting"]["fm_calls"],
        "fm_steps": result["policy_accounting"]["fm_steps"],
        "exit_layer_counts": result["policy_accounting"]["exit_layer_counts"],
        "policy_wall_seconds": result["policy_wall_seconds"],
        "rollout_wall_seconds": result["rollout_wall_seconds"],
        "seed": result["seed"],
        "initial_state_sha256": result["initial_state_sha256"],
        "source_git_commit": result["source_git_commit"],
        "physical_gpu_index": result["gpu"]["physical_index"],
        "gpu_uuid": result["gpu"]["uuid"],
        "result_sha256": observed,
    }


def _run_arm_impl(
    *,
    cfg: GenerateConfig,
    environment: Any,
    task_description: str,
    model: torch.nn.Module,
    controller: ExitController,
    device: torch.device,
    runtime: Any,
    record: Any,
    arm: str,
    initial_state: np.ndarray,
    initial_state_sha256: str,
    pair_dir: Path,
    source_commit: str,
    physical_gpu_index: int,
    gpu_uuid: str,
    model_audit: Mapping[str, Any],
    d9b_audit: Mapping[str, Any],
    runner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    final_dir = pair_dir / arm
    if final_dir.is_dir():
        return _completed_arm_summary(
            final_dir,
            arm=arm,
            canonical_key=record.canonical_key,
            task_id=record.task_id,
            episode_index=record.episode_index,
            seed=record.seed,
            state_sha256=initial_state_sha256,
            source_commit=source_commit,
        )
    if final_dir.exists():
        raise D9CCollectionError(f"arm output is not a directory: {final_dir}")

    attempt = _next_attempt(pair_dir, arm)
    (attempt / "command.txt").write_text(_command_text(), encoding="utf-8")
    telemetry_path = attempt / "policy_calls.jsonl"
    runtime_path = attempt / "phase_route_runtime.jsonl"
    cache_dir = attempt / "same_noise_cache"
    log_path = attempt / "eval.log"
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=10)
    teacher = None
    if arm == PHASE_ROUTE_ARM:
        teacher = SafeVisionTeacherCacheWriter(
            cache_dir,
            feature_dtype="float16",
            teacher_kind=PHASE_ROUTE_TEACHER_KIND,
            checkpoint_sha256=D2_CHECKPOINT_SHA256,
        )
    runtime_before = runtime.policy_calls if arm == PHASE_ROUTE_ARM else 0
    counted = CountingEnvironment(environment)
    rollout_started = time.perf_counter()
    success = False
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            set_seed_everywhere(record.seed)
            success, replay_images, _ = run_episode(
                cfg=cfg,
                env=counted,
                task_description=task_description,
                model=model,
                exit_controller=controller,
                device=device,
                resize_size=model.config.vision_backbone.image_default_input_size,
                initial_state=np.array(initial_state, copy=True),
                log_file=log_file,
                task_id=record.task_id,
                episode_idx=record.episode_index,
                telemetry_logger=telemetry,
                vision_teacher_cache_writer=teacher,
                phase_depth_runtime=None,
                phase_depth_control_enabled=False,
                episode_id_override=record.canonical_key,
                phase_route_runtime=runtime if arm == PHASE_ROUTE_ARM else None,
            )
            del replay_images
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except BaseException as error:
        telemetry.close()
        if teacher is not None:
            teacher.close()
        abort = {
            "status": "ABORT_V3_D9C_INFRASTRUCTURE_FAILURE",
            "timestamp_utc": utc_now(),
            "arm": arm,
            "canonical_key": record.canonical_key,
            "task_id": record.task_id,
            "episode_index": record.episode_index,
            "seed": record.seed,
            "initial_state_sha256": initial_state_sha256,
            "source_git_commit": source_commit,
            "physical_gpu_index": physical_gpu_index,
            "gpu_uuid": gpu_uuid,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "same_tuple_required_for_retry": True,
            "outcome_based_retry": False,
        }
        write_json(attempt / "abort.json", abort)
        raise
    rollout_wall = time.perf_counter() - rollout_started
    telemetry.close()
    if teacher is not None:
        teacher.close()
    if telemetry.error_count:
        raise D9CCollectionError(
            f"telemetry writer failed: {telemetry.last_error}"
        )
    if teacher is not None and teacher.error_count:
        raise D9CCollectionError(f"cache writer failed: {teacher.last_error}")

    telemetry_records = read_jsonl(telemetry_path)
    policy = summarize_policy_telemetry(
        telemetry_records,
        arm=arm,
        expected_episode_id=record.canonical_key,
        expected_task_id=record.task_id,
    )
    runtime_summary = None
    cache_summary = None
    cache_inventory_sha = None
    if arm == PHASE_ROUTE_ARM:
        current_records = runtime.records
        runtime_records = tuple(current_records[runtime_before:])
        runtime_summary = validate_phase_route_runtime_records(
            runtime_records,
            expected_episode_id=record.canonical_key,
            expected_task_id=record.task_id,
            expected_policy_calls=policy["policy_calls"],
        )
        write_jsonl(runtime_path, runtime_records)
        manifest_path = cache_dir / "manifest.jsonl"
        manifest = read_jsonl(manifest_path)
        cache_summary = validate_phase_route_cache(
            manifest,
            expected_episode_id=record.canonical_key,
            expected_task_id=record.task_id,
            expected_policy_calls=policy["policy_calls"],
        )
        telemetry_keys = {
            (str(item["episode_id"]), int(item["step_id"]))
            for item in telemetry_records
        }
        cache_keys = {
            (str(item["episode_id"]), int(item["step_id"])) for item in manifest
        }
        if telemetry_keys != cache_keys:
            raise D9CCollectionError("PhaseRoute telemetry/cache keys differ")
        inventory = build_file_inventory(
            cache_dir, (str(item["array_path"]) for item in manifest)
        )
        inventory_path = cache_dir / "inventory.jsonl"
        write_jsonl(inventory_path, tuple(item.to_dict() for item in inventory))
        cache_inventory_sha = sha256_file(inventory_path)
        cache_summary["cache_bytes"] = sum(item.bytes for item in inventory)

    result = {
        "status": "COMPLETE_V3_D9C_ARM_ROLLOUT",
        "schema_version": D9C_ARM_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "arm": arm,
        "canonical_key": record.canonical_key,
        "task_id": record.task_id,
        "episode_index": record.episode_index,
        "seed": record.seed,
        "arm_order": list(record.arm_order),
        "initial_state_sha256": initial_state_sha256,
        "success": bool(success),
        "environment_steps": counted.steps,
        "control_steps": max(0, counted.steps - cfg.num_steps_wait),
        "policy_accounting": policy,
        "policy_wall_seconds": policy["policy_latency_ms"]["sum"] / 1000.0,
        "rollout_wall_seconds": rollout_wall,
        "telemetry": {
            "path": telemetry_path.relative_to(attempt).as_posix(),
            "sha256": sha256_file(telemetry_path),
            "records": len(telemetry_records),
        },
        "phase_route_runtime": (
            {
                **runtime_summary,
                "path": runtime_path.relative_to(attempt).as_posix(),
                "sha256": sha256_file(runtime_path),
            }
            if runtime_summary is not None
            else None
        ),
        "same_noise_cache": (
            {
                **cache_summary,
                "manifest_path": (cache_dir / "manifest.jsonl")
                .relative_to(attempt)
                .as_posix(),
                "manifest_sha256": sha256_file(cache_dir / "manifest.jsonl"),
                "inventory_path": (cache_dir / "inventory.jsonl")
                .relative_to(attempt)
                .as_posix(),
                "inventory_sha256": cache_inventory_sha,
            }
            if cache_summary is not None
            else None
        ),
        "source_git_commit": source_commit,
        "source_worktree_dirty": False,
        "gpu": {
            "physical_index": physical_gpu_index,
            "uuid": gpu_uuid,
            "visible_count": torch.cuda.device_count(),
            "visible_name": torch.cuda.get_device_name(0),
        },
        "model_audit": dict(model_audit),
        "prerequisites": {"D9B": dict(d9b_audit), "D9C_runner": dict(runner_audit)},
        "retry_policy": {
            "same_arm_task_episode_seed_state_commit": True,
            "outcome_based_retry": False,
            "replacement_episode_or_seed": False,
        },
        "claim_boundary": {
            "per_rollout_success_is_raw_evidence": True,
            "cross_pair_aggregate_computed": False,
            "D9_primary_gate_evaluated": False,
        },
    }
    result_path = attempt / "result.json"
    write_json(result_path, result)
    result_sha = sha256_file(result_path)
    (attempt / "result.sha256").write_text(
        f"{result_sha}  result.json\n", encoding="utf-8"
    )
    attempt.replace(final_dir)
    return _completed_arm_summary(
        final_dir,
        arm=arm,
        canonical_key=record.canonical_key,
        task_id=record.task_id,
        episode_index=record.episode_index,
        seed=record.seed,
        state_sha256=initial_state_sha256,
        source_commit=source_commit,
    )


def _run_arm(**kwargs: Any) -> dict[str, Any]:
    """Add an immutable abort ledger around rollout and evidence validation."""

    try:
        return _run_arm_impl(**kwargs)
    except BaseException as error:
        pair_dir = Path(kwargs["pair_dir"])
        arm = str(kwargs["arm"])
        record = kwargs["record"]
        attempt_root = pair_dir / ".attempts" / arm
        candidates = sorted(
            attempt_root.glob("attempt_*.incomplete"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for attempt in candidates:
            if (attempt / "abort.json").exists() or (attempt / "result.json").exists():
                continue
            abort = {
                "status": "ABORT_V3_D9C_INFRASTRUCTURE_FAILURE",
                "timestamp_utc": utc_now(),
                "arm": arm,
                "canonical_key": record.canonical_key,
                "task_id": record.task_id,
                "episode_index": record.episode_index,
                "seed": record.seed,
                "initial_state_sha256": kwargs["initial_state_sha256"],
                "source_git_commit": kwargs["source_commit"],
                "physical_gpu_index": kwargs["physical_gpu_index"],
                "gpu_uuid": kwargs["gpu_uuid"],
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "same_tuple_required_for_retry": True,
                "outcome_based_retry": False,
            }
            write_json(attempt / "abort.json", abort)
            break
        raise


def _run(args: argparse.Namespace) -> None:
    source_status = git_output("status", "--porcelain=v1")
    if source_status:
        raise PermissionError("D9C requires a clean frozen-runner worktree")
    if not _tracked_readiness_is_current():
        raise PermissionError("D9C runner readiness is not the tracked HEAD blob")
    source_commit = git_output("rev-parse", "HEAD")
    schedule = task_schedule(REPO_ROOT, args.task_id)
    output = validate_task_output(REPO_ROOT, args.task_id, args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError("D9C output exists; explicit --resume is required")
    if not output.exists() and args.resume:
        raise FileNotFoundError("D9C --resume requires an existing task output")

    d9b_audit = validate_d9b_readiness(REPO_ROOT)
    runner_audit = validate_runner_readiness(REPO_ROOT)
    checkpoint = args.checkpoint.resolve(strict=True)
    model_audit = validate_runtime_model_directory(
        checkpoint, args.model_attestation.resolve(strict=True)
    )
    router = args.router.resolve(strict=True)
    phase_checkpoint = args.phase_checkpoint.resolve(strict=True)
    if sha256_file(router) != EXPECTED_ROUTER_SHA256:
        raise D9CCollectionError("D9C router SHA-256 differs")
    if sha256_file(phase_checkpoint) != EXPECTED_PHASE_SHA256:
        raise D9CCollectionError("D9C phase checkpoint SHA-256 differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
        raise PermissionError("D9C CUDA_VISIBLE_DEVICES differs from assignment")
    if not torch.cuda.is_available():
        raise RuntimeError("D9C collection requires CUDA")
    torch.cuda.set_device(0)
    observed_uuid = str(torch.cuda.get_device_properties(0).uuid)
    validate_gpu_contract(
        task_id=args.task_id,
        physical_gpu_index=args.physical_gpu_index,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_gpu_count=torch.cuda.device_count(),
        expected_gpu_uuid=args.expected_gpu_uuid,
        observed_gpu_uuid=observed_uuid,
    )
    gpu_uuid = "GPU-" + observed_uuid.removeprefix("GPU-")

    suite = benchmark.get_benchmark_dict()[SUITE]()
    if suite.n_tasks != 10:
        raise D9CCollectionError("LIBERO-10 task count differs")
    states = suite.get_task_init_states(args.task_id)
    if len(states) <= max(record.episode_index for record in schedule):
        raise D9CCollectionError("LIBERO-10 does not contain frozen test indices")
    selected_states = {
        record.episode_index: np.asarray(states[record.episode_index])
        for record in schedule
    }
    state_hashes = {
        str(index): sha256_array(state) for index, state in selected_states.items()
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_V3_D9C_TASK_PREFLIGHT",
                    "task_id": args.task_id,
                    "canonical_keys": [record.canonical_key for record in schedule],
                    "episode_indices": [record.episode_index for record in schedule],
                    "seeds": {str(record.episode_index): record.seed for record in schedule},
                    "arm_orders": {
                        str(record.episode_index): list(record.arm_order)
                        for record in schedule
                    },
                    "initial_state_sha256": state_hashes,
                    "physical_gpu_index": args.physical_gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "D9B": d9b_audit,
                    "D9C_runner": runner_audit,
                    "model_audit": model_audit,
                    "model_loaded": False,
                    "environment_created": False,
                    "rollouts_run": 0,
                    "cross_pair_aggregate_computed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    completed_task_path = output / "task_result.json"
    if completed_task_path.is_file():
        completed = read_json_object(completed_task_path)
        if (
            not args.resume
            or completed.get("status") != D9C_TASK_STATUS
            or completed.get("schema_version") != D9C_TASK_SCHEMA_VERSION
            or completed.get("task_id") != args.task_id
            or completed.get("source_git_commit") != source_commit
            or completed.get("complete_pair_count") != len(schedule)
            or completed.get("complete_rollout_count") != 2 * len(schedule)
            or completed.get("initial_state_sha256") != state_hashes
        ):
            raise D9CCollectionError("completed D9C task evidence differs")
        for record in schedule:
            pair_path = output / f"pair_episode{record.episode_index}" / "pair_record.json"
            pair_value = read_json_object(pair_path)
            validate_pair_record(pair_value, record=record)
            if completed["pair_record_sha256"].get(record.canonical_key) != sha256_file(
                pair_path
            ):
                raise D9CCollectionError("completed D9C pair SHA-256 differs")
        print(D9C_TASK_STATUS, flush=True)
        return

    output.mkdir(parents=True, exist_ok=args.resume)
    if not (output / "command.txt").exists():
        (output / "command.txt").write_text(_command_text(), encoding="utf-8")
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=SUITE,
        num_trials_per_task=len(schedule),
        action_head_flow_matching_inference_steps=FM_STEPS,
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
        seed=schedule[0].seed,
        run_id_note=f"v3-d9c-task{args.task_id}",
        vision_aggregation_enabled=False,
        rp_pep_enabled=False,
        phase_route_v3_enabled=False,
    )
    threshold_path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    threshold_before = threshold_path.read_bytes()
    set_seed_everywhere(cfg.seed)
    model, device, _ = initialize_and_load_model(cfg)
    if threshold_path.read_bytes() != threshold_before:
        raise D9CCollectionError("model initialization changed frozen thresholds")
    runtime = load_frozen_phase_route_runtime(router, phase_checkpoint)
    original_controller = _original_a1_controller(cfg, model, device)
    phase_controller = _phase_route_controller(cfg, model, device, runtime)
    if threshold_path.read_bytes() != threshold_before:
        raise D9CCollectionError("controller initialization changed frozen thresholds")
    task = suite.get_task(args.task_id)
    environment, task_description = get_libero_env(
        task, cfg.model_family, resolution=cfg.env_img_res
    )
    pair_shas: dict[str, str] = {}
    try:
        for record in schedule:
            pair_dir = output / f"pair_episode{record.episode_index}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            pair_path = pair_dir / "pair_record.json"
            if pair_path.is_file():
                pair_value = read_json_object(pair_path)
                validate_pair_record(pair_value, record=record)
                pair_shas[record.canonical_key] = sha256_file(pair_path)
                print(f"D9C pair already complete: {record.canonical_key}", flush=True)
                continue
            state = selected_states[record.episode_index]
            state_sha = state_hashes[str(record.episode_index)]
            arm_summaries: dict[str, Any] = {}
            for arm in record.arm_order:
                controller = (
                    original_controller if arm == ORIGINAL_A1_ARM else phase_controller
                )
                arm_summaries[arm] = _run_arm(
                    cfg=cfg,
                    environment=environment,
                    task_description=task_description,
                    model=model,
                    controller=controller,
                    device=device,
                    runtime=runtime,
                    record=record,
                    arm=arm,
                    initial_state=state,
                    initial_state_sha256=state_sha,
                    pair_dir=pair_dir,
                    source_commit=source_commit,
                    physical_gpu_index=args.physical_gpu_index,
                    gpu_uuid=gpu_uuid,
                    model_audit=model_audit,
                    d9b_audit=d9b_audit,
                    runner_audit=runner_audit,
                )
                print(
                    f"D9C raw arm complete: {record.canonical_key} arm={arm}",
                    flush=True,
                )
            pair_value = {
                "status": "COMPLETE_V3_D9C_PAIRED_ACTIVE_PAIR",
                "schema_version": D9C_PAIR_SCHEMA_VERSION,
                "timestamp_utc": utc_now(),
                "canonical_key": record.canonical_key,
                "task_id": record.task_id,
                "episode_index": record.episode_index,
                "seed": record.seed,
                "arm_order": list(record.arm_order),
                "arms": arm_summaries,
                "claim_boundary": {
                    "raw_pair_evidence_only": True,
                    "cross_pair_aggregate_computed": False,
                    "D9_primary_gate_evaluated": False,
                },
            }
            validate_pair_record(pair_value, record=record)
            write_json(pair_path, pair_value)
            pair_sha = sha256_file(pair_path)
            (pair_dir / "pair_record.sha256").write_text(
                f"{pair_sha}  pair_record.json\n", encoding="utf-8"
            )
            pair_shas[record.canonical_key] = pair_sha
            print(f"D9C pair complete: {record.canonical_key}", flush=True)
    finally:
        if hasattr(environment, "close"):
            environment.close()

    if len(pair_shas) != len(schedule):
        raise D9CCollectionError("D9C task did not complete all ten pairs")
    task_result_path = output / "task_result.json"
    if task_result_path.exists():
        raise FileExistsError("D9C task result already exists")
    task_result = {
        "status": D9C_TASK_STATUS,
        "schema_version": D9C_TASK_SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "task_id": args.task_id,
        "physical_gpu_index": args.physical_gpu_index,
        "gpu_uuid": gpu_uuid,
        "source_git_commit": source_commit,
        "complete_pair_count": len(pair_shas),
        "complete_rollout_count": 2 * len(pair_shas),
        "pair_record_sha256": pair_shas,
        "initial_state_sha256": state_hashes,
        "interim_aggregate": {
            "success": False,
            "safety": False,
            "efficiency": False,
        },
        "claim_boundary": {
            "task_success_rate_computed": False,
            "overall_or_per_task_gate_evaluated": False,
            "D9C_collection_is_D9_result": False,
        },
    }
    write_json(task_result_path, task_result)
    task_sha = sha256_file(task_result_path)
    (output / "task_result.sha256").write_text(
        f"{task_sha}  task_result.json\n", encoding="utf-8"
    )
    print(D9C_TASK_STATUS, flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "ABORT_V3_D9C_TASK",
                    "task_id": args.task_id,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "incomplete_status": (
                        "INCOMPLETE_V3_D9_INDEPENDENT_TEST_NOT_PASS_OR_NEGATIVE"
                    ),
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
