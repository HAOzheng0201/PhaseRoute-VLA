#!/usr/bin/env python3
"""Collect one Stage-11D development task under frozen original A1 control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Mapping


_REQUESTED_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
# State lineage must pass before this process can initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "scripts/dynamic_compute"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from a1.vla.dynamic_compute.route_first_reliability_artifacts import (  # noqa: E402
    sha256_file,
)
from a1.vla.dynamic_compute.route_first_reliability import (  # noqa: E402
    STAGE11D_TASK_IDS,
)
from a1.vla.dynamic_compute.route_first_reliability_collection import (  # noqa: E402
    STAGE11D_A1_CHECKPOINT_SHA256,
    STAGE11D_COLLECTION_CLUSTERS_PER_TASK,
    STAGE11D_COLLECTION_FM_STEPS,
    STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH,
    STAGE11D_COLLECTION_REPLAY_LAYERS,
    STAGE11D_COLLECTION_ROLE,
    STAGE11D_COLLECTION_SUITE,
    STAGE11D_RAW_TASK_SCHEMA,
    Stage11DCollectionError,
    Stage11DDevelopmentTaskSuite,
    load_development_states,
    load_development_task_calls,
    task_development_schedule,
    validate_collection_readiness,
    validate_episode_id_override,
    validate_gpu_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=40_000)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "model/libero_exit"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Must equal runs/route_first_stage11d_development_raw/taskXX.",
    )
    parser.add_argument("--cpu-preflight-only", action="store_true")
    parser.add_argument("--gpu-preflight-only", action="store_true")
    parser.add_argument("--model-load-smoke", action="store_true")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True
    ).strip()


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise Stage11DCollectionError(f"empty JSONL line in {path}")
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise Stage11DCollectionError(f"JSONL object required in {path}")
        records.append(dict(value))
    return tuple(records)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _state_sha256(value: Any) -> str:
    from a1.vla.dynamic_compute.route_first_reliability_artifacts import (
        canonical_state_bytes,
    )

    return canonical_state_bytes(value)[2]


def _artifact_inventory(root: Path, paths: tuple[Path, ...]) -> dict[str, Any]:
    inventory = {}
    for path in paths:
        target = path.resolve(strict=True)
        relative = str(target.relative_to(root.resolve(strict=True)))
        inventory[relative] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    return inventory


def _run(args: argparse.Namespace) -> None:
    if args.cpu_preflight_only and (
        args.gpu_preflight_only or args.model_load_smoke
    ):
        raise ValueError("CPU preflight cannot be combined with GPU operations")
    if args.gpu_preflight_only and args.model_load_smoke:
        raise ValueError("GPU preflight and model-load smoke are separate gates")
    if _git("status", "--porcelain=v1"):
        raise PermissionError("Stage-11D collection requires a clean worktree")

    readiness = validate_collection_readiness(REPO_ROOT)
    schedule, states_by_task, state_attestation = load_development_states(REPO_ROOT)
    task_schedule = task_development_schedule(schedule, args.task_id)
    output = (
        REPO_ROOT
        / STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH
        / f"task{args.task_id:02d}"
    ).resolve()
    if args.output_dir is not None and args.output_dir.resolve() != output:
        raise PermissionError("Stage-11D task output path differs")
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("Stage-11D collection refuses existing task evidence")
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint != (REPO_ROOT / "model/libero_exit").resolve(strict=True):
        raise PermissionError("Stage-11D checkpoint path differs from readiness")
    state_hashes = {
        str(record.replicate_id): _state_sha256(
            states_by_task[args.task_id][record.replicate_id]
        )
        for record in task_schedule
    }
    common_preflight = {
        "task_id": args.task_id,
        "replicate_ids": [record.replicate_id for record in task_schedule],
        "cluster_keys": [record.cluster_key for record in task_schedule],
        "policy_seeds": {
            str(record.replicate_id): record.policy_seed for record in task_schedule
        },
        "state_sha256": state_hashes,
        "source_git_commit": _git("rev-parse", "HEAD"),
        "readiness_status": readiness["status"],
        "state_payload_sha256": state_attestation["payload_sha256"],
        "output_absent": True,
        "original_A1_only": True,
        "calibration_or_shadow_rollout_opened": False,
        "same_noise_replay_run": False,
        "training_run": False,
        "active_new_router_control": False,
    }
    if args.cpu_preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_ROUTE_FIRST_STAGE11D_COLLECTION_CPU_PREFLIGHT",
                    **common_preflight,
                    "CUDA_initialized": False,
                    "model_loaded": False,
                    "LIBERO_environment_opened": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    if _REQUESTED_VISIBLE_DEVICES is None:
        raise PermissionError("Stage-11D task has no CUDA_VISIBLE_DEVICES binding")
    os.environ["CUDA_VISIBLE_DEVICES"] = _REQUESTED_VISIBLE_DEVICES
    import torch

    if torch.cuda.is_initialized():
        raise PermissionError("CUDA initialized before Stage-11D state validation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage-11D collection requires exactly one visible GPU")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_properties(0)
    validate_gpu_contract(
        physical_gpu_index=args.physical_gpu_index,
        visible_devices=_REQUESTED_VISIBLE_DEVICES,
        visible_gpu_count=torch.cuda.device_count(),
        expected_gpu_uuid=args.expected_gpu_uuid,
        observed_gpu_uuid=str(gpu.uuid),
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_memory_mib = free_bytes // (1024 * 1024)
    if free_memory_mib < args.minimum_free_memory_mib:
        raise RuntimeError("Stage-11D assigned GPU no longer has enough free memory")
    if args.gpu_preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_ROUTE_FIRST_STAGE11D_COLLECTION_GPU_PREFLIGHT",
                    **common_preflight,
                    "physical_gpu_index": args.physical_gpu_index,
                    "gpu_uuid": str(gpu.uuid),
                    "gpu_name": gpu.name,
                    "free_memory_mib": free_memory_mib,
                    "total_memory_mib": total_bytes // (1024 * 1024),
                    "model_loaded": False,
                    "LIBERO_environment_opened": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    from libero.libero import benchmark
    import robot_experiments.libero.eval_libero_early_exit as eval_module
    from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
    from a1.vla.dynamic_compute.vision_teacher_cache import (
        SafeVisionTeacherCacheWriter,
        VISION_TEACHER_CACHE_SCHEMA_VERSION,
        has_complete_candidate_fm_traces,
    )
    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        get_image_resize_size,
        initialize_and_load_model,
        run_task,
        setup_logging,
    )
    from robot_experiments.robot_utils import set_seed_everywhere
    from smoke_m1_telemetry import make_exit_controller

    config = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=STAGE11D_COLLECTION_SUITE,
        num_trials_per_task=STAGE11D_COLLECTION_CLUSTERS_PER_TASK,
        action_head_flow_matching_inference_steps=STAGE11D_COLLECTION_FM_STEPS,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        local_log_dir=str(incomplete / "eval_logs"),
        save_rollout_video=False,
        save_rollout_video_path=str(incomplete),
        use_wandb=False,
        reseed_each_episode=True,
        seed=94_260_830,
        run_id_note=f"route-first-stage11d-original-a1-task{args.task_id}",
        vision_aggregation_enabled=False,
        learned_vision_aggregation_checkpoint=None,
        phase_depth_enabled=False,
        rp_pep_enabled=False,
        phase_route_v3_enabled=False,
    )
    thresholds_path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    thresholds_before = thresholds_path.read_bytes()
    set_seed_everywhere(config.seed)
    print(
        f"[Stage11D] loading frozen original A1 task={args.task_id}", flush=True
    )
    model, device, _ = initialize_and_load_model(config)
    controller = make_exit_controller(config, model, device)
    if (
        thresholds_path.read_bytes() != thresholds_before
        or tuple(controller.exit_id_list) != tuple(range(1, 28, 2))
        or set(controller.thresholds) != set(range(1, 28, 2))
    ):
        raise Stage11DCollectionError("Frozen original A1 controller differs")
    if args.model_load_smoke:
        print(
            json.dumps(
                {
                    "status": "PASS_ROUTE_FIRST_STAGE11D_A1_MODEL_LOAD_SMOKE",
                    **common_preflight,
                    "physical_gpu_index": args.physical_gpu_index,
                    "gpu_uuid": str(gpu.uuid),
                    "model_loaded": True,
                    "eligible_exit_layers": list(controller.exit_id_list),
                    "LIBERO_environment_opened": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

    incomplete.mkdir(parents=True, exist_ok=False)
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "DATA_DIR",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLA_CONFIG_YAML",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "CUBLAS_WORKSPACE_CONFIG",
    )
    environment = {
        name: os.environ[name] for name in environment_names if name in os.environ
    }
    command = shlex.join(
        ["env", *(f"{key}={value}" for key, value in environment.items()), sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(
        "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + command + "\n",
        encoding="utf-8",
    )

    base_suite = benchmark.get_benchmark_dict()[STAGE11D_COLLECTION_SUITE]()
    if base_suite.n_tasks != len(STAGE11D_TASK_IDS):
        raise Stage11DCollectionError("LIBERO-10 task count differs")
    suite = Stage11DDevelopmentTaskSuite(base_suite, states_by_task)
    telemetry_path = incomplete / "policy_calls.jsonl"
    cache_directory = incomplete / "observation_calls"
    resize_size = get_image_resize_size(config)
    log_file, eval_log_path, _ = setup_logging(config, model.config.action_head)
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=25)
    observer = SafeVisionTeacherCacheWriter(
        cache_directory,
        feature_dtype="float16",
        teacher_kind="frozen_original_a1_observer",
        checkpoint_sha256=STAGE11D_A1_CHECKPOINT_SHA256,
    )
    original_run_episode = eval_module.run_episode

    def run_episode_with_identity(*call_args: Any, **call_kwargs: Any):
        if len(call_args) > 10:
            replicate_id = int(call_args[10])
            task_id = int(call_args[9])
        else:
            replicate_id = int(call_kwargs["episode_idx"])
            task_id = int(call_kwargs["task_id"])
        record = task_schedule[replicate_id]
        cluster_key = validate_episode_id_override(
            record.cluster_key, task_id=task_id, replicate_id=replicate_id
        )
        mutable_kwargs = dict(call_kwargs)
        if "episode_id_override" in mutable_kwargs:
            raise PermissionError("Stage-11D identity override was already supplied")
        mutable_kwargs["episode_id_override"] = cluster_key
        return original_run_episode(*call_args, **mutable_kwargs)

    eval_module.run_episode = run_episode_with_identity
    try:
        episodes, successes, exit_sum, exit_count = run_task(
            cfg=config,
            task_suite=suite,
            task_id=args.task_id,
            model=model,
            exit_controller=controller,
            device=device,
            num_tasks=1,
            resize_size=resize_size,
            total_episodes=0,
            total_successes=0,
            log_file=log_file,
            total_exit_mean_sum=0.0,
            total_exit_mean_count=0,
            telemetry_logger=telemetry,
            vision_teacher_cache_writer=observer,
            phase_depth_runtime=None,
            phase_depth_control_enabled=False,
            phase_route_runtime=None,
        )
    finally:
        eval_module.run_episode = original_run_episode
        telemetry.close()
        observer.close()
        log_file.close()

    telemetry_records = _jsonl(telemetry_path)
    manifest_path = cache_directory / "manifest.jsonl"
    manifest_records = _jsonl(manifest_path)
    validated_calls = load_development_task_calls(incomplete, task_id=args.task_id)
    telemetry_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in telemetry_records
    }
    manifest_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in manifest_records
    }
    expected_ids = {record.cluster_key for record in task_schedule}
    observed_ids = {str(record["episode_id"]) for record in manifest_records}
    payloads = [cache_directory / str(record["array_path"]) for record in manifest_records]
    latencies = [float(record["latency_ms"]) for record in telemetry_records]
    source_status = _git("status", "--porcelain=v1")
    checks = {
        "readiness_and_state_binding_current": bool(readiness),
        "one_verified_visible_gpu": True,
        "all_12_development_clusters_completed": episodes
        == STAGE11D_COLLECTION_CLUSTERS_PER_TASK,
        "development_cluster_identity_exact": observed_ids == expected_ids,
        "calibration_and_shadow_replicates_absent": all(
            int(call.replicate_id) in range(12) for call in validated_calls
        ),
        "telemetry_and_observation_keys_align": telemetry_by_key.keys()
        == manifest_by_key.keys(),
        "exit_and_fm_metadata_align": telemetry_by_key.keys()
        == manifest_by_key.keys()
        and all(
            int(manifest_by_key[key]["teacher_exit_layer"])
            == int(telemetry_by_key[key]["exit_layer"])
            and int(manifest_by_key[key]["fm_calls"])
            == int(telemetry_by_key[key]["fm_calls"])
            for key in manifest_by_key
        ),
        "raw_cache_schema_and_fm_trace_complete": bool(manifest_records)
        and all(
            record["schema_version"] == VISION_TEACHER_CACHE_SCHEMA_VERSION
            and record["checkpoint_sha256"] == STAGE11D_A1_CHECKPOINT_SHA256
            and int(record["source_projected_tokens"]) == 576
            and int(record["unique_visual_slots"]) == 288
            and int(record["valid_crop_count"]) == 4
            and has_complete_candidate_fm_traces(record)
            for record in manifest_records
        ),
        "all_cache_payloads_present": bool(payloads)
        and all(path.is_file() for path in payloads),
        "observer_writes_error_free": telemetry.error_count == 0
        and observer.error_count == 0,
        "router_phase_and_vision_compression_disabled": not config.phase_route_v3_enabled
        and not config.phase_depth_enabled
        and not config.vision_aggregation_enabled,
        "source_worktree_clean": not bool(source_status),
    }
    passed = all(checks.values())
    result = {
        "schema_version": STAGE11D_RAW_TASK_SCHEMA,
        "status": (
            "PASS_ROUTE_FIRST_STAGE11D_RAW_TASK"
            if passed
            else "FAIL_ROUTE_FIRST_STAGE11D_RAW_TASK"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": STAGE11D_COLLECTION_ROLE,
        "suite": STAGE11D_COLLECTION_SUITE,
        "task_id": args.task_id,
        "replicate_ids": [record.replicate_id for record in task_schedule],
        "cluster_keys": [record.cluster_key for record in task_schedule],
        "state_sha256": state_hashes,
        "policy_seeds": {
            str(record.replicate_id): record.policy_seed for record in task_schedule
        },
        "completed_clusters": episodes,
        "behavior_successes": successes,
        "behavior_success_rate": successes / episodes if episodes else 0.0,
        "policy_calls": len(telemetry_records),
        "observation_cache_calls": len(manifest_records),
        "validated_manifest_calls": len(validated_calls),
        "behavior_mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "cache_bytes": sum(path.stat().st_size for path in payloads if path.is_file()),
        "source_git_commit": _git("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "gpu_audit": {
            "physical_gpu_index": args.physical_gpu_index,
            "expected_gpu_uuid": args.expected_gpu_uuid,
            "visible_gpu_uuid": str(gpu.uuid),
            "visible_gpu_count": torch.cuda.device_count(),
            "free_memory_mib_before_model_load": free_memory_mib,
        },
        "lineage": {
            "protocol_sha256": readiness["protocol_sha256"],
            "state_binding_sha256": readiness["state_binding_sha256"],
            "state_payload_sha256": state_attestation["payload_sha256"],
            "checkpoint_sha256": STAGE11D_A1_CHECKPOINT_SHA256,
            "readiness_sha256": sha256_file(
                REPO_ROOT
                / "results/route_first/route_first_stage11d_collection_runner_readiness.json"
            ),
        },
        "checks": checks,
        "access_ledger": {
            "development_clusters_opened": STAGE11D_COLLECTION_CLUSTERS_PER_TASK,
            "calibration_clusters_rolled_out": 0,
            "shadow_confirmation_clusters_rolled_out": 0,
            "behavior_policy": "frozen_original_A1_early_exit",
            "new_router_loaded": False,
            "same_noise_counterfactual_replay_calls": 0,
            "new_router_active_control_calls": 0,
        },
        "claim_boundary": {
            "behavior_success_is_descriptive_only": True,
            "L13_L27_reliability_truth_computed": False,
            "feasibility_gate_evaluated": False,
            "active_new_method_control": False,
            "speedup_or_superiority_claim_authorized": False,
        },
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    inventory = _artifact_inventory(
        incomplete,
        (result_path, telemetry_path, manifest_path, *payloads),
    )
    _write_json_once(incomplete / "artifact_inventory.json", inventory)
    if not passed:
        raise RuntimeError("Stage-11D raw task failed one or more gates")
    incomplete.rename(output)
    print("PASS_ROUTE_FIRST_STAGE11D_RAW_TASK", flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        output = (
            REPO_ROOT
            / STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH
            / f"task{args.task_id:02d}"
        ).resolve()
        incomplete = output.with_name(output.name + ".incomplete")
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_ROUTE_FIRST_STAGE11D_RAW_TASK",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
