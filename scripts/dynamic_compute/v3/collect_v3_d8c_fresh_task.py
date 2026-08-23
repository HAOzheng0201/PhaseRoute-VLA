#!/usr/bin/env python3
"""Collect one D8C task on 20 frozen generated states under original A1."""

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
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "dynamic_compute"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from libero.libero import benchmark  # noqa: E402
import robot_experiments.libero.eval_libero_early_exit as eval_module  # noqa: E402

from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger  # noqa: E402
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_CHECKPOINT_SHA256,
    D2_EXIT_THRESHOLDS_SHA256,
    stream_sha256,
    validate_runtime_model_directory,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_POLICY_SEED_BASE,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation_collection import (  # noqa: E402
    D8C_FM_STEPS,
    D8C_RAW_TASK_SCHEMA_VERSION,
    D8C_ROLE,
    D8C_SUITE,
    FreshStateTaskSuite,
    hash_state,
    load_fresh_states,
    task_fresh_schedule,
    validate_d8c_gpu_contract,
    validate_d8c_prerequisites,
    validate_episode_id_override,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (  # noqa: E402
    SafeVisionTeacherCacheWriter,
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)
from robot_experiments.libero.eval_libero_early_exit import (  # noqa: E402
    GenerateConfig,
    get_image_resize_size,
    initialize_and_load_model,
    run_task,
    setup_logging,
)
from robot_experiments.robot_utils import set_seed_everywhere  # noqa: E402
from smoke_m1_telemetry import make_exit_controller  # noqa: E402


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model-load-smoke", action="store_true")
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(args: argparse.Namespace) -> None:
    prerequisites = validate_d8c_prerequisites(REPO_ROOT)
    schedule, states_by_task = load_fresh_states(REPO_ROOT)
    task_schedule = task_fresh_schedule(schedule, args.task_id)
    expected_output = (
        REPO_ROOT / "reports" / "v3_d8_fresh_raw" / f"task{args.task_id}"
    ).resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("D8C raw task output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D8C refuses to overwrite raw task evidence")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8C raw collection requires a clean worktree")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
        raise PermissionError("D8C CUDA_VISIBLE_DEVICES differs from assignment")
    if not torch.cuda.is_available():
        raise RuntimeError("D8C raw collection requires CUDA")
    torch.cuda.set_device(0)
    validate_d8c_gpu_contract(
        physical_gpu_index=args.physical_gpu_index,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_gpu_count=torch.cuda.device_count(),
        expected_gpu_uuid=args.expected_gpu_uuid,
        observed_gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    frozen_source = (REPO_ROOT.parents[1] / "source" / "model" / "libero_exit").resolve()
    if checkpoint == frozen_source:
        raise PermissionError("D8C cannot use the writable-sidecar source directory")
    model_audit = validate_runtime_model_directory(checkpoint, args.model_attestation)
    base_suite = benchmark.get_benchmark_dict()[D8C_SUITE]()
    suite = FreshStateTaskSuite(base_suite, states_by_task)
    initial_states = suite.get_task_init_states(args.task_id)
    initial_state_sha256 = {
        str(record.replicate_id): hash_state(initial_states[record.replicate_id])
        for record in task_schedule
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_V3_D8C_RAW_TASK_PREFLIGHT",
                    "task_id": args.task_id,
                    "replicate_ids": [record.replicate_id for record in task_schedule],
                    "policy_seeds": {
                        str(record.replicate_id): record.policy_seed
                        for record in task_schedule
                    },
                    "initial_state_sha256": initial_state_sha256,
                    "prerequisites": prerequisites,
                    "model_audit": model_audit,
                    "physical_gpu_index": args.physical_gpu_index,
                    "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                    "output_absent": not output.exists() and not incomplete.exists(),
                    "model_loaded": False,
                    "rollout_run": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    smoke_cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=D8C_SUITE,
        num_trials_per_task=len(task_schedule),
        action_head_flow_matching_inference_steps=D8C_FM_STEPS,
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
        seed=D8_POLICY_SEED_BASE,
        run_id_note=f"v3-d8c-fresh-task{args.task_id}",
        vision_aggregation_enabled=False,
    )
    threshold_path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    threshold_before = json.loads(threshold_path.read_text(encoding="utf-8"))
    set_seed_everywhere(smoke_cfg.seed)
    model, device, _ = initialize_and_load_model(smoke_cfg)
    exit_controller = make_exit_controller(smoke_cfg, model, device)
    threshold_after = json.loads(threshold_path.read_text(encoding="utf-8"))
    if (
        threshold_after != threshold_before
        or stream_sha256(threshold_path) != D2_EXIT_THRESHOLDS_SHA256
    ):
        raise RuntimeError("D8C model initialization changed frozen thresholds")
    if args.model_load_smoke:
        print(
            json.dumps(
                {
                    "status": "PASS_V3_D8C_MODEL_LOAD_SMOKE",
                    "task_id": args.task_id,
                    "device": str(device),
                    "n_layers": int(model.config.n_layers),
                    "eligible_exit_layers": list(exit_controller.exit_id_list),
                    "rollout_run": False,
                    "output_written": False,
                },
                ensure_ascii=False,
                indent=2,
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
        [
            "env",
            *(f"{key}={value}" for key, value in environment.items()),
            sys.executable,
            *sys.argv,
        ]
    )
    (incomplete / "command.txt").write_text(
        "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + command + "\n",
        encoding="utf-8",
    )

    cfg = smoke_cfg
    cfg.local_log_dir = str(incomplete / "eval_logs")
    cfg.save_rollout_video_path = str(incomplete)
    telemetry_path = incomplete / "policy_calls.jsonl"
    teacher_cache_dir = incomplete / "teacher_calls"
    resize_size = get_image_resize_size(cfg)
    log_file, eval_log_path, _ = setup_logging(cfg, model.config.action_head)
    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=25)
    teacher = SafeVisionTeacherCacheWriter(
        teacher_cache_dir,
        feature_dtype="float16",
        teacher_kind="a1_early_exit",
        checkpoint_sha256=D2_CHECKPOINT_SHA256,
    )
    original_run_episode = eval_module.run_episode

    def run_episode_with_fresh_identity(*call_args: Any, **call_kwargs: Any):
        if len(call_args) > 10:
            local_replicate = int(call_args[10])
            task_id = int(call_args[9])
        else:
            local_replicate = int(call_kwargs["episode_idx"])
            task_id = int(call_kwargs["task_id"])
        record = task_schedule[local_replicate]
        cluster_key = validate_episode_id_override(
            record.cluster_key,
            task_id=task_id,
            replicate_id=local_replicate,
        )
        mutable_kwargs = dict(call_kwargs)
        if "episode_id_override" in mutable_kwargs:
            raise PermissionError("D8C caller unexpectedly supplied an identity override")
        mutable_kwargs["episode_id_override"] = cluster_key
        return original_run_episode(*call_args, **mutable_kwargs)

    eval_module.run_episode = run_episode_with_fresh_identity
    try:
        episodes, successes, exit_sum, exit_count = run_task(
            cfg=cfg,
            task_suite=suite,
            task_id=args.task_id,
            model=model,
            exit_controller=exit_controller,
            device=device,
            num_tasks=1,
            resize_size=resize_size,
            total_episodes=0,
            total_successes=0,
            log_file=log_file,
            total_exit_mean_sum=0.0,
            total_exit_mean_count=0,
            telemetry_logger=telemetry,
            vision_teacher_cache_writer=teacher,
            phase_depth_runtime=None,
            phase_depth_control_enabled=False,
        )
    finally:
        eval_module.run_episode = original_run_episode
        telemetry.close()
        teacher.close()
        log_file.close()

    telemetry_records = _jsonl(telemetry_path)
    manifest_path = teacher_cache_dir / "manifest.jsonl"
    manifest_records = _jsonl(manifest_path)
    telemetry_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in telemetry_records
    }
    teacher_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in manifest_records
    }
    expected_ids = {record.cluster_key for record in task_schedule}
    observed_ids = {str(record["episode_id"]) for record in manifest_records}
    shards = [teacher_cache_dir / str(record["array_path"]) for record in manifest_records]
    latencies = [float(record["latency_ms"]) for record in telemetry_records]
    source_status = git_output("status", "--porcelain=v1")
    checks = {
        "D8_readiness_and_bound_artifacts_current": bool(prerequisites),
        "runtime_checkpoint_attested_and_sidecars_local": bool(model_audit),
        "gpu_is_one_verified_physical_front_four_device": True,
        "all_20_fresh_replicates_completed": episodes == len(task_schedule),
        "fresh_cluster_identity_exact": observed_ids == expected_ids,
        "telemetry_and_teacher_keys_align": telemetry_by_key.keys()
        == teacher_by_key.keys(),
        "exit_and_fm_metadata_align": telemetry_by_key.keys()
        == teacher_by_key.keys()
        and all(
            int(teacher_by_key[key]["teacher_exit_layer"])
            == int(telemetry_by_key[key]["exit_layer"])
            and int(teacher_by_key[key]["fm_calls"])
            == int(telemetry_by_key[key]["fm_calls"])
            for key in teacher_by_key
        ),
        "raw_cache_schema_and_fm_trace_complete": bool(manifest_records)
        and all(
            record["schema_version"] == VISION_TEACHER_CACHE_SCHEMA_VERSION
            and record["checkpoint_sha256"] == D2_CHECKPOINT_SHA256
            and int(record["source_projected_tokens"]) == 576
            and int(record["unique_visual_slots"]) == 288
            and int(record["valid_crop_count"]) == 4
            and has_complete_candidate_fm_traces(record)
            for record in manifest_records
        ),
        "all_npz_shards_present": bool(shards) and all(path.is_file() for path in shards),
        "observer_writes_error_free": telemetry.error_count == 0
        and teacher.error_count == 0,
        "official_episode_40_49_absent": all(
            ":episode" not in str(record["episode_id"])
            for record in telemetry_records + manifest_records
        ),
        "D7_shadow_only_and_not_active_control": True,
        "source_worktree_clean": not bool(source_status),
    }
    passed = all(checks.values())
    result = {
        "status": "PASS_V3_D8C_RAW_TASK" if passed else "FAIL_V3_D8C_RAW_TASK",
        "schema_version": D8C_RAW_TASK_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "task_id": args.task_id,
        "replicate_ids": [record.replicate_id for record in task_schedule],
        "cluster_keys": [record.cluster_key for record in task_schedule],
        "state_seeds": {
            str(record.replicate_id): record.state_seed for record in task_schedule
        },
        "policy_seeds": {
            str(record.replicate_id): record.policy_seed for record in task_schedule
        },
        "initial_state_sha256": initial_state_sha256,
        "completed_clusters": episodes,
        "behavior_successes": successes,
        "behavior_success_rate": successes / episodes if episodes else 0.0,
        "policy_calls": len(telemetry_records),
        "teacher_cache_calls": len(manifest_records),
        "mean_exit_ratio": exit_sum / exit_count if exit_count else None,
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "cache_bytes": sum(path.stat().st_size for path in shards if path.is_file()),
        "prerequisites": prerequisites,
        "model_audit": model_audit,
        "gpu_audit": {
            "physical_gpu_index": args.physical_gpu_index,
            "expected_gpu_uuid": args.expected_gpu_uuid,
            "visible_gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "visible_gpu_count": torch.cuda.device_count(),
        },
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_git_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "source_worktree_dirty": bool(source_status),
        "telemetry_sha256": stream_sha256(telemetry_path),
        "teacher_manifest_sha256": stream_sha256(manifest_path),
        "eval_log": str(Path(eval_log_path).relative_to(incomplete)),
        "checks": checks,
        "access_ledger": {
            "fresh_generated_state_clusters_opened": 20,
            "fresh_policy_rollouts": 20,
            "behavior_policy": "frozen_original_A1_early_exit_controller",
            "final_router_loaded": False,
            "D7_active_control_calls": 0,
            "official_episode_40_49_opened": False,
            "calibration_or_test_payload_opened": False,
        },
        "claim_boundary": {
            "behavior_success_is_descriptive_only": True,
            "generated_states_are_official_benchmark_states": False,
            "candidate_truth_computed": False,
            "confirmation_gate_inspected": False,
            "active_control": False,
            "superiority_claim_authorized": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("D8C raw task failed one or more gates")
    incomplete.rename(output)
    print("PASS_V3_D8C_RAW_TASK", flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        incomplete = args.output_dir.resolve().with_name(args.output_dir.name + ".incomplete")
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_V3_D8C_RAW_TASK",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
