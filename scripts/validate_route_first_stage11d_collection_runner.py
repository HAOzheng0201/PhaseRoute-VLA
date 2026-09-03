#!/usr/bin/env python3
"""Publish readiness for Stage-11D original-A1 development collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_reliability import (  # noqa: E402
    STAGE11D_PROTOCOL_SHA256,
    validate_stage11d_protocol,
)
from a1.vla.dynamic_compute.route_first_reliability_artifacts import (  # noqa: E402
    sha256_file,
)
from a1.vla.dynamic_compute.route_first_reliability_collection import (  # noqa: E402
    STAGE11D_A1_ACTION_DELTA_SHA256,
    STAGE11D_A1_CHECKPOINT_BYTES,
    STAGE11D_A1_CHECKPOINT_SHA256,
    STAGE11D_A1_CONFIG_SHA256,
    STAGE11D_A1_DATASET_STATISTICS_SHA256,
    STAGE11D_A1_THRESHOLDS_SHA256,
    STAGE11D_COLLECTION_CLUSTER_COUNT,
    STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH,
    STAGE11D_COLLECTION_READINESS_SCHEMA,
    STAGE11D_COLLECTION_READINESS_STATUS,
    Stage11DDevelopmentTaskSuite,
    development_schedule,
    load_development_states,
    task_development_schedule,
)
from a1.vla.dynamic_compute.route_first_reliability_state_binding import (  # noqa: E402
    STAGE11D_STATE_BINDING_SHA256,
    load_stage11d_state_binding,
    validate_local_stage11d_state_artifacts,
)


RUNNER_FILES = (
    "a1/vla/dynamic_compute/route_first_reliability.py",
    "a1/vla/dynamic_compute/route_first_reliability_artifacts.py",
    "a1/vla/dynamic_compute/route_first_reliability_state_binding.py",
    "a1/vla/dynamic_compute/route_first_reliability_collection.py",
    "configs/research/route_first_stage11d_reliability_protocol.json",
    "configs/research/route_first_stage11d_fresh_state_binding.json",
    "results/route_first/route_first_stage11d_fresh_states.json",
    "scripts/dynamic_compute/route_first_stage11d/collect_original_a1_task.py",
    "scripts/dynamic_compute/route_first_stage11d/launch_original_a1_development.py",
    "scripts/validate_route_first_stage11d_collection_runner.py",
    "tests/dynamic_compute/test_route_first_reliability_collection.py",
)
PROTECTED_FILES = {
    "a1/vla/value_net.py": (
        "ec3a860427f32d5837e279eb17eeb28befaee9dd7944d46482173c85e8847dc1"
    ),
    "robot_experiments/libero/exit_vla_utils.py": (
        "e5c88b72199c1354fc7b3f2fa22e056b593ee5cdadf7185cc7d1c09fe768051a"
    ),
    "robot_experiments/libero/eval_libero_early_exit.py": (
        "a4e3b1b49cdaf2021b3cd370d8a1e89c927906e7cbd5f8afdccd5ceb5b1826cd"
    ),
    "robot_experiments/libero/stage1_vla_utils.py": (
        "b45cad88585611a16cc92229becdfd6a2466e3fb0f859302846ae29dbff815a4"
    ),
    "a1/vla/dynamic_compute/vision_teacher_cache.py": (
        "3e9b0d4f0d55539dfc13ccbbabf51ee09fcfa66873a126221338d37b9e118d7f"
    ),
}
SMALL_CHECKPOINT_FILES = {
    "config.yaml": STAGE11D_A1_CONFIG_SHA256,
    "exit_thresholds_libero_10_exp_1.0.json": STAGE11D_A1_THRESHOLDS_SHA256,
    "dataset_statistics.json": STAGE11D_A1_DATASET_STATISTICS_SHA256,
    "exit_action_delta_matrix_libero_10_fm_steps10.json": (
        STAGE11D_A1_ACTION_DELTA_SHA256
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_once(path: Path, payload: dict[str, object]) -> None:
    target = path.expanduser().resolve()
    temporary = target.with_name(target.name + ".incomplete")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _checkpoint_inventory() -> dict[str, object]:
    directory = REPO_ROOT / "model/libero_exit"
    model_path = directory / "model.pt"
    metadata = model_path.stat()
    if (
        model_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != STAGE11D_A1_CHECKPOINT_BYTES
    ):
        raise RuntimeError("Stage-11D A1 model file type or size differs")
    print("Hashing the 33.8 GB A1 checkpoint once for readiness...", flush=True)
    model_sha = sha256_file(model_path)
    if model_sha != STAGE11D_A1_CHECKPOINT_SHA256:
        raise RuntimeError("Stage-11D A1 model SHA-256 differs")
    result: dict[str, object] = {
        "model.pt": {
            "path": "model/libero_exit/model.pt",
            "bytes": metadata.st_size,
            "sha256": model_sha,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        }
    }
    for name, expected_sha in SMALL_CHECKPOINT_FILES.items():
        path = directory / name
        observed_sha = sha256_file(path)
        if path.is_symlink() or observed_sha != expected_sha:
            raise RuntimeError(f"Stage-11D A1 checkpoint sidecar differs: {name}")
        result[name] = {
            "path": f"model/libero_exit/{name}",
            "bytes": path.stat().st_size,
            "sha256": observed_sha,
        }
    return result


def main() -> None:
    args = parse_args()
    protocol = validate_stage11d_protocol(REPO_ROOT)
    state_binding = load_stage11d_state_binding(REPO_ROOT)
    local_state = validate_local_stage11d_state_artifacts(REPO_ROOT)
    schedule, states, _attestation = load_development_states(REPO_ROOT)
    development = development_schedule(schedule)
    suite = Stage11DDevelopmentTaskSuite(object(), states)
    task_schedules = {
        str(task_id): task_development_schedule(development, task_id)
        for task_id in range(10)
    }
    sources = {
        relative: (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in RUNNER_FILES
        if relative.endswith(".py")
    }
    worker = sources[
        "scripts/dynamic_compute/route_first_stage11d/collect_original_a1_task.py"
    ]
    launcher = sources[
        "scripts/dynamic_compute/route_first_stage11d/launch_original_a1_development.py"
    ]
    protected_hashes = {
        relative: sha256_file(REPO_ROOT / relative) for relative in PROTECTED_FILES
    }
    output_root = REPO_ROOT / STAGE11D_COLLECTION_OUTPUT_RELATIVE_PATH
    launch_logs = REPO_ROOT / "runs/route_first_stage11d_development_launch_logs"
    checks = {
        "protocol_and_state_binding_exact": protocol["protocol_sha256"]
        == STAGE11D_PROTOCOL_SHA256
        and state_binding["local_state_payload"]["records"] == 200
        and local_state["attestation"]["audit"]["byte_identical_records"] == 200,
        "development_schedule_is_exact_120": len(development)
        == STAGE11D_COLLECTION_CLUSTER_COUNT
        and all(record.split == "development_train" for record in development),
        "each_task_exposes_only_replicates_0_to_11": all(
            tuple(record.replicate_id for record in task_schedules[str(task_id)])
            == tuple(range(12))
            and len(suite.get_task_init_states(task_id)) == 12
            for task_id in range(10)
        ),
        "original_A1_is_the_only_behavior_controller": "make_exit_controller" in worker
        and "load_route_first_active_runtime" not in worker
        and "load_frozen_phase_route_runtime" not in worker,
        "observer_is_noncontrolling": "vision_teacher_cache_writer=observer" in worker
        and "phase_depth_runtime=None" in worker
        and "phase_route_runtime=None" in worker
        and "phase_route_v3_enabled=False" in worker
        and "vision_aggregation_enabled=False" in worker,
        "cpu_gpu_and_model_load_gates_are_separate": all(
            value in worker
            for value in (
                "--cpu-preflight-only",
                "--gpu-preflight-only",
                "--model-load-smoke",
            )
        ),
        "idle_GPU_selection_and_one_process_one_GPU": "_select_gpus" in launcher
        and '"CUDA_VISIBLE_DEVICES": str(gpu["index"])' in launcher
        and "minimum_free_memory_mib" in launcher,
        "no_outcome_based_retry_path": launcher.count("subprocess.Popen(") == 1
        and "no retry allowed" in launcher,
        "collection_outputs_absent": not output_root.exists()
        and not launch_logs.exists(),
        "protected_A1_files_exact": protected_hashes == PROTECTED_FILES,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Stage-11D collection readiness failed: " + ", ".join(failed))
    checkpoint_inventory = _checkpoint_inventory()
    payload: dict[str, object] = {
        "schema_version": STAGE11D_COLLECTION_READINESS_SCHEMA,
        "status": STAGE11D_COLLECTION_READINESS_STATUS,
        "protocol_sha256": STAGE11D_PROTOCOL_SHA256,
        "state_binding_sha256": STAGE11D_STATE_BINDING_SHA256,
        "state_payload_sha256": state_binding["local_state_payload"]["sha256"],
        "runner_files": {
            relative: sha256_file(REPO_ROOT / relative) for relative in RUNNER_FILES
        },
        "protected_files": protected_hashes,
        "checkpoint_inventory": checkpoint_inventory,
        "schedule": {
            "development_clusters": len(development),
            "tasks": 10,
            "replicates_per_task": 12,
            "replicate_ids": list(range(12)),
            "policy_seed_formula": "94260830 + task_id * 10000 + replicate_id",
        },
        "checks": checks,
        "access_boundary": {
            "development_train": True,
            "calibration": False,
            "shadow_confirmation": False,
            "Stage12_reserved_states": False,
        },
        "execution": {
            "GPU_queried_or_initialized": False,
            "A1_model_loaded": False,
            "LIBERO_environment_opened": False,
            "development_collection_started": False,
            "same_noise_replay_started": False,
            "training_started": False,
            "active_new_router_control_started": False,
        },
        "authorization": {
            "original_A1_observation_collection": True,
            "same_noise_replay": False,
            "training": False,
            "active_control": False,
            "requires_clean_committed_runner_and_live_GPU_preflight": True,
        },
        "next_stage": "COMMIT_RUNNER_THEN_RUN_CPU_AND_LIVE_GPU_PREFLIGHTS",
    }
    _write_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
