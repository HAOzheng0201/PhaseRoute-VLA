#!/usr/bin/env python3
"""Run one frozen Route-first Stage-11B development profiling shard."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._route_first_stage10_contracts import ACTIVE  # noqa: E402


PROTOCOL_SCHEMA = "phase-route-vla.route-first-stage11b-profile-protocol.v1"
PROTOCOL_RELATIVE_PATH = Path(
    "configs/research/route_first_stage11b_profile_protocol.json"
)
STAGE10_READINESS_SHA256 = (
    "a4cd7726a03be9705fe8410ca1df92657dac281cfc1180c3e3edb661552b95d7"
)
STAGE10_READINESS_PATH = Path(
    "results/route_first/route_first_stage10_runner_readiness.json"
)
PROFILE_SCHEMA = "phase-route-vla.route-first-stage11b-profile-result.v1"
THRESHOLD_SHA256 = (
    "a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        output.append(value)
    if not output:
        raise ValueError(f"JSONL file is empty: {path}")
    return tuple(output)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
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


def _parse_task_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("task ids must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or any(item < 0 or item > 9 for item in result):
        raise argparse.ArgumentTypeError("task ids must be unique values in 0..9")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-stage", choices=("smoke", "shard0", "shard1", "shard2", "shard3"), required=True)
    parser.add_argument("--task-ids", type=_parse_task_ids, required=True)
    parser.add_argument("--physical-gpu-index", type=int, choices=range(8), required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "model/libero_exit")
    parser.add_argument(
        "--calibrated-router",
        type=Path,
        default=REPO_ROOT / "runs/route_first_calibration_stage6/router_calibrated.npz",
    )
    parser.add_argument(
        "--stage7-holdout",
        type=Path,
        default=REPO_ROOT / "results/route_first/route_first_stage7_holdout.json",
    )
    parser.add_argument(
        "--context-router",
        type=Path,
        default=REPO_ROOT / "artifacts/phase_route_v3/final_router.pt",
    )
    parser.add_argument(
        "--phase-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts/phase_route_v3/phase_estimator.pt",
    )
    return parser.parse_args()


def _normalize_uuid(value: Any) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def _thresholds(checkpoint: Path) -> dict[int, float]:
    path = checkpoint / "exit_thresholds_libero_10_exp_1.0.json"
    if sha256_file(path) != THRESHOLD_SHA256:
        raise PermissionError("frozen A1 threshold SHA-256 differs")
    value = _load_json(path)
    return {int(layer): float(threshold) for layer, threshold in value.items()}


def _stage11_sparse_controller(cfg: Any, model: Any, device: Any) -> Any:
    """Reproduce the Stage-10 sparse constructor without its CUDA-masking import."""

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


def _gpu_processes(expected_uuid: str) -> tuple[dict[str, Any], ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query failed: {completed.stderr.strip()}")
    output = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or _normalize_uuid(fields[1]) != _normalize_uuid(expected_uuid):
            continue
        output.append(
            {"pid": int(fields[0]), "gpu_uuid": fields[1], "used_memory_mib": int(fields[2])}
        )
    return tuple(output)


class GPUProcessMonitor:
    def __init__(self, expected_uuid: str, interval_seconds: float = 1.0) -> None:
        self.expected_uuid = expected_uuid
        self.interval_seconds = interval_seconds
        self.foreign: dict[int, dict[str, Any]] = {}
        self.samples = 0
        self.query_errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for record in _gpu_processes(self.expected_uuid):
                    if record["pid"] != os.getpid():
                        self.foreign[int(record["pid"])] = dict(record)
                self.samples += 1
            except Exception as error:
                self.query_errors.append(f"{type(error).__name__}: {error}")
            self._stop.wait(self.interval_seconds)

    def stop(self) -> Mapping[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 3.0))
        if self._thread.is_alive():
            self.query_errors.append("monitor thread did not stop")
        return {
            "interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "foreign_processes": [self.foreign[pid] for pid in sorted(self.foreign)],
            "query_errors": self.query_errors,
            "clean": not self.foreign and not self.query_errors,
        }


def _validate_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / PROTOCOL_RELATIVE_PATH
    protocol = _load_json(path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_DEVELOPMENT_PROFILE_NOT_RUN"
    ):
        raise ValueError("Stage-11B protocol contract differs")
    schedule = protocol.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("Stage-11B schedule is missing")
    if args.profile_stage == "smoke":
        expected = tuple(schedule.get("smoke", {}).get("task_ids", ()))
    else:
        expected = tuple(schedule.get("full_shards", {}).get(args.profile_stage, ()))
    if args.task_ids != expected:
        raise ValueError(
            f"task selection {args.task_ids} differs from frozen {args.profile_stage} {expected}"
        )
    return protocol, sha256_file(path)


def _source_and_artifact_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip()
    if worktree:
        raise PermissionError("Stage-11B profile requires a clean worktree")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    protected = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in ACTIVE.PROTECTED_CODE_SHA256
    }
    if protected != dict(ACTIVE.PROTECTED_CODE_SHA256):
        raise PermissionError("D9 protected source SHA differs")
    readiness_path = REPO_ROOT / STAGE10_READINESS_PATH
    if sha256_file(readiness_path) != STAGE10_READINESS_SHA256:
        raise PermissionError("Stage-10 readiness evidence SHA differs")
    readiness = _load_json(readiness_path)
    model_binding = readiness.get("bound_artifacts", {}).get("model/libero_exit/model.pt")
    model_path = args.checkpoint.resolve(strict=True) / "model.pt"
    stat = model_path.stat()
    if not isinstance(model_binding, Mapping) or (
        stat.st_size != model_binding.get("bytes")
        or stat.st_mtime_ns != model_binding.get("mtime_ns")
        or stat.st_ino != model_binding.get("inode")
    ):
        raise PermissionError("A1 model file identity differs from Stage-10 binding")
    return {
        "source_git_commit": source_commit,
        "source_worktree_dirty": False,
        "protected_code_sha256": protected,
        "stage10_readiness_path": str(STAGE10_READINESS_PATH),
        "stage10_readiness_sha256": STAGE10_READINESS_SHA256,
        "model_binding": dict(model_binding),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    protocol, protocol_sha = _validate_protocol(args)
    source = _source_and_artifact_preflight(args)
    visible_text = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_text not in (str(args.physical_gpu_index), args.expected_gpu_uuid):
        raise PermissionError("CUDA_VISIBLE_DEVICES differs from requested physical GPU")
    preflight_processes = _gpu_processes(args.expected_gpu_uuid)
    if preflight_processes:
        raise PermissionError(f"requested GPU already has compute processes: {preflight_processes}")
    output.mkdir(parents=True)
    # Sampling starts before CUDA initialization/model loading so transient
    # overlap cannot hide between endpoint-only checks.
    monitor = GPUProcessMonitor(args.expected_gpu_uuid)
    monitor.start()

    # Heavy imports happen only after the CPU-only evidence checks above.
    import numpy as np
    import torch
    from libero.libero import benchmark

    from a1.vla.dynamic_compute.route_first_controller import RouteFirstExitController
    from a1.vla.dynamic_compute.route_first_runtime import load_route_first_active_runtime
    from a1.vla.dynamic_compute.stage1_measurement import summarize_stage1_records
    from a1.vla.dynamic_compute.stage11_compute_measurement import (
        summarize_stage11_compute_records,
    )
    from a1.vla.dynamic_compute.telemetry import SafeJSONLTelemetryLogger
    from a1.vla.dynamic_compute.v3.release import summarize_runtime_records
    import robot_experiments.libero.eval_libero_early_exit as frozen_evaluator
    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
        run_episode,
        validate_config,
    )
    from robot_experiments.libero.libero_utils import get_libero_env
    from robot_experiments.libero.stage1_vla_utils import STAGE1_TIMING_ENV
    from robot_experiments.libero.stage11_vla_utils import (
        STAGE11_COMPUTE_ENV,
        get_vla_action as get_stage11_vla_action,
    )
    from robot_experiments.robot_utils import set_seed_everywhere
    from scripts.run_route_first_active import summarize_route_first_integrity

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage-11B requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if _normalize_uuid(properties.uuid) != _normalize_uuid(args.expected_gpu_uuid):
        raise RuntimeError("visible CUDA UUID differs")

    telemetry_path = output / "policy_telemetry.jsonl"
    runtime_path = output / "phase_route_runtime.jsonl"
    stage1_path = output / "stage1_measurement.jsonl"
    compute_path = output / "stage11_compute_measurement.jsonl"
    result_path = output / "result.json"
    episode_log_dir = output / "episode_logs"
    episode_log_dir.mkdir()
    os.environ[STAGE1_TIMING_ENV] = str(stage1_path)
    os.environ[STAGE11_COMPUTE_ENV] = str(compute_path)
    frozen_evaluator.get_vla_action = get_stage11_vla_action

    seed_base = int(protocol["schedule"]["seed_base"])
    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
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
        reseed_each_episode=True,
        seed=seed_base,
        run_id_note="route-first-stage11b-profile",
        vision_aggregation_enabled=False,
        learned_vision_aggregation_checkpoint=None,
        phase_depth_enabled=False,
        rp_pep_enabled=False,
        phase_route_v3_enabled=False,
    )
    validate_config(cfg)
    set_seed_everywhere(seed_base)
    print(
        f"[Stage11B] loading A1 on physical GPU {args.physical_gpu_index} "
        f"for {args.profile_stage}",
        flush=True,
    )
    model, device, _ = initialize_and_load_model(cfg)
    runtime = load_route_first_active_runtime(
        args.calibrated_router.resolve(strict=True),
        args.stage7_holdout.resolve(strict=True),
        args.context_router.resolve(strict=True),
        args.phase_checkpoint.resolve(strict=True),
    )
    base_controller = _stage11_sparse_controller(cfg, model, device)
    controller = RouteFirstExitController.from_frozen_sparse_controller(base_controller)
    controller.install_route_first_adapter(runtime.adapter)
    controller.to(device)
    controller.eval()

    telemetry = SafeJSONLTelemetryLogger(telemetry_path, flush_every=1)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    episodes = []
    started = time.perf_counter()
    try:
        for task_id in args.task_ids:
            episode_index = 0
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            environment, task_description = get_libero_env(
                task, cfg.model_family, resolution=cfg.env_img_res
            )
            try:
                episode_seed = seed_base + task_id * 10_000 + episode_index
                set_seed_everywhere(episode_seed)
                calls_before = runtime.policy_calls
                episode_started = time.perf_counter()
                log_path = episode_log_dir / f"task{task_id}_state0.log"
                print(f"[Stage11B] start task={task_id} state=0", flush=True)
                with log_path.open("x", encoding="utf-8") as episode_log:
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
                selected = {
                    f"L{layer}": sum(
                        record.get("selected_layer") == layer
                        for record in runtime.records[calls_before:]
                    )
                    for layer in (13, 27)
                }
                episodes.append(
                    {
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "success": bool(success),
                        "policy_calls": calls,
                        "selected_layer_counts": selected,
                        "wall_seconds": time.perf_counter() - episode_started,
                    }
                )
                print(
                    f"[Stage11B] complete task={task_id} success={success} "
                    f"calls={calls} selected={selected}",
                    flush=True,
                )
            finally:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()
    finally:
        telemetry.close()
        gpu_monitor = monitor.stop()

    if telemetry.error_count:
        raise RuntimeError(f"telemetry write failed: {telemetry.last_error}")
    stage1_records = _load_jsonl(stage1_path)
    compute_records = _load_jsonl(compute_path)
    runtime_records = tuple(runtime.records)
    stage1_summary = summarize_stage1_records(stage1_records)
    compute_summary = summarize_stage11_compute_records(compute_records)
    runtime_summary = summarize_runtime_records(runtime_records)
    route_integrity = summarize_route_first_integrity(runtime_records)
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
    _write_jsonl(runtime_path, runtime_records)
    policy_calls = runtime.policy_calls
    gates = {
        "runtime_complete": bool(
            runtime.error_count == 0
            and policy_calls == runtime.prepared_calls == runtime.committed_calls
        ),
        "exactly_one_authoritative_fm": (
            route_integrity["valid_calls_with_exactly_one_fm"] == policy_calls
        ),
        "stage1_measurement_complete": bool(
            stage1_summary["records"] == policy_calls
            and stage1_summary["records_with_errors"] == 0
            and stage1_summary["records_with_nonfinite_actions"] == 0
        ),
        "stage11_compute_complete": bool(
            compute_summary["records"] == policy_calls
            and compute_summary["valid_records"] == policy_calls
        ),
        "gpu_sampling_monitor_clean": bool(gpu_monitor["clean"]),
    }
    result = {
        "schema_version": PROFILE_SCHEMA,
        "status": (
            "COMPLETE_STAGE11B_DEVELOPMENT_PROFILE"
            if all(gates.values())
            else "INVALID_STAGE11B_DEVELOPMENT_PROFILE"
        ),
        "profile_stage": args.profile_stage,
        "task_ids": list(args.task_ids),
        "episode_indices": [0],
        "protocol_path": str(PROTOCOL_RELATIVE_PATH),
        "protocol_sha256": protocol_sha,
        "source": source,
        "gpu": {
            "physical_index": args.physical_gpu_index,
            "uuid": "GPU-" + str(properties.uuid).removeprefix("GPU-"),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "preflight_processes": list(preflight_processes),
            "sampling_monitor": gpu_monitor,
        },
        "episodes": episodes,
        "successes_descriptive": sum(int(item["success"]) for item in episodes),
        "policy_calls": policy_calls,
        "runtime": runtime_summary,
        "stage1_measurement": stage1_summary,
        "stage11_compute": compute_summary,
        "wall_seconds": time.perf_counter() - started,
        "gates": gates,
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_json(result_path, result)
    if not all(gates.values()):
        raise RuntimeError(f"Stage-11B profile gates failed: {gates}")
    print(
        f"[Stage11B] PASS stage={args.profile_stage} calls={policy_calls} "
        f"success={result['successes_descriptive']}/{len(episodes)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
