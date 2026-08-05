"""Profile one M4.23 policy on preregistered fixed observations.

The three policies are run in separate sessions so the measurement uses the
actual full-depth and early-exit model classes.  Selection and result helpers
are deliberately pure enough to audit without loading the 34 GB checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POLICIES = ("early_exit", "rp_pep", "full_depth")
EXPECTED_MODEL_CLASSES = {
    "early_exit": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
    "rp_pep": "a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
    "full_depth": "a1.vla.affordvla.AffordVLA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--order-position", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--exit-layer", type=int, action="append")
    parser.add_argument("--records-per-exit", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def canonical_record_identity(cache_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_dir": str(cache_dir.resolve()),
        "array_path": str(record["array_path"]),
        "episode_id": str(record["episode_id"]),
        "task_id": int(record["task_id"]),
        "step_id": int(record["step_id"]),
        "teacher_exit_layer": int(record["teacher_exit_layer"]),
    }


def record_key(item: tuple[Path, Mapping[str, Any]]) -> tuple[Any, ...]:
    cache_dir, record = item
    return (
        int(record["task_id"]),
        str(record["episode_id"]),
        int(record["step_id"]),
        str(cache_dir.resolve()),
        str(record["array_path"]),
    )


def select_stratified_entries(
    entries: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    exit_layers: Sequence[int],
    records_per_exit: int,
) -> list[tuple[Path, Mapping[str, Any]]]:
    """Freeze a deterministic, task-diverse set within each teacher exit layer."""

    if records_per_exit <= 0:
        raise ValueError("records_per_exit must be positive")
    layers = tuple(int(layer) for layer in exit_layers)
    if len(set(layers)) != len(layers):
        raise ValueError("exit_layers must be unique")
    by_layer: dict[int, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    identities: set[str] = set()
    for item in entries:
        identity = json.dumps(
            canonical_record_identity(*item), sort_keys=True, separators=(",", ":")
        )
        if identity in identities:
            raise ValueError(f"duplicate cache record: {identity}")
        identities.add(identity)
        by_layer[int(item[1]["teacher_exit_layer"])].append(item)

    selected: list[tuple[Path, Mapping[str, Any]]] = []
    for layer in layers:
        candidates = sorted(by_layer.get(layer, []), key=record_key)
        if len(candidates) < records_per_exit:
            raise ValueError(
                f"teacher exit layer {layer} has {len(candidates)} records; "
                f"need {records_per_exit}"
            )
        layer_selected: list[tuple[Path, Mapping[str, Any]]] = []
        seen_tasks: set[int] = set()
        for item in candidates:
            task_id = int(item[1]["task_id"])
            if task_id in seen_tasks:
                continue
            layer_selected.append(item)
            seen_tasks.add(task_id)
            if len(layer_selected) == records_per_exit:
                break
        if len(layer_selected) < records_per_exit:
            selected_ids = {id(item[1]) for item in layer_selected}
            for item in candidates:
                if id(item[1]) in selected_ids:
                    continue
                layer_selected.append(item)
                if len(layer_selected) == records_per_exit:
                    break
        selected.extend(layer_selected)
    return selected


def selection_sha256(entries: Sequence[tuple[Path, Mapping[str, Any]]]) -> str:
    payload = [canonical_record_identity(*item) for item in entries]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def summarize(values: Iterable[float]) -> dict[str, float]:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("cannot summarize an empty sequence")

    def percentile(fraction: float) -> float:
        position = (len(rows) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(rows) - 1)
        weight = position - lower
        return rows[lower] * (1.0 - weight) + rows[upper] * weight

    return {
        "count": len(rows),
        "total": float(sum(rows)),
        "mean": float(statistics.fmean(rows)),
        "median": float(statistics.median(rows)),
        "p95": float(percentile(0.95)),
        "min": rows[0],
        "max": rows[-1],
    }


class CudaComponentProfiler:
    """CUDA-event component timing installed only for the diagnostic pass."""

    def __init__(self, model: torch.nn.Module):
        if int(model.config.block_group_size) != 1:
            raise ValueError("M4.23 component profiler requires block_group_size=1")
        self.model = model
        self.active = False
        self.block_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.fm_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._block_starts: dict[int, list[torch.cuda.Event]] = defaultdict(list)
        self.handles = []
        for block in model.transformer.blocks:
            self.handles.append(block.register_forward_pre_hook(self._block_pre_hook))
            self.handles.append(block.register_forward_hook(self._block_post_hook))
        self.original_predict_vector_field = model.action_head.predict_vector_field

        def profiled_predict_vector_field(*args: Any, **kwargs: Any):
            if not self.active:
                return self.original_predict_vector_field(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = self.original_predict_vector_field(*args, **kwargs)
            end.record()
            self.fm_pairs.append((start, end))
            return output

        model.action_head.predict_vector_field = profiled_predict_vector_field

    def _block_pre_hook(self, module: torch.nn.Module, _inputs: Any) -> None:
        if self.active:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._block_starts[id(module)].append(start)

    def _block_post_hook(self, module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        if self.active:
            starts = self._block_starts[id(module)]
            if not starts:
                raise RuntimeError("missing transformer block start event")
            start = starts.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.block_pairs.append((start, end))

    def start(self) -> None:
        if self.active:
            raise RuntimeError("component profiler is already active")
        self.block_pairs = []
        self.fm_pairs = []
        self._block_starts.clear()
        self.active = True

    def finish(self) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("component profiler is not active")
        self.active = False
        if any(self._block_starts.values()):
            raise RuntimeError("unbalanced transformer hooks")
        return {
            "transformer_ms": float(
                sum(start.elapsed_time(end) for start, end in self.block_pairs)
            ),
            "fm_head_ms": float(
                sum(start.elapsed_time(end) for start, end in self.fm_pairs)
            ),
            "transformer_block_calls": len(self.block_pairs),
            "fm_vector_field_calls": len(self.fm_pairs),
        }

    def close(self) -> None:
        self.active = False
        self.model.action_head.predict_vector_field = self.original_predict_vector_field
        for handle in self.handles:
            handle.remove()
        self.handles = []


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_policy(args: argparse.Namespace):
    from robot_experiments.robot_utils import set_seed_everywhere

    checkpoint = args.checkpoint.resolve()
    common = dict(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_spatial",
        action_head_flow_matching_inference_steps=args.fm_steps,
        use_wandb=False,
        save_rollout_video=False,
        seed=args.seed,
    )
    set_seed_everywhere(args.seed)
    if args.policy == "full_depth":
        import robot_experiments.libero.eval_libero as evaluation

        cfg = evaluation.GenerateConfig(**common)
        model, device = evaluation.initialize_and_load_model(cfg)
        return cfg, model, device, None, None

    import robot_experiments.libero.eval_libero_early_exit as evaluation
    from scripts.dynamic_compute.replay_m420b_rp_pep import _make_controllers

    cfg = evaluation.GenerateConfig(
        **common,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
    )
    model, device, _ = evaluation.initialize_and_load_model(cfg)
    baseline, rp_pep, threshold_path, plan = _make_controllers(cfg, model, device)
    controller = baseline if args.policy == "early_exit" else rp_pep
    return cfg, model, device, controller, (threshold_path, plan)


def full_depth_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(inputs)
    projected = converted.pop("precomputed_projected_features")
    image_idx = converted["image_input_idx"]
    if projected.ndim != 4 or image_idx.ndim != 3:
        raise ValueError("cached vision tensors do not have [B,C,M,*] shapes")
    converted["pre_extracted_image_features"] = projected.flatten(1, 2)
    converted["image_input_idx"] = image_idx.flatten(1, 2)
    return converted


def execute_policy(
    *,
    policy: str,
    model: torch.nn.Module,
    controller: Any,
    inputs: Mapping[str, Any],
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
    device: torch.device,
    timestep: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device=device)
    telemetry_events: list[dict[str, Any]] = []
    if controller is not None:
        controller.value_net.reset_actions()
        controller.set_timestep(timestep)

    def telemetry_callback(event_type: str, payload: Mapping[str, Any]) -> None:
        if event_type == "exit_candidate":
            telemetry_events.append(dict(payload))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start_event.record()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        if policy == "full_depth":
            action = model.predict_actions(**full_depth_inputs(inputs))
            exit_layer = int(model.config.n_layers) - 1
        else:
            output = model.forward(
                **inputs,
                exit_controller=controller,
                telemetry_callback=telemetry_callback,
            )
            if output.exit_action is None:
                raise RuntimeError(f"{policy} returned no exit action")
            action = output.exit_action
            exit_layer = int(output.exit_layer)
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    action_cpu = action.detach().to(device="cpu", dtype=torch.float32)
    if policy == "full_depth":
        fm_calls = 1
        fm_steps = int(model.config.num_diffusion_inference_steps)
        rng_burns = 0
    else:
        fm_calls = sum(int(event.get("fm_calls", 0)) for event in telemetry_events)
        fm_steps = sum(int(event.get("fm_steps", 0)) for event in telemetry_events)
        rng_burns = sum(int(event.get("rng_burns", 0)) for event in telemetry_events)
    return {
        "exit_layer": exit_layer,
        "transformer_layers_executed": exit_layer + 1,
        "action_sha256": sha256_tensor(action_cpu),
        "action_finite": bool(torch.isfinite(action_cpu).all()),
        "action_shape": list(action_cpu.shape),
        "fm_calls": fm_calls,
        "fm_steps": fm_steps,
        "rng_burns": rng_burns,
        "cuda_latency_ms": cuda_ms,
        "wall_latency_ms": wall_ms,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.physical_gpu_index not in (0, 1, 2, 3):
        raise ValueError("M4.23 only permits physical GPUs 0-3")
    if args.fm_steps != 10:
        raise ValueError("M4.23 is frozen to FM10")
    if args.repeats <= 0 or args.warmup_calls < 0:
        raise ValueError("invalid repeats/warmup-calls")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M4.23 requires exactly one visible CUDA device")

    from scripts.dynamic_compute.replay_m420b_rp_pep import (
        _load_call_inputs,
        _load_manifest_entries,
        normalize_gpu_uuid,
    )

    device = torch.device("cuda:0")
    visible_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if normalize_gpu_uuid(visible_uuid) != normalize_gpu_uuid(args.expected_gpu_uuid):
        raise RuntimeError(
            f"GPU UUID mismatch: expected {args.expected_gpu_uuid}, visible {visible_uuid}"
        )

    cache_dirs = [path.resolve() for path in args.cache_dir]
    all_entries = _load_manifest_entries(cache_dirs, args.checkpoint_sha256)
    exit_layers = tuple(args.exit_layer or (11, 13, 27))
    entries = select_stratified_entries(
        all_entries,
        exit_layers=exit_layers,
        records_per_exit=args.records_per_exit,
    )
    selected_sha = selection_sha256(entries)
    selected_records = [canonical_record_identity(*item) for item in entries]

    cfg, model, device, controller, early_metadata = load_policy(args)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_class != EXPECTED_MODEL_CLASSES[args.policy]:
        raise RuntimeError(f"unexpected model class for {args.policy}: {model_class}")
    model_dtype = model.transformer.wte.embedding.dtype
    amp_enabled = not args.disable_amp

    first_inputs, first_cpu_rng, first_cuda_rng = _load_call_inputs(
        entries[0][0], entries[0][1], device=device, model_dtype=model_dtype
    )
    for _ in range(args.warmup_calls):
        execute_policy(
            policy=args.policy,
            model=model,
            controller=controller,
            inputs=first_inputs,
            cpu_rng_state=first_cpu_rng,
            cuda_rng_state=first_cuda_rng,
            device=device,
            timestep=int(entries[0][1]["step_id"]),
            amp_enabled=amp_enabled,
        )

    torch.cuda.reset_peak_memory_stats(device)
    timed_samples: list[dict[str, Any]] = []
    for record_index, (cache_dir, record) in enumerate(entries):
        inputs, cpu_rng_state, cuda_rng_state = _load_call_inputs(
            cache_dir, record, device=device, model_dtype=model_dtype
        )
        for repeat in range(args.repeats):
            random.seed(args.seed + record_index * 1000 + repeat)
            measurement = execute_policy(
                policy=args.policy,
                model=model,
                controller=controller,
                inputs=inputs,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
                device=device,
                timestep=int(record["step_id"]),
                amp_enabled=amp_enabled,
            )
            row = {
                **canonical_record_identity(cache_dir, record),
                "record_index": record_index,
                "repeat": repeat,
                **measurement,
            }
            timed_samples.append(row)
            print(
                f"timed policy={args.policy} record={record_index + 1:02d}/{len(entries):02d} "
                f"repeat={repeat + 1}/{args.repeats} teacher_exit={record['teacher_exit_layer']} "
                f"actual_exit={measurement['exit_layer']} fm={measurement['fm_calls']} "
                f"cuda_ms={measurement['cuda_latency_ms']:.2f}",
                flush=True,
            )
    timed_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    timed_peak_reserved = int(torch.cuda.max_memory_reserved(device))

    profiler = CudaComponentProfiler(model)
    component_samples: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for record_index, (cache_dir, record) in enumerate(entries):
            inputs, cpu_rng_state, cuda_rng_state = _load_call_inputs(
                cache_dir, record, device=device, model_dtype=model_dtype
            )
            random.seed(args.seed + record_index * 1000)
            profiler.start()
            measurement = execute_policy(
                policy=args.policy,
                model=model,
                controller=controller,
                inputs=inputs,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
                device=device,
                timestep=int(record["step_id"]),
                amp_enabled=amp_enabled,
            )
            components = profiler.finish()
            other_ms = (
                measurement["cuda_latency_ms"]
                - components["transformer_ms"]
                - components["fm_head_ms"]
            )
            row = {
                **canonical_record_identity(cache_dir, record),
                "record_index": record_index,
                **measurement,
                **components,
                "instrumented_other_ms": float(other_ms),
            }
            component_samples.append(row)
            print(
                f"component policy={args.policy} record={record_index + 1:02d}/{len(entries):02d} "
                f"total={measurement['cuda_latency_ms']:.2f} "
                f"transformer={components['transformer_ms']:.2f} "
                f"fm_head={components['fm_head_ms']:.2f} other={other_ms:.2f}",
                flush=True,
            )
    finally:
        profiler.close()
    component_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    component_peak_reserved = int(torch.cuda.max_memory_reserved(device))

    expected_timed = len(entries) * args.repeats
    local_checks = {
        "timed_sample_count": len(timed_samples) == expected_timed,
        "component_sample_count": len(component_samples) == len(entries),
        "all_actions_finite": all(row["action_finite"] for row in timed_samples + component_samples),
        "full_depth_single_solve": (
            True
            if args.policy != "full_depth"
            else all(
                row["fm_calls"] == 1
                and row["fm_steps"] == args.fm_steps
                and row["fm_vector_field_calls"] == args.fm_steps
                for row in component_samples
            )
        ),
        "component_events_consistent": all(
            row["transformer_block_calls"] == row["transformer_layers_executed"]
            and row["fm_vector_field_calls"] == row["fm_steps"]
            for row in component_samples
        ),
        "component_other_nonnegative": all(
            row["instrumented_other_ms"] >= -1.0 for row in component_samples
        ),
    }
    status = "PASS" if all(local_checks.values()) else "FAIL"
    source_status = git_output("status", "--porcelain=v1")
    threshold_path = early_metadata[0] if early_metadata is not None else None
    plan = early_metadata[1] if early_metadata is not None else None
    result = {
        "status": status,
        "scope": "m423_fixed_observation_policy_profile",
        "policy": args.policy,
        "model_class": model_class,
        "checkpoint": str(args.checkpoint.resolve() / "model.pt"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "cache_dirs": [str(path) for path in cache_dirs],
        "cache_records_available": len(all_entries),
        "selected_records": selected_records,
        "selection_sha256": selected_sha,
        "exit_layers": list(exit_layers),
        "records_per_exit": args.records_per_exit,
        "records": len(entries),
        "repeats": args.repeats,
        "warmup_calls": args.warmup_calls,
        "fm_steps": args.fm_steps,
        "amp_enabled": amp_enabled,
        "seed": args.seed,
        "physical_gpu_index": args.physical_gpu_index,
        "order_position": args.order_position,
        "physical_gpu_uuid_nvidia_smi": args.expected_gpu_uuid,
        "physical_gpu_uuid_visible": visible_uuid,
        "gpu_name": torch.cuda.get_device_name(device),
        "threshold_path": str(threshold_path.resolve()) if threshold_path else None,
        "threshold_sha256": (
            hashlib.sha256(threshold_path.read_bytes()).hexdigest()
            if threshold_path
            else None
        ),
        "productive_exit_plan": (
            {
                "name": plan.name,
                "eligible_exit_layers": list(plan.eligible_exit_layers),
                "comparison_references": plan.comparison_references,
                "rng_burns": plan.rng_burns,
            }
            if args.policy == "rp_pep"
            else None
        ),
        "memory_bytes": {
            "timed_peak_allocated": timed_peak_allocated,
            "timed_peak_reserved": timed_peak_reserved,
            "component_peak_allocated": component_peak_allocated,
            "component_peak_reserved": component_peak_reserved,
        },
        "summary": {
            "timed_cuda_latency_ms": summarize(
                row["cuda_latency_ms"] for row in timed_samples
            ),
            "timed_wall_latency_ms": summarize(
                row["wall_latency_ms"] for row in timed_samples
            ),
            "fm_calls": summarize(row["fm_calls"] for row in timed_samples),
            "component_transformer_ms": summarize(
                row["transformer_ms"] for row in component_samples
            ),
            "component_fm_head_ms": summarize(
                row["fm_head_ms"] for row in component_samples
            ),
            "component_other_ms": summarize(
                row["instrumented_other_ms"] for row in component_samples
            ),
        },
        "local_checks": local_checks,
        "timed_samples": timed_samples,
        "component_samples": component_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, **result["summary"]}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
