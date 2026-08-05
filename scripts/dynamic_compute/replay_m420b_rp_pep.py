"""Paired fixed-observation gate for M4.20b RP-PEP.

Each cached call runs the original A1 controller first, restores the exact
CPU/CUDA RNG state, and then runs the sparse controller on the same projected
visual features.  This script intentionally loads the 34 GB model only once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_BASELINE_CALLS = {3: 3, 11: 7, 13: 8, 27: 15}
EXPECTED_RP_PEP_CALLS = {3: 2, 11: 4, 13: 5, 27: 7}
EXPECTED_RP_PEP_BURNS = {3: 1, 11: 3, 13: 3, 27: 8}


def normalize_gpu_uuid(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20264804)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty latency values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latencies(values: list[float]) -> dict[str, float]:
    return {
        "total_ms": float(sum(values)),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(_percentile(values, 0.95)),
    }


def evaluate_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("RP-PEP gate requires at least one row")
    baseline_cuda = [float(row["baseline"]["cuda_latency_ms"]) for row in rows]
    sparse_cuda = [float(row["rp_pep"]["cuda_latency_ms"]) for row in rows]
    baseline_wall = [float(row["baseline"]["wall_latency_ms"]) for row in rows]
    sparse_wall = [float(row["rp_pep"]["wall_latency_ms"]) for row in rows]
    baseline_calls = sum(int(row["baseline"]["fm_calls"]) for row in rows)
    sparse_calls = sum(int(row["rp_pep"]["fm_calls"]) for row in rows)
    baseline_cuda_summary = summarize_latencies(baseline_cuda)
    sparse_cuda_summary = summarize_latencies(sparse_cuda)
    baseline_wall_summary = summarize_latencies(baseline_wall)
    sparse_wall_summary = summarize_latencies(sparse_wall)
    fm_reduction = 1.0 - sparse_calls / baseline_calls
    cuda_reduction = 1.0 - (
        sparse_cuda_summary["total_ms"] / baseline_cuda_summary["total_ms"]
    )
    wall_reduction = 1.0 - (
        sparse_wall_summary["total_ms"] / baseline_wall_summary["total_ms"]
    )
    counters = {
        "exit_mismatches": sum(not bool(row["exit_match"]) for row in rows),
        "action_nonexact": sum(not bool(row["action_exact"]) for row in rows),
        "action_over_1e_6": sum(
            float(row["action_max_abs_error"]) > 1e-6 for row in rows
        ),
        "gripper_direction_mismatches": sum(
            int(row["gripper_direction_mismatches"]) for row in rows
        ),
        "nonfinite_rows": sum(not bool(row["finite"]) for row in rows),
        "fm_formula_mismatches": sum(
            not bool(row["fm_formula_match"]) for row in rows
        ),
        "trace_mismatches": sum(not bool(row["retained_trace_match"]) for row in rows),
        "threshold_event_mismatches": sum(
            not bool(row["threshold_event_match"]) for row in rows
        ),
        "telemetry_errors": sum(int(row.get("telemetry_errors", 0)) for row in rows),
    }
    strict_equivalence = all(value == 0 for value in counters.values())
    passed = strict_equivalence and fm_reduction >= 0.35 and cuda_reduction >= 0.15
    return {
        "status": "PASS" if passed else "FAIL",
        "calls": len(rows),
        "exact_action_calls": len(rows) - counters["action_nonexact"],
        "max_action_abs_error": max(float(row["action_max_abs_error"]) for row in rows),
        "counters": counters,
        "fm_solver_calls": {
            "baseline": baseline_calls,
            "rp_pep": sparse_calls,
            "reduction_fraction": fm_reduction,
        },
        "cuda_policy_latency": {
            "baseline": baseline_cuda_summary,
            "rp_pep": sparse_cuda_summary,
            "reduction_fraction": cuda_reduction,
        },
        "wall_policy_latency": {
            "baseline": baseline_wall_summary,
            "rp_pep": sparse_wall_summary,
            "reduction_fraction": wall_reduction,
        },
        "gates": {
            "strict_equivalence": strict_equivalence,
            "fm_reduction_at_least_35_percent": fm_reduction >= 0.35,
            "cuda_latency_reduction_at_least_15_percent": cuda_reduction >= 0.15,
        },
    }


def _optional(array: np.ndarray, *, device: torch.device, dtype: torch.dtype):
    if array.size == 0:
        return None
    return torch.from_numpy(array).to(device=device, dtype=dtype).unsqueeze(0)


def _load_call_inputs(
    cache_dir: Path,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    with np.load(cache_dir / str(record["array_path"])) as shard:
        arrays = {name: shard[name].copy() for name in shard.files}
    inputs = {
        "input_ids": torch.from_numpy(arrays["input_ids"]).to(
            device=device, dtype=torch.int64
        ).unsqueeze(0),
        "attention_mask": _optional(
            arrays["attention_mask"], device=device, dtype=torch.bool
        ),
        "attention_bias": _optional(
            arrays["attention_bias"], device=device, dtype=torch.float32
        ),
        "response_mask": _optional(
            arrays["response_mask"], device=device, dtype=torch.bool
        ),
        "image_input_idx": torch.from_numpy(arrays["image_input_idx"]).to(
            device=device, dtype=torch.int64
        ).unsqueeze(0),
        "subsegment_ids": _optional(
            arrays["subsegment_ids"], device=device, dtype=torch.int64
        ),
        "position_ids": _optional(
            arrays["position_ids"], device=device, dtype=torch.int64
        ),
        "action_proprio": torch.from_numpy(arrays["action_proprio"]).to(
            device=device, dtype=torch.float32
        ),
        "proprio_token_idx": torch.from_numpy(arrays["proprio_token_idx"]).to(
            device=device, dtype=torch.int64
        ),
        "output_hidden_states": False,
        "use_cache": True,
        "precomputed_projected_features": torch.from_numpy(
            arrays["projected_features"]
        ).to(device=device, dtype=model_dtype).unsqueeze(0),
    }
    return (
        inputs,
        torch.from_numpy(arrays["cpu_rng_state"]),
        torch.from_numpy(arrays["cuda_rng_state"]),
    )


def _make_controllers(cfg, model, device):
    from a1.vla.dynamic_compute.productive_exit import a1_fm10_rp_pep_plan
    from a1.vla.value_net import ActionValueNet, ExitController

    original = tuple(model.get_all_exit_idx(cfg.exit_interval))
    plan = a1_fm10_rp_pep_plan(original)
    threshold_path = Path(cfg.pretrained_checkpoint) / (
        f"exit_thresholds_{cfg.task_suite_name}_{cfg.exit_dist}_{cfg.exit_ratio}.json"
    )
    with threshold_path.open("r", encoding="utf-8") as input_file:
        thresholds = {
            int(layer): float(value) for layer, value in json.load(input_file).items()
        }
    selected = plan.select_eligible_thresholds(thresholds, lower_is_easier=True)

    def make(exit_layers, productive_plan=None):
        value_net = ActionValueNet(
            exit_list=list(exit_layers),
            exit_head=model.action_head,
            model=model,
            interval=cfg.exit_interval,
            threshold_type=cfg.threshold_type,
            anchor=False,
            productive_exit_plan=productive_plan,
        )
        controller = ExitController(
            value_net,
            exit_id_list=list(exit_layers),
            steps_per_stage=cfg.steps_per_stage,
            leq=True,
            exit_dist=cfg.exit_dist,
            max_layer=model.config.n_layers,
        )
        controller.to(device)
        controller.eval()
        return controller

    baseline = make(original)
    baseline.thresholds = dict(thresholds)
    sparse = make(plan.eligible_exit_layers, plan)
    sparse._set_threshold_value(selected)
    return baseline, sparse, threshold_path, plan


def _run_policy(
    model,
    controller,
    inputs: Mapping[str, Any],
    *,
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
    device: torch.device,
    timestep: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device=device)
    controller.set_timestep(timestep)
    telemetry_events: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    def telemetry_callback(event_type, payload):
        if event_type == "exit_candidate":
            telemetry_events.append(dict(payload))

    def trace_callback(payload):
        traces.append(dict(payload))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()
    start_event.record()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        output = model.forward(
            **inputs,
            exit_controller=controller,
            telemetry_callback=telemetry_callback,
            fm_trace_callback=trace_callback,
        )
    end_event.record()
    torch.cuda.synchronize(device)
    wall_latency_ms = (time.perf_counter() - wall_started) * 1000.0
    cuda_latency_ms = float(start_event.elapsed_time(end_event))
    action = output.exit_action.detach().to(device="cpu", dtype=torch.float32)
    trace_rows = []
    for trace in traces:
        trace_rows.append(
            {
                "layer": int(trace["candidate_layer"]),
                "role": str(trace["candidate_role"]),
                "input_x": trace["input_x"].detach().to(device="cpu", dtype=torch.float32),
                "output_action": trace["output_action"].detach().to(
                    device="cpu", dtype=torch.float32
                ),
            }
        )
    return {
        "exit_layer": int(output.exit_layer),
        "action": action,
        "action_sha256": _sha256_tensor(action),
        "fm_calls": sum(int(event.get("fm_calls", 0)) for event in telemetry_events),
        "fm_steps": sum(int(event.get("fm_steps", 0)) for event in telemetry_events),
        "rng_burns": sum(int(event.get("rng_burns", 0)) for event in telemetry_events),
        "cuda_latency_ms": cuda_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "events": telemetry_events,
        "traces": trace_rows,
    }


def _event_signature(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(event["layer_idx"]),
        bool(event["evaluated"]),
        bool(event["should_exit"]),
        event.get("threshold_passed"),
        bool(event.get("forced_by_max_layer", False)),
        event.get("action_delta"),
        event.get("threshold"),
        event.get("base_threshold"),
        bool(event.get("leq", True)),
    )


def _compare_pair(record, baseline, sparse) -> dict[str, Any]:
    baseline_action = baseline["action"]
    sparse_action = sparse["action"]
    error = (baseline_action - sparse_action).abs()
    exact = bool(torch.equal(baseline_action, sparse_action))
    finite = bool(torch.isfinite(baseline_action).all() and torch.isfinite(sparse_action).all())
    gripper_mismatches = 0
    if baseline_action.shape[-1] >= 7:
        gripper_mismatches = int(
            ((baseline_action[..., 6:] >= 0) != (sparse_action[..., 6:] >= 0))
            .sum()
            .item()
        )

    baseline_traces = {int(trace["layer"]): trace for trace in baseline["traces"]}
    retained_trace_match = True
    trace_error = 0.0
    for trace in sparse["traces"]:
        reference = baseline_traces.get(int(trace["layer"]))
        if reference is None:
            retained_trace_match = False
            continue
        for name in ("input_x", "output_action"):
            difference = (trace[name] - reference[name]).abs()
            trace_error = max(trace_error, float(difference.max().item()))
            retained_trace_match &= bool(torch.equal(trace[name], reference[name]))

    baseline_events = {
        int(event["layer_idx"]): event
        for event in baseline["events"]
        if bool(event.get("evaluated", False))
    }
    threshold_event_match = True
    for event in sparse["events"]:
        if not bool(event.get("evaluated", False)):
            continue
        reference = baseline_events.get(int(event["layer_idx"]))
        threshold_event_match &= reference is not None and (
            _event_signature(event) == _event_signature(reference)
        )

    exit_layer = int(baseline["exit_layer"])
    formula_match = (
        exit_layer in EXPECTED_BASELINE_CALLS
        and int(sparse["exit_layer"]) == exit_layer
        and int(baseline["fm_calls"]) == EXPECTED_BASELINE_CALLS[exit_layer]
        and int(sparse["fm_calls"]) == EXPECTED_RP_PEP_CALLS[exit_layer]
        and int(sparse["rng_burns"]) == EXPECTED_RP_PEP_BURNS[exit_layer]
        and len(baseline["traces"]) == EXPECTED_BASELINE_CALLS[exit_layer]
        and len(sparse["traces"]) == EXPECTED_RP_PEP_CALLS[exit_layer]
    )
    return {
        "task_id": int(record["task_id"]),
        "step_id": int(record["step_id"]),
        "episode_id": str(record["episode_id"]),
        "cache_teacher_exit_layer": int(record["teacher_exit_layer"]),
        "exit_match": exit_layer == int(sparse["exit_layer"]),
        "action_exact": exact,
        "action_max_abs_error": float(error.max().item()),
        "gripper_direction_mismatches": gripper_mismatches,
        "finite": finite,
        "fm_formula_match": formula_match,
        "retained_trace_match": retained_trace_match,
        "retained_trace_max_abs_error": trace_error,
        "threshold_event_match": threshold_event_match,
        "telemetry_errors": 0,
        "baseline": {
            key: baseline[key]
            for key in (
                "exit_layer",
                "action_sha256",
                "fm_calls",
                "fm_steps",
                "rng_burns",
                "cuda_latency_ms",
                "wall_latency_ms",
            )
        },
        "rp_pep": {
            key: sparse[key]
            for key in (
                "exit_layer",
                "action_sha256",
                "fm_calls",
                "fm_steps",
                "rng_burns",
                "cuda_latency_ms",
                "wall_latency_ms",
            )
        },
    }


def _load_manifest_entries(cache_dirs: list[Path], checkpoint_sha256: str):
    entries = []
    for cache_dir in cache_dirs:
        manifest = cache_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "phase-route-vla.vision-teacher-call.v3":
                raise ValueError(f"M4.20b requires v3 cache: {manifest}")
            if record.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"cache/checkpoint SHA mismatch: {manifest}")
            entries.append((cache_dir, record))
    if not entries:
        raise ValueError("no cache records selected")
    return entries


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.fm_steps != 10:
        raise ValueError("M4.20b RP-PEP v1 requires --fm-steps 10")
    if args.warmup_pairs < 0:
        raise ValueError("warmup-pairs cannot be negative")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("paired replay requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    visible_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if normalize_gpu_uuid(visible_uuid) != normalize_gpu_uuid(args.expected_gpu_uuid):
        raise RuntimeError(
            f"GPU UUID mismatch: expected {args.expected_gpu_uuid}, visible {visible_uuid}"
        )

    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
    )
    from robot_experiments.robot_utils import set_seed_everywhere

    cache_dirs = [path.resolve() for path in args.cache_dir]
    entries = _load_manifest_entries(cache_dirs, args.checkpoint_sha256)
    checkpoint = args.checkpoint.resolve()
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_spatial",
        action_head_flow_matching_inference_steps=args.fm_steps,
        exit_interval=2,
        steps_per_stage=1,
        threshold_type="cosine",
        exit_dist="exp",
        exit_ratio=1.0,
        use_wandb=False,
        save_rollout_video=False,
        seed=args.seed,
    )
    set_seed_everywhere(args.seed)
    model, device, _ = initialize_and_load_model(cfg)
    baseline_controller, sparse_controller, threshold_path, plan = _make_controllers(
        cfg, model, device
    )
    model_dtype = model.transformer.wte.embedding.dtype
    amp_enabled = not args.disable_amp

    first_inputs, first_cpu_rng, first_cuda_rng = _load_call_inputs(
        entries[0][0], entries[0][1], device=device, model_dtype=model_dtype
    )
    for _ in range(args.warmup_pairs):
        _run_policy(
            model,
            baseline_controller,
            first_inputs,
            cpu_rng_state=first_cpu_rng,
            cuda_rng_state=first_cuda_rng,
            device=device,
            timestep=int(entries[0][1]["step_id"]),
            amp_enabled=amp_enabled,
        )
        _run_policy(
            model,
            sparse_controller,
            first_inputs,
            cpu_rng_state=first_cpu_rng,
            cuda_rng_state=first_cuda_rng,
            device=device,
            timestep=int(entries[0][1]["step_id"]),
            amp_enabled=amp_enabled,
        )
    baseline_controller.value_net.reset_actions()
    sparse_controller.value_net.reset_actions()

    rows = []
    for index, (cache_dir, record) in enumerate(entries):
        inputs, cpu_rng_state, cuda_rng_state = _load_call_inputs(
            cache_dir, record, device=device, model_dtype=model_dtype
        )
        baseline = _run_policy(
            model,
            baseline_controller,
            inputs,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            device=device,
            timestep=int(record["step_id"]),
            amp_enabled=amp_enabled,
        )
        sparse = _run_policy(
            model,
            sparse_controller,
            inputs,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_state=cuda_rng_state,
            device=device,
            timestep=int(record["step_id"]),
            amp_enabled=amp_enabled,
        )
        row = _compare_pair(record, baseline, sparse)
        row["record_index"] = index
        row["cache_dir"] = str(cache_dir)
        rows.append(row)
        print(
            f"[{index + 1:03d}/{len(entries):03d}] task={row['task_id']} "
            f"step={row['step_id']} exit={row['baseline']['exit_layer']} "
            f"exact={row['action_exact']} fm={row['baseline']['fm_calls']}->"
            f"{row['rp_pep']['fm_calls']} cuda_ms="
            f"{row['baseline']['cuda_latency_ms']:.2f}->"
            f"{row['rp_pep']['cuda_latency_ms']:.2f}",
            flush=True,
        )

    gate = evaluate_gate(rows)
    result = {
        "status": gate["status"],
        "scope": "m420b_rp_pep_paired_fixed_observation",
        "checkpoint": str(checkpoint / "model.pt"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "threshold_path": str(threshold_path.resolve()),
        "threshold_sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
        "cache_dirs": [str(path) for path in cache_dirs],
        "records": len(rows),
        "fm_steps": args.fm_steps,
        "seed": args.seed,
        "amp_enabled": amp_enabled,
        "warmup_pairs": args.warmup_pairs,
        "physical_gpu_uuid_visible": visible_uuid,
        "physical_gpu_uuid_nvidia_smi": args.expected_gpu_uuid,
        "productive_exit_plan": {
            "name": plan.name,
            "original_exit_layers": list(plan.original_exit_layers),
            "eligible_exit_layers": list(plan.eligible_exit_layers),
            "comparison_references": plan.comparison_references,
            "rng_burns": plan.rng_burns,
        },
        "gate": gate,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
