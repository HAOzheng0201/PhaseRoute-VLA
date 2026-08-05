"""M4.24 oracle route-then-solve fixed-observation profiler.

The route and expected action SHA come from a frozen M4.23 early-exit result.
The oracle never reads the action tensor: it runs the transformer to the
frozen route, burns only the skipped initial-noise draws, and performs one
final FM10 solve.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.profile_m423_fixed_observations import (  # noqa: E402
    CudaComponentProfiler,
    canonical_record_identity,
    select_stratified_entries,
    selection_sha256,
    sha256_tensor,
    summarize,
)
from scripts.dynamic_compute.replay_m420b_rp_pep import (  # noqa: E402
    _load_call_inputs,
    _load_manifest_entries,
    normalize_gpu_uuid,
)


ORIGINAL_FM_CALLS = {3: 3, 11: 7, 13: 8, 27: 15}
ORACLE_RNG_BURNS = {layer: calls - 1 for layer, calls in ORIGINAL_FM_CALLS.items()}
EXPECTED_MODEL_CLASS = "a1.vla.affordvla_early_exit.AffordVLAEarlyExit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--route-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--order-position", type=int, choices=(1, 2), required=True)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--exit-layer", type=int, action="append")
    parser.add_argument("--records-per-exit", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(Path(row["cache_dir"]).resolve()),
        str(row["array_path"]),
        str(row["episode_id"]),
        int(row["task_id"]),
        int(row["step_id"]),
        int(row["teacher_exit_layer"]),
    )


def entry_route_key(cache_dir: Path, record: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = canonical_record_identity(cache_dir, record)
    return route_key(identity)


def load_frozen_routes(
    result: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    expected_selection_sha256: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Validate the frozen M4.23 route source and collapse its two repeats."""

    if result.get("scope") != "m423_fixed_observation_policy_profile":
        raise ValueError("route source has unexpected scope")
    if result.get("status") != "PASS" or result.get("policy") != "early_exit":
        raise ValueError("route source must be a PASS early_exit M4.23 profile")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("route/checkpoint SHA mismatch")
    if result.get("selection_sha256") != expected_selection_sha256:
        raise ValueError("route/cache selection SHA mismatch")

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in result.get("timed_samples", []):
        grouped[route_key(row)].append(row)
    if not grouped:
        raise ValueError("route source has no timed samples")

    routes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, rows in grouped.items():
        repeats = {int(row["repeat"]) for row in rows}
        if repeats != {0, 1} or len(rows) != 2:
            raise ValueError(f"route source must contain repeats 0/1 exactly once: {key}")
        exit_layers = {int(row["exit_layer"]) for row in rows}
        fm_calls = {int(row["fm_calls"]) for row in rows}
        action_hashes = {str(row["action_sha256"]) for row in rows}
        if len(exit_layers) != 1 or len(fm_calls) != 1 or len(action_hashes) != 1:
            raise ValueError(f"route source repeats disagree: {key}")
        exit_layer = next(iter(exit_layers))
        calls = next(iter(fm_calls))
        if exit_layer not in ORIGINAL_FM_CALLS:
            raise ValueError(f"unsupported frozen exit layer {exit_layer}")
        if calls != ORIGINAL_FM_CALLS[exit_layer]:
            raise ValueError(
                f"frozen FM formula mismatch at layer {exit_layer}: {calls}"
            )
        routes[key] = {
            "route_layer": exit_layer,
            "original_fm_calls": calls,
            "rng_burns": ORACLE_RNG_BURNS[exit_layer],
            "expected_action_sha256": next(iter(action_hashes)),
        }
    return routes


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_model(args: argparse.Namespace):
    import robot_experiments.libero.eval_libero_early_exit as evaluation
    from robot_experiments.robot_utils import set_seed_everywhere

    cfg = evaluation.GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
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
    model, device, _ = evaluation.initialize_and_load_model(cfg)
    return cfg, model, device


def execute_oracle(
    *,
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
    device: torch.device,
    route: Mapping[str, Any],
    amp_enabled: bool,
) -> dict[str, Any]:
    route_layer = int(route["route_layer"])
    burns = int(route["rng_burns"])
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start_event.record()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled
    ):
        transformer_output = model.forward(**inputs, exit_id=route_layer)
        if int(transformer_output.exit_layer) != route_layer:
            raise RuntimeError("forced transformer route did not stop at requested layer")
        kvs = transformer_output.attn_key_values
        if not kvs or len(kvs) != route_layer + 1:
            raise RuntimeError("forced transformer route returned invalid KV depth")
        first = kvs[0][0]
        noise_shape = (
            first.shape[0],
            int(model.config.num_actions_chunk),
            int(model.config.fixed_action_dim),
        )
        for _ in range(burns):
            torch.randn(noise_shape, device=first.device, dtype=first.dtype)
        action = model.predict_actions_flow_matching(
            kvs,
            inputs["action_proprio"],
            transformer_output.fm_pos_offset,
        )
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    action_cpu = action.detach().to(device="cpu", dtype=torch.float32)
    action_sha = sha256_tensor(action_cpu)
    return {
        "route_layer": route_layer,
        "exit_layer": route_layer,
        "transformer_layers_executed": route_layer + 1,
        "original_fm_calls": int(route["original_fm_calls"]),
        "fm_calls": 1,
        "fm_steps": int(model.config.num_diffusion_inference_steps),
        "rng_burns": burns,
        "expected_action_sha256": str(route["expected_action_sha256"]),
        "action_sha256": action_sha,
        "action_exact": action_sha == str(route["expected_action_sha256"]),
        "action_finite": bool(torch.isfinite(action_cpu).all()),
        "action_shape": list(action_cpu.shape),
        "cuda_latency_ms": cuda_ms,
        "wall_latency_ms": wall_ms,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.physical_gpu_index not in (0, 1, 2, 3):
        raise ValueError("M4.24 only permits physical GPUs 0-3")
    if args.fm_steps != 10:
        raise ValueError("M4.24 oracle is frozen to FM10")
    if args.repeats != 2 or args.warmup_calls != 1:
        raise ValueError("M4.24 is frozen to repeats=2 and warmup-calls=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M4.24 requires exactly one visible CUDA device")
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
    route_path = args.route_result.resolve()
    route_result = json.loads(route_path.read_text(encoding="utf-8"))
    routes = load_frozen_routes(
        route_result,
        checkpoint_sha256=args.checkpoint_sha256,
        expected_selection_sha256=selected_sha,
    )
    entry_keys = {entry_route_key(*item) for item in entries}
    if routes.keys() != entry_keys:
        missing = sorted(entry_keys - routes.keys())
        extra = sorted(routes.keys() - entry_keys)
        raise ValueError(f"route/cache grids differ; missing={missing}, extra={extra}")

    _, model, device = load_model(args)
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_class != EXPECTED_MODEL_CLASS:
        raise RuntimeError(f"unexpected oracle model class: {model_class}")
    model_dtype = model.transformer.wte.embedding.dtype
    amp_enabled = not args.disable_amp

    first_inputs, first_cpu_rng, first_cuda_rng = _load_call_inputs(
        entries[0][0], entries[0][1], device=device, model_dtype=model_dtype
    )
    first_route = routes[entry_route_key(*entries[0])]
    execute_oracle(
        model=model,
        inputs=first_inputs,
        cpu_rng_state=first_cpu_rng,
        cuda_rng_state=first_cuda_rng,
        device=device,
        route=first_route,
        amp_enabled=amp_enabled,
    )

    torch.cuda.reset_peak_memory_stats(device)
    timed_samples: list[dict[str, Any]] = []
    for record_index, (cache_dir, record) in enumerate(entries):
        inputs, cpu_rng_state, cuda_rng_state = _load_call_inputs(
            cache_dir, record, device=device, model_dtype=model_dtype
        )
        route = routes[entry_route_key(cache_dir, record)]
        for repeat in range(args.repeats):
            random.seed(args.seed + record_index * 1000 + repeat)
            measurement = execute_oracle(
                model=model,
                inputs=inputs,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
                device=device,
                route=route,
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
                f"timed policy=oracle_rts record={record_index + 1:02d}/{len(entries):02d} "
                f"repeat={repeat + 1}/{args.repeats} route={route['route_layer']} "
                f"burns={route['rng_burns']} exact={measurement['action_exact']} "
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
            route = routes[entry_route_key(cache_dir, record)]
            random.seed(args.seed + record_index * 1000)
            profiler.start()
            measurement = execute_oracle(
                model=model,
                inputs=inputs,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
                device=device,
                route=route,
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
                f"component policy=oracle_rts record={record_index + 1:02d}/{len(entries):02d} "
                f"route={route['route_layer']} total={measurement['cuda_latency_ms']:.2f} "
                f"transformer={components['transformer_ms']:.2f} "
                f"fm_head={components['fm_head_ms']:.2f} other={other_ms:.2f}",
                flush=True,
            )
    finally:
        profiler.close()
    component_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    component_peak_reserved = int(torch.cuda.max_memory_reserved(device))

    all_rows = timed_samples + component_samples
    local_checks = {
        "timed_sample_count": len(timed_samples) == len(entries) * args.repeats,
        "component_sample_count": len(component_samples) == len(entries),
        "strict_action_equivalence": all(row["action_exact"] for row in all_rows),
        "all_actions_finite": all(row["action_finite"] for row in all_rows),
        "one_fm_solve": all(
            row["fm_calls"] == 1 and row["fm_steps"] == args.fm_steps
            for row in all_rows
        ),
        "rng_burn_formula": all(
            row["rng_burns"] == ORACLE_RNG_BURNS[row["route_layer"]]
            and row["original_fm_calls"] == ORIGINAL_FM_CALLS[row["route_layer"]]
            for row in all_rows
        ),
        "component_events_consistent": all(
            row["transformer_block_calls"] == row["route_layer"] + 1
            and row["fm_vector_field_calls"] == args.fm_steps
            for row in component_samples
        ),
        "component_other_nonnegative": all(
            row["instrumented_other_ms"] >= -1.0 for row in component_samples
        ),
    }
    status = "PASS" if all(local_checks.values()) else "FAIL"
    source_status = git_output("status", "--porcelain=v1")
    result = {
        "status": status,
        "scope": "m424_oracle_route_then_solve_profile",
        "policy": "oracle_rts",
        "model_class": model_class,
        "checkpoint": str(args.checkpoint.resolve() / "model.pt"),
        "checkpoint_sha256": args.checkpoint_sha256,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "cache_dirs": [str(path) for path in cache_dirs],
        "cache_records_available": len(all_entries),
        "selected_records": [canonical_record_identity(*item) for item in entries],
        "selection_sha256": selected_sha,
        "route_source": str(route_path),
        "route_source_sha256": sha256_file(route_path),
        "route_distribution": {
            str(layer): sum(route["route_layer"] == layer for route in routes.values())
            for layer in sorted(ORIGINAL_FM_CALLS)
        },
        "original_fm_calls": ORIGINAL_FM_CALLS,
        "oracle_rng_burns": ORACLE_RNG_BURNS,
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
