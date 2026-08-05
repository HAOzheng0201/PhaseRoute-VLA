"""Localize compressed-vision gripper errors inside A1's FM Euler solve."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.fm_diagnostics import (
    flow_matching_euler_trajectory,
)
from a1.vla.dynamic_compute.frozen_a1_distillation import (
    freeze_a1_for_action_distillation,
    frozen_a1_context_forward,
)
from scripts.dynamic_compute.analyze_m49_paired_action_replay import (
    _parse_named_paths,
    action_error_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--aggregator", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--step-id", type=int, required=True)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--input-perturbation", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gripper_signature(action: torch.Tensor) -> dict[str, Any]:
    gripper = action.detach().float()[0, :, 6]
    return {
        "chunk": gripper.cpu().tolist(),
        "positive_indices": torch.nonzero(
            gripper >= 0.5, as_tuple=False
        ).flatten().cpu().tolist(),
        "negative_indices": torch.nonzero(
            gripper <= -0.5, as_tuple=False
        ).flatten().cpu().tolist(),
    }


def _trajectory_rows(trace, reference_states: torch.Tensor | None):
    rows = []
    steps = trace.vector_fields.shape[0]
    for index in range(steps + 1):
        state = trace.states[index].detach().float()
        row = {
            "state_index": index,
            "state_time": 1.0 - index / float(steps),
            "gripper_chunk": state[0, :, 6].cpu().tolist(),
            "vs_full_token": (
                action_error_metrics(state, reference_states[index].float())
                if reference_states is not None
                else None
            ),
        }
        if index < steps:
            row["vector_field_gripper"] = (
                trace.vector_fields[index]
                .detach()
                .float()[0, :, 6]
                .cpu()
                .tolist()
            )
        rows.append(row)
    return rows


def _solve(model, context, batch, input_x, fm_steps, amp_enabled):
    device_type = input_x.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=amp_enabled and device_type == "cuda",
    ):
        return flow_matching_euler_trajectory(
            model,
            context.attn_key_values,
            batch["action_proprio"],
            context.fm_pos_offset,
            input_x,
            steps=fm_steps,
        )


def _input_gripper_sensitivity(
    model,
    context,
    batch,
    base_trace,
    fm_steps,
    epsilon,
    amp_enabled,
):
    input_x = batch["teacher_exit_input_x"]
    horizon = input_x.shape[1]
    base_gripper = base_trace.final_action.detach().float()[0, :, 6]
    columns = []
    for input_index in range(horizon):
        perturbed = input_x.clone()
        perturbed[:, input_index, 6] += epsilon
        trace = _solve(
            model,
            context,
            batch,
            perturbed,
            fm_steps,
            amp_enabled,
        )
        columns.append(
            (
                trace.final_action.detach().float()[0, :, 6] - base_gripper
            )
            / epsilon
        )
    jacobian = torch.stack(columns, dim=1)
    return {
        "epsilon": epsilon,
        "jacobian": jacobian.cpu().tolist(),
        "frobenius_norm": float(torch.linalg.vector_norm(jacobian).cpu()),
        "max_abs": float(jacobian.abs().max().cpu()),
        "mean_abs_diagonal": float(jacobian.diagonal().abs().mean().cpu()),
        "max_row_l1": float(jacobian.abs().sum(dim=1).max().cpu()),
    }


def _seed_interventions(
    model,
    context,
    batch,
    base_trace,
    teacher,
    fm_steps,
    amp_enabled,
):
    base = base_trace.final_action.detach().float()
    interventions = {}
    for name, replacement in (
        ("zero_gripper", torch.zeros_like(batch["teacher_exit_input_x"][..., 6:])),
        ("teacher_gripper", teacher[..., 6:]),
    ):
        input_x = batch["teacher_exit_input_x"].clone()
        input_x[..., 6:] = replacement.to(input_x)
        trace = _solve(
            model,
            context,
            batch,
            input_x,
            fm_steps,
            amp_enabled,
        )
        action = trace.final_action.detach().float()
        interventions[name] = {
            "gripper": _gripper_signature(action),
            "vs_teacher": action_error_metrics(action, teacher),
            "change_from_cached_seed": action_error_metrics(action, base),
        }
    return interventions


def _set_residual_scale(aggregator, scale: float) -> None:
    if not 0.0 <= scale < 1.0:
        raise ValueError("residual scale must be in [0, 1)")
    value = torch.atanh(
        torch.tensor(
            scale,
            dtype=aggregator.residual_gate_logit.dtype,
            device=aggregator.residual_gate_logit.device,
        )
    )
    aggregator.residual_gate_logit.copy_(value)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.fm_steps < 1:
        raise ValueError("fm-steps must be positive")
    if args.input_perturbation <= 0.0:
        raise ValueError("input-perturbation must be positive")
    scales = args.residual_scale or [
        0.0,
        0.1,
        0.25,
        0.5,
        0.9,
        0.95,
        0.98,
        0.99,
    ]
    if len(set(scales)) != len(scales):
        raise ValueError("residual scales must be unique")
    for scale in scales:
        if not 0.0 <= scale < 1.0:
            raise ValueError("residual scales must be in [0, 1)")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M4.11 FM sensitivity analysis requires CUDA")

    # Imports below need the full A1/LIBERO stack only for an actual replay.
    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
    )
    from scripts.dynamic_compute.train_m46_frozen_a1_distillation import (
        FrozenA1TeacherDataset,
        _load_aggregator,
        _move,
    )

    named_paths = _parse_named_paths(args.aggregator)
    cache_dirs = [path.resolve() for path in args.cache_dir]
    dataset = FrozenA1TeacherDataset(
        cache_dirs,
        args.checkpoint_sha256,
        require_candidate_traces=False,
    )
    selected = [
        index
        for index, record in enumerate(dataset.records)
        if int(record["task_id"]) == args.task_id
        and int(record["step_id"]) == args.step_id
    ]
    if len(selected) != 1:
        raise ValueError("task/step filter must select exactly one cache record")
    batch = _move(
        next(iter(DataLoader(Subset(dataset, selected), batch_size=1))),
        device,
    )
    teacher = batch["teacher_action"].float()
    amp_enabled = not args.disable_amp

    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint.resolve()),
        action_head_flow_matching_inference_steps=args.fm_steps,
        use_wandb=False,
        seed=args.seed,
    )
    model, _, _ = initialize_and_load_model(cfg)
    freeze_a1_for_action_distillation(model)
    aggregators = {}
    aggregator_metadata = {}
    for name, path in named_paths:
        aggregator, config, checkpoint = _load_aggregator(path, device)
        if checkpoint.get("teacher_checkpoint_sha256") != args.checkpoint_sha256:
            raise ValueError(f"{name} and A1 checkpoint fingerprints differ")
        aggregator.eval()
        aggregators[name] = aggregator
        aggregator_metadata[name] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _file_sha256(path),
            "output_tokens": config.output_tokens,
            "residual_scale": float(
                torch.tanh(aggregator.residual_gate_logit.float()).item()
            ),
        }

    methods = {}
    reference_states = None
    finite = True
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for name, aggregator in [("full_token", None), *aggregators.items()]:
            started = time.perf_counter()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                context = frozen_a1_context_forward(model, aggregator, batch)
            trace = _solve(
                model,
                context,
                batch,
                batch["teacher_exit_input_x"],
                args.fm_steps,
                amp_enabled,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                standard = model.predict_actions_flow_matching(
                    context.attn_key_values,
                    batch["action_proprio"],
                    context.fm_pos_offset,
                    input_x=batch["teacher_exit_input_x"],
                )
            replay_max_abs = float(
                (trace.final_action.float() - standard.float()).abs().max().cpu()
            )
            final_action = trace.final_action.detach().float()
            method = {
                "exit_layer": context.exit_layer,
                "fm_pos_offset": int(context.fm_pos_offset[0].cpu()),
                "inference_seconds": time.perf_counter() - started,
                "standard_replay_max_abs_error": replay_max_abs,
                "final_action": final_action[0].cpu().tolist(),
                "gripper": _gripper_signature(final_action),
                "vs_teacher": action_error_metrics(final_action, teacher),
                "trajectory": _trajectory_rows(trace, reference_states),
                "input_gripper_sensitivity": _input_gripper_sensitivity(
                    model,
                    context,
                    batch,
                    trace,
                    args.fm_steps,
                    args.input_perturbation,
                    amp_enabled,
                ),
                "seed_interventions": _seed_interventions(
                    model,
                    context,
                    batch,
                    trace,
                    teacher,
                    args.fm_steps,
                    amp_enabled,
                ),
            }
            if name == "full_token":
                reference_states = trace.states.detach().float()
            else:
                method["vs_full_token"] = action_error_metrics(
                    final_action, reference_states[-1]
                )
            methods[name] = method
            finite = finite and replay_max_abs <= 1e-4
            del context, trace, standard
            torch.cuda.empty_cache()

        residual_sweeps = {}
        for name, aggregator in aggregators.items():
            original_logit = aggregator.residual_gate_logit.detach().clone()
            rows = []
            try:
                for scale in scales:
                    _set_residual_scale(aggregator, scale)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        context = frozen_a1_context_forward(
                            model, aggregator, batch
                        )
                    trace = _solve(
                        model,
                        context,
                        batch,
                        batch["teacher_exit_input_x"],
                        args.fm_steps,
                        amp_enabled,
                    )
                    action = trace.final_action.detach().float()
                    rows.append(
                        {
                            "residual_scale": scale,
                            "gripper": _gripper_signature(action),
                            "vs_teacher": action_error_metrics(action, teacher),
                            "vs_full_token": action_error_metrics(
                                action, reference_states[-1]
                            ),
                        }
                    )
                    del context, trace
                    torch.cuda.empty_cache()
            finally:
                aggregator.residual_gate_logit.copy_(original_logit)
            residual_sweeps[name] = rows

    finite = finite and all(
        math.isfinite(float(value))
        for method in methods.values()
        for value in method["vs_teacher"].values()
    )
    result = {
        "status": "PASS" if finite else "FAIL",
        "scope": "m411_fm_trajectory_and_seed_sensitivity",
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_checkpoint_sha256": args.checkpoint_sha256,
        "cache_dirs": [str(path) for path in cache_dirs],
        "source_record_index": selected[0],
        "task_id": args.task_id,
        "step_id": args.step_id,
        "fm_steps": args.fm_steps,
        "seed": args.seed,
        "amp_enabled": amp_enabled,
        "input_perturbation": args.input_perturbation,
        "teacher_action": teacher[0].cpu().tolist(),
        "cached_input_x": batch["teacher_exit_input_x"][0].float().cpu().tolist(),
        "aggregators": aggregator_metadata,
        "methods": methods,
        "residual_scale_sweeps": residual_sweeps,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
