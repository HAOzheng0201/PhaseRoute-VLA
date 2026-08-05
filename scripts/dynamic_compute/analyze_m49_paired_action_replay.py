"""Replay cached A1 states to localize learned-vision action drift."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--aggregator",
        action="append",
        required=True,
        help="Named checkpoint in NAME=PATH form; may be warmup or distilled schema.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--step-id", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def _parse_named_paths(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"aggregator must use NAME=PATH form: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name == "full_token":
            raise ValueError("aggregator name must be non-empty and not full_token")
        if name in seen:
            raise ValueError(f"duplicate aggregator name: {name}")
        seen.add(name)
        result.append((name, Path(raw_path).expanduser().resolve()))
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_error_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    predicted = predicted.detach().float()
    target = target.detach().float()
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("paired actions must share shape [B, H, A]")
    error = (predicted - target).abs()
    metrics = {
        "mae": float(error.mean().cpu()),
        "max_abs_error": float(error.max().cpu()),
        "first_step_mae": float(error[:, 0].mean().cpu()),
    }
    if predicted.shape[-1] >= 7:
        metrics.update(
            translation_mae=float(error[..., :3].mean().cpu()),
            rotation_mae=float(error[..., 3:6].mean().cpu()),
            gripper_mae=float(error[..., 6:].mean().cpu()),
            first_translation_mae=float(error[:, 0, :3].mean().cpu()),
            first_rotation_mae=float(error[:, 0, 3:6].mean().cpu()),
            first_gripper_mae=float(error[:, 0, 6:].mean().cpu()),
            first_gripper_direction_mismatch=float(
                ((predicted[:, 0, 6:] >= 0) != (target[:, 0, 6:] >= 0))
                .float()
                .mean()
                .cpu()
            ),
        )
    return metrics


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        name: statistics.fmean(float(row[name]) for row in rows)
        for name in rows[0]
    }


def _aggregate_replay_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_names = list(rows[0]["methods"])
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)

    def summarize(selected: list[dict[str, Any]]):
        methods = {}
        for method in method_names:
            method_rows = [row["methods"][method] for row in selected]
            methods[method] = {
                "vs_teacher": _aggregate_metric_rows(
                    [row["vs_teacher"] for row in method_rows]
                ),
                "vs_full_token": (
                    _aggregate_metric_rows(
                        [row["vs_full_token"] for row in method_rows]
                    )
                    if method != "full_token"
                    else None
                ),
                "inference_seconds_total": sum(
                    float(row["inference_seconds"]) for row in method_rows
                ),
            }
        return {
            "calls": len(selected),
            "mean_exit_layer": statistics.fmean(
                float(row["exit_layer"]) for row in selected
            ),
            "methods": methods,
        }

    return {
        "overall": summarize(rows),
        "by_task": {
            str(task_id): summarize(task_rows)
            for task_id, task_rows in sorted(by_task.items())
        },
    }


def main() -> None:
    # Parse first so ``--help`` and argument errors never initialize the A1
    # model stack or MuJoCo/EGL.  The heavyweight imports below are only
    # needed after argparse has accepted an actual replay request.
    args = parse_args()
    # Keep model/LIBERO imports lazy so metric helpers remain testable in a
    # restricted CPU environment without initializing MuJoCo EGL.
    from a1.vla.dynamic_compute.frozen_a1_distillation import (
        freeze_a1_for_action_distillation,
        frozen_a1_action_forward,
    )
    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
    )
    from scripts.dynamic_compute.train_m46_frozen_a1_distillation import (
        FrozenA1TeacherDataset,
        _load_aggregator,
        _move,
    )

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.fm_steps < 1:
        raise ValueError("fm-steps must be positive")
    named_paths = _parse_named_paths(args.aggregator)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paired A1 action replay requires CUDA")

    cache_dirs = [path.resolve() for path in args.cache_dir]
    dataset = FrozenA1TeacherDataset(
        cache_dirs,
        args.checkpoint_sha256,
        require_candidate_traces=False,
    )
    selected_indices = [
        index
        for index, record in enumerate(dataset.records)
        if (args.task_id is None or int(record["task_id"]) == args.task_id)
        and (args.step_id is None or int(record["step_id"]) == args.step_id)
    ]
    if not selected_indices:
        raise ValueError("task/step filters selected no cache records")
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
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
        teacher_hash = checkpoint.get("teacher_checkpoint_sha256")
        if teacher_hash != args.checkpoint_sha256:
            raise ValueError(f"{name} and frozen A1 checkpoint fingerprints differ")
        aggregator.eval()
        aggregators[name] = aggregator
        aggregator_metadata[name] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _file_sha256(path),
            "schema_version": checkpoint.get("schema_version"),
            "output_tokens": config.output_tokens,
        }

    amp_enabled = not args.disable_amp
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            teacher = batch["teacher_action"].float()
            method_actions = {}
            method_seconds = {}
            for name, aggregator in [("full_token", None), *aggregators.items()]:
                started = time.perf_counter()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    output = frozen_a1_action_forward(model, aggregator, batch)
                torch.cuda.synchronize(device)
                method_seconds[name] = time.perf_counter() - started
                method_actions[name] = output.normalized_action.detach().float()

            full_token = method_actions["full_token"]
            methods = {}
            for name, action in method_actions.items():
                methods[name] = {
                    "vs_teacher": action_error_metrics(action, teacher),
                    "vs_full_token": (
                        action_error_metrics(action, full_token)
                        if name != "full_token"
                        else None
                    ),
                    "inference_seconds": method_seconds[name],
                    "action_chunk": action[0].cpu().tolist(),
                }
            rows.append(
                {
                    "record_index": int(batch["source_record_index"].item()),
                    "task_id": int(batch["source_task_id"].item()),
                    "step_id": int(batch["source_step_id"].item()),
                    "episode_id": dataset.episode_ids[
                        int(batch["source_record_index"].item())
                    ],
                    "exit_layer": int(batch["teacher_exit_layer"].item()),
                    "teacher_action_chunk": teacher[0].cpu().tolist(),
                    "methods": methods,
                }
            )

    aggregates = _aggregate_replay_rows(rows)
    finite = all(
        math.isfinite(float(metric_value))
        for row in rows
        for method in row["methods"].values()
        for comparison in ("vs_teacher", "vs_full_token")
        if method[comparison] is not None
        for metric_value in method[comparison].values()
    )
    result = {
        "status": "PASS" if finite else "FAIL",
        "scope": "m49_cached_observation_paired_action_replay",
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_checkpoint_sha256": args.checkpoint_sha256,
        "cache_dirs": [str(path) for path in cache_dirs],
        "cache_records_total": len(dataset),
        "cache_records_selected": len(rows),
        "task_id_filter": args.task_id,
        "step_id_filter": args.step_id,
        "fm_steps": args.fm_steps,
        "seed": args.seed,
        "amp_enabled": amp_enabled,
        "aggregators": aggregator_metadata,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "aggregates": aggregates,
        "rows": rows,
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
