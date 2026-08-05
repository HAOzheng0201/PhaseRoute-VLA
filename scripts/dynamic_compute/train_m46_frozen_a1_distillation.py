"""Train learnable EFA with real frozen-A1 normalized-action supervision."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.frozen_a1_distillation import (
    configure_frozen_a1_activation_checkpointing,
    freeze_a1_for_action_distillation,
    frozen_a1_action_distillation_loss,
    frozen_a1_action_forward,
    gripper_transition_mask,
    select_cached_candidate_supervision,
)
from a1.vla.dynamic_compute.learnable_vision_aggregation import (
    LearnableVisionAggregationConfig,
    LearnableVisionAggregator,
    reparameterize_residual_scale,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    VISION_TEACHER_CACHE_SCHEMA_VERSION_V2,
)
from a1.vla.dynamic_compute.learnable_vision_runtime import (
    DISTILLED_EFA_CHECKPOINT_SCHEMA_VERSION,
)
CHECKPOINT_SCHEMA_VERSION = "phase-route-vla.frozen-a1-efa-distillation.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("model/libero_exit"))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--warmup-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-full-token-replay", action="store_true")
    parser.add_argument("--residual-scale-reparameterization", type=float)
    parser.add_argument("--first-step-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--gripper-transition-loss-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--gripper-transition-threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--final-supervision-probability", type=float, default=0.0
    )
    parser.add_argument(
        "--gripper-transition-replay-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of drawing a cached final-action gripper-transition "
            "record and forcing final supervision for one optimizer step."
        ),
    )
    parser.add_argument(
        "--activation-checkpointing",
        default="none",
        help=(
            "Frozen-LLM activation recomputation strategy. Use whole_layer "
            "for deep candidate exits that do not fit in one 48 GiB GPU."
        ),
    )
    parser.add_argument(
        "--supervision",
        choices=("final", "random_candidate", "deepest_candidate"),
        default="final",
    )
    return parser.parse_args()


class FrozenA1TeacherDataset(Dataset):
    def __init__(
        self,
        cache_dirs: list[Path],
        checkpoint_sha256: str,
        *,
        require_candidate_traces: bool,
    ):
        self.entries: list[tuple[Path, dict[str, Any]]] = []
        for cache_dir in cache_dirs:
            manifest_path = cache_dir / "manifest.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.entries.extend((cache_dir, record) for record in records)
        self.records = [record for _, record in self.entries]
        if len(self.records) < 2:
            raise ValueError("frozen-A1 distillation requires at least two cache calls")
        for record in self.records:
            allowed_schemas = {
                VISION_TEACHER_CACHE_SCHEMA_VERSION_V2,
                VISION_TEACHER_CACHE_SCHEMA_VERSION,
            }
            if record.get("schema_version") not in allowed_schemas:
                raise ValueError("frozen-A1 distillation requires v2 or v3 cache")
            if (
                require_candidate_traces
                and record.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION
            ):
                raise ValueError("candidate supervision requires v3 teacher cache")
            if record.get("teacher_kind") != "a1_early_exit":
                raise ValueError("cache teacher_kind is not a1_early_exit")
            if record.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError("cache and requested teacher checkpoint do not match")
            if float(record.get("teacher_trace_max_abs_error", math.inf)) > 1e-5:
                raise ValueError("cache contains a misaligned final FM trace")
        self.checkpoint_sha256 = checkpoint_sha256
        self.episode_ids = [str(record["episode_id"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cache_dir, record = self.entries[index]
        shard_path = cache_dir / record["array_path"]
        with np.load(shard_path) as shard:
            result = {
                "projected_features": torch.from_numpy(
                    shard["projected_features"].astype(np.float32)
                ),
                "image_input_idx": torch.from_numpy(
                    shard["image_input_idx"].astype(np.int64)
                ),
                "instruction_summary": torch.from_numpy(
                    shard["instruction_summary"].astype(np.float32)
                ),
                "teacher_action": torch.from_numpy(
                    shard["teacher_normalized_action"].astype(np.float32)
                ),
                "teacher_exit_input_x": torch.from_numpy(
                    shard["teacher_exit_input_x"].astype(np.float32)
                ),
                "teacher_exit_layer": torch.tensor(
                    int(record["teacher_exit_layer"]), dtype=torch.int64
                ),
                "source_record_index": torch.tensor(index, dtype=torch.int64),
                "source_task_id": torch.tensor(
                    int(record["task_id"]), dtype=torch.int64
                ),
                "source_step_id": torch.tensor(
                    int(record["step_id"]), dtype=torch.int64
                ),
            }
            tensor_dtypes = {
                "input_ids": np.int64,
                "attention_mask": np.bool_,
                "attention_bias": np.float32,
                "response_mask": np.bool_,
                "subsegment_ids": np.int64,
                "position_ids": np.int64,
                "action_proprio": np.float32,
                "proprio_token_idx": np.int64,
            }
            for name, dtype in tensor_dtypes.items():
                result[name] = torch.from_numpy(shard[name].astype(dtype))
            trace_dtypes = {
                "fm_trace_layers": np.int64,
                "fm_trace_roles": np.uint8,
                "fm_trace_input_x": np.float32,
                "fm_trace_output_action": np.float32,
            }
            for name, dtype in trace_dtypes.items():
                if name in shard:
                    result[name] = torch.from_numpy(shard[name].astype(dtype))
        return result


def _cached_final_gripper_transition_indices(
    dataset: FrozenA1TeacherDataset,
    indices: list[int],
    threshold: float,
) -> list[int]:
    """Find rare final-action transition calls without loading image features."""

    selected = []
    for index in indices:
        cache_dir, record = dataset.entries[index]
        shard_path = cache_dir / record["array_path"]
        with np.load(shard_path) as shard:
            action = torch.from_numpy(
                shard["teacher_normalized_action"].astype(np.float32)
            )
        if action.ndim != 2:
            raise ValueError("cached teacher action must have shape [H, A]")
        if bool(gripper_transition_mask(action.unsqueeze(0), threshold).item()):
            selected.append(index)
    return selected


def _move(batch: dict[str, torch.Tensor], device: torch.device):
    return {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


def _state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_aggregator(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    schema = checkpoint.get("schema_version")
    if schema == "phase-route-vla.learnable-efa-warmup.v1":
        config = LearnableVisionAggregationConfig(**checkpoint["model_config"])
        prefix = "aggregator."
        state = {
            name[len(prefix):]: value
            for name, value in checkpoint["model_state_dict"].items()
            if name.startswith(prefix)
        }
    elif schema == DISTILLED_EFA_CHECKPOINT_SCHEMA_VERSION:
        config = LearnableVisionAggregationConfig(
            **checkpoint["aggregator_config"]
        )
        state = checkpoint["aggregator_state_dict"]
    else:
        raise ValueError("unexpected EFA initialization checkpoint schema")
    aggregator = LearnableVisionAggregator(config)
    aggregator.load_state_dict(state, strict=True)
    return aggregator.to(device), config, checkpoint


def _select_supervision(
    batch: dict[str, torch.Tensor],
    mode: str,
    *,
    rng: random.Random,
    deterministic_index: int = 0,
) -> dict[str, torch.Tensor]:
    if mode == "final":
        return batch
    roles = batch.get("fm_trace_roles")
    if roles is None or roles.ndim != 2 or roles.shape[0] != 1:
        raise ValueError("candidate supervision requires batch size 1 v3 traces")
    candidate_count = int((roles[0] == 1).sum().item())
    if mode == "deepest_candidate":
        ordinal = candidate_count - 1
    else:
        ordinal = (
            deterministic_index
            if deterministic_index
            else rng.randrange(candidate_count)
        )
    batch["candidate_ordinal"] = torch.tensor(
        [ordinal % candidate_count], dtype=torch.int64
    )
    batch["candidate_count"] = torch.tensor(
        [candidate_count], dtype=torch.int64
    )
    return select_cached_candidate_supervision(batch, ordinal)


def _forward(
    model,
    aggregator,
    batch,
    amp_enabled: bool,
    loss_kwargs: dict[str, float] | None = None,
):
    device_type = batch["input_ids"].device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=amp_enabled and device_type == "cuda",
    ):
        output = frozen_a1_action_forward(model, aggregator, batch)
        loss, parts = frozen_a1_action_distillation_loss(
            output,
            batch["teacher_action"],
            **(loss_kwargs or {}),
        )
    metrics = {
        name: float(value.detach().float().cpu()) for name, value in parts.items()
    }
    metrics.update(
        exit_layer=output.exit_layer,
        fm_pos_offset=int(output.fm_pos_offset[0].detach().cpu()),
    )
    for name in (
        "source_record_index",
        "source_task_id",
        "source_step_id",
        "candidate_ordinal",
        "candidate_count",
        "supervision_is_final",
    ):
        if name in batch:
            metrics[name] = int(batch[name].reshape(-1)[0].detach().cpu())
    return loss, metrics


@torch.no_grad()
def _evaluate(
    model,
    aggregator,
    loader,
    device,
    amp_enabled,
    supervision,
    seed,
    loss_kwargs=None,
):
    aggregator.eval()
    rows = []
    rng = random.Random(seed)
    for index, batch in enumerate(loader):
        batch = _select_supervision(
            batch,
            supervision,
            rng=rng,
            deterministic_index=index + 1,
        )
        _, metrics = _forward(
            model,
            aggregator,
            _move(batch, device),
            amp_enabled,
            loss_kwargs,
        )
        rows.append(metrics)
    result = {
        name: statistics.fmean(float(row[name]) for row in rows)
        for name in ("total", "mae", "first_step_mae", "translation_mae", "rotation_mae", "gripper_mae")
    }
    result["mean_supervision_layer"] = statistics.fmean(
        float(row["exit_layer"]) for row in rows
    )
    return result


@torch.no_grad()
def _full_token_replay(model, sample, device, amp_enabled):
    batch = _move(sample, device)
    device_type = device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=amp_enabled and device_type == "cuda",
    ):
        output = frozen_a1_action_forward(model, None, batch)
    predicted = output.normalized_action.float()
    target = batch["teacher_action"].float()
    return {
        "mae": float((predicted - target).abs().mean().cpu()),
        "max_abs_error": float((predicted - target).abs().max().cpu()),
        "exit_layer": output.exit_layer,
        "fm_pos_offset": int(output.fm_pos_offset[0].cpu()),
    }


def main() -> None:
    args = parse_args()
    # Keep LIBERO/MuJoCo imports behind argument parsing so help and dataset
    # utilities remain usable in CPU-only test environments.
    from robot_experiments.libero.eval_libero_early_exit import (
        GenerateConfig,
        initialize_and_load_model,
    )

    if args.steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 <= args.final_supervision_probability <= 1.0:
        raise ValueError("final-supervision-probability must be in [0, 1]")
    if not 0.0 <= args.gripper_transition_replay_probability <= 1.0:
        raise ValueError(
            "gripper-transition-replay-probability must be in [0, 1]"
        )
    if args.supervision == "final" and args.final_supervision_probability:
        raise ValueError(
            "final-supervision-probability is only valid for candidate supervision"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real frozen-A1 distillation smoke requires CUDA")

    cache_dirs = [path.resolve() for path in args.cache_dir]
    dataset = FrozenA1TeacherDataset(
        cache_dirs,
        args.checkpoint_sha256,
        require_candidate_traces=args.supervision != "final",
    )
    split_rng = random.Random(args.seed)
    episode_ids = sorted(set(dataset.episode_ids))
    split_rng.shuffle(episode_ids)
    if len(episode_ids) >= 2:
        validation_episode_count = max(1, len(episode_ids) // 4)
        validation_episodes = set(episode_ids[:validation_episode_count])
        validation_indices = [
            index
            for index, episode_id in enumerate(dataset.episode_ids)
            if episode_id in validation_episodes
        ]
        training_indices = [
            index
            for index, episode_id in enumerate(dataset.episode_ids)
            if episode_id not in validation_episodes
        ]
        split_kind = "episode"
    else:
        indices = list(range(len(dataset)))
        split_rng.shuffle(indices)
        validation_count = max(1, len(indices) // 5)
        validation_indices = sorted(indices[:validation_count])
        training_indices = sorted(indices[validation_count:])
        validation_episodes = set(episode_ids)
        split_kind = "call_fallback_single_episode"
    training_loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    transition_training_indices = _cached_final_gripper_transition_indices(
        dataset,
        training_indices,
        args.gripper_transition_threshold,
    )
    if (
        args.gripper_transition_replay_probability > 0.0
        and not transition_training_indices
    ):
        raise ValueError(
            "transition replay requested but the training split has no "
            "final-action gripper transition records"
        )
    transition_loader = (
        DataLoader(
            Subset(dataset, transition_training_indices),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed + 2),
        )
        if transition_training_indices
        else None
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
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
    activation_checkpointing = configure_frozen_a1_activation_checkpointing(
        model, args.activation_checkpointing
    )
    aggregator, aggregator_config, warmup = _load_aggregator(
        args.warmup_checkpoint.resolve(), device
    )
    if warmup.get("teacher_checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("warmup and frozen A1 checkpoint fingerprints differ")
    residual_reparameterization = None
    if args.residual_scale_reparameterization is not None:
        residual_reparameterization = reparameterize_residual_scale(
            aggregator, args.residual_scale_reparameterization
        )
    amp_enabled = not args.disable_amp
    loss_kwargs = {
        "first_step_weight": args.first_step_loss_weight,
        "gripper_weight": args.gripper_loss_weight,
        "gripper_transition_weight": args.gripper_transition_loss_weight,
        "gripper_transition_threshold": args.gripper_transition_threshold,
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    replay = None
    if not args.skip_full_token_replay:
        replay_sample = next(iter(DataLoader(Subset(dataset, [validation_indices[0]]))))
        replay = _full_token_replay(
            model, replay_sample, device, amp_enabled
        )

    initial_validation = _evaluate(
        model,
        aggregator,
        validation_loader,
        device,
        amp_enabled,
        args.supervision,
        args.seed,
        loss_kwargs,
    )
    initial_final_validation = (
        _evaluate(
            model,
            aggregator,
            validation_loader,
            device,
            amp_enabled,
            "final",
            args.seed,
            loss_kwargs,
        )
        if args.supervision != "final"
        else initial_validation
    )
    initial_hash = _state_sha256(aggregator.state_dict())
    optimizer = torch.optim.AdamW(
        aggregator.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metrics_path = args.output_dir / "training_metrics.jsonl"
    iterator = iter(training_loader)
    transition_iterator = iter(transition_loader) if transition_loader else None
    started = time.perf_counter()
    gradient_observed = False
    aggregator.train()
    supervision_rng = random.Random(args.seed + 1)
    transition_rng = random.Random(args.seed + 2)
    transition_replay_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            replay_transition = (
                transition_loader is not None
                and args.gripper_transition_replay_probability > 0.0
                and transition_rng.random()
                < args.gripper_transition_replay_probability
            )
            if replay_transition:
                try:
                    batch = next(transition_iterator)
                except StopIteration:
                    transition_iterator = iter(transition_loader)
                    batch = next(transition_iterator)
                transition_replay_steps += 1
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(training_loader)
                    batch = next(iterator)
            use_final = (
                replay_transition
                or (
                    args.supervision != "final"
                    and args.final_supervision_probability > 0.0
                    and supervision_rng.random()
                    < args.final_supervision_probability
                )
            )
            if use_final:
                roles = batch.get("fm_trace_roles")
                candidate_count = (
                    int((roles[0] == 1).sum().item())
                    if roles is not None
                    else 0
                )
                batch["candidate_ordinal"] = torch.tensor(
                    [-1], dtype=torch.int64
                )
                batch["candidate_count"] = torch.tensor(
                    [candidate_count], dtype=torch.int64
                )
                batch["supervision_is_final"] = torch.tensor(
                    [1], dtype=torch.int64
                )
            else:
                batch = _select_supervision(
                    batch,
                    args.supervision,
                    rng=supervision_rng,
                )
                batch["supervision_is_final"] = torch.tensor(
                    [int(args.supervision == "final")], dtype=torch.int64
                )
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _forward(
                model, aggregator, batch, amp_enabled, loss_kwargs
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite action loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                aggregator.parameters(), args.grad_clip
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite EFA gradient at step {step}")
            gradient_observed = gradient_observed or float(grad_norm) > 0.0
            optimizer.step()
            row: dict[str, Any] = {
                "step": step,
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": args.learning_rate,
                "transition_replay_sample": int(replay_transition),
                **metrics,
            }
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
    training_seconds = time.perf_counter() - started
    final_validation = _evaluate(
        model,
        aggregator,
        validation_loader,
        device,
        amp_enabled,
        args.supervision,
        args.seed,
        loss_kwargs,
    )
    final_final_validation = (
        _evaluate(
            model,
            aggregator,
            validation_loader,
            device,
            amp_enabled,
            "final",
            args.seed,
            loss_kwargs,
        )
        if args.supervision != "final"
        else final_validation
    )
    final_hash = _state_sha256(aggregator.state_dict())
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    status_ok = (
        gradient_observed
        and initial_hash != final_hash
        and all(math.isfinite(value) for value in final_validation.values())
        and all(
            math.isfinite(value) for value in final_final_validation.values()
        )
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "aggregator_config": asdict(aggregator_config),
        "aggregator_state_dict": {
            name: value.detach().cpu()
            for name, value in aggregator.state_dict().items()
        },
        "teacher_checkpoint_sha256": args.checkpoint_sha256,
        "teacher_caches": [str(path) for path in cache_dirs],
        "warmup_checkpoint": str(args.warmup_checkpoint.resolve()),
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "split_kind": split_kind,
        "validation_episodes": sorted(validation_episodes),
        "training_args": vars(args),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "initial_final_validation": initial_final_validation,
        "final_final_validation": final_final_validation,
        "residual_reparameterization": residual_reparameterization,
        "transition_replay": {
            "probability": args.gripper_transition_replay_probability,
            "threshold": args.gripper_transition_threshold,
            "training_record_indices": transition_training_indices,
            "training_records": len(transition_training_indices),
            "steps": transition_replay_steps,
        },
    }
    checkpoint_path = args.output_dir / "efa_frozen_a1_distilled.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "scope": "real_frozen_a1_multi_layer_action_distillation",
        "teacher_kind": "a1_early_exit_fixed_cached_exit_layer",
        "teacher_checkpoint_sha256": args.checkpoint_sha256,
        "cache_schema_version": VISION_TEACHER_CACHE_SCHEMA_VERSION,
        "cache_records": len(dataset),
        "cache_dirs": [str(path) for path in cache_dirs],
        "supervision": args.supervision,
        "split_kind": split_kind,
        "validation_episodes": sorted(validation_episodes),
        "training_records": len(training_indices),
        "validation_records": len(validation_indices),
        "steps": args.steps,
        "fm_steps": args.fm_steps,
        "amp_enabled": amp_enabled,
        "activation_checkpointing": activation_checkpointing,
        "residual_reparameterization": residual_reparameterization,
        "transition_replay": checkpoint["transition_replay"],
        "loss_weights": loss_kwargs,
        "full_token_replay": replay,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "initial_final_validation": initial_final_validation,
        "final_final_validation": final_final_validation,
        "gradient_observed": gradient_observed,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "training_seconds": training_seconds,
        "peak_cuda_memory_bytes": peak_memory,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "metrics_path": str(metrics_path),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
