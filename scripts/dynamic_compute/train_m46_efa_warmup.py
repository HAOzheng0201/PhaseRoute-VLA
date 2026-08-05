"""Warm up learnable EFA on projected-feature teacher cache.

This is deliberately an auxiliary pretraining stage.  It does not claim that
the disposable action readout is equivalent to distillation through frozen A1.
"""

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
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.learnable_vision_aggregation import (
    LearnableEFAWarmupModel,
    LearnableVisionAggregationConfig,
    efa_warmup_loss,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (
    SUPPORTED_VISION_TEACHER_CACHE_SCHEMA_VERSIONS,
)


CHECKPOINT_SCHEMA_VERSION = "phase-route-vla.learnable-efa-warmup.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


class VisionTeacherDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        manifest_path = cache_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.records) < 2:
            raise ValueError("warmup requires at least two cached calls")
        checkpoint_hashes = {record.get("checkpoint_sha256") for record in self.records}
        teacher_kinds = {record.get("teacher_kind") for record in self.records}
        if len(checkpoint_hashes) != 1 or None in checkpoint_hashes:
            raise ValueError("cache mixes checkpoint fingerprints")
        if teacher_kinds != {"a1_early_exit"}:
            raise ValueError("warmup expects only a1_early_exit teacher records")
        if any(
            record.get("schema_version")
            not in SUPPORTED_VISION_TEACHER_CACHE_SCHEMA_VERSIONS
            for record in self.records
        ):
            raise ValueError("cache contains an unsupported schema")
        self.checkpoint_sha256 = next(iter(checkpoint_hashes))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        shard_path = self.cache_dir / record["array_path"]
        with np.load(shard_path) as shard:
            return {
                "projected_features": torch.from_numpy(
                    shard["projected_features"].astype(np.float32)
                ),
                "image_input_idx": torch.from_numpy(
                    shard["image_input_idx"].astype(np.int64)
                ),
                "instruction_summary": torch.from_numpy(
                    shard["instruction_summary"].astype(np.float32)
                ),
                "normalized_proprio": torch.from_numpy(
                    shard["normalized_proprio"].astype(np.float32)
                ),
                "teacher_action": torch.from_numpy(
                    shard["teacher_normalized_action"].astype(np.float32)
                ),
            }


def _move(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device=device, non_blocking=True) for name, value in batch.items()}


def _forward_loss(model, batch, amp_enabled):
    device_type = batch["projected_features"].device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=amp_enabled and device_type == "cuda",
    ):
        output = model(
            batch["projected_features"],
            batch["image_input_idx"],
            batch["instruction_summary"],
            batch["normalized_proprio"],
        )
        loss, parts = efa_warmup_loss(
            output,
            batch["projected_features"],
            batch["image_input_idx"],
            batch["teacher_action"],
        )
    anchor_cosine = torch.nn.functional.cosine_similarity(
        output.aggregation.aggregated.features.float(),
        output.aggregation.anchor_features.float(),
        dim=-1,
    ).mean()
    attention = output.aggregation.attention_weights
    batch_size, query_count, _ = attention.shape
    crops = batch["image_input_idx"].shape[1]
    patches = batch["image_input_idx"].shape[2]
    per_crop_attention = attention.reshape(
        batch_size, query_count, crops, patches
    ).sum(dim=(1, 3))
    valid_crops = (batch["image_input_idx"] >= 0).any(dim=-1)
    minimum_valid_crop_attention = per_crop_attention[valid_crops].min()
    metrics = {
        **{name: float(value.float().cpu()) for name, value in parts.items()},
        "anchor_cosine": float(anchor_cosine.detach().cpu()),
        "minimum_valid_crop_attention": float(
            minimum_valid_crop_attention.detach().cpu()
        ),
        "residual_scale": float(
            output.aggregation.residual_scale.detach().float().cpu()
        ),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    rows = []
    for batch in loader:
        _, metrics = _forward_loss(model, _move(batch, device), amp_enabled)
        rows.append(metrics)
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }


def state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset = VisionTeacherDataset(args.cache_dir.resolve())
    sample = dataset[0]
    crops, patches, hidden_dim = sample["projected_features"].shape
    proprio_dim = sample["normalized_proprio"].numel()
    action_horizon, action_dim = sample["teacher_action"].shape
    config = LearnableVisionAggregationConfig(
        hidden_dim=hidden_dim,
        attention_dim=args.attention_dim,
        output_tokens=args.output_tokens,
        num_heads=args.num_heads,
        max_crops=crops,
        max_patches_per_crop=patches,
        min_tokens_per_crop=4,
        proprio_dim=proprio_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    model = LearnableEFAWarmupModel(config).to(device)
    initial_state_hash = state_sha256(model.state_dict())

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    validation_count = max(1, len(indices) // 5)
    validation_indices = sorted(indices[:validation_count])
    training_indices = sorted(indices[validation_count:])
    train_loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_eval_loader = DataLoader(
        Subset(dataset, training_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    amp_enabled = not args.disable_amp
    initial_train = evaluate(model, train_eval_loader, device, amp_enabled)
    initial_validation = evaluate(model, val_loader, device, amp_enabled)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metrics_path = args.output_dir / "training_metrics.jsonl"
    train_iterator = iter(train_loader)
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _forward_loss(model, batch, amp_enabled)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step()
            row: dict[str, Any] = {
                "step": step,
                "learning_rate": args.learning_rate,
                "grad_norm": float(grad_norm.detach().cpu()),
                **metrics,
            }
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()

    final_train = evaluate(model, train_eval_loader, device, amp_enabled)
    final_validation = evaluate(model, val_loader, device, amp_enabled)
    final_state_hash = state_sha256(model.state_dict())
    train_improvement = 1.0 - final_train["total"] / initial_train["total"]
    validation_improvement = (
        1.0 - final_validation["total"] / initial_validation["total"]
    )
    status_ok = (
        final_state_hash != initial_state_hash
        and math.isfinite(train_improvement)
        and train_improvement > 0.05
        and final_train["minimum_valid_crop_attention"] > 0.0
        and final_train["anchor_cosine"] > 0.95
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": asdict(config),
        "model_state_dict": {
            name: value.detach().to(device="cpu")
            for name, value in model.state_dict().items()
        },
        "teacher_checkpoint_sha256": dataset.checkpoint_sha256,
        "cache_dir": str(args.cache_dir.resolve()),
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "training_args": vars(args),
        "metrics": {
            "initial_train": initial_train,
            "final_train": final_train,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "train_improvement": train_improvement,
            "validation_improvement": validation_improvement,
        },
    }
    checkpoint_path = args.output_dir / "efa_warmup.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "status": "PASS" if status_ok else "FAIL",
        "scope": "auxiliary_warmup_not_frozen_a1_action_distillation",
        "cache_dir": str(args.cache_dir.resolve()),
        "teacher_checkpoint_sha256": dataset.checkpoint_sha256,
        "records": len(dataset),
        "training_records": len(training_indices),
        "validation_records": len(validation_indices),
        "steps": args.steps,
        "device": str(device),
        "amp_enabled": amp_enabled and device.type == "cuda",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": asdict(config),
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "train_improvement": train_improvement,
        "validation_improvement": validation_improvement,
        "initial_state_sha256": initial_state_hash,
        "final_state_sha256": final_state_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics_path": str(metrics_path),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not status_ok:
        raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
