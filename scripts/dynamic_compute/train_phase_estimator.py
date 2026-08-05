"""Train and evaluate the observer-only M2 PhaseStateEstimator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Dict

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseLossConfig,
    PhaseStateEstimator,
    phase_estimator_loss,
)
from a1.vla.dynamic_compute.phase_training import (
    ESTIMATOR_INPUT_NAMES,
    SPLIT_IDS,
    baseline_metrics,
    boundary_metrics,
    load_phase_dataset,
    make_torch_batch,
    progress_metrics,
    select_f1_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: batch[name] for name in ESTIMATOR_INPUT_NAMES}


def compute_loss(
    model: PhaseStateEstimator,
    batch: Dict[str, torch.Tensor],
    loss_config: PhaseLossConfig,
) -> Dict[str, torch.Tensor]:
    state = model(**model_inputs(batch))
    return phase_estimator_loss(
        state,
        progress_target=batch["progress_target"],
        boundary_target=batch["boundary_target"],
        episode_index=batch["episode_index"],
        call_index=batch["call_index"],
        config=loss_config,
    )


@torch.inference_mode()
def predict(
    model: PhaseStateEstimator,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    model.eval()
    state = model(**model_inputs(batch))
    return {
        "progress": state.progress.detach().cpu().numpy(),
        "boundary_prob": state.boundary_prob.detach().cpu().numpy(),
        "uncertainty": state.uncertainty.detach().cpu().numpy(),
    }


def split_metrics(
    prediction: Dict[str, np.ndarray],
    batch: Dict[str, torch.Tensor],
    calibrated_threshold: float,
) -> Dict[str, Any]:
    progress_target = batch["progress_target"].detach().cpu().numpy()
    boundary_target = batch["boundary_target"].detach().cpu().numpy()
    return {
        "records": int(progress_target.shape[0]),
        "progress": progress_metrics(prediction["progress"], progress_target),
        "boundary_fixed_0_5": boundary_metrics(
            prediction["boundary_prob"], boundary_target, 0.5
        ),
        "boundary_validation_calibrated": boundary_metrics(
            prediction["boundary_prob"], boundary_target, calibrated_threshold
        ),
        "uncertainty_mean": float(np.mean(prediction["uncertainty"])),
        "uncertainty_max": float(np.max(prediction["uncertainty"])),
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("Invalid optimizer hyperparameters")
    checkpoint_path = args.output_dir / "phase_estimator.pt"
    result_path = args.output_dir / "result.json"
    history_path = args.output_dir / "training_history.json"
    predictions_path = args.output_dir / "predictions.npz"
    for path in (checkpoint_path, result_path, history_path, predictions_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite training artifact: {path}")

    set_determinism(args.seed)
    device = resolve_device(args.device)
    bundle = load_phase_dataset(args.dataset, args.metadata)
    batches = {
        split_name: make_torch_batch(bundle, split_name, device)
        for split_name in SPLIT_IDS
    }
    arrays = bundle.arrays
    model_config = PhaseEstimatorConfig(
        visual_summary_dim=arrays["visual_summary"].shape[-1],
        instruction_dim=arrays["instruction_summary"].shape[-1],
        proprio_dim=arrays["current_proprio"].shape[-1],
        action_horizon=arrays["action_history"].shape[-2],
        action_dim=arrays["action_history"].shape[-1],
    )
    loss_config = PhaseLossConfig()
    model = PhaseStateEstimator(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation = float("inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = compute_loss(model, batches["train"], loss_config)
        train_losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_losses = compute_loss(
                model, batches["validation"], loss_config
            )
        train_values = {
            name: float(value.detach().cpu())
            for name, value in train_losses.items()
        }
        validation_values = {
            name: float(value.detach().cpu())
            for name, value in validation_losses.items()
        }
        history.append(
            {
                "epoch": epoch,
                "train": train_values,
                "validation": validation_values,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            }
        )
        validation_total = validation_values["total"]
        if validation_total < best_validation - 1e-8:
            best_validation = validation_total
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch} train_total={train_values['total']:.6f} "
                f"validation_total={validation_total:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    predictions = {
        split_name: predict(model, batch)
        for split_name, batch in batches.items()
    }
    validation_threshold = select_f1_threshold(
        predictions["validation"]["boundary_prob"],
        batches["validation"]["boundary_target"].detach().cpu().numpy(),
    )
    metrics = {
        split_name: split_metrics(
            predictions[split_name], batches[split_name], validation_threshold
        )
        for split_name in SPLIT_IDS
    }

    train_indices = arrays["split"] == SPLIT_IDS["train"]
    baselines = {}
    for offset, (split_name, split_id) in enumerate(SPLIT_IDS.items()):
        split_indices = arrays["split"] == split_id
        baselines[split_name] = baseline_metrics(
            arrays["progress_target"][train_indices],
            arrays["boundary_target"][train_indices],
            arrays["progress_target"][split_indices],
            arrays["boundary_target"][split_indices],
            seed=args.seed + offset,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "phase-route-vla.phase-estimator-checkpoint.v1",
            "model_state_dict": best_state,
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
            "dataset_sha256": bundle.dataset_sha256,
            "best_epoch": best_epoch,
            "validation_boundary_threshold": validation_threshold,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        predictions_path,
        **{
            f"{split_name}_{name}": value
            for split_name, split_predictions in predictions.items()
            for name, value in split_predictions.items()
        },
    )
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "observer_only": True,
        "controls_early_exit": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "dataset": str(bundle.dataset_path),
        "dataset_sha256": bundle.dataset_sha256,
        "split_records": bundle.metadata["split_records"],
        "split_episodes": bundle.metadata["split_episodes"],
        "model_config": asdict(model_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss_config": asdict(loss_config),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "validation_boundary_threshold": validation_threshold,
        "metrics": metrics,
        "baselines": baselines,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "predictions": str(predictions_path),
        },
        "limitations": [
            "Weak labels are heuristics derived from A1 rollout telemetry.",
            "Validation chooses the boundary threshold; test data is never used for calibration.",
            "M2 is observer-only and does not change early-exit or action decisions.",
        ],
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
