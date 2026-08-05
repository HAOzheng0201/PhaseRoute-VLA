"""Evaluate whether causal phase features predict EFA/full-token action drift.

The analysis reconstructs PhaseEstimator state on the exact cached baseline
trajectory used by paired action replay.  Model selection uses nested
leave-one-task-out ridge regression so the held-out task never participates in
feature normalization, regularization selection, or parameter fitting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.phase_estimator import (  # noqa: E402
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-result", type=Path, action="append", required=True
    )
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--aggregator-name", default="m410")
    parser.add_argument("--target", default="mae")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--call-normalizer", type=float, default=40.0)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 1.0, 10.0, 100.0),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ridge_fit(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    if features.ndim != 2 or targets.shape != (features.shape[0],):
        raise ValueError("ridge inputs have incompatible shapes")
    if features.shape[0] < 2 or alpha < 0 or not math.isfinite(alpha):
        raise ValueError("ridge fit requires samples and finite nonnegative alpha")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(features.shape[0]), normalized))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = design.T @ design + float(alpha) * penalty
    weights = np.linalg.solve(gram, design.T @ targets)
    return {"mean": mean, "scale": scale, "weights": weights}


def _ridge_predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    normalized = (features - model["mean"]) / model["scale"]
    design = np.column_stack((np.ones(features.shape[0]), normalized))
    return design @ model["weights"]


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("target and prediction must be aligned vectors")
    residual = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target.size > 1 and target.std() > 0 and prediction.std() > 0
        else 0.0
    )
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "pearson": correlation,
        "r2": 1.0 - float(np.square(residual).sum()) / denominator
        if denominator > 0
        else 0.0,
    }


def _select_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    alphas: Iterable[float],
) -> tuple[float, dict[str, float]]:
    unique_groups = sorted(set(int(value) for value in groups))
    if len(unique_groups) < 2:
        raise ValueError("alpha selection requires at least two training groups")
    scores: dict[str, float] = {}
    for alpha in alphas:
        errors = []
        for held_group in unique_groups:
            train = groups != held_group
            test = ~train
            model = _ridge_fit(features[train], targets[train], float(alpha))
            prediction = _ridge_predict(model, features[test])
            errors.extend(np.abs(prediction - targets[test]).tolist())
        scores[str(float(alpha))] = statistics.fmean(errors)
    best = min((value, float(name)) for name, value in scores.items())[1]
    return best, scores


def nested_group_ridge_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    alphas: Iterable[float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.empty_like(targets, dtype=np.float64)
    folds = []
    for held_group in sorted(set(int(value) for value in groups)):
        train = groups != held_group
        test = ~train
        alpha, inner_scores = _select_alpha(
            features[train], targets[train], groups[train], alphas
        )
        model = _ridge_fit(features[train], targets[train], alpha)
        prediction[test] = _ridge_predict(model, features[test])
        folds.append(
            {
                "held_task": held_group,
                "training_records": int(train.sum()),
                "test_records": int(test.sum()),
                "selected_alpha": alpha,
                "inner_mae_by_alpha": inner_scores,
                "metrics": _regression_metrics(targets[test], prediction[test]),
            }
        )
    return prediction, folds


def _load_replay_labels(
    paths: list[Path],
    aggregator_name: str,
    target_name: str,
) -> tuple[dict[tuple[int, int], float], list[dict[str, Any]]]:
    labels = {}
    replays = []
    for path in paths:
        replay = json.loads(path.read_text(encoding="utf-8"))
        if replay.get("status") != "PASS":
            raise ValueError("paired replay result is not PASS")
        for row in replay["rows"]:
            key = (int(row["task_id"]), int(row["step_id"]))
            if key in labels:
                raise ValueError(f"duplicate paired replay label for {key}")
            comparison = row["methods"][aggregator_name]["vs_full_token"]
            labels[key] = float(comparison[target_name])
        replays.append(replay)
    return labels, replays


def _load_entries(cache_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    entries = []
    for cache_dir in cache_dirs:
        manifest = cache_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append((cache_dir, json.loads(line)))
    entries.sort(
        key=lambda item: (
            int(item[1]["task_id"]),
            str(item[1]["episode_id"]),
            int(item[1]["step_id"]),
        )
    )
    return entries


def _phase_features(
    entries: list[tuple[Path, dict[str, Any]]],
    checkpoint_path: Path,
    history_len: int,
    call_normalizer: float,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "phase-route-vla.phase-estimator-checkpoint.v1":
        raise ValueError("unexpected PhaseEstimator checkpoint schema")
    config = PhaseEstimatorConfig(**checkpoint["model_config"])
    estimator = PhaseStateEstimator(config)
    estimator.load_state_dict(checkpoint["model_state_dict"])
    estimator.eval()
    histories: dict[str, deque[tuple[np.ndarray, np.ndarray]]] = defaultdict(
        lambda: deque(maxlen=history_len)
    )
    call_indices: dict[str, int] = defaultdict(int)
    rows = []
    with torch.inference_mode():
        for cache_dir, record in entries:
            episode_id = str(record["episode_id"])
            call_indices[episode_id] += 1
            with np.load(cache_dir / record["array_path"]) as shard:
                projected = shard["projected_features"].astype(np.float32)
                positions = shard["image_input_idx"]
                valid = positions >= 0
                if not valid.any():
                    raise ValueError("cache call contains no valid visual tokens")
                visual_summary = projected[valid].mean(axis=0)
                instruction = shard["instruction_summary"].astype(np.float32)
                proprio = shard["normalized_proprio"].astype(np.float32)
                action = shard["teacher_normalized_action"].astype(np.float32)

            history = histories[episode_id]
            proprio_history = np.zeros(
                (1, history_len, config.proprio_dim), dtype=np.float32
            )
            action_history = np.zeros(
                (1, history_len, config.action_horizon, config.action_dim),
                dtype=np.float32,
            )
            history_mask = np.zeros((1, history_len), dtype=np.bool_)
            start = history_len - len(history)
            for offset, (past_proprio, past_action) in enumerate(history, start=start):
                proprio_history[0, offset] = past_proprio
                action_history[0, offset] = past_action
                history_mask[0, offset] = True
            state = estimator(
                visual_summary=torch.from_numpy(visual_summary[None]),
                instruction_summary=torch.from_numpy(instruction[None]),
                current_proprio=torch.from_numpy(proprio[None]),
                proprio_history=torch.from_numpy(proprio_history),
                proprio_history_mask=torch.from_numpy(history_mask),
                action_history=torch.from_numpy(action_history),
                action_history_mask=torch.from_numpy(history_mask),
            )
            stage = state.stage_embedding[0].detach().float().numpy()
            scalars = np.array(
                [
                    call_indices[episode_id] / call_normalizer,
                    float(state.progress[0, 0]),
                    float(state.boundary_prob[0, 0]),
                    float(state.uncertainty[0, 0]),
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "task_id": int(record["task_id"]),
                    "step_id": int(record["step_id"]),
                    "episode_id": episode_id,
                    "call_index": call_indices[episode_id],
                    "scalars": scalars,
                    "stage_embedding": stage.astype(np.float64),
                }
            )
            history.append((proprio.copy(), action.copy()))
    return rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.history_len < 1 or args.call_normalizer <= 0:
        raise ValueError("history-len and call-normalizer must be positive")
    if not args.alphas or any(alpha < 0 for alpha in args.alphas):
        raise ValueError("alphas must be nonnegative")
    replay_paths = [path.resolve() for path in args.replay_result]
    labels, replays = _load_replay_labels(
        replay_paths, args.aggregator_name, args.target
    )
    teacher_hashes = {
        str(replay["teacher_checkpoint_sha256"]) for replay in replays
    }
    if len(teacher_hashes) != 1:
        raise ValueError("paired replay teacher checkpoints do not match")
    cache_dirs = [path.resolve() for path in args.cache_dir]
    entries = _load_entries(cache_dirs)
    rows = _phase_features(
        entries,
        args.phase_checkpoint.resolve(),
        args.history_len,
        args.call_normalizer,
    )
    if {(row["task_id"], row["step_id"]) for row in rows} != set(labels):
        raise ValueError("phase cache calls and replay labels are not exactly aligned")
    groups = np.array([row["task_id"] for row in rows], dtype=np.int64)
    targets = np.array(
        [labels[(row["task_id"], row["step_id"])] for row in rows],
        dtype=np.float64,
    )
    feature_sets = {
        "uncertainty": np.stack([row["scalars"][[3]] for row in rows]),
        "phase_scalars": np.stack([row["scalars"] for row in rows]),
        "phase_stage": np.stack(
            [np.concatenate((row["scalars"], row["stage_embedding"])) for row in rows]
        ),
    }
    analyses = {}
    all_predictions = {}
    for name, features in feature_sets.items():
        prediction, folds = nested_group_ridge_predictions(
            features, targets, groups, args.alphas
        )
        deployment_alpha, deployment_cv = _select_alpha(
            features, targets, groups, args.alphas
        )
        deployment_model = _ridge_fit(features, targets, deployment_alpha)
        analyses[name] = {
            "feature_dim": int(features.shape[1]),
            "nested_loto_metrics": _regression_metrics(targets, prediction),
            "folds": folds,
            "deployment_alpha": deployment_alpha,
            "deployment_loto_mae_by_alpha": deployment_cv,
            "deployment_model": {
                key: value.tolist() for key, value in deployment_model.items()
            },
        }
        all_predictions[name] = prediction

    constant_prediction = np.empty_like(targets)
    for task_id in sorted(set(groups.tolist())):
        train = groups != task_id
        constant_prediction[~train] = targets[train].mean()
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "task_id": row["task_id"],
                "step_id": row["step_id"],
                "call_index": row["call_index"],
                "target": float(targets[index]),
                "phase_progress": float(row["scalars"][1]),
                "phase_boundary_prob": float(row["scalars"][2]),
                "phase_uncertainty": float(row["scalars"][3]),
                "predictions": {
                    name: float(prediction[index])
                    for name, prediction in all_predictions.items()
                },
            }
        )
    result = {
        "status": "PASS",
        "scope": "m416_nested_loto_phase_action_drift_risk",
        "replay_results": [str(path) for path in replay_paths],
        "replay_result_sha256": {
            str(path): _file_sha256(path) for path in replay_paths
        },
        "replay_teacher_checkpoint_sha256": next(iter(teacher_hashes)),
        "aggregator_name": args.aggregator_name,
        "target": args.target,
        "phase_checkpoint": str(args.phase_checkpoint.resolve()),
        "phase_checkpoint_sha256": _file_sha256(args.phase_checkpoint.resolve()),
        "cache_dirs": [str(path) for path in cache_dirs],
        "records": len(rows),
        "tasks": sorted(set(groups.tolist())),
        "history_len": args.history_len,
        "call_normalizer": args.call_normalizer,
        "alphas": [float(alpha) for alpha in args.alphas],
        "constant_nested_loto_metrics": _regression_metrics(
            targets, constant_prediction
        ),
        "feature_analyses": analyses,
        "rows": output_rows,
    }
    finite = all(
        math.isfinite(float(value))
        for analysis in analyses.values()
        for value in analysis["nested_loto_metrics"].values()
    )
    if not finite:
        result["status"] = "FAIL"
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
