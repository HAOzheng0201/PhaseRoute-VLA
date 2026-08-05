"""Fit M4.26 route13/27 routers without reading sealed episode4/5 metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.causal_route_router import (  # noqa: E402
    calibrate_zero_false_positive,
)
from a1.vla.dynamic_compute.risk_route13_router import (  # noqa: E402
    M426A_FEATURE_SCHEMA_VERSION,
    M426_FEATURE_SCHEMA_VERSION,
    M426_VARIANTS,
    RiskRoute13Model,
    fit_route13_head,
    route13_metrics,
    route13_or_27,
)


EXPECTED_SCOPE = "m426_temporal_route_feature_table"
DEV_EPISODES = (0, 1, 2)
CALIBRATION_EPISODES = (3,)
TEST_EPISODES = (4, 5)
M426A_EXPECTED_SCOPE = "m426a_temporal_route_feature_table"
M426A_DEV_EPISODES = (0, 1, 2)
M426A_CALIBRATION_EPISODES = (3, 4)
M426A_TEST_EPISODES = (5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--layer-norm-eps", type=float, default=1e-6)
    parser.add_argument("--protocol", choices=("m426", "m426a"), default="m426")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_feature_table(
    path: Path,
    checkpoint_sha256: str,
    phase_checkpoint_sha256: str,
    *,
    protocol: str = "m426",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if protocol == "m426":
        expected_scope = EXPECTED_SCOPE
        expected_schema = M426_FEATURE_SCHEMA_VERSION
        expected_episode_grid = set(range(6))
    elif protocol == "m426a":
        expected_scope = M426A_EXPECTED_SCOPE
        expected_schema = M426A_FEATURE_SCHEMA_VERSION
        expected_episode_grid = set(range(7))
    else:
        raise ValueError("unsupported M4.26 data protocol")
    source = path.resolve()
    result = json.loads(source.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS"
        or result.get("scope") != expected_scope
        or result.get("schema_version") != expected_schema
        or result.get("checkpoint_sha256") != checkpoint_sha256
        or result.get("phase_checkpoint_sha256") != phase_checkpoint_sha256
        or not result.get("data_sufficient")
        or not all(bool(value) for value in result.get("local_checks", {}).values())
    ):
        raise ValueError("M4.26 feature result failed frozen checks")
    arrays_path = Path(result["arrays_path"])
    if sha256_file(arrays_path) != result.get("arrays_sha256"):
        raise ValueError("M4.26 feature array SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as source_arrays:
        arrays = {name: source_arrays[name].copy() for name in source_arrays.files}
    required = {
        "layer13_hidden",
        "current_proprio",
        "proprio_history",
        "action_history",
        "history_mask",
        "phase_stage",
        "phase_scalars",
        "step_feature",
        "task_id",
        "episode_index",
        "step_id",
        "call_index",
        "teacher_route",
        "identity_sha256",
    }
    if not required.issubset(arrays):
        raise KeyError(f"feature table misses arrays: {sorted(required - set(arrays))}")
    rows = int(result["records"])
    if any(arrays[name].shape[0] != rows for name in required):
        raise ValueError("M4.26 feature arrays have inconsistent row counts")
    if set(arrays["episode_index"].tolist()) != expected_episode_grid:
        raise ValueError("M4.26 episode grid differs")
    if set(arrays["task_id"].tolist()) != set(range(10)):
        raise ValueError("M4.26 task grid differs")
    if not set(np.unique(arrays["teacher_route"]).tolist()).issubset({11, 13, 27}):
        raise ValueError("M4.26 teacher route grid differs")
    return arrays, {
        "path": str(source),
        "sha256": sha256_file(source),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": result["arrays_sha256"],
        "records": rows,
        "role_summaries": result["role_summaries"],
    }


def fit_variant(
    arrays: Mapping[str, np.ndarray],
    *,
    variant: str,
    pca_rank: int,
    l2: float,
    max_iter: int,
    eps: float,
    development_episodes: tuple[int, ...] = DEV_EPISODES,
    calibration_episodes: tuple[int, ...] = CALIBRATION_EPISODES,
    test_episodes: tuple[int, ...] = TEST_EPISODES,
) -> tuple[RiskRoute13Model, dict[str, Any]]:
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_index"], dtype=np.int64)
    group = task * 10 + episode
    dev = np.isin(episode, development_episodes)
    calibration = np.isin(episode, calibration_episodes)
    expected_groups = {
        task_id * 10 + episode_id
        for task_id in range(10)
        for episode_id in development_episodes
    }
    if set(np.unique(group[dev]).tolist()) != expected_groups:
        raise ValueError("development episode groups are incomplete")
    probability_oof = np.full(teacher.shape, np.nan, dtype=np.float64)
    folds = []
    for fold_index, held_group in enumerate(sorted(expected_groups)):
        held = dev & (group == held_group)
        train = dev & (group != held_group)
        _, head, features = fit_route13_head(
            arrays,
            train,
            variant=variant,
            pca_rank=pca_rank,
            l2=l2,
            max_iter=max_iter,
            layer_norm_eps=eps,
        )
        probability_oof[held] = head.probabilities(features[held])
        folds.append(
            {
                "held_group": int(held_group),
                "held_task": int(held_group // 10),
                "held_episode": int(held_group % 10),
                "train_rows": int(train.sum()),
                "held_rows": int(held.sum()),
                "pca_rank": int(head.pca_rank),
            }
        )
        print(
            f"variant={variant} OOF fold={fold_index + 1}/30 "
            f"task={held_group // 10} episode={held_group % 10}",
            flush=True,
        )
    if not np.isfinite(probability_oof[dev]).all():
        raise RuntimeError("M4.26 OOF prediction grid is incomplete")
    labels = (teacher <= 13).astype(np.int64)
    if not np.any(dev & (teacher == 27)):
        raise ValueError("development has no route27 negative")
    oof_threshold = calibrate_zero_false_positive(
        probability_oof[dev], labels[dev]
    )
    oof_routes = route13_or_27(probability_oof[dev], threshold=oof_threshold)
    oof_metrics = route13_metrics(oof_routes, teacher[dev])

    preprocessor, head, features = fit_route13_head(
        arrays,
        dev,
        variant=variant,
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        layer_norm_eps=eps,
    )
    if not np.any(calibration & (teacher == 27)):
        raise ValueError("calibration has no route27 negative")
    probability_calibration = head.probabilities(features[calibration])
    threshold = calibrate_zero_false_positive(
        probability_calibration, labels[calibration]
    )
    calibration_routes = route13_or_27(
        probability_calibration, threshold=threshold
    )
    calibration_metrics = route13_metrics(
        calibration_routes, teacher[calibration]
    )
    model = RiskRoute13Model(variant, preprocessor, head, threshold)
    report = {
        "variant": variant,
        "development_rows": int(dev.sum()),
        "calibration_rows": int(calibration.sum()),
        "sealed_test_rows_not_evaluated": int(
            np.isin(episode, test_episodes).sum()
        ),
        "oof_threshold": oof_threshold,
        "oof_metrics": oof_metrics,
        "oof_folds": folds,
        "calibration_threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "calibration_probability_range": {
            "min": float(probability_calibration.min()),
            "max": float(probability_calibration.max()),
        },
        "processed_feature_dim": int(head.weight.size),
        "pca_rank": int(head.pca_rank),
    }
    return model, report


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.pca_rank != 64 or args.l2 != 1.0 or args.max_iter != 100:
        raise ValueError("M4.26 freezes PCA=64, L2=1.0 and max_iter=100")
    if args.protocol == "m426":
        development_episodes = DEV_EPISODES
        calibration_episodes = CALIBRATION_EPISODES
        test_episodes = TEST_EPISODES
        fit_scope = "m426_grouped_oof_and_calibration_fit"
    else:
        development_episodes = M426A_DEV_EPISODES
        calibration_episodes = M426A_CALIBRATION_EPISODES
        test_episodes = M426A_TEST_EPISODES
        fit_scope = "m426a_grouped_oof_and_calibration_fit"
    arrays, source = load_feature_table(
        args.feature_result,
        args.checkpoint_sha256,
        args.phase_checkpoint_sha256,
        protocol=args.protocol,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    analyses = {}
    checkpoint_files = {}
    for variant in M426_VARIANTS:
        model, analysis = fit_variant(
            arrays,
            variant=variant,
            pca_rank=args.pca_rank,
            l2=args.l2,
            max_iter=args.max_iter,
            eps=args.layer_norm_eps,
            development_episodes=development_episodes,
            calibration_episodes=calibration_episodes,
            test_episodes=test_episodes,
        )
        if (
            int(analysis["oof_metrics"]["route27_false_shallow"]) != 0
            or int(analysis["calibration_metrics"]["route27_false_shallow"]) != 0
        ):
            raise RuntimeError(f"{variant} is not fail-closed before sealed test")
        checkpoint_path = args.output_dir / f"{variant}_router.npz"
        model.save(
            checkpoint_path,
            source_feature_sha256=source["sha256"],
            source_arrays_sha256=source["arrays_sha256"],
            checkpoint_sha256=args.checkpoint_sha256,
            phase_checkpoint_sha256=args.phase_checkpoint_sha256,
        )
        reloaded = RiskRoute13Model.load(checkpoint_path)
        calibration = np.isin(arrays["episode_index"], calibration_episodes)
        calibration_arrays = {
            name: np.asarray(value)[calibration] for name, value in arrays.items()
        }
        expected = model.probabilities(calibration_arrays)
        actual = reloaded.probabilities(calibration_arrays)
        if not np.array_equal(expected, actual):
            raise RuntimeError(f"{variant} checkpoint prediction roundtrip differs")
        analysis["checkpoint_path"] = str(checkpoint_path.resolve())
        analysis["checkpoint_sha256"] = sha256_file(checkpoint_path)
        analyses[variant] = analysis
        checkpoint_files[variant] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        }
    result = {
        "status": "PASS",
        "scope": fit_scope,
        "protocol": args.protocol,
        "sealed_test_evaluated": False,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "feature_source": source,
        "development_episodes": list(development_episodes),
        "calibration_episodes": list(calibration_episodes),
        "sealed_test_episodes": list(test_episodes),
        "pca_rank": args.pca_rank,
        "l2": args.l2,
        "max_iter": args.max_iter,
        "layer_norm_eps": args.layer_norm_eps,
        "checkpoint_files": checkpoint_files,
        "analyses": analyses,
    }
    result_path = args.output_dir / "fit_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "sealed_test_evaluated": False,
                "variants": {
                    name: {
                        "oof": value["oof_metrics"],
                        "calibration": value["calibration_metrics"],
                        "checkpoint_sha256": value["checkpoint_sha256"],
                    }
                    for name, value in analyses.items()
                },
                "fit_result_sha256": sha256_file(result_path),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
