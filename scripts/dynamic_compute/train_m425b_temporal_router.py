"""Fit M4.25b routers without evaluating sealed episode4/5 predictions."""

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
    route_metrics,
    sequential_routes,
)
from a1.vla.dynamic_compute.temporal_route_features import (  # noqa: E402
    M425B_FEATURE_SCHEMA_VERSION,
)
from a1.vla.dynamic_compute.temporal_route_router import (  # noqa: E402
    M425B_VARIANTS,
    FeaturePreprocessor,
    TemporalRouteModel,
    fit_processed_pca_logistic,
)


EXPECTED_SCOPE = "m425b_temporal_route_feature_table"
DEV_EPISODES = (0, 1, 2)
CALIBRATION_EPISODES = (3,)
TEST_EPISODES = (4, 5)


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
    path: Path, checkpoint_sha256: str, phase_checkpoint_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = path.resolve()
    result = json.loads(source.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS"
        or result.get("scope") != EXPECTED_SCOPE
        or result.get("schema_version") != M425B_FEATURE_SCHEMA_VERSION
        or result.get("checkpoint_sha256") != checkpoint_sha256
        or result.get("phase_checkpoint_sha256") != phase_checkpoint_sha256
        or not result.get("data_sufficient")
        or not all(bool(value) for value in result.get("local_checks", {}).values())
    ):
        raise ValueError("M4.25b feature result failed frozen checks")
    arrays_path = Path(result["arrays_path"])
    if sha256_file(arrays_path) != result.get("arrays_sha256"):
        raise ValueError("M4.25b feature array SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as source_arrays:
        arrays = {name: source_arrays[name].copy() for name in source_arrays.files}
    required = {
        "layer11_hidden",
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
        raise ValueError("M4.25b feature arrays have inconsistent row counts")
    if set(arrays["episode_index"].tolist()) != set(range(6)):
        raise ValueError("M4.25b episode grid differs")
    if set(arrays["task_id"].tolist()) != set(range(10)):
        raise ValueError("M4.25b task grid differs")
    if not set(np.unique(arrays["teacher_route"]).tolist()).issubset({11, 13, 27}):
        raise ValueError("M4.25b teacher route grid differs")
    return arrays, {
        "path": str(source),
        "sha256": sha256_file(source),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": result["arrays_sha256"],
        "records": rows,
        "role_summaries": result["role_summaries"],
    }


def _fit_head(
    arrays: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    *,
    variant: str,
    layer: int,
    labels: np.ndarray,
    pca_rank: int,
    l2: float,
    max_iter: int,
    eps: float,
):
    preprocessor = FeaturePreprocessor.fit(
        arrays, fit_mask, variant=variant, layer_norm_eps=eps
    )
    features = preprocessor.transform(arrays, layer=layer)
    head = fit_processed_pca_logistic(
        features[fit_mask],
        labels[fit_mask],
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
    )
    return preprocessor, head, features


def fit_variant(
    arrays: Mapping[str, np.ndarray],
    *,
    variant: str,
    pca_rank: int,
    l2: float,
    max_iter: int,
    eps: float,
) -> tuple[TemporalRouteModel, dict[str, Any]]:
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_index"], dtype=np.int64)
    group = task * 10 + episode
    dev = np.isin(episode, DEV_EPISODES)
    calibration = np.isin(episode, CALIBRATION_EPISODES)
    if set(np.unique(group[dev]).tolist()) != {
        task_id * 10 + episode_id
        for task_id in range(10)
        for episode_id in DEV_EPISODES
    }:
        raise ValueError("development episode groups are incomplete")
    p11_oof = np.full(teacher.shape, np.nan, dtype=np.float64)
    p13_oof = np.full(teacher.shape, np.nan, dtype=np.float64)
    folds = []
    for fold_index, held_group in enumerate(sorted(np.unique(group[dev]).tolist())):
        held = dev & (group == held_group)
        train = dev & (group != held_group)
        labels11 = (teacher == 11).astype(np.int64)
        pp11, head11, features11 = _fit_head(
            arrays,
            train,
            variant=variant,
            layer=11,
            labels=labels11,
            pca_rank=pca_rank,
            l2=l2,
            max_iter=max_iter,
            eps=eps,
        )
        train13 = train & (teacher >= 13)
        labels13 = (teacher == 13).astype(np.int64)
        pp13, head13, features13 = _fit_head(
            arrays,
            train13,
            variant=variant,
            layer=13,
            labels=labels13,
            pca_rank=pca_rank,
            l2=l2,
            max_iter=max_iter,
            eps=eps,
        )
        p11_oof[held] = head11.probabilities(features11[held])
        p13_oof[held] = head13.probabilities(features13[held])
        folds.append(
            {
                "held_group": int(held_group),
                "held_task": int(held_group // 10),
                "held_episode": int(held_group % 10),
                "train_rows": int(train.sum()),
                "held_rows": int(held.sum()),
                "head11_rank": head11.pca_rank,
                "head13_rank": head13.pca_rank,
            }
        )
        print(
            f"variant={variant} OOF fold={fold_index + 1}/30 "
            f"task={held_group // 10} episode={held_group % 10}",
            flush=True,
        )
    if not np.isfinite(p11_oof[dev]).all() or not np.isfinite(p13_oof[dev]).all():
        raise RuntimeError("OOF prediction grid is incomplete")
    oof_threshold11 = calibrate_zero_false_positive(
        p11_oof[dev], (teacher[dev] == 11).astype(np.int64)
    )
    dev13 = dev & (teacher >= 13)
    oof_threshold13 = calibrate_zero_false_positive(
        p13_oof[dev13], (teacher[dev13] == 13).astype(np.int64)
    )
    oof_routes = sequential_routes(
        p11_oof[dev],
        p13_oof[dev],
        threshold11=oof_threshold11,
        threshold13=oof_threshold13,
    )
    oof_metrics = route_metrics(oof_routes, teacher[dev])

    labels11 = (teacher == 11).astype(np.int64)
    final_pp11, final_head11, final_features11 = _fit_head(
        arrays,
        dev,
        variant=variant,
        layer=11,
        labels=labels11,
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        eps=eps,
    )
    dev13 = dev & (teacher >= 13)
    labels13 = (teacher == 13).astype(np.int64)
    final_pp13, final_head13, final_features13 = _fit_head(
        arrays,
        dev13,
        variant=variant,
        layer=13,
        labels=labels13,
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        eps=eps,
    )
    p11_calibration = final_head11.probabilities(final_features11[calibration])
    p13_calibration = final_head13.probabilities(final_features13[calibration])
    threshold11 = calibrate_zero_false_positive(
        p11_calibration, (teacher[calibration] == 11).astype(np.int64)
    )
    calibration13 = calibration & (teacher >= 13)
    if not np.any(calibration13 & (teacher == 27)):
        raise ValueError("calibration set contains no route27 negative for head13")
    p13_calibration_eligible = final_head13.probabilities(
        final_features13[calibration13]
    )
    threshold13 = calibrate_zero_false_positive(
        p13_calibration_eligible,
        (teacher[calibration13] == 13).astype(np.int64),
    )
    calibration_routes = sequential_routes(
        p11_calibration,
        p13_calibration,
        threshold11=threshold11,
        threshold13=threshold13,
    )
    calibration_metrics = route_metrics(
        calibration_routes, teacher[calibration]
    )
    model = TemporalRouteModel(
        variant,
        final_pp11,
        final_pp13,
        final_head11,
        final_head13,
        threshold11,
        threshold13,
    )
    report = {
        "variant": variant,
        "development_rows": int(dev.sum()),
        "calibration_rows": int(calibration.sum()),
        "sealed_test_rows_not_evaluated": int(np.isin(episode, TEST_EPISODES).sum()),
        "oof_threshold11": oof_threshold11,
        "oof_threshold13": oof_threshold13,
        "oof_metrics": oof_metrics,
        "oof_folds": folds,
        "calibration_threshold11": threshold11,
        "calibration_threshold13": threshold13,
        "calibration_metrics": calibration_metrics,
        "calibration_probability_ranges": {
            "p11_min": float(p11_calibration.min()),
            "p11_max": float(p11_calibration.max()),
            "p13_min": float(p13_calibration.min()),
            "p13_max": float(p13_calibration.max()),
        },
        "processed_feature_dim11": int(final_head11.weight.size),
        "processed_feature_dim13": int(final_head13.weight.size),
        "pca_rank11": final_head11.pca_rank,
        "pca_rank13": final_head13.pca_rank,
    }
    return model, report


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.pca_rank != 64 or args.l2 != 1.0 or args.max_iter != 100:
        raise ValueError("M4.25b freezes PCA=64, L2=1.0 and max_iter=100")
    arrays, source = load_feature_table(
        args.feature_result,
        args.checkpoint_sha256,
        args.phase_checkpoint_sha256,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    analyses = {}
    checkpoint_files = {}
    for variant in M425B_VARIANTS:
        model, analysis = fit_variant(
            arrays,
            variant=variant,
            pca_rank=args.pca_rank,
            l2=args.l2,
            max_iter=args.max_iter,
            eps=args.layer_norm_eps,
        )
        if (
            int(analysis["oof_metrics"]["false_shallow"]) != 0
            or int(analysis["calibration_metrics"]["false_shallow"]) != 0
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
        reloaded = TemporalRouteModel.load(checkpoint_path)
        calibration = np.isin(arrays["episode_index"], CALIBRATION_EPISODES)
        calibration_arrays = {
            name: np.asarray(value)[calibration] for name, value in arrays.items()
        }
        expected = model.probabilities(calibration_arrays)
        actual = reloaded.probabilities(calibration_arrays)
        if not (
            np.array_equal(expected[0], actual[0])
            and np.array_equal(expected[1], actual[1])
        ):
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
        "scope": "m425b_grouped_oof_and_calibration_fit",
        "sealed_test_evaluated": False,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "feature_source": source,
        "development_episodes": list(DEV_EPISODES),
        "calibration_episodes": list(CALIBRATION_EPISODES),
        "sealed_test_episodes": list(TEST_EPISODES),
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
