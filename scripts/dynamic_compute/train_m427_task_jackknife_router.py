"""Fit and freeze M4.27 without producing sealed episode10--14 predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from a1.vla.dynamic_compute.m427_task_jackknife_router import (  # noqa: E402
    M427_TASKS,
    TaskJackknifeRoute13Ensemble,
    aggregate_safe13_probabilities,
    calibrate_strict_negative_max,
    episode_group_risk_metrics,
    strict_route13_or_27,
    zero_error_clopper_pearson_upper,
)
from a1.vla.dynamic_compute.risk_route13_router import (  # noqa: E402
    M427_FEATURE_SCHEMA_VERSION,
    M428_FEATURE_SCHEMA_VERSION,
    RiskRoute13Model,
    fit_route13_head,
    route13_metrics,
)



@dataclass(frozen=True)
class TaskJackknifeProtocol:
    name: str
    feature_schema_version: str
    feature_scope: str
    fit_scope: str
    sealed_scope: str
    development_episodes: tuple[int, ...]
    calibration_episodes: tuple[int, ...]
    sealed_episodes: tuple[int, ...]
    minimum_route27_rows: int
    minimum_positive_groups: int
    maximum_calibration_cp_upper: float


M427_PROTOCOL = TaskJackknifeProtocol(
    name="m427",
    feature_schema_version=M427_FEATURE_SCHEMA_VERSION,
    feature_scope="m427_temporal_route_feature_table",
    fit_scope="m427_task_jackknife_fit_and_calibration",
    sealed_scope="m427_sealed_task_jackknife_evaluation",
    development_episodes=tuple(range(5)),
    calibration_episodes=tuple(range(5, 10)),
    sealed_episodes=tuple(range(10, 15)),
    minimum_route27_rows=30,
    minimum_positive_groups=19,
    maximum_calibration_cp_upper=0.15,
)
M428_PROTOCOL = TaskJackknifeProtocol(
    name="m428",
    feature_schema_version=M428_FEATURE_SCHEMA_VERSION,
    feature_scope="m428_temporal_route_feature_table",
    fit_scope="m428_task_jackknife_fit_and_calibration",
    sealed_scope="m428_sealed_task_jackknife_evaluation",
    development_episodes=tuple(range(10)),
    calibration_episodes=tuple(range(10, 20)),
    sealed_episodes=tuple(range(20, 30)),
    minimum_route27_rows=30,
    minimum_positive_groups=29,
    maximum_calibration_cp_upper=0.10,
)


def get_protocol_config(name: str) -> TaskJackknifeProtocol:
    protocols = {item.name: item for item in (M427_PROTOCOL, M428_PROTOCOL)}
    try:
        return protocols[name]
    except KeyError as error:
        raise ValueError(f"unsupported task-jackknife protocol: {name}") from error


DEV_EPISODES = M427_PROTOCOL.development_episodes
CALIBRATION_EPISODES = M427_PROTOCOL.calibration_episodes
SEALED_EPISODES = M427_PROTOCOL.sealed_episodes
EXPECTED_SCOPE = M427_PROTOCOL.feature_scope
EXPECTED_FEATURES = {
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def subset_arrays(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    selected = np.asarray(mask, dtype=np.bool_).reshape(-1)
    return {name: np.asarray(value)[selected].copy() for name, value in arrays.items()}


def load_nonsealed_features(
    result_path: Path,
    checkpoint_sha256: str,
    phase_checkpoint_sha256: str,
    protocol: TaskJackknifeProtocol = M427_PROTOCOL,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = result_path.resolve()
    result = json.loads(source.read_text(encoding="utf-8"))
    test_summary = result.get("role_summaries", {}).get("test", {})
    if (
        result.get("status") != "PASS"
        or result.get("scope") != protocol.feature_scope
        or result.get("schema_version") != protocol.feature_schema_version
        or result.get("protocol") != protocol.name
        or result.get("checkpoint_sha256") != checkpoint_sha256
        or result.get("phase_checkpoint_sha256") != phase_checkpoint_sha256
        or not result.get("data_sufficient")
        or not all(bool(value) for value in result.get("local_checks", {}).values())
        or test_summary.get("sealed") is not True
        or "teacher_distribution" in test_summary
    ):
        raise ValueError(
            f"{protocol.name.upper()} feature result failed frozen nonsealed checks"
        )
    arrays_path = Path(result["arrays_path"])
    if sha256_file(arrays_path) != result.get("arrays_sha256"):
        raise ValueError("M4.27 feature array SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as source_arrays:
        if not EXPECTED_FEATURES.issubset(source_arrays.files):
            raise KeyError("M4.27 feature table misses required arrays")
        episode_index = source_arrays["episode_index"]
        nonsealed_mask = np.isin(
            episode_index,
            protocol.development_episodes + protocol.calibration_episodes,
        )
        arrays = {
            name: source_arrays[name][nonsealed_mask].copy()
            for name in EXPECTED_FEATURES
        }
    if set(arrays["episode_index"].tolist()) != set(
        protocol.development_episodes + protocol.calibration_episodes
    ):
        raise ValueError(f"{protocol.name.upper()} nonsealed episode grid differs")
    if set(arrays["task_id"].tolist()) != set(M427_TASKS):
        raise ValueError(f"{protocol.name.upper()} task grid differs")
    if np.unique(arrays["identity_sha256"]).size != arrays["teacher_route"].size:
        raise ValueError(f"{protocol.name.upper()} nonsealed identities are duplicated")
    return arrays, {
        "path": str(source),
        "sha256": sha256_file(source),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": result["arrays_sha256"],
        "records_total": int(result["records"]),
        "nonsealed_rows": int(arrays["teacher_route"].size),
        "role_summaries": {
            "development": result["role_summaries"]["development"],
            "calibration": result["role_summaries"]["calibration"],
            "test": test_summary,
        },
    }


def fit_base_model(
    arrays: Mapping[str, np.ndarray],
    fit_mask: np.ndarray,
    *,
    pca_rank: int,
    l2: float,
    max_iter: int,
    eps: float,
) -> RiskRoute13Model:
    preprocessor, head, _ = fit_route13_head(
        arrays,
        fit_mask,
        variant="temporal_phase_step",
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        layer_norm_eps=eps,
    )
    return RiskRoute13Model("temporal_phase_step", preprocessor, head, 1.0)


def task_jackknife_fit_masks(
    task_id: np.ndarray,
    episode_index: np.ndarray,
    protocol: TaskJackknifeProtocol = M427_PROTOCOL,
) -> dict[int, np.ndarray]:
    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if tasks.shape != episodes.shape or tasks.size < 1:
        raise ValueError("task and episode arrays must be aligned and non-empty")
    development = np.isin(episodes, protocol.development_episodes)
    if set(tasks[development].tolist()) != set(M427_TASKS):
        raise ValueError(f"{protocol.name.upper()} development task grid differs")
    return {
        excluded_task: development & (tasks != excluded_task)
        for excluded_task in M427_TASKS
    }


def method_report(
    scores: np.ndarray,
    threshold: float,
    teacher: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    routes = strict_route13_or_27(scores[mask], threshold=threshold)
    group = episode_group_risk_metrics(
        routes, teacher[mask], task[mask], episode[mask]
    )
    return {
        "threshold": threshold,
        "score_range": {
            "min": float(scores[mask].min()),
            "max": float(scores[mask].max()),
        },
        "metrics": route13_metrics(routes, teacher[mask]),
        "group_risk": group,
    }


def role_data_summary(
    teacher: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    always27 = np.full(int(mask.sum()), 27, dtype=np.int64)
    groups = episode_group_risk_metrics(
        always27, teacher[mask], task[mask], episode[mask]
    )
    return {
        "rows": int(mask.sum()),
        "route27_rows": int(np.sum(teacher[mask] == 27)),
        "route27_positive_groups": int(groups["route27_positive_groups"]),
    }


def main(protocol_name: str = "m427") -> None:
    args = parse_args()
    protocol = get_protocol_config(protocol_name)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    arrays, feature_source = load_nonsealed_features(
        args.feature_result,
        args.checkpoint_sha256,
        args.phase_checkpoint_sha256,
        protocol,
    )
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_index"], dtype=np.int64)
    development = np.isin(episode, protocol.development_episodes)
    calibration = np.isin(episode, protocol.calibration_episodes)
    development_data = role_data_summary(
        teacher, task, episode, development
    )
    calibration_data = role_data_summary(
        teacher, task, episode, calibration
    )
    data_gates = {
        "development_route27_rows_at_least_30": development_data["route27_rows"]
        >= protocol.minimum_route27_rows,
        "development_positive_groups_at_least_19": development_data[
            "route27_positive_groups"
        ]
        >= protocol.minimum_positive_groups,
        "calibration_route27_rows_at_least_30": calibration_data["route27_rows"]
        >= protocol.minimum_route27_rows,
        "calibration_positive_groups_at_least_19": calibration_data[
            "route27_positive_groups"
        ]
        >= protocol.minimum_positive_groups,
    }
    if not all(data_gates.values()):
        raise ValueError(
            f"{protocol.name.upper()} feature result passed but trainer data gates differ"
        )

    fit_masks = task_jackknife_fit_masks(task, episode, protocol)
    learners = []
    learner_reports = []
    for excluded_task in M427_TASKS:
        fit_mask = fit_masks[excluded_task]
        if np.any(fit_mask & (task == excluded_task)):
            raise RuntimeError("task-jackknife exclusion failed")
        if not np.any(fit_mask & (teacher == 27)):
            raise ValueError(f"exclude-task{excluded_task} training has no route27")
        model = fit_base_model(
            arrays,
            fit_mask,
            pca_rank=args.pca_rank,
            l2=args.l2,
            max_iter=args.max_iter,
            eps=args.layer_norm_eps,
        )
        learners.append(model)
        learner_reports.append(
            {
                "excluded_task": excluded_task,
                "train_rows": int(fit_mask.sum()),
                "train_route27_rows": int(np.sum(fit_mask & (teacher == 27))),
                "excluded_task_rows_used": int(
                    np.sum(fit_mask & (task == excluded_task))
                ),
                "pca_rank": int(model.head.pca_rank),
            }
        )
        print(f"fit task-jackknife learner exclude_task={excluded_task}", flush=True)

    learner_probabilities = np.stack(
        [model.probabilities(arrays) for model in learners], axis=1
    )
    min_scores = aggregate_safe13_probabilities(
        learner_probabilities, aggregation="min"
    )
    mean_scores = aggregate_safe13_probabilities(
        learner_probabilities, aggregation="mean"
    )
    min_threshold = calibrate_strict_negative_max(
        min_scores[calibration], teacher[calibration]
    )
    mean_threshold = calibrate_strict_negative_max(
        mean_scores[calibration], teacher[calibration]
    )

    single_full = fit_base_model(
        arrays,
        development,
        pca_rank=args.pca_rank,
        l2=args.l2,
        max_iter=args.max_iter,
        eps=args.layer_norm_eps,
    )
    single_scores = single_full.probabilities(arrays)
    single_threshold = calibrate_strict_negative_max(
        single_scores[calibration], teacher[calibration]
    )

    analyses = {
        "ensemble_min": method_report(
            min_scores,
            min_threshold,
            teacher,
            task,
            episode,
            calibration,
        ),
        "ensemble_mean": method_report(
            mean_scores,
            mean_threshold,
            teacher,
            task,
            episode,
            calibration,
        ),
        "single_full": method_report(
            single_scores,
            single_threshold,
            teacher,
            task,
            episode,
            calibration,
        ),
    }
    main_analysis = analyses["ensemble_min"]
    main_group_count = int(
        main_analysis["group_risk"]["route27_positive_groups"]
    )
    main_group_errors = int(main_analysis["group_risk"]["route27_error_groups"])
    cp_upper = (
        zero_error_clopper_pearson_upper(main_group_count)
        if main_group_errors == 0
        else 1.0
    )
    main_analysis["group_risk"]["one_sided_95pct_cp_upper"] = cp_upper
    calibration_gates = {
        **data_gates,
        "calibration_route27_false_shallow_rows_zero": int(
            main_analysis["metrics"]["route27_false_shallow"]
        )
        == 0,
        "calibration_route27_error_groups_zero": main_group_errors == 0,
        "calibration_safe13_recall_at_least_15_percent": float(
            main_analysis["metrics"]["safe13_recall"]
        )
        >= 0.15,
        "calibration_predicted13_coverage_at_least_15_percent": float(
            main_analysis["metrics"]["predicted13_coverage"]
        )
        >= 0.15,
        f"calibration_group_cp_upper_at_most_{int(protocol.maximum_calibration_cp_upper * 100)}_percent": (
            cp_upper <= protocol.maximum_calibration_cp_upper
        ),
    }
    calibration_gate = (
        "PASS" if all(calibration_gates.values()) else "NOT_VIABLE_CALIBRATION"
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    min_ensemble = TaskJackknifeRoute13Ensemble(
        tuple(learners), M427_TASKS, "min", min_threshold
    )
    mean_ensemble = TaskJackknifeRoute13Ensemble(
        tuple(learners), M427_TASKS, "mean", mean_threshold
    )
    min_descriptor = min_ensemble.save(args.output_dir / "ensemble_min")
    mean_descriptor = mean_ensemble.save(args.output_dir / "ensemble_mean")
    single_path = args.output_dir / "single_full_router.npz"
    single_full.save(single_path, aggregation="single_full")

    loaded_min = TaskJackknifeRoute13Ensemble.load(min_descriptor)
    loaded_mean = TaskJackknifeRoute13Ensemble.load(mean_descriptor)
    loaded_single = RiskRoute13Model.load(single_path)
    roundtrip_checks = {
        "ensemble_min_scores_exact": np.array_equal(
            loaded_min.scores(arrays), min_scores
        ),
        "ensemble_mean_scores_exact": np.array_equal(
            loaded_mean.scores(arrays), mean_scores
        ),
        "single_full_scores_exact": np.array_equal(
            loaded_single.probabilities(arrays), single_scores
        ),
        "all_excluded_task_rows_zero": all(
            item["excluded_task_rows_used"] == 0 for item in learner_reports
        ),
    }
    if not all(roundtrip_checks.values()):
        raise RuntimeError(f"{protocol.name.upper()} checkpoint roundtrip failed")

    result = {
        "status": "PASS" if all(roundtrip_checks.values()) else "FAIL",
        "scope": protocol.fit_scope,
        "protocol": protocol.name,
        "router_calibration_gate": calibration_gate,
        "sealed_test_evaluated": False,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "feature_source": feature_source,
        "development_episodes": list(protocol.development_episodes),
        "calibration_episodes": list(protocol.calibration_episodes),
        "sealed_test_episodes": list(protocol.sealed_episodes),
        "pca_rank": args.pca_rank,
        "l2": args.l2,
        "max_iter": args.max_iter,
        "layer_norm_eps": args.layer_norm_eps,
        "role_data": {
            "development": development_data,
            "calibration": calibration_data,
            "test": {
                "episode_indices": list(protocol.sealed_episodes),
                "sealed": True,
            },
        },
        "learner_reports": learner_reports,
        "checkpoint_files": {
            "ensemble_min": {
                "path": str(min_descriptor.resolve()),
                "sha256": sha256_file(min_descriptor),
            },
            "ensemble_mean": {
                "path": str(mean_descriptor.resolve()),
                "sha256": sha256_file(mean_descriptor),
            },
            "single_full": {
                "path": str(single_path.resolve()),
                "sha256": sha256_file(single_path),
                "threshold": single_threshold,
            },
        },
        "roundtrip_checks": roundtrip_checks,
        "analyses": analyses,
        "calibration_gates": calibration_gates,
    }
    result_path = args.output_dir / "fit_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if calibration_gate != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
