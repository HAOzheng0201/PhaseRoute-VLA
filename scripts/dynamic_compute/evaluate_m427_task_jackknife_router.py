"""One-shot sealed evaluation for frozen M4.27 task-jackknife routers."""

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

from a1.vla.dynamic_compute.m427_task_jackknife_router import (  # noqa: E402
    TaskJackknifeRoute13Ensemble,
    episode_group_risk_metrics,
    strict_route13_or_27,
)
from a1.vla.dynamic_compute.risk_route13_router import (  # noqa: E402
    RiskRoute13Model,
    route13_metrics,
)
from scripts.dynamic_compute.train_m427_task_jackknife_router import (  # noqa: E402
    EXPECTED_FEATURES,
    M427_PROTOCOL,
    SEALED_EPISODES,
    TaskJackknifeProtocol,
    get_protocol_config,
    subset_arrays,
)


EXPECTED_SCOPE = "m427_temporal_route_feature_table"
EXPECTED_FIT_SCOPE = "m427_task_jackknife_fit_and_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--m424-result", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_sealed_features(
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
        raise ValueError(f"{protocol.name.upper()} feature result failed sealed checks")
    arrays_path = Path(result["arrays_path"])
    if sha256_file(arrays_path) != result.get("arrays_sha256"):
        raise ValueError("M4.27 feature array SHA-256 differs")
    with np.load(arrays_path, allow_pickle=False) as source_arrays:
        if not EXPECTED_FEATURES.issubset(source_arrays.files):
            raise KeyError("M4.27 feature table misses required arrays")
        sealed_mask = np.isin(
            source_arrays["episode_index"], protocol.sealed_episodes
        )
        arrays = {
            name: source_arrays[name][sealed_mask].copy()
            for name in EXPECTED_FEATURES
        }
    if set(arrays["episode_index"].tolist()) != set(protocol.sealed_episodes):
        raise ValueError(f"{protocol.name.upper()} sealed episode grid differs")
    if set(arrays["task_id"].tolist()) != set(range(10)):
        raise ValueError(f"{protocol.name.upper()} sealed task grid differs")
    if np.unique(arrays["identity_sha256"]).size != arrays["teacher_route"].size:
        raise ValueError(f"{protocol.name.upper()} sealed identities are duplicated")
    return arrays, {
        "path": str(source),
        "sha256": sha256_file(source),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": result["arrays_sha256"],
        "sealed_rows": int(arrays["teacher_route"].size),
    }


def latency_estimate(
    predicted_routes: np.ndarray, m424: Mapping[str, Any]
) -> dict[str, Any]:
    route_latency = {
        int(route): float(values["oracle_latency_ms"]["mean"])
        for route, values in m424["by_oracle_route_layer"].items()
        if int(route) in (13, 27)
    }
    if set(route_latency) != {13, 27}:
        raise ValueError("M4.24 route13/27 latency grid differs")
    routes = np.asarray(predicted_routes, dtype=np.int64).reshape(-1)
    per_row = np.asarray([route_latency[int(route)] for route in routes])
    full_mean = float(
        m424["policy_summary"]["full_depth"]["cuda_latency_ms"]["mean"]
    )
    mean = float(per_row.mean())
    return {
        "route_latency_ms": {str(key): value for key, value in route_latency.items()},
        "estimated_mean_ms": mean,
        "full_depth_mean_ms": full_mean,
        "reduction_fraction": 1.0 - mean / full_mean,
    }


def evaluate_scores(
    scores: np.ndarray,
    threshold: float,
    arrays: Mapping[str, np.ndarray],
    m424: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    routes = strict_route13_or_27(scores, threshold=threshold)
    return {
        "threshold": threshold,
        "score_range": {"min": float(scores.min()), "max": float(scores.max())},
        "metrics": route13_metrics(routes, teacher),
        "group_risk": episode_group_risk_metrics(
            routes, teacher, arrays["task_id"], arrays["episode_index"]
        ),
        "estimated_latency": latency_estimate(routes, m424),
        "predicted_routes": routes,
    }


def false_shallow_records(
    arrays: Mapping[str, np.ndarray], scores: np.ndarray, routes: np.ndarray
) -> list[dict[str, Any]]:
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    failures = np.flatnonzero((teacher == 27) & (routes == 13))
    return [
        {
            "task_id": int(arrays["task_id"][index]),
            "episode_index": int(arrays["episode_index"][index]),
            "step_id": int(arrays["step_id"][index]),
            "call_index": int(arrays["call_index"][index]),
            "score_safe13": float(scores[index]),
            "identity_sha256": arrays["identity_sha256"][index].decode("ascii"),
        }
        for index in failures
    ]


def m427_science_gates(
    main_method: Mapping[str, Any],
    controls: list[Mapping[str, Any]],
    protocol: TaskJackknifeProtocol = M427_PROTOCOL,
) -> dict[str, bool]:
    return {
        f"sealed_route27_rows_at_least_{protocol.minimum_route27_rows}": int(
            main_method["metrics"]["route27_rows"]
        )
        >= protocol.minimum_route27_rows,
        f"sealed_positive_groups_at_least_{protocol.minimum_positive_groups}": int(
            main_method["group_risk"]["route27_positive_groups"]
        )
        >= protocol.minimum_positive_groups,
        "sealed_route27_false_shallow_rows_zero": int(
            main_method["metrics"]["route27_false_shallow"]
        )
        == 0,
        "sealed_route27_error_groups_zero": int(
            main_method["group_risk"]["route27_error_groups"]
        )
        == 0,
        "sealed_binary_exact_at_least_70_percent": float(
            main_method["metrics"]["binary_exact_accuracy"]
        )
        >= 0.70,
        "sealed_safe13_recall_at_least_25_percent": float(
            main_method["metrics"]["safe13_recall"]
        )
        >= 0.25,
        "sealed_predicted13_coverage_at_least_25_percent": float(
            main_method["metrics"]["predicted13_coverage"]
        )
        >= 0.25,
        "sealed_latency_reduction_at_least_10_percent": float(
            main_method["estimated_latency"]["reduction_fraction"]
        )
        >= 0.10,
        "safety_not_worse_than_learning_controls": all(
            int(main_method["metrics"]["route27_false_shallow"])
            <= int(control["metrics"]["route27_false_shallow"])
            and int(main_method["group_risk"]["route27_error_groups"])
            <= int(control["group_risk"]["route27_error_groups"])
            for control in controls
        ),
        "binary_exact_not_below_worse_learning_control": float(
            main_method["metrics"]["binary_exact_accuracy"]
        )
        >= min(
            float(control["metrics"]["binary_exact_accuracy"])
            for control in controls
        ),
    }


def main(protocol_name: str = "m427") -> None:
    args = parse_args()
    protocol = get_protocol_config(protocol_name)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    arrays, feature_source = load_sealed_features(
        args.feature_result,
        args.checkpoint_sha256,
        args.phase_checkpoint_sha256,
        protocol,
    )
    fit_path = args.fit_result.resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if (
        fit.get("status") != "PASS"
        or fit.get("scope") != protocol.fit_scope
        or fit.get("protocol") != protocol.name
        or fit.get("router_calibration_gate") != "PASS"
        or fit.get("sealed_test_evaluated") is not False
        or fit.get("checkpoint_sha256") != args.checkpoint_sha256
        or fit.get("phase_checkpoint_sha256") != args.phase_checkpoint_sha256
        or fit.get("feature_source", {}).get("sha256") != feature_source["sha256"]
        or not all(bool(value) for value in fit.get("roundtrip_checks", {}).values())
        or not all(bool(value) for value in fit.get("calibration_gates", {}).values())
    ):
        raise ValueError(
            f"{protocol.name.upper()} fit result failed sealed-evaluation checks"
        )

    m424_path = args.m424_result.resolve()
    m424 = json.loads(m424_path.read_text(encoding="utf-8"))
    if (
        m424.get("status") != "PASS"
        or m424.get("oracle_ceiling", {}).get("status") != "VIABLE"
        or not bool(m424.get("oracle_ceiling", {}).get("viable_for_router_training"))
        or m424.get("checkpoint_sha256") != args.checkpoint_sha256
    ):
        raise ValueError("M4.24 latency source failed frozen checks")

    descriptors = fit["checkpoint_files"]
    for name in ("ensemble_min", "ensemble_mean", "single_full"):
        path = Path(descriptors[name]["path"])
        if sha256_file(path) != descriptors[name]["sha256"]:
            raise ValueError(
                f"{protocol.name.upper()} {name} checkpoint SHA-256 differs"
            )
    ensemble_min = TaskJackknifeRoute13Ensemble.load(
        descriptors["ensemble_min"]["path"]
    )
    ensemble_mean = TaskJackknifeRoute13Ensemble.load(
        descriptors["ensemble_mean"]["path"]
    )
    single_full = RiskRoute13Model.load(descriptors["single_full"]["path"])

    min_scores = ensemble_min.scores(arrays)
    mean_scores = ensemble_mean.scores(arrays)
    single_scores = single_full.probabilities(arrays)
    methods = {
        "ensemble_min": evaluate_scores(
            min_scores, ensemble_min.threshold, arrays, m424
        ),
        "ensemble_mean": evaluate_scores(
            mean_scores, ensemble_mean.threshold, arrays, m424
        ),
        "single_full": evaluate_scores(
            single_scores,
            float(descriptors["single_full"]["threshold"]),
            arrays,
            m424,
        ),
    }
    for name, scores in (
        ("ensemble_min", min_scores),
        ("ensemble_mean", mean_scores),
        ("single_full", single_scores),
    ):
        routes = methods[name].pop("predicted_routes")
        methods[name]["false_shallow_records"] = false_shallow_records(
            arrays, scores, routes
        )

    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    always27 = np.full(teacher.shape, 27, dtype=np.int64)
    always27_group = episode_group_risk_metrics(
        always27, teacher, arrays["task_id"], arrays["episode_index"]
    )
    constant_methods = {
        "always27": {
            "metrics": route13_metrics(always27, teacher),
            "group_risk": always27_group,
            "estimated_latency": latency_estimate(always27, m424),
        }
    }

    main_method = methods["ensemble_min"]
    controls = [methods["single_full"], methods["ensemble_mean"]]
    science_gates = m427_science_gates(main_method, controls, protocol)
    router_gate = "PASS" if all(science_gates.values()) else "NOT_VIABLE"
    engineering_checks = {
        "feature_source_match": True,
        "fit_result_frozen_before_sealed": True,
        "checkpoint_hashes_match": True,
        "sealed_episode_grid": set(arrays["episode_index"].tolist())
        == set(protocol.sealed_episodes),
        "sealed_task_grid": set(arrays["task_id"].tolist()) == set(range(10)),
        "sealed_unique_identity": np.unique(arrays["identity_sha256"]).size
        == teacher.size,
        "all_scores_finite": all(
            np.isfinite(values).all()
            for values in (min_scores, mean_scores, single_scores)
        ),
    }
    if not all(engineering_checks.values()):
        raise RuntimeError(
            f"{protocol.name.upper()} sealed engineering checks failed"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "PASS",
        "scope": protocol.sealed_scope,
        "protocol": protocol.name,
        "router_offline_gate": router_gate,
        "runtime_integration_allowed": router_gate == "PASS",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "feature_source": feature_source,
        "fit_result": str(fit_path),
        "fit_result_sha256": sha256_file(fit_path),
        "m424_result": str(m424_path),
        "m424_result_sha256": sha256_file(m424_path),
        "sealed_episode_indices": list(protocol.sealed_episodes),
        "sealed_rows": int(teacher.size),
        "sealed_teacher_distribution": {
            str(route): int(np.sum(teacher == route)) for route in (11, 13, 27)
        },
        "engineering_checks": engineering_checks,
        "science_gates": science_gates,
        "science_gates_passed": int(sum(science_gates.values())),
        "science_gates_total": len(science_gates),
        "methods": methods,
        "constant_methods": constant_methods,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
