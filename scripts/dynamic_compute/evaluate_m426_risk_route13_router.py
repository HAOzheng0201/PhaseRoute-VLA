"""One-shot sealed episode4/5 evaluation for frozen M4.26 routers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.risk_route13_router import (  # noqa: E402
    M426_VARIANTS,
    RiskRoute13Model,
    route13_metrics,
)
from scripts.dynamic_compute.train_m426_risk_route13_router import (  # noqa: E402
    M426A_TEST_EPISODES,
    TEST_EPISODES,
    load_feature_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--m424-result", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def latency_estimate(
    predicted_routes: np.ndarray, m424: dict[str, Any]
) -> dict[str, Any]:
    route_latency = {
        int(route): float(values["oracle_latency_ms"]["mean"])
        for route, values in m424["by_oracle_route_layer"].items()
        if int(route) in (13, 27)
    }
    if set(route_latency) != {13, 27}:
        raise ValueError("M4.24 route13/27 latency grid differs")
    routes = np.asarray(predicted_routes, dtype=np.int64).reshape(-1)
    if not set(np.unique(routes).tolist()).issubset({13, 27}):
        raise ValueError("latency estimate received a non-M4.26 route")
    per_row = np.asarray([route_latency[int(route)] for route in routes])
    full_mean = float(m424["policy_summary"]["full_depth"]["cuda_latency_ms"]["mean"])
    mean = float(per_row.mean())
    return {
        "route_latency_ms": {str(key): value for key, value in route_latency.items()},
        "estimated_mean_ms": mean,
        "full_depth_mean_ms": full_mean,
        "reduction_fraction": 1.0 - mean / full_mean,
    }


def science_gates(
    main: dict[str, Any],
    controls: dict[str, dict[str, Any]],
    fit_analysis: dict[str, Any],
) -> dict[str, bool]:
    metrics = main["metrics"]
    return {
        "development_oof_route27_false_shallow_zero": int(
            fit_analysis["oof_metrics"]["route27_false_shallow"]
        )
        == 0,
        "calibration_route27_false_shallow_zero": int(
            fit_analysis["calibration_metrics"]["route27_false_shallow"]
        )
        == 0,
        "sealed_route27_false_shallow_zero": int(
            metrics["route27_false_shallow"]
        )
        == 0
        and int(metrics["route27_rows"]) > 0,
        "sealed_binary_exact_at_least_70_percent": float(
            metrics["binary_exact_accuracy"]
        )
        >= 0.70,
        "sealed_safe13_recall_at_least_25_percent": float(
            metrics["safe13_recall"]
        )
        >= 0.25,
        "sealed_predicted13_coverage_at_least_25_percent": float(
            metrics["predicted13_coverage"]
        )
        >= 0.25,
        "sealed_latency_reduction_at_least_10_percent": float(
            main["estimated_latency"]["reduction_fraction"]
        )
        >= 0.10,
        "binary_exact_not_below_controls": all(
            float(metrics["binary_exact_accuracy"])
            >= float(control["metrics"]["binary_exact_accuracy"])
            for control in controls.values()
        ),
        "route27_false_shallow_not_above_controls": all(
            int(metrics["route27_false_shallow"])
            <= int(control["metrics"]["route27_false_shallow"])
            for control in controls.values()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    arrays, feature_source = load_feature_table(
        args.feature_result,
        args.checkpoint_sha256,
        args.phase_checkpoint_sha256,
        protocol=args.protocol,
    )
    if args.protocol == "m426":
        expected_fit_scope = "m426_grouped_oof_and_calibration_fit"
        test_episodes = TEST_EPISODES
        result_scope = "m426_sealed_risk_route13_evaluation"
    else:
        expected_fit_scope = "m426a_grouped_oof_and_calibration_fit"
        test_episodes = M426A_TEST_EPISODES
        result_scope = "m426a_sealed_risk_route13_evaluation"
    fit_path = args.fit_result.resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if (
        fit.get("status") != "PASS"
        or fit.get("scope") != expected_fit_scope
        or fit.get("protocol") != args.protocol
        or fit.get("sealed_test_evaluated") is not False
        or fit.get("checkpoint_sha256") != args.checkpoint_sha256
        or fit.get("phase_checkpoint_sha256") != args.phase_checkpoint_sha256
        or fit.get("feature_source", {}).get("sha256") != feature_source["sha256"]
    ):
        raise ValueError("M4.26 fit result failed sealed-evaluation checks")
    m424_path = args.m424_result.resolve()
    m424 = json.loads(m424_path.read_text(encoding="utf-8"))
    if (
        m424.get("status") != "PASS"
        or m424.get("oracle_ceiling", {}).get("status") != "VIABLE"
        or not bool(m424.get("oracle_ceiling", {}).get("viable_for_router_training"))
        or m424.get("checkpoint_sha256") != args.checkpoint_sha256
    ):
        raise ValueError("M4.24 latency source failed frozen checks")
    test = np.isin(arrays["episode_index"], test_episodes)
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)[test]
    identities = arrays["identity_sha256"][test]
    if teacher.size < 1 or np.unique(identities).size != teacher.size:
        raise ValueError("sealed test rows are empty or duplicated")

    methods = {}
    for variant in M426_VARIANTS:
        descriptor = fit["checkpoint_files"][variant]
        checkpoint_path = Path(descriptor["path"])
        if sha256_file(checkpoint_path) != descriptor["sha256"]:
            raise ValueError(f"{variant} checkpoint SHA-256 differs")
        model = RiskRoute13Model.load(checkpoint_path)
        if model.variant != variant:
            raise ValueError(f"{variant} checkpoint variant differs")
        probability = model.probabilities(arrays)[test]
        if not np.isfinite(probability).all():
            raise ValueError(f"{variant} sealed probabilities are non-finite")
        routes = np.where(probability >= model.threshold, 13, 27).astype(np.int64)
        methods[variant] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": descriptor["sha256"],
            "threshold": model.threshold,
            "probability_range": {
                "min": float(probability.min()),
                "max": float(probability.max()),
            },
            "metrics": route13_metrics(routes, teacher),
            "estimated_latency": latency_estimate(routes, m424),
        }

    always27_routes = np.full(teacher.shape, 27, dtype=np.int64)
    constant_methods = {
        "always27": {
            "metrics": route13_metrics(always27_routes, teacher),
            "estimated_latency": latency_estimate(always27_routes, m424),
        }
    }
    main_method = methods["temporal_phase_step"]
    controls = {name: methods[name] for name in ("hidden_only", "step_proprio")}
    gates = science_gates(
        main_method, controls, fit["analyses"]["temporal_phase_step"]
    )
    offline_viable = all(gates.values())
    engineering_checks = {
        "feature_source_match": fit["feature_source"]["sha256"]
        == feature_source["sha256"],
        "checkpoint_hashes_match": all(
            methods[name]["checkpoint_sha256"]
            == fit["checkpoint_files"][name]["sha256"]
            for name in M426_VARIANTS
        ),
        "sealed_episode_grid": set(
            np.asarray(arrays["episode_index"])[test].tolist()
        )
        == set(test_episodes),
        "sealed_task_grid": set(np.asarray(arrays["task_id"])[test].tolist())
        == set(range(10)),
        "sealed_unique_identity": np.unique(identities).size == teacher.size,
        "sealed_route27_present": int(np.sum(teacher == 27)) > 0,
        "all_predictions_finite": all(
            np.isfinite(
                [
                    method["probability_range"]["min"],
                    method["probability_range"]["max"],
                ]
            ).all()
            for method in methods.values()
        ),
        "route_domain_is_13_or_27": all(
            set(method["metrics"]["predicted_distribution"]) == {"13", "27"}
            for method in methods.values()
        ),
    }
    engineering_pass = all(engineering_checks.values())
    result = {
        "status": "PASS" if engineering_pass else "FAIL",
        "scope": result_scope,
        "protocol": args.protocol,
        "router_offline_gate": "VIABLE" if offline_viable else "NOT_VIABLE",
        "runtime_integration_allowed": bool(engineering_pass and offline_viable),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "feature_source": feature_source,
        "fit_result": str(fit_path),
        "fit_result_sha256": sha256_file(fit_path),
        "m424_result": str(m424_path),
        "m424_result_sha256": sha256_file(m424_path),
        "sealed_episode_indices": list(test_episodes),
        "sealed_rows": int(teacher.size),
        "sealed_teacher_distribution": {
            str(route): int(np.sum(teacher == route)) for route in (11, 13, 27)
        },
        "engineering_checks": engineering_checks,
        "science_gates": gates,
        "science_gates_passed": int(sum(gates.values())),
        "science_gates_total": len(gates),
        "methods": methods,
        "constant_methods": constant_methods,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "router_offline_gate": result["router_offline_gate"],
                "runtime_integration_allowed": result["runtime_integration_allowed"],
                "science_gates": gates,
                "methods": {
                    name: {
                        "metrics": value["metrics"],
                        "estimated_latency": value["estimated_latency"],
                    }
                    for name, value in methods.items()
                },
                "result_sha256": sha256_file(result_path),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    if not engineering_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
