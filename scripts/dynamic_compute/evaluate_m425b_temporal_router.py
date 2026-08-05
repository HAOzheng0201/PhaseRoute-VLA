"""One-shot sealed episode4/5 evaluation for frozen M4.25b routers."""

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

from a1.vla.dynamic_compute.causal_route_router import (  # noqa: E402
    route_metrics,
    sequential_routes,
)
from a1.vla.dynamic_compute.temporal_route_router import (  # noqa: E402
    M425B_VARIANTS,
    TemporalRouteModel,
)
from scripts.dynamic_compute.train_m425b_temporal_router import (  # noqa: E402
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
    }
    if set(route_latency) != {11, 13, 27}:
        raise ValueError("M4.24 route latency grid differs")
    per_row = np.asarray([route_latency[int(route)] for route in predicted_routes])
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
        "development_oof_false_shallow_zero": int(
            fit_analysis["oof_metrics"]["false_shallow"]
        )
        == 0,
        "calibration_false_shallow_zero": int(
            fit_analysis["calibration_metrics"]["false_shallow"]
        )
        == 0,
        "sealed_false_shallow_zero": int(metrics["false_shallow"]) == 0,
        "sealed_route27_false_shallow_zero": int(
            metrics["teacher27_false_shallow"]
        )
        == 0
        and int(metrics["teacher27_rows"]) > 0,
        "sealed_exact_at_least_35_percent": float(metrics["exact_accuracy"]) >= 0.35,
        "sealed_shallow_coverage_at_least_25_percent": float(
            metrics["shallow_coverage"]
        )
        >= 0.25,
        "sealed_latency_reduction_at_least_10_percent": float(
            main["estimated_latency"]["reduction_fraction"]
        )
        >= 0.10,
        "exact_not_below_controls": all(
            float(metrics["exact_accuracy"])
            >= float(control["metrics"]["exact_accuracy"])
            for control in controls.values()
        ),
        "false_shallow_not_above_controls": all(
            int(metrics["false_shallow"])
            <= int(control["metrics"]["false_shallow"])
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
    )
    fit_path = args.fit_result.resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if (
        fit.get("status") != "PASS"
        or fit.get("scope") != "m425b_grouped_oof_and_calibration_fit"
        or fit.get("sealed_test_evaluated") is not False
        or fit.get("checkpoint_sha256") != args.checkpoint_sha256
        or fit.get("phase_checkpoint_sha256") != args.phase_checkpoint_sha256
        or fit.get("feature_source", {}).get("sha256") != feature_source["sha256"]
    ):
        raise ValueError("M4.25b fit result failed sealed-evaluation checks")
    m424_path = args.m424_result.resolve()
    m424 = json.loads(m424_path.read_text(encoding="utf-8"))
    if (
        m424.get("status") != "PASS"
        or m424.get("oracle_ceiling", {}).get("status") != "VIABLE"
        or not bool(m424.get("oracle_ceiling", {}).get("viable_for_router_training"))
        or m424.get("checkpoint_sha256") != args.checkpoint_sha256
    ):
        raise ValueError("M4.24 latency source failed frozen checks")
    test = np.isin(arrays["episode_index"], TEST_EPISODES)
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)[test]
    identities = arrays["identity_sha256"][test]
    if teacher.size < 1 or np.unique(identities).size != teacher.size:
        raise ValueError("sealed test rows are empty or duplicated")

    methods = {}
    for variant in M425B_VARIANTS:
        descriptor = fit["checkpoint_files"][variant]
        checkpoint_path = Path(descriptor["path"])
        if sha256_file(checkpoint_path) != descriptor["sha256"]:
            raise ValueError(f"{variant} checkpoint SHA-256 differs")
        model = TemporalRouteModel.load(checkpoint_path)
        if model.variant != variant:
            raise ValueError(f"{variant} checkpoint variant differs")
        probability11, probability13 = model.probabilities(arrays)
        if not np.isfinite(probability11[test]).all() or not np.isfinite(
            probability13[test]
        ).all():
            raise ValueError(f"{variant} sealed probabilities are non-finite")
        routes = sequential_routes(
            probability11[test],
            probability13[test],
            threshold11=model.threshold11,
            threshold13=model.threshold13,
        )
        methods[variant] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": descriptor["sha256"],
            "threshold11": model.threshold11,
            "threshold13": model.threshold13,
            "probability_ranges": {
                "p11_min": float(probability11[test].min()),
                "p11_max": float(probability11[test].max()),
                "p13_min": float(probability13[test].min()),
                "p13_max": float(probability13[test].max()),
            },
            "metrics": route_metrics(routes, teacher),
            "estimated_latency": latency_estimate(routes, m424),
        }

    constant_methods = {}
    for route in (11, 13, 27):
        routes = np.full(teacher.shape, route, dtype=np.int64)
        constant_methods[f"always{route}"] = {
            "metrics": route_metrics(routes, teacher),
            "estimated_latency": latency_estimate(routes, m424),
        }
    main_method = methods["temporal_phase"]
    controls = {
        name: methods[name] for name in ("hidden_only", "step_proprio")
    }
    gates = science_gates(
        main_method, controls, fit["analyses"]["temporal_phase"]
    )
    offline_viable = all(gates.values())
    engineering_checks = {
        "feature_source_match": fit["feature_source"]["sha256"]
        == feature_source["sha256"],
        "checkpoint_hashes_match": all(
            methods[name]["checkpoint_sha256"]
            == fit["checkpoint_files"][name]["sha256"]
            for name in M425B_VARIANTS
        ),
        "sealed_episode_grid": set(
            np.asarray(arrays["episode_index"])[test].tolist()
        )
        == set(TEST_EPISODES),
        "sealed_task_grid": set(np.asarray(arrays["task_id"])[test].tolist())
        == set(range(10)),
        "sealed_unique_identity": np.unique(identities).size == teacher.size,
        "sealed_route27_present": int(np.sum(teacher == 27)) > 0,
        "all_predictions_finite": all(
            np.isfinite(
                [
                    method["probability_ranges"]["p11_min"],
                    method["probability_ranges"]["p11_max"],
                    method["probability_ranges"]["p13_min"],
                    method["probability_ranges"]["p13_max"],
                ]
            ).all()
            for method in methods.values()
        ),
    }
    engineering_pass = all(engineering_checks.values())
    result = {
        "status": "PASS" if engineering_pass else "FAIL",
        "scope": "m425b_sealed_temporal_router_evaluation",
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
        "sealed_episode_indices": list(TEST_EPISODES),
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
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not engineering_pass:
        raise SystemExit(1)
    if not offline_viable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
