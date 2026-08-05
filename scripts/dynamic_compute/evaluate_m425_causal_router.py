"""One-shot sealed task8/9 evaluation for the frozen M4.25 routers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.causal_route_router import (  # noqa: E402
    CausalRouteRouter,
    route_metrics,
    sequential_routes,
)
from scripts.dynamic_compute.train_m425_causal_router import (  # noqa: E402
    DEV_TASKS,
    TEST_TASKS,
    load_feature_table,
    low_cost_features,
    router_probabilities_from_npz,
)


EXPECTED_M424_SHA256 = "329db77d02f360c9eeed3721aa49157b7b552d756174305cf28ab7211c22cc1e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, action="append", required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--m424-result", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty values")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def add_estimated_latency(
    metrics: dict[str, Any],
    predicted: np.ndarray,
    *,
    route_latency_ms: Mapping[int, float],
    full_latency_ms: float,
) -> dict[str, Any]:
    latencies = np.asarray([route_latency_ms[int(route)] for route in predicted])
    metrics = dict(metrics)
    metrics["estimated_cuda_latency_ms"] = {
        "mean": float(latencies.mean()),
        "median": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
    }
    metrics["estimated_reduction_vs_full_mean"] = float(
        1.0 - latencies.mean() / full_latency_ms
    )
    return metrics


def evaluate_predictions(
    predicted: np.ndarray,
    teacher: np.ndarray,
    *,
    route_latency_ms: Mapping[int, float],
    full_latency_ms: float,
) -> dict[str, Any]:
    return add_estimated_latency(
        route_metrics(predicted, teacher),
        predicted,
        route_latency_ms=route_latency_ms,
        full_latency_ms=full_latency_ms,
    )


def microbenchmark_router(
    router: CausalRouteRouter,
    feature11: np.ndarray,
    feature13: np.ndarray,
    repeats: int,
) -> dict[str, float | int]:
    if repeats < 100:
        raise ValueError("router latency benchmark requires at least 100 repeats")
    x11 = torch.from_numpy(feature11[:1].astype(np.float32))
    x13 = torch.from_numpy(feature13[:1].astype(np.float32))
    with torch.inference_mode():
        for _ in range(20):
            router.probability(11, x11)
            router.probability(13, x13)
        values = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            router.probability(11, x11)
            router.probability(13, x13)
            values.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "repeats": repeats,
        "two_head_mean_ms": float(statistics.fmean(values)),
        "two_head_median_ms": float(statistics.median(values)),
        "two_head_p95_ms": percentile(values, 95),
        "two_head_max_ms": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    fit_path = args.fit_result.resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("status") != "PASS" or fit.get("scope") != "m425_causal_router_grouped_fit":
        raise ValueError("M4.25 evaluation requires a PASS grouped fit result")
    if fit.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise ValueError("fit/checkpoint SHA mismatch")
    if tuple(fit["fit_config"]["development_tasks"]) != DEV_TASKS:
        raise ValueError("fit development task split differs")
    if tuple(fit["fit_config"]["sealed_test_tasks"]) != TEST_TASKS:
        raise ValueError("fit sealed test task split differs")
    hidden_path = Path(fit["hidden_router"]["path"])
    lowcost_path = Path(fit["step_proprio_router"]["path"])
    if sha256_file(hidden_path) != fit["hidden_router"]["sha256"]:
        raise ValueError("hidden router checkpoint SHA mismatch")
    if sha256_file(lowcost_path) != fit["step_proprio_router"]["sha256"]:
        raise ValueError("step+proprio router checkpoint SHA mismatch")

    table = load_feature_table(
        args.feature_result,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    test = np.isin(table.task_id, TEST_TASKS)
    if int(test.sum()) != 41 or set(np.unique(table.task_id[test]).tolist()) != set(TEST_TASKS):
        raise ValueError("sealed test grid must be exactly task8/9 with 41 rows")
    teacher = table.teacher_route[test]

    m424_path = args.m424_result.resolve()
    if sha256_file(m424_path) != EXPECTED_M424_SHA256:
        raise ValueError("M4.24 latency source SHA differs from preregistration")
    m424 = json.loads(m424_path.read_text(encoding="utf-8"))
    if m424.get("status") != "PASS" or m424.get("oracle_ceiling", {}).get("status") != "VIABLE":
        raise ValueError("M4.24 latency source is not a viable PASS result")
    route_latency = {
        int(layer): float(summary["oracle_latency_ms"]["mean"])
        for layer, summary in m424["by_oracle_route_layer"].items()
    }
    if route_latency.keys() != {11, 13, 27}:
        raise ValueError("M4.24 route latency table is incomplete")
    full_latency = float(m424["policy_summary"]["full_depth"]["cuda_latency_ms"]["mean"])

    p11, p13, threshold11, threshold13 = router_probabilities_from_npz(
        hidden_path, table.layer11[test], table.layer13[test]
    )
    hidden_routes = sequential_routes(
        p11, p13, threshold11=threshold11, threshold13=threshold13
    )
    cheap = low_cost_features(table)
    cheap11, cheap13, cheap_t11, cheap_t13 = router_probabilities_from_npz(
        lowcost_path, cheap[test], cheap[test]
    )
    cheap_routes = sequential_routes(
        cheap11, cheap13, threshold11=cheap_t11, threshold13=cheap_t13
    )

    methods = {
        "always27": evaluate_predictions(
            np.full(teacher.shape, 27, dtype=np.int64), teacher,
            route_latency_ms=route_latency, full_latency_ms=full_latency,
        ),
        "always11": evaluate_predictions(
            np.full(teacher.shape, 11, dtype=np.int64), teacher,
            route_latency_ms=route_latency, full_latency_ms=full_latency,
        ),
        "always13": evaluate_predictions(
            np.full(teacher.shape, 13, dtype=np.int64), teacher,
            route_latency_ms=route_latency, full_latency_ms=full_latency,
        ),
        "step_proprio": evaluate_predictions(
            cheap_routes, teacher,
            route_latency_ms=route_latency, full_latency_ms=full_latency,
        ),
        "layer_hidden": evaluate_predictions(
            hidden_routes, teacher,
            route_latency_ms=route_latency, full_latency_ms=full_latency,
        ),
    }
    hidden = methods["layer_hidden"]
    lowcost = methods["step_proprio"]
    gates = {
        "development_oof_zero_false_shallow": int(
            fit["hidden_router"]["oof_metrics"]["false_shallow"]
        ) == 0,
        "heldout_zero_false_shallow": int(hidden["false_shallow"]) == 0,
        "heldout_route27_zero_false_shallow": int(hidden["teacher27_false_shallow"]) == 0,
        "heldout_exact_accuracy_at_least_35_percent": float(hidden["exact_accuracy"]) >= 0.35,
        "heldout_shallow_coverage_at_least_25_percent": float(hidden["shallow_coverage"]) >= 0.25,
        "estimated_mean_latency_reduction_at_least_10_percent": float(
            hidden["estimated_reduction_vs_full_mean"]
        ) >= 0.10,
        "not_worse_than_step_proprio_exact": float(hidden["exact_accuracy"])
        >= float(lowcost["exact_accuracy"]),
        "not_worse_than_step_proprio_false_shallow": int(hidden["false_shallow"])
        <= int(lowcost["false_shallow"]),
    }
    viable = all(gates.values())
    router = CausalRouteRouter.from_npz(hidden_path)
    latency = microbenchmark_router(
        router, table.layer11[test], table.layer13[test], args.latency_repeats
    )

    result = {
        "status": "PASS",
        "scope": "m425_causal_router_sealed_task_holdout",
        "router_offline_gate": {
            "status": "ROUTER_OFFLINE_VIABLE" if viable else "NOT_VIABLE",
            "viable_for_runtime_integration": viable,
            "gates": gates,
        },
        "checkpoint_sha256": args.checkpoint_sha256,
        "cache_index_sha256": table.cache_index_sha256,
        "fit_result": {
            "path": str(fit_path),
            "sha256": sha256_file(fit_path),
        },
        "hidden_router": {
            "path": str(hidden_path.resolve()),
            "sha256": fit["hidden_router"]["sha256"],
            "threshold11": threshold11,
            "threshold13": threshold13,
        },
        "step_proprio_router": {
            "path": str(lowcost_path.resolve()),
            "sha256": fit["step_proprio_router"]["sha256"],
            "threshold11": cheap_t11,
            "threshold13": cheap_t13,
        },
        "development_tasks": list(DEV_TASKS),
        "sealed_test_tasks": list(TEST_TASKS),
        "sealed_test_rows": int(test.sum()),
        "teacher_distribution": {
            str(layer): int(np.sum(teacher == layer)) for layer in (11, 13, 27)
        },
        "latency_source": {
            "path": str(m424_path),
            "sha256": EXPECTED_M424_SHA256,
            "route_mean_ms": {str(key): value for key, value in route_latency.items()},
            "full_depth_mean_ms": full_latency,
        },
        "methods": methods,
        "router_cpu_latency": latency,
        "probability_audit": {
            "hidden11_min": float(p11.min()),
            "hidden11_max": float(p11.max()),
            "hidden13_min": float(p13.min()),
            "hidden13_max": float(p13.max()),
            "all_finite": bool(
                np.isfinite(p11).all()
                and np.isfinite(p13).all()
                and np.isfinite(cheap11).all()
                and np.isfinite(cheap13).all()
            ),
        },
        "inputs": list(table.input_files),
    }
    if not result["probability_audit"]["all_finite"]:
        result["status"] = "FAIL"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "result.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)
    if not viable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
