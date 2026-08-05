"""Aggregate compatible M2 PhaseEstimator runs without selecting the best seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate: {args.output}")
    runs: List[Dict[str, Any]] = []
    dataset_hashes = set()
    for result_path in args.results:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            raise ValueError(f"Run did not pass: {result_path}")
        if not result.get("observer_only") or result.get("controls_early_exit"):
            raise ValueError(f"Run is not observer-only: {result_path}")
        checkpoint_path = Path(result["artifacts"]["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        dataset_hashes.add(result["dataset_sha256"])
        runs.append(
            {
                "seed": int(result["seed"]),
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "best_epoch": int(result["best_epoch"]),
                "validation_boundary_threshold": float(
                    result["validation_boundary_threshold"]
                ),
                "test_progress_mae": float(
                    result["metrics"]["test"]["progress"]["mae"]
                ),
                "test_boundary_f1_fixed_0_5": float(
                    result["metrics"]["test"]["boundary_fixed_0_5"]["f1"]
                ),
                "test_boundary_f1_validation_calibrated": float(
                    result["metrics"]["test"]
                    ["boundary_validation_calibrated"]["f1"]
                ),
                "test_majority_boundary_f1": float(
                    result["baselines"]["test"]["majority_boundary"]["f1"]
                ),
                "test_constant_progress_mae": float(
                    result["baselines"]["test"]["constant_progress"]["mae"]
                ),
            }
        )
    if len(dataset_hashes) != 1:
        raise ValueError(f"Runs use different datasets: {sorted(dataset_hashes)}")
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training seed in aggregate")

    metric_names = (
        "test_progress_mae",
        "test_boundary_f1_fixed_0_5",
        "test_boundary_f1_validation_calibrated",
        "test_majority_boundary_f1",
        "test_constant_progress_mae",
    )
    aggregate = {
        "status": "PASS",
        "observer_only": True,
        "controls_early_exit": False,
        "dataset_sha256": next(iter(dataset_hashes)),
        "num_seeds": len(runs),
        "seeds": sorted(seeds),
        "metrics": {
            name: summarize([float(run[name]) for run in runs])
            for name in metric_names
        },
        "runs": sorted(runs, key=lambda run: run["seed"]),
        "interpretation_limits": [
            "All seeds share the same small four-task episode split.",
            "These metrics measure agreement with weak labels, not task success causality.",
            "No run controlled early exit or token routing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
