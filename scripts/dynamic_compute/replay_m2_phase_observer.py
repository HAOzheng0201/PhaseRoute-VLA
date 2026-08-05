"""Replay M2 phase inputs through the causal observer and benchmark latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from a1.vla.dynamic_compute.phase_observer import SafePhaseObserver
from a1.vla.dynamic_compute.phase_training import SPLIT_IDS, load_phase_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile, method="linear"))


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    calls_path = args.output_dir / "phase_observer_calls.jsonl"
    if result_path.exists() or calls_path.exists():
        raise FileExistsError(f"Refusing to overwrite observer replay in {args.output_dir}")
    bundle = load_phase_dataset(args.dataset, args.metadata)
    arrays = bundle.arrays
    episode_ids = bundle.metadata["episode_ids"]
    instruction_hashes = bundle.metadata["instruction_hashes"]
    order = np.lexsort((arrays["call_index"], arrays["episode_index"]))

    observer = SafePhaseObserver(
        args.phase_checkpoint,
        calls_path,
        device=args.device,
        history_len=int(bundle.metadata["history_len"]),
    )
    try:
        for row_index in order:
            episode_index = int(arrays["episode_index"][row_index])
            instruction_index = int(arrays["instruction_index"][row_index])
            ok = observer.log_call(
                context={
                    "episode_id": episode_ids[episode_index],
                    "step_id": int(arrays["step_id"][row_index]),
                    "task_id": int(arrays["task_id"][row_index]),
                },
                instruction=f"hash:{instruction_hashes[instruction_index]}",
                raw_proprio=arrays["current_raw_proprio"][row_index],
                normalized_proprio=arrays["current_proprio"][row_index],
                previous_action=(
                    arrays["previous_executed_action"][row_index]
                    if arrays["previous_executed_action_mask"][row_index]
                    else None
                ),
                normalized_action_chunk=arrays[
                    "current_normalized_action_chunk"
                ][row_index],
                action_chunk=arrays["current_normalized_action_chunk"][row_index],
                visual_summary=arrays["visual_summary"][row_index],
                instruction_summary=arrays["instruction_summary"][row_index],
                visual_token_count=576,
                instruction_token_count=1,
            )
            if not ok:
                raise RuntimeError(observer.last_error)
    finally:
        observer.close()

    records = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(order):
        raise ValueError("Observer replay record count differs from dataset")
    progress_by_row = np.empty(arrays["split"].shape[0], dtype=np.float64)
    boundary_by_row = np.empty_like(progress_by_row)
    for row_index, record in zip(order, records):
        progress_by_row[row_index] = float(record["progress"])
        boundary_by_row[row_index] = float(record["boundary_prob"])

    expected_progress = np.empty_like(progress_by_row)
    expected_boundary = np.empty_like(boundary_by_row)
    with np.load(args.reference_predictions, allow_pickle=False) as reference:
        for split_name, split_id in SPLIT_IDS.items():
            indices = np.flatnonzero(arrays["split"] == split_id)
            expected_progress[indices] = reference[
                f"{split_name}_progress"
            ].reshape(-1)
            expected_boundary[indices] = reference[
                f"{split_name}_boundary_prob"
            ].reshape(-1)
    progress_max_abs_diff = float(
        np.max(np.abs(progress_by_row - expected_progress))
    )
    boundary_max_abs_diff = float(
        np.max(np.abs(boundary_by_row - expected_boundary))
    )
    prediction_tolerance = 1e-5
    predictions_aligned = (
        progress_max_abs_diff <= prediction_tolerance
        and boundary_max_abs_diff <= prediction_tolerance
    )

    latencies = np.asarray([record["latency_ms"] for record in records], dtype=np.float64)
    steady_latencies = latencies[1:]
    source_results = [source["result"] for source in bundle.metadata["source_runs"]]
    total_policy_calls = sum(int(result["telemetry_calls"]) for result in source_results)
    weighted_a1_latency_ms = sum(
        float(result["latency_ms_mean"]) * int(result["telemetry_calls"])
        for result in source_results
    ) / total_policy_calls
    status_ok = (
        observer.error_count == 0
        and len(records) == bundle.metadata["records"]
        and predictions_aligned
        and np.isfinite(latencies).all()
    )
    result: Dict[str, object] = {
        "status": "PASS" if status_ok else "FAIL",
        "observer_only": True,
        "controls_early_exit": False,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "dataset_sha256": bundle.dataset_sha256,
        "phase_checkpoint": str(args.phase_checkpoint),
        "phase_checkpoint_sha256": observer.checkpoint_sha256,
        "records": len(records),
        "episodes": int(bundle.metadata["episodes"]),
        "observer_errors": observer.error_count,
        "observer_last_error": observer.last_error,
        "predictions_aligned_with_training_evaluation": predictions_aligned,
        "prediction_tolerance": prediction_tolerance,
        "progress_max_abs_diff": progress_max_abs_diff,
        "boundary_prob_max_abs_diff": boundary_max_abs_diff,
        "latency_ms": {
            "first_call": float(latencies[0]),
            "mean_all": float(latencies.mean()),
            "mean_excluding_first": float(steady_latencies.mean()),
            "median_excluding_first": float(np.median(steady_latencies)),
            "p95_excluding_first": percentile(steady_latencies, 95.0),
            "max_excluding_first": float(steady_latencies.max()),
            "std_excluding_first": float(statistics.pstdev(steady_latencies)),
        },
        "a1_policy_latency_ms_weighted_mean": weighted_a1_latency_ms,
        "observer_mean_excluding_first_to_a1_percent": (
            float(steady_latencies.mean()) / weighted_a1_latency_ms * 100.0
        ),
        "history_contract": (
            "Each prediction uses only prior calls from the same episode; the current "
            "action chunk is appended after prediction."
        ),
        "latency_limitations": [
            "Offline replay excludes A1/observer kernel contention.",
            "The first observer call is reported separately as cold-start latency.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
