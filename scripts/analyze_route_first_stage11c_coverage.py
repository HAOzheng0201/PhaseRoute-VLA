#!/usr/bin/env python3
"""Diagnose the coverage ceiling of the frozen Route-first score head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_calibration import (  # noqa: E402
    evaluate_route_first_threshold,
)
from a1.vla.dynamic_compute.route_first_router import (  # noqa: E402
    route_first_group_weights,
)


PROTOCOL_SCHEMA = "phase-route-vla.route-first-stage11c-coverage-protocol.v1"
RESULT_SCHEMA = "phase-route-vla.route-first-stage11c-coverage-result.v1"
RUNTIME_SCHEMA = "phase-route-vla.route-first-active-runtime.v1"
DEFAULT_PROTOCOL = (
    REPO_ROOT / "configs/research/route_first_stage11c_coverage_protocol.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/route_first/route_first_stage11c_coverage_diagnosis.json"
)


class Stage11CCoverageError(RuntimeError):
    """Raised when frozen diagnosis evidence is missing or inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11CCoverageError(message)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"JSON object required: {path}")
    return dict(value)


def _records(path: Path) -> list[dict[str, Any]]:
    output = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(
            isinstance(value, Mapping),
            f"JSONL object required: {path}:{line_number}",
        )
        output.append(dict(value))
    _require(bool(output), f"empty JSONL evidence: {path}")
    return output


def _resolve_bound_file(binding: Mapping[str, Any]) -> Path:
    _require(set(binding) >= {"path", "sha256"}, "file binding is incomplete")
    path = (REPO_ROOT / str(binding["path"])).resolve(strict=True)
    _require(sha256_file(path) == binding["sha256"], f"SHA-256 differs: {path}")
    return path


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot summarize empty scores")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _score_summary(values: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(
        scores.size > 0
        and np.isfinite(scores).all()
        and np.all((scores >= 0.0) & (scores <= 1.0)),
        "scores are invalid",
    )
    materialized = scores.tolist()
    return {
        "count": int(scores.size),
        "min": float(scores.min()),
        "p10": _nearest_rank(materialized, 0.10),
        "p25": _nearest_rank(materialized, 0.25),
        "p50": _nearest_rank(materialized, 0.50),
        "p75": _nearest_rank(materialized, 0.75),
        "p90": _nearest_rank(materialized, 0.90),
        "p95": _nearest_rank(materialized, 0.95),
        "p99": _nearest_rank(materialized, 0.99),
        "max": float(scores.max()),
    }


def summarize_live_scores(
    scores: np.ndarray,
    task_ids: np.ndarray,
    selected_layers: np.ndarray,
    *,
    threshold_grid: Sequence[float],
    frozen_threshold: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    tasks = np.asarray(task_ids, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected_layers, dtype=np.int64).reshape(-1)
    _require(values.shape == tasks.shape == selected.shape, "live arrays differ")
    _require(set(np.unique(tasks).tolist()) == set(range(10)), "task grid differs")
    _require(set(np.unique(selected).tolist()).issubset({13, 27}), "layers differ")
    _score_summary(values)
    expected = np.where(values >= frozen_threshold, 13, 27)
    _require(np.array_equal(expected, selected), "live threshold decisions differ")

    task_weights = np.zeros(values.size, dtype=np.float64)
    for task_id in range(10):
        mask = tasks == task_id
        task_weights[mask] = 0.1 / int(mask.sum())
    curve = []
    for threshold in threshold_grid:
        chosen = values >= float(threshold)
        curve.append(
            {
                "threshold": float(threshold),
                "selected_rows": int(chosen.sum()),
                "raw_policy_call_coverage": float(chosen.mean()),
                "equal_task_coverage": float(task_weights[chosen].sum()),
            }
        )
    rows = []
    for task_id in range(10):
        mask = tasks == task_id
        task_scores = values[mask]
        chosen = task_scores >= frozen_threshold
        below = task_scores[~chosen]
        rows.append(
            {
                "task_id": task_id,
                "policy_calls": int(mask.sum()),
                "L13_calls": int(chosen.sum()),
                "L13_fraction": float(chosen.mean()),
                "near_miss_within_0_02": int(
                    np.sum(
                        (task_scores < frozen_threshold)
                        & (task_scores >= frozen_threshold - 0.02)
                    )
                ),
                "maximum_score_below_threshold": (
                    float(below.max()) if below.size else None
                ),
            }
        )
    return {
        "rows": int(values.size),
        "score13": _score_summary(values),
        "frozen_threshold13": float(frozen_threshold),
        "observed_L13_calls": int(np.sum(selected == 13)),
        "observed_L13_fraction": float(np.mean(selected == 13)),
        "threshold_curve_descriptive_only": curve,
        "per_task": rows,
    }


def summarize_teacher_scores(
    scores: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    *,
    threshold_grid: Sequence[float],
    confidence_level: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 2, "teacher scores differ")
    layer = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    _require(
        values.shape[0] == layer.size == tasks.size == episodes.size,
        "teacher arrays differ",
    )
    score13 = values[:, 1]
    safe13 = layer <= 13
    weights = route_first_group_weights(tasks, episodes)
    curve = [
        evaluate_route_first_threshold(
            score13,
            safe13,
            weights,
            threshold=float(threshold),
            confidence_level=confidence_level,
        )
        for threshold in threshold_grid
    ]
    unsafe_scores = score13[~safe13]
    return {
        "rows": int(score13.size),
        "episode_indices": sorted(int(value) for value in np.unique(episodes)),
        "teacher_safe13_rows": int(safe13.sum()),
        "teacher_safe13_raw_fraction": float(safe13.mean()),
        "teacher_safe13_group_equal_ceiling": float(
            weights[safe13].sum() / weights.sum()
        ),
        "score13_all": _score_summary(score13),
        "score13_safe": _score_summary(score13[safe13]),
        "score13_unsafe": _score_summary(unsafe_scores),
        "maximum_unsafe_score13": float(unsafe_scores.max()),
        "threshold_curve_descriptive_only": curve,
    }


def classify_diagnosis(
    live: Mapping[str, Any],
    calibration: Mapping[str, Any],
    holdout: Mapping[str, Any],
    stage11b: Mapping[str, Any],
    *,
    rules: Mapping[str, float],
) -> tuple[str, dict[str, bool]]:
    probe = float(rules["threshold_probe"])

    def curve_row(result: Mapping[str, Any], threshold: float) -> Mapping[str, Any]:
        matches = [
            row
            for row in result["threshold_curve_descriptive_only"]
            if math.isclose(float(row["threshold"]), threshold, abs_tol=1e-12)
        ]
        _require(len(matches) == 1, "diagnostic threshold probe is missing")
        return matches[0]

    live_probe = curve_row(live, probe)
    holdout_probe = curve_row(holdout, probe)
    maximum_ceiling = float(
        rules["maximum_teacher_safe13_ceiling_for_low_ceiling"]
    )
    checks = {
        "calibration_teacher_safe13_ceiling_is_low": float(
            calibration["teacher_safe13_group_equal_ceiling"]
        )
        <= maximum_ceiling,
        "holdout_teacher_safe13_ceiling_is_low": float(
            holdout["teacher_safe13_group_equal_ceiling"]
        )
        <= maximum_ceiling,
        "threshold_relaxation_live_coverage_remains_weak": float(
            live_probe["raw_policy_call_coverage"]
        )
        <= float(rules["maximum_live_raw_coverage_for_weak_relaxation"]),
        "threshold_relaxation_holdout_risk_bound_is_high": float(
            holdout_probe["false_safe_upper_bound"]
        )
        >= float(rules["minimum_holdout_false_safe_upper_for_unsafe_relaxation"]),
        "unsafe_examples_have_high_confidence_overlap": float(
            holdout["maximum_unsafe_score13"]
        )
        >= float(rules["minimum_unsafe_score_for_high_confidence_overlap"]),
        "current_decoder_block_reduction_is_small": float(
            stage11b["routing_usage"]["decoder_block_reduction_fraction"]
        )
        <= float(rules["maximum_current_decoder_block_reduction"]),
    }
    status = (
        "THRESHOLD_ONLY_NOT_VIABLE_NEW_DEVELOPMENT_TARGET_REQUIRED"
        if all(checks.values())
        else "THRESHOLD_ONLY_REMAINS_DIAGNOSTIC_CANDIDATE"
    )
    return status, checks


def analyze(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = protocol_path.resolve(strict=True)
    protocol = _object(protocol_path)
    _require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "protocol differs")
    _require(
        protocol.get("decision", {}).get("emits_new_threshold") is False,
        "protocol attempted to select a threshold",
    )
    forbidden = protocol.get("forbidden")
    _require(
        isinstance(forbidden, Mapping)
        and forbidden
        and all(value is True for value in forbidden.values()),
        "forbidden operations are not fail-closed",
    )
    threshold_grid = [float(value) for value in protocol["diagnostic_threshold_grid"]]
    frozen_threshold = float(protocol["frozen_threshold13"])
    _require(frozen_threshold in threshold_grid, "frozen threshold is absent")
    confidence = float(protocol["weighting"]["confidence_level"])

    inputs = protocol["inputs"]
    stage11b_path = _resolve_bound_file(inputs["stage11b_aggregate"])
    stage11b = _object(stage11b_path)
    _require(stage11b.get("status") == "PASS", "Stage 11B aggregate did not pass")
    runtime_records = []
    runtime_manifest = {}
    for shard_name in sorted(inputs["stage11b_runtime_jsonl"]):
        binding = inputs["stage11b_runtime_jsonl"][shard_name]
        path = _resolve_bound_file(binding)
        rows = _records(path)
        runtime_records.extend(rows)
        runtime_manifest[shard_name] = {
            "path": str(path),
            "sha256": binding["sha256"],
            "records": len(rows),
        }
    _require(
        len(runtime_records) == stage11b["policy_calls"],
        "runtime count differs from Stage 11B",
    )
    _require(
        all(row.get("schema_version") == RUNTIME_SCHEMA for row in runtime_records),
        "runtime schema differs",
    )
    live = summarize_live_scores(
        np.asarray([row["route_first_scores"][1] for row in runtime_records]),
        np.asarray([row["context"]["task_id"] for row in runtime_records]),
        np.asarray([row["selected_layer"] for row in runtime_records]),
        threshold_grid=threshold_grid,
        frozen_threshold=frozen_threshold,
    )

    calibration_path = _resolve_bound_file(inputs["historical_calibration_scores"])
    holdout_path = _resolve_bound_file(inputs["historical_holdout_scores"])
    _resolve_bound_file(inputs["calibrated_router"])
    calibration_arrays = np.load(calibration_path, allow_pickle=False)
    holdout_arrays = np.load(holdout_path, allow_pickle=False)
    calibration = summarize_teacher_scores(
        calibration_arrays["scores"],
        calibration_arrays["teacher_layer"],
        calibration_arrays["task_id"],
        calibration_arrays["episode_index"],
        threshold_grid=threshold_grid,
        confidence_level=confidence,
    )
    holdout = summarize_teacher_scores(
        holdout_arrays["scores"],
        holdout_arrays["teacher_layer"],
        holdout_arrays["task_id"],
        holdout_arrays["episode_index"],
        threshold_grid=threshold_grid,
        confidence_level=confidence,
    )
    _require(
        calibration["episode_indices"]
        == inputs["historical_calibration_scores"]["episode_indices"],
        "calibration episode binding differs",
    )
    _require(
        holdout["episode_indices"]
        == inputs["historical_holdout_scores"]["episode_indices"],
        "holdout episode binding differs",
    )
    status, checks = classify_diagnosis(
        live,
        calibration,
        holdout,
        stage11b,
        rules=protocol["diagnostic_rules"],
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "stage11b_live_development": live,
        "historical_calibration_reused_diagnostic_only": calibration,
        "historical_holdout_reused_diagnostic_only": holdout,
        "diagnostic_checks": checks,
        "new_threshold": None,
        "runtime_change_authorized": False,
        "next_development_hypothesis": {
            "primary": (
                "replace conservative teacher-imitation target with a newly "
                "collected direct L13-to-L27 action-reliability target"
            ),
            "secondary": (
                "reduce fixed L27 and selected-action FM cost without changing "
                "decisions"
            ),
            "reject": (
                "do not lower the scalar threshold and promote it directly to "
                "Stage 12"
            ),
        },
        "inputs": {
            "stage11b_aggregate": {
                "path": str(stage11b_path),
                "sha256": inputs["stage11b_aggregate"]["sha256"],
            },
            "runtime": runtime_manifest,
            "historical_calibration_scores": {
                "path": str(calibration_path),
                "sha256": inputs["historical_calibration_scores"]["sha256"],
            },
            "historical_holdout_scores": {
                "path": str(holdout_path),
                "sha256": inputs["historical_holdout_scores"]["sha256"],
            },
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    args = parse_args()
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).strip():
        raise PermissionError("Stage 11C diagnosis requires a clean worktree")
    output = args.output.resolve()
    temporary = output.with_name(output.name + ".incomplete")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or temporary.exists() or sidecar.exists():
        raise FileExistsError("Stage 11C diagnosis refuses to overwrite evidence")
    value = analyze(args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
