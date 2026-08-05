#!/usr/bin/env python3
"""Reproducible post-sealed failure attribution for the M4.28 router.

M4.28's test split is already opened, so this script may inspect it for method
design.  It never rewrites M4.28 artifacts and it cannot authorize the failed
router for runtime use.  The output explicitly separates post-hoc diagnosis
from evidence supporting the already-frozen RP-PEP release.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.m427_task_jackknife_router import (  # noqa: E402
    TaskJackknifeRoute13Ensemble,
    episode_group_risk_metrics,
    strict_route13_or_27,
)
from a1.vla.dynamic_compute.release import (  # noqa: E402
    CHECKPOINT_SHA256,
    sha256_file,
    validate_rp_pep_release,
)
from a1.vla.dynamic_compute.risk_route13_router import route13_metrics  # noqa: E402
from scripts.dynamic_compute.build_m425b_temporal_features import (  # noqa: E402
    load_cache_entries,
)


M429_SCHEMA_VERSION = "phase-route-vla.m429-failure-analysis.v1"
M428_FEATURE_SHA256 = (
    "9b094e57ea513c75ad67d51596ae160155fe2bbf0cdf511395bea2252547f4a7"
)
M428_FIT_RESULT_SHA256 = (
    "48b1fb8c11443cd917c600eece37876d0470c535e8835edef37f98006711c15f"
)
M428_SEALED_RESULT_SHA256 = (
    "ef07df9e0aeb009295d29e0d8cdc988f9c9693b633fc68d9ef36183bb0ffc897"
)
M428_SEALED_EPISODES = tuple(range(20, 30))
M428_SEED = 20261228


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-result",
        type=Path,
        default=Path("reports/m428_temporal_features_20260805_v1/result.json"),
    )
    parser.add_argument(
        "--fit-result",
        type=Path,
        default=Path(
            "reports/m428_task_jackknife_router_fit_20260805_v1/fit_result.json"
        ),
    )
    parser.add_argument(
        "--sealed-result",
        type=Path,
        default=Path(
            "reports/m428_task_jackknife_router_sealed_20260805_v1/result.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m429_failure_analysis_20260805_v2"),
    )
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def cosine_action_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Match A1's mean cosine action delta over horizon positions."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("actions must have equal [H, A] shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("actions must be finite")
    left_norm = np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-5)
    right_norm = np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-5)
    similarity = np.sum((left / left_norm) * (right / right_norm), axis=-1)
    return float(np.mean(1.0 - similarity))


def threshold_scan(
    scores: np.ndarray,
    teacher_route: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    teacher = np.asarray(teacher_route, dtype=np.int64).reshape(-1)
    tasks = np.asarray(task_id, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    if not (values.shape == teacher.shape == tasks.shape == episodes.shape):
        raise ValueError("threshold-scan arrays must align")
    rows = []
    for threshold in thresholds:
        predicted = strict_route13_or_27(values, threshold=float(threshold))
        metrics = route13_metrics(predicted, teacher)
        groups = episode_group_risk_metrics(predicted, teacher, tasks, episodes)
        rows.append(
            {
                "threshold": float(threshold),
                "false_shallow_rows": int(metrics["route27_false_shallow"]),
                "false_shallow_groups": int(groups["route27_error_groups"]),
                "binary_exact_accuracy": float(metrics["binary_exact_accuracy"]),
                "safe13_recall": float(metrics["safe13_recall"]),
                "predicted13_coverage": float(metrics["predicted13_coverage"]),
            }
        )
    return rows


def score_tail_summary(scores: np.ndarray, teacher_route: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    teacher = np.asarray(teacher_route, dtype=np.int64).reshape(-1)
    if values.shape != teacher.shape:
        raise ValueError("score and teacher arrays must align")
    result = {}
    for name, mask in (("safe13", teacher <= 13), ("required27", teacher == 27)):
        selected = values[mask]
        if selected.size == 0:
            raise ValueError(f"empty score class: {name}")
        result[name] = {
            "rows": int(selected.size),
            "min": float(selected.min()),
            "median": float(np.median(selected)),
            "p95": float(np.quantile(selected, 0.95)),
            "p99": float(np.quantile(selected, 0.99)),
            "max": float(selected.max()),
        }
    # A dangerous negative is indistinguishable from at least half of the
    # positive class by score alone once its maximum reaches the safe median.
    # Comparing against the safe p95 would incorrectly label the overlap as
    # absent merely because many easy positives saturate near one.
    result["high_score_overlap_rule"] = "required27_max >= safe13_median"
    result["high_score_tail_overlaps"] = bool(
        result["required27"]["max"] >= result["safe13"]["median"]
    )
    return result


def _candidate_trace(shard: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roles = np.asarray(shard["fm_trace_roles"], dtype=np.uint8)
    candidate = roles == 1
    layers = np.asarray(shard["fm_trace_layers"], dtype=np.int64)[candidate]
    inputs = np.asarray(shard["fm_trace_input_x"], dtype=np.float32)[candidate]
    outputs = np.asarray(shard["fm_trace_output_action"], dtype=np.float32)[candidate]
    if layers.size < 2 or inputs.shape[0] != layers.size or outputs.shape[0] != layers.size:
        raise ValueError("candidate trace is incomplete")
    return layers, inputs, outputs


def trace_diagnostics(shard: Mapping[str, np.ndarray]) -> dict[str, Any]:
    layers, inputs, outputs = _candidate_trace(shard)
    adjacent = []
    for index in range(1, layers.size):
        left = int(layers[index - 1])
        right = int(layers[index])
        action_difference = outputs[index] - outputs[index - 1]
        input_difference = inputs[index] - inputs[index - 1]
        adjacent.append(
            {
                "layers": f"{left}->{right}",
                "available_at_layer13": right <= 13,
                "action_cosine_delta": cosine_action_distance(
                    outputs[index - 1], outputs[index]
                ),
                "action_mae": float(np.mean(np.abs(action_difference))),
                "action_max_abs": float(np.max(np.abs(action_difference))),
                "input_mae": float(np.mean(np.abs(input_difference))),
                "gripper_sign_mismatches": int(
                    np.sum(
                        (outputs[index - 1, :, -1] >= 0)
                        != (outputs[index, :, -1] >= 0)
                    )
                ),
            }
        )
    causal_layers = [int(value) for value in layers if int(value) <= 13]
    future_layers = [int(value) for value in layers if int(value) > 13]
    return {
        "candidate_layers": [int(value) for value in layers],
        "causal_candidate_layers_at_route13": causal_layers,
        "future_candidate_layers_unavailable_at_route13": future_layers,
        "adjacent_candidate_diagnostics": adjacent,
    }


def _find_false_shallow_cache_rows(
    feature_result: Mapping[str, Any], false_records: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    cache_dirs = [
        Path(item["manifest_path"]).resolve().parent
        for item in feature_result["teacher_cache_sources"]
    ]
    entries, _ = load_cache_entries(
        cache_dirs,
        CHECKPOINT_SHA256,
        M428_SEED,
        expected_episodes=30,
    )
    wanted = {str(item["identity_sha256"]) for item in false_records}
    matched = {
        identity: (cache_dir, record)
        for cache_dir, record, identity in entries
        if identity in wanted
    }
    if set(matched) != wanted:
        raise ValueError("failed to align all false-shallow identities to teacher cache")
    return matched


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    feature_path = args.feature_result.resolve()
    fit_path = args.fit_result.resolve()
    sealed_path = args.sealed_result.resolve()
    feature_result = load_json(feature_path)
    fit_result = load_json(fit_path)
    sealed_result = load_json(sealed_path)
    arrays_path = Path(feature_result["arrays_path"])

    immutable_checks = {
        "feature_status": feature_result.get("status") == "PASS",
        "feature_protocol": feature_result.get("protocol") == "m428",
        "feature_arrays_sha256": sha256_file(arrays_path) == M428_FEATURE_SHA256,
        "fit_result_sha256": sha256_file(fit_path) == M428_FIT_RESULT_SHA256,
        "sealed_result_sha256": sha256_file(sealed_path)
        == M428_SEALED_RESULT_SHA256,
        "fit_precedes_sealed": fit_result.get("sealed_test_evaluated") is False,
        "sealed_engineering_pass": sealed_result.get("status") == "PASS",
        "sealed_router_not_viable": sealed_result.get("router_offline_gate")
        == "NOT_VIABLE",
        "sealed_runtime_forbidden": sealed_result.get("runtime_integration_allowed")
        is False,
    }
    if not all(immutable_checks.values()):
        raise ValueError("M4.28 immutable input checks failed")

    with np.load(arrays_path, allow_pickle=False) as source:
        mask = np.isin(source["episode_index"], M428_SEALED_EPISODES)
        arrays = {name: source[name][mask].copy() for name in source.files}
    if set(arrays["episode_index"].tolist()) != set(M428_SEALED_EPISODES):
        raise ValueError("opened M4.28 test episode grid differs")

    ensemble_descriptor = Path(
        fit_result["checkpoint_files"]["ensemble_min"]["path"]
    )
    ensemble = TaskJackknifeRoute13Ensemble.load(ensemble_descriptor)
    scores = ensemble.scores(arrays)
    frozen_threshold = float(ensemble.threshold)
    teacher = np.asarray(arrays["teacher_route"], dtype=np.int64)
    frozen_routes = strict_route13_or_27(scores, threshold=frozen_threshold)
    failures = np.flatnonzero((teacher == 27) & (frozen_routes == 13))
    reconstructed_false_records = [
        {
            "task_id": int(arrays["task_id"][index]),
            "episode_index": int(arrays["episode_index"][index]),
            "step_id": int(arrays["step_id"][index]),
            "call_index": int(arrays["call_index"][index]),
            "score_safe13": float(scores[index]),
            "identity_sha256": arrays["identity_sha256"][index].decode("ascii"),
            "phase_progress": float(arrays["phase_scalars"][index, 0]),
            "phase_boundary_prob": float(arrays["phase_scalars"][index, 1]),
            "phase_uncertainty": float(arrays["phase_scalars"][index, 2]),
        }
        for index in failures
    ]
    recorded_false = sealed_result["methods"]["ensemble_min"][
        "false_shallow_records"
    ]
    recorded_identities = {str(item["identity_sha256"]) for item in recorded_false}
    reconstructed_identities = {
        str(item["identity_sha256"]) for item in reconstructed_false_records
    }
    reconstruction_checks = {
        "sealed_rows": int(teacher.size) == 1314,
        "frozen_threshold_matches": math.isclose(
            frozen_threshold,
            float(sealed_result["methods"]["ensemble_min"]["threshold"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "four_false_shallow_rows": int(failures.size) == 4,
        "false_identity_set_matches": recorded_identities
        == reconstructed_identities,
    }
    if not all(reconstruction_checks.values()):
        raise RuntimeError("M4.28 sealed reconstruction failed")

    threshold_values = sorted(
        {frozen_threshold, 0.99, 0.994, float(np.nextafter(scores[teacher == 27].max(), 1.0))}
    )
    scan = threshold_scan(
        scores,
        teacher,
        arrays["task_id"],
        arrays["episode_index"],
        threshold_values,
    )

    matched = _find_false_shallow_cache_rows(feature_result, reconstructed_false_records)
    trace_rows = []
    for false_record in reconstructed_false_records:
        identity = str(false_record["identity_sha256"])
        cache_dir, manifest_record = matched[identity]
        shard_path = cache_dir / str(manifest_record["array_path"])
        with np.load(shard_path, allow_pickle=False) as shard:
            diagnostics = trace_diagnostics(shard)
        trace_rows.append(
            {
                **false_record,
                "cache_array": str(shard_path),
                "teacher_exit_layer": int(manifest_record["teacher_exit_layer"]),
                **diagnostics,
            }
        )

    release = validate_rp_pep_release(REPO_ROOT)
    release_checks = {
        "rp_pep_release_gate_pass": release["status"] == "PASS",
        "rp_pep_default_disabled": release["runtime_default_enabled"] is False,
        "learned_router_runtime_forbidden": release["learned_router_runtime_allowed"]
        is False,
    }
    engineering_status = (
        "PASS"
        if all(immutable_checks.values())
        and all(reconstruction_checks.values())
        and all(release_checks.values())
        else "FAIL"
    )
    result = {
        "schema_version": M429_SCHEMA_VERSION,
        "status": engineering_status,
        "scope": "m429_postsealed_failure_attribution_and_release_selection",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "analysis_is_posthoc": True,
        "m428_sealed_may_not_be_reused_as_unseen_test": True,
        "immutable_inputs": {
            "feature_result": str(feature_path),
            "feature_result_sha256": sha256_file(feature_path),
            "feature_arrays": str(arrays_path.resolve()),
            "feature_arrays_sha256": sha256_file(arrays_path),
            "fit_result": str(fit_path),
            "fit_result_sha256": sha256_file(fit_path),
            "sealed_result": str(sealed_path),
            "sealed_result_sha256": sha256_file(sealed_path),
        },
        "immutable_checks": immutable_checks,
        "reconstruction_checks": reconstruction_checks,
        "frozen_threshold": frozen_threshold,
        "score_tail_summary": score_tail_summary(scores, teacher),
        "threshold_scan": scan,
        "false_shallow_records": reconstructed_false_records,
        "false_shallow_trace_diagnostics": trace_rows,
        "verification_cost": {
            "original_a1_fm_solves_by_exit": {"3": 3, "11": 7, "13": 8, "27": 15},
            "rp_pep_fm_solves_by_exit": {"3": 2, "11": 4, "13": 5, "27": 7},
            "rp_pep_rng_burns_by_exit": {"3": 1, "11": 3, "13": 3, "27": 8},
            "oracle_route_then_solve_fm_solves": 1,
            "exact_online_verification_through_layer13_fm_solves": 5,
            "exact_online_verification_reject_then_layer27_total_fm_solves": 7,
            "interpretation": (
                "Exact action-consistency verification through layer13 consumes the "
                "same five real solves as RP-PEP's layer13 path; future trace deltas "
                "are unavailable at a causal layer13 decision. It is a safe fallback, "
                "not a free route-then-solve verifier."
            ),
        },
        "release_selection": {
            "selected_runtime": "rp_pep",
            "selected_runtime_gate": "PASS",
            "m428_learned_router": "PROHIBITED",
            "m428_learned_router_gate": "NOT_VIABLE",
            "new_learned_router_claim": "NOT_MADE",
            "reason": (
                "RP-PEP already has strict action/trajectory equivalence and measured "
                "latency gains. M4.28 high-score class overlap, phase-boundary failures, "
                "and non-zero verification cost do not justify runtime integration."
            ),
        },
        "rp_pep_release": release,
        "release_checks": release_checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if engineering_status == "PASS" else 1)


if __name__ == "__main__":
    main()
