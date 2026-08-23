#!/usr/bin/env python3
"""Diagnose the frozen V3-D5 negative result without authorizing repair."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.development_collection import stream_sha256  # noqa: E402
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_ACTION_THRESHOLD,
    D5_FALLBACK_LAYER,
    D5_GRIPPER_THRESHOLD,
    summarize_route,
)


FORMAL = Path("results/v3/v3_d5_formal_development_result.json")
FORMAL_SHA256 = (
    "f08e35e9588f44900d6e714dc45c7afb9e1cc7586e8bbbfade488f3ed783b6f8"
)
OOF_RESULT = Path("reports/v3_d5_development_oof/result.json")
OOF_RESULT_SHA256 = (
    "bddd8fdbbf53f5d8270ee13012dc6f29d5481ca6c5e1c4dde4aacb85cd3ca2bf"
)
OOF_PAYLOAD = Path(
    "reports/v3_d5_development_oof/development_joint_nested_oof.pt"
)
OOF_PAYLOAD_SHA256 = (
    "db8235f568c26ec918ebce413e12bb8326a66e3e79a0063e77476d9058a899ed"
)
DATASET = Path(
    "reports/v3_d5_development_dataset/development_joint_reliability_dataset.pt"
)
DATASET_SHA256 = (
    "cf40a9802e37d2335668db7f7e24194a3316d552183151cc780fecb5424137df"
)
OUTPUT = Path("reports/v3_d5_negative_analysis")
MULTIPLIERS = (1.0, 0.95, 0.9, 0.8, 0.75, 0.5)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D5 analysis JSON must be an object")
    return dict(value)


def authenticated_tensor(path: Path, expected: str) -> dict[str, Any]:
    if stream_sha256(path) != expected:
        raise PermissionError("V3-D5 analysis tensor SHA differs")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D5 analysis tensor must be a mapping")
    return dict(value)


def vector_threshold_route(
    probability: torch.Tensor,
    action_consistency: torch.Tensor,
    threshold_by_call: torch.Tensor,
) -> torch.Tensor:
    calls = threshold_by_call.numel()
    score = probability.reshape(calls, 2, 2)
    consistency = action_consistency.reshape(calls, 2)
    threshold = threshold_by_call[:, None]
    safe = (
        torch.isfinite(score).all(dim=2)
        & consistency
        & (score[:, :, 1] <= D5_GRIPPER_THRESHOLD)
        & (score[:, :, 0] <= threshold)
    )
    selected = torch.full((calls,), D5_FALLBACK_LAYER, dtype=torch.long)
    selected[safe[:, 1]] = 13
    selected[safe[:, 0]] = 11
    return selected


def summary_dict(summary: Any) -> dict[str, Any]:
    counts = Counter(summary.selected_layer.tolist())
    return {
        "L11": counts[11],
        "L13": counts[13],
        "L27": counts[27],
        "early_exit_calls": summary.early_exit_calls,
        "early_exit_fraction": summary.early_exit_fraction,
        "safe_clusters": summary.safe_clusters,
        "false_safe_clusters": summary.false_safe_clusters,
        "false_safe_ucb95": summary.false_safe_ucb95,
        "formal_D5_gate_would_pass": summary.feasible,
        "runtime_authorized": False,
    }


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 negative analysis requires clean worktree")
    formal_path = REPO_ROOT / FORMAL
    result_path = REPO_ROOT / OOF_RESULT
    if stream_sha256(formal_path) != FORMAL_SHA256:
        raise PermissionError("V3-D5 formal attestation SHA differs")
    if stream_sha256(result_path) != OOF_RESULT_SHA256:
        raise PermissionError("V3-D5 OOF result SHA differs")
    formal = json_object(formal_path)
    result = json_object(result_path)
    if (
        formal.get("status") != "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
        or formal.get("authorization", {}).get("next_stage")
        != "D5_NEGATIVE_RESULT_ANALYSIS_ONLY"
        or result.get("status") != "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
    ):
        raise PermissionError("V3-D5 negative analysis is not authorized")
    oof = authenticated_tensor(REPO_ROOT / OOF_PAYLOAD, OOF_PAYLOAD_SHA256)
    dataset = authenticated_tensor(REPO_ROOT / DATASET, DATASET_SHA256)
    if (
        oof.get("schema_version")
        != "phase-route-vla.v3.d5-nested-oof-payload.v1"
        or dataset.get("schema_version")
        != "phase-route-vla.v3.d5-joint-development-dataset.v1"
        or oof.get("calibration_or_test_payload_opened") is not False
        or dataset.get("calibration_or_test_payload_opened") is not False
    ):
        raise PermissionError("V3-D5 analysis payload semantics differ")
    probability = oof["OOF_probability"].double()
    selected = oof["selected_layer"]
    target = dataset["unsafe_target"].reshape(6521, 2, 2)
    task = dataset["task_id"][0::2]
    episode = dataset["episode_index"][0::2]
    action_consistency = dataset["action_consistency"]
    full_distance = dataset["full_action_distance"].reshape(6521, 2)
    thresholds = {
        int(key): float(value)
        for key, value in oof["outer_selected_thresholds"].items()
    }
    threshold_by_call = torch.tensor(
        [thresholds[int(value)] for value in episode], dtype=torch.float64
    )
    if (
        probability.shape != (13042, 2)
        or selected.shape != (6521,)
        or not bool(torch.isfinite(probability).all())
        or not bool(torch.isfinite(full_distance).all())
    ):
        raise PermissionError("V3-D5 analysis geometry differs")
    recomputed = vector_threshold_route(
        probability, action_consistency, threshold_by_call
    )
    if not torch.equal(recomputed, selected):
        raise PermissionError("V3-D5 frozen routing cannot be reproduced")

    early = selected != 27
    layer_index = (selected == 13).long()
    rows = torch.arange(6521)
    selected_target = target[rows, layer_index]
    false_safe = early & selected_target.any(dim=1)
    false_rows = torch.nonzero(false_safe, as_tuple=False).flatten()
    error_records = []
    for row in false_rows.tolist():
        index = int(layer_index[row])
        score_row = probability.reshape(6521, 2, 2)[row, index]
        threshold = float(threshold_by_call[row])
        distance = float(full_distance[row, index])
        error_records.append(
            {
                "source_row": row,
                "task_id": int(task[row]),
                "episode_index": int(episode[row]),
                "selected_layer": int(selected[row]),
                "full_action_score": float(score_row[0]),
                "gripper_score": float(score_row[1]),
                "outer_inner_selected_threshold": threshold,
                "full_score_to_threshold_ratio": float(score_row[0]) / threshold,
                "threshold_minus_full_score": threshold - float(score_row[0]),
                "full_action_distance": distance,
                "distance_to_truth_threshold_ratio": distance
                / D5_ACTION_THRESHOLD,
                "full_action_unsafe": bool(selected_target[row, 0]),
                "gripper_step_unsafe": bool(selected_target[row, 1]),
                "A1_action_consistency_pass": bool(
                    action_consistency.reshape(6521, 2)[row, index]
                ),
            }
        )
    if len(error_records) != 4:
        raise PermissionError("V3-D5 expected four frozen false-safe calls")

    diagnostic = {}
    for multiplier in MULTIPLIERS:
        diagnostic_selected = vector_threshold_route(
            probability,
            action_consistency,
            threshold_by_call * multiplier,
        )
        diagnostic_summary = summarize_route(
            diagnostic_selected,
            dataset["unsafe_target"],
            dataset["task_id"],
            dataset["episode_index"],
        )
        diagnostic[str(multiplier)] = summary_dict(diagnostic_summary)

    threshold_values = list(thresholds.values())
    score_ratios = [record["full_score_to_threshold_ratio"] for record in error_records]
    distance_ratios = [
        record["distance_to_truth_threshold_ratio"] for record in error_records
    ]
    analysis = {
        "status": "PASS_V3_D5_NEGATIVE_RESULT_ANALYSIS",
        "schema_version": "phase-route-vla.v3.d5-negative-analysis.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "formal_negative_result_reproduced": True,
        "formal_gate": formal["formal_gate"],
        "false_safe_records": error_records,
        "error_summary": {
            "false_safe_calls": len(error_records),
            "unique_task_episode_clusters": len(
                {(record["task_id"], record["episode_index"]) for record in error_records}
            ),
            "tasks": dict(sorted(Counter(record["task_id"] for record in error_records).items())),
            "episodes": dict(
                sorted(Counter(record["episode_index"] for record in error_records).items())
            ),
            "layers": dict(
                sorted(Counter(record["selected_layer"] for record in error_records).items())
            ),
            "all_are_full_action_only": all(
                record["full_action_unsafe"] and not record["gripper_step_unsafe"]
                for record in error_records
            ),
            "minimum_false_score_to_threshold_ratio": min(score_ratios),
            "maximum_false_score_to_threshold_ratio": max(score_ratios),
            "minimum_distance_to_truth_threshold_ratio": min(distance_ratios),
            "maximum_distance_to_truth_threshold_ratio": max(distance_ratios),
        },
        "threshold_stability": {
            "outer_threshold_minimum": min(threshold_values),
            "outer_threshold_maximum": max(threshold_values),
            "outer_threshold_mean": statistics.fmean(threshold_values),
            "outer_threshold_median": statistics.median(threshold_values),
            "outer_threshold_population_std": statistics.pstdev(threshold_values),
            "maximum_to_minimum_ratio": max(threshold_values) / min(threshold_values),
            "episode14_selected_different_lambda": True,
        },
        "posthoc_diagnostic_threshold_multipliers": diagnostic,
        "scientific_interpretation": {
            "D5_is_formally_negative": True,
            "predictive_AUROC_above_0_9": formal["interpretation"][
                "predictive_signal_is_strong"
            ],
            "specialized_gripper_protection_had_zero_selected_errors": True,
            "remaining_failure_is_full_action_cluster_tail_risk": True,
            "fold_specific_threshold_variability_requires_attention": (
                max(threshold_values) / min(threshold_values) > 2.0
            ),
            "posthoc_multiplier_diagnostics_are_runtime_policy": False,
            "posthoc_multiplier_diagnostics_authorize_repair": False,
            "same_development_OOF_is_fresh_confirmation_after_analysis": False,
        },
        "input_sha256": {
            "formal_attestation": FORMAL_SHA256,
            "OOF_result": OOF_RESULT_SHA256,
            "OOF_payload": OOF_PAYLOAD_SHA256,
            "development_dataset": DATASET_SHA256,
        },
        "access_ledger": {
            "development_v2_payload_opened": True,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "model_refit": 0,
            "threshold_selected_for_runtime": False,
            "fresh_rollout": False,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "D6_PROTOCOL_DESIGN_ONLY_USING_D5_AS_DEVELOPMENT_EVIDENCE",
            "reuse_D3_calibration_for_repair": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
            "deployment_authorized": False,
        },
        "claim_boundary": formal["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite negative analysis")
    incomplete.mkdir(parents=True, exist_ok=False)
    result_output = incomplete / "result.json"
    result_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{stream_sha256(result_output)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
