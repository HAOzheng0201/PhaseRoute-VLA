#!/usr/bin/env python3
"""Diagnose frozen V3-D6 without fitting, repair, or sealed-data access."""

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


D6_FORMAL = Path("results/v3/v3_d6_formal_development_result.json")
D6_FORMAL_SHA256 = (
    "c8bda5b40afb93c5fe815e71224da1e0f99570e4b73970e4cf8489b78fd62fc6"
)
D6_OOF_RESULT = Path("reports/v3_d6_development_oof/result.json")
D6_OOF_RESULT_SHA256 = (
    "7f82486c38ea3b01fd64332db04cf5f56ee81bb34ae4965e29fd632cc5a83ec2"
)
D6_OOF_PAYLOAD = Path(
    "reports/v3_d6_development_oof/development_severity_nested_oof.pt"
)
D6_OOF_PAYLOAD_SHA256 = (
    "f4230860e6c45fd1a60db66330775eaf791a130e4ab0eb9dc282a2941cfed296"
)
D5_FORMAL = Path("results/v3/v3_d5_formal_development_result.json")
D5_FORMAL_SHA256 = (
    "f08e35e9588f44900d6e714dc45c7afb9e1cc7586e8bbbfade488f3ed783b6f8"
)
D5_OOF_PAYLOAD = Path(
    "reports/v3_d5_development_oof/development_joint_nested_oof.pt"
)
D5_OOF_PAYLOAD_SHA256 = (
    "db8235f568c26ec918ebce413e12bb8326a66e3e79a0063e77476d9058a899ed"
)
DATASET = Path(
    "reports/v3_d5_development_dataset/development_joint_reliability_dataset.pt"
)
DATASET_SHA256 = (
    "cf40a9802e37d2335668db7f7e24194a3316d552183151cc780fecb5424137df"
)
OUTPUT = Path("reports/v3_d6_negative_analysis")
BASE_MULTIPLIERS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.60, 0.50, 0.40)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D6 analysis JSON must be an object")
    return dict(value)


def authenticated_tensor(path: Path, expected: str) -> dict[str, Any]:
    if stream_sha256(path) != expected:
        raise PermissionError("V3-D6 analysis tensor SHA differs")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D6 analysis tensor must be a mapping")
    return dict(value)


def vector_threshold_route(
    score: torch.Tensor,
    action_consistency: torch.Tensor,
    threshold_by_call: torch.Tensor,
) -> torch.Tensor:
    calls = threshold_by_call.numel()
    paired = score.reshape(calls, 2, 2)
    consistency = action_consistency.reshape(calls, 2)
    threshold = threshold_by_call[:, None]
    safe = (
        torch.isfinite(paired).all(dim=2)
        & consistency
        & (paired[:, :, 1] <= D5_GRIPPER_THRESHOLD)
        & (paired[:, :, 0] <= threshold)
    )
    selected = torch.full((calls,), D5_FALLBACK_LAYER, dtype=torch.long)
    selected[safe[:, 1]] = 13
    selected[safe[:, 0]] = 11
    return selected


def selected_unsafe(
    selected: torch.Tensor, paired_target: torch.Tensor
) -> torch.Tensor:
    early = selected != D5_FALLBACK_LAYER
    layer_index = (selected == 13).long()
    rows = torch.arange(selected.numel())
    unsafe = torch.zeros(selected.numel(), dtype=torch.bool)
    unsafe[early] = paired_target[rows[early], layer_index[early]].any(dim=1)
    return unsafe


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
        "frozen_numeric_gate_would_pass": summary.feasible,
        "runtime_authorized": False,
    }


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D6 negative analysis requires clean worktree")
    d6_formal_path = REPO_ROOT / D6_FORMAL
    d6_result_path = REPO_ROOT / D6_OOF_RESULT
    d5_formal_path = REPO_ROOT / D5_FORMAL
    if (
        stream_sha256(d6_formal_path) != D6_FORMAL_SHA256
        or stream_sha256(d6_result_path) != D6_OOF_RESULT_SHA256
        or stream_sha256(d5_formal_path) != D5_FORMAL_SHA256
    ):
        raise PermissionError("V3-D6 analysis metadata SHA differs")
    d6_formal = json_object(d6_formal_path)
    d6_result = json_object(d6_result_path)
    d5_formal = json_object(d5_formal_path)
    if (
        d6_formal.get("status") != "NEGATIVE_V3_D6_DEVELOPMENT_SELECTION"
        or d6_formal.get("authorization", {}).get("next_stage")
        != "D6_NEGATIVE_RESULT_ANALYSIS_ONLY"
        or d6_result.get("status") != "NEGATIVE_V3_D6_DEVELOPMENT_SELECTION"
        or d5_formal.get("status") != "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
    ):
        raise PermissionError("V3-D6 negative analysis is not authorized")

    d6 = authenticated_tensor(REPO_ROOT / D6_OOF_PAYLOAD, D6_OOF_PAYLOAD_SHA256)
    d5 = authenticated_tensor(REPO_ROOT / D5_OOF_PAYLOAD, D5_OOF_PAYLOAD_SHA256)
    dataset = authenticated_tensor(REPO_ROOT / DATASET, DATASET_SHA256)
    if (
        d6.get("schema_version")
        != "phase-route-vla.v3.d6-development-selection-payload.v1"
        or d5.get("schema_version")
        != "phase-route-vla.v3.d5-nested-oof-payload.v1"
        or dataset.get("schema_version")
        != "phase-route-vla.v3.d5-joint-development-dataset.v1"
        or d6.get("calibration_or_test_payload_opened") is not False
        or d5.get("calibration_or_test_payload_opened") is not False
        or dataset.get("calibration_or_test_payload_opened") is not False
    ):
        raise PermissionError("V3-D6 analysis payload semantics differ")

    score6 = d6["OOF_score"].double()
    score5 = d5["OOF_probability"].double()
    selected6 = d6["selected_layer"]
    selected5 = d5["selected_layer"]
    paired_target = dataset["unsafe_target"].reshape(6521, 2, 2)
    task = dataset["task_id"][0::2]
    episode = dataset["episode_index"][0::2]
    action_consistency = dataset["action_consistency"]
    distance = dataset["full_action_distance"].reshape(6521, 2).double()
    severity = d6["severity_weight"].reshape(6521, 2).double()
    robust = d6["outer_robust_thresholds"]
    threshold5 = {
        int(key): float(value) for key, value in d5["outer_selected_thresholds"].items()
    }
    base6_by_call = torch.tensor(
        [float(robust[str(int(value))]["pre_shrink_threshold"]) for value in episode],
        dtype=torch.float64,
    )
    runtime6_by_call = 0.95 * base6_by_call
    threshold5_by_call = torch.tensor(
        [threshold5[int(value)] for value in episode], dtype=torch.float64
    )
    if (
        score6.shape != (13042, 2)
        or score5.shape != (13042, 2)
        or selected6.shape != (6521,)
        or selected5.shape != (6521,)
        or not bool(torch.isfinite(score6).all())
        or not bool(torch.isfinite(score5).all())
    ):
        raise PermissionError("V3-D6 analysis geometry differs")
    recomputed6 = vector_threshold_route(score6, action_consistency, runtime6_by_call)
    if not torch.equal(recomputed6, selected6):
        raise PermissionError("V3-D6 frozen routing cannot be reproduced")

    unsafe5 = selected_unsafe(selected5, paired_target)
    unsafe6 = selected_unsafe(selected6, paired_target)
    false_rows5 = torch.nonzero(unsafe5, as_tuple=False).flatten()
    false_rows6 = torch.nonzero(unsafe6, as_tuple=False).flatten()
    if not torch.equal(false_rows5, false_rows6) or false_rows6.numel() != 4:
        raise PermissionError("V3-D5/D6 frozen false-safe identities differ")

    paired5 = score5.reshape(6521, 2, 2)
    paired6 = score6.reshape(6521, 2, 2)
    error_records = []
    for row in false_rows6.tolist():
        layer_index = int((selected6[row] == 13).long())
        if selected5[row] != selected6[row]:
            raise PermissionError("V3-D6 false-safe selected layer changed")
        value5 = float(paired5[row, layer_index, 0])
        value6 = float(paired6[row, layer_index, 0])
        base6 = float(base6_by_call[row])
        runtime6 = float(runtime6_by_call[row])
        old_threshold = float(threshold5_by_call[row])
        error_records.append(
            {
                "source_row": row,
                "task_id": int(task[row]),
                "episode_index": int(episode[row]),
                "selected_layer": int(selected6[row]),
                "distance_to_truth_threshold_ratio": float(distance[row, layer_index])
                / D5_ACTION_THRESHOLD,
                "D6_severity_weight": float(severity[row, layer_index]),
                "D5_full_action_score": value5,
                "D5_runtime_threshold": old_threshold,
                "D5_score_to_runtime_threshold_ratio": value5 / old_threshold,
                "D6_full_action_risk_score": value6,
                "D6_pre_shrink_base_threshold": base6,
                "D6_runtime_threshold": runtime6,
                "D6_score_to_runtime_threshold_ratio": value6 / runtime6,
                "D6_score_to_pre_shrink_base_ratio": value6 / base6,
                "D6_minus_D5_raw_score": value6 - value5,
                "minimum_base_multiplier_that_still_selects_call": value6 / base6,
                "full_action_unsafe": bool(paired_target[row, layer_index, 0]),
                "gripper_step_unsafe": bool(paired_target[row, layer_index, 1]),
                "A1_action_consistency_pass": bool(
                    action_consistency.reshape(6521, 2)[row, layer_index]
                ),
            }
        )

    threshold_diagnostic = {}
    for multiplier in BASE_MULTIPLIERS:
        diagnostic_selected = vector_threshold_route(
            score6, action_consistency, base6_by_call * multiplier
        )
        diagnostic_summary = summarize_route(
            diagnostic_selected,
            dataset["unsafe_target"],
            dataset["task_id"],
            dataset["episode_index"],
        )
        threshold_diagnostic[str(multiplier)] = summary_dict(diagnostic_summary)

    fold_threshold_records = {}
    active_folds = 0
    folds_with_any_lower_view = 0
    runtime_values = []
    base_values = []
    for episode_index in range(12, 30):
        value = robust[str(episode_index)]
        full = float(value["full_threshold"])
        order = float(value["order_statistic_threshold"])
        base = float(value["pre_shrink_threshold"])
        runtime = float(value["runtime_threshold"])
        jackknife = [float(item) for item in value["jackknife_thresholds"].values()]
        active = order < full and math.isclose(base, order, rel_tol=0.0, abs_tol=1.0e-15)
        any_lower = min(jackknife) < full
        active_folds += int(active)
        folds_with_any_lower_view += int(any_lower)
        base_values.append(base)
        runtime_values.append(runtime)
        fold_threshold_records[str(episode_index)] = {
            "full_threshold": full,
            "jackknife_minimum": min(jackknife),
            "jackknife_maximum": max(jackknife),
            "jackknife_unique_values": len(set(jackknife)),
            "fifth_smallest_threshold": order,
            "jackknife_changed_pre_shrink_base": active,
            "pre_shrink_base": base,
            "runtime_threshold": runtime,
        }
    if active_folds != 0:
        raise PermissionError("V3-D6 expected frozen jackknife branch to be inactive")

    early5 = selected5 != 27
    early6 = selected6 != 27
    newly_early = (~early5) & early6
    withdrawn_early = early5 & (~early6)
    transition = Counter(zip(selected5.tolist(), selected6.tolist()))
    transition_table = {
        f"D5_L{old}_to_D6_L{new}": transition[(old, new)]
        for old in (11, 13, 27)
        for new in (11, 13, 27)
    }
    new_early_cells = set(zip(task[newly_early].tolist(), episode[newly_early].tolist()))
    route_change = {
        "changed_calls": int((selected5 != selected6).sum()),
        "unchanged_calls": int((selected5 == selected6).sum()),
        "newly_early_calls": int(newly_early.sum()),
        "newly_early_task_episode_clusters": len(new_early_cells),
        "newly_early_false_safe_calls_on_reused_development": int(
            (newly_early & unsafe6).sum()
        ),
        "withdrawn_early_calls": int(withdrawn_early.sum()),
        "withdrawn_early_false_safe_calls_in_D5": int(
            (withdrawn_early & unsafe5).sum()
        ),
        "net_early_exit_change": int(early6.sum()) - int(early5.sum()),
        "transition_table": transition_table,
        "same_four_false_safe_source_rows_retained": false_rows6.tolist(),
        "route_change_is_unbiased_comparison": False,
    }

    analysis = {
        "status": "PASS_V3_D6_NEGATIVE_RESULT_ANALYSIS",
        "schema_version": "phase-route-vla.v3.d6-negative-analysis.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "formal_negative_result_reproduced": True,
        "formal_selection": d6_formal["formal_selection"],
        "false_safe_records_with_D5_comparison": error_records,
        "error_summary": {
            "D5_false_safe_source_rows": false_rows5.tolist(),
            "D6_false_safe_source_rows": false_rows6.tolist(),
            "same_four_failures_retained": True,
            "unique_task_episode_clusters": 4,
            "all_are_full_action_only": all(
                item["full_action_unsafe"] and not item["gripper_step_unsafe"]
                for item in error_records
            ),
            "minimum_D6_score_to_runtime_threshold_ratio": min(
                item["D6_score_to_runtime_threshold_ratio"] for item in error_records
            ),
            "maximum_D6_score_to_runtime_threshold_ratio": max(
                item["D6_score_to_runtime_threshold_ratio"] for item in error_records
            ),
            "uniform_base_multiplier_must_be_strictly_below_to_reject_all_four": min(
                item["minimum_base_multiplier_that_still_selects_call"]
                for item in error_records
            ),
        },
        "route_change_from_locked_D5": route_change,
        "jackknife_effectiveness": {
            "outer_folds": 18,
            "folds_with_at_least_one_jackknife_view_below_full_threshold": (
                folds_with_any_lower_view
            ),
            "folds_where_fifth_smallest_changed_pre_shrink_base": active_folds,
            "jackknife_branch_activation_fraction": active_folds / 18,
            "fixed_0_95_multiplier_was_only_active_global_shrink": True,
            "per_outer_fold": fold_threshold_records,
        },
        "threshold_stability": {
            "pre_shrink_base_minimum": min(base_values),
            "pre_shrink_base_maximum": max(base_values),
            "pre_shrink_base_mean": statistics.fmean(base_values),
            "pre_shrink_base_population_std": statistics.pstdev(base_values),
            "pre_shrink_base_maximum_to_minimum_ratio": max(base_values)
            / min(base_values),
            "runtime_threshold_minimum": min(runtime_values),
            "runtime_threshold_maximum": max(runtime_values),
            "runtime_threshold_maximum_to_minimum_ratio": max(runtime_values)
            / min(runtime_values),
            "D5_threshold_maximum_to_minimum_ratio": d5_formal["training_audit"][
                "selected_thresholds"
            ]["16"]
            / d5_formal["training_audit"]["selected_thresholds"]["14"],
        },
        "posthoc_diagnostic_base_multipliers": threshold_diagnostic,
        "scientific_interpretation": {
            "D6_is_formally_negative": True,
            "severity_weighting_and_threshold_repair_removed_a_false_cluster": False,
            "D6_added_102_development_early_exits_without_new_observed_error": True,
            "that_coverage_observation_is_fresh_or_unbiased": False,
            "jackknife_order_statistic_affected_runtime_threshold": False,
            "severity_score_gain_was_partly_cancelled_by_threshold_rescaling": True,
            "uniform_multiplier_alone_addresses_sample_uncertainty": False,
            "remaining_problem_is_sample_level_tail_misranking_or_uncertainty": True,
            "posthoc_multiplier_diagnostics_authorize_runtime_repair": False,
            "same_development_data_can_confirm_D7": False,
        },
        "D7_design_requirements": {
            "must_target_sample_level_epistemic_or_OOD_risk": True,
            "must_not_be_only_another_posthoc_uniform_multiplier": True,
            "must_preserve_specialized_gripper_gate": True,
            "must_fit_within_runtime_parameter_and_latency_budget": True,
            "must_pre_register_before_any_new_model_fit": True,
            "development_12_29_may_only_support_design_not_confirmation": True,
            "calibration_30_39_may_not_be_reused_for_repair_or_selection": True,
            "independent_test_40_49_remains_sealed": True,
        },
        "input_sha256": {
            "D6_formal_attestation": D6_FORMAL_SHA256,
            "D6_OOF_result": D6_OOF_RESULT_SHA256,
            "D6_OOF_payload": D6_OOF_PAYLOAD_SHA256,
            "D5_formal_attestation": D5_FORMAL_SHA256,
            "D5_OOF_payload": D5_OOF_PAYLOAD_SHA256,
            "development_dataset": DATASET_SHA256,
        },
        "access_ledger": {
            "development_v2_payload_opened": True,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "model_refit": 0,
            "runtime_threshold_selected": False,
            "fresh_rollout": False,
            "active_control": False,
            "gpu_query_or_initialization": 0,
        },
        "authorization": {
            "next_stage": "D7_PROTOCOL_DESIGN_ONLY_USING_D5_D6_AS_REUSED_DEVELOPMENT_EVIDENCE",
            "reuse_D3_calibration_for_repair": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
            "deployment_authorized": False,
        },
        "claim_boundary": d6_formal["claim_boundary"],
    }
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D6 refuses to overwrite negative analysis")
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
