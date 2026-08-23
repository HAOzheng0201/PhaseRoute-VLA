#!/usr/bin/env python3
"""Aggregate 18 immutable V3-D6 folds into development-selection evidence."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from a1.vla.dynamic_compute.v3.development_collection import stream_sha256  # noqa: E402
from a1.vla.dynamic_compute.v3.gripper_v2_models import tie_aware_auroc  # noqa: E402
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_CONTRACT_SHA256,
    D5_EPISODES,
    D5_FALLBACK_LAYER,
    D5_MAX_FALSE_SAFE_UCB95,
    D5_MIN_EARLY_FRACTION,
    D5_MIN_SAFE_CLUSTERS,
    D5_TASK_IDS,
    development_data_from_mapping,
    summarize_route,
)
from a1.vla.dynamic_compute.v3.severity_reliability import (  # noqa: E402
    D6_CONTRACT_SHA256,
    load_d6_contract,
    severity_weights,
)
from a1.vla.dynamic_compute.v3.severity_reliability_oof import (  # noqa: E402
    D6_FITS_PER_OUTER,
    D6_OOF_SCHEMA_VERSION,
)


DATASET_RESULT = Path("reports/v3_d5_development_dataset/result.json")
DATASET_RESULT_SHA256 = (
    "7b4facd767594974359bef11edec83bbe3df66c3ee4c5c3981814992f792186d"
)
DATASET_PAYLOAD_SHA256 = (
    "cf40a9802e37d2335668db7f7e24194a3316d552183151cc780fecb5424137df"
)
DATASET_PAYLOAD_SCHEMA = "phase-route-vla.v3.d5-joint-development-dataset.v1"
CONTRACT_VALIDATION = Path("results/v3/v3_d6_repair_contract_validation.json")
CONTRACT_VALIDATION_SHA256 = (
    "1e14491bfe256377762d47d007bc943677e990b474321913e6e912e98ec4e422"
)
D5_FORMAL_RESULT = Path("results/v3/v3_d5_formal_development_result.json")
D5_FORMAL_RESULT_SHA256 = (
    "f08e35e9588f44900d6e714dc45c7afb9e1cc7586e8bbbfade488f3ed783b6f8"
)
FOLD_ROOT = Path("reports/v3_d6_development_oof_folds")
OUTPUT = Path("reports/v3_d6_development_oof")
RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d6-development-selection-result.v1"
PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d6-development-selection-payload.v1"
RP_PEP_FM_CALLS = {11: 4, 13: 5, 27: 7}


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"V3-D6 aggregate JSON must be an object: {path}")
    return dict(value)


def target_metrics(score: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if (
        score.ndim != 1
        or target.shape != score.shape
        or target.dtype != torch.bool
        or not bool(torch.isfinite(score).all())
        or not bool(((score > 0.0) & (score < 1.0)).all())
    ):
        raise PermissionError("V3-D6 metric geometry differs")
    return {
        "rows": int(target.numel()),
        "positive": int(target.sum()),
        "positive_fraction": float(target.double().mean()),
        "binary_log_loss": float(F.binary_cross_entropy(score.double(), target.double())),
        "brier_score": float((score.double() - target.double()).square().mean()),
        "auroc": tie_aware_auroc(score.double(), target),
    }


def severity_distribution(weight: torch.Tensor) -> dict[str, Any]:
    quantiles = torch.quantile(
        weight.double(), torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float64)
    )
    return {
        "rows": int(weight.numel()),
        "mean": float(weight.double().mean()),
        "quantiles": {
            name: float(value)
            for name, value in zip(("min", "p50", "p90", "p95", "p99", "max"), quantiles)
        },
        "bands": {
            "equal_1": int((weight == 1.0).sum()),
            "gt_1_le_2": int(((weight > 1.0) & (weight <= 2.0)).sum()),
            "gt_2_le_3": int(((weight > 2.0) & (weight <= 3.0)).sum()),
            "gt_3_le_4": int(((weight > 3.0) & (weight <= 4.0)).sum()),
            "gt_4_le_5": int(((weight > 4.0) & (weight <= 5.0)).sum()),
        },
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D6 OOF aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D6 OOF aggregation requires a clean worktree")
    load_d6_contract(REPO_ROOT)
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D6 refuses to overwrite aggregate evidence")

    validation_path = REPO_ROOT / CONTRACT_VALIDATION
    validation = json_object(validation_path)
    if (
        stream_sha256(validation_path) != CONTRACT_VALIDATION_SHA256
        or validation.get("status") != "PASS_V3_D6_REPAIR_CONTRACT_FROZEN"
        or validation.get("contract", {}).get("sha256") != D6_CONTRACT_SHA256
    ):
        raise PermissionError("V3-D6 contract validation differs")
    d5_path = REPO_ROOT / D5_FORMAL_RESULT
    d5_result = json_object(d5_path)
    if (
        stream_sha256(d5_path) != D5_FORMAL_RESULT_SHA256
        or d5_result.get("status") != "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
    ):
        raise PermissionError("V3-D6 locked D5 comparison differs")

    dataset_result_path = REPO_ROOT / DATASET_RESULT
    if stream_sha256(dataset_result_path) != DATASET_RESULT_SHA256:
        raise PermissionError("V3-D6 aggregate dataset result SHA differs")
    dataset_result = json_object(dataset_result_path)
    dataset_path = dataset_result_path.parent / str(dataset_result["payload"])
    if (
        dataset_result.get("status") != "PASS_V3_D5_DEVELOPMENT_DATASET"
        or dataset_result.get("payload_sha256") != DATASET_PAYLOAD_SHA256
        or stream_sha256(dataset_path) != DATASET_PAYLOAD_SHA256
    ):
        raise PermissionError("V3-D6 aggregate dataset evidence differs")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if (
        dataset.get("schema_version") != DATASET_PAYLOAD_SCHEMA
        or dataset.get("contract_sha256") != D5_CONTRACT_SHA256
        or "full_action_distance" not in dataset
        or dataset.get("calibration_or_test_payload_opened") is not False
    ):
        raise PermissionError("V3-D6 aggregate dataset payload differs")
    data = development_data_from_mapping(dataset)
    distance = dataset["full_action_distance"].detach().cpu().contiguous()
    row_severity = severity_weights(distance)

    started = time.perf_counter()
    score = torch.full((data.rows, 2), float("nan"), dtype=torch.float64)
    selected_layer = torch.full((data.calls,), -1, dtype=torch.long)
    assigned = torch.zeros(data.rows, dtype=torch.long)
    fold_results: dict[str, Any] = {}
    selected_lambdas: dict[str, float] = {}
    robust_thresholds: dict[str, Any] = {}
    current_commit = git_output("rev-parse", "HEAD")
    for episode in D5_EPISODES:
        result_path = REPO_ROOT / FOLD_ROOT / f"episode{episode}/result.json"
        result = json_object(result_path)
        payload_path = result_path.parent / str(result["payload"])
        if (
            result.get("status") != "PASS_V3_D6_OOF_OUTER_FOLD"
            or result.get("outer_episode") != episode
            or result.get("source_git_commit") != current_commit
            or result.get("source_worktree_dirty") is not False
            or result.get("dataset_result_sha256") != DATASET_RESULT_SHA256
            or result.get("dataset_payload_sha256") != DATASET_PAYLOAD_SHA256
            or result.get("d6_contract_sha256") != D6_CONTRACT_SHA256
            or result.get("d6_contract_validation_sha256") != CONTRACT_VALIDATION_SHA256
            or stream_sha256(payload_path) != result.get("payload_sha256")
        ):
            raise PermissionError(f"V3-D6 outer result differs: episode {episode}")
        fold = torch.load(payload_path, map_location="cpu", weights_only=True)
        indices = fold["validation_indices"]
        expected_indices = torch.nonzero(data.episode_index == episode, as_tuple=False).flatten()
        robust = fold.get("robust_threshold")
        if (
            fold.get("schema_version") != D6_OOF_SCHEMA_VERSION
            or fold.get("outer_episode") != episode
            or fold.get("fit_count") != D6_FITS_PER_OUTER
            or not torch.equal(indices, expected_indices)
            or fold["validation_score"].shape != (indices.numel(), 2)
            or fold["selected_layer"].shape != (indices.numel() // 2,)
            or not isinstance(robust, Mapping)
            or result.get("robust_threshold") != robust
        ):
            raise PermissionError(f"V3-D6 outer payload differs: episode {episode}")
        score[indices] = fold["validation_score"]
        assigned[indices] += 1
        source_rows = data.source_row[indices][0::2]
        if (
            not torch.equal(data.source_row[indices][0::2], data.source_row[indices][1::2])
            or bool((selected_layer[source_rows] != -1).any())
        ):
            raise PermissionError("V3-D6 outer call assignment differs")
        selected_layer[source_rows] = fold["selected_layer"]
        result_hash = stream_sha256(result_path)
        fold_results[str(episode)] = {
            "path": str(result_path.relative_to(REPO_ROOT)),
            "sha256": result_hash,
            "payload_sha256": result["payload_sha256"],
            "validation_rows": int(indices.numel()),
            "fit_count": int(result["fit_count"]),
            "selected_lambda": float(result["selected_lambda"]),
            "robust_threshold_feasible": bool(robust["feasible"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
        }
        selected_lambdas[str(episode)] = float(result["selected_lambda"])
        robust_thresholds[str(episode)] = dict(robust)
    if (
        not torch.equal(assigned, torch.ones_like(assigned))
        or not bool(torch.isfinite(score).all())
        or bool((selected_layer == -1).any())
    ):
        raise PermissionError("V3-D6 aggregate OOF assignment differs")

    summary = summarize_route(selected_layer, data.unsafe_target, data.task_id, data.episode_index)
    paired_target = data.unsafe_target.reshape(data.calls, 2, 2)
    early = selected_layer != D5_FALLBACK_LAYER
    selected_index = (selected_layer == 13).long()
    call_rows = torch.arange(data.calls)
    false_full = torch.zeros(data.calls, dtype=torch.bool)
    false_gripper = torch.zeros(data.calls, dtype=torch.bool)
    false_full[early] = paired_target[call_rows[early], selected_index[early], 0]
    false_gripper[early] = paired_target[call_rows[early], selected_index[early], 1]
    task_call = data.task_id[0::2]
    episode_call = data.episode_index[0::2]
    false_full_clusters = set(zip(task_call[false_full].tolist(), episode_call[false_full].tolist()))
    false_gripper_clusters = set(
        zip(task_call[false_gripper].tolist(), episode_call[false_gripper].tolist())
    )
    selection_counts = Counter(selected_layer.tolist())
    per_task = {}
    for task in D5_TASK_IDS:
        mask = task_call == task
        per_task[str(task)] = {
            "calls": int(mask.sum()),
            "L11": int(((selected_layer == 11) & mask).sum()),
            "L13": int(((selected_layer == 13) & mask).sum()),
            "L27": int(((selected_layer == 27) & mask).sum()),
            "false_full_action_calls": int((false_full & mask).sum()),
            "false_gripper_calls": int((false_gripper & mask).sum()),
            "false_safe_calls": int((summary.selected_unsafe & mask).sum()),
        }

    false_records = []
    for call in torch.nonzero(false_full | false_gripper, as_tuple=False).flatten().tolist():
        candidate = int(selected_index[call])
        row = 2 * call + candidate
        episode = int(episode_call[call])
        false_records.append(
            {
                "source_row": call,
                "task_id": int(task_call[call]),
                "episode_index": episode,
                "selected_layer": int(selected_layer[call]),
                "full_action_unsafe": bool(false_full[call]),
                "gripper_step_unsafe": bool(false_gripper[call]),
                "full_action_distance": float(distance[row]),
                "severity_weight": float(row_severity[row]),
                "full_action_risk_score": float(score[row, 0]),
                "gripper_score": float(score[row, 1]),
                "outer_runtime_threshold": robust_thresholds[str(episode)]["runtime_threshold"],
                "action_consistency": bool(data.action_consistency[row]),
            }
        )

    d5_selection = d5_result["formal_gate"]
    d5_efficiency = d5_result["estimated_efficiency"]
    infeasible_folds = sum(
        not bool(value["robust_threshold_feasible"]) for value in fold_results.values()
    )
    gate_checks = {
        "all_6521_calls_13042_rows_and_180_clusters_present": (
            data.calls == 6521
            and data.rows == 13042
            and len(set(zip(data.task_id.tolist(), data.episode_index.tolist()))) == 180
        ),
        "all_18_outer_predictions_finite_and_assigned_once": bool(torch.isfinite(score).all())
        and int(assigned.min()) == 1
        and int(assigned.max()) == 1,
        "all_18_outer_robust_thresholds_feasible": infeasible_folds == 0,
        "minimum_60_safe_clusters": summary.safe_clusters >= D5_MIN_SAFE_CLUSTERS,
        "minimum_5_percent_early_exit_calls": summary.early_exit_fraction >= D5_MIN_EARLY_FRACTION,
        "all_10_tasks_have_nonzero_early_exit_calls": all(
            value > 0 for value in summary.per_task_early_calls
        ),
        "false_safe_cluster_exact_ucb95_at_most_5_percent": (
            summary.false_safe_ucb95 <= D5_MAX_FALSE_SAFE_UCB95
        ),
        "false_gripper_calls_at_most_D5_locked_zero": int(false_gripper.sum())
        <= int(d5_selection["false_gripper_calls"]),
        "false_full_action_clusters_at_most_D5_locked_four": len(false_full_clusters)
        <= 4,
        "always_defer_rejected": summary.early_exit_calls > 0,
        "calibration_test_rollout_and_active_control_not_run": True,
    }
    status = (
        "PROMISING_V3_D6_DEVELOPMENT_SELECTION"
        if all(gate_checks.values())
        else "NEGATIVE_V3_D6_DEVELOPMENT_SELECTION"
    )
    metrics = {
        "full_action_severity_risk_score": target_metrics(score[:, 0], data.unsafe_target[:, 0]),
        "gripper_step_unsafe_probability": target_metrics(score[:, 1], data.unsafe_target[:, 1]),
    }
    estimated_fm_calls = sum(
        selection_counts[layer] * RP_PEP_FM_CALLS[layer] for layer in (11, 13, 27)
    )
    behavior_fm_calls = int(dataset_result["behavior_a1"]["behavior_fm_calls"])
    estimated_reduction = 1.0 - estimated_fm_calls / behavior_fm_calls
    result = {
        "status": status,
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "role": "development_v2_reused_for_method_selection",
        "suite": "libero_10",
        "rows": data.rows,
        "policy_calls": data.calls,
        "clusters": 180,
        "outer_fold_count": 18,
        "inner_fold_count_per_outer": 17,
        "fits_per_outer": D6_FITS_PER_OUTER,
        "total_model_fits": 18 * D6_FITS_PER_OUTER,
        "selected_lambdas": selected_lambdas,
        "robust_thresholds": robust_thresholds,
        "infeasible_outer_folds": infeasible_folds,
        "severity_weight_distribution": severity_distribution(row_severity),
        "OOF_metrics": metrics,
        "selection": {
            "L11": selection_counts[11],
            "L13": selection_counts[13],
            "L27": selection_counts[27],
            "early_exit_calls": summary.early_exit_calls,
            "early_exit_fraction": summary.early_exit_fraction,
            "safe_clusters": summary.safe_clusters,
            "false_safe_clusters": summary.false_safe_clusters,
            "false_safe_cluster_rate": (
                summary.false_safe_clusters / summary.safe_clusters if summary.safe_clusters else None
            ),
            "false_safe_cluster_ucb95": summary.false_safe_ucb95,
            "false_full_action_calls": int(false_full.sum()),
            "false_full_action_clusters": len(false_full_clusters),
            "false_gripper_calls": int(false_gripper.sum()),
            "false_gripper_clusters": len(false_gripper_clusters),
            "false_safe_calls": int(summary.selected_unsafe.sum()),
            "per_task_early_calls": list(summary.per_task_early_calls),
            "false_records": false_records,
        },
        "estimated_efficiency": {
            "shadow_rp_pep_fm_calls": estimated_fm_calls,
            "observed_behavior_A1_fm_calls": behavior_fm_calls,
            "estimated_fm_call_reduction_fraction": estimated_reduction,
            "risk_head_latency_included": False,
            "measured_end_to_end_latency": False,
        },
        "locked_D5_comparison": {
            "D5_result_sha256": D5_FORMAL_RESULT_SHA256,
            "D5_status": d5_result["status"],
            "D5_early_exit_fraction": d5_selection["early_exit_fraction"],
            "D6_minus_D5_early_exit_fraction": summary.early_exit_fraction
            - float(d5_selection["early_exit_fraction"]),
            "D5_false_safe_clusters": d5_selection["false_safe_clusters"],
            "D6_minus_D5_false_safe_clusters": summary.false_safe_clusters
            - int(d5_selection["false_safe_clusters"]),
            "D5_false_safe_cluster_ucb95": d5_selection["false_safe_cluster_ucb95"],
            "D6_minus_D5_false_safe_cluster_ucb95": summary.false_safe_ucb95
            - float(d5_selection["false_safe_cluster_ucb95"]),
            "D5_false_full_action_calls": d5_selection["false_full_action_calls"],
            "D6_false_full_action_calls": int(false_full.sum()),
            "D5_false_gripper_calls": d5_selection["false_gripper_calls"],
            "D6_false_gripper_calls": int(false_gripper.sum()),
            "D5_estimated_fm_reduction_fraction": d5_efficiency[
                "estimated_fm_call_reduction_fraction"
            ],
            "D6_minus_D5_estimated_fm_reduction_fraction": estimated_reduction
            - float(d5_efficiency["estimated_fm_call_reduction_fraction"]),
            "comparison_is_unbiased_or_fresh": False,
        },
        "per_task": per_task,
        "gate_checks": gate_checks,
        "input_sha256": {
            "D6_contract": D6_CONTRACT_SHA256,
            "D6_contract_validation": CONTRACT_VALIDATION_SHA256,
            "D5_formal_result": D5_FORMAL_RESULT_SHA256,
            "dataset_result": DATASET_RESULT_SHA256,
            "dataset_payload": DATASET_PAYLOAD_SHA256,
            "outer_fold_results": {
                episode: value["sha256"] for episode, value in fold_results.items()
            },
        },
        "outer_folds": fold_results,
        "access_ledger": {
            "development_v2_payload_reopened_after_D5_analysis": True,
            "development_model_fits": 18 * D6_FITS_PER_OUTER,
            "outer_truth_used_for_model_lambda_or_threshold_selection": False,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "fresh_rollout": False,
            "active_control": False,
        },
        "next_stage": {
            "authorized": (
                "FRESH_CALIBRATION_PROTOCOL_DESIGN_ONLY"
                if status == "PROMISING_V3_D6_DEVELOPMENT_SELECTION"
                else "D6_NEGATIVE_RESULT_ANALYSIS_ONLY"
            ),
            "reuse_D3_calibration_for_repair": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
        },
        "claim_boundary": {
            "result_is_fresh_confirmation": False,
            "result_is_unbiased_D5_comparison": False,
            "result_is_closed_loop_success": False,
            "estimated_FM_reduction_is_measured_latency": False,
            "layer27_consistency_is_task_success": False,
            "deployment_authorized": False,
            "superiority_claim_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    payload_path = incomplete / "development_severity_nested_oof.pt"
    torch.save(
        {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "contract_sha256": D6_CONTRACT_SHA256,
            "task_id": data.task_id.clone(),
            "episode_index": data.episode_index.clone(),
            "candidate_layer": data.candidate_layer.clone(),
            "unsafe_target": data.unsafe_target.clone(),
            "full_action_distance": distance.clone(),
            "severity_weight": row_severity.clone(),
            "OOF_score": score.clone(),
            "selected_layer": selected_layer.clone(),
            "selected_unsafe": summary.selected_unsafe.clone(),
            "outer_selected_lambdas": selected_lambdas,
            "outer_robust_thresholds": robust_thresholds,
            "development_data_reused_after_D5_analysis": True,
            "calibration_or_test_payload_opened": False,
            "active_control": False,
        },
        payload_path,
    )
    result["payload"] = payload_path.name
    result["payload_sha256"] = stream_sha256(payload_path)
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{stream_sha256(result_path)}  result.json\n", encoding="utf-8"
    )
    shutil.move(str(incomplete), str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
