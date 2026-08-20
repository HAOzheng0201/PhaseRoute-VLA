#!/usr/bin/env python3
"""Aggregate all 18 immutable V3-D5 outer folds into the formal dev gate."""

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
    D5_CANDIDATE_LAYERS,
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
from a1.vla.dynamic_compute.v3.joint_reliability_oof import (  # noqa: E402
    D5_FITS_PER_OUTER,
    D5_OOF_SCHEMA_VERSION,
)


DATASET_RESULT = Path("reports/v3_d5_development_dataset/result.json")
DATASET_RESULT_SHA256 = (
    "7b4facd767594974359bef11edec83bbe3df66c3ee4c5c3981814992f792186d"
)
DATASET_PAYLOAD_SHA256 = (
    "cf40a9802e37d2335668db7f7e24194a3316d552183151cc780fecb5424137df"
)
DATASET_PAYLOAD_SCHEMA = "phase-route-vla.v3.d5-joint-development-dataset.v1"
FOLD_ROOT = Path("reports/v3_d5_development_oof_folds")
OUTPUT = Path("reports/v3_d5_development_oof")
RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d5-nested-oof-result.v1"
PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d5-nested-oof-payload.v1"
RP_PEP_FM_CALLS = {11: 4, 13: 5, 27: 7}


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"V3-D5 aggregate JSON must be an object: {path}")
    return dict(value)


def target_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if (
        probability.ndim != 1
        or target.shape != probability.shape
        or target.dtype != torch.bool
        or not bool(torch.isfinite(probability).all())
    ):
        raise PermissionError("V3-D5 metric geometry differs")
    return {
        "rows": int(target.numel()),
        "positive": int(target.sum()),
        "positive_fraction": float(target.double().mean()),
        "binary_log_loss": float(
            F.binary_cross_entropy(probability.double(), target.double())
        ),
        "brier_score": float((probability.double() - target.double()).square().mean()),
        "auroc": tie_aware_auroc(probability.double(), target),
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D5 OOF aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 OOF aggregation requires clean worktree")
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite aggregate evidence")
    dataset_result_path = REPO_ROOT / DATASET_RESULT
    if stream_sha256(dataset_result_path) != DATASET_RESULT_SHA256:
        raise PermissionError("V3-D5 aggregate dataset result SHA differs")
    dataset_result = json_object(dataset_result_path)
    dataset_path = dataset_result_path.parent / str(dataset_result["payload"])
    if (
        dataset_result.get("status") != "PASS_V3_D5_DEVELOPMENT_DATASET"
        or dataset_result.get("payload_sha256") != DATASET_PAYLOAD_SHA256
        or stream_sha256(dataset_path) != DATASET_PAYLOAD_SHA256
    ):
        raise PermissionError("V3-D5 aggregate dataset evidence differs")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if (
        dataset.get("schema_version") != DATASET_PAYLOAD_SCHEMA
        or dataset.get("contract_sha256") != D5_CONTRACT_SHA256
    ):
        raise PermissionError("V3-D5 aggregate dataset payload differs")
    data = development_data_from_mapping(dataset)

    started = time.perf_counter()
    probability = torch.full((data.rows, 2), float("nan"), dtype=torch.float64)
    selected_layer = torch.full((data.calls,), -1, dtype=torch.long)
    assigned = torch.zeros(data.rows, dtype=torch.long)
    fold_results: dict[str, Any] = {}
    thresholds: dict[str, float | None] = {}
    selected_lambdas: dict[str, float] = {}
    current_commit = git_output("rev-parse", "HEAD")
    for episode in D5_EPISODES:
        result_path = REPO_ROOT / FOLD_ROOT / f"episode{episode}/result.json"
        result = json_object(result_path)
        payload_path = result_path.parent / str(result["payload"])
        if (
            result.get("status") != "PASS_V3_D5_OOF_OUTER_FOLD"
            or result.get("outer_episode") != episode
            or result.get("source_git_commit") != current_commit
            or result.get("source_worktree_dirty") is not False
            or result.get("dataset_result_sha256") != DATASET_RESULT_SHA256
            or result.get("dataset_payload_sha256") != DATASET_PAYLOAD_SHA256
            or stream_sha256(payload_path) != result.get("payload_sha256")
        ):
            raise PermissionError(f"V3-D5 outer result differs: episode {episode}")
        fold = torch.load(payload_path, map_location="cpu", weights_only=True)
        indices = fold["validation_indices"]
        expected_indices = torch.nonzero(
            data.episode_index == episode, as_tuple=False
        ).flatten()
        if (
            fold.get("schema_version") != D5_OOF_SCHEMA_VERSION
            or fold.get("outer_episode") != episode
            or fold.get("fit_count") != D5_FITS_PER_OUTER
            or not torch.equal(indices, expected_indices)
            or fold["validation_probability"].shape != (indices.numel(), 2)
            or fold["selected_layer"].shape != (indices.numel() // 2,)
        ):
            raise PermissionError(f"V3-D5 outer payload differs: episode {episode}")
        probability[indices] = fold["validation_probability"]
        assigned[indices] += 1
        source_rows = data.source_row[indices][0::2]
        if (
            not torch.equal(data.source_row[indices][0::2], data.source_row[indices][1::2])
            or bool((selected_layer[source_rows] != -1).any())
        ):
            raise PermissionError("V3-D5 outer call assignment differs")
        selected_layer[source_rows] = fold["selected_layer"]
        result_hash = stream_sha256(result_path)
        fold_results[str(episode)] = {
            "path": str(result_path.relative_to(REPO_ROOT)),
            "sha256": result_hash,
            "payload_sha256": result["payload_sha256"],
            "validation_rows": int(indices.numel()),
            "fit_count": int(result["fit_count"]),
            "selected_lambda": float(result["selected_lambda"]),
            "inner_threshold_feasible": bool(result["inner_threshold_feasible"]),
            "inner_selected_threshold": result["inner_selected_threshold"],
            "elapsed_seconds": float(result["elapsed_seconds"]),
        }
        thresholds[str(episode)] = result["inner_selected_threshold"]
        selected_lambdas[str(episode)] = float(result["selected_lambda"])
    if (
        not torch.equal(assigned, torch.ones_like(assigned))
        or not bool(torch.isfinite(probability).all())
        or bool((selected_layer == -1).any())
    ):
        raise PermissionError("V3-D5 aggregate OOF assignment differs")

    summary = summarize_route(
        selected_layer,
        data.unsafe_target,
        data.task_id,
        data.episode_index,
    )
    paired_target = data.unsafe_target.reshape(data.calls, 2, 2)
    early = selected_layer != D5_FALLBACK_LAYER
    selected_index = (selected_layer == 13).long()
    rows = torch.arange(data.calls)
    false_full = torch.zeros(data.calls, dtype=torch.bool)
    false_gripper = torch.zeros(data.calls, dtype=torch.bool)
    false_full[early] = paired_target[rows[early], selected_index[early], 0]
    false_gripper[early] = paired_target[rows[early], selected_index[early], 1]
    task_call = data.task_id[0::2]
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
    infeasible_folds = sum(
        not bool(value["inner_threshold_feasible"])
        for value in fold_results.values()
    )
    gate_checks = {
        "all_6521_calls_13042_rows_and_180_clusters_present": (
            data.calls == 6521
            and data.rows == 13042
            and len(set(zip(data.task_id.tolist(), data.episode_index.tolist()))) == 180
        ),
        "all_18_outer_predictions_finite_and_assigned_once": bool(
            torch.isfinite(probability).all()
        )
        and int(assigned.min()) == 1
        and int(assigned.max()) == 1,
        "all_18_inner_thresholds_feasible": infeasible_folds == 0,
        "minimum_60_safe_clusters": summary.safe_clusters >= D5_MIN_SAFE_CLUSTERS,
        "minimum_5_percent_early_exit_calls": (
            summary.early_exit_fraction >= D5_MIN_EARLY_FRACTION
        ),
        "all_10_tasks_have_nonzero_early_exit_calls": all(
            value > 0 for value in summary.per_task_early_calls
        ),
        "false_safe_cluster_exact_ucb95_at_most_5_percent": (
            summary.false_safe_ucb95 <= D5_MAX_FALSE_SAFE_UCB95
        ),
        "always_defer_rejected": summary.early_exit_calls > 0,
        "calibration_test_rollout_and_active_control_not_run": True,
    }
    status = (
        "PASS_V3_D5_DEVELOPMENT_GATE"
        if all(gate_checks.values())
        else "NEGATIVE_V3_D5_DEVELOPMENT_GATE"
    )
    metric = {
        "full_action_unsafe": target_metrics(
            probability[:, 0], data.unsafe_target[:, 0]
        ),
        "gripper_step_unsafe": target_metrics(
            probability[:, 1], data.unsafe_target[:, 1]
        ),
    }
    estimated_fm_calls = sum(
        selection_counts[layer] * RP_PEP_FM_CALLS[layer]
        for layer in (11, 13, 27)
    )
    behavior_fm_calls = int(dataset_result["behavior_a1"]["behavior_fm_calls"])
    result = {
        "status": status,
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "role": "development_v2",
        "suite": "libero_10",
        "rows": data.rows,
        "policy_calls": data.calls,
        "clusters": 180,
        "outer_fold_count": 18,
        "inner_fold_count_per_outer": 17,
        "fits_per_outer": D5_FITS_PER_OUTER,
        "total_model_fits": 18 * D5_FITS_PER_OUTER,
        "selected_lambdas": selected_lambdas,
        "selected_thresholds": thresholds,
        "infeasible_outer_folds": infeasible_folds,
        "OOF_metrics": metric,
        "selection": {
            "L11": selection_counts[11],
            "L13": selection_counts[13],
            "L27": selection_counts[27],
            "early_exit_calls": summary.early_exit_calls,
            "early_exit_fraction": summary.early_exit_fraction,
            "safe_clusters": summary.safe_clusters,
            "false_safe_clusters": summary.false_safe_clusters,
            "false_safe_cluster_rate": (
                summary.false_safe_clusters / summary.safe_clusters
                if summary.safe_clusters
                else None
            ),
            "false_safe_cluster_ucb95": summary.false_safe_ucb95,
            "false_full_action_calls": int(false_full.sum()),
            "false_gripper_calls": int(false_gripper.sum()),
            "false_safe_calls": int(summary.selected_unsafe.sum()),
            "per_task_early_calls": list(summary.per_task_early_calls),
        },
        "estimated_efficiency": {
            "shadow_rp_pep_fm_calls": estimated_fm_calls,
            "observed_behavior_A1_fm_calls": behavior_fm_calls,
            "estimated_fm_call_reduction_fraction": (
                1.0 - estimated_fm_calls / behavior_fm_calls
            ),
            "risk_head_latency_included": False,
            "measured_end_to_end_latency": False,
        },
        "per_task": per_task,
        "gate_checks": gate_checks,
        "input_sha256": {
            "contract": D5_CONTRACT_SHA256,
            "dataset_result": DATASET_RESULT_SHA256,
            "dataset_payload": DATASET_PAYLOAD_SHA256,
            "outer_fold_results": {
                episode: value["sha256"] for episode, value in fold_results.items()
            },
        },
        "outer_folds": fold_results,
        "access_ledger": {
            "development_v2_payload_opened": True,
            "development_model_fits": 18 * D5_FITS_PER_OUTER,
            "outer_truth_used_for_model_or_threshold_selection": False,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "fresh_rollout": False,
            "active_control": False,
        },
        "next_stage": {
            "authorized": (
                "DESIGN_FRESH_CALIBRATION_PROTOCOL"
                if status == "PASS_V3_D5_DEVELOPMENT_GATE"
                else "D5_NEGATIVE_RESULT_ANALYSIS_ONLY"
            ),
            "reuse_D3_calibration_for_repair": False,
            "independent_test_authorized": False,
            "active_control_authorized": False,
        },
        "claim_boundary": {
            "development_nested_OOF_is_closed_loop_success": False,
            "development_nested_OOF_is_independent_test": False,
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
    payload_path = incomplete / "development_joint_nested_oof.pt"
    torch.save(
        {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "contract_sha256": D5_CONTRACT_SHA256,
            "task_id": data.task_id.clone(),
            "episode_index": data.episode_index.clone(),
            "candidate_layer": data.candidate_layer.clone(),
            "unsafe_target": data.unsafe_target.clone(),
            "OOF_probability": probability.clone(),
            "selected_layer": selected_layer.clone(),
            "selected_unsafe": summary.selected_unsafe.clone(),
            "outer_selected_lambdas": selected_lambdas,
            "outer_selected_thresholds": thresholds,
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
