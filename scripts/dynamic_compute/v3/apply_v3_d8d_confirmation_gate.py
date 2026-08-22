#!/usr/bin/env python3
"""Apply the frozen D8B router once and freeze the prospective D8D gate."""

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

from a1.vla.dynamic_compute.v3.d8_confirmation_scoring import (  # noqa: E402
    D8D_PAYLOAD_SCHEMA_VERSION,
    D8D_RESULT_SCHEMA_VERSION,
    confirmation_data_from_mapping,
    score_frozen_router_predictions,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    stream_sha256,
)
from a1.vla.dynamic_compute.v3.epistemic_ensemble import (  # noqa: E402
    D7_HEAD_COUNT,
    D7_MIN_HEAD_RANGE,
)
from a1.vla.dynamic_compute.v3.final_router import (  # noqa: E402
    D8B_PAYLOAD_SCHEMA_VERSION,
    final_router_from_mapping,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_CLUSTER_COUNT,
    D8_CLUSTERS_PER_TASK,
    D8_CONTRACT_SHA256,
    D8_SCHEDULE_SHA256,
    D8_TASK_IDS,
    load_d8_contract,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation_collection import (  # noqa: E402
    D8C_DATASET_SCHEMA_VERSION,
    D8C_ROLE,
    D8C_SUITE,
)
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_ACTION_THRESHOLD,
    mean_action_cosine_distance,
)
from a1.vla.dynamic_compute.v3.shadow_decision import (  # noqa: E402
    D4_RP_PEP_FM_CALLS,
)


D8C_FORMAL_RESULT = Path("results/v3/v3_d8c_formal_collection_result.json")
D8C_FORMAL_RESULT_SHA256 = (
    "5b0f47de0cefabf6dc6da14860b6a4e7a5cdb34866654bbc5a4d1ed30d72fcf2"
)
D8C_DATASET_RESULT = Path("reports/v3_d8_fresh_dataset/result.json")
D8C_DATASET_RESULT_SHA256 = (
    "3b60e241de444f9cce7839290be39740a3920dcddfe8e2159da25bb3d724ae63"
)
D8C_DATASET_PAYLOAD = Path(
    "reports/v3_d8_fresh_dataset/fresh_confirmation_dataset.pt"
)
D8C_DATASET_PAYLOAD_SHA256 = (
    "411b3d68b2e4326573722a616b5fcf7862fbcc6b85f499be7cdf0877a8889327"
)
D8B_RESULT = Path("reports/v3_d8_final_router/result.json")
D8B_RESULT_SHA256 = (
    "76d209ef3e92dcf2a4edb329337a0481d8976ee2382d634de172904724cda70d"
)
D8B_PAYLOAD = Path("reports/v3_d8_final_router/final_router.pt")
D8B_PAYLOAD_SHA256 = (
    "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830"
)
OUTPUT = Path("reports/v3_d8_confirmation")
EXPECTED_POLICY_CALLS = 7140
EXPECTED_CANDIDATE_ROWS = 14280
EXPECTED_BEHAVIOR_FM_CALLS = 73716


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"D8D JSON must be an object: {path}")
    return dict(value)


def tensor_distribution(value: torch.Tensor) -> dict[str, Any]:
    if (
        value.device.type != "cpu"
        or value.ndim != 1
        or not value.is_floating_point()
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
    ):
        raise PermissionError("D8D distribution tensor differs")
    levels = torch.tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64
    )
    quantiles = torch.quantile(value.double(), levels)
    names = ("min", "p01", "p05", "p50", "p95", "p99", "max")
    return {
        "rows": int(value.numel()),
        "mean": float(value.double().mean()),
        "quantiles": {
            name: float(item) for name, item in zip(names, quantiles)
        },
    }


def _authenticate_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    contract = load_d8_contract(REPO_ROOT)
    truth = contract.get("truth_and_routing", {})
    gate = contract.get("confirmation_gate", {})
    execution = contract.get("execution_order", {})
    if (
        truth.get("route_safe")
        != (
            "A1_action_consistency_AND_head0_gripper_safe_AND_"
            "five_head_full_action_upper_score_safe"
        )
        or truth.get("selection") != "L11_if_safe_else_L13_if_safe_else_L27"
        or truth.get("full_action_unsafe_threshold") != D5_ACTION_THRESHOLD
        or truth.get("full_action_severe_false_ratio_threshold") != 4.0
        or truth.get(
            "confirmation_labels_used_for_refit_threshold_or_feature_selection"
        )
        is not False
        or gate.get("all_criteria_are_conjunctive") is not True
        or gate.get("estimated_reduction_includes_router_latency") is not False
        or execution.get("D8D")
        != "apply_frozen_router_aggregate_gate_and_freeze_result"
    ):
        raise PermissionError("D8D frozen contract semantics differ")

    formal_path = REPO_ROOT / D8C_FORMAL_RESULT
    dataset_result_path = REPO_ROOT / D8C_DATASET_RESULT
    dataset_payload_path = REPO_ROOT / D8C_DATASET_PAYLOAD
    router_result_path = REPO_ROOT / D8B_RESULT
    router_payload_path = REPO_ROOT / D8B_PAYLOAD
    observed = {
        "D8C_formal_result": stream_sha256(formal_path),
        "D8C_dataset_result": stream_sha256(dataset_result_path),
        "D8C_dataset_payload": stream_sha256(dataset_payload_path),
        "D8B_result": stream_sha256(router_result_path),
        "D8B_payload": stream_sha256(router_payload_path),
    }
    expected = {
        "D8C_formal_result": D8C_FORMAL_RESULT_SHA256,
        "D8C_dataset_result": D8C_DATASET_RESULT_SHA256,
        "D8C_dataset_payload": D8C_DATASET_PAYLOAD_SHA256,
        "D8B_result": D8B_RESULT_SHA256,
        "D8B_payload": D8B_PAYLOAD_SHA256,
    }
    if observed != expected:
        raise PermissionError("D8D bound input SHA-256 differs")

    formal = json_object(formal_path)
    dataset_result = json_object(dataset_result_path)
    router_result = json_object(router_result_path)
    if (
        formal.get("status")
        != "PASS_V3_D8C_PROSPECTIVE_COLLECTION_AND_REPLAY"
        or formal.get("clusters") != D8_CLUSTER_COUNT
        or formal.get("clusters_per_task")
        != [D8_CLUSTERS_PER_TASK] * len(D8_TASK_IDS)
        or formal.get("policy_calls") != EXPECTED_POLICY_CALLS
        or formal.get("candidate_rows") != EXPECTED_CANDIDATE_ROWS
        or formal.get("authorization", {}).get("next_stage")
        != "D8D_APPLY_FROZEN_ROUTER_AND_AGGREGATE_CONFIRMATION_GATE"
        or formal.get("authorization", {}).get("refit_or_threshold_change")
        is not False
        or formal.get("authorization", {}).get("open_episode_40_49") is not False
        or formal.get("authorization", {}).get("active_control") is not False
        or formal.get("bound_artifacts", {}).get("D8_contract_sha256")
        != D8_CONTRACT_SHA256
        or formal.get("bound_artifacts", {}).get("D8_schedule_sha256")
        != D8_SCHEDULE_SHA256
        or formal.get("bound_artifacts", {}).get("D8B_payload_sha256")
        != D8B_PAYLOAD_SHA256
        or formal.get("bound_artifacts", {}).get("D8C_dataset_payload_sha256")
        != D8C_DATASET_PAYLOAD_SHA256
    ):
        raise PermissionError("D8D formal D8C authorization differs")
    if (
        dataset_result.get("status") != "PASS_V3_D8C_DATASET"
        or dataset_result.get("policy_calls") != EXPECTED_POLICY_CALLS
        or dataset_result.get("candidate_rows") != EXPECTED_CANDIDATE_ROWS
        or dataset_result.get("payload_sha256") != D8C_DATASET_PAYLOAD_SHA256
        or not all(dataset_result.get("checks", {}).values())
        or dataset_result.get("behavior_a1", {}).get("behavior_fm_calls")
        != EXPECTED_BEHAVIOR_FM_CALLS
        or dataset_result.get("access_ledger", {}).get("final_router_loaded")
        is not False
        or dataset_result.get("access_ledger", {}).get("confirmation_gate_inspected")
        is not False
        or dataset_result.get("access_ledger", {}).get("official_episode_40_49_opened")
        is not False
    ):
        raise PermissionError("D8D D8C dataset metadata differs")
    if (
        router_result.get("status") != "PASS_V3_D8B_FINAL_ROUTER_FROZEN"
        or router_result.get("payload_sha256") != D8B_PAYLOAD_SHA256
        or router_result.get("head_count") != D7_HEAD_COUNT
        or router_result.get("final_model_fits") != D7_HEAD_COUNT
        or router_result.get("access_ledger", {}).get(
            "confirmation_state_or_rollout_accessed"
        )
        is not False
        or router_result.get("access_ledger", {}).get("official_episode_40_49_opened")
        is not False
    ):
        raise PermissionError("D8D frozen router metadata differs")
    return contract, formal, dataset_result, expected


def _load_and_validate_dataset() -> tuple[dict[str, Any], Any]:
    payload = torch.load(
        REPO_ROOT / D8C_DATASET_PAYLOAD, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise PermissionError("D8D D8C payload must be a mapping")
    value = dict(payload)
    if (
        value.get("schema_version") != D8C_DATASET_SCHEMA_VERSION
        or value.get("role") != D8C_ROLE
        or value.get("suite") != D8C_SUITE
        or value.get("feature_dimension") != 97
        or value.get("target_axis_order")
        != ["full_action_unsafe", "gripper_step_unsafe"]
        or value.get("layer27_runtime_visible") is not False
        or value.get("layer27_is_consistency_teacher_only") is not True
        or value.get("task_replicate_identity_is_runtime_input") is not False
        or value.get("router_scored") is not False
        or value.get("confirmation_gate_inspected") is not False
        or value.get("official_episode_40_49_opened") is not False
    ):
        raise PermissionError("D8D D8C payload boundary differs")
    data = confirmation_data_from_mapping(
        value, expected_policy_calls=EXPECTED_POLICY_CALLS
    )
    actions = value.get("candidate_actions")
    l11_delta = value.get("l11_telemetry_action_delta")
    l13_delta = value.get("l13_same_noise_action_delta")
    if (
        not isinstance(actions, torch.Tensor)
        or actions.device.type != "cpu"
        or actions.shape != (EXPECTED_POLICY_CALLS, 3, 8, 7)
        or not actions.is_floating_point()
        or not bool(torch.isfinite(actions).all())
        or not isinstance(l11_delta, torch.Tensor)
        or l11_delta.shape != (EXPECTED_POLICY_CALLS,)
        or not bool(torch.isfinite(l11_delta).all())
        or not isinstance(l13_delta, torch.Tensor)
        or l13_delta.shape != (EXPECTED_POLICY_CALLS,)
        or not bool(torch.isfinite(l13_delta).all())
    ):
        raise PermissionError("D8D truth-source geometry differs")
    actions = actions.detach().cpu().contiguous()
    recomputed_l13 = mean_action_cosine_distance(actions[:, 0], actions[:, 1])
    recomputed_full = torch.stack(
        (
            mean_action_cosine_distance(actions[:, 0], actions[:, 2]),
            mean_action_cosine_distance(actions[:, 1], actions[:, 2]),
        ),
        dim=1,
    )
    recomputed_consistency = torch.stack(
        (
            l11_delta <= D5_ACTION_THRESHOLD,
            l13_delta <= D5_ACTION_THRESHOLD,
        ),
        dim=1,
    )
    recomputed_gripper = (
        (actions[:, :2, :, 6] >= 0.0)
        != (actions[:, 2, :, 6] >= 0.0)[:, None]
    ).any(dim=2)
    recomputed_target = torch.stack(
        (recomputed_full > D5_ACTION_THRESHOLD, recomputed_gripper), dim=2
    )
    if (
        not torch.equal(recomputed_l13, l13_delta)
        or not torch.equal(recomputed_full.reshape(-1), data.full_action_distance)
        or not torch.equal(
            recomputed_consistency.reshape(-1), data.action_consistency
        )
        or not torch.equal(
            recomputed_target.reshape(EXPECTED_CANDIDATE_ROWS, 2),
            data.unsafe_target,
        )
    ):
        raise PermissionError("D8D independent truth recomputation differs")
    return value, data


def _load_router() -> tuple[dict[str, Any], Any]:
    payload = torch.load(
        REPO_ROOT / D8B_PAYLOAD, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise PermissionError("D8D router payload must be a mapping")
    value = dict(payload)
    if (
        value.get("schema_version") != D8B_PAYLOAD_SCHEMA_VERSION
        or value.get("D8_contract_sha256") != D8_CONTRACT_SHA256
        or value.get("D8_schedule_sha256") != D8_SCHEDULE_SHA256
        or value.get("feature_dimension") != 97
        or value.get("head_count") != D7_HEAD_COUNT
        or value.get("confirmation_state_or_rollout_accessed") is not False
        or value.get("calibration_or_test_payload_opened") is not False
        or value.get("gpu_query_or_initialization") != 0
    ):
        raise PermissionError("D8D router payload semantics differ")
    router = final_router_from_mapping(value)
    return value, router


def _summary_mapping(summary: Any) -> dict[str, Any]:
    return {
        "clusters": summary.total_clusters,
        "clusters_per_task": list(summary.clusters_per_task),
        "safe_clusters": summary.safe_clusters,
        "safe_clusters_per_task": list(summary.safe_clusters_per_task),
        "policy_calls": summary.policy_calls,
        "early_exit_calls": summary.early_exit_calls,
        "early_exit_fraction": summary.early_exit_calls / summary.policy_calls,
        "early_exit_calls_per_task": list(summary.early_exit_calls_per_task),
        "false_safe_clusters": summary.false_safe_clusters,
        "false_safe_cluster_rate": (
            summary.false_safe_clusters / summary.safe_clusters
            if summary.safe_clusters
            else None
        ),
        "false_safe_cluster_exact_cp_ucb95": summary.false_safe_ucb95,
        "false_full_action_clusters": summary.false_full_action_clusters,
        "false_gripper_calls": summary.false_gripper_calls,
        "severe_false_full_action_clusters": (
            summary.severe_false_full_action_clusters
        ),
        "nondegenerate_row_fraction": summary.nondegenerate_row_fraction,
        "estimated_fm_reduction_fraction": (
            summary.estimated_fm_reduction_fraction
        ),
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D8D confirmation scoring is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8D confirmation scoring requires a clean worktree")
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D8D refuses to overwrite confirmation evidence")

    started = time.perf_counter()
    _contract, formal, dataset_result, input_hashes = _authenticate_inputs()
    _dataset_payload, data = _load_and_validate_dataset()
    router_payload, router = _load_router()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # This is the sole formal router application and gate aggregation point.
    head_prediction, combined, head_range, full_upper = router.predict(
        data.features, data.candidate_layer
    )
    if (
        not torch.equal(combined[:, 0], full_upper)
        or not torch.equal(combined[:, 1], head_prediction[0, :, 1])
        or not torch.equal(
            head_range,
            head_prediction[:, :, 0].max(dim=0).values
            - head_prediction[:, :, 0].min(dim=0).values,
        )
    ):
        raise PermissionError("D8D router output semantics differ")
    scored = score_frozen_router_predictions(
        data,
        head_prediction,
        runtime_threshold=router.runtime_threshold,
        gripper_threshold=router.gripper_threshold,
        action_consistency_threshold=router.action_consistency_threshold,
        behavior_fm_calls=EXPECTED_BEHAVIOR_FM_CALLS,
    )
    gate_checks = scored.summary.gate_checks()
    status = (
        "PASS_V3_D8_PROSPECTIVE_SHADOW_CONFIRMATION"
        if all(gate_checks.values())
        else "NEGATIVE_V3_D8_PROSPECTIVE_SHADOW_CONFIRMATION"
    )

    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    early = scored.selected_layer != 27
    call_task = data.task_id[0::2]
    call_replicate = data.replicate_id[0::2]
    call_keys = data.cluster_keys[0::2]
    selection_counts = Counter(scored.selected_layer.tolist())
    estimated_fm_calls = sum(
        selection_counts[layer] * D4_RP_PEP_FM_CALLS[layer]
        for layer in (11, 13, 27)
    )

    error_records = []
    for call in torch.nonzero(
        early & scored.selected_unsafe, as_tuple=False
    ).flatten().tolist():
        candidate = int(scored.selected_candidate_index[call])
        row = 2 * call + candidate
        error_records.append(
            {
                "source_row": call,
                "task_id": int(call_task[call]),
                "replicate_id": int(call_replicate[call]),
                "cluster_key": call_keys[call],
                "call_ordinal": int(data.call_ordinal[row]),
                "step_id": int(data.step_id[row]),
                "selected_layer": int(scored.selected_layer[call]),
                "full_action_unsafe": bool(
                    scored.selected_full_action_unsafe[call]
                ),
                "gripper_step_unsafe": bool(scored.selected_gripper_unsafe[call]),
                "severe_full_action_false": bool(
                    scored.severe_false_full_action[call]
                ),
                "full_action_distance": float(
                    scored.selected_full_action_distance[call]
                ),
                "full_action_distance_to_threshold_ratio": float(
                    scored.selected_full_action_distance[call]
                    / router.action_consistency_threshold
                ),
                "action_consistency": bool(data.action_consistency[row]),
                "five_head_full_action_probability": [
                    float(value)
                    for value in scored.head_prediction[:, row, 0].tolist()
                ],
                "five_head_gripper_probability": [
                    float(value)
                    for value in scored.head_prediction[:, row, 1].tolist()
                ],
                "max_full_action_score": float(scored.combined_score[row, 0]),
                "head0_gripper_score": float(scored.combined_score[row, 1]),
                "full_action_head_range": float(scored.full_head_range[row]),
            }
        )
    error_path = incomplete / "false_safe_records.jsonl"
    with error_path.open("w", encoding="utf-8") as output_file:
        for record in error_records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )

    payload_path = incomplete / "confirmation_scoring.pt"
    torch.save(
        {
            "schema_version": D8D_PAYLOAD_SCHEMA_VERSION,
            "D8_contract_sha256": D8_CONTRACT_SHA256,
            "D8_schedule_sha256": D8_SCHEDULE_SHA256,
            "D8C_dataset_payload_sha256": D8C_DATASET_PAYLOAD_SHA256,
            "D8B_router_payload_sha256": D8B_PAYLOAD_SHA256,
            "task_id": data.task_id.clone(),
            "replicate_id": data.replicate_id.clone(),
            "cluster_keys": list(data.cluster_keys),
            "call_ordinal": data.call_ordinal.clone(),
            "step_id": data.step_id.clone(),
            "candidate_layer": data.candidate_layer.clone(),
            "action_consistency": data.action_consistency.clone(),
            "five_head_prediction": scored.head_prediction.clone(),
            "combined_score": scored.combined_score.clone(),
            "full_action_head_range": scored.full_head_range.clone(),
            "candidate_safe": scored.candidate_safe.clone(),
            "selected_layer": scored.selected_layer.clone(),
            "selected_candidate_index": scored.selected_candidate_index.clone(),
            "selected_full_action_unsafe": (
                scored.selected_full_action_unsafe.clone()
            ),
            "selected_gripper_unsafe": scored.selected_gripper_unsafe.clone(),
            "selected_full_action_distance": (
                scored.selected_full_action_distance.clone()
            ),
            "selected_unsafe": scored.selected_unsafe.clone(),
            "severe_false_full_action": scored.severe_false_full_action.clone(),
            "safe_cluster_keys": list(scored.safe_cluster_keys),
            "false_safe_cluster_keys": list(scored.false_safe_cluster_keys),
            "false_full_action_cluster_keys": list(
                scored.false_full_action_cluster_keys
            ),
            "severe_false_full_action_cluster_keys": list(
                scored.severe_false_full_action_cluster_keys
            ),
            "gate_checks": dict(gate_checks),
            "active_control": False,
            "official_episode_40_49_opened": False,
            "router_refit_or_threshold_selection": False,
        },
        payload_path,
    )

    safe_set = set(scored.safe_cluster_keys)
    false_set = set(scored.false_safe_cluster_keys)
    false_full_set = set(scored.false_full_action_cluster_keys)
    severe_set = set(scored.severe_false_full_action_cluster_keys)
    per_task = {}
    for task in D8_TASK_IDS:
        mask = call_task == task
        task_keys = {
            call_keys[index]
            for index in torch.nonzero(mask, as_tuple=False).flatten().tolist()
        }
        per_task[str(task)] = {
            "clusters": len(task_keys),
            "calls": int(mask.sum()),
            "L11": int(((scored.selected_layer == 11) & mask).sum()),
            "L13": int(((scored.selected_layer == 13) & mask).sum()),
            "L27": int(((scored.selected_layer == 27) & mask).sum()),
            "early_exit_calls": int((early & mask).sum()),
            "safe_clusters": len(task_keys & safe_set),
            "false_safe_clusters": len(task_keys & false_set),
            "false_full_action_clusters": len(task_keys & false_full_set),
            "false_gripper_calls": int(
                (scored.selected_gripper_unsafe & mask).sum()
            ),
            "severe_false_full_action_clusters": len(task_keys & severe_set),
        }

    result = {
        "status": status,
        "schema_version": D8D_RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "prospective_generated_state_shadow_confirmation",
        "suite": D8C_SUITE,
        "router": {
            "head_count": D7_HEAD_COUNT,
            "lambda": router_payload["head_states"][0]["l2_lambda"],
            "full_threshold": router.full_threshold,
            "runtime_threshold": router.runtime_threshold,
            "fixed_safety_multiplier": (
                router.runtime_threshold / router.full_threshold
            ),
            "gripper_threshold": router.gripper_threshold,
            "action_consistency_threshold": router.action_consistency_threshold,
            "model_or_normalizer_refits": 0,
            "feature_or_threshold_searches": 0,
        },
        "confirmation": _summary_mapping(scored.summary),
        "selection": {
            "L11": selection_counts[11],
            "L13": selection_counts[13],
            "L27": selection_counts[27],
            "L11_fraction": selection_counts[11] / EXPECTED_POLICY_CALLS,
            "L13_fraction": selection_counts[13] / EXPECTED_POLICY_CALLS,
            "L27_fraction": selection_counts[27] / EXPECTED_POLICY_CALLS,
        },
        "safety_audit": {
            "false_safe_calls": int(scored.selected_unsafe.sum()),
            "false_full_action_calls": int(
                scored.selected_full_action_unsafe.sum()
            ),
            "false_gripper_calls": int(scored.selected_gripper_unsafe.sum()),
            "severe_false_full_action_calls": int(
                scored.severe_false_full_action.sum()
            ),
            "safe_cluster_keys": list(scored.safe_cluster_keys),
            "false_safe_cluster_keys": list(scored.false_safe_cluster_keys),
            "false_full_action_cluster_keys": list(
                scored.false_full_action_cluster_keys
            ),
            "severe_false_full_action_cluster_keys": list(
                scored.severe_false_full_action_cluster_keys
            ),
            "false_safe_records": len(error_records),
            "full_action_truth_threshold": router.action_consistency_threshold,
            "severe_ratio_threshold": 4.0,
            "layer27_is_consistency_teacher_only": True,
        },
        "score_distribution": {
            "max_five_head_full_action": tensor_distribution(
                scored.combined_score[:, 0]
            ),
            "head0_gripper": tensor_distribution(scored.combined_score[:, 1]),
            "full_action_head_range": {
                **tensor_distribution(scored.full_head_range),
                "rows_above_1e-6": int(
                    (scored.full_head_range > D7_MIN_HEAD_RANGE).sum()
                ),
                "fraction_above_1e-6": scored.summary.nondegenerate_row_fraction,
            },
        },
        "estimated_efficiency": {
            "shadow_rp_pep_fm_calls": estimated_fm_calls,
            "observed_behavior_A1_fm_calls": EXPECTED_BEHAVIOR_FM_CALLS,
            "estimated_fm_call_reduction_fraction": (
                scored.summary.estimated_fm_reduction_fraction
            ),
            "five_head_router_latency_included": False,
            "measured_end_to_end_latency": False,
        },
        "behavior_a1_descriptive_only": {
            **formal["behavior_a1"],
            "is_D8_router_closed_loop_result": False,
        },
        "per_task": per_task,
        "gate_checks": gate_checks,
        "input_sha256": input_hashes,
        "artifacts": {
            "payload": payload_path.name,
            "payload_sha256": stream_sha256(payload_path),
            "false_safe_records": error_path.name,
            "false_safe_records_sha256": stream_sha256(error_path),
        },
        "access_ledger": {
            "fresh_confirmation_dataset_opened": True,
            "frozen_final_router_loaded": True,
            "router_predictions_computed": EXPECTED_CANDIDATE_ROWS,
            "confirmation_gate_evaluations": 1,
            "model_refits": 0,
            "threshold_or_feature_searches": 0,
            "gpu_query_or_initialization": 0,
            "official_episode_40_49_opened": False,
            "calibration_or_test_payload_opened": False,
            "active_control": False,
        },
        "next_stage": {
            "authorized": (
                "INDEPENDENT_TEST_V2_PROTOCOL_DESIGN_ONLY"
                if status == "PASS_V3_D8_PROSPECTIVE_SHADOW_CONFIRMATION"
                else "D8_NEGATIVE_RESULT_ANALYSIS_ONLY"
            ),
            "open_episode_40_49_authorized": False,
            "active_control_authorized": False,
            "deployment_authorized": False,
        },
        "claim_boundary": {
            "generated_states_are_official_benchmark_fixed_states": False,
            "prospective_shadow_confirmation_is_closed_loop_D8_success": False,
            "layer27_consistency_is_task_success_or_expert_certificate": False,
            "behavior_A1_success_is_D8_router_success": False,
            "estimated_FM_reduction_is_measured_latency": False,
            "superiority_claim_authorized": False,
            "deployment_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
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
