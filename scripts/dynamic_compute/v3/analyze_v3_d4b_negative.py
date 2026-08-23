#!/usr/bin/env python3
"""Diagnose the frozen D4B negative result without changing any threshold."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.gripper_v2_calibration import (  # noqa: E402
    clopper_pearson_upper,
)
from a1.vla.dynamic_compute.v3.shadow_signal_adapter import (  # noqa: E402
    authenticated_weights_only_load,
    stream_sha256,
)


D4B_ATTESTATION = Path("results/v3/v3_d4b_formal_shadow_result.json")
D4B_ATTESTATION_SHA256 = (
    "53b4462903cc009c5d5b02434d045935aa5b9915790666b1f5ec01a6a1cab27f"
)
D4B_PAYLOAD = Path("reports/v3_d4b_formal_shadow/shadow_payload.pt")
D4B_PAYLOAD_SHA256 = (
    "90d6083ff887e78ee4fc4edacd893a6ca4a3f6efc3a7b72338e9a7a41f21f83f"
)
D3_PREDICTIONS = Path(
    "reports/v3_d3_calibration_result/calibration_predictions.pt"
)
D3_PREDICTIONS_SHA256 = (
    "55dfae85be7609c5fe2319752dc17538b042eed2fec8fb02afce1faabedea607"
)
OUTPUT = Path("reports/v3_d4b_negative_analysis")
GATE_NAMES = (
    "action_consistency",
    "motion_safe",
    "tail_ucb_safe",
    "gripper_safe",
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_file(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("D4B analysis JSON must be an object")
    return value


def evaluate_policy(
    *,
    gates: Mapping[str, torch.Tensor],
    included: tuple[str, ...],
    full_action_truth: torch.Tensor,
    gripper_mismatch_truth: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
) -> dict[str, Any]:
    candidate_safe = torch.ones_like(full_action_truth)
    for name in included:
        candidate_safe &= gates[name]
    rows = candidate_safe.shape[0]
    selected = torch.full((rows,), -1, dtype=torch.long)
    selected[candidate_safe[:, 1]] = 1
    selected[candidate_safe[:, 0]] = 0
    early = selected >= 0
    false_full = torch.zeros(rows, dtype=torch.bool)
    false_gripper = torch.zeros(rows, dtype=torch.bool)
    selected_rows = torch.arange(rows)[early]
    selected_index = selected[early]
    false_full[early] = ~full_action_truth[selected_rows, selected_index]
    false_gripper[early] = gripper_mismatch_truth[selected_rows, selected_index]
    false_safe = false_full | false_gripper
    clusters = task_id * 10 + (episode_index - 30)
    safe_clusters = set(int(value) for value in clusters[early].tolist())
    false_clusters = set(int(value) for value in clusters[false_safe].tolist())
    ucb = clopper_pearson_upper(len(false_clusters), len(safe_clusters))
    return {
        "included_gates": list(included),
        "selection_counts": {
            "L11": int((selected == 0).sum()),
            "L13": int((selected == 1).sum()),
            "L27": int((selected == -1).sum()),
        },
        "early_exit_calls": int(early.sum()),
        "early_exit_fraction": float(early.double().mean()),
        "false_safe_calls": int(false_safe.sum()),
        "false_full_action_calls": int(false_full.sum()),
        "false_gripper_calls": int(false_gripper.sum()),
        "safe_clusters": len(safe_clusters),
        "false_safe_clusters": len(false_clusters),
        "false_safe_cluster_ucb95": ucb,
        "would_pass_frozen_D4B_statistical_gate": (
            bool(early.any())
            and float(early.double().mean()) >= 0.01
            and ucb <= 0.05
        ),
        "runtime_authorized": False,
    }


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D4B negative analysis requires clean worktree")
    attestation_path = REPO_ROOT / D4B_ATTESTATION
    if stream_sha256(attestation_path) != D4B_ATTESTATION_SHA256:
        raise PermissionError("D4B negative attestation SHA differs")
    attestation = json_file(attestation_path)
    if (
        attestation.get("status") != "NEGATIVE_V3_D4B_FORMAL_SHADOW_GATE"
        or attestation.get("authorization", {}).get("next_stage")
        != "D4B_NEGATIVE_RESULT_ANALYSIS_ONLY"
        or attestation.get("authorization", {}).get("active_control_authorized")
        is not False
        or attestation.get("authorization", {}).get("independent_test_authorized")
        is not False
    ):
        raise PermissionError("D4B negative analysis is not authorized")
    payload = authenticated_weights_only_load(
        REPO_ROOT / D4B_PAYLOAD,
        expected_sha256=D4B_PAYLOAD_SHA256,
        context="D4B shadow payload",
    )
    predictions = authenticated_weights_only_load(
        REPO_ROOT / D3_PREDICTIONS,
        expected_sha256=D3_PREDICTIONS_SHA256,
        context="D4B D3 predictions",
    )
    if (
        payload.get("schema_version")
        != "phase-route-vla.v3.d4b-shadow-payload.v1"
        or payload.get("role") != "calibration_v2"
        or payload.get("active_control") is not False
        or payload.get("independent_test_payload_opened") is not False
    ):
        raise PermissionError("D4B analysis payload boundary differs")
    rows = 3516
    gates = {
        name: payload[name]
        for name in GATE_NAMES
    }
    for name, value in gates.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.device.type != "cpu"
            or tuple(value.shape) != (rows, 2)
            or not value.is_contiguous()
        ):
            raise ValueError(f"D4B analysis gate geometry differs: {name}")
    full_action_truth = payload["candidate_to_l27_full_action_safe"]
    gripper_mismatch = predictions["step_mismatch"].reshape(rows, 2)
    task = payload["task_id"]
    episode = payload["episode_index"]
    if (
        full_action_truth.dtype != torch.bool
        or full_action_truth.shape != (rows, 2)
        or gripper_mismatch.dtype != torch.bool
        or gripper_mismatch.shape != (rows, 2)
        or task.shape != (rows,)
        or episode.shape != (rows,)
        or len(set((task * 10 + episode - 30).tolist())) != 100
    ):
        raise ValueError("D4B analysis truth/identity geometry differs")
    policies = {
        "frozen_four_gate": GATE_NAMES,
        "action_plus_gripper_only": (
            "action_consistency",
            "gripper_safe",
        ),
        "leave_out_motion": (
            "action_consistency",
            "tail_ucb_safe",
            "gripper_safe",
        ),
        "leave_out_tail": (
            "action_consistency",
            "motion_safe",
            "gripper_safe",
        ),
        "leave_out_action_consistency": (
            "motion_safe",
            "tail_ucb_safe",
            "gripper_safe",
        ),
        "leave_out_gripper": (
            "action_consistency",
            "motion_safe",
            "tail_ucb_safe",
        ),
    }
    counterfactual = {
        name: evaluate_policy(
            gates=gates,
            included=included,
            full_action_truth=full_action_truth,
            gripper_mismatch_truth=gripper_mismatch,
            task_id=task,
            episode_index=episode,
        )
        for name, included in policies.items()
    }
    if (
        counterfactual["frozen_four_gate"]["selection_counts"]
        != {"L11": 29, "L13": 123, "L27": 3364}
        or counterfactual["frozen_four_gate"]["safe_clusters"] != 58
        or counterfactual["frozen_four_gate"]["false_safe_calls"] != 0
    ):
        raise RuntimeError("D4B analysis does not reproduce formal result")
    stacked = torch.stack([gates[name] for name in GATE_NAMES], dim=-1)
    fail_count = (~stacked).sum(dim=-1)
    unique_veto_calls = Counter()
    for gate_index, name in enumerate(GATE_NAMES):
        unique_veto_calls[name] = int(
            ((fail_count == 1) & ~stacked[:, :, gate_index]).sum()
        )
    cluster = task * 10 + (episode - 30)
    full_candidate_safe = stacked.all(dim=-1)
    covered = set(int(value) for value in cluster[full_candidate_safe.any(dim=1)].tolist())
    uncovered = set(range(100)) - covered
    near_miss_clusters: dict[str, set[int]] = {name: set() for name in GATE_NAMES}
    for row in range(rows):
        cluster_id = int(cluster[row])
        if cluster_id not in uncovered:
            continue
        for layer_index in range(2):
            if int(fail_count[row, layer_index]) == 1:
                missing = int((~stacked[row, layer_index]).nonzero()[0])
                near_miss_clusters[GATE_NAMES[missing]].add(cluster_id)
    gate_support = {}
    for gate_index, name in enumerate(GATE_NAMES):
        gate_support[name] = {
            "L11_pass_calls": int(stacked[:, 0, gate_index].sum()),
            "L13_pass_calls": int(stacked[:, 1, gate_index].sum()),
            "unique_candidate_vetoes": unique_veto_calls[name],
            "uncovered_clusters_with_a_sole_veto_near_miss": len(
                near_miss_clusters[name]
            ),
        }
    result = {
        "status": "PASS_V3_D4B_NEGATIVE_RESULT_ANALYSIS",
        "schema_version": "phase-route-vla.v3.d4b-negative-analysis.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "formal_negative_result_reproduced": True,
        "gate_support": gate_support,
        "uncovered_clusters": len(uncovered),
        "counterfactual_ablations": counterfactual,
        "interpretation_policy": {
            "ablations_are_diagnostic_not_runtime_candidates": True,
            "posthoc_threshold_or_gate_change_allowed": False,
            "same_calibration_data_repair_allowed": False,
            "independent_test_or_active_control_authorized": False,
        },
        "input_sha256": {
            "d4b_attestation": D4B_ATTESTATION_SHA256,
            "d4b_payload": D4B_PAYLOAD_SHA256,
            "d3_predictions": D3_PREDICTIONS_SHA256,
        },
        "next_stage": {
            "authorized": "NEW_DEVELOPMENT_ONLY_JOINT_RELIABILITY_PROTOCOL_DESIGN",
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
    }
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D4B negative analysis refuses to overwrite evidence")
    incomplete.mkdir(parents=True)
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
