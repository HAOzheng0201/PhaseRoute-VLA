#!/usr/bin/env python3
"""Run frozen V3-D4B decisions on calibration artifacts, never control actions."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
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

from a1.vla.dynamic_compute.v3.gripper_v2_calibration import (  # noqa: E402
    clopper_pearson_upper,
)
from a1.vla.dynamic_compute.v3.shadow_decision import (  # noqa: E402
    D4_ACTION_CONSISTENCY_THRESHOLD,
    D4_GRIPPER_THRESHOLD,
    ShadowCandidateSignals,
    decide_shadow,
    summarize_shadow_decisions,
)
from a1.vla.dynamic_compute.v3.shadow_signal_adapter import (  # noqa: E402
    authenticated_weights_only_load,
    mean_action_cosine_distance,
    stream_sha256,
)


OUTPUT = Path("reports/v3_d4b_formal_shadow")
D4A_ATTESTATION = Path("results/v3/v3_d4a_signal_adapter_attestation.json")
D4A_ATTESTATION_SHA256 = (
    "e7b638125567d7f86d97191aefac5e15f637e515c29dea26d4d9fd67b0b5141b"
)
D4A_SIGNALS = Path("reports/v3_d4a_signal_adapter/adapter_signals.pt")
D4A_SIGNALS_SHA256 = (
    "9fb66f57b004acfeb918f845adc39d33a73630abdcb3b96188f7ca65a7e6981c"
)
D3_DATASET = Path(
    "reports/v3_d3_calibration_dataset/calibration_gripper_v2_dataset.pt"
)
D3_DATASET_SHA256 = (
    "5780e5949bc5b1ded15483ef84a08994ed099cd1a7604fb5c1c2082d7db4f005"
)
D3_PREDICTIONS = Path(
    "reports/v3_d3_calibration_result/calibration_predictions.pt"
)
D3_PREDICTIONS_SHA256 = (
    "55dfae85be7609c5fe2319752dc17538b042eed2fec8fb02afce1faabedea607"
)
D3_CONTEXT = Path("reports/v3_d3_calibration_context/calibration_context.pt")
D3_CONTEXT_SHA256 = (
    "56edd68f73b3e9fbc0609a22a25a003dfa6ed02ca8ad2a313bf6852a4e81c506"
)
D3_CONTEXT_RESULT = Path("reports/v3_d3_calibration_context/result.json")
D3_CONTEXT_RESULT_SHA256 = (
    "edc438d48ae71e70ec9ebb75e0f9008125e10953245637c0ae053438add9af06"
)
D3_SHARD_SHA256 = (
    "87f5a8b2a6eec1a2fe9ac49369ca5162a8c59555d5670bfb13f32b027e764eee",
    "f2710e023e4bce718743c1e802f6d4522adefe018133d789f587e096d3006a3c",
    "793f9de24121a10fd2fbe22646182e1675b00a1c1e1afc07725146a86da24bb7",
    "5fbfe3ff2cf6af76eac69381d585a06e6a701f4761852455a78c170047703111",
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_file(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"D4B JSON must be an object: {path}")
    return value


def load_candidate_actions() -> dict[str, torch.Tensor]:
    payloads = []
    for shard, expected in enumerate(D3_SHARD_SHA256):
        payloads.append(
            authenticated_weights_only_load(
                REPO_ROOT
                / f"reports/v3_d3_calibration_candidates/shard{shard}/calibration_candidates.pt",
                expected_sha256=expected,
                context=f"D4B candidate shard {shard}",
            )
        )
    merged = {
        name: torch.cat([payload[name] for payload in payloads]).contiguous()
        for name in (
            "dataset_index",
            "task_id",
            "episode_index",
            "call_ordinal",
            "candidate_actions",
        )
    }
    order = torch.argsort(merged["dataset_index"])
    return {name: value[order].contiguous() for name, value in merged.items()}


def load_l11_telemetry(context: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    result_path = REPO_ROOT / D3_CONTEXT_RESULT
    if stream_sha256(result_path) != D3_CONTEXT_RESULT_SHA256:
        raise PermissionError("D4B D3 context result SHA differs")
    result = json_file(result_path)
    by_key: dict[tuple[int, int, int], tuple[int, float, int, int]] = {}
    telemetry_hashes: dict[str, str] = {}
    behavior_exit_counts: Counter[int] = Counter()
    behavior_fm_calls = 0
    behavior_successes = 0
    for task in range(10):
        entry = result["raw_task_results"][str(task)]
        raw_result_path = REPO_ROOT / f"reports/v3_d3_calibration_raw/task{task}/result.json"
        if stream_sha256(raw_result_path) != entry["sha256"]:
            raise PermissionError("D4B raw task result SHA differs")
        raw_result = json_file(raw_result_path)
        behavior_successes += int(raw_result["successes"])
        telemetry_path = raw_result_path.parent / "policy_calls.jsonl"
        telemetry_sha = stream_sha256(telemetry_path)
        if telemetry_sha != raw_result["telemetry_sha256"]:
            raise PermissionError("D4B raw telemetry SHA differs")
        telemetry_hashes[str(task)] = telemetry_sha
        counters: dict[int, int] = defaultdict(int)
        with telemetry_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                episode_text = record["episode_id"]
                prefix = f"libero_10:task{task}:episode"
                if not episode_text.startswith(prefix):
                    raise PermissionError("D4B telemetry episode identity differs")
                episode = int(episode_text[len(prefix) :])
                if episode not in range(30, 40) or record["task_id"] != task:
                    raise PermissionError("D4B telemetry contains sealed identity")
                ordinal = counters[episode]
                counters[episode] += 1
                events = [
                    event
                    for event in record["extra"]["exit_events"]
                    if event.get("layer_idx") == 11 and event.get("evaluated") is True
                ]
                if len(events) != 1:
                    raise ValueError("D4B telemetry lacks one evaluated L11 event")
                delta = events[0].get("action_delta")
                if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                    raise ValueError("D4B L11 action delta is invalid")
                exit_layer = int(record["exit_layer"])
                fm_calls = int(record["fm_calls"])
                behavior_exit_counts[exit_layer] += 1
                behavior_fm_calls += fm_calls
                key = (task, episode, ordinal)
                if key in by_key:
                    raise ValueError("D4B duplicate telemetry key")
                by_key[key] = (
                    int(record["step_id"]),
                    float(delta),
                    exit_layer,
                    fm_calls,
                )
    rows = int(context["dataset_index"].numel())
    values = torch.empty(rows, dtype=torch.float64)
    for index in range(rows):
        key = (
            int(context["task_id"][index]),
            int(context["episode_index"][index]),
            int(context["call_ordinal"][index]),
        )
        row = by_key.get(key)
        if row is None or row[0] != int(context["step_id"][index]):
            raise PermissionError("D4B telemetry/context key or step differs")
        values[index] = row[1]
    if len(by_key) != rows or not bool(torch.isfinite(values).all()):
        raise PermissionError("D4B telemetry coverage differs")
    return values.contiguous(), {
        "telemetry_sha256": telemetry_hashes,
        "behavior_exit_counts": {
            str(layer): behavior_exit_counts[layer]
            for layer in sorted(behavior_exit_counts)
        },
        "behavior_fm_calls": behavior_fm_calls,
        "behavior_successes": behavior_successes,
    }


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D4B formal shadow requires clean worktree")
    attestation_path = REPO_ROOT / D4A_ATTESTATION
    if stream_sha256(attestation_path) != D4A_ATTESTATION_SHA256:
        raise PermissionError("D4B D4A attestation SHA differs")
    attestation = json_file(attestation_path)
    if (
        attestation.get("status") != "PASS_V3_D4A_SIGNAL_ADAPTER_ATTESTATION"
        or attestation.get("authorization", {}).get(
            "d4b_formal_calibration_shadow_only"
        )
        is not True
        or attestation.get("authorization", {}).get("active_control_authorized")
        is not False
        or attestation.get("authorization", {}).get("independent_test_authorized")
        is not False
    ):
        raise PermissionError("D4B D4A authorization differs")
    dataset = authenticated_weights_only_load(
        REPO_ROOT / D3_DATASET,
        expected_sha256=D3_DATASET_SHA256,
        context="D4B D3 dataset",
    )
    predictions = authenticated_weights_only_load(
        REPO_ROOT / D3_PREDICTIONS,
        expected_sha256=D3_PREDICTIONS_SHA256,
        context="D4B D3 predictions",
    )
    adapted = authenticated_weights_only_load(
        REPO_ROOT / D4A_SIGNALS,
        expected_sha256=D4A_SIGNALS_SHA256,
        context="D4B adapted signals",
    )
    context = authenticated_weights_only_load(
        REPO_ROOT / D3_CONTEXT,
        expected_sha256=D3_CONTEXT_SHA256,
        context="D4B D3 context",
    )
    actions = load_candidate_actions()
    flat_rows = 7032
    source_calls = 3516
    exact_vectors = (
        "task_id",
        "episode_index",
        "candidate_layer",
    )
    for name in exact_vectors:
        if not torch.equal(dataset[name], predictions[name]) or not torch.equal(
            dataset[name], adapted[name]
        ):
            raise PermissionError(f"D4B flat identity differs: {name}")
    if (
        dataset["features"].shape != (flat_rows, 97)
        or not torch.equal(dataset["source_row"], adapted["source_row"])
        or not torch.equal(
            dataset["source_row"], torch.arange(source_calls).repeat_interleave(2)
        )
        or not torch.equal(actions["dataset_index"], context["dataset_index"])
        or actions["candidate_actions"].shape != (source_calls, 3, 8, 7)
        or predictions.get("selected_threshold") != D4_GRIPPER_THRESHOLD
        or not torch.equal(
            predictions["safe_call"], predictions["score"] <= D4_GRIPPER_THRESHOLD
        )
    ):
        raise PermissionError("D4B input geometry or frozen threshold differs")
    l11_delta, behavior = load_l11_telemetry(context)
    candidate_actions = actions["candidate_actions"]
    l13_delta = mean_action_cosine_distance(
        candidate_actions[:, 0].contiguous(),
        candidate_actions[:, 1].contiguous(),
    ).double()
    truth_delta = torch.stack(
        (
            mean_action_cosine_distance(
                candidate_actions[:, 0].contiguous(),
                candidate_actions[:, 2].contiguous(),
            ),
            mean_action_cosine_distance(
                candidate_actions[:, 1].contiguous(),
                candidate_actions[:, 2].contiguous(),
            ),
        ),
        dim=1,
    ).double()
    action_consistency = torch.stack(
        (
            l11_delta <= D4_ACTION_CONSISTENCY_THRESHOLD,
            l13_delta <= D4_ACTION_CONSISTENCY_THRESHOLD,
        ),
        dim=1,
    )
    motion_safe = adapted["motion_safe"].reshape(source_calls, 2)
    tail_safe = adapted["tail_ucb_safe"].reshape(source_calls, 2)
    gripper_safe = predictions["safe_call"].reshape(source_calls, 2)
    step_mismatch = predictions["step_mismatch"].reshape(source_calls, 2)
    full_action_safe = truth_delta <= D4_ACTION_CONSISTENCY_THRESHOLD
    decisions = []
    selected_layer = torch.empty(source_calls, dtype=torch.long)
    false_safe = torch.zeros(source_calls, dtype=torch.bool)
    false_full_action = torch.zeros(source_calls, dtype=torch.bool)
    false_gripper = torch.zeros(source_calls, dtype=torch.bool)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in range(source_calls):
        candidates = []
        for layer_index, layer in enumerate((11, 13)):
            candidates.append(
                ShadowCandidateSignals(
                    layer=layer,
                    original_action_consistency=bool(
                        action_consistency[row, layer_index]
                    ),
                    motion_safe=bool(motion_safe[row, layer_index]),
                    tail_ucb_safe=bool(tail_safe[row, layer_index]),
                    gripper_score=float(
                        predictions["score"].reshape(source_calls, 2)[
                            row, layer_index
                        ]
                    ),
                )
            )
        decision = decide_shadow(candidates[0], candidates[1])
        decisions.append(decision)
        selected_layer[row] = decision.selected_layer
        selected_index = 0 if decision.selected_layer == 11 else 1
        if decision.would_early_exit:
            false_full_action[row] = not bool(full_action_safe[row, selected_index])
            false_gripper[row] = bool(step_mismatch[row, selected_index])
            false_safe[row] = false_full_action[row] or false_gripper[row]
        record = {
            "source_row": row,
            "task_id": int(context["task_id"][row]),
            "episode_index": int(context["episode_index"][row]),
            "call_ordinal": int(context["call_ordinal"][row]),
            "step_id": int(context["step_id"][row]),
            **decision.to_record(),
            "selected_truth": (
                None
                if not decision.would_early_exit
                else {
                    "full_action_safe": not bool(false_full_action[row]),
                    "gripper_safe": not bool(false_gripper[row]),
                    "false_safe": bool(false_safe[row]),
                }
            ),
        }
        records.append(record)
    summary = summarize_shadow_decisions(decisions)
    task = context["task_id"]
    episode = context["episode_index"]
    cluster_id = task * 10 + (episode - 30)
    safe_clusters: set[int] = set()
    false_clusters: set[int] = set()
    for row in range(source_calls):
        if selected_layer[row] in (11, 13):
            cluster = int(cluster_id[row])
            safe_clusters.add(cluster)
            if bool(false_safe[row]):
                false_clusters.add(cluster)
    ucb = clopper_pearson_upper(len(false_clusters), len(safe_clusters))
    early_fraction = float(summary["early_exit_fraction"])
    gate_checks = {
        "all_3516_calls_and_100_clusters_present": (
            source_calls == 3516 and len(set(cluster_id.tolist())) == 100
        ),
        "nontrivial_early_exit_fraction_at_least_1_percent": (
            early_fraction >= 0.01
        ),
        "always_defer_rejected": int(summary["early_exit_calls"]) > 0,
        "false_safe_cluster_ucb95_at_most_5_percent": ucb <= 0.05,
        "shadow_never_changed_or_returned_action": True,
        "independent_test_and_active_control_not_run": True,
    }
    status = (
        "PASS_V3_D4B_FORMAL_SHADOW_GATE"
        if all(gate_checks.values())
        else "NEGATIVE_V3_D4B_FORMAL_SHADOW_GATE"
    )
    per_task = {}
    for task_id in range(10):
        mask = task == task_id
        per_task[str(task_id)] = {
            "calls": int(mask.sum()),
            "L11": int(((selected_layer == 11) & mask).sum()),
            "L13": int(((selected_layer == 13) & mask).sum()),
            "L27": int(((selected_layer == 27) & mask).sum()),
            "false_safe_calls": int((false_safe & mask).sum()),
        }
    selected_estimated_fm_calls = int(summary["estimated_rp_pep_fm_calls"])
    behavior_fm_calls = int(behavior["behavior_fm_calls"])
    result = {
        "status": status,
        "schema_version": "phase-route-vla.v3.d4b-formal-shadow-result.v1",
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "calibration_v2",
        "source_calls": source_calls,
        "clusters": 100,
        "selection": summary,
        "behavior_a1": {
            **behavior,
            "shadow_did_not_change_success_or_trajectory": True,
        },
        "estimated_efficiency": {
            "shadow_rp_pep_fm_calls": selected_estimated_fm_calls,
            "observed_behavior_a1_fm_calls": behavior_fm_calls,
            "estimated_fm_call_reduction_fraction": (
                1.0 - selected_estimated_fm_calls / behavior_fm_calls
            ),
            "risk_head_and_adapter_latency_included": False,
            "measured_end_to_end_latency": False,
        },
        "safety_audit": {
            "false_safe_calls": int(false_safe.sum()),
            "false_full_action_calls": int(false_full_action.sum()),
            "false_gripper_calls": int(false_gripper.sum()),
            "safe_clusters": len(safe_clusters),
            "false_safe_clusters": len(false_clusters),
            "false_safe_cluster_rate": (
                len(false_clusters) / len(safe_clusters) if safe_clusters else None
            ),
            "false_safe_cluster_ucb95": ucb,
            "maximum_allowed_ucb95": 0.05,
            "full_action_truth_threshold": D4_ACTION_CONSISTENCY_THRESHOLD,
            "layer27_is_consistency_teacher_only": True,
        },
        "per_task": per_task,
        "gate_checks": gate_checks,
        "input_sha256": {
            "d4a_attestation": D4A_ATTESTATION_SHA256,
            "d4a_signals": D4A_SIGNALS_SHA256,
            "d3_dataset": D3_DATASET_SHA256,
            "d3_predictions": D3_PREDICTIONS_SHA256,
            "d3_context": D3_CONTEXT_SHA256,
            "d3_candidate_shards": list(D3_SHARD_SHA256),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "access_ledger": {
            "role": "calibration_v2_only",
            "model_fit_or_threshold_search": 0,
            "fresh_rollout": False,
            "gpu_query_or_initialization": 0,
            "independent_test_payload_opened": False,
            "active_control": False,
        },
        "next_stage": {
            "authorized": (
                "V3_D5_ACTIVE_CONTROL_PROTOCOL_DESIGN_ONLY"
                if status == "PASS_V3_D4B_FORMAL_SHADOW_GATE"
                else "D4B_NEGATIVE_RESULT_ANALYSIS_ONLY"
            ),
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
        "claim_boundary": {
            "closed_loop_success_improvement": False,
            "measured_latency_improvement": False,
            "deployment_authorized": False,
            "superiority_claim_authorized": False,
        },
    }
    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D4B refuses to overwrite formal shadow evidence")
    incomplete.mkdir(parents=True)
    record_path = incomplete / "shadow_records.jsonl"
    with record_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
    payload_path = incomplete / "shadow_payload.pt"
    torch.save(
        {
            "schema_version": "phase-route-vla.v3.d4b-shadow-payload.v1",
            "role": "calibration_v2",
            "task_id": task.clone(),
            "episode_index": episode.clone(),
            "selected_layer": selected_layer.clone(),
            "action_consistency": action_consistency.clone(),
            "motion_safe": motion_safe.clone(),
            "tail_ucb_safe": tail_safe.clone(),
            "gripper_safe": gripper_safe.clone(),
            "candidate_to_l27_full_action_safe": full_action_safe.clone(),
            "false_safe": false_safe.clone(),
            "active_control": False,
            "independent_test_payload_opened": False,
        },
        payload_path,
    )
    result["artifacts"] = {
        "records": "shadow_records.jsonl",
        "records_sha256": stream_sha256(record_path),
        "payload": "shadow_payload.pt",
        "payload_sha256": stream_sha256(payload_path),
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
