#!/usr/bin/env python3
"""Build the authenticated V3-D5 development-only joint target dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
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

from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_CANDIDATE_SCHEMA_VERSION,
    D2_CONTEXT_SCHEMA_VERSION,
    D2_DATASET_SCHEMA_VERSION,
    D2_REPLAY_LAYERS,
    D2_ROLE,
    D2_SUITE,
    stream_sha256,
)
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_ACTION_THRESHOLD,
    D5_CONTRACT_SHA256,
    D5_EPISODES,
    D5_SCHEMA_VERSION,
    mean_action_cosine_distance,
)


CONTRACT_VALIDATION = Path(
    "results/v3/v3_d5_development_contract_validation.json"
)
CONTRACT_VALIDATION_SHA256 = (
    "9e758e12f50dbd72ed7c52c608605d491029d0f7332ff56964335ccff488fa75"
)
CONTEXT_RESULT = Path("reports/v3_d2_development_context/result.json")
CONTEXT_RESULT_SHA256 = (
    "b637299ee317caa17472c8202b5477f22b92bd125082dacaec776d222ae7b7a3"
)
CONTEXT_PAYLOAD_SHA256 = (
    "82ea239b9ab8835b8f76dd2b1328b630c49ba04f1f52a8392cca9adeff5089c3"
)
D2_DATASET_RESULT = Path("reports/v3_d2_development_dataset/result.json")
D2_DATASET_RESULT_SHA256 = (
    "469ae38e246e473f9df38ee2f2c8b444d853ba27d949d07090f85187cd576b30"
)
D2_DATASET_PAYLOAD_SHA256 = (
    "d2e21932cadbf683f8607791627a825efc352fd6677a046670de19d65e51433d"
)
CANDIDATE_RESULT_SHA256 = (
    "5b68cd3d3a9008355e008795f8779047db1e828bf48774925cc241f4359f2a73",
    "4d6f47ad8b6a2dbb700a08c4740da20c6b1b5f1d005963a1bb5f0f5fe9d8c964",
    "9175c3fb5574435f3382320bf490fee5afbaacb6e2f397c6cd007de3cbe7c00b",
    "dd5ea95fc94cc545cac89de5284191ebf72a7269ba67723e1535252c21c5a2cd",
)
CANDIDATE_PAYLOAD_SHA256 = (
    "f907ef1c80a16bdbef104aeee28b5b005ac46ec93a5ba7feca4ad11d2769c235",
    "e4a42b7d3bf8e30de058d13c7398474fe7fc13506eb0ad6e90a2a1d7eabec11b",
    "585327487e719739a2e14921e934f2a4a239e7588bc61205573270cd969f036f",
    "6004ced6ebe603d28fc9236839b038cdc2c20dc7f9884e1a703c020d9dcafb04",
)
OUTPUT = Path("reports/v3_d5_development_dataset")
RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d5-dataset-result.v1"
PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d5-joint-development-dataset.v1"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError(f"V3-D5 JSON must be an object: {path}")
    return dict(value)


def authenticated_load(
    path: Path, *, expected_sha256: str, context: str
) -> dict[str, Any]:
    if stream_sha256(path) != expected_sha256:
        raise PermissionError(f"V3-D5 payload SHA differs: {context}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise PermissionError(f"V3-D5 payload must be a mapping: {context}")
    return dict(value)


def load_context() -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = REPO_ROOT / CONTEXT_RESULT
    if stream_sha256(result_path) != CONTEXT_RESULT_SHA256:
        raise PermissionError("V3-D5 context result SHA differs")
    result = json_object(result_path)
    if (
        result.get("status") != "PASS_V3_D2_CONTEXT"
        or result.get("role") != D2_ROLE
        or result.get("payload_sha256") != CONTEXT_PAYLOAD_SHA256
    ):
        raise PermissionError("V3-D5 context result semantics differ")
    payload = authenticated_load(
        result_path.parent / str(result["payload"]),
        expected_sha256=CONTEXT_PAYLOAD_SHA256,
        context="D2 development context",
    )
    if (
        payload.get("schema_version") != D2_CONTEXT_SCHEMA_VERSION
        or payload.get("role") != D2_ROLE
        or payload.get("suite") != D2_SUITE
        or payload.get("teacher_action_is_runtime_input") is not False
        or payload.get("layer27_is_runtime_input") is not False
    ):
        raise PermissionError("V3-D5 context payload semantics differ")
    return result, payload


def load_candidate_actions(context: Mapping[str, Any]) -> torch.Tensor:
    payloads = []
    for shard in range(4):
        result_path = (
            REPO_ROOT
            / f"reports/v3_d2_development_candidates/shard{shard}/result.json"
        )
        if stream_sha256(result_path) != CANDIDATE_RESULT_SHA256[shard]:
            raise PermissionError(f"V3-D5 candidate result SHA differs: {shard}")
        result = json_object(result_path)
        if (
            result.get("status") != "PASS_V3_D2_CANDIDATE_SHARD"
            or result.get("shard_index") != shard
            or result.get("payload_sha256") != CANDIDATE_PAYLOAD_SHA256[shard]
        ):
            raise PermissionError(f"V3-D5 candidate result semantics differ: {shard}")
        payload = authenticated_load(
            result_path.parent / str(result["payload"]),
            expected_sha256=CANDIDATE_PAYLOAD_SHA256[shard],
            context=f"D2 candidate shard {shard}",
        )
        if (
            payload.get("schema_version") != D2_CANDIDATE_SCHEMA_VERSION
            or payload.get("layer27_is_consistency_teacher_only") is not True
            or not torch.equal(
                payload["candidate_layers"],
                torch.tensor(D2_REPLAY_LAYERS, dtype=torch.long),
            )
        ):
            raise PermissionError(f"V3-D5 candidate payload semantics differ: {shard}")
        payloads.append(payload)
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
    merged = {name: value[order].contiguous() for name, value in merged.items()}
    for name in ("dataset_index", "task_id", "episode_index", "call_ordinal"):
        if not torch.equal(merged[name], context[name]):
            raise PermissionError(f"V3-D5 candidate/context identity differs: {name}")
    actions = merged["candidate_actions"]
    if actions.shape != (6521, 3, 8, 7) or not bool(torch.isfinite(actions).all()):
        raise PermissionError("V3-D5 candidate action geometry differs")
    return actions.float().contiguous()


def load_d2_runtime_dataset() -> dict[str, Any]:
    result_path = REPO_ROOT / D2_DATASET_RESULT
    if stream_sha256(result_path) != D2_DATASET_RESULT_SHA256:
        raise PermissionError("V3-D5 D2 dataset result SHA differs")
    result = json_object(result_path)
    if (
        result.get("status") != "PASS_V3_D2_DATASET"
        or result.get("payload_sha256") != D2_DATASET_PAYLOAD_SHA256
    ):
        raise PermissionError("V3-D5 D2 dataset result semantics differ")
    payload = authenticated_load(
        result_path.parent / str(result["payload"]),
        expected_sha256=D2_DATASET_PAYLOAD_SHA256,
        context="D2 97D development dataset",
    )
    if (
        payload.get("schema_version") != D2_DATASET_SCHEMA_VERSION
        or payload.get("teacher_or_layer27_runtime_visible") is not False
        or payload.get("other_candidate_runtime_visible") is not False
        or payload.get("feature_dimension") != 97
    ):
        raise PermissionError("V3-D5 D2 runtime dataset semantics differ")
    return payload


def load_l11_telemetry(
    context_result: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    by_key: dict[tuple[int, int, int], tuple[int, float]] = {}
    telemetry_hashes: dict[str, str] = {}
    behavior_exit_counts: Counter[int] = Counter()
    behavior_fm_calls = 0
    for task in range(10):
        entry = context_result["raw_task_results"][str(task)]
        result_path = REPO_ROOT / f"reports/v3_d2_development_raw/task{task}/result.json"
        if stream_sha256(result_path) != entry["sha256"]:
            raise PermissionError("V3-D5 raw task result SHA differs")
        result = json_object(result_path)
        telemetry_path = result_path.parent / "policy_calls.jsonl"
        observed_hash = stream_sha256(telemetry_path)
        if observed_hash != result["telemetry_sha256"]:
            raise PermissionError("V3-D5 raw telemetry SHA differs")
        telemetry_hashes[str(task)] = observed_hash
        counters: dict[int, int] = defaultdict(int)
        with telemetry_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                prefix = f"libero_10:task{task}:episode"
                episode_text = str(record["episode_id"])
                if not episode_text.startswith(prefix):
                    raise PermissionError("V3-D5 telemetry episode identity differs")
                episode = int(episode_text[len(prefix) :])
                if episode not in D5_EPISODES or int(record["task_id"]) != task:
                    raise PermissionError("V3-D5 telemetry contains sealed identity")
                ordinal = counters[episode]
                counters[episode] += 1
                events = [
                    event
                    for event in record["extra"]["exit_events"]
                    if event.get("layer_idx") == 11
                    and event.get("evaluated") is True
                ]
                if len(events) != 1:
                    raise PermissionError("V3-D5 telemetry lacks one L11 event")
                delta = events[0].get("action_delta")
                if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                    raise PermissionError("V3-D5 L11 action delta is invalid")
                key = (task, episode, ordinal)
                if key in by_key:
                    raise PermissionError("V3-D5 duplicate telemetry key")
                by_key[key] = (int(record["step_id"]), float(delta))
                behavior_exit_counts[int(record["exit_layer"])] += 1
                behavior_fm_calls += int(record["fm_calls"])
    rows = int(context["dataset_index"].numel())
    values = torch.empty(rows, dtype=torch.float64)
    for index in range(rows):
        key = (
            int(context["task_id"][index]),
            int(context["episode_index"][index]),
            int(context["call_ordinal"][index]),
        )
        record = by_key.get(key)
        if record is None or record[0] != int(context["step_id"][index]):
            raise PermissionError("V3-D5 telemetry/context identity differs")
        values[index] = record[1]
    if len(by_key) != rows or not bool(torch.isfinite(values).all()):
        raise PermissionError("V3-D5 telemetry coverage differs")
    return values.contiguous(), {
        "telemetry_sha256": telemetry_hashes,
        "behavior_exit_counts": {
            str(layer): behavior_exit_counts[layer]
            for layer in sorted(behavior_exit_counts)
        },
        "behavior_fm_calls": behavior_fm_calls,
    }


def main() -> None:
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 dataset build requires clean worktree")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D5 dataset build is CPU-only")
    validation_path = REPO_ROOT / CONTRACT_VALIDATION
    if stream_sha256(validation_path) != CONTRACT_VALIDATION_SHA256:
        raise PermissionError("V3-D5 contract validation SHA differs")
    validation = json_object(validation_path)
    if (
        validation.get("status")
        != "PASS_V3_D5_DEVELOPMENT_CONTRACT_FROZEN"
        or validation.get("contract", {}).get("sha256") != D5_CONTRACT_SHA256
        or validation.get("authorization", {}).get("next_stage")
        != "D5_DEVELOPMENT_ONLY_NESTED_OOF_TRAINING"
    ):
        raise PermissionError("V3-D5 contract validation authorization differs")

    output = REPO_ROOT / OUTPUT
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite development dataset")

    started = time.perf_counter()
    context_result, context = load_context()
    actions = load_candidate_actions(context)
    runtime = load_d2_runtime_dataset()
    l11_delta, behavior = load_l11_telemetry(context_result, context)
    l13_delta = mean_action_cosine_distance(actions[:, 0], actions[:, 1])
    action_consistency = torch.stack(
        (l11_delta <= D5_ACTION_THRESHOLD, l13_delta <= D5_ACTION_THRESHOLD),
        dim=1,
    )
    full_distance = torch.stack(
        (
            mean_action_cosine_distance(actions[:, 0], actions[:, 2]),
            mean_action_cosine_distance(actions[:, 1], actions[:, 2]),
        ),
        dim=1,
    )
    full_unsafe = full_distance > D5_ACTION_THRESHOLD
    candidate_state = actions[:, :2, :, 6] >= 0.0
    teacher_state = actions[:, 2, :, 6] >= 0.0
    gripper_unsafe = (candidate_state != teacher_state[:, None]).any(dim=2)
    runtime_gripper = runtime["occurrence"][:, 0].reshape(6521, 2)
    if not torch.equal(gripper_unsafe, runtime_gripper):
        raise PermissionError("V3-D5 gripper target differs from frozen D2 target")
    expected_layer = torch.tensor([11, 13], dtype=torch.long).repeat(6521)
    expected_source = torch.arange(6521).repeat_interleave(2)
    if (
        runtime["features"].shape != (13042, 97)
        or not torch.equal(runtime["candidate_layer"], expected_layer)
        or not torch.equal(runtime["source_row"], expected_source)
        or not torch.equal(runtime["task_id"][0::2], context["task_id"])
        or not torch.equal(runtime["episode_index"][0::2], context["episode_index"])
    ):
        raise PermissionError("V3-D5 frozen runtime dataset identity differs")

    flat_full_unsafe = full_unsafe.reshape(-1).contiguous()
    flat_gripper_unsafe = gripper_unsafe.reshape(-1).contiguous()
    unsafe_target = torch.stack(
        (flat_full_unsafe, flat_gripper_unsafe), dim=1
    ).contiguous()
    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "contract_schema_version": D5_SCHEMA_VERSION,
        "contract_sha256": D5_CONTRACT_SHA256,
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "features": runtime["features"].float().contiguous(),
        "candidate_layer": runtime["candidate_layer"].clone(),
        "source_row": runtime["source_row"].clone(),
        "task_id": runtime["task_id"].clone(),
        "episode_index": runtime["episode_index"].clone(),
        "action_consistency": action_consistency.reshape(-1).contiguous(),
        "unsafe_target": unsafe_target,
        "target_axis_order": ["full_action_unsafe", "gripper_step_unsafe"],
        "full_action_distance": full_distance.reshape(-1).contiguous(),
        "l11_telemetry_action_delta": l11_delta.contiguous(),
        "l13_same_noise_action_delta": l13_delta.contiguous(),
        "layer27_runtime_visible": False,
        "layer27_is_consistency_teacher_only": True,
        "task_episode_identity_is_runtime_input": False,
        "calibration_or_test_payload_opened": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.move(str(output), str(incomplete))
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    payload_path = incomplete / "development_joint_reliability_dataset.pt"
    torch.save(payload, payload_path)
    per_layer = {}
    for layer_index, layer in enumerate((11, 13)):
        per_layer[str(layer)] = {
            "rows": 6521,
            "action_consistency_safe": int(action_consistency[:, layer_index].sum()),
            "full_action_unsafe": int(full_unsafe[:, layer_index].sum()),
            "gripper_step_unsafe": int(gripper_unsafe[:, layer_index].sum()),
            "joint_unsafe": int(
                (full_unsafe[:, layer_index] | gripper_unsafe[:, layer_index]).sum()
            ),
        }
    result = {
        "status": "PASS_V3_D5_DEVELOPMENT_DATASET",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "policy_calls": 6521,
        "candidate_rows": 13042,
        "clusters": 180,
        "feature_dimension": 97,
        "target_axis_order": payload["target_axis_order"],
        "target_support": per_layer,
        "behavior_a1": behavior,
        "input_sha256": {
            "contract": D5_CONTRACT_SHA256,
            "contract_validation": CONTRACT_VALIDATION_SHA256,
            "context_result": CONTEXT_RESULT_SHA256,
            "context_payload": CONTEXT_PAYLOAD_SHA256,
            "d2_dataset_result": D2_DATASET_RESULT_SHA256,
            "d2_dataset_payload": D2_DATASET_PAYLOAD_SHA256,
            "candidate_results": list(CANDIDATE_RESULT_SHA256),
            "candidate_payloads": list(CANDIDATE_PAYLOAD_SHA256),
        },
        "payload": payload_path.name,
        "payload_sha256": stream_sha256(payload_path),
        "checks": {
            "all_authenticated_development_inputs_match": True,
            "candidate_context_and_runtime_row_identity_exact": True,
            "gripper_target_recomputed_and_matches_D2_exactly": True,
            "full_action_target_uses_same_noise_L27_offline_only": True,
            "A1_L11_and_same_noise_L13_consistency_gates_exact": True,
            "calibration_and_independent_test_payload_not_opened": True,
            "no_model_fit_threshold_selection_rollout_or_control": True,
        },
        "access_ledger": {
            "development_context_payload_opened": True,
            "development_candidate_payloads_opened": 4,
            "development_97D_dataset_payload_opened": True,
            "D5_joint_target_distribution_opened_after_contract_freeze": True,
            "calibration_v2_payload_opened": False,
            "independent_test_payload_opened": False,
            "gpu_query_or_initialization": 0,
            "model_fits": 0,
            "active_control": False,
        },
        "claim_boundary": {
            "model_trained": False,
            "runtime_threshold_selected": False,
            "closed_loop_success": False,
            "independent_test_result": False,
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
