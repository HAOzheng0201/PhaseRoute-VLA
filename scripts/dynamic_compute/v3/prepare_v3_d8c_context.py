#!/usr/bin/env python3
"""Build past-only D8C runtime context from authenticated fresh raw caches."""

from __future__ import annotations

import argparse
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
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.vla.dynamic_compute.phase_estimator import (  # noqa: E402
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_PHASE_CHECKPOINT_SHA256,
    PastOnlyHistory,
    pool_visual_features,
    stream_sha256,
    validate_runtime_context,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation import (  # noqa: E402
    D8_TASK_IDS,
)
from a1.vla.dynamic_compute.v3.fresh_confirmation_collection import (  # noqa: E402
    D8C_CONTEXT_SCHEMA_VERSION,
    D8C_RAW_TASK_SCHEMA_VERSION,
    D8C_ROLE,
    D8C_SUITE,
    load_fresh_task_calls,
    resolve_fresh_call_payload,
    validate_d8c_prerequisites,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d8c-context-result.v1"
PHASE_SCHEMA_VERSION = "phase-route-vla.phase-estimator-checkpoint.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _load_phase_estimator(path: Path) -> PhaseStateEstimator:
    if stream_sha256(path) != D2_PHASE_CHECKPOINT_SHA256:
        raise PermissionError("D8C phase checkpoint SHA-256 differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != PHASE_SCHEMA_VERSION:
        raise ValueError("D8C PhaseStateEstimator schema differs")
    config = PhaseEstimatorConfig(**checkpoint["model_config"])
    if config != PhaseEstimatorConfig():
        raise ValueError("D8C PhaseStateEstimator geometry differs")
    model = PhaseStateEstimator(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _stack(values: list[torch.Tensor], name: str) -> torch.Tensor:
    if not values:
        raise ValueError(f"D8C context column is empty: {name}")
    return torch.stack(values).contiguous()


def _run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D8C context construction is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D8C context construction requires a clean worktree")
    prerequisites = validate_d8c_prerequisites(REPO_ROOT)
    expected_raw = (REPO_ROOT / "reports" / "v3_d8_fresh_raw").resolve()
    raw_root = args.raw_root.resolve(strict=True)
    if raw_root != expected_raw:
        raise PermissionError("D8C raw root path differs")
    expected_output = (REPO_ROOT / "reports" / "v3_d8_fresh_context").resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("D8C context output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("D8C refuses to overwrite context evidence")
    phase = _load_phase_estimator(args.phase_checkpoint.resolve(strict=True))

    calls = []
    raw_results: dict[str, dict[str, Any]] = {}
    next_index = 0
    current_commit = git_output("rev-parse", "HEAD")
    for task_id in D8_TASK_IDS:
        task_dir = raw_root / f"task{task_id}"
        result_path = task_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS_V3_D8C_RAW_TASK"
            or result.get("schema_version") != D8C_RAW_TASK_SCHEMA_VERSION
            or result.get("role") != D8C_ROLE
            or result.get("suite") != D8C_SUITE
            or result.get("task_id") != task_id
            or result.get("completed_clusters") != 20
            or result.get("source_git_commit") != current_commit
            or result.get("source_worktree_dirty") is not False
            or result.get("access_ledger", {}).get("official_episode_40_49_opened")
            is not False
        ):
            raise PermissionError(f"D8C raw task {task_id} result differs")
        task_calls = load_fresh_task_calls(
            task_dir, task_id=task_id, dataset_index_start=next_index
        )
        calls.extend(task_calls)
        next_index += len(task_calls)
        raw_results[str(task_id)] = {
            "path": str(result_path),
            "sha256": stream_sha256(result_path),
            "calls": len(task_calls),
            "behavior_successes": int(result["behavior_successes"]),
            "telemetry_sha256": str(result["telemetry_sha256"]),
            "teacher_manifest_sha256": str(result["teacher_manifest_sha256"]),
        }

    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "instruction_summary",
            "vision_crop_summary",
            "vision_crop_mask",
            "phase_embedding",
            "phase_scalars",
            "normalized_proprio",
            "proprio_history",
            "action_history",
            "history_mask",
        )
    }
    histories = PastOnlyHistory()
    records_path = incomplete / "context_records.jsonl"
    opened_payloads: list[Path] = []
    cluster_histogram: Counter[str] = Counter()
    started = time.perf_counter()
    with torch.inference_mode(), records_path.open("w", encoding="utf-8") as record_file:
        for ordinal, call in enumerate(calls, start=1):
            payload_path = resolve_fresh_call_payload(call)
            payload_sha256 = stream_sha256(payload_path)
            with np.load(payload_path, allow_pickle=False) as arrays:
                projected = arrays["projected_features"].astype(np.float32)
                positions = arrays["image_input_idx"].astype(np.int64)
                instruction = arrays["instruction_summary"].astype(np.float32)
                proprio = arrays["normalized_proprio"].astype(np.float32)
                behavior_action = arrays["teacher_normalized_action"].astype(np.float32)
            opened_payloads.append(payload_path)
            if instruction.shape != (3584,) or not np.isfinite(instruction).all():
                raise ValueError("D8C instruction summary differs")
            global_visual, crop_summary, crop_mask = pool_visual_features(
                projected, positions
            )
            history = histories.window_then_commit(call, proprio, behavior_action)
            expected_history = min(call.call_ordinal, 8)
            if int(history.history_mask.sum()) != expected_history:
                raise RuntimeError("D8C past-only history length differs")
            state = phase(
                visual_summary=torch.from_numpy(global_visual[None]),
                instruction_summary=torch.from_numpy(instruction[None]),
                current_proprio=torch.from_numpy(proprio[None]),
                proprio_history=torch.from_numpy(history.proprio_history[None]),
                proprio_history_mask=torch.from_numpy(history.history_mask[None]),
                action_history=torch.from_numpy(history.action_history[None]),
                action_history_mask=torch.from_numpy(history.history_mask[None]),
            )
            row_values = {
                "instruction_summary": torch.from_numpy(instruction).float(),
                "vision_crop_summary": torch.from_numpy(crop_summary).float(),
                "vision_crop_mask": torch.from_numpy(crop_mask).bool(),
                "phase_embedding": state.stage_embedding[0].float().cpu(),
                "phase_scalars": torch.tensor(
                    [
                        float(state.progress[0, 0]),
                        float(state.boundary_prob[0, 0]),
                        float(state.uncertainty[0, 0]),
                    ],
                    dtype=torch.float32,
                ),
                "normalized_proprio": torch.from_numpy(proprio).float(),
                "proprio_history": torch.from_numpy(history.proprio_history).float(),
                "action_history": torch.from_numpy(history.action_history).float(),
                "history_mask": torch.from_numpy(history.history_mask).bool(),
            }
            for name, value in row_values.items():
                values[name].append(value.contiguous())
            record = {
                "dataset_index": call.dataset_index,
                "task_id": call.task_id,
                "replicate_id": call.replicate_id,
                "policy_seed": call.policy_seed,
                "cluster_key": call.cluster_key,
                "call_ordinal": call.call_ordinal,
                "step_id": call.step_id,
                "behavior_exit_layer": call.behavior_exit_layer,
                "source_manifest_line": call.source_manifest_line,
                "source_payload_sha256": payload_sha256,
                "history_valid_rows": expected_history,
                "current_action_excluded_from_history": True,
                "role": D8C_ROLE,
            }
            record_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            record_file.flush()
            cluster_histogram[call.cluster_key] += 1
            if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(calls):
                print(
                    f"D8C context row={ordinal}/{len(calls)} task={call.task_id} "
                    f"replicate={call.replicate_id} call={call.call_ordinal}",
                    flush=True,
                )

    runtime_inputs = {name: _stack(column, name) for name, column in values.items()}
    validate_runtime_context(runtime_inputs, rows=len(calls))
    payload = {
        "schema_version": D8C_CONTEXT_SCHEMA_VERSION,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "D8_readiness_sha256": prerequisites["D8_readiness_sha256"],
        "phase_checkpoint_sha256": D2_PHASE_CHECKPOINT_SHA256,
        "dataset_index": torch.tensor(
            [call.dataset_index for call in calls], dtype=torch.long
        ),
        "task_id": torch.tensor([call.task_id for call in calls], dtype=torch.long),
        "replicate_id": torch.tensor(
            [call.replicate_id for call in calls], dtype=torch.long
        ),
        "policy_seed": torch.tensor(
            [call.policy_seed for call in calls], dtype=torch.long
        ),
        "cluster_keys": [call.cluster_key for call in calls],
        "call_ordinal": torch.tensor(
            [call.call_ordinal for call in calls], dtype=torch.long
        ),
        "step_id": torch.tensor([call.step_id for call in calls], dtype=torch.long),
        "behavior_exit_layer": torch.tensor(
            [call.behavior_exit_layer for call in calls], dtype=torch.long
        ),
        "runtime_inputs": runtime_inputs,
        "task_replicate_identity_is_runtime_input": False,
        "teacher_action_is_runtime_input": False,
        "layer27_is_runtime_input": False,
    }
    payload_path = incomplete / "fresh_context.pt"
    torch.save(payload, payload_path)
    checks = {
        "all_ten_raw_tasks_pass_same_clean_commit": len(raw_results) == 10,
        "all_200_clusters_have_policy_calls": len(cluster_histogram) == 200
        and all(value > 0 for value in cluster_histogram.values()),
        "all_raw_npz_payloads_opened_once": len(opened_payloads) == len(calls),
        "past_only_history_and_runtime_geometry_exact": True,
        "fresh_identity_not_runtime_feature": True,
        "layer27_and_teacher_action_not_runtime_inputs": True,
        "official_episode_40_49_not_opened": all(
            ":episode" not in call.cluster_key for call in calls
        ),
        "no_router_scoring_training_threshold_or_control": True,
    }
    result = {
        "status": "PASS_V3_D8C_CONTEXT"
        if all(checks.values())
        else "FAIL_V3_D8C_CONTEXT",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": current_commit,
        "source_worktree_dirty": False,
        "role": D8C_ROLE,
        "suite": D8C_SUITE,
        "rows": len(calls),
        "clusters": len(cluster_histogram),
        "raw_task_results": raw_results,
        "payload": payload_path.name,
        "payload_sha256": stream_sha256(payload_path),
        "records": records_path.name,
        "records_sha256": stream_sha256(records_path),
        "phase_checkpoint_sha256": D2_PHASE_CHECKPOINT_SHA256,
        "checks": checks,
        "access_ledger": {
            "fresh_raw_call_payloads_opened": len(opened_payloads),
            "official_episode_40_49_opened": False,
            "final_router_loaded": False,
            "confirmation_gate_inspected": False,
            "gpu_query_or_initialization": 0,
            "active_control": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = incomplete / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        raise RuntimeError("D8C context failed one or more gates")
    shutil.move(str(incomplete), str(output))
    print("PASS_V3_D8C_CONTEXT", flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        incomplete = args.output_dir.resolve().with_name(args.output_dir.name + ".incomplete")
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_V3_D8C_CONTEXT",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
