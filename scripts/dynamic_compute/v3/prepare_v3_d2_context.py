#!/usr/bin/env python3
"""Build V3-D2 past-only context tensors from the ten raw task caches."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.phase_estimator import (  # noqa: E402
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_CONTEXT_SCHEMA_VERSION,
    D2_PHASE_CHECKPOINT_SHA256,
    D2_ROLE,
    D2_SELECTION_SHA256,
    D2_SUITE,
    D2_TASK_IDS,
    PastOnlyHistory,
    load_development_selection,
    load_task_calls,
    pool_visual_features,
    resolve_call_payload,
    stream_sha256,
    validate_frozen_d2_inputs,
    validate_runtime_context,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d2-context-result.v1"
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
        raise PermissionError("V3-D2 phase checkpoint SHA-256 differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != PHASE_SCHEMA_VERSION:
        raise ValueError("V3-D2 PhaseStateEstimator schema differs")
    config = PhaseEstimatorConfig(**checkpoint["model_config"])
    expected = PhaseEstimatorConfig()
    if config != expected:
        raise ValueError("V3-D2 PhaseStateEstimator geometry differs")
    model = PhaseStateEstimator(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _stack(values: list[torch.Tensor], name: str) -> torch.Tensor:
    if not values:
        raise ValueError(f"V3-D2 context column is empty: {name}")
    return torch.stack(values).contiguous()


def _run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise PermissionError("V3-D2 context construction is CPU-only")
    validate_frozen_d2_inputs(REPO_ROOT)
    load_development_selection(REPO_ROOT)
    expected_raw = (REPO_ROOT / "reports" / "v3_d2_development_raw").resolve()
    raw_root = args.raw_root.resolve(strict=True)
    if raw_root != expected_raw:
        raise PermissionError("V3-D2 raw root path differs")
    expected_output = (REPO_ROOT / "reports" / "v3_d2_development_context").resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D2 context output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D2 refuses to overwrite context evidence")
    phase = _load_phase_estimator(args.phase_checkpoint.resolve(strict=True))

    calls = []
    raw_results: dict[str, dict[str, Any]] = {}
    next_index = 0
    for task_id in D2_TASK_IDS:
        task_dir = raw_root / f"task{task_id}"
        result_path = task_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS_V3_D2_RAW_TASK"
            or result.get("role") != D2_ROLE
            or result.get("suite") != D2_SUITE
            or result.get("task_id") != task_id
            or result.get("source_worktree_dirty") is not False
        ):
            raise PermissionError(f"V3-D2 raw task {task_id} result differs")
        task_calls = load_task_calls(
            task_dir, task_id=task_id, dataset_index_start=next_index
        )
        calls.extend(task_calls)
        next_index += len(task_calls)
        raw_results[str(task_id)] = {
            "path": str(result_path),
            "sha256": stream_sha256(result_path),
            "calls": len(task_calls),
            "successes": int(result["successes"]),
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
    episode_histogram: Counter[int] = Counter()
    started = time.perf_counter()
    with torch.inference_mode(), records_path.open("w", encoding="utf-8") as record_file:
        for ordinal, call in enumerate(calls, start=1):
            payload_path = resolve_call_payload(call)
            payload_sha256 = stream_sha256(payload_path)
            with np.load(payload_path, allow_pickle=False) as arrays:
                projected = arrays["projected_features"].astype(np.float32)
                positions = arrays["image_input_idx"].astype(np.int64)
                instruction = arrays["instruction_summary"].astype(np.float32)
                proprio = arrays["normalized_proprio"].astype(np.float32)
                behavior_action = arrays["teacher_normalized_action"].astype(np.float32)
            opened_payloads.append(payload_path)
            if instruction.shape != (3584,) or not np.isfinite(instruction).all():
                raise ValueError("V3-D2 instruction summary differs")
            global_visual, crop_summary, crop_mask = pool_visual_features(
                projected, positions
            )
            history = histories.window_then_commit(call, proprio, behavior_action)
            expected_history = min(call.call_ordinal, 8)
            if int(history.history_mask.sum()) != expected_history:
                raise RuntimeError("V3-D2 past-only history length differs")
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
                "episode_index": call.episode_index,
                "call_ordinal": call.call_ordinal,
                "step_id": call.step_id,
                "behavior_exit_layer": call.behavior_exit_layer,
                "source_manifest_line": call.source_manifest_line,
                "source_payload_sha256": payload_sha256,
                "history_valid_rows": expected_history,
                "current_action_excluded_from_history": True,
                "role": D2_ROLE,
            }
            record_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            record_file.flush()
            episode_histogram[call.episode_index] += 1
            if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(calls):
                print(
                    f"V3-D2 context row={ordinal}/{len(calls)} "
                    f"task={call.task_id} episode={call.episode_index} "
                    f"call={call.call_ordinal}",
                    flush=True,
                )

    runtime_inputs = {name: _stack(column, name) for name, column in values.items()}
    validate_runtime_context(runtime_inputs, rows=len(calls))
    payload = {
        "schema_version": D2_CONTEXT_SCHEMA_VERSION,
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "selection_sha256": D2_SELECTION_SHA256,
        "phase_checkpoint_sha256": D2_PHASE_CHECKPOINT_SHA256,
        "dataset_index": torch.tensor(
            [call.dataset_index for call in calls], dtype=torch.long
        ),
        "task_id": torch.tensor([call.task_id for call in calls], dtype=torch.long),
        "episode_index": torch.tensor(
            [call.episode_index for call in calls], dtype=torch.long
        ),
        "call_ordinal": torch.tensor(
            [call.call_ordinal for call in calls], dtype=torch.long
        ),
        "step_id": torch.tensor([call.step_id for call in calls], dtype=torch.long),
        "behavior_exit_layer": torch.tensor(
            [call.behavior_exit_layer for call in calls], dtype=torch.long
        ),
        "runtime_inputs": runtime_inputs,
        "task_episode_identity_is_runtime_input": False,
        "teacher_action_is_runtime_input": False,
        "layer27_is_runtime_input": False,
    }
    payload_path = incomplete / "development_context.pt"
    torch.save(payload, payload_path)
    first_calls = [call for call in calls if call.call_ordinal == 0]
    checks = {
        "all_ten_raw_task_results_pass_and_clean": len(raw_results) == 10,
        "only_development_v2_payloads_opened": len(opened_payloads) == len(calls)
        and all(call.episode_index in range(12, 30) for call in calls),
        "all_180_task_episode_groups_have_calls": len(first_calls) == 180,
        "past_only_history_and_episode_reset_exact": all(
            values["history_mask"][call.dataset_index].sum().item() == 0
            for call in first_calls
        ),
        "runtime_context_geometry_and_finiteness_exact": True,
        "phase_checkpoint_frozen_and_current": not phase.training
        and not any(parameter.requires_grad for parameter in phase.parameters()),
        "no_candidate_teacher_label_training_or_control": True,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS_V3_D2_CONTEXT" if passed else "FAIL_V3_D2_CONTEXT",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "rows": len(calls),
        "groups": len(first_calls),
        "task_rows": {
            task: sum(call.task_id == int(task) for call in calls)
            for task in map(str, D2_TASK_IDS)
        },
        "episode_rows": {
            str(episode): episode_histogram[episode] for episode in range(12, 30)
        },
        "raw_task_results": raw_results,
        "phase_checkpoint": str(args.phase_checkpoint.resolve()),
        "phase_checkpoint_sha256": D2_PHASE_CHECKPOINT_SHA256,
        "records": records_path.name,
        "records_sha256": stream_sha256(records_path),
        "payload": payload_path.name,
        "payload_sha256": stream_sha256(payload_path),
        "elapsed_seconds": time.perf_counter() - started,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "checks": checks,
        "claim_boundary": {
            "development_v2_payload_opened": True,
            "candidate_actions_computed": False,
            "gripper_targets_computed": False,
            "model_trained": False,
            "calibration_or_test_payload_opened": False,
            "active_control": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V3-D2 context failed one or more gates")
    incomplete.rename(output)
    print("PASS_V3_D2_CONTEXT", flush=True)


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
                        "status": "ABORT_V3_D2_CONTEXT",
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
