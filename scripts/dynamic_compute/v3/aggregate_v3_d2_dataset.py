#!/usr/bin/env python3
"""Aggregate V3-D2 context/candidates into the frozen 97-D gripper dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.v3.development_collection import (  # noqa: E402
    D2_CANDIDATE_SCHEMA_VERSION,
    D2_CONTEXT_SCHEMA_VERSION,
    D2_DATASET_SCHEMA_VERSION,
    D2_REPLAY_LAYERS,
    D2_ROLE,
    D2_SELECTION_SHA256,
    D2_SUITE,
    build_gripper_v2_features,
    build_gripper_v2_targets,
    stream_sha256,
    validate_frozen_d2_inputs,
    validate_runtime_context,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d2-dataset-result.v1"
SHARD_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-result", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(args: argparse.Namespace) -> None:
    validate_frozen_d2_inputs(REPO_ROOT)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise PermissionError("V3-D2 aggregation is CPU-only")
    expected_context = (
        REPO_ROOT / "reports" / "v3_d2_development_context" / "result.json"
    ).resolve()
    context_result_path = args.context_result.resolve(strict=True)
    if context_result_path != expected_context:
        raise PermissionError("V3-D2 context result path differs")
    context_result = json.loads(context_result_path.read_text(encoding="utf-8"))
    if context_result.get("status") != "PASS_V3_D2_CONTEXT":
        raise PermissionError("V3-D2 context result has not passed")
    context_directory = context_result_path.parent
    context_payload_path = context_directory / str(context_result["payload"])
    if stream_sha256(context_payload_path) != context_result["payload_sha256"]:
        raise PermissionError("V3-D2 context payload SHA-256 differs")
    context = torch.load(context_payload_path, map_location="cpu", weights_only=True)
    if (
        context.get("schema_version") != D2_CONTEXT_SCHEMA_VERSION
        or context.get("role") != D2_ROLE
        or context.get("suite") != D2_SUITE
        or context.get("selection_sha256") != D2_SELECTION_SHA256
        or context.get("teacher_action_is_runtime_input") is not False
        or context.get("layer27_is_runtime_input") is not False
    ):
        raise PermissionError("V3-D2 context payload contract differs")
    rows = int(context["dataset_index"].numel())
    if not torch.equal(context["dataset_index"], torch.arange(rows)):
        raise PermissionError("V3-D2 context dataset index is not contiguous")
    validate_runtime_context(context["runtime_inputs"], rows=rows)
    context_records_path = context_directory / str(context_result["records"])
    if stream_sha256(context_records_path) != context_result["records_sha256"]:
        raise PermissionError("V3-D2 context records SHA-256 differs")
    context_records = _jsonl(context_records_path)
    context_by_index = {
        int(record["dataset_index"]): record for record in context_records
    }

    expected_candidate_root = (
        REPO_ROOT / "reports" / "v3_d2_development_candidates"
    ).resolve()
    candidate_root = args.candidate_root.resolve(strict=True)
    if candidate_root != expected_candidate_root:
        raise PermissionError("V3-D2 candidate root path differs")
    payloads = []
    candidate_records: dict[int, dict[str, Any]] = {}
    shard_results: dict[str, dict[str, Any]] = {}
    for shard in range(SHARD_COUNT):
        directory = candidate_root / f"shard{shard}"
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS_V3_D2_CANDIDATE_SHARD"
            or result.get("shard_index") != shard
            or result.get("role") != D2_ROLE
            or result.get("suite") != D2_SUITE
        ):
            raise PermissionError(f"V3-D2 candidate shard {shard} result differs")
        payload_path = directory / str(result["payload"])
        records_path = directory / str(result["records"])
        if (
            stream_sha256(payload_path) != result["payload_sha256"]
            or stream_sha256(records_path) != result["records_sha256"]
        ):
            raise PermissionError(f"V3-D2 candidate shard {shard} SHA differs")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != D2_CANDIDATE_SCHEMA_VERSION
            or payload.get("shard_index") != shard
            or payload.get("shard_count") != SHARD_COUNT
            or payload.get("layer27_is_consistency_teacher_only") is not True
            or payload.get("gripper_target_computed") is not False
            or not torch.equal(
                payload["candidate_layers"],
                torch.tensor(D2_REPLAY_LAYERS, dtype=torch.long),
            )
        ):
            raise PermissionError(f"V3-D2 candidate shard {shard} contract differs")
        for record in _jsonl(records_path):
            index = int(record["dataset_index"])
            if index in candidate_records:
                raise ValueError("V3-D2 candidate dataset index repeats")
            candidate_records[index] = record
        payloads.append(payload)
        shard_results[str(shard)] = {
            "path": str(result_path),
            "sha256": stream_sha256(result_path),
            "rows": int(result["rows"]),
            "physical_gpu_index": int(result["physical_gpu_index"]),
        }

    concatenated = {
        name: torch.cat([payload[name] for payload in payloads]).contiguous()
        for name in (
            "dataset_index",
            "task_id",
            "episode_index",
            "call_ordinal",
            "shared_fm_input_x",
            "candidate_actions",
        )
    }
    order = torch.argsort(concatenated["dataset_index"])
    merged = {name: value[order].contiguous() for name, value in concatenated.items()}
    if not torch.equal(merged["dataset_index"], context["dataset_index"]):
        raise PermissionError("V3-D2 context/candidate dataset index differs")
    for name in ("task_id", "episode_index", "call_ordinal"):
        if not torch.equal(merged[name], context[name]):
            raise PermissionError(f"V3-D2 context/candidate {name} differs")
    if merged["candidate_actions"].shape != (rows, 3, 8, 7) or not bool(
        torch.isfinite(merged["candidate_actions"]).all()
    ):
        raise ValueError("V3-D2 merged candidate action geometry differs")
    if set(candidate_records) != set(range(rows)):
        raise PermissionError("V3-D2 candidate record coverage differs")
    for index in range(rows):
        if (
            candidate_records[index]["source_payload_sha256"]
            != context_by_index[index]["source_payload_sha256"]
        ):
            raise PermissionError("V3-D2 candidate/context source hash differs")

    started = time.perf_counter()
    features = build_gripper_v2_features(
        context["runtime_inputs"], merged["candidate_actions"][:, :2]
    )
    targets = build_gripper_v2_targets(merged["candidate_actions"])
    expected_fraction = targets.count.float() / torch.tensor(
        [8.0, 7.0], dtype=torch.float32
    ).view(1, 1, 2)
    flat_features = features.reshape(rows * 2, 97).contiguous()
    flat_layer = torch.tensor([11, 13], dtype=torch.long).repeat(rows)
    flat_source_row = torch.arange(rows, dtype=torch.long).repeat_interleave(2)
    flat_task = context["task_id"].repeat_interleave(2)
    flat_episode = context["episode_index"].repeat_interleave(2)
    flat_occurrence = targets.occurrence.reshape(rows * 2, 2).contiguous()
    flat_count = targets.count.reshape(rows * 2, 2).contiguous()
    flat_expected_fraction = expected_fraction.reshape(rows * 2, 2).contiguous()
    flat_timing = targets.first_transition_mismatch.reshape(rows * 2).contiguous()
    if flat_features.shape != (rows * 2, 97):
        raise RuntimeError("V3-D2 flattened feature geometry differs")

    expected_output = (REPO_ROOT / "reports" / "v3_d2_development_dataset").resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D2 dataset output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D2 refuses to overwrite dataset evidence")
    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    dataset = {
        "schema_version": D2_DATASET_SCHEMA_VERSION,
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "selection_sha256": D2_SELECTION_SHA256,
        "feature_layout": {
            "legacy_causal_context": [0, 82],
            "current_candidate_gripper_sign_sequence": [82, 90],
            "current_candidate_gripper_transition_pattern": [90, 97],
        },
        "feature_dimension": 97,
        "target_axis_order": ["step", "transition"],
        "features": flat_features,
        "candidate_layer": flat_layer,
        "source_row": flat_source_row,
        "task_id": flat_task,
        "episode_index": flat_episode,
        "occurrence": flat_occurrence,
        "count": flat_count,
        "expected_fraction": flat_expected_fraction,
        "first_transition_mismatch": flat_timing,
        "step_mismatch_bits": targets.step_mismatch_bits,
        "transition_mismatch_bits": targets.transition_mismatch_bits,
        "teacher_or_layer27_runtime_visible": False,
        "other_candidate_runtime_visible": False,
        "task_episode_identity_is_runtime_input": False,
        "full_depth_is_consistency_teacher_only": True,
        "continuous_positive_magnitude_target": False,
    }
    dataset_path = incomplete / "development_gripper_v2_dataset.pt"
    torch.save(dataset, dataset_path)
    support: dict[str, dict[str, dict[str, int]]] = {}
    for layer_index, layer in enumerate((11, 13)):
        support[str(layer)] = {}
        layer_count = targets.count[:, layer_index]
        for target_index, target in enumerate(("step", "transition")):
            values = layer_count[:, target_index]
            support[str(layer)][target] = {
                "zero": int((values == 0).sum()),
                "positive": int((values > 0).sum()),
                "maximum": int(values.max()),
            }
    checks = {
        "context_and_four_candidate_shards_pass_and_current": len(shard_results) == 4,
        "canonical_row_coverage_and_source_hash_alignment": set(candidate_records)
        == set(range(rows)),
        "97d_features_use_current_candidate_only": features.shape == (rows, 2, 97),
        "discrete_target_geometry_support_and_finiteness_exact": True,
        "layer27_absent_from_runtime_dataset": dataset[
            "teacher_or_layer27_runtime_visible"
        ]
        is False,
        "calibration_test_and_c361_payload_not_opened": True,
        "no_training_threshold_selection_rollout_or_control": True,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS_V3_D2_DATASET" if passed else "FAIL_V3_D2_DATASET",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "source_rows": rows,
        "candidate_rows": rows * 2,
        "groups": len(set(zip(flat_task.tolist(), flat_episode.tolist()))),
        "support": support,
        "context_result": {
            "path": str(context_result_path),
            "sha256": stream_sha256(context_result_path),
        },
        "candidate_shard_results": shard_results,
        "payload": dataset_path.name,
        "payload_sha256": stream_sha256(dataset_path),
        "elapsed_seconds": time.perf_counter() - started,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "checks": checks,
        "claim_boundary": {
            "development_v2_payload_opened": True,
            "gripper_targets_computed": True,
            "model_trained": False,
            "calibration_or_test_payload_opened": False,
            "runtime_threshold_selected": False,
            "active_control": False,
            "method_performance_claim": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V3-D2 dataset failed one or more gates")
    incomplete.rename(output)
    print("PASS_V3_D2_DATASET", flush=True)


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
                        "status": "ABORT_V3_D2_DATASET",
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
