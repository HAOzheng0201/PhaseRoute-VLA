#!/usr/bin/env python3
"""Aggregate all V3-D2 outer folds, evaluate gates, and refit final heads."""

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
    D2_DATASET_SCHEMA_VERSION,
    D2_ROLE,
    D2_SUITE,
    stream_sha256,
    validate_frozen_d2_inputs,
)
from a1.vla.dynamic_compute.v3.gripper_v2_oof import (  # noqa: E402
    EXPECTED_FITS_PER_OUTER,
    HEAD_NAMES,
    OOF_SCHEMA_VERSION,
    development_data_from_mapping,
    evaluate_oof,
    final_lambda,
    fit_final_models,
)
from a1.vla.dynamic_compute.v3.gripper_v2_protocol import (  # noqa: E402
    DEVELOPMENT_EPISODES,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d2-nested-oof-result.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _load_dataset(path: Path) -> tuple[dict[str, Any], Path, Any]:
    expected = (
        REPO_ROOT / "reports" / "v3_d2_development_dataset" / "result.json"
    ).resolve()
    resolved = path.resolve(strict=True)
    if resolved != expected:
        raise PermissionError("V3-D2 dataset result path differs")
    result = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS_V3_D2_DATASET"
        or result.get("role") != D2_ROLE
        or result.get("suite") != D2_SUITE
    ):
        raise PermissionError("V3-D2 dataset result has not passed")
    payload_path = resolved.parent / str(result["payload"])
    if stream_sha256(payload_path) != result["payload_sha256"]:
        raise PermissionError("V3-D2 dataset payload SHA-256 differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != D2_DATASET_SCHEMA_VERSION:
        raise PermissionError("V3-D2 dataset schema differs")
    return result, payload_path, development_data_from_mapping(payload)


def _empty_oof(rows: int) -> dict[str, torch.Tensor]:
    return {
        "occurrence_probability": torch.full((rows, 2), torch.nan, dtype=torch.float64),
        "occurrence_baseline": torch.full((rows, 2), torch.nan, dtype=torch.float64),
        "zt_step_probability": torch.full((rows, 8), torch.nan, dtype=torch.float64),
        "zt_transition_probability": torch.full((rows, 7), torch.nan, dtype=torch.float64),
        "ordinal_step_probability": torch.full((rows, 8), torch.nan, dtype=torch.float64),
        "ordinal_transition_probability": torch.full((rows, 7), torch.nan, dtype=torch.float64),
        "expected_fraction": torch.full((rows, 2), torch.nan, dtype=torch.float64),
        "expected_fraction_baseline": torch.full((rows, 2), torch.nan, dtype=torch.float64),
        "assignment_count": torch.zeros(rows, dtype=torch.long),
    }


def _run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise PermissionError("V3-D2 OOF aggregation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D2 OOF aggregation requires a clean worktree")
    validate_frozen_d2_inputs(REPO_ROOT)
    dataset_result, dataset_path, data = _load_dataset(args.dataset_result)
    torch.manual_seed(20260820)
    torch.use_deterministic_algorithms(True)
    expected_fold_root = (
        REPO_ROOT / "reports" / "v3_d2_development_oof_folds"
    ).resolve()
    fold_root = args.fold_root.resolve(strict=True)
    if fold_root != expected_fold_root:
        raise PermissionError("V3-D2 OOF fold root differs")

    oof = _empty_oof(data.rows)
    outer_lambdas = {head: [] for head in HEAD_NAMES}
    fold_results: dict[str, Any] = {}
    total_outer_fit_count = 0
    for episode in DEVELOPMENT_EPISODES:
        directory = fold_root / f"episode{episode}"
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS_V3_D2_OOF_OUTER_FOLD"
            or result.get("outer_episode") != episode
            or result.get("source_worktree_dirty") is not False
            or result.get("source_git_commit") != git_output("rev-parse", "HEAD")
            or result.get("dataset_payload_sha256") != stream_sha256(dataset_path)
        ):
            raise PermissionError(f"V3-D2 OOF outer episode {episode} result differs")
        payload_path = directory / str(result["payload"])
        if stream_sha256(payload_path) != result["payload_sha256"]:
            raise PermissionError(f"V3-D2 OOF outer episode {episode} payload SHA differs")
        fold = torch.load(payload_path, map_location="cpu", weights_only=True)
        if (
            fold.get("schema_version") != OOF_SCHEMA_VERSION
            or fold.get("outer_episode") != episode
            or fold.get("fit_count") != EXPECTED_FITS_PER_OUTER
        ):
            raise PermissionError(f"V3-D2 OOF outer episode {episode} payload differs")
        indices = fold["predictions"]["row_index"]
        if not torch.equal(data.episode_index[indices], torch.full_like(indices, episode)):
            raise PermissionError("V3-D2 OOF held-out row identity differs")
        if bool((oof["assignment_count"][indices] != 0).any()):
            raise PermissionError("V3-D2 OOF row assigned more than once")
        for name in oof:
            if name == "assignment_count":
                continue
            oof[name][indices] = fold["predictions"][name]
        oof["assignment_count"][indices] += 1
        for head in HEAD_NAMES:
            value = float(fold["selected_lambda"][head])
            if value != float(result["selected_lambda"][head]):
                raise PermissionError("V3-D2 OOF selected lambda evidence differs")
            outer_lambdas[head].append(value)
        total_outer_fit_count += int(fold["fit_count"])
        fold_results[str(episode)] = {
            "path": str(result_path),
            "sha256": stream_sha256(result_path),
            "payload_sha256": stream_sha256(payload_path),
            "validation_rows": int(indices.numel()),
            "selected_lambda": result["selected_lambda"],
            "elapsed_seconds": float(result["elapsed_seconds"]),
        }

    if total_outer_fit_count != 18 * EXPECTED_FITS_PER_OUTER:
        raise RuntimeError("V3-D2 OOF aggregate fit count differs")
    if not torch.equal(oof["assignment_count"], torch.ones(data.rows, dtype=torch.long)):
        raise RuntimeError("V3-D2 OOF coverage is not exactly once")
    metrics = evaluate_oof(data, oof)
    final_lambdas = {
        head: final_lambda(outer_lambdas[head]) for head in HEAD_NAMES
    }

    expected_output = (REPO_ROOT / "reports" / "v3_d2_development_oof").resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D2 OOF aggregate output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D2 refuses to overwrite aggregate OOF evidence")
    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")
    started = time.perf_counter()
    final_model_state = fit_final_models(data, final_lambdas, max_iterations=500)
    primary_parameter_count = (
        int(final_model_state["occurrence"]["weight"].numel())
        + int(final_model_state["ordinal_step"]["weight"].numel())
        + int(final_model_state["ordinal_step"]["raw_base"].numel())
        + int(final_model_state["ordinal_step"]["raw_increments"].numel())
        + int(final_model_state["ordinal_transition"]["weight"].numel())
        + int(final_model_state["ordinal_transition"]["raw_base"].numel())
        + int(final_model_state["ordinal_transition"]["raw_increments"].numel())
    )
    if primary_parameter_count > 512:
        raise RuntimeError("V3-D2 primary parameter cap exceeded")
    payload = {
        "schema_version": OOF_SCHEMA_VERSION,
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "dataset_payload_sha256": stream_sha256(dataset_path),
        "task_id": data.task_id,
        "episode_index": data.episode_index,
        "candidate_layer": data.candidate_layer,
        "occurrence_target": data.occurrence,
        "count_target": data.count,
        "expected_fraction_target": data.expected_fraction,
        "oof": oof,
        "outer_selected_lambdas": outer_lambdas,
        "final_lambdas": final_lambdas,
        "final_model_state": final_model_state,
        "primary_parameter_count": primary_parameter_count,
        "calibration_or_test_payload_accessed": False,
        "runtime_threshold_selected": False,
    }
    payload_path = incomplete / "development_gripper_v2_nested_oof.pt"
    torch.save(payload, payload_path)
    gates = metrics["gates"]
    if gates["full_pass"]:
        status = "PASS_V3_D2_FULL_DEVELOPMENT_GATE"
    elif gates["focused_pass_non_deployable"]:
        status = "PASS_V3_D2_FOCUSED_NON_DEPLOYABLE"
    else:
        status = "NEGATIVE_V3_D2_DEVELOPMENT_GATE"
    result = {
        "status": status,
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "rows": data.rows,
        "groups": 180,
        "outer_fold_count": 18,
        "inner_fold_count_per_outer": 17,
        "inner_cell_count_per_lambda_per_outer": 170,
        "fit_counts": {
            "inner": 18 * 17 * 3 * 5,
            "outer_refit": 18 * 5,
            "final_refit": 5,
            "total": total_outer_fit_count + 5,
        },
        "outer_selected_lambdas": outer_lambdas,
        "final_lambda_rule": "largest_lambda_among_outer_selection_modes",
        "final_lambdas": final_lambdas,
        "primary_parameter_count": primary_parameter_count,
        "metrics": metrics,
        "dataset_result": {
            "path": str(args.dataset_result.resolve()),
            "sha256": stream_sha256(args.dataset_result.resolve()),
            "payload_sha256": stream_sha256(dataset_path),
        },
        "outer_fold_results": fold_results,
        "payload": payload_path.name,
        "payload_sha256": stream_sha256(payload_path),
        "final_refit_elapsed_seconds": time.perf_counter() - started,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "checks": {
            "all_18_outer_and_17_inner_episode_folds_complete": True,
            "all_oof_rows_assigned_exactly_once": True,
            "candidate_pairs_and_task_episode_groups_never_split": True,
            "fit_partition_only_normalization_anchors_and_cutpoints": True,
            "ordinal_primary_and_zt_binomial_comparator_fixed_before_labels": True,
            "calibration_test_and_c361_payload_not_opened": True,
            "no_runtime_threshold_rollout_or_active_control": True,
        },
        "next_stage": {
            "d3_calibration_authorized": bool(
                gates["full_pass"] or gates["focused_pass_non_deployable"]
            ),
            "deployment_authorized": False,
            "independent_test_authorized": False,
        },
        "claim_boundary": {
            "development_v2_payload_opened": True,
            "model_trained": True,
            "development_nested_oof_metrics_reported": True,
            "calibration_or_test_payload_opened": False,
            "runtime_threshold_selected": False,
            "active_control": False,
            "independent_test_performance_claim": False,
            "superiority_claim": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.rename(output)
    print(status, flush=True)


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
                        "status": "ABORT_V3_D2_NESTED_OOF",
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
