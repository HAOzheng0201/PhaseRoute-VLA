#!/usr/bin/env python3
"""Run one immutable outer episode fold of V3-D2 nested OOF."""

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
    OOF_SCHEMA_VERSION,
    development_data_from_mapping,
    fit_outer_fold,
)
from a1.vla.dynamic_compute.v3.gripper_v2_protocol import (  # noqa: E402
    DEVELOPMENT_EPISODES,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d2-oof-outer-result.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--outer-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise PermissionError("V3-D2 nested OOF is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D2 nested OOF requires a clean worktree")
    validate_frozen_d2_inputs(REPO_ROOT)
    if args.outer_episode not in DEVELOPMENT_EPISODES:
        raise ValueError("V3-D2 outer episode differs")
    expected_result = (
        REPO_ROOT / "reports" / "v3_d2_development_dataset" / "result.json"
    ).resolve()
    dataset_result_path = args.dataset_result.resolve(strict=True)
    if dataset_result_path != expected_result:
        raise PermissionError("V3-D2 dataset result path differs")
    dataset_result = json.loads(dataset_result_path.read_text(encoding="utf-8"))
    if (
        dataset_result.get("status") != "PASS_V3_D2_DATASET"
        or dataset_result.get("role") != D2_ROLE
        or dataset_result.get("suite") != D2_SUITE
    ):
        raise PermissionError("V3-D2 dataset result has not passed")
    dataset_path = dataset_result_path.parent / str(dataset_result["payload"])
    if stream_sha256(dataset_path) != dataset_result["payload_sha256"]:
        raise PermissionError("V3-D2 dataset payload SHA-256 differs")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != D2_DATASET_SCHEMA_VERSION
        or payload.get("role") != D2_ROLE
        or payload.get("suite") != D2_SUITE
        or payload.get("teacher_or_layer27_runtime_visible") is not False
        or payload.get("other_candidate_runtime_visible") is not False
    ):
        raise PermissionError("V3-D2 dataset payload contract differs")
    data = development_data_from_mapping(payload)
    torch.manual_seed(20260820 + args.outer_episode)
    torch.use_deterministic_algorithms(True)

    expected_output = (
        REPO_ROOT
        / "reports"
        / "v3_d2_development_oof_folds"
        / f"episode{args.outer_episode}"
    ).resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D2 outer-fold output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D2 refuses to overwrite outer-fold evidence")
    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")

    started = time.perf_counter()
    fold = fit_outer_fold(data, args.outer_episode, max_iterations=500)
    if fold.get("schema_version") != OOF_SCHEMA_VERSION:
        raise RuntimeError("V3-D2 outer-fold schema differs")
    fold_path = incomplete / "outer_fold.pt"
    torch.save(fold, fold_path)
    result = {
        "status": "PASS_V3_D2_OOF_OUTER_FOLD",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": D2_ROLE,
        "suite": D2_SUITE,
        "outer_episode": args.outer_episode,
        "inner_episode_count": len(fold["inner_episodes"]),
        "validation_rows": int(fold["predictions"]["row_index"].numel()),
        "fit_count": int(fold["fit_count"]),
        "selected_lambda": fold["selected_lambda"],
        "one_standard_error": fold["one_standard_error"],
        "dataset_result": str(dataset_result_path),
        "dataset_result_sha256": stream_sha256(dataset_result_path),
        "dataset_payload_sha256": stream_sha256(dataset_path),
        "payload": fold_path.name,
        "payload_sha256": stream_sha256(fold_path),
        "elapsed_seconds": time.perf_counter() - started,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "checks": {
            "development_dataset_pass_and_hash_current": True,
            "outer_episode_group_held_out_exactly": True,
            "seventeen_inner_episode_folds_and_170_equal_cells": True,
            "all_five_head_lambdas_selected_by_one_standard_error": True,
            "cpu_fp64_deterministic_full_batch_lbfgs": True,
            "calibration_test_and_c361_payload_not_opened": True,
            "no_threshold_rollout_or_active_control": True,
        },
        "claim_boundary": {
            "development_v2_payload_opened": True,
            "model_trained": True,
            "outer_validation_used_for_selection": False,
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
    incomplete.rename(output)
    print(f"PASS_V3_D2_OOF_OUTER_FOLD episode={args.outer_episode}", flush=True)


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
                        "status": "ABORT_V3_D2_OOF_OUTER_FOLD",
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
