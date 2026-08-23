#!/usr/bin/env python3
"""Run one immutable outer episode fold of V3-D5 nested OOF."""

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
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from a1.vla.dynamic_compute.v3.development_collection import stream_sha256  # noqa: E402
from a1.vla.dynamic_compute.v3.joint_reliability import (  # noqa: E402
    D5_CONTRACT_SHA256,
    D5_EPISODES,
    development_data_from_mapping,
)
from a1.vla.dynamic_compute.v3.joint_reliability_oof import (  # noqa: E402
    D5_FITS_PER_OUTER,
    D5_OOF_SCHEMA_VERSION,
    fit_outer_fold,
)


DATASET_RESULT = Path("reports/v3_d5_development_dataset/result.json")
DATASET_RESULT_SHA256 = (
    "7b4facd767594974359bef11edec83bbe3df66c3ee4c5c3981814992f792186d"
)
DATASET_PAYLOAD_SHA256 = (
    "cf40a9802e37d2335668db7f7e24194a3316d552183151cc780fecb5424137df"
)
PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d5-joint-development-dataset.v1"
RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d5-oof-outer-result.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--outer-episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PermissionError("V3-D5 outer metadata must be an object")
    return dict(value)


def run(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("V3-D5 nested OOF is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("V3-D5 nested OOF requires a clean worktree")
    if args.outer_episode not in D5_EPISODES:
        raise PermissionError("V3-D5 outer episode differs")
    expected_result = (REPO_ROOT / DATASET_RESULT).resolve(strict=True)
    result_path = args.dataset_result.resolve(strict=True)
    if result_path != expected_result or stream_sha256(result_path) != DATASET_RESULT_SHA256:
        raise PermissionError("V3-D5 dataset result path or SHA differs")
    result = json_object(result_path)
    if (
        result.get("status") != "PASS_V3_D5_DEVELOPMENT_DATASET"
        or result.get("source_worktree_dirty") is not False
        or result.get("payload_sha256") != DATASET_PAYLOAD_SHA256
        or result.get("input_sha256", {}).get("contract") != D5_CONTRACT_SHA256
    ):
        raise PermissionError("V3-D5 dataset result semantics differ")
    payload_path = result_path.parent / str(result["payload"])
    if stream_sha256(payload_path) != DATASET_PAYLOAD_SHA256:
        raise PermissionError("V3-D5 dataset payload SHA differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION
        or payload.get("contract_sha256") != D5_CONTRACT_SHA256
        or payload.get("layer27_runtime_visible") is not False
        or payload.get("calibration_or_test_payload_opened") is not False
    ):
        raise PermissionError("V3-D5 dataset payload semantics differ")
    data = development_data_from_mapping(payload)
    torch.manual_seed(20260820 + args.outer_episode)
    torch.use_deterministic_algorithms(True)

    expected_output = (
        REPO_ROOT
        / "reports/v3_d5_development_oof_folds"
        / f"episode{args.outer_episode}"
    ).resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D5 outer output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D5 refuses to overwrite outer-fold evidence")
    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")

    started = time.perf_counter()
    fold = fit_outer_fold(data, args.outer_episode, max_iterations=500)
    if (
        fold.get("schema_version") != D5_OOF_SCHEMA_VERSION
        or fold.get("fit_count") != D5_FITS_PER_OUTER
    ):
        raise RuntimeError("V3-D5 outer fold output semantics differ")
    fold_path = incomplete / "outer_fold.pt"
    torch.save(fold, fold_path)
    outer_result = {
        "status": "PASS_V3_D5_OOF_OUTER_FOLD",
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "role": "development_v2",
        "suite": "libero_10",
        "outer_episode": args.outer_episode,
        "inner_episode_count": 17,
        "validation_rows": int(fold["validation_indices"].numel()),
        "fit_count": int(fold["fit_count"]),
        "selected_lambda": float(fold["selected_lambda"]),
        "inner_threshold_feasible": bool(fold["inner_threshold_feasible"]),
        "inner_selected_threshold": fold["inner_selected_threshold"],
        "inner_threshold_summary": fold["inner_threshold_summary"],
        "inner_evaluated_thresholds": int(fold["inner_evaluated_thresholds"]),
        "dataset_result_sha256": DATASET_RESULT_SHA256,
        "dataset_payload_sha256": DATASET_PAYLOAD_SHA256,
        "payload": fold_path.name,
        "payload_sha256": stream_sha256(fold_path),
        "elapsed_seconds": time.perf_counter() - started,
        "checks": {
            "authenticated_D5_dataset_and_contract_current": True,
            "outer_episode_held_out_from_normalizer_anchor_model_and_threshold": True,
            "seventeen_inner_episode_folds_and_170_task_cells_exact": True,
            "lambda_selected_by_frozen_one_standard_error_rule": True,
            "threshold_selected_from_inner_OOF_only": True,
            "CPU_FP64_deterministic_full_batch_LBFGS": True,
            "calibration_and_independent_test_payload_not_opened": True,
            "no_rollout_active_control_or_deployment": True,
        },
        "claim_boundary": {
            "development_model_trained": True,
            "outer_truth_used_for_selection": False,
            "calibration_or_test_payload_opened": False,
            "closed_loop_result": False,
            "active_control": False,
            "superiority_claim": False,
        },
    }
    result_output = incomplete / "result.json"
    result_output.write_text(
        json.dumps(outer_result, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (incomplete / "result.sha256").write_text(
        f"{stream_sha256(result_output)}  result.json\n", encoding="utf-8"
    )
    incomplete.rename(output)
    print(
        "PASS_V3_D5_OOF_OUTER_FOLD "
        f"episode={args.outer_episode} lambda={fold['selected_lambda']} "
        f"threshold={fold['inner_selected_threshold']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except BaseException as error:
        incomplete = args.output_dir.resolve().with_name(
            args.output_dir.name + ".incomplete"
        )
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_V3_D5_OOF_OUTER_FOLD",
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
