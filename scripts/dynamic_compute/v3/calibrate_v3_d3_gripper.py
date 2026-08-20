#!/usr/bin/env python3
"""Apply the frozen D2 Gripper-v2 heads and calibrate one global D3 threshold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
    stream_sha256,
)
from a1.vla.dynamic_compute.v3.gripper_v2_calibration import (  # noqa: E402
    D3_CLUSTER_COUNT,
    D3_CONTRACT_SHA256,
    D3_DATASET_SCHEMA_VERSION,
    D3_D2_PAYLOAD_SHA256,
    D3_EPISODES,
    D3_ROLE,
    D3_SELECTION_SHA256,
    D3_SUITE,
    load_frozen_d2_final_state,
    score_calibration_features,
    select_global_threshold,
    validate_d3_prerequisites,
)


RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d3-calibration-result.v1"
PREDICTION_SCHEMA_VERSION = "phase-route-vla.v3.d3-calibration-predictions.v1"
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 1 or not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("V3-D3 quantile values must be non-empty and finite")
    probabilities = torch.tensor(QUANTILES, dtype=torch.float64)
    estimates = torch.quantile(values.detach().cpu().double(), probabilities)
    return {
        format(probability, ".2f"): float(estimate)
        for probability, estimate in zip(QUANTILES, estimates)
    }


def _validate_dataset(payload: Any, result: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("V3-D3 calibration dataset must be a mapping")
    if (
        payload.get("schema_version") != D3_DATASET_SCHEMA_VERSION
        or payload.get("role") != D3_ROLE
        or payload.get("suite") != D3_SUITE
        or payload.get("selection_sha256") != D3_SELECTION_SHA256
        or payload.get("feature_dimension") != 97
        or payload.get("target_axis_order") != ["step", "transition"]
        or payload.get("teacher_or_layer27_runtime_visible") is not False
        or payload.get("other_candidate_runtime_visible") is not False
        or payload.get("task_episode_identity_is_runtime_input") is not False
        or payload.get("full_depth_is_consistency_teacher_only") is not True
        or payload.get("model_refit_allowed") is not False
        or payload.get("runtime_threshold_selected") is not False
        or payload.get("independent_test_payload_opened") is not False
    ):
        raise PermissionError("V3-D3 calibration dataset contract differs")
    names = (
        "features",
        "candidate_layer",
        "source_row",
        "task_id",
        "episode_index",
        "occurrence",
        "count",
        "expected_fraction",
        "step_mismatch_bits",
        "transition_mismatch_bits",
    )
    if any(not isinstance(payload.get(name), torch.Tensor) for name in names):
        raise ValueError("V3-D3 calibration dataset tensor columns differ")
    data = {name: payload[name].detach().cpu().contiguous() for name in names}
    rows = int(data["features"].shape[0])
    source_rows = rows // 2
    if (
        rows < D3_CLUSTER_COUNT * 2
        or rows % 2
        or data["features"].shape != (rows, 97)
        or not data["features"].is_floating_point()
        or not bool(torch.isfinite(data["features"]).all())
    ):
        raise ValueError("V3-D3 calibration feature geometry differs")
    for name in ("candidate_layer", "source_row", "task_id", "episode_index"):
        if data[name].dtype != torch.long or data[name].shape != (rows,):
            raise ValueError(f"V3-D3 calibration {name} geometry differs")
    if (
        data["occurrence"].dtype != torch.bool
        or data["occurrence"].shape != (rows, 2)
        or data["count"].dtype != torch.long
        or data["count"].shape != (rows, 2)
        or not data["expected_fraction"].is_floating_point()
        or data["expected_fraction"].shape != (rows, 2)
        or not bool(torch.isfinite(data["expected_fraction"]).all())
        or data["step_mismatch_bits"].dtype != torch.bool
        or data["step_mismatch_bits"].shape != (source_rows, 2, 8)
        or data["transition_mismatch_bits"].dtype != torch.bool
        or data["transition_mismatch_bits"].shape != (source_rows, 2, 7)
    ):
        raise ValueError("V3-D3 calibration target geometry differs")
    paired_layers = torch.tensor([11, 13], dtype=torch.long).repeat(source_rows)
    paired_source = torch.arange(source_rows, dtype=torch.long).repeat_interleave(2)
    if (
        not torch.equal(data["candidate_layer"], paired_layers)
        or not torch.equal(data["source_row"], paired_source)
        or not torch.equal(data["task_id"][0::2], data["task_id"][1::2])
        or not torch.equal(
            data["episode_index"][0::2], data["episode_index"][1::2]
        )
    ):
        raise PermissionError("V3-D3 calibration candidate pairing differs")
    support = torch.tensor([8, 7], dtype=torch.long)
    expected_fraction = data["count"].double() / support.double()
    if (
        not torch.equal(data["occurrence"], data["count"] > 0)
        or not torch.equal(
            data["occurrence"][:, 0],
            data["step_mismatch_bits"].reshape(rows, 8).any(dim=1),
        )
        or not torch.equal(
            data["occurrence"][:, 1],
            data["transition_mismatch_bits"].reshape(rows, 7).any(dim=1),
        )
        or not torch.equal(
            data["count"][:, 0],
            data["step_mismatch_bits"].reshape(rows, 8).sum(dim=1),
        )
        or not torch.equal(
            data["count"][:, 1],
            data["transition_mismatch_bits"].reshape(rows, 7).sum(dim=1),
        )
        or not torch.allclose(
            data["expected_fraction"].double(),
            expected_fraction,
            rtol=0.0,
            atol=1.0e-7,
        )
        or bool((data["occurrence"][:, 1] & ~data["occurrence"][:, 0]).any())
    ):
        raise ValueError("V3-D3 calibration target identities differ")
    if (
        not bool(((data["task_id"] >= 0) & (data["task_id"] <= 9)).all())
        or not all(value in D3_EPISODES for value in data["episode_index"].tolist())
        or len(set(zip(data["task_id"].tolist(), data["episode_index"].tolist())))
        != D3_CLUSTER_COUNT
    ):
        raise PermissionError("V3-D3 calibration cluster boundary differs")
    if (
        result.get("candidate_rows") != rows
        or result.get("source_rows") != source_rows
        or result.get("groups") != D3_CLUSTER_COUNT
    ):
        raise ValueError("V3-D3 calibration dataset result counts differ")
    return data


def _stratum_metrics(
    *,
    mask: torch.Tensor,
    safe: torch.Tensor | None,
    step_mismatch: torch.Tensor,
    transition_mismatch: torch.Tensor,
) -> dict[str, Any]:
    rows = int(mask.sum())
    step_events = int((step_mismatch & mask).sum())
    transition_events = int((transition_mismatch & mask).sum())
    result: dict[str, Any] = {
        "candidate_rows": rows,
        "step_mismatch_rows": step_events,
        "step_mismatch_rate": step_events / rows if rows else None,
        "transition_mismatch_rows": transition_events,
        "transition_mismatch_rate": transition_events / rows if rows else None,
        "selected_threshold_available": safe is not None,
    }
    if safe is None:
        result.update(
            {
                "predicted_safe_rows": None,
                "safe_step_mismatch_rows": None,
                "safe_step_mismatch_rate": None,
                "safe_transition_mismatch_rows": None,
                "safe_transition_mismatch_rate": None,
            }
        )
        return result
    safe_mask = safe & mask
    safe_rows = int(safe_mask.sum())
    safe_step = int((step_mismatch & safe_mask).sum())
    safe_transition = int((transition_mismatch & safe_mask).sum())
    result.update(
        {
            "predicted_safe_rows": safe_rows,
            "safe_step_mismatch_rows": safe_step,
            "safe_step_mismatch_rate": safe_step / safe_rows if safe_rows else None,
            "safe_transition_mismatch_rows": safe_transition,
            "safe_transition_mismatch_rate": (
                safe_transition / safe_rows if safe_rows else None
            ),
        }
    )
    return result


def _run(args: argparse.Namespace) -> None:
    prerequisite_audit = validate_d3_prerequisites(REPO_ROOT)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise PermissionError("V3-D3 threshold calibration is CPU-only")
    expected_dataset_result = (
        REPO_ROOT / "reports/v3_d3_calibration_dataset/result.json"
    ).resolve()
    dataset_result_path = args.dataset_result.resolve(strict=True)
    if dataset_result_path != expected_dataset_result:
        raise PermissionError("V3-D3 calibration dataset result path differs")
    dataset_result = json.loads(dataset_result_path.read_text(encoding="utf-8"))
    current_commit = git_output("rev-parse", "HEAD")
    if (
        dataset_result.get("status") != "PASS_V3_D3_DATASET"
        or dataset_result.get("role") != D3_ROLE
        or dataset_result.get("suite") != D3_SUITE
        or dataset_result.get("source_worktree_dirty") is not False
        or dataset_result.get("source_git_commit") != current_commit
        or dataset_result.get("claim_boundary", {}).get(
            "independent_test_payload_opened"
        )
        is not False
    ):
        raise PermissionError("V3-D3 calibration dataset result has not passed")
    dataset_path = dataset_result_path.parent / str(dataset_result["payload"])
    if stream_sha256(dataset_path) != dataset_result.get("payload_sha256"):
        raise PermissionError("V3-D3 calibration dataset payload SHA-256 differs")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    data = _validate_dataset(dataset, dataset_result)

    expected_output = (REPO_ROOT / "reports/v3_d3_calibration_result").resolve()
    output = args.output_dir.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output != expected_output:
        raise PermissionError("V3-D3 calibration output path differs")
    if output.exists() or incomplete.exists():
        raise FileExistsError("V3-D3 refuses to overwrite calibration evidence")
    incomplete.mkdir(parents=True, exist_ok=False)
    command = "cd " + shlex.quote(str(REPO_ROOT)) + "\n\n" + shlex.join(
        [sys.executable, *sys.argv]
    )
    (incomplete / "command.txt").write_text(command + "\n", encoding="utf-8")

    started = time.perf_counter()
    final_state = load_frozen_d2_final_state(REPO_ROOT)
    predictions = score_calibration_features(
        final_state, data["features"], data["candidate_layer"]
    )
    step_mismatch = data["occurrence"][:, 0]
    transition_mismatch = data["occurrence"][:, 1]
    selection = select_global_threshold(
        score=predictions["score"],
        step_mismatch=step_mismatch,
        transition_mismatch=transition_mismatch,
        task_id=data["task_id"],
        episode_index=data["episode_index"],
        candidate_layer=data["candidate_layer"],
    )
    selected = selection["selected"]
    selected_threshold = None if selected is None else float(selected["threshold"])
    safe = (
        None
        if selected_threshold is None
        else predictions["score"] <= selected_threshold
    )

    curve_path = incomplete / "threshold_curve.jsonl"
    with curve_path.open("w", encoding="utf-8") as output_file:
        for index, record in enumerate(selection["curve"]):
            output_file.write(
                json.dumps(
                    {"candidate_index": index, **record},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    prediction_payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "role": D3_ROLE,
        "suite": D3_SUITE,
        "contract_sha256": D3_CONTRACT_SHA256,
        "selection_sha256": D3_SELECTION_SHA256,
        "dataset_payload_sha256": stream_sha256(dataset_path),
        "d2_final_model_payload_sha256": D3_D2_PAYLOAD_SHA256,
        "score_name": selection["score_name"],
        "score": predictions["score"],
        "occurrence_probability": predictions["occurrence_probability"],
        "ordinal_step_probability": predictions["ordinal_step_probability"],
        "ordinal_transition_probability": predictions[
            "ordinal_transition_probability"
        ],
        "ordinal_expected_fraction": predictions["ordinal_expected_fraction"],
        "task_id": data["task_id"],
        "episode_index": data["episode_index"],
        "candidate_layer": data["candidate_layer"],
        "step_mismatch": step_mismatch,
        "transition_mismatch": transition_mismatch,
        "selected_threshold": selected_threshold,
        "model_refit": False,
        "independent_test_payload_opened": False,
        "active_control": False,
    }
    if safe is not None:
        prediction_payload["safe_call"] = safe
    prediction_path = incomplete / "calibration_predictions.pt"
    torch.save(prediction_payload, prediction_path)

    all_rows = torch.ones_like(step_mismatch)
    per_layer = {
        str(layer): _stratum_metrics(
            mask=data["candidate_layer"] == layer,
            safe=safe,
            step_mismatch=step_mismatch,
            transition_mismatch=transition_mismatch,
        )
        for layer in (11, 13)
    }
    per_task = {
        str(task): _stratum_metrics(
            mask=data["task_id"] == task,
            safe=safe,
            step_mismatch=step_mismatch,
            transition_mismatch=transition_mismatch,
        )
        for task in range(10)
    }
    overall = _stratum_metrics(
        mask=all_rows,
        safe=safe,
        step_mismatch=step_mismatch,
        transition_mismatch=transition_mismatch,
    )
    if selected is None:
        primary = {
            "selected_threshold": None,
            "safe_clusters": 0,
            "false_safe_clusters": 0,
            "safe_cluster_coverage": 0.0,
            "false_safe_cluster_rate": None,
            "false_safe_cluster_ucb95": 1.0,
        }
    else:
        primary = {
            "selected_threshold": selected_threshold,
            "safe_clusters": int(selected["safe_clusters"]),
            "false_safe_clusters": int(selected["false_safe_clusters"]),
            "safe_cluster_coverage": float(selected["safe_cluster_coverage"]),
            "false_safe_cluster_rate": float(selected["false_safe_cluster_rate"]),
            "false_safe_cluster_ucb95": float(
                selected["false_safe_cluster_ucb95"]
            ),
        }
    checks = {
        "frozen_d2_final_state_restored_without_refit": True,
        "predeclared_step_occurrence_score_only": selection["score_name"]
        == "step_any_mismatch_probability",
        "one_global_threshold_shared_across_layers_tasks_and_time": True,
        "all_100_calibration_clusters_present": selection["checks"][
            "all_100_calibration_clusters_present"
        ],
        "transition_mismatch_implies_step_mismatch": selection["checks"][
            "transition_mismatch_implies_step_mismatch"
        ],
        "complete_sorted_unique_threshold_curve_saved": len(selection["curve"])
        == selection["candidate_threshold_count"],
        "always_defer_not_accepted_as_pass": (
            selection["status"] != "PASS_V3_D3_CALIBRATION_GATE"
            or selected is not None
        ),
        "independent_test_shadow_and_active_control_not_run": True,
    }
    source_status = git_output("status", "--porcelain=v1")
    result = {
        "status": selection["status"],
        "schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "role": D3_ROLE,
        "suite": D3_SUITE,
        "contract_sha256": D3_CONTRACT_SHA256,
        "prerequisite_audit": prerequisite_audit,
        "dataset_result": {
            "path": str(dataset_result_path),
            "sha256": stream_sha256(dataset_result_path),
            "payload_sha256": stream_sha256(dataset_path),
        },
        "frozen_d2_final_model_payload_sha256": D3_D2_PAYLOAD_SHA256,
        "candidate_rows": int(data["features"].shape[0]),
        "clusters": D3_CLUSTER_COUNT,
        "candidate_threshold_count": selection["candidate_threshold_count"],
        "feasible_threshold_count": selection["feasible_threshold_count"],
        "primary": primary,
        "secondary": {
            "predicted_safe_candidate_rows": overall["predicted_safe_rows"],
            "call_level_false_safe_rate": overall["safe_step_mismatch_rate"],
            "overall_support_and_error": overall,
            "per_layer_support_and_error": per_layer,
            "per_task_support_and_error": per_task,
            "score_quantiles": _quantiles(predictions["score"]),
            "ordinal_expected_fraction_quantiles": {
                "step": _quantiles(predictions["ordinal_expected_fraction"][:, 0]),
                "transition": _quantiles(
                    predictions["ordinal_expected_fraction"][:, 1]
                ),
            },
        },
        "threshold_curve": {
            "path": curve_path.name,
            "sha256": stream_sha256(curve_path),
            "records": selection["candidate_threshold_count"],
        },
        "predictions": {
            "path": prediction_path.name,
            "sha256": stream_sha256(prediction_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source_git_commit": current_commit,
        "source_git_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "source_worktree_dirty": bool(source_status),
        "checks": checks,
        "next_stage": {
            "d4_shadow_decision_only_authorized": selection["status"]
            == "PASS_V3_D3_CALIBRATION_GATE",
            "active_control_authorized": False,
            "independent_test_authorized": False,
        },
        "claim_boundary": {
            "calibration_v2_payload_opened": True,
            "frozen_model_scored": True,
            "model_refit": False,
            "runtime_threshold_selected": selected is not None,
            "independent_test_payload_opened": False,
            "shadow_decision_run": False,
            "active_control_run": False,
            "success_rate_or_superiority_claim": False,
        },
    }
    (incomplete / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()) or result["source_worktree_dirty"]:
        raise RuntimeError("V3-D3 calibration audit failed")
    incomplete.rename(output)
    print(selection["status"], flush=True)


def main() -> None:
    args = parse_args()
    try:
        _run(args)
    except BaseException as error:
        incomplete = args.output_dir.resolve().with_name(
            args.output_dir.name + ".incomplete"
        )
        if incomplete.is_dir() and not (incomplete / "abort.json").exists():
            (incomplete / "abort.json").write_text(
                json.dumps(
                    {
                        "status": "ABORT_V3_D3_CALIBRATION",
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
