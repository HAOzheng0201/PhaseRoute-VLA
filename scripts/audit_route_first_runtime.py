#!/usr/bin/env python3
"""Replay the Stage-8 route-first runtime rule on sealed Stage-7 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_calibration import (  # noqa: E402
    load_calibrated_route_first_router,
)
from a1.vla.dynamic_compute.route_first_dataset import (  # noqa: E402
    load_route_first_teacher_aggregate,
)
from a1.vla.dynamic_compute.route_first_runtime import (  # noqa: E402
    ROUTE_FIRST_CALIBRATED_ROUTER_SHA256,
    ROUTE_FIRST_STAGE7_HOLDOUT_SHA256,
    route_first_target_layers,
)


RESULT_SCHEMA = "phase-route-vla.route-first-stage8-runtime-replay.v1"
PASS_STATUS = "PASS_ROUTE_FIRST_RUNTIME_REPLAY_AND_SINGLE_FM_CONTRACT"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_exclusive(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as output_file:
            output_file.write(payload)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=REPO_ROOT
        / "runs/route_first_teacher_holdout_states10_11/aggregate_states10_11.npz",
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=REPO_ROOT
        / "runs/route_first_calibration_stage6/router_calibrated.npz",
    )
    parser.add_argument(
        "--stage7-result",
        type=Path,
        default=REPO_ROOT
        / "results/route_first/route_first_stage7_holdout.json",
    )
    parser.add_argument(
        "--stage7-scores",
        type=Path,
        default=REPO_ROOT / "runs/route_first_holdout_stage7/holdout_scores.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_path = args.aggregate.expanduser().resolve(strict=True)
    router_path = args.router.expanduser().resolve(strict=True)
    stage7_path = args.stage7_result.expanduser().resolve(strict=True)
    scores_path = args.stage7_scores.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()

    router_sha = _sha256_file(router_path)
    stage7_sha = _sha256_file(stage7_path)
    if router_sha != ROUTE_FIRST_CALIBRATED_ROUTER_SHA256:
        raise ValueError("Stage-8 router SHA-256 differs")
    if stage7_sha != ROUTE_FIRST_STAGE7_HOLDOUT_SHA256:
        raise ValueError("Stage-8 holdout result SHA-256 differs")
    stage7 = json.loads(stage7_path.read_text(encoding="utf-8"))
    if stage7.get("status") != "PASS_ENGINEERING_HOLDOUT_RUNTIME_INTEGRATION_READY":
        raise ValueError("Stage-7 result does not authorize runtime integration")
    if bool(stage7["authorization"]["active_control"]):
        raise ValueError("Stage-7 active-control boundary changed")

    aggregate = load_route_first_teacher_aggregate(aggregate_path)
    router, metadata = load_calibrated_route_first_router(router_path)
    scores, selected = route_first_target_layers(
        router,
        aggregate.features,
        enabled11=bool(metadata["enabled11"]),
        enabled13=bool(metadata["enabled13"]),
        threshold13=float(metadata["threshold13"]),
    )
    with np.load(scores_path, allow_pickle=False) as sealed:
        sealed_scores = sealed["scores"].astype(np.float64)
        sealed_selected = sealed["selected_layer"].astype(np.int16)
        if str(sealed["router_file_sha256"].item()) != router_sha:
            raise ValueError("sealed Stage-7 scores bind a different router")
    score_max_abs_error = float(np.max(np.abs(scores - sealed_scores)))
    selected_exact = int(np.count_nonzero(selected == sealed_selected))
    if selected_exact != aggregate.rows or score_max_abs_error > 1e-7:
        raise RuntimeError("route-first runtime replay differs from Stage 7")
    if bool(np.any(selected == 11)):
        raise RuntimeError("runtime replay selected disabled L11")

    route_counts = Counter(int(value) for value in selected)
    teacher_counts = Counter(int(value) for value in aggregate.teacher_layer)
    calls = aggregate.rows
    # The candidate-first teacher executes the RP-PEP L3 reference/candidate,
    # L11 reference/candidate, L13 candidate when L11 vetoes, and L27
    # reference/candidate when both early candidates veto.  This is an
    # invocation-count audit, not a wall-clock speed claim.
    candidate_first_fm_calls = (
        2 * calls
        + 2 * calls
        + (calls - teacher_counts.get(11, 0))
        + 2 * teacher_counts.get(27, 0)
    )
    route_first_fm_calls = calls
    baseline_full_decoder_blocks = calls * 28
    route_first_decoder_blocks = sum(
        count * (layer + 1) for layer, count in route_counts.items()
    )
    candidate_first_decoder_blocks = sum(
        count * (layer + 1) for layer, count in teacher_counts.items()
    )

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": PASS_STATUS,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "offline_replay_only_no_active_control",
        "inputs": {
            "aggregate": _display(aggregate_path),
            "aggregate_file_sha256": aggregate.file_sha256,
            "aggregate_payload_sha256": aggregate.payload_sha256,
            "rows": calls,
            "episodes": len(aggregate.episode_grid),
            "router": _display(router_path),
            "router_file_sha256": router_sha,
            "stage7_result": _display(stage7_path),
            "stage7_result_sha256": stage7_sha,
            "stage7_scores": _display(scores_path),
            "stage7_scores_sha256": _sha256_file(scores_path),
        },
        "runtime_contract": {
            "route_before_flow_matching": True,
            "action_free_feature_dimension": int(aggregate.features.shape[1]),
            "enabled_layers": [13, 27],
            "disabled_layers": [11],
            "fallback_layer": 27,
            "flow_matching_calls_per_valid_policy_call": 1,
            "threshold13": float(metadata["threshold13"]),
            "threshold_changed": False,
        },
        "replay": {
            "selected_layer_counts": {
                str(layer): int(route_counts.get(layer, 0))
                for layer in (11, 13, 27)
            },
            "stage7_selected_layer_exact_matches": selected_exact,
            "stage7_selected_layer_total": calls,
            "stage7_score_max_abs_error": score_max_abs_error,
            "l11_selected_rows": int(route_counts.get(11, 0)),
        },
        "static_compute_audit": {
            "route_first_flow_matching_invocations": route_first_fm_calls,
            "candidate_first_teacher_flow_matching_invocations": (
                candidate_first_fm_calls
            ),
            "flow_matching_invocation_reduction_fraction": (
                1.0 - route_first_fm_calls / candidate_first_fm_calls
            ),
            "route_first_decoder_blocks": route_first_decoder_blocks,
            "full_l27_decoder_blocks": baseline_full_decoder_blocks,
            "decoder_block_reduction_vs_full_l27_fraction": (
                1.0 - route_first_decoder_blocks / baseline_full_decoder_blocks
            ),
            "candidate_first_teacher_decoder_blocks": (
                candidate_first_decoder_blocks
            ),
            "note": (
                "Static invocation/block accounting only; no wall-clock speedup "
                "or closed-loop quality claim."
            ),
        },
        "authorization": {
            "runtime_integration_verified": True,
            "active_control_run": False,
            "generated_state_active_test": False,
            "historical_D9_states40_to49": False,
        },
        "claim_boundary": {
            "offline_replay_passed": True,
            "single_fm_controller_contract_requires_unit_test": True,
            "closed_loop_improvement_demonstrated": False,
            "wall_clock_speedup_demonstrated": False,
            "formal_safety_guarantee": False,
        },
    }
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_exclusive(output, payload)
    digest = _sha256_file(output)
    _write_exclusive(
        output.with_suffix(".sha256"),
        f"{digest}  {output.name}\n".encode("ascii"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
