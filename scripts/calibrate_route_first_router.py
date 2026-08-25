#!/usr/bin/env python3
"""Select on state 8 and one-shot confirm on state 9 without active control."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_calibration import (  # noqa: E402
    ROUTE_FIRST_CALIBRATION_STATUS,
    RouteFirstThresholdRule,
    confirm_route_first_threshold,
    load_calibrated_route_first_router,
    route_first_confirmed_layers,
    route_first_safe_label,
    save_calibrated_route_first_router,
    select_route_first_threshold,
)
from a1.vla.dynamic_compute.route_first_dataset import (  # noqa: E402
    load_route_first_teacher_aggregate,
)
from a1.vla.dynamic_compute.route_first_router import (  # noqa: E402
    ROUTE_FIRST_ROUTER_CALIBRATION_STATUS,
    load_uncalibrated_route_first_router,
    route_first_group_weights,
)


PROTOCOL_SCHEMA = "phase-route-vla.route-first-calibration-protocol.v1"
RESULT_SCHEMA = "phase-route-vla.route-first-calibration-result.v1"
SCORES_SCHEMA = "phase-route-vla.route-first-calibration-scores.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def _load_protocol(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as input_file:
        protocol = json.load(input_file)
    required = {
        "schema_version",
        "seed",
        "frozen_score_model",
        "data",
        "statistics",
        "threshold_selection",
        "one_shot_confirmation",
        "fail_closed_policy",
        "claim_boundary",
    }
    if set(protocol) != required or protocol["schema_version"] != PROTOCOL_SCHEMA:
        raise ValueError("route-first calibration protocol fields differ")
    data = protocol["data"]
    if (
        data["suite"] != "libero_10"
        or data["task_ids"] != list(range(10))
        or data["threshold_selection_episode_indices"] != [8]
        or data["one_shot_confirmation_episode_indices"] != [9]
        or data["engineering_holdout_episode_indices_not_opened"] != [10, 11]
        or data["historical_D9_episode_indices_forbidden"] != list(range(40, 50))
        or bool(data["control_influence"])
        or bool(data["identity_is_model_input"])
    ):
        raise ValueError("route-first calibration data boundary differs")
    statistics = protocol["statistics"]
    if statistics != {
        "episode_cells": "equal_total_weight",
        "confidence_method": "one_sided_weighted_wilson_effective_sample_size",
        "confidence_level": 0.9,
        "threshold_candidates": "unique_observed_score_selecting_score_greater_than_or_equal",
        "selection_objective": (
            "maximum_group_equal_coverage_then_lower_false_safe_then_higher_threshold"
        ),
    }:
        raise ValueError("route-first calibration statistic contract differs")
    fail_closed = protocol["fail_closed_policy"]
    if fail_closed != {
        "selection_without_feasible_threshold_disables_head": True,
        "confirmation_failure_disables_only_that_head": True,
        "state9_may_not_change_a_threshold": True,
        "safe13_confirmation_required_for_engineering_holdout": True,
        "all_disabled_route": 27,
    }:
        raise ValueError("route-first fail-closed policy differs")
    return protocol


def _save_scores(
    path: Path,
    *,
    aggregate,
    scores: np.ndarray,
    selected_layers: np.ndarray,
    selection11: dict[str, object],
    selection13: dict[str, object],
    confirmation11: dict[str, object],
    confirmation13: dict[str, object],
) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

    def threshold(selection: dict[str, object]) -> float:
        value = selection.get("threshold")
        return float(value) if isinstance(value, (float, int)) else 1.0

    try:
        with temporary.open("xb") as output_file:
            np.savez_compressed(
                output_file,
                schema_version=np.asarray(SCORES_SCHEMA),
                calibration_status=np.asarray(ROUTE_FIRST_CALIBRATION_STATUS),
                aggregate_payload_sha256=np.asarray(aggregate.payload_sha256),
                aggregate_file_sha256=np.asarray(aggregate.file_sha256),
                scores=np.asarray(scores, dtype=np.float32),
                teacher_layer=aggregate.teacher_layer.astype(np.int16),
                task_id=aggregate.task_id.astype(np.int16),
                episode_index=aggregate.episode_index.astype(np.int16),
                call_ordinal=aggregate.call_ordinal.astype(np.int32),
                state9_selected_layer=np.asarray(selected_layers, dtype=np.int16),
                selection_threshold11=np.asarray(
                    threshold(selection11), dtype=np.float64
                ),
                selection_threshold13=np.asarray(
                    threshold(selection13), dtype=np.float64
                ),
                confirmed_enabled11=np.asarray(
                    bool(confirmation11["active_enabled"]), dtype=np.bool_
                ),
                confirmed_enabled13=np.asarray(
                    bool(confirmation13["active_enabled"]), dtype=np.bool_
                ),
            )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _routing_summary(
    layers: np.ndarray,
    teacher_layer: np.ndarray,
    task_id: np.ndarray,
    episode_index: np.ndarray,
) -> dict[str, object]:
    selected = np.asarray(layers, dtype=np.int64).reshape(-1)
    teacher = np.asarray(teacher_layer, dtype=np.int64).reshape(-1)
    weights = route_first_group_weights(task_id, episode_index)
    early = selected < 27
    false_shallow = ((selected == 11) & (teacher != 11)) | (
        (selected == 13) & (teacher == 27)
    )
    early_mass = float(weights[early].sum())
    false_mass = float(weights[false_shallow].sum())
    counts = Counter(int(value) for value in selected)
    return {
        "rows": int(selected.size),
        "selected_layer_counts": {
            str(layer): int(counts.get(layer, 0)) for layer in (11, 13, 27)
        },
        "group_equal_early_exit_coverage": float(early_mass / weights.sum()),
        "group_equal_false_shallow_rate_among_early_exits": float(
            false_mass / early_mass
        )
        if early_mass > 0.0
        else 1.0,
        "group_equal_executed_depth_ratio_to_l27": float(
            np.sum(weights * selected) / (weights.sum() * 27.0)
        ),
        "group_equal_layer_count_reduction_ratio": float(
            1.0 - np.sum(weights * selected) / (weights.sum() * 27.0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/route_first_calibration_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_path = args.aggregate.expanduser().resolve(strict=True)
    router_path = args.router.expanduser().resolve(strict=True)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    incomplete_dir = output_dir.with_name(output_dir.name + ".incomplete")
    published_result = args.published_result.expanduser().resolve()
    published_sha = published_result.with_suffix(".sha256")
    if output_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if published_result.exists() or published_sha.exists():
        raise FileExistsError(f"refusing to overwrite {published_result}")

    print("[1/5] validating pre-registered protocol and exact states-8/9 grid")
    protocol = _load_protocol(protocol_path)
    protocol_sha256 = _sha256_file(protocol_path)
    aggregate = load_route_first_teacher_aggregate(aggregate_path)
    expected_grid = tuple((task, episode) for task in range(10) for episode in (8, 9))
    if aggregate.episode_grid != expected_grid:
        raise ValueError("calibration aggregate must be exact tasks 0-9 x states 8-9")
    frozen = protocol["frozen_score_model"]
    if (
        _sha256_file(router_path) != frozen["file_sha256"]
        or frozen["calibration_status_required"]
        != ROUTE_FIRST_ROUTER_CALIBRATION_STATUS
        or frozen["selected_candidate"] != "pca64_l2_0.3"
    ):
        raise ValueError("frozen route-first score model binding differs")
    router, router_metadata = load_uncalibrated_route_first_router(router_path)
    if router.head11.pca_rank != 64 or router.head13.pca_rank != 64:
        raise ValueError("frozen route-first PCA rank differs")
    if router.head11.l2 != 0.3 or router.head13.l2 != 0.3:
        raise ValueError("frozen route-first L2 differs")
    scores = router.probabilities(aggregate.features)
    if np.any(scores[:, 0] > scores[:, 1]):
        raise RuntimeError("route-first score nesting differs")

    confidence = float(protocol["statistics"]["confidence_level"])
    state8 = aggregate.episode_index == 8
    state9 = aggregate.episode_index == 9
    selection_rules = {
        head: RouteFirstThresholdRule.from_mapping(
            protocol["threshold_selection"][f"safe{head}"], selection=True
        )
        for head in (11, 13)
    }
    confirmation_rules = {
        head: RouteFirstThresholdRule.from_mapping(
            protocol["one_shot_confirmation"][f"safe{head}"], selection=False
        )
        for head in (11, 13)
    }

    print("[2/5] selecting thresholds on state 8 only")
    selection11 = select_route_first_threshold(
        scores[state8, 0],
        route_first_safe_label(aggregate.teacher_layer[state8], head=11),
        aggregate.task_id[state8],
        aggregate.episode_index[state8],
        rule=selection_rules[11],
        confidence_level=confidence,
    )
    selection13 = select_route_first_threshold(
        scores[state8, 1],
        route_first_safe_label(aggregate.teacher_layer[state8], head=13),
        aggregate.task_id[state8],
        aggregate.episode_index[state8],
        rule=selection_rules[13],
        confidence_level=confidence,
    )

    print("[3/5] one-shot confirming exact thresholds on state 9")
    confirmation11 = confirm_route_first_threshold(
        selection11,
        scores[state9, 0],
        route_first_safe_label(aggregate.teacher_layer[state9], head=11),
        aggregate.task_id[state9],
        aggregate.episode_index[state9],
        rule=confirmation_rules[11],
        confidence_level=confidence,
    )
    confirmation13 = confirm_route_first_threshold(
        selection13,
        scores[state9, 1],
        route_first_safe_label(aggregate.teacher_layer[state9], head=13),
        aggregate.task_id[state9],
        aggregate.episode_index[state9],
        rule=confirmation_rules[13],
        confidence_level=confidence,
    )
    holdout_authorized = bool(confirmation13["active_enabled"])
    state9_layers = route_first_confirmed_layers(
        scores[state9], confirmation11, confirmation13
    )
    routing = _routing_summary(
        state9_layers,
        aggregate.teacher_layer[state9],
        aggregate.task_id[state9],
        aggregate.episode_index[state9],
    )

    print("[4/5] publishing fail-closed calibrated artifact and audit scores")
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    calibrated_path = incomplete_dir / "router_calibrated.npz"
    scores_path = incomplete_dir / "calibration_scores.npz"
    try:
        save_calibrated_route_first_router(
            calibrated_path,
            router,
            source_router_sha256=frozen["file_sha256"],
            calibration_payload_sha256=aggregate.payload_sha256,
            calibration_file_sha256=aggregate.file_sha256,
            protocol_file_sha256=protocol_sha256,
            selection11=selection11,
            selection13=selection13,
            confirmation11=confirmation11,
            confirmation13=confirmation13,
            engineering_holdout_authorized=holdout_authorized,
        )
        loaded, calibrated_metadata = load_calibrated_route_first_router(
            calibrated_path
        )
        maximum_score_error = float(
            np.max(np.abs(loaded.probabilities(aggregate.features) - scores))
        )
        if not np.allclose(
            loaded.probabilities(aggregate.features), scores, rtol=2e-5, atol=2e-6
        ):
            raise RuntimeError("calibrated route-first score roundtrip differs")
        _save_scores(
            scores_path,
            aggregate=aggregate,
            scores=scores,
            selected_layers=state9_layers,
            selection11=selection11,
            selection13=selection13,
            confirmation11=confirmation11,
            confirmation13=confirmation13,
        )
        label_counts = Counter(int(value) for value in aggregate.teacher_layer)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": (
                "PASS_ONE_SHOT_CONFIRMATION_ENGINEERING_HOLDOUT_READY"
                if holdout_authorized
                else "FAIL_SAFE13_CONFIRMATION_HOLDOUT_NOT_AUTHORIZED"
            ),
            "created_at": datetime.now().astimezone().isoformat(),
            "scope": "states8_selection_state9_confirmation_not_holdout_not_D9",
            "inputs": {
                "aggregate": _display_path(aggregate_path),
                "aggregate_rows": aggregate.rows,
                "aggregate_episodes": len(aggregate.episode_grid),
                "aggregate_payload_sha256": aggregate.payload_sha256,
                "aggregate_file_sha256": aggregate.file_sha256,
                "teacher_layer_counts": {
                    str(layer): int(label_counts.get(layer, 0))
                    for layer in (11, 13, 27)
                },
                "router": _display_path(router_path),
                "router_file_sha256": frozen["file_sha256"],
                "router_metadata": router_metadata,
                "protocol": _display_path(protocol_path),
                "protocol_file_sha256": protocol_sha256,
            },
            "state8_threshold_selection": {
                "safe11": selection11,
                "safe13": selection13,
            },
            "state9_one_shot_confirmation": {
                "safe11": confirmation11,
                "safe13": confirmation13,
                "thresholds_changed_from_state8": False,
                "routing_summary": routing,
            },
            "authorization": {
                "engineering_holdout_states10_to11": holdout_authorized,
                "active_control": False,
                "historical_D9_states40_to49": False,
            },
            "artifacts": {
                "calibrated_router": _display_path(
                    output_dir / calibrated_path.name
                ),
                "calibrated_router_file_sha256": _sha256_file(calibrated_path),
                "calibration_scores": _display_path(output_dir / scores_path.name),
                "calibration_scores_file_sha256": _sha256_file(scores_path),
                "maximum_score_roundtrip_error": maximum_score_error,
                "calibrated_metadata": calibrated_metadata,
            },
            "claim_boundary": {
                "engineering_thresholds_confirmed": holdout_authorized,
                "formal_safety_guarantee": False,
                "states10_to11_opened": False,
                "historical_D9_states40_to49_opened": False,
                "active_control_run": False,
                "closed_loop_improvement_demonstrated": False,
            },
        }
        result_payload = _json_bytes(result)
        with (incomplete_dir / "route_first_stage6_calibration.json").open(
            "xb"
        ) as output_file:
            output_file.write(result_payload)
        incomplete_dir.replace(output_dir)
    except Exception:
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir)
        raise

    _write_exclusive(published_result, result_payload)
    result_digest = _sha256_file(published_result)
    _write_exclusive(
        published_sha,
        f"{result_digest}  {published_result.name}\n".encode("ascii"),
    )
    print("[5/5] calibration result published; active control remains disabled")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
