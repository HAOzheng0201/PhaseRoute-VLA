#!/usr/bin/env python3
"""Evaluate frozen route-first thresholds once on states 10--11."""

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
    RouteFirstThresholdRule,
    load_calibrated_route_first_router,
)
from a1.vla.dynamic_compute.route_first_dataset import (  # noqa: E402
    load_route_first_teacher_aggregate,
)
from a1.vla.dynamic_compute.route_first_holdout import (  # noqa: E402
    evaluate_route_first_holdout,
    load_route_first_holdout_protocol,
)


RESULT_SCHEMA = "phase-route-vla.route-first-holdout-result.v1"
SCORES_SCHEMA = "phase-route-vla.route-first-holdout-scores.v1"


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


def _save_scores(
    path: Path,
    *,
    aggregate,
    scores: np.ndarray,
    selected_layers: np.ndarray,
    threshold13: float,
    router_sha256: str,
    protocol_sha256: str,
) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    try:
        with temporary.open("xb") as output_file:
            np.savez_compressed(
                output_file,
                schema_version=np.asarray(SCORES_SCHEMA),
                aggregate_payload_sha256=np.asarray(aggregate.payload_sha256),
                aggregate_file_sha256=np.asarray(aggregate.file_sha256),
                router_file_sha256=np.asarray(router_sha256),
                protocol_file_sha256=np.asarray(protocol_sha256),
                scores=np.asarray(scores, dtype=np.float32),
                selected_layer=np.asarray(selected_layers, dtype=np.int16),
                teacher_layer=aggregate.teacher_layer.astype(np.int16),
                task_id=aggregate.task_id.astype(np.int16),
                episode_index=aggregate.episode_index.astype(np.int16),
                call_ordinal=aggregate.call_ordinal.astype(np.int32),
                threshold13=np.asarray(threshold13, dtype=np.float64),
                enabled11=np.asarray(False, dtype=np.bool_),
                enabled13=np.asarray(True, dtype=np.bool_),
                active_control_authorized=np.asarray(False, dtype=np.bool_),
            )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/route_first_holdout_protocol.json",
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

    print("[1/4] validating frozen Stage 6 router and exact states-10/11 grid")
    protocol = load_route_first_holdout_protocol(protocol_path)
    protocol_sha256 = _sha256_file(protocol_path)
    frozen = protocol["frozen_calibrated_router"]
    expected_router = (REPO_ROOT / frozen["path"]).resolve(strict=True)
    if router_path != expected_router or _sha256_file(router_path) != frozen[
        "file_sha256"
    ]:
        raise ValueError("holdout calibrated router binding differs")
    for path_key, sha_key in (
        ("calibration_result_path", "calibration_result_sha256"),
        ("calibration_verification_path", "calibration_verification_sha256"),
    ):
        evidence = (REPO_ROOT / frozen[path_key]).resolve(strict=True)
        if _sha256_file(evidence) != frozen[sha_key]:
            raise ValueError(f"holdout evidence binding differs: {path_key}")

    aggregate = load_route_first_teacher_aggregate(aggregate_path)
    expected_grid = tuple((task, state) for task in range(10) for state in (10, 11))
    if aggregate.episode_grid != expected_grid:
        raise ValueError("holdout aggregate must be exact tasks 0-9 x states 10-11")
    router, metadata = load_calibrated_route_first_router(router_path)
    expected_metadata = {
        "source_router_sha256": frozen["source_router_sha256"],
        "threshold11": frozen["threshold11"],
        "enabled11": frozen["enabled11"],
        "threshold13": frozen["threshold13"],
        "enabled13": frozen["enabled13"],
        "engineering_holdout_authorized": frozen[
            "engineering_holdout_authorized"
        ],
        "active_control_authorized": frozen["active_control_authorized"],
        "calibration_status": frozen["calibration_status_required"],
    }
    for key, value in expected_metadata.items():
        if metadata[key] != value:
            raise ValueError(f"holdout router metadata differs: {key}")

    print("[2/4] applying exact L13 threshold without refit or movement")
    scores = router.probabilities(aggregate.features)
    gate = protocol["holdout_gate"]
    diagnostics = protocol["diagnostics"]
    audit = evaluate_route_first_holdout(
        scores,
        aggregate.teacher_layer,
        aggregate.task_id,
        aggregate.episode_index,
        threshold13=float(frozen["threshold13"]),
        enabled11=bool(frozen["enabled11"]),
        enabled13=bool(frozen["enabled13"]),
        expected_episode_indices=(10, 11),
        pooled_rule=RouteFirstThresholdRule.from_mapping(
            gate["pooled_safe13"], selection=False
        ),
        per_episode_rule=RouteFirstThresholdRule.from_mapping(
            gate["per_episode_index_safe13"], selection=False
        ),
        confidence_level=float(protocol["statistics"]["confidence_level"]),
        score_quantiles=tuple(diagnostics["score13_quantiles_report_only"]),
    )
    selected_layers = audit.pop("selected_layers")

    print("[3/4] saving fail-closed audit scores and immutable result")
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    scores_path = incomplete_dir / "holdout_scores.npz"
    try:
        _save_scores(
            scores_path,
            aggregate=aggregate,
            scores=scores,
            selected_layers=selected_layers,
            threshold13=float(frozen["threshold13"]),
            router_sha256=frozen["file_sha256"],
            protocol_sha256=protocol_sha256,
        )
        teacher_counts = Counter(int(value) for value in aggregate.teacher_layer)
        integration_ready = bool(audit["passed"])
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": audit["status"],
            "created_at": datetime.now().astimezone().isoformat(),
            "scope": "states10_to11_engineering_holdout_observation_only",
            "inputs": {
                "aggregate": _display_path(aggregate_path),
                "aggregate_rows": aggregate.rows,
                "aggregate_episodes": len(aggregate.episode_grid),
                "aggregate_payload_sha256": aggregate.payload_sha256,
                "aggregate_file_sha256": aggregate.file_sha256,
                "teacher_layer_counts": {
                    str(layer): int(teacher_counts.get(layer, 0))
                    for layer in (11, 13, 27)
                },
                "calibrated_router": _display_path(router_path),
                "calibrated_router_file_sha256": frozen["file_sha256"],
                "calibrated_router_metadata": metadata,
                "protocol": _display_path(protocol_path),
                "protocol_file_sha256": protocol_sha256,
            },
            "holdout_audit": audit,
            "authorization": {
                "runtime_integration_implementation": integration_ready,
                "active_control": False,
                "generated_state_active_test": False,
                "historical_D9_states40_to49": False,
            },
            "artifacts": {
                "holdout_scores": _display_path(output_dir / scores_path.name),
                "holdout_scores_file_sha256": _sha256_file(scores_path),
            },
            "claim_boundary": {
                "engineering_holdout_passed": integration_ready,
                "formal_safety_guarantee": False,
                "states10_to11_opened": True,
                "historical_D9_states40_to49_opened": False,
                "active_control_run": False,
                "closed_loop_improvement_demonstrated": False,
                "wall_clock_speedup_demonstrated": False,
            },
        }
        result_payload = _json_bytes(result)
        with (incomplete_dir / "route_first_stage7_holdout.json").open(
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
    print("[4/4] holdout result published; active control remains disabled")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
