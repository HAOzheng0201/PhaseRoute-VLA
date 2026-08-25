#!/usr/bin/env python3
"""Train and audit the uncalibrated route-first ordinal score model.

This entrypoint is deliberately unable to choose deployment thresholds.  It
uses only the frozen states-0/7 teacher aggregate for candidate selection and
emits two nested safety scores plus leakage-audited OOF evidence.
"""

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

from a1.vla.dynamic_compute.route_first_dataset import (  # noqa: E402
    load_route_first_teacher_aggregate,
)
from a1.vla.dynamic_compute.route_first_router import (  # noqa: E402
    ROUTE_FIRST_ROUTER_CALIBRATION_STATUS,
    load_uncalibrated_route_first_router,
    save_uncalibrated_route_first_router,
)
from a1.vla.dynamic_compute.route_first_training import (  # noqa: E402
    RouteFirstCandidate,
    episode_index_candidate_search,
    fit_final_route_first_router,
    leave_one_task_out_scores,
    ranking_gates,
)


PROTOCOL_SCHEMA = "phase-route-vla.route-first-router-protocol.v1"
RESULT_SCHEMA = "phase-route-vla.route-first-router-training.v1"
OOF_SCHEMA = "phase-route-vla.route-first-oof-scores.v1"


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


def _load_and_validate_protocol(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as input_file:
        protocol = json.load(input_file)
    required = {
        "schema_version",
        "seed",
        "training_episode_indices",
        "calibration_episode_indices_not_opened",
        "engineering_holdout_episode_indices_not_opened",
        "historical_D9_episode_indices_forbidden",
        "training_dataset",
        "targets",
        "candidate_grid",
        "optimization",
        "selection",
        "robustness",
        "weighting",
        "ranking_gates",
        "calibration_boundary",
    }
    if set(protocol) != required or protocol["schema_version"] != PROTOCOL_SCHEMA:
        raise ValueError("route-first router protocol schema or fields differ")
    expected_training = list(range(8))
    expected_calibration = [8, 9]
    expected_holdout = [10, 11]
    expected_d9 = list(range(40, 50))
    if (
        protocol["training_episode_indices"] != expected_training
        or protocol["calibration_episode_indices_not_opened"]
        != expected_calibration
        or protocol["engineering_holdout_episode_indices_not_opened"]
        != expected_holdout
        or protocol["historical_D9_episode_indices_forbidden"] != expected_d9
    ):
        raise ValueError("route-first train/calibration/holdout boundary differs")
    if protocol["targets"] != {
        "safe11": "teacher_layer == 11",
        "safe13": "teacher_layer <= 13",
        "nested_scores": "score11 <= score13",
    }:
        raise ValueError("route-first target contract differs")
    boundary = protocol["calibration_boundary"]
    if boundary != {
        "thresholds_in_training_artifact": False,
        "states_0_to_7_may_set_deployment_thresholds": False,
        "states_8_to_9_required_for_thresholds": True,
        "active_control_before_calibration_gate": False,
    }:
        raise ValueError("route-first calibration boundary differs")
    if not bool(protocol["optimization"].get("deterministic")):
        raise ValueError("route-first training must be deterministic")
    return protocol


def _save_oof_scores(
    path: Path,
    *,
    aggregate,
    candidate: RouteFirstCandidate,
    episode_scores: np.ndarray,
    task_scores: np.ndarray,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite {temporary}")
    try:
        with temporary.open("xb") as output_file:
            np.savez_compressed(
                output_file,
                schema_version=np.asarray(OOF_SCHEMA),
                calibration_status=np.asarray(
                    ROUTE_FIRST_ROUTER_CALIBRATION_STATUS
                ),
                training_payload_sha256=np.asarray(aggregate.payload_sha256),
                training_file_sha256=np.asarray(aggregate.file_sha256),
                selected_pca_rank=np.asarray(candidate.pca_rank, dtype=np.int32),
                selected_l2=np.asarray(candidate.l2, dtype=np.float64),
                episode_oof_scores=np.asarray(episode_scores, dtype=np.float32),
                task_oof_scores=np.asarray(task_scores, dtype=np.float32),
                teacher_layer=aggregate.teacher_layer.astype(np.int16),
                task_id=aggregate.task_id.astype(np.int16),
                episode_index=aggregate.episode_index.astype(np.int16),
                call_ordinal=aggregate.call_ordinal.astype(np.int32),
            )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/route_first_router_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_path = args.aggregate.expanduser().resolve(strict=True)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    incomplete_dir = output_dir.with_name(output_dir.name + ".incomplete")
    published_result = args.published_result.expanduser().resolve()
    published_sha = published_result.with_suffix(".sha256")
    if output_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output_dir}")
    if published_result.exists() or published_sha.exists():
        raise FileExistsError(f"refusing to overwrite {published_result}")

    print("[1/5] validating frozen protocol and states-0/7 aggregate", flush=True)
    protocol = _load_and_validate_protocol(protocol_path)
    aggregate = load_route_first_teacher_aggregate(aggregate_path)
    dataset_contract = protocol["training_dataset"]
    expected_tasks = tuple(int(value) for value in dataset_contract["task_ids"])
    expected_episodes = tuple(
        int(value) for value in protocol["training_episode_indices"]
    )
    expected_grid = tuple(
        (task, episode) for task in expected_tasks for episode in expected_episodes
    )
    if (
        dataset_contract["suite"] != "libero_10"
        or aggregate.rows != int(dataset_contract["rows"])
        or len(aggregate.episode_grid) != int(dataset_contract["episodes"])
        or aggregate.payload_sha256 != dataset_contract["payload_sha256"]
        or aggregate.file_sha256 != dataset_contract["file_sha256"]
        or aggregate.episode_grid != expected_grid
    ):
        raise ValueError("training aggregate differs from the frozen protocol")

    grid = protocol["candidate_grid"]
    candidates = tuple(
        RouteFirstCandidate(int(rank), float(l2))
        for rank in grid["pca_rank"]
        for l2 in grid["l2"]
    )
    maximum_iterations = int(
        protocol["optimization"]["maximum_newton_iterations"]
    )
    if maximum_iterations < 1:
        raise ValueError("maximum Newton iterations must be positive")

    print(
        f"[2/5] episode-index OOF candidate search: {len(candidates)} candidates",
        flush=True,
    )
    search = episode_index_candidate_search(
        aggregate.features,
        aggregate.teacher_layer,
        aggregate.task_id,
        aggregate.episode_index,
        candidates=candidates,
        max_iter=maximum_iterations,
    )
    selected = search["selected"]
    print(f"      selected {selected.name}", flush=True)

    print("[3/5] leave-one-task-out robustness audit", flush=True)
    task_audit = leave_one_task_out_scores(
        aggregate.features,
        aggregate.teacher_layer,
        aggregate.task_id,
        aggregate.episode_index,
        candidate=selected,
        max_iter=maximum_iterations,
    )
    gates = ranking_gates(
        search["selected_metrics"],
        task_audit["metrics"],
        protocol["ranking_gates"],
    )
    passed_gates = all(gates.values())

    print("[4/5] fitting and round-tripping threshold-free affine heads", flush=True)
    router = fit_final_route_first_router(
        aggregate.features,
        aggregate.teacher_layer,
        aggregate.task_id,
        aggregate.episode_index,
        candidate=selected,
        max_iter=maximum_iterations,
    )
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    router_path = incomplete_dir / "router_uncalibrated.npz"
    oof_path = incomplete_dir / "oof_scores.npz"
    try:
        save_uncalibrated_route_first_router(
            router_path,
            router,
            training_payload_sha256=aggregate.payload_sha256,
            training_file_sha256=aggregate.file_sha256,
            task_ids=expected_tasks,
            episode_indices=expected_episodes,
            seed=int(protocol["seed"]),
        )
        loaded_router, loaded_metadata = load_uncalibrated_route_first_router(
            router_path
        )
        fitted_scores = router.probabilities(aggregate.features)
        loaded_scores = loaded_router.probabilities(aggregate.features)
        maximum_roundtrip_error = float(
            np.max(np.abs(fitted_scores - loaded_scores))
        )
        roundtrip_passed = bool(
            np.allclose(fitted_scores, loaded_scores, rtol=2e-5, atol=2e-6)
        )
        metadata_passed = loaded_metadata == {
            "training_payload_sha256": aggregate.payload_sha256,
            "training_file_sha256": aggregate.file_sha256,
            "training_task_ids": list(expected_tasks),
            "training_episode_indices": list(expected_episodes),
            "seed": int(protocol["seed"]),
            "calibration_status": ROUTE_FIRST_ROUTER_CALIBRATION_STATUS,
        }
        if not roundtrip_passed or not metadata_passed:
            raise RuntimeError("uncalibrated router roundtrip verification failed")
        _save_oof_scores(
            oof_path,
            aggregate=aggregate,
            candidate=selected,
            episode_scores=search["selected_scores"],
            task_scores=task_audit["scores"],
        )

        label_counts = Counter(int(value) for value in aggregate.teacher_layer)
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": (
                "PASS_RANKING_GATES_UNCALIBRATED"
                if passed_gates
                else "FAIL_RANKING_GATES_UNCALIBRATED_NOT_DEPLOYABLE"
            ),
            "created_at": datetime.now().astimezone().isoformat(),
            "scope": "states_0_to_7_model_selection_not_calibration_not_holdout_not_D9",
            "inputs": {
                "aggregate": _display_path(aggregate_path),
                "aggregate_rows": aggregate.rows,
                "aggregate_episodes": len(aggregate.episode_grid),
                "aggregate_payload_sha256": aggregate.payload_sha256,
                "aggregate_file_sha256": aggregate.file_sha256,
                "protocol": _display_path(protocol_path),
                "protocol_file_sha256": _sha256_file(protocol_path),
                "task_ids": list(expected_tasks),
                "episode_indices": list(expected_episodes),
                "teacher_layer_counts": {
                    str(layer): int(label_counts.get(layer, 0))
                    for layer in (11, 13, 27)
                },
            },
            "candidate_selection": {
                "split": search["split"],
                "folds": search["folds"],
                "candidate_count": len(candidates),
                "candidates": search["candidates"],
                "selected": {
                    "name": selected.name,
                    "pca_rank": selected.pca_rank,
                    "l2": selected.l2,
                },
                "selected_metrics": search["selected_metrics"],
            },
            "task_robustness_audit": {
                "split": task_audit["split"],
                "folds": task_audit["folds"],
                "metrics": task_audit["metrics"],
                "used_for_candidate_selection": False,
            },
            "ranking_gate": {
                "thresholds": protocol["ranking_gates"],
                "checks": gates,
                "failed_checks": [name for name, passed in gates.items() if not passed],
                "all_passed": passed_gates,
                "standards_changed_after_results": False,
            },
            "runtime_model": {
                "input_dimension": 199,
                "output_scores": ["safe11", "safe13"],
                "nested_scores_enforced": True,
                "affine_parameters": 400,
                "affine_multiply_accumulates": 398,
                "pca_required_at_runtime": False,
                "head11_weight_l2_norm": float(np.linalg.norm(router.head11.weight)),
                "head13_weight_l2_norm": float(np.linalg.norm(router.head13.weight)),
                "calibration_status": ROUTE_FIRST_ROUTER_CALIBRATION_STATUS,
            },
            "artifacts": {
                "router": _display_path(output_dir / router_path.name),
                "router_file_sha256": _sha256_file(router_path),
                "oof_scores": _display_path(output_dir / oof_path.name),
                "oof_scores_file_sha256": _sha256_file(oof_path),
                "roundtrip_maximum_absolute_score_error": maximum_roundtrip_error,
                "roundtrip_allclose": roundtrip_passed,
                "metadata_roundtrip_exact": metadata_passed,
                "contains_deployment_thresholds": False,
            },
            "claim_boundary": {
                "ranking_signal_demonstrated": passed_gates,
                "calibration_states8_to9_opened": False,
                "engineering_holdout_states10_to11_opened": False,
                "historical_D9_states40_to49_opened": False,
                "deployment_thresholds_selected": False,
                "active_control_authorized": False,
                "closed_loop_improvement_demonstrated": False,
            },
        }
        result_payload = _json_bytes(result)
        with (incomplete_dir / "route_first_stage5_training.json").open(
            "xb"
        ) as output_file:
            output_file.write(result_payload)
        incomplete_dir.replace(output_dir)
    except Exception:
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir)
        raise

    _write_exclusive(published_result, result_payload)
    published_digest = _sha256_file(published_result)
    _write_exclusive(
        published_sha,
        f"{published_digest}  {published_result.name}\n".encode("ascii"),
    )
    print("[5/5] published immutable Stage 5 result", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
