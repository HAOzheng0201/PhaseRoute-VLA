from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from a1.vla.dynamic_compute.risk_route13_router import (
    M426A_FEATURE_SCHEMA_VERSION,
    M427_FEATURE_SCHEMA_VERSION,
)
from scripts.dynamic_compute.build_m426_temporal_features import (
    M427_EXPECTED_EPISODES,
    M427_EXPECTED_SEED,
    M427_ROLE_BY_EPISODE,
    format_progress_line,
    m427_data_sufficient,
)
from scripts.dynamic_compute.evaluate_m427_task_jackknife_router import (
    main as evaluate_main,
    m427_science_gates,
)
from scripts.dynamic_compute.train_m427_task_jackknife_router import (
    load_nonsealed_features,
    main as train_main,
    task_jackknife_fit_masks,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m427_frozen_split_and_nonsealed_data_gate() -> None:
    assert M427_EXPECTED_SEED == 20261127
    assert M427_EXPECTED_EPISODES == tuple(range(15))
    assert [index for index, role in M427_ROLE_BY_EPISODE.items() if role == "development"] == list(range(5))
    assert [index for index, role in M427_ROLE_BY_EPISODE.items() if role == "calibration"] == list(range(5, 10))
    assert [index for index, role in M427_ROLE_BY_EPISODE.items() if role == "test"] == list(range(10, 15))
    assert M427_FEATURE_SCHEMA_VERSION != M426A_FEATURE_SCHEMA_VERSION
    assert m427_data_sufficient(
        {"development": 30, "calibration": 30},
        {"development": 19, "calibration": 19},
    )
    assert not m427_data_sufficient(
        {"development": 30, "calibration": 30, "test": 999},
        {"development": 19, "calibration": 19},
    )


def test_m427_builder_direct_cli_resolves_repo_package() -> None:
    command = [
        sys.executable,
        "scripts/dynamic_compute/build_m427_temporal_features.py",
        "--help",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--feature-result" in completed.stdout


def test_m427_progress_redacts_only_sealed_labels() -> None:
    common = {
        "protocol": "m427",
        "index": 1,
        "total": 2,
        "task_id": 0,
        "episode_index": 10,
        "step_id": 42,
        "history_count": 3,
        "raw_exit": 27,
        "route": 27,
    }
    sealed = format_progress_line(role="test", **common)
    assert "REDACTED" in sealed
    assert "raw_exit=27" not in sealed
    assert "binary_target=27" not in sealed
    calibration = format_progress_line(role="calibration", **common)
    assert "raw_exit=27" in calibration
    assert "binary_target=27" in calibration


def test_task_jackknife_masks_exclude_exact_task_and_sealed_rows() -> None:
    task = np.repeat(np.arange(10), 15)
    episode = np.tile(np.arange(15), 10)
    masks = task_jackknife_fit_masks(task, episode)
    assert set(masks) == set(range(10))
    for excluded_task, mask in masks.items():
        assert int(mask.sum()) == 45
        assert not np.any(task[mask] == excluded_task)
        assert np.all(episode[mask] < 5)


def _write_feature_result(tmp_path: Path) -> tuple[Path, str, str]:
    task = np.repeat(np.arange(10), 15)
    episode = np.tile(np.arange(15), 10)
    rows = task.size
    rng = np.random.default_rng(20261127)
    teacher = np.where(episode >= 10, 999, np.where((task + episode) % 4 == 0, 27, 13))
    arrays = {
        "layer13_hidden": rng.normal(size=(rows, 8)),
        "current_proprio": rng.normal(size=(rows, 2)),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": rng.random(size=(rows, 2)) > 0.5,
        "phase_stage": rng.normal(size=(rows, 4)),
        "phase_scalars": rng.random(size=(rows, 3)),
        "step_feature": episode.astype(np.float32) / 15.0,
        "task_id": task,
        "episode_index": episode,
        "step_id": episode,
        "call_index": np.zeros(rows, dtype=np.int16),
        "teacher_route": teacher,
        "identity_sha256": np.asarray(
            [f"{index:064x}".encode("ascii") for index in range(rows)], dtype="S64"
        ),
    }
    arrays_path = tmp_path / "features.npz"
    np.savez_compressed(arrays_path, **arrays)
    checkpoint_sha = "a" * 64
    phase_sha = "b" * 64
    result = {
        "status": "PASS",
        "scope": "m427_temporal_route_feature_table",
        "schema_version": M427_FEATURE_SCHEMA_VERSION,
        "protocol": "m427",
        "checkpoint_sha256": checkpoint_sha,
        "phase_checkpoint_sha256": phase_sha,
        "data_sufficient": True,
        "local_checks": {"synthetic": True},
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "records": rows,
        "role_summaries": {
            "development": {"episode_indices": list(range(5)), "rows": 50},
            "calibration": {"episode_indices": list(range(5, 10)), "rows": 50},
            "test": {"episode_indices": list(range(10, 15)), "rows": 50, "sealed": True},
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path, checkpoint_sha, phase_sha


def test_trainer_loader_returns_no_sealed_rows_or_labels(tmp_path) -> None:
    result_path, checkpoint_sha, phase_sha = _write_feature_result(tmp_path)
    arrays, source = load_nonsealed_features(
        result_path, checkpoint_sha, phase_sha
    )
    assert arrays["teacher_route"].shape == (100,)
    assert set(arrays["episode_index"].tolist()) == set(range(10))
    assert 999 not in arrays["teacher_route"]
    assert source["role_summaries"]["test"]["sealed"] is True
    assert "teacher_distribution" not in source["role_summaries"]["test"]


def _method(
    *,
    exact: float = 0.78,
    false_rows: int = 0,
    error_groups: int = 0,
    recall: float = 0.5,
    coverage: float = 0.5,
    reduction: float = 0.2,
):
    return {
        "metrics": {
            "route27_rows": 30,
            "route27_false_shallow": false_rows,
            "binary_exact_accuracy": exact,
            "safe13_recall": recall,
            "predicted13_coverage": coverage,
        },
        "group_risk": {
            "route27_positive_groups": 19,
            "route27_error_groups": error_groups,
        },
        "estimated_latency": {"reduction_fraction": reduction},
    }


def test_m427_science_gate_requires_group_and_row_safety() -> None:
    controls = [_method(exact=0.75), _method(exact=0.80)]
    assert all(m427_science_gates(_method(), controls).values())
    row_failure = m427_science_gates(_method(false_rows=1), controls)
    assert row_failure["sealed_route27_false_shallow_rows_zero"] is False
    assert row_failure["safety_not_worse_than_learning_controls"] is False
    group_failure = m427_science_gates(_method(error_groups=1), controls)
    assert group_failure["sealed_route27_error_groups_zero"] is False
    assert group_failure["safety_not_worse_than_learning_controls"] is False


def _write_end_to_end_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    tasks = []
    episodes = []
    teachers = []
    signals = []
    for task in range(10):
        for episode in range(15):
            positive_group = (task + episode) % 5 < 3
            for row in range(2):
                route = 27 if positive_group and row == 0 else 13
                tasks.append(task)
                episodes.append(episode)
                teachers.append(route)
                signals.append(-3.0 if route == 27 else 3.0)
    task = np.asarray(tasks, dtype=np.int16)
    episode = np.asarray(episodes, dtype=np.int8)
    teacher = np.asarray(teachers, dtype=np.int16)
    signal = np.asarray(signals, dtype=np.float64)
    rows = task.size
    rng = np.random.default_rng(20261127)
    arrays = {
        "layer13_hidden": signal[:, None] + rng.normal(scale=0.05, size=(rows, 8)),
        "current_proprio": np.stack([signal, task / 9.0], axis=1),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": np.ones((rows, 2), dtype=np.bool_),
        "phase_stage": signal[:, None] + rng.normal(scale=0.05, size=(rows, 4)),
        "phase_scalars": np.stack(
            [episode / 15.0, teacher == 27, np.full(rows, 0.1)], axis=1
        ),
        "step_feature": episode.astype(np.float32) / 15.0,
        "task_id": task,
        "episode_index": episode,
        "step_id": episode.astype(np.int32),
        "call_index": np.tile(np.arange(2), rows // 2).astype(np.int16),
        "teacher_route": teacher,
        "identity_sha256": np.asarray(
            [f"{index:064x}".encode("ascii") for index in range(rows)], dtype="S64"
        ),
    }
    arrays_path = tmp_path / "e2e_features.npz"
    np.savez_compressed(arrays_path, **arrays)
    checkpoint_sha = "c" * 64
    phase_sha = "d" * 64
    result = {
        "status": "PASS",
        "scope": "m427_temporal_route_feature_table",
        "schema_version": M427_FEATURE_SCHEMA_VERSION,
        "protocol": "m427",
        "checkpoint_sha256": checkpoint_sha,
        "phase_checkpoint_sha256": phase_sha,
        "data_sufficient": True,
        "local_checks": {"synthetic": True},
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "records": rows,
        "role_summaries": {
            "development": {"episode_indices": list(range(5)), "rows": 100},
            "calibration": {"episode_indices": list(range(5, 10)), "rows": 100},
            "test": {"episode_indices": list(range(10, 15)), "rows": 100, "sealed": True},
        },
    }
    result_path = tmp_path / "e2e_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    m424 = {
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha,
        "oracle_ceiling": {"status": "VIABLE", "viable_for_router_training": True},
        "by_oracle_route_layer": {
            "13": {"oracle_latency_ms": {"mean": 2.0}},
            "27": {"oracle_latency_ms": {"mean": 4.0}},
        },
        "policy_summary": {"full_depth": {"cuda_latency_ms": {"mean": 4.2}}},
    }
    m424_path = tmp_path / "m424.json"
    m424_path.write_text(json.dumps(m424), encoding="utf-8")
    return result_path, m424_path, checkpoint_sha, phase_sha


def test_m427_synthetic_fit_save_load_and_sealed_evaluation(
    tmp_path, monkeypatch, capsys
) -> None:
    result_path, m424_path, checkpoint_sha, phase_sha = _write_end_to_end_fixture(
        tmp_path
    )
    fit_dir = tmp_path / "fit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_m427",
            "--feature-result",
            str(result_path),
            "--checkpoint-sha256",
            checkpoint_sha,
            "--phase-checkpoint-sha256",
            phase_sha,
            "--pca-rank",
            "4",
            "--max-iter",
            "30",
            "--output-dir",
            str(fit_dir),
        ],
    )
    train_main()
    fit = json.loads((fit_dir / "fit_result.json").read_text(encoding="utf-8"))
    assert fit["router_calibration_gate"] == "PASS"
    assert fit["sealed_test_evaluated"] is False
    assert all(fit["roundtrip_checks"].values())

    holdout_dir = tmp_path / "holdout"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_m427",
            "--feature-result",
            str(result_path),
            "--fit-result",
            str(fit_dir / "fit_result.json"),
            "--m424-result",
            str(m424_path),
            "--checkpoint-sha256",
            checkpoint_sha,
            "--phase-checkpoint-sha256",
            phase_sha,
            "--output-dir",
            str(holdout_dir),
        ],
    )
    evaluate_main()
    result = json.loads((holdout_dir / "result.json").read_text(encoding="utf-8"))
    # The deliberately conservative min ensemble is safer but less exact than
    # both controls in this synthetic fixture, so the frozen utility-parity
    # gate must reject runtime integration without breaking the pipeline.
    assert result["router_offline_gate"] == "NOT_VIABLE"
    assert result["runtime_integration_allowed"] is False
    assert all(result["engineering_checks"].values())
    assert result["science_gates_passed"] == 9
    assert (
        result["science_gates"]["binary_exact_not_below_worse_learning_control"]
        is False
    )
    capsys.readouterr()
