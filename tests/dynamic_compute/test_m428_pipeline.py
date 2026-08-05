from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from a1.vla.dynamic_compute.m427_task_jackknife_router import (
    zero_error_clopper_pearson_upper,
)
from a1.vla.dynamic_compute.risk_route13_router import (
    M427_FEATURE_SCHEMA_VERSION,
    M428_FEATURE_SCHEMA_VERSION,
)
from scripts.dynamic_compute.build_m426_temporal_features import (
    M428_EXPECTED_EPISODES,
    M428_EXPECTED_SEED,
    M428_ROLE_BY_EPISODE,
    format_progress_line,
    m428_data_sufficient,
)
from scripts.dynamic_compute.evaluate_m427_task_jackknife_router import (
    main as evaluate_main,
    m427_science_gates,
)
from scripts.dynamic_compute.train_m427_task_jackknife_router import (
    M428_PROTOCOL,
    load_nonsealed_features,
    main as train_main,
    task_jackknife_fit_masks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m428_frozen_split_schema_and_data_gate() -> None:
    assert M428_EXPECTED_SEED == 20261228
    assert M428_EXPECTED_EPISODES == tuple(range(30))
    assert [
        index
        for index, role in M428_ROLE_BY_EPISODE.items()
        if role == "development"
    ] == list(range(10))
    assert [
        index
        for index, role in M428_ROLE_BY_EPISODE.items()
        if role == "calibration"
    ] == list(range(10, 20))
    assert [
        index for index, role in M428_ROLE_BY_EPISODE.items() if role == "test"
    ] == list(range(20, 30))
    assert M428_FEATURE_SCHEMA_VERSION != M427_FEATURE_SCHEMA_VERSION
    assert m428_data_sufficient(
        {"development": 30, "calibration": 30},
        {"development": 29, "calibration": 29},
    )
    assert not m428_data_sufficient(
        {"development": 30, "calibration": 30, "test": 999},
        {"development": 29, "calibration": 29},
    )
    assert not m428_data_sufficient(
        {"development": 30, "calibration": 30},
        {"development": 29, "calibration": 28},
    )
    assert zero_error_clopper_pearson_upper(29) <= 0.10
    assert zero_error_clopper_pearson_upper(28) > 0.10


def test_m428_wrappers_resolve_repo_and_freeze_protocol() -> None:
    for script in (
        "build_m428_temporal_features.py",
        "train_m428_task_jackknife_router.py",
        "evaluate_m428_task_jackknife_router.py",
    ):
        completed = subprocess.run(
            [sys.executable, f"scripts/dynamic_compute/{script}", "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--output-dir" in completed.stdout

    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_compute/build_m428_temporal_features.py",
            "--protocol",
            "m427",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "freezes protocol=m428" in rejected.stderr


def test_m428_front4_launcher_freezes_seed_episode_count_and_gpu_set() -> None:
    wrapper = (
        REPO_ROOT / "scripts/dynamic_compute/run_m428_teacher_cache_front4.sh"
    ).read_text(encoding="utf-8")
    common = (
        REPO_ROOT / "scripts/dynamic_compute/run_m425b_teacher_cache_front4.sh"
    ).read_text(encoding="utf-8")
    assert "M428_ALLOW_THIRTY_EPISODES=1" in wrapper
    assert "20261228" in wrapper
    assert "  30" in wrapper
    assert "declare -a gpus=(0 1 2 3)" in common
    assert "M428_ALLOW_THIRTY_EPISODES" in common


def test_m428_progress_redacts_only_sealed_labels() -> None:
    common = {
        "protocol": "m428",
        "index": 1,
        "total": 2,
        "task_id": 0,
        "episode_index": 20,
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


def test_m428_task_jackknife_masks_exclude_task_and_nondevelopment() -> None:
    task = np.repeat(np.arange(10), 30)
    episode = np.tile(np.arange(30), 10)
    masks = task_jackknife_fit_masks(task, episode, M428_PROTOCOL)
    assert set(masks) == set(range(10))
    for excluded_task, mask in masks.items():
        assert int(mask.sum()) == 90
        assert not np.any(task[mask] == excluded_task)
        assert np.all(episode[mask] < 10)


def _write_m428_feature_result(tmp_path: Path) -> tuple[Path, str, str]:
    task = np.repeat(np.arange(10), 30)
    episode = np.tile(np.arange(30), 10)
    rows = task.size
    rng = np.random.default_rng(20261228)
    teacher = np.where(
        episode >= 20,
        999,
        np.where((task + episode) % 4 == 0, 27, 13),
    )
    arrays = {
        "layer13_hidden": rng.normal(size=(rows, 8)),
        "current_proprio": rng.normal(size=(rows, 2)),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": rng.random(size=(rows, 2)) > 0.5,
        "phase_stage": rng.normal(size=(rows, 4)),
        "phase_scalars": rng.random(size=(rows, 3)),
        "step_feature": episode.astype(np.float32) / 30.0,
        "task_id": task,
        "episode_index": episode,
        "step_id": episode,
        "call_index": np.zeros(rows, dtype=np.int16),
        "teacher_route": teacher,
        "identity_sha256": np.asarray(
            [f"{index:064x}".encode("ascii") for index in range(rows)],
            dtype="S64",
        ),
    }
    arrays_path = tmp_path / "features.npz"
    np.savez_compressed(arrays_path, **arrays)
    checkpoint_sha = "a" * 64
    phase_sha = "b" * 64
    result = {
        "status": "PASS",
        "scope": M428_PROTOCOL.feature_scope,
        "schema_version": M428_FEATURE_SCHEMA_VERSION,
        "protocol": "m428",
        "checkpoint_sha256": checkpoint_sha,
        "phase_checkpoint_sha256": phase_sha,
        "data_sufficient": True,
        "local_checks": {"synthetic": True},
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "records": rows,
        "role_summaries": {
            "development": {"episode_indices": list(range(10)), "rows": 100},
            "calibration": {
                "episode_indices": list(range(10, 20)),
                "rows": 100,
            },
            "test": {
                "episode_indices": list(range(20, 30)),
                "rows": 100,
                "sealed": True,
            },
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path, checkpoint_sha, phase_sha


def test_m428_trainer_loader_never_returns_sealed_rows(tmp_path: Path) -> None:
    result_path, checkpoint_sha, phase_sha = _write_m428_feature_result(tmp_path)
    arrays, source = load_nonsealed_features(
        result_path,
        checkpoint_sha,
        phase_sha,
        M428_PROTOCOL,
    )
    assert arrays["teacher_route"].shape == (200,)
    assert set(arrays["episode_index"].tolist()) == set(range(20))
    assert 999 not in arrays["teacher_route"]
    assert source["role_summaries"]["test"]["sealed"] is True
    assert "teacher_distribution" not in source["role_summaries"]["test"]


def _method(
    *,
    exact: float = 0.78,
    route27_rows: int = 30,
    positive_groups: int = 29,
    false_rows: int = 0,
    error_groups: int = 0,
    recall: float = 0.5,
    coverage: float = 0.5,
    reduction: float = 0.2,
) -> dict:
    return {
        "metrics": {
            "route27_rows": route27_rows,
            "route27_false_shallow": false_rows,
            "binary_exact_accuracy": exact,
            "safe13_recall": recall,
            "predicted13_coverage": coverage,
        },
        "group_risk": {
            "route27_positive_groups": positive_groups,
            "route27_error_groups": error_groups,
        },
        "estimated_latency": {"reduction_fraction": reduction},
    }


def test_m428_science_gate_requires_29_sealed_groups() -> None:
    controls = [_method(exact=0.75), _method(exact=0.80)]
    assert all(m427_science_gates(_method(), controls, M428_PROTOCOL).values())
    insufficient = m427_science_gates(
        _method(positive_groups=28), controls, M428_PROTOCOL
    )
    assert insufficient["sealed_positive_groups_at_least_29"] is False
    unsafe = m427_science_gates(
        _method(false_rows=1, error_groups=1), controls, M428_PROTOCOL
    )
    assert unsafe["sealed_route27_false_shallow_rows_zero"] is False
    assert unsafe["sealed_route27_error_groups_zero"] is False


def _write_m428_end_to_end_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, str, str]:
    tasks = []
    episodes = []
    teachers = []
    signals = []
    for task_id in range(10):
        for episode_index in range(30):
            positive_group = (task_id + episode_index) % 5 < 3
            for row in range(2):
                route = 27 if positive_group and row == 0 else 13
                tasks.append(task_id)
                episodes.append(episode_index)
                teachers.append(route)
                signals.append(-3.0 if route == 27 else 3.0)
    task = np.asarray(tasks, dtype=np.int16)
    episode = np.asarray(episodes, dtype=np.int8)
    teacher = np.asarray(teachers, dtype=np.int16)
    signal = np.asarray(signals, dtype=np.float64)
    rows = task.size
    rng = np.random.default_rng(20261228)
    arrays = {
        "layer13_hidden": signal[:, None]
        + rng.normal(scale=0.05, size=(rows, 8)),
        "current_proprio": np.stack([signal, task / 9.0], axis=1),
        "proprio_history": rng.normal(size=(rows, 2, 2)),
        "action_history": rng.normal(size=(rows, 2, 2, 1)),
        "history_mask": np.ones((rows, 2), dtype=np.bool_),
        "phase_stage": signal[:, None]
        + rng.normal(scale=0.05, size=(rows, 4)),
        "phase_scalars": np.stack(
            [episode / 30.0, teacher == 27, np.full(rows, 0.1)], axis=1
        ),
        "step_feature": episode.astype(np.float32) / 30.0,
        "task_id": task,
        "episode_index": episode,
        "step_id": episode.astype(np.int32),
        "call_index": np.tile(np.arange(2), rows // 2).astype(np.int16),
        "teacher_route": teacher,
        "identity_sha256": np.asarray(
            [f"{index:064x}".encode("ascii") for index in range(rows)],
            dtype="S64",
        ),
    }
    arrays_path = tmp_path / "e2e_features.npz"
    np.savez_compressed(arrays_path, **arrays)
    checkpoint_sha = "c" * 64
    phase_sha = "d" * 64
    result = {
        "status": "PASS",
        "scope": M428_PROTOCOL.feature_scope,
        "schema_version": M428_FEATURE_SCHEMA_VERSION,
        "protocol": "m428",
        "checkpoint_sha256": checkpoint_sha,
        "phase_checkpoint_sha256": phase_sha,
        "data_sufficient": True,
        "local_checks": {"synthetic": True},
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
        "records": rows,
        "role_summaries": {
            "development": {"episode_indices": list(range(10)), "rows": 200},
            "calibration": {
                "episode_indices": list(range(10, 20)),
                "rows": 200,
            },
            "test": {
                "episode_indices": list(range(20, 30)),
                "rows": 200,
                "sealed": True,
            },
        },
    }
    result_path = tmp_path / "e2e_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    m424 = {
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha,
        "oracle_ceiling": {
            "status": "VIABLE",
            "viable_for_router_training": True,
        },
        "by_oracle_route_layer": {
            "13": {"oracle_latency_ms": {"mean": 2.0}},
            "27": {"oracle_latency_ms": {"mean": 4.0}},
        },
        "policy_summary": {
            "full_depth": {"cuda_latency_ms": {"mean": 4.2}}
        },
    }
    m424_path = tmp_path / "m424.json"
    m424_path.write_text(json.dumps(m424), encoding="utf-8")
    return result_path, m424_path, checkpoint_sha, phase_sha


def test_m428_synthetic_fit_roundtrip_and_sealed_evaluation(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    result_path, m424_path, checkpoint_sha, phase_sha = (
        _write_m428_end_to_end_fixture(tmp_path)
    )
    fit_dir = tmp_path / "fit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_m428",
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
    train_main("m428")
    fit = json.loads((fit_dir / "fit_result.json").read_text(encoding="utf-8"))
    assert fit["protocol"] == "m428"
    assert fit["scope"] == M428_PROTOCOL.fit_scope
    assert fit["router_calibration_gate"] == "PASS"
    assert fit["sealed_test_evaluated"] is False
    assert fit["sealed_test_episodes"] == list(range(20, 30))
    assert all(fit["roundtrip_checks"].values())

    holdout_dir = tmp_path / "holdout"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_m428",
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
    evaluate_main("m428")
    holdout = json.loads(
        (holdout_dir / "result.json").read_text(encoding="utf-8")
    )
    assert holdout["protocol"] == "m428"
    assert holdout["scope"] == M428_PROTOCOL.sealed_scope
    assert holdout["sealed_episode_indices"] == list(range(20, 30))
    assert all(holdout["engineering_checks"].values())
    assert "sealed_positive_groups_at_least_29" in holdout["science_gates"]
    capsys.readouterr()
