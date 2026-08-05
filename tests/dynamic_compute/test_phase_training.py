import hashlib
import json

import numpy as np
import pytest
import torch

from a1.vla.dynamic_compute.phase_dataset import PHASE_DATASET_SCHEMA_VERSION
from a1.vla.dynamic_compute.phase_training import (
    ESTIMATOR_INPUT_NAMES,
    baseline_metrics,
    boundary_metrics,
    load_phase_dataset,
    make_torch_batch,
    select_f1_threshold,
)


def _write_dataset(tmp_path, *, cross_split=False):
    rows = 6
    episode_index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    split = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    if cross_split:
        split[1] = 1
    arrays = {
        "visual_summary": np.ones((rows, 4), dtype=np.float16),
        "instruction_summary": np.ones((rows, 5), dtype=np.float16),
        "current_proprio": np.ones((rows, 3), dtype=np.float32),
        "proprio_history": np.ones((rows, 2, 3), dtype=np.float32),
        "proprio_history_mask": np.array([[0, 1]] * rows, dtype=np.bool_),
        "action_history": np.ones((rows, 2, 2, 2), dtype=np.float32),
        "action_history_mask": np.array([[0, 1]] * rows, dtype=np.bool_),
        "progress_target": np.array(
            [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=np.float32
        ),
        "boundary_target": np.array(
            [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=np.float32
        ),
        "episode_index": episode_index,
        "call_index": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "split": split,
        "current_normalized_action_chunk": np.full(
            (rows, 2, 2), 99.0, dtype=np.float32
        ),
    }
    dataset_path = tmp_path / "phase_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": PHASE_DATASET_SCHEMA_VERSION,
        "records": rows,
        "dataset_sha256": digest,
        "split_records": {"train": 2, "validation": 2, "test": 2},
        "split_episodes": {"train": 1, "validation": 1, "test": 1},
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return dataset_path, metadata_path


def test_loader_validates_hash_splits_and_exposes_only_estimator_inputs(tmp_path):
    dataset_path, metadata_path = _write_dataset(tmp_path)
    bundle = load_phase_dataset(dataset_path, metadata_path)
    batch = make_torch_batch(bundle, "train", torch.device("cpu"))

    assert set(ESTIMATOR_INPUT_NAMES).issubset(batch)
    assert "current_normalized_action_chunk" not in batch
    assert batch["visual_summary"].shape == (2, 4)
    assert batch["action_history_mask"].dtype == torch.bool


def test_loader_rejects_episode_split_leakage(tmp_path):
    dataset_path, metadata_path = _write_dataset(tmp_path, cross_split=True)
    with pytest.raises(ValueError, match="crosses dataset splits"):
        load_phase_dataset(dataset_path, metadata_path)


def test_boundary_threshold_and_baselines_are_deterministic():
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    target = np.array([0, 1, 1, 0])
    threshold = select_f1_threshold(probability, target)
    metrics = boundary_metrics(probability, target, threshold)

    assert metrics["f1"] >= boundary_metrics(probability, target, 0.5)["f1"]
    first = baseline_metrics(target, target, target, target, seed=17)
    second = baseline_metrics(target, target, target, target, seed=17)
    assert first == second
