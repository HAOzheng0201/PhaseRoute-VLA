import numpy as np

from scripts.dynamic_compute.train_m46_frozen_a1_distillation import (
    FrozenA1TeacherDataset,
    _cached_final_gripper_transition_indices,
)


def test_cached_transition_index_scan_reads_final_action_only(tmp_path):
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    entries = []
    for index, transition in enumerate((False, True, False)):
        action = np.zeros((8, 7), dtype=np.float32)
        if transition:
            action[-1, 6] = 1.0
        path = arrays / f"call_{index:06d}.npz"
        np.savez(path, teacher_normalized_action=action)
        entries.append((tmp_path, {"array_path": str(path.relative_to(tmp_path))}))

    dataset = FrozenA1TeacherDataset.__new__(FrozenA1TeacherDataset)
    dataset.entries = entries

    assert _cached_final_gripper_transition_indices(
        dataset, [0, 1, 2], threshold=0.5
    ) == [1]


def test_transition_scan_respects_requested_training_subset(tmp_path):
    path = tmp_path / "sample.npz"
    action = np.zeros((8, 7), dtype=np.float32)
    action[-1, 6] = 1.0
    np.savez(path, teacher_normalized_action=action)
    dataset = FrozenA1TeacherDataset.__new__(FrozenA1TeacherDataset)
    dataset.entries = [(tmp_path, {"array_path": path.name})] * 2

    assert _cached_final_gripper_transition_indices(
        dataset, [1], threshold=0.5
    ) == [1]
