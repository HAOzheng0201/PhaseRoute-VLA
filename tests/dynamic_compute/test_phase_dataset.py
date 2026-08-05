import numpy as np

from a1.vla.dynamic_compute.phase_dataset import (
    PhaseDatasetConfig,
    build_phase_dataset_arrays,
)


def _phase_call(episode_id, step_id, value, task_id=0):
    del task_id
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "raw_proprio": np.full(8, value, dtype=np.float32),
        "normalized_proprio": np.full(8, value + 0.5, dtype=np.float32),
        "previous_action": np.empty((0,), dtype=np.float32),
        "normalized_action_chunk": np.full((2, 7), value, dtype=np.float32),
        "action_chunk": np.full((2, 7), value, dtype=np.float32),
        "visual_summary": np.full(4, value, dtype=np.float16),
        "instruction_summary": np.full(6, value, dtype=np.float16),
    }


def _telemetry(episode_id, step_id, task_id=0):
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "task_id": task_id,
        "instruction_hash": f"instruction-{task_id}",
        "gripper_state": 0.0,
        "translation_speed": 0.0,
        "rotation_speed": 0.0,
        "candidate_exit_layers": [1],
        "action_delta_by_exit": [0.0],
        "exit_layer": 1,
        "extra": {},
    }


def test_histories_are_right_aligned_and_do_not_cross_episodes():
    phase_calls = [
        _phase_call("episode-a", 10, 1.0),
        _phase_call("episode-a", 18, 2.0),
        _phase_call("episode-a", 26, 3.0),
        _phase_call("episode-b", 10, 9.0),
    ]
    telemetry = [
        _telemetry("episode-a", 10),
        _telemetry("episode-a", 18),
        _telemetry("episode-a", 26),
        _telemetry("episode-b", 10),
    ]

    arrays, metadata = build_phase_dataset_arrays(
        phase_calls,
        telemetry,
        config=PhaseDatasetConfig(history_len=2),
    )

    assert metadata["records"] == 4
    assert arrays["proprio_history_mask"].tolist() == [
        [False, False],
        [False, True],
        [True, True],
        [False, False],
    ]
    np.testing.assert_array_equal(
        arrays["proprio_history"][2, :, 0],
        np.array([1.5, 2.5], dtype=np.float32),
    )
    assert not arrays["action_history_mask"][3].any()
    assert arrays["call_index"].tolist() == [0, 1, 2, 0]


def test_join_rejects_missing_phase_or_telemetry_call():
    phase_calls = [_phase_call("episode", 10, 1.0)]
    telemetry = [_telemetry("episode", 18)]

    try:
        build_phase_dataset_arrays(phase_calls, telemetry)
    except ValueError as error:
        assert "keys differ" in str(error)
    else:
        raise AssertionError("missing aligned call should have failed")


def test_splits_keep_each_episode_wholly_in_one_partition():
    phase_calls = []
    telemetry = []
    for episode_index in range(5):
        episode_id = f"episode-{episode_index}"
        for call_index in range(2):
            step_id = 10 + call_index * 8
            phase_calls.append(_phase_call(episode_id, step_id, episode_index))
            telemetry.append(_telemetry(episode_id, step_id))

    arrays, metadata = build_phase_dataset_arrays(
        phase_calls,
        telemetry,
        config=PhaseDatasetConfig(history_len=2),
    )

    assert metadata["split_episodes"] == {"train": 3, "validation": 1, "test": 1}
    for episode_index in np.unique(arrays["episode_index"]):
        episode_splits = np.unique(arrays["split"][arrays["episode_index"] == episode_index])
        assert len(episode_splits) == 1
