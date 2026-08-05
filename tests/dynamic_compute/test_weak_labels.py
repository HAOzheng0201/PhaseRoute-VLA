import pytest

from a1.vla.dynamic_compute.weak_labels import (
    BoundaryLabelConfig,
    build_episode_weak_labels,
    build_weak_labels,
)


def _record(
    episode_id,
    step_id,
    *,
    gripper_state=0.08,
    translation_speed=0.8,
    rotation_speed=0.02,
    action_delta=0.002,
    previous_action_vector=None,
):
    return {
        "episode_id": episode_id,
        "task_id": 0,
        "step_id": step_id,
        "gripper_state": gripper_state,
        "translation_speed": translation_speed,
        "rotation_speed": rotation_speed,
        "candidate_exit_layers": [1, 3, 5],
        "action_delta_by_exit": [0.1, action_delta, None],
        "exit_layer": 3,
        "extra": {"previous_action_vector": previous_action_vector},
    }


def test_progress_uses_policy_call_timebase():
    records = [_record("ep", step) for step in [10, 18, 26]]
    labels = build_episode_weak_labels(records)

    assert [label.call_index for label in labels] == [0, 1, 2]
    assert [label.progress_target for label in labels] == [0.0, 0.5, 1.0]
    assert [label.environment_step_id for label in labels] == [10, 18, 26]


def test_gripper_flip_is_dilated_within_episode():
    records = [
        _record("ep", 10, gripper_state=0.08),
        _record("ep", 18, gripper_state=0.08),
        _record("ep", 26, gripper_state=0.01),
        _record("ep", 34, gripper_state=0.01),
        _record("ep", 42, gripper_state=0.01),
    ]
    labels = build_episode_weak_labels(
        records,
        BoundaryLabelConfig(dilation_radius=1),
    )

    assert labels[2].boundary_events["gripper_flip"] is True
    assert [label.boundary_target_raw for label in labels] == [0, 0, 1, 0, 0]
    assert [label.boundary_target for label in labels] == [0, 1, 1, 1, 0]


def test_speed_rotation_and_fine_transition_events_are_configurable():
    records = [
        _record("ep", 10, translation_speed=0.9, rotation_speed=0.01),
        _record("ep", 18, translation_speed=0.2, rotation_speed=0.10),
    ]
    labels = build_episode_weak_labels(
        records,
        BoundaryLabelConfig(dilation_radius=0),
    )
    events = labels[1].boundary_events

    assert events["translation_speed_change"] is True
    assert events["rotation_speed_change"] is True
    assert events["fine_transition"] is True
    assert labels[1].boundary_score == 3.0


def test_direction_and_action_delta_increase_events():
    records = [
        _record(
            "ep",
            10,
            action_delta=0.001,
            previous_action_vector=[1.0, 0.0, 0.0, 0, 0, 0, 1],
        ),
        _record(
            "ep",
            18,
            action_delta=0.02,
            previous_action_vector=[-1.0, 0.0, 0.0, 0, 0, 0, 1],
        ),
    ]
    labels = build_episode_weak_labels(
        records,
        BoundaryLabelConfig(dilation_radius=0),
    )

    assert labels[1].boundary_events["direction_change"] is True
    assert labels[1].boundary_events["action_delta_increase"] is True


def test_grouping_does_not_dilate_across_episode_boundaries():
    records = [
        _record("ep-a", 10, gripper_state=0.08),
        _record("ep-a", 18, gripper_state=0.01),
        _record("ep-b", 10, gripper_state=0.01),
        _record("ep-b", 18, gripper_state=0.01),
    ]
    labels = build_weak_labels(records, BoundaryLabelConfig(dilation_radius=1))
    by_episode = {}
    for label in labels:
        by_episode.setdefault(label.episode_id, []).append(label)

    assert [label.boundary_target for label in by_episode["ep-a"]] == [1, 1]
    assert [label.boundary_target for label in by_episode["ep-b"]] == [0, 0]


def test_episode_builder_rejects_mixed_episodes():
    with pytest.raises(ValueError, match="exactly one episode"):
        build_episode_weak_labels([_record("a", 0), _record("b", 1)])


def test_boundary_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="low_speed"):
        BoundaryLabelConfig(
            fine_transition_low_speed=0.8,
            fine_transition_high_speed=0.2,
        )
    with pytest.raises(ValueError, match="direction_cosine"):
        BoundaryLabelConfig(direction_cosine_threshold=2.0)

