from scripts.dynamic_compute.collect_m418_persistent_tasks import (
    InitialStateWindowTaskSuite,
    POLICY_MODEL_CLASSES,
    SelectedInitialStateTaskSuite,
    engineering_status_ok,
    resolve_episode_indices,
)


def _row(task_id, episode_idx):
    return {
        "status": "PASS",
        "task_id": task_id,
        "episode_idx": episode_idx,
        "policy_calls": 2,
        "action_chunk_lengths": [8, 8],
        "action_chunk_sha256": ["a" * 64, "b" * 64],
        "latency_ms_by_call": [1.0, 2.0],
    }


def test_engineering_status_requires_the_complete_task_episode_grid():
    rows = [_row(task, episode) for task in (0, 2) for episode in range(3)]

    assert engineering_status_ok(
        policy="full_depth",
        model_class=POLICY_MODEL_CLASSES["full_depth"],
        task_ids=[0, 2],
        episodes_per_task=3,
        episode_records=rows,
        telemetry_errors=0,
    )
    assert not engineering_status_ok(
        policy="full_depth",
        model_class=POLICY_MODEL_CLASSES["full_depth"],
        task_ids=[0, 2],
        episodes_per_task=3,
        episode_records=rows[:-1],
        telemetry_errors=0,
    )


def test_engineering_status_rejects_wrong_policy_class_and_telemetry_error():
    rows = [_row(1, 0)]

    assert not engineering_status_ok(
        policy="early_exit",
        model_class=POLICY_MODEL_CLASSES["full_depth"],
        task_ids=[1],
        episodes_per_task=1,
        episode_records=rows,
        telemetry_errors=0,
    )
    assert not engineering_status_ok(
        policy="early_exit",
        model_class=POLICY_MODEL_CLASSES["early_exit"],
        task_ids=[1],
        episodes_per_task=1,
        episode_records=rows,
        telemetry_errors=1,
    )


def test_engineering_status_accepts_rp_pep_with_early_exit_model_class():
    rows = [_row(0, 27), _row(0, 28)]

    assert engineering_status_ok(
        policy="rp_pep",
        model_class=POLICY_MODEL_CLASSES["rp_pep"],
        task_ids=[0],
        episodes_per_task=2,
        episode_indices=[27, 28],
        episode_records=rows,
        telemetry_errors=0,
    )


def test_engineering_status_accepts_a_nonzero_episode_window_only_when_complete():
    rows = [_row(5, episode) for episode in range(3, 7)]

    assert engineering_status_ok(
        policy="early_exit",
        model_class=POLICY_MODEL_CLASSES["early_exit"],
        task_ids=[5],
        episodes_per_task=4,
        episode_start_index=3,
        episode_records=rows,
        telemetry_errors=0,
    )
    assert not engineering_status_ok(
        policy="early_exit",
        model_class=POLICY_MODEL_CLASSES["early_exit"],
        task_ids=[5],
        episodes_per_task=4,
        episode_start_index=2,
        episode_records=rows,
        telemetry_errors=0,
    )


def test_initial_state_window_forwards_suite_and_rejects_out_of_range():
    class FakeSuite:
        n_tasks = 10

        def get_task_init_states(self, task_id):
            return [f"task{task_id}-episode{episode}" for episode in range(8)]

        def get_task(self, task_id):
            return f"task{task_id}"

    window = InitialStateWindowTaskSuite(FakeSuite(), 3, 4)

    assert window.n_tasks == 10
    assert window.get_task(5) == "task5"
    assert window.get_task_init_states(5) == [
        "task5-episode3",
        "task5-episode4",
        "task5-episode5",
        "task5-episode6",
    ]

    too_far = InitialStateWindowTaskSuite(FakeSuite(), 6, 3)
    try:
        too_far.get_task_init_states(5)
    except ValueError as error:
        assert "cannot select [6:9]" in str(error)
    else:
        raise AssertionError("out-of-range initial-state window was accepted")


def test_arbitrary_episode_selection_requires_the_exact_grid():
    rows = [_row(5, episode) for episode in (2, 14, 22)]

    assert engineering_status_ok(
        policy="full_depth",
        model_class=POLICY_MODEL_CLASSES["full_depth"],
        task_ids=[5],
        episodes_per_task=3,
        episode_indices=[2, 14, 22],
        episode_records=rows,
        telemetry_errors=0,
    )
    assert not engineering_status_ok(
        policy="full_depth",
        model_class=POLICY_MODEL_CLASSES["full_depth"],
        task_ids=[5],
        episodes_per_task=3,
        episode_indices=[2, 14, 21],
        episode_records=rows,
        telemetry_errors=0,
    )


def test_selected_initial_states_preserve_requested_order_and_validate_indices():
    class FakeSuite:
        label = "forwarded"

        def get_task_init_states(self, task_id):
            return [f"task{task_id}-episode{episode}" for episode in range(25)]

    selected = SelectedInitialStateTaskSuite(FakeSuite(), [2, 14, 22])

    assert selected.label == "forwarded"
    assert selected.get_task_init_states(5) == [
        "task5-episode2",
        "task5-episode14",
        "task5-episode22",
    ]

    try:
        SelectedInitialStateTaskSuite(FakeSuite(), [2, 2])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate episode indices were accepted")


def test_resolve_episode_indices_separates_windows_from_explicit_selection():
    assert resolve_episode_indices(
        episode_indices=None,
        episode_start_index=3,
        episodes_per_task=4,
    ) == [3, 4, 5, 6]
    assert resolve_episode_indices(
        episode_indices=[2, 14, 22],
        episode_start_index=0,
        episodes_per_task=3,
    ) == [2, 14, 22]

    for kwargs in (
        {
            "episode_indices": [2, 14, 22],
            "episode_start_index": 1,
            "episodes_per_task": 3,
        },
        {
            "episode_indices": [2, 14],
            "episode_start_index": 0,
            "episodes_per_task": 3,
        },
    ):
        try:
            resolve_episode_indices(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid episode selection was accepted")
