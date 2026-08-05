from pathlib import Path

import pytest

from scripts.dynamic_compute.profile_m423_fixed_observations import (
    select_stratified_entries,
    selection_sha256,
)


def _entry(layer: int, task: int, step: int):
    return (
        Path(f"/cache/task{task}"),
        {
            "array_path": f"arrays/call_{step:06d}.npz",
            "episode_id": f"suite:task{task}:episode0",
            "task_id": task,
            "step_id": step,
            "teacher_exit_layer": layer,
        },
    )


def test_selection_is_deterministic_stratified_and_task_diverse():
    entries = []
    for layer in (11, 13, 27):
        for task in range(4):
            entries.append(_entry(layer, task, 10 + layer + task))
            entries.append(_entry(layer, task, 100 + layer + task))

    selected = select_stratified_entries(
        list(reversed(entries)), exit_layers=(11, 13, 27), records_per_exit=4
    )
    selected_again = select_stratified_entries(
        entries, exit_layers=(11, 13, 27), records_per_exit=4
    )

    assert selection_sha256(selected) == selection_sha256(selected_again)
    assert [row[1]["teacher_exit_layer"] for row in selected] == [11] * 4 + [13] * 4 + [27] * 4
    for offset in (0, 4, 8):
        assert {row[1]["task_id"] for row in selected[offset : offset + 4]} == {0, 1, 2, 3}


def test_selection_fails_closed_when_a_stratum_is_too_small():
    entries = [_entry(11, 0, 10), _entry(11, 1, 18)]
    with pytest.raises(ValueError, match="need 4"):
        select_stratified_entries(entries, exit_layers=(11,), records_per_exit=4)


def test_selection_rejects_duplicate_records():
    entry = _entry(11, 0, 10)
    with pytest.raises(ValueError, match="duplicate cache record"):
        select_stratified_entries([entry, entry], exit_layers=(11,), records_per_exit=1)
