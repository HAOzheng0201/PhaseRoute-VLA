import json

import pytest

from scripts.dynamic_compute.replay_m420_depth_hysteresis import (
    load_episode_sequences,
    replay_split,
    split_name,
)


LAYERS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]


def _episode(raw_layers):
    rows = []
    for index, layer in enumerate(raw_layers):
        rows.append(
            {
                "source": "synthetic",
                "source_path": "/synthetic.jsonl",
                "episode_id": "libero_spatial:task0:episode0",
                "task_id": 0,
                "episode_idx": 0,
                "step_id": 10 + index * 8,
                "raw_layer": layer,
                "candidate_exit_layers": tuple(LAYERS),
                "raw_fm_calls": LAYERS.index(layer) + 1,
            }
        )
    return {"episode": rows}


def test_replay_reports_stability_cost_and_safety_invariant():
    result = replay_split(_episode([13, 11, 13, 11, 13, 11, 11]), 2)

    assert result["raw"]["switches"] == 5
    assert result["routed"]["switches"] < result["raw"]["switches"]
    assert result["comparison"]["route_changes"] > 0
    assert result["comparison"]["estimated_fm_call_increase"] > 0
    assert result["comparison"]["shallower_than_raw"] == 0
    assert result["routed"]["final_layer_calls"] == 0


@pytest.mark.parametrize(
    "source,task,episode,expected",
    [
        ("m418", 0, 0, "calibration"),
        ("m418", 4, 2, "calibration"),
        ("m418b", 5, 13, "calibration"),
        ("m418", 6, 0, "offline_held_out"),
        ("m418", 9, 2, "offline_held_out"),
        ("m418b", 5, 15, "offline_held_out"),
        ("m418b", 5, 26, "offline_held_out"),
        ("m418", 5, 2, "secondary_audit"),
        ("m418b", 5, 14, "secondary_audit"),
        ("m419", 5, 22, "secondary_audit"),
    ],
)
def test_preregistered_split_mapping(source, task, episode, expected):
    assert split_name(source, task, episode) == expected


def test_loader_keeps_same_episode_id_from_different_runs_distinct(tmp_path):
    record = {
        "episode_id": "libero_spatial:task5:episode2",
        "step_id": 10,
        "task_id": 5,
        "candidate_exit_layers": LAYERS,
        "exit_layer": 11,
        "fm_calls": 6,
        "extra": {},
    }
    paths = [tmp_path / "seed1.jsonl", tmp_path / "seed2.jsonl"]
    for path in paths:
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    episodes = load_episode_sequences([("m419", path) for path in paths])

    assert len(episodes) == 2


def test_loader_rejects_duplicate_step_within_one_run(tmp_path):
    record = {
        "episode_id": "libero_spatial:task5:episode2",
        "step_id": 10,
        "task_id": 5,
        "candidate_exit_layers": LAYERS,
        "exit_layer": 11,
        "fm_calls": 6,
        "extra": {},
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate policy call"):
        load_episode_sequences([("m419", path)])
