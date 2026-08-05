import torch

from scripts.dynamic_compute.analyze_m49_paired_action_replay import (
    _parse_named_paths,
    action_error_metrics,
)


def test_action_error_metrics_separates_robot_components():
    target = torch.zeros(1, 2, 7)
    predicted = target.clone()
    predicted[..., :3] = 1.0
    predicted[..., 3:6] = 2.0
    predicted[..., 6:] = -3.0

    metrics = action_error_metrics(predicted, target)

    assert metrics["translation_mae"] == 1.0
    assert metrics["rotation_mae"] == 2.0
    assert metrics["gripper_mae"] == 3.0
    assert metrics["first_gripper_direction_mismatch"] == 1.0


def test_named_aggregators_reject_reserved_and_duplicate_names(tmp_path):
    checkpoint = tmp_path / "efa.pt"
    parsed = _parse_named_paths([f"warmup={checkpoint}"])
    assert parsed == [("warmup", checkpoint.resolve())]

    for values in (
        [f"full_token={checkpoint}"],
        [f"efa={checkpoint}", f"efa={checkpoint}"],
        [str(checkpoint)],
    ):
        try:
            _parse_named_paths(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid aggregator specification was accepted")
