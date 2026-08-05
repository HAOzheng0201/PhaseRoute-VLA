import torch

from a1.vla.dynamic_compute.budget_controller import (
    TransparentPhaseBudgetController,
)
from a1.vla.dynamic_compute.phase_estimator import PhaseState


def _state(boundary, uncertainty):
    batch_size = len(boundary)
    return PhaseState(
        stage_embedding=torch.zeros(batch_size, 4),
        progress=torch.full((batch_size, 1), 0.5),
        boundary_prob=torch.tensor(boundary, dtype=torch.float32).unsqueeze(1),
        uncertainty=torch.tensor(uncertainty, dtype=torch.float32).unsqueeze(1),
        next_hidden=torch.zeros(1, batch_size, 4),
    )


def test_transparent_rule_maps_default_low_motion_rapid_and_risk_to_b0_b3():
    controller = TransparentPhaseBudgetController()
    selection = controller(
        _state([0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.1, 0.1]),
        progress_delta=torch.tensor([[0.0], [0.0], [0.2], [0.0]]),
        motion_speed=torch.tensor([[1.0], [0.2], [1.0], [1.0]]),
    )

    assert selection.profile_id.tolist() == [0, 1, 2, 3]
    assert selection.reasons == (
        "default",
        "low_motion",
        "rapid_progress",
        "boundary_or_uncertainty",
    )
    torch.testing.assert_close(selection.profile_probs.sum(dim=1), torch.ones(4))


def test_high_uncertainty_overrides_lower_budget_rules():
    controller = TransparentPhaseBudgetController()
    selection = controller(
        _state([0.1], [0.9]),
        progress_delta=torch.tensor([[0.3]]),
        motion_speed=torch.tensor([[0.0]]),
    )
    assert selection.profile_id.item() == 3
