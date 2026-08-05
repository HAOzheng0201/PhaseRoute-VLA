import torch

from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseStateEstimator,
    phase_estimator_loss,
)


def _inputs(config, batch_size=3, history_len=4):
    return {
        "visual_summary": torch.randn(batch_size, config.visual_summary_dim),
        "instruction_summary": torch.randn(batch_size, config.instruction_dim),
        "current_proprio": torch.randn(batch_size, config.proprio_dim),
        "proprio_history": torch.randn(batch_size, history_len, config.proprio_dim),
        "proprio_history_mask": torch.tensor(
            [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool
        ),
        "action_history": torch.randn(
            batch_size,
            history_len,
            config.action_horizon,
            config.action_dim,
        ),
        "action_history_mask": torch.tensor(
            [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool
        ),
    }


def test_phase_estimator_output_contract_and_ranges():
    config = PhaseEstimatorConfig(
        visual_summary_dim=16,
        instruction_dim=12,
        visual_proj_dim=8,
        instruction_proj_dim=8,
        proprio_proj_dim=6,
        action_proj_dim=6,
        gru_hidden_dim=10,
        stage_dim=7,
    )
    estimator = PhaseStateEstimator(config)
    state = estimator(**_inputs(config))

    assert state.stage_embedding.shape == (3, 7)
    assert state.progress.shape == (3, 1)
    assert state.boundary_prob.shape == (3, 1)
    assert state.uncertainty.shape == (3, 1)
    assert state.next_hidden.shape == (1, 3, 10)
    assert torch.all((0 <= state.progress) & (state.progress <= 1))
    assert torch.all((0 <= state.boundary_prob) & (state.boundary_prob <= 1))
    assert torch.all((0 <= state.uncertainty) & (state.uncertainty <= 1))
    assert all(torch.isfinite(value).all() for value in state.__dict__.values())


def test_masked_history_values_cannot_change_phase_state():
    torch.manual_seed(7)
    config = PhaseEstimatorConfig(
        visual_summary_dim=8,
        instruction_dim=8,
        visual_proj_dim=6,
        instruction_proj_dim=6,
        proprio_proj_dim=5,
        action_proj_dim=5,
        gru_hidden_dim=9,
        stage_dim=4,
    )
    estimator = PhaseStateEstimator(config).eval()
    inputs = _inputs(config)
    baseline = estimator(**inputs)
    changed = {name: value.clone() for name, value in inputs.items()}
    history_mask = changed["action_history_mask"]
    changed["action_history"][~history_mask] = 1e6
    changed["proprio_history"][~history_mask] = -1e6
    output = estimator(**changed)

    torch.testing.assert_close(output.stage_embedding, baseline.stage_embedding)
    torch.testing.assert_close(output.progress, baseline.progress)
    torch.testing.assert_close(output.boundary_prob, baseline.boundary_prob)
    torch.testing.assert_close(output.next_hidden, baseline.next_hidden)


def test_phase_loss_is_finite_and_uses_only_within_episode_order_pairs():
    config = PhaseEstimatorConfig(
        visual_summary_dim=8,
        instruction_dim=8,
        visual_proj_dim=6,
        instruction_proj_dim=6,
        proprio_proj_dim=5,
        action_proj_dim=5,
        gru_hidden_dim=9,
        stage_dim=4,
    )
    estimator = PhaseStateEstimator(config)
    state = estimator(**_inputs(config))
    losses = phase_estimator_loss(
        state,
        progress_target=torch.tensor([[0.0], [0.5], [1.0]]),
        boundary_target=torch.tensor([[0.0], [1.0], [0.0]]),
        episode_index=torch.tensor([0, 0, 1]),
        call_index=torch.tensor([0, 1, 0]),
    )

    assert set(losses) == {"total", "progress", "boundary", "order", "order_pairs"}
    assert losses["order_pairs"].item() == 1
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["total"].backward()
    assert estimator.progress_head.weight.grad is not None
