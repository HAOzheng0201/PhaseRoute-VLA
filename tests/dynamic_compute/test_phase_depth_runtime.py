import pytest
import torch

from a1.vla.dynamic_compute.phase_depth_runtime import SafePhaseDepthRuntime
from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseState,
    PhaseStateEstimator,
)


def _checkpoint(path):
    torch.manual_seed(5)
    config = PhaseEstimatorConfig(
        visual_summary_dim=4,
        instruction_dim=5,
        proprio_dim=3,
        action_horizon=2,
        action_dim=2,
        visual_proj_dim=4,
        instruction_proj_dim=4,
        proprio_proj_dim=3,
        action_proj_dim=3,
        gru_hidden_dim=5,
        stage_dim=4,
    )
    model = PhaseStateEstimator(config)
    torch.save(
        {
            "schema_version": "phase-route-vla.phase-estimator-checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "model_config": dict(config.__dict__),
            "dataset_sha256": "b" * 64,
        },
        path,
    )


def test_runtime_is_causal_and_uses_actual_exit_list(tmp_path):
    checkpoint = tmp_path / "phase.pt"
    _checkpoint(checkpoint)
    runtime_a = SafePhaseDepthRuntime(
        checkpoint,
        device="cpu",
        eligible_exit_layers=(1, 3, 7, 9),
        history_len=2,
    )
    runtime_b = SafePhaseDepthRuntime(
        checkpoint,
        device="cpu",
        eligible_exit_layers=(1, 3, 7, 9),
        history_len=2,
    )
    inputs = {
        "context": {"episode_id": "episode-0", "step_id": 0},
        "visual_summary": torch.ones(1, 4),
        "instruction_summary": torch.ones(1, 5),
        "normalized_proprio": torch.zeros(3),
    }
    first_a = runtime_a.prepare_plan(**inputs)
    assert runtime_a.current_plan is first_a
    runtime_a.clear_current_plan()
    assert runtime_a.current_plan is None
    runtime_a.update_after_action(
        context=inputs["context"],
        normalized_proprio=torch.zeros(3),
        normalized_action_chunk=torch.ones(2, 2),
    )
    first_b = runtime_b.prepare_plan(**inputs)

    torch.testing.assert_close(
        first_a.phase_state.progress,
        first_b.phase_state.progress,
    )
    assert first_a.budget.min_exit_layer in (1, 3, 7, 9)
    assert not first_a.fallback
    assert runtime_a.error_count == 0


def test_runtime_failure_falls_back_to_high_risk_b3(tmp_path):
    checkpoint = tmp_path / "phase.pt"
    _checkpoint(checkpoint)
    runtime = SafePhaseDepthRuntime(
        checkpoint,
        device="cpu",
        eligible_exit_layers=(1, 3, 5),
    )
    plan = runtime.prepare_plan(
        context={"episode_id": "episode", "step_id": 0},
        visual_summary=torch.ones(1, 99),
        instruction_summary=torch.ones(1, 5),
        normalized_proprio=torch.zeros(3),
    )

    assert plan.fallback
    assert plan.budget.profile.name == "B3"
    assert plan.phase_state.boundary_prob.item() == 1.0
    assert plan.phase_state.uncertainty.item() == 1.0
    assert runtime.error_count == 1


def test_persistent_boundary_probability_routes_only_on_rising_edge(tmp_path):
    checkpoint = tmp_path / "phase.pt"
    _checkpoint(checkpoint)
    runtime = SafePhaseDepthRuntime(
        checkpoint,
        device="cpu",
        eligible_exit_layers=(1, 3, 5, 7, 9, 11, 13),
    )
    def state(boundary):
        return PhaseState(
            stage_embedding=torch.zeros(1, 4),
            progress=torch.zeros(1, 1),
            boundary_prob=torch.tensor([[boundary]]),
            uncertainty=torch.zeros(1, 1),
            next_hidden=torch.zeros(1, 1, 5),
        )

    low, _, low_crossed = runtime._routing_state("episode", state(0.2))
    edge, rise, edge_crossed = runtime._routing_state("episode", state(0.8))
    sustained, _, sustained_crossed = runtime._routing_state("episode", state(0.95))

    assert low.boundary_prob.item() == 0.0 and not low_crossed
    assert edge.boundary_prob.item() == 1.0 and edge_crossed
    assert rise == pytest.approx(0.6)
    assert sustained.boundary_prob.item() == 0.0 and not sustained_crossed
