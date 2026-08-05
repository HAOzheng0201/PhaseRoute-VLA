from types import SimpleNamespace

import torch

from a1.vla.dynamic_compute.budget_profiles import (
    BudgetProfileResolver,
    m3_depth_profiles,
)
from a1.vla.dynamic_compute.exit_policy import PhaseAwareExitPolicy
from a1.vla.dynamic_compute.phase_estimator import PhaseState
from a1.vla.value_net import ActionValueNet, ExitController


class CountingActionValueNet(ActionValueNet):
    def __init__(self, value, action):
        torch.nn.Module.__init__(self)
        self.value = value
        self.action = action
        self.calls = 0
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_diffusion_inference_steps=10)
        )

    def forward(self, x, i, proprio, start_idx, end_idx, pos_offset):
        del x, i, proprio, start_idx, end_idx, pos_offset
        self.calls += 1
        return torch.tensor([self.value]), self.action


def _phase(boundary=0.1, uncertainty=0.1):
    return PhaseState(
        stage_embedding=torch.zeros(1, 4),
        progress=torch.tensor([[0.5]]),
        boundary_prob=torch.tensor([[boundary]]),
        uncertainty=torch.tensor([[uncertainty]]),
        next_hidden=torch.zeros(1, 1, 4),
    )


def _controller(value=0.1):
    action = torch.randn(1, 8, 7)
    value_net = CountingActionValueNet(value, action)
    controller = ExitController(
        value_net,
        exit_id_list=[1, 3, 5, 7, 9],
        steps_per_stage=1,
        max_layer=10,
    )
    controller.thresholds = {1: 0.2, 3: 0.2, 5: 0.2, 7: 0.2, 9: 1e8}
    controller.set_timestep(0)
    return controller, value_net, action


def test_phase_plan_skips_fm_below_minimum_exit_and_evaluates_at_minimum():
    controller, value_net, action = _controller()
    resolver = BudgetProfileResolver(m3_depth_profiles(), controller.exit_id_list)
    budget = resolver.resolve(2)
    controller.set_phase_plan(
        policy=PhaseAwareExitPolicy(),
        phase_state=_phase(),
        budget=budget,
        profile_reason="rapid_progress",
    )

    assert budget.min_exit_rank > 0
    early_layer = controller.exit_id_list[budget.min_exit_rank - 1]
    early_flag, early_action = controller(None, early_layer, None, 0, 0, None)
    min_flag, min_action = controller(None, budget.min_exit_layer, None, 0, 0, None)

    assert not early_flag and early_action is None
    assert value_net.calls == 1
    assert min_flag
    torch.testing.assert_close(min_action, action)


def test_phase_threshold_can_change_decision_only_when_plan_is_installed():
    baseline, _, _ = _controller(value=0.22)
    phased, _, _ = _controller(value=0.22)
    budget = BudgetProfileResolver(
        m3_depth_profiles(), phased.exit_id_list
    ).resolve(0)
    phased.set_phase_plan(
        policy=PhaseAwareExitPolicy(),
        phase_state=_phase(),
        budget=budget,
    )

    baseline_flag, _ = baseline(None, 1, None, 0, 0, None)
    phased_flag, _ = phased(None, 1, None, 0, 0, None)
    assert not baseline_flag
    assert phased_flag


def test_phase_risk_guard_defers_until_final_exit_and_emits_reason():
    controller, value_net, action = _controller(value=0.0)
    budget = BudgetProfileResolver(
        m3_depth_profiles(), controller.exit_id_list
    ).resolve(0)
    controller.set_phase_plan(
        policy=PhaseAwareExitPolicy(),
        phase_state=_phase(boundary=0.9),
        budget=budget,
    )
    events = []

    first_flag, first_action = controller(
        None,
        1,
        None,
        0,
        0,
        None,
        telemetry_callback=lambda name, payload: events.append((name, payload)),
    )
    final_flag, final_action = controller(None, 9, None, 0, 0, None)

    assert not first_flag and first_action is None
    assert events[0][1]["phase_reason"] == "phase_risk_guard"
    assert final_flag
    torch.testing.assert_close(final_action, action)
    assert value_net.calls == 2


def test_clearing_phase_plan_restores_baseline_path():
    controller, _, _ = _controller(value=0.22)
    budget = BudgetProfileResolver(
        m3_depth_profiles(), controller.exit_id_list
    ).resolve(0)
    controller.set_phase_plan(
        policy=PhaseAwareExitPolicy(),
        phase_state=_phase(),
        budget=budget,
    )
    controller.clear_phase_plan()
    exit_flag, _ = controller(None, 1, None, 0, 0, None)
    assert not exit_flag
