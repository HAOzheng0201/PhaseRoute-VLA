from types import SimpleNamespace

import pytest
import torch

from a1.vla.dynamic_compute.productive_exit import (
    ProductiveExitPlan,
    a1_fm10_rp_pep_plan,
)
from a1.vla.value_net import ActionValueNet, ExitController


ORIGINAL_EXITS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27)


class FakeFlowModel:
    def __init__(self):
        self.config = SimpleNamespace(
            action_head="flow_matching",
            num_diffusion_inference_steps=10,
            num_actions_chunk=2,
            fixed_action_dim=3,
        )

    def predict_actions_flow_matching(
        self,
        kvs,
        proprio,
        pos_offset,
        input_x=None,
        fm_trace_callback=None,
        fm_trace_context=None,
    ):
        del proprio, pos_offset
        if input_x is None:
            input_x = torch.randn(1, 2, 3)
        output = input_x + len(kvs) * 0.01
        if fm_trace_callback is not None:
            fm_trace_callback(
                {
                    **dict(fm_trace_context or {}),
                    "input_x": input_x,
                    "output_action": output,
                    "fm_steps": 10,
                }
            )
        return output


def _value_net(exits, *, plan=None):
    return ActionValueNet(
        exit_list=list(exits),
        exit_head=None,
        model=FakeFlowModel(),
        interval=2,
        threshold_type="cosine",
        anchor=False,
        productive_exit_plan=plan,
    )


def test_preregistered_plan_and_threshold_validation():
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    thresholds = {layer: 0.0 for layer in ORIGINAL_EXITS}
    thresholds.update({3: 0.1, 11: 0.1, 13: 0.1, 27: 1e8})

    plan.validate_thresholds(thresholds, lower_is_easier=True)

    assert plan.eligible_exit_layers == (3, 11, 13, 27)
    assert plan.comparison_reference(3) == 1
    assert plan.comparison_reference(11) == 9
    assert plan.comparison_reference(27) == 25
    assert plan.rng_burn_count(27) == 5
    assert plan.select_eligible_thresholds(
        thresholds, lower_is_easier=True
    ) == (0.1, 0.1, 0.1, 1e8)


def test_plan_rejects_pruned_positive_threshold():
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    thresholds = {layer: 0.0 for layer in ORIGINAL_EXITS}
    thresholds[5] = 0.001
    thresholds[27] = 1e8

    with pytest.raises(ValueError, match="positive thresholds"):
        plan.validate_thresholds(thresholds, lower_is_easier=True)


def test_rng_preserving_pruning_reproduces_retained_actions_and_deltas():
    feats = [(torch.zeros(1, 1), torch.zeros(1, 1)) for _ in range(28)]
    baseline = _value_net(ORIGINAL_EXITS)
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    sparse = _value_net(plan.eligible_exit_layers, plan=plan)

    torch.manual_seed(20264804)
    baseline_rows = {}
    for layer in ORIGINAL_EXITS:
        value, action = baseline(feats, layer, None, 0, 0, None)
        if layer in plan.eligible_exit_layers:
            baseline_rows[layer] = (value.clone(), action.clone())

    torch.manual_seed(20264804)
    sparse_rows = {}
    expected_counts = {3: (2, 1), 11: (2, 2), 13: (1, 0), 27: (2, 5)}
    for layer in plan.eligible_exit_layers:
        value, action = sparse(feats, layer, None, 0, 0, None)
        sparse_rows[layer] = (value.clone(), action.clone())
        calls, burns = expected_counts[layer]
        assert sparse.last_fm_calls == calls
        assert sparse.last_rng_burns == burns

    for layer in plan.eligible_exit_layers:
        torch.testing.assert_close(
            sparse_rows[layer][0],
            baseline_rows[layer][0],
            rtol=0,
            atol=0,
            msg=lambda message, layer=layer: f"layer={layer} delta mismatch: {message}",
        )
        torch.testing.assert_close(
            sparse_rows[layer][1],
            baseline_rows[layer][1],
            rtol=0,
            atol=0,
            msg=lambda message, layer=layer: f"layer={layer} action mismatch: {message}",
        )


def test_productive_controller_resets_only_new_mode_history_per_call():
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    productive_net = _value_net(plan.eligible_exit_layers, plan=plan)
    baseline_net = _value_net(ORIGINAL_EXITS)
    productive_net.action_list.append(torch.ones(1))
    baseline_net.action_list.append(torch.ones(1))
    productive = ExitController(
        productive_net,
        list(plan.eligible_exit_layers),
        steps_per_stage=1,
        max_layer=28,
    )
    baseline = ExitController(
        baseline_net,
        list(ORIGINAL_EXITS),
        steps_per_stage=1,
        max_layer=28,
    )

    productive.set_timestep(10)
    baseline.set_timestep(10)

    assert productive_net.action_list == []
    assert len(baseline_net.action_list) == 1


def test_plan_rejects_anchor_and_mismatched_exit_list():
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    with pytest.raises(ValueError, match="anchor"):
        ActionValueNet(
            exit_list=list(plan.eligible_exit_layers),
            exit_head=None,
            model=FakeFlowModel(),
            interval=2,
            threshold_type="cosine",
            anchor=True,
            productive_exit_plan=plan,
        )
    with pytest.raises(ValueError, match="do not match"):
        _value_net((3, 11, 27), plan=plan)


def test_invalid_plan_shapes_are_rejected():
    with pytest.raises(ValueError, match="retain the final"):
        ProductiveExitPlan(
            original_exit_layers=(1, 3, 5),
            eligible_exit_layers=(1, 3),
            comparison_reference_by_exit=(),
            rng_burns_by_exit=(),
        )
