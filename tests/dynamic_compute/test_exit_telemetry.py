from types import SimpleNamespace

import pytest
import torch

from a1.vla.value_net import ActionValueNet, ExitController


class DummyActionValueNet(ActionValueNet):
    def __init__(self, value: float, action: torch.Tensor, fm_steps: int = 10):
        torch.nn.Module.__init__(self)
        self._value = value
        self._action = action
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_diffusion_inference_steps=fm_steps)
        )

    def forward(self, x, i, proprio, start_idx, end_idx, pos_offset):
        del x, i, proprio, start_idx, end_idx, pos_offset
        return torch.tensor([self._value]), self._action


def _build_controller(action: torch.Tensor) -> ExitController:
    controller = ExitController(
        DummyActionValueNet(0.1, action),
        exit_id_list=[1, 3, 5],
        steps_per_stage=8,
        max_layer=6,
    )
    controller.thresholds = {1: 0.2, 3: 0.2, 5: 1e8}
    controller.set_timestep(0)
    return controller


def test_exit_controller_callback_preserves_decision_and_action_bitwise():
    expected_action = torch.randn(1, 8, 7)
    baseline_controller = _build_controller(expected_action)
    telemetry_controller = _build_controller(expected_action)
    events = []

    baseline_flag, baseline_action = baseline_controller(
        None, 1, None, 0, 0, None
    )
    observed_flag, observed_action = telemetry_controller(
        None,
        1,
        None,
        0,
        0,
        None,
        telemetry_callback=lambda name, payload: events.append((name, dict(payload))),
    )

    assert observed_flag is baseline_flag is True
    torch.testing.assert_close(observed_action, baseline_action, rtol=0, atol=0)
    assert len(events) == 1
    name, payload = events[0]
    assert name == "exit_candidate"
    assert payload["layer_idx"] == 1
    assert payload["action_delta"] == pytest.approx(0.1)
    assert payload["fm_calls"] == 1
    assert payload["fm_steps"] == 10
    assert payload["should_exit"] is True


def test_exit_controller_contains_callback_failure():
    expected_action = torch.randn(1, 8, 7)
    controller = _build_controller(expected_action)

    def broken_callback(name, payload):
        del name, payload
        raise RuntimeError("telemetry sink unavailable")

    exit_flag, action = controller(
        None, 1, None, 0, 0, None, telemetry_callback=broken_callback
    )

    assert exit_flag is True
    torch.testing.assert_close(action, expected_action, rtol=0, atol=0)


def test_reused_stage_emits_no_fm_call():
    expected_action = torch.randn(1, 8, 7)
    controller = _build_controller(expected_action)
    controller.cur_exit_id = 3
    controller.set_timestep(1)
    events = []

    exit_flag, action = controller(
        None,
        3,
        None,
        0,
        0,
        None,
        telemetry_callback=lambda name, payload: events.append((name, dict(payload))),
    )

    assert exit_flag is True
    assert action is None
    assert events[0][1]["evaluated"] is False
    assert events[0][1]["fm_calls"] == 0
    assert events[0][1]["fm_steps"] == 0
