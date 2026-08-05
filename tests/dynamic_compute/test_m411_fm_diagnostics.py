from types import SimpleNamespace

import torch
from torch import nn

from a1.vla.dynamic_compute.fm_diagnostics import (
    flow_matching_euler_trajectory,
)


class _IdentityVectorField(nn.Module):
    def predict_vector_field(self, kv, proprio, x, t, pos_offset):
        del kv, proprio, t, pos_offset
        return x


class _TinyFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_diffusion_inference_steps=2)
        self.action_head = _IdentityVectorField()


def test_euler_trajectory_retains_states_fields_and_times():
    model = _TinyFlowModel()
    kv = [(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))]
    input_x = torch.full((1, 2, 1), 2.0)

    trace = flow_matching_euler_trajectory(
        model,
        kv,
        proprio=torch.zeros(1, 1, 1),
        pos_offset=torch.zeros(1, dtype=torch.long),
        input_x=input_x,
    )

    assert trace.states.shape == (3, 1, 2, 1)
    assert trace.vector_fields.shape == (2, 1, 2, 1)
    assert trace.times.tolist() == [1.0, 0.5]
    assert trace.step_size == -0.5
    torch.testing.assert_close(trace.states[:, 0, 0, 0], torch.tensor([2.0, 1.0, 0.5]))
    torch.testing.assert_close(trace.final_action, torch.full((1, 2, 1), 0.5))


def test_euler_trajectory_rejects_invalid_steps_and_input_shape():
    model = _TinyFlowModel()
    kv = [(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))]
    kwargs = {
        "model": model,
        "attn_key_values": kv,
        "proprio": torch.zeros(1, 1, 1),
        "pos_offset": torch.zeros(1, dtype=torch.long),
        "input_x": torch.zeros(1, 2, 1),
    }
    for override in ({"steps": 0}, {"input_x": torch.zeros(2, 1)}):
        try:
            flow_matching_euler_trajectory(**(kwargs | override))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid FM diagnostic input was accepted")
