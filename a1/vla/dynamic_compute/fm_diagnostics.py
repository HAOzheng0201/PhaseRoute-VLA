"""Side-effect-free diagnostics for A1's Euler Flow Matching solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class FlowMatchingEulerTrajectory:
    states: torch.Tensor
    vector_fields: torch.Tensor
    times: torch.Tensor
    step_size: float

    @property
    def final_action(self) -> torch.Tensor:
        return self.states[-1]


def flow_matching_euler_trajectory(
    model: nn.Module,
    attn_key_values: Any,
    proprio: torch.Tensor,
    pos_offset: torch.Tensor,
    input_x: torch.Tensor,
    *,
    steps: int | None = None,
) -> FlowMatchingEulerTrajectory:
    """Replay A1's exact Euler update while retaining every intermediate state."""

    if not attn_key_values or not isinstance(attn_key_values[0], (tuple, list)):
        raise ValueError("attn_key_values must contain (key, value) entries")
    if input_x.ndim != 3:
        raise ValueError("input_x must have shape [B, H, A]")
    configured_steps = int(
        getattr(model.config, "num_diffusion_inference_steps", 10)
    )
    steps = configured_steps if steps is None else int(steps)
    if steps < 1:
        raise ValueError("steps must be positive")
    device = attn_key_values[0][0].device
    dtype = attn_key_values[0][0].dtype
    if input_x.shape[0] != attn_key_values[0][0].shape[0]:
        raise ValueError("input_x batch size does not match KV cache")

    x = input_x.to(device=device, dtype=dtype)
    dt = -1.0 / float(steps)
    states = [x]
    vector_fields = []
    times = []
    t_float = 1.0
    for _ in range(steps):
        t = torch.full((x.shape[0],), t_float, device=device, dtype=dtype)
        vector_field = model.action_head.predict_vector_field(
            attn_key_values,
            proprio,
            x,
            t,
            pos_offset=pos_offset,
        )
        if vector_field.shape != x.shape:
            raise ValueError("Flow Matching vector field does not match action state")
        x = x + dt * vector_field
        vector_fields.append(vector_field)
        states.append(x)
        times.append(t_float)
        t_float += dt

    return FlowMatchingEulerTrajectory(
        states=torch.stack(states),
        vector_fields=torch.stack(vector_fields),
        times=torch.tensor(times, device=device, dtype=torch.float32),
        step_size=dt,
    )
