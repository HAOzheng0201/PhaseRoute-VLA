"""Exact one-layer flow-matching controller for Stage-1 depth baselines."""

from __future__ import annotations

from typing import Any, Optional

import torch

from a1.vla.dynamic_compute.telemetry import (
    TelemetryEventCallback,
    emit_telemetry_event,
)
from a1.vla.dynamic_compute.vision_teacher_cache import FlowMatchingTraceCallback


class FixedLayerControllerError(ValueError):
    """Raised when a fixed-depth experiment violates its frozen contract."""


class FixedLayerFlowMatchingController(torch.nn.Module):
    """Generate exactly one FM action when the transformer reaches ``layer``.

    A1's legacy ``exit_id`` branch returns a hidden/KV prefix but no
    flow-matching action.  This controller follows the normal early-exit
    interface and invokes the frozen action head once at the requested layer;
    it does not compute an adjacent-layer comparison action or threshold.
    """

    stage1_fixed_layer = True

    def __init__(self, model: Any, layer: int) -> None:
        super().__init__()
        if type(layer) is not int or layer not in (11, 13, 27):
            raise FixedLayerControllerError("fixed layer must be one of 11,13,27")
        if getattr(model.config, "action_head", None) != "flow_matching":
            raise FixedLayerControllerError("fixed controller requires flow matching")
        if layer >= int(model.config.n_layers):
            raise FixedLayerControllerError("fixed layer exceeds model depth")
        self.model = model
        self.layer = layer
        self.exit_id_list = [layer]
        self.max_layer = layer
        self.cur_exit_id = layer
        self.cur_step = 0

    def set_timestep(self, timestep: int) -> None:
        if type(timestep) is not int or timestep < 0:
            raise FixedLayerControllerError("timestep must be a non-negative integer")
        self.cur_step = timestep

    @torch.no_grad()
    def forward(
        self,
        x: Any,
        i: int,
        proprio: Any,
        start_idx: int,
        end_idx: int,
        pos_offset: Any,
        telemetry_callback: Optional[TelemetryEventCallback] = None,
        fm_trace_callback: Optional[FlowMatchingTraceCallback] = None,
    ) -> tuple[bool, Optional[torch.Tensor]]:
        del start_idx, end_idx
        if i != self.layer:
            return False, None
        if not isinstance(x, list) or len(x) < self.layer + 1:
            raise FixedLayerControllerError("KV cache does not reach fixed layer")
        trace_kwargs = {}
        if fm_trace_callback is not None:
            trace_kwargs = {
                "fm_trace_callback": fm_trace_callback,
                "fm_trace_context": {
                    "candidate_layer": self.layer,
                    "candidate_role": "fixed_layer_action",
                },
            }
        action = self.model.predict_actions_flow_matching(
            x[: self.layer + 1],
            proprio,
            pos_offset,
            **trace_kwargs,
        )
        self.cur_exit_id = self.layer
        emit_telemetry_event(
            telemetry_callback,
            "exit_candidate",
            {
                "layer_idx": self.layer,
                "evaluated": True,
                "should_exit": True,
                "fixed_layer": True,
                "action_delta": None,
                "threshold": None,
                "fm_calls": 1,
                "fm_steps": int(self.model.config.num_diffusion_inference_steps),
            },
        )
        return True, action


__all__ = [
    "FixedLayerControllerError",
    "FixedLayerFlowMatchingController",
]
