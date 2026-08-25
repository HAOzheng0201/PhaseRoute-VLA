"""Isolated exit controller for route-first single-FM inference.

The historical A1/V3 controller is SHA-protected by the D9 evidence bundle.
This subclass keeps those bytes untouched and is installed only by the new
route-first experiment entrypoint.  It also exposes the historical
``phase_route_runtime_adapter`` attribute so the frozen LIBERO observation
hooks can feed the new runtime without editing their protected source.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from a1.vla.dynamic_compute.telemetry import (
    TelemetryEventCallback,
    emit_telemetry_event,
)
from a1.vla.dynamic_compute.vision_teacher_cache import FlowMatchingTraceCallback
from a1.vla.value_net import ExitController


class RouteFirstControllerError(ValueError):
    """Raised when the isolated controller contract is violated."""


class RouteFirstExitController(ExitController):
    """Pre-route to L13/L27, then execute one flow-matching action head."""

    route_first = True

    @classmethod
    def from_frozen_sparse_controller(
        cls, controller: ExitController
    ) -> "RouteFirstExitController":
        """Clone only immutable configuration/thresholds from an A1 controller."""

        if type(controller) is not ExitController:
            raise RouteFirstControllerError("source must be the frozen ExitController")
        if tuple(controller.exit_id_list) != (3, 11, 13, 27):
            raise RouteFirstControllerError("source sparse layers differ")
        if controller.steps_per_stage != 1:
            raise RouteFirstControllerError("source controller stage width differs")
        if controller.thresholds is None:
            raise RouteFirstControllerError("source controller thresholds are missing")
        cloned = cls(
            controller.value_net,
            exit_id_list=list(controller.exit_id_list),
            steps_per_stage=controller.steps_per_stage,
            exit_dist=controller.exit_dist,
            leq=controller.leq,
            max_layer=controller.max_layer + 1,
        )
        cloned.thresholds = dict(controller.thresholds)
        return cloned

    def install_route_first_adapter(self, adapter: Any) -> None:
        required = (
            "begin_policy_call",
            "target_for_layer",
            "select_action",
            "fail_closed",
        )
        if adapter is None or any(not hasattr(adapter, name) for name in required):
            raise RouteFirstControllerError("route-first adapter interface differs")
        if not bool(getattr(adapter, "route_first", False)):
            raise RouteFirstControllerError("adapter is not marked route-first")
        if not bool(getattr(adapter, "evidence_verified", False)):
            raise RouteFirstControllerError("route-first evidence is not verified")
        if self.phase_plan_active:
            raise RouteFirstControllerError("phase-depth plan is already active")
        if self.phase_route_runtime_adapter is not None:
            raise RouteFirstControllerError("a runtime adapter is already installed")
        if tuple(self.exit_id_list) != (3, 11, 13, 27):
            raise RouteFirstControllerError("route-first sparse layers differ")
        if self.steps_per_stage != 1:
            raise RouteFirstControllerError("route-first requires one call per stage")
        if self.value_net.model.config.action_head != "flow_matching":
            raise RouteFirstControllerError("route-first requires flow matching")
        self.value_net.configure_phase_route_shared_candidates(None)
        # Frozen observation hooks identify an active runtime by this exact
        # attribute.  The subclass owns its semantics; frozen V3 never sees it.
        self.phase_route_runtime_adapter = adapter

    def clear_route_first_adapter(self) -> None:
        self.phase_route_runtime_adapter = None

    def _predict_once(
        self,
        feats,
        layer: int,
        proprio,
        pos_offset,
        fm_trace_callback: Optional[FlowMatchingTraceCallback],
    ) -> torch.Tensor:
        self.value_net.last_fm_calls = 0
        self.value_net.last_fm_steps = 0
        self.value_net.last_rng_burns = 0
        if layer not in (13, 27):
            raise RouteFirstControllerError("selected depth must be L13 or L27")
        if not isinstance(feats, (tuple, list)) or len(feats) != layer + 1:
            raise RouteFirstControllerError(
                "KV cache must end exactly at the selected layer"
            )
        kwargs = {}
        if fm_trace_callback is not None:
            kwargs.update(
                fm_trace_callback=fm_trace_callback,
                fm_trace_context={
                    "candidate_layer": layer,
                    "candidate_role": "route_first_selected_action",
                },
            )
        action = self.value_net.model.predict_actions_flow_matching(
            feats,
            proprio,
            pos_offset,
            **kwargs,
        )
        self.value_net.last_fm_calls = 1
        self.value_net.last_fm_steps = int(
            getattr(
                self.value_net.model.config,
                "num_diffusion_inference_steps",
                0,
            )
        )
        return action

    @torch.no_grad()
    def forward(
        self,
        x,
        i,
        proprio,
        start_idx,
        end_idx,
        pos_offset,
        telemetry_callback: Optional[TelemetryEventCallback] = None,
        fm_trace_callback: Optional[FlowMatchingTraceCallback] = None,
    ):
        del start_idx, end_idx
        if self.thresholds is None:
            raise RouteFirstControllerError("frozen sparse thresholds are missing")
        if type(i) is not int:
            raise RouteFirstControllerError("decoder layer must be an integer")
        if i not in self.exit_id_list:
            return False, None
        adapter = self.phase_route_runtime_adapter
        if adapter is None:
            raise RouteFirstControllerError("route-first adapter is not installed")

        try:
            if not getattr(adapter, "active", False):
                adapter.begin_policy_call(None)
            target_layer = int(adapter.target_for_layer(i))
        except Exception as error:
            if getattr(adapter, "active", False):
                adapter.fail_closed(error)
            target_layer = 27

        if i != target_layer:
            emit_telemetry_event(
                telemetry_callback,
                "exit_candidate",
                {
                    "layer_idx": i,
                    "evaluated": False,
                    "should_exit": False,
                    "route_first_active": True,
                    "route_first_target_layer": target_layer,
                    "action_delta": None,
                    "threshold": None,
                    "fm_calls": 0,
                    "fm_steps": 0,
                    "rng_burns": 0,
                },
            )
            return False, None

        try:
            action = self._predict_once(
                x,
                i,
                proprio,
                pos_offset,
                fm_trace_callback,
            )
        except Exception as error:
            adapter.fail_closed(error)
            emit_telemetry_event(
                telemetry_callback,
                "route_first_action_error",
                {
                    "layer_idx": i,
                    "fallback_layer": 27,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "fm_calls": int(
                        getattr(self.value_net, "last_fm_calls", 0)
                    ),
                },
            )
            if i < 27:
                return False, None
            raise

        decision = adapter.select_action(
            i,
            action,
            fm_calls=int(getattr(self.value_net, "last_fm_calls", 0)),
            telemetry_callback=telemetry_callback,
        )
        emit_telemetry_event(
            telemetry_callback,
            "exit_candidate",
            {
                "layer_idx": i,
                "evaluated": True,
                "should_exit": decision.should_exit,
                "route_first_active": True,
                "route_first_target_layer": target_layer,
                "route_first_gate": decision.telemetry,
                "action_delta": None,
                "threshold": getattr(adapter, "threshold13", None),
                "fm_calls": int(getattr(self.value_net, "last_fm_calls", 0)),
                "fm_steps": int(getattr(self.value_net, "last_fm_steps", 0)),
                "rng_burns": 0,
            },
        )
        if decision.should_exit:
            self.cur_exit_id = i
            return True, decision.selected_action
        return False, None


__all__ = [
    "RouteFirstControllerError",
    "RouteFirstExitController",
]
