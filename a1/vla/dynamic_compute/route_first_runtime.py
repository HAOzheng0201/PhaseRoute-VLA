"""Fail-closed route-first runtime for the calibrated L13/L27 router.

This module deliberately reuses the frozen V3 observation/phase context
builder while replacing its action-dependent candidate adapter.  The depth is
therefore selected from the 199-D action-free context before decoder layer 0.
The controller can then skip every non-selected action-head call and execute
flow matching exactly once at L13 or L27.

The loader binds the implementation to the sealed Stage-6 calibrated router
and the Stage-7 engineering-holdout result.  Stage 7 authorizes runtime
integration only, not an active rollout; callers must still opt in through a
separately preregistered experiment entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .route_first_calibration import load_calibrated_route_first_router
from .route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    build_route_first_context_features,
)
from .route_first_router import RouteFirstOrdinalRouter
from .v3.active_runtime import (
    ActivePhaseRouteRuntime,
    ActiveRuntimeArtifacts,
    ActiveRuntimeError,
    load_frozen_phase_route_runtime,
    sha256_file,
)
from .v3.final_router import FinalFiveHeadRouter
from .v3.gripper_v2_protocol import ACTION_DIMENSION, HORIZON
from .v3.runtime_adapter import RuntimeStepDecision, TelemetryCallback


ROUTE_FIRST_ACTIVE_RUNTIME_SCHEMA_VERSION = (
    "phase-route-vla.route-first-active-runtime.v1"
)
ROUTE_FIRST_RUNTIME_STATUS = "PASS_STAGE7_RUNTIME_INTEGRATION_READY"
ROUTE_FIRST_RUNTIME_LAYERS = (13, 27)
ROUTE_FIRST_CALIBRATED_ROUTER_SHA256 = (
    "ae561b77c01bd4c7eee6cc0ff91e215733662544cc1af2e5039b0a8f02c60cc2"
)
ROUTE_FIRST_STAGE7_HOLDOUT_SHA256 = (
    "d9780a5e4765ee9a80165eb790b99b4e9e85fcb1ae6d34ae006ddb72ce48f258"
)
ROUTE_FIRST_STAGE7_HOLDOUT_STATUS = (
    "PASS_ENGINEERING_HOLDOUT_RUNTIME_INTEGRATION_READY"
)


class RouteFirstRuntimeError(ValueError):
    """Raised by strict artifact loaders and invalid controller use."""


def route_first_router_state_sha256(router: RouteFirstOrdinalRouter) -> str:
    """Hash the exact affine heads so in-memory mutation is fail-closed."""

    digest = hashlib.sha256()
    for name, head in (("head11", router.head11), ("head13", router.head13)):
        weight = np.asarray(head.weight, dtype=np.float64).reshape(-1)
        digest.update(name.encode("ascii"))
        digest.update(weight.tobytes(order="C"))
        digest.update(
            json.dumps(
                {
                    "bias": float(head.bias),
                    "pca_rank": int(head.pca_rank),
                    "l2": float(head.l2),
                    "iterations": int(head.iterations),
                },
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        )
    return digest.hexdigest()


def route_first_target_layers(
    router: RouteFirstOrdinalRouter,
    features: np.ndarray,
    *,
    enabled11: bool,
    enabled13: bool,
    threshold13: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scores and fail-closed L13/L27 targets for offline/live parity."""

    if bool(enabled11):
        raise RouteFirstRuntimeError("L11 must remain disabled after Stage 6")
    if not bool(enabled13):
        raise RouteFirstRuntimeError("L13 must remain enabled after Stage 7")
    if not math.isfinite(float(threshold13)) or not 0.0 <= threshold13 <= 1.0:
        raise RouteFirstRuntimeError("L13 threshold is invalid")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ROUTE_FIRST_FEATURE_DIMENSION:
        raise RouteFirstRuntimeError(
            f"route-first features must have shape [N,{ROUTE_FIRST_FEATURE_DIMENSION}]"
        )
    scores = router.probabilities(values)
    targets = np.full(values.shape[0], 27, dtype=np.int16)
    targets[scores[:, 1] >= float(threshold13)] = 13
    return scores, targets


@dataclass(frozen=True)
class RouteFirstRuntimeArtifacts:
    calibrated_router_path: str
    calibrated_router_sha256: str
    stage7_holdout_path: str
    stage7_holdout_sha256: str
    stage7_status: str
    v3_context_artifacts: ActiveRuntimeArtifacts


class RouteFirstRuntimeAdapter:
    """One-call state machine that chooses depth before any candidate action."""

    route_first = True
    def __init__(
        self,
        router: RouteFirstOrdinalRouter,
        metadata: Mapping[str, object],
        artifacts: RouteFirstRuntimeArtifacts,
    ) -> None:
        self.router = router
        self.metadata = dict(metadata)
        self.artifacts = artifacts
        self.evidence_verified = bool(
            artifacts.calibrated_router_sha256
            == ROUTE_FIRST_CALIBRATED_ROUTER_SHA256
            and artifacts.stage7_holdout_sha256
            == ROUTE_FIRST_STAGE7_HOLDOUT_SHA256
            and artifacts.stage7_status == ROUTE_FIRST_STAGE7_HOLDOUT_STATUS
        )
        if not self.evidence_verified:
            raise RouteFirstRuntimeError("route-first Stage-6/7 evidence differs")
        self.router_state_sha256 = route_first_router_state_sha256(router)
        self.enabled11 = bool(self.metadata.get("enabled11"))
        self.enabled13 = bool(self.metadata.get("enabled13"))
        self.threshold13 = float(self.metadata.get("threshold13", float("nan")))
        if self.enabled11 or not self.enabled13:
            raise RouteFirstRuntimeError("runtime requires disabled L11 and enabled L13")
        if not math.isfinite(self.threshold13) or not 0.0 <= self.threshold13 <= 1.0:
            raise RouteFirstRuntimeError("runtime L13 threshold is invalid")
        if not bool(self.metadata.get("engineering_holdout_authorized")):
            raise RouteFirstRuntimeError("engineering holdout did not authorize integration")
        if bool(self.metadata.get("active_control_authorized")):
            raise RouteFirstRuntimeError("Stage-6 artifact active-control flag changed")
        self._active = False
        self._target_layer = 27
        self._scores: tuple[float, float] | None = None
        self._fail_reason: str | None = None
        self._fm_calls = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def target_layer(self) -> int:
        return self._target_layer

    @property
    def scores(self) -> tuple[float, float] | None:
        return self._scores

    @property
    def fail_reason(self) -> str | None:
        return self._fail_reason

    @property
    def fm_calls(self) -> int:
        return self._fm_calls

    def _latch_fallback(self, error: Exception | str) -> None:
        text = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self._fail_reason = text
        self._target_layer = 27

    def fail_closed(self, error: Exception | str) -> None:
        """Public controller hook: veto L13 and lock the call to L27."""

        if not self._active:
            raise RouteFirstRuntimeError("no active route-first policy call")
        self._latch_fallback(error)

    def begin_policy_call(
        self, runtime_inputs: Mapping[str, torch.Tensor] | None
    ) -> None:
        self._active = True
        self._target_layer = 27
        self._scores = None
        self._fail_reason = None
        self._fm_calls = 0
        try:
            if route_first_router_state_sha256(self.router) != self.router_state_sha256:
                raise RouteFirstRuntimeError("route-first router mutated in memory")
            if runtime_inputs is None:
                raise RouteFirstRuntimeError("runtime context is missing")
            feature = build_route_first_context_features(runtime_inputs)
            scores, targets = route_first_target_layers(
                self.router,
                feature.detach().cpu().numpy(),
                enabled11=self.enabled11,
                enabled13=self.enabled13,
                threshold13=self.threshold13,
            )
            self._scores = (float(scores[0, 0]), float(scores[0, 1]))
            self._target_layer = int(targets[0])
        except Exception as error:
            self._latch_fallback(error)

    def target_for_layer(self, layer: int) -> int:
        if not self._active:
            raise RouteFirstRuntimeError("no active route-first policy call")
        if type(layer) is not int or layer < 0:
            self._latch_fallback("decoder layer is invalid")
        if self._target_layer not in ROUTE_FIRST_RUNTIME_LAYERS:
            self._latch_fallback("route-first target layer is invalid")
        return self._target_layer

    @staticmethod
    def _emit(
        callback: TelemetryCallback | None,
        event_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            callback(event_name, payload)
        except Exception:
            pass

    def select_action(
        self,
        layer: int,
        selected_action: torch.Tensor,
        *,
        fm_calls: int,
        telemetry_callback: TelemetryCallback | None = None,
    ) -> RuntimeStepDecision:
        if not self._active:
            raise RouteFirstRuntimeError("no active route-first policy call")
        target = self.target_for_layer(layer)
        valid_action = bool(
            isinstance(selected_action, torch.Tensor)
            and selected_action.is_floating_point()
            and tuple(selected_action.shape) == (1, HORIZON, ACTION_DIMENSION)
            and bool(torch.isfinite(selected_action).all())
        )
        valid_call_count = type(fm_calls) is int and fm_calls == 1
        if layer != target or not valid_action or not valid_call_count:
            reasons = []
            if layer != target:
                reasons.append("selected_layer_differs_from_locked_target")
            if not valid_action:
                reasons.append("selected_action_is_invalid")
            if not valid_call_count:
                reasons.append("flow_matching_call_count_differs_from_one")
            self._latch_fallback(",".join(reasons))
            record = {
                "layer_idx": layer,
                "target_layer": self._target_layer,
                "should_exit": False,
                "fallback": True,
                "fm_calls": int(fm_calls) if type(fm_calls) is int else None,
                "score11": self._scores[0] if self._scores is not None else None,
                "score13": self._scores[1] if self._scores is not None else None,
                "threshold13": self.threshold13,
                "fail_reason": self._fail_reason,
            }
            self._emit(telemetry_callback, "route_first_action_rejected", record)
            return RuntimeStepDecision(layer, False, None, False, record)

        self._fm_calls += fm_calls
        self._active = False
        fallback = layer == 27
        record = {
            "layer_idx": layer,
            "target_layer": layer,
            "should_exit": True,
            "fallback": fallback,
            "fm_calls": self._fm_calls,
            "score11": self._scores[0] if self._scores is not None else None,
            "score13": self._scores[1] if self._scores is not None else None,
            "threshold13": self.threshold13,
            "enabled11": False,
            "enabled13": True,
            "fail_reason": self._fail_reason,
            "calibrated_router_sha256": self.artifacts.calibrated_router_sha256,
            "stage7_holdout_sha256": self.artifacts.stage7_holdout_sha256,
        }
        self._emit(telemetry_callback, "route_first_selected_action", record)
        self._emit(
            telemetry_callback,
            "phase_route_decision",
            {
                "mode": "route_first",
                "selected_layer": layer,
                "fallback": fallback,
                "fm_calls": self._fm_calls,
                "score13": record["score13"],
                "threshold13": self.threshold13,
                "fail_reason": self._fail_reason,
            },
        )
        return RuntimeStepDecision(layer, True, selected_action, True, record)


class ActiveRouteFirstRuntime(ActivePhaseRouteRuntime):
    """V3 context builder paired with the action-free route-first adapter."""

    enabled = True

    def __init__(
        self,
        context_router: FinalFiveHeadRouter,
        phase_estimator: torch.nn.Module,
        route_first_router: RouteFirstOrdinalRouter,
        route_first_metadata: Mapping[str, object],
        route_first_artifacts: RouteFirstRuntimeArtifacts,
        *,
        artifacts: ActiveRuntimeArtifacts,
    ) -> None:
        super().__init__(context_router, phase_estimator, artifacts=artifacts)
        self.route_first_artifacts = route_first_artifacts
        self.adapter = RouteFirstRuntimeAdapter(
            route_first_router,
            route_first_metadata,
            route_first_artifacts,
        )

    def begin_policy_call(self, **kwargs: Any) -> bool:
        result = super().begin_policy_call(**kwargs)
        if self._current is not None:
            self._current["schema_version"] = ROUTE_FIRST_ACTIVE_RUNTIME_SCHEMA_VERSION
            self._current["runtime_mode"] = "route_first_l13_l27"
        return result

    def prepare_policy_call(self) -> bool:
        result = super().prepare_policy_call()
        if self._current is not None:
            self._current.update(
                {
                    "route_first_target_layer": self.adapter.target_layer,
                    "route_first_scores": self.adapter.scores,
                    "route_first_fail_reason": self.adapter.fail_reason,
                }
            )
        return result


def _validate_stage7_evidence(
    holdout_path: Path,
    *,
    holdout_sha256: str,
    calibrated_router_sha256: str,
    router_metadata: Mapping[str, object],
) -> str:
    try:
        result = json.loads(holdout_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RouteFirstRuntimeError("Stage-7 holdout result is unreadable") from error
    if result.get("status") != ROUTE_FIRST_STAGE7_HOLDOUT_STATUS:
        raise RouteFirstRuntimeError("Stage-7 holdout status does not authorize integration")
    inputs = result.get("inputs")
    authorization = result.get("authorization")
    claim = result.get("claim_boundary")
    if not isinstance(inputs, Mapping) or not isinstance(authorization, Mapping):
        raise RouteFirstRuntimeError("Stage-7 evidence structure differs")
    if not isinstance(claim, Mapping):
        raise RouteFirstRuntimeError("Stage-7 claim boundary is missing")
    if inputs.get("calibrated_router_file_sha256") != calibrated_router_sha256:
        raise RouteFirstRuntimeError("Stage-7 evidence binds a different calibrated router")
    sealed_metadata = inputs.get("calibrated_router_metadata")
    if not isinstance(sealed_metadata, Mapping):
        raise RouteFirstRuntimeError("Stage-7 router metadata is missing")
    for name in ("threshold11", "enabled11", "threshold13", "enabled13"):
        if sealed_metadata.get(name) != router_metadata.get(name):
            raise RouteFirstRuntimeError(f"Stage-7 router metadata differs at {name}")
    if (
        not bool(authorization.get("runtime_integration_implementation"))
        or bool(authorization.get("active_control"))
        or bool(authorization.get("generated_state_active_test"))
        or not bool(claim.get("engineering_holdout_passed"))
        or bool(claim.get("active_control_run"))
    ):
        raise RouteFirstRuntimeError("Stage-7 authorization boundary differs")
    if holdout_sha256 != ROUTE_FIRST_STAGE7_HOLDOUT_SHA256:
        raise RouteFirstRuntimeError("Stage-7 holdout SHA-256 differs")
    return str(result["status"])


def load_route_first_active_runtime(
    calibrated_router_path: str | Path,
    stage7_holdout_path: str | Path,
    context_router_path: str | Path,
    phase_checkpoint_path: str | Path,
    *,
    expected_calibrated_router_sha256: str = ROUTE_FIRST_CALIBRATED_ROUTER_SHA256,
    expected_stage7_holdout_sha256: str = ROUTE_FIRST_STAGE7_HOLDOUT_SHA256,
) -> ActiveRouteFirstRuntime:
    """Strictly load Stage-6/7 evidence and the frozen V3 context models."""

    calibrated_path = Path(calibrated_router_path).expanduser().resolve(strict=True)
    holdout_path = Path(stage7_holdout_path).expanduser().resolve(strict=True)
    calibrated_sha = sha256_file(calibrated_path)
    holdout_sha = sha256_file(holdout_path)
    if calibrated_sha != expected_calibrated_router_sha256:
        raise RouteFirstRuntimeError("calibrated route-first router SHA-256 differs")
    if holdout_sha != expected_stage7_holdout_sha256:
        raise RouteFirstRuntimeError("Stage-7 holdout result SHA-256 differs")
    route_router, metadata = load_calibrated_route_first_router(calibrated_path)
    stage7_status = _validate_stage7_evidence(
        holdout_path,
        holdout_sha256=holdout_sha,
        calibrated_router_sha256=calibrated_sha,
        router_metadata=metadata,
    )
    try:
        context_runtime = load_frozen_phase_route_runtime(
            context_router_path,
            phase_checkpoint_path,
        )
    except ActiveRuntimeError as error:
        raise RouteFirstRuntimeError(str(error)) from error
    if context_runtime.artifacts is None:
        raise RouteFirstRuntimeError("frozen V3 context artifacts are missing")
    route_artifacts = RouteFirstRuntimeArtifacts(
        calibrated_router_path=str(calibrated_path),
        calibrated_router_sha256=calibrated_sha,
        stage7_holdout_path=str(holdout_path),
        stage7_holdout_sha256=holdout_sha,
        stage7_status=stage7_status,
        v3_context_artifacts=context_runtime.artifacts,
    )
    return ActiveRouteFirstRuntime(
        context_runtime.adapter.router,
        context_runtime.phase_estimator,
        route_router,
        metadata,
        route_artifacts,
        artifacts=context_runtime.artifacts,
    )


__all__ = [
    "ActiveRouteFirstRuntime",
    "ROUTE_FIRST_ACTIVE_RUNTIME_SCHEMA_VERSION",
    "ROUTE_FIRST_CALIBRATED_ROUTER_SHA256",
    "ROUTE_FIRST_RUNTIME_LAYERS",
    "ROUTE_FIRST_RUNTIME_STATUS",
    "ROUTE_FIRST_STAGE7_HOLDOUT_SHA256",
    "ROUTE_FIRST_STAGE7_HOLDOUT_STATUS",
    "RouteFirstRuntimeAdapter",
    "RouteFirstRuntimeArtifacts",
    "RouteFirstRuntimeError",
    "load_route_first_active_runtime",
    "route_first_router_state_sha256",
    "route_first_target_layers",
]
