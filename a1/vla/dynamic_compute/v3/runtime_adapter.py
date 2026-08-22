"""Fail-closed active runtime adapter for the frozen V3-D8 router.

The adapter is deliberately CPU-only.  Candidate actions may be produced by a
GPU policy, but the 97-D route feature and the five small GLM heads are scored
on detached CPU float32/float64 tensors.  The selected action object is never
rewritten: early exits return the exact candidate supplied by the action head,
and a vetoed or malformed call returns the exact L27 fallback supplied later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import numpy as np
import torch

from a1.vla.dynamic_compute.temporal_route_features import right_aligned_history

from .development_collection import (
    D2_HISTORY_LENGTH,
    build_gripper_v2_feature,
    build_gripper_v2_features,
    validate_runtime_context,
)
from .final_router import FinalFiveHeadRouter, final_router_state
from .gripper_v2_protocol import (
    ACTION_DIMENSION,
    DECISION_LAYERS,
    HORIZON,
    RUNTIME_CONTEXT_NAMES,
    TEACHER_LAYER,
)


D9A_RUNTIME_SCHEMA_VERSION = "phase-route-vla.v3.d9a-runtime-adapter.v1"
D9A_RUNTIME_STATUS = "PASS_V3_D9A_RUNTIME_ADAPTER_AND_D8_PARITY"
ROUTE_PRIORITY = (*DECISION_LAYERS, TEACHER_LAYER)


class RuntimeAdapterError(ValueError):
    """Raised for programmer errors outside the fail-closed candidate path."""


TelemetryCallback = Callable[[str, Mapping[str, Any]], None]


def _hash_tensor(digest: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def frozen_router_sha256(router: FinalFiveHeadRouter) -> str:
    """Hash every frozen head, normalizer, and scalar threshold."""

    state = final_router_state(router)
    digest = hashlib.sha256()
    for head_index, head in enumerate(state["head_states"]):
        for name in (
            "normalizer_mean",
            "normalizer_scale",
            "anchor_score",
            "weight",
        ):
            _hash_tensor(digest, f"head{head_index}.{name}", head[name])
        digest.update(
            json.dumps(
                {
                    "head": head_index,
                    "l2_lambda": float(head["l2_lambda"]),
                    "final_loss": float(head["final_loss"]),
                },
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        )
    digest.update(
        json.dumps(
            {
                "full_threshold": float(state["full_threshold"]),
                "runtime_threshold": float(state["runtime_threshold"]),
                "gripper_threshold": float(state["gripper_threshold"]),
                "action_consistency_threshold": float(
                    state["action_consistency_threshold"]
                ),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    )
    return digest.hexdigest()


def _copy_runtime_context(
    runtime_inputs: Mapping[str, torch.Tensor], *, rows: int
) -> dict[str, torch.Tensor]:
    if not isinstance(runtime_inputs, Mapping) or tuple(runtime_inputs) != (
        RUNTIME_CONTEXT_NAMES
    ):
        raise RuntimeAdapterError("runtime context names or order differ")
    copied = {
        name: value.detach().cpu().contiguous().clone()
        if isinstance(value, torch.Tensor)
        else value
        for name, value in runtime_inputs.items()
    }
    validate_runtime_context(copied, rows=rows)
    return copied


@dataclass(frozen=True)
class CachedBatchRoute:
    """Vectorized D8-compatible route result for already-cached inputs."""

    features: torch.Tensor
    five_head_prediction: torch.Tensor
    combined_score: torch.Tensor
    full_action_head_range: torch.Tensor
    candidate_safe: torch.Tensor
    selected_layer: torch.Tensor


def route_cached_candidate_pairs(
    router: FinalFiveHeadRouter,
    runtime_inputs: Mapping[str, torch.Tensor],
    candidate_actions: torch.Tensor,
    action_consistency: torch.Tensor,
) -> CachedBatchRoute:
    """Apply the online rule in one vectorized, D8-parity-preserving batch."""

    if (
        not isinstance(candidate_actions, torch.Tensor)
        or candidate_actions.device.type != "cpu"
        or candidate_actions.dtype != torch.float32
        or candidate_actions.ndim != 4
        or tuple(candidate_actions.shape[1:]) != (2, HORIZON, ACTION_DIMENSION)
        or not bool(torch.isfinite(candidate_actions).all())
    ):
        raise RuntimeAdapterError("candidate actions must be finite CPU float32 [B,2,8,7]")
    calls = int(candidate_actions.shape[0])
    if (
        not isinstance(action_consistency, torch.Tensor)
        or action_consistency.device.type != "cpu"
        or action_consistency.dtype != torch.bool
        or tuple(action_consistency.shape) != (calls, 2)
    ):
        raise RuntimeAdapterError("action consistency must be CPU bool [B,2]")
    context = _copy_runtime_context(runtime_inputs, rows=calls)
    before = frozen_router_sha256(router)
    features = build_gripper_v2_features(context, candidate_actions)
    flat_features = features.reshape(calls * 2, -1).contiguous()
    layers = torch.tensor(DECISION_LAYERS, dtype=torch.long).repeat(calls)
    head, combined, head_range, full_upper = router.predict(flat_features, layers)
    after = frozen_router_sha256(router)
    if before != after:
        raise RuntimeAdapterError("frozen router mutated during prediction")
    if (
        not torch.equal(combined[:, 0], full_upper)
        or not torch.equal(combined[:, 1], head[0, :, 1])
    ):
        raise RuntimeAdapterError("frozen router output semantics differ")
    safe = (
        action_consistency.reshape(-1)
        & (combined[:, 0] <= router.runtime_threshold)
        & (combined[:, 1] <= router.gripper_threshold)
    ).contiguous()
    paired_safe = safe.reshape(calls, 2)
    selected = torch.full((calls,), TEACHER_LAYER, dtype=torch.long)
    selected[paired_safe[:, 1]] = DECISION_LAYERS[1]
    selected[paired_safe[:, 0]] = DECISION_LAYERS[0]
    return CachedBatchRoute(
        features=flat_features,
        five_head_prediction=head.double().contiguous(),
        combined_score=combined.double().contiguous(),
        full_action_head_range=head_range.double().contiguous(),
        candidate_safe=safe,
        selected_layer=selected,
    )


@dataclass(frozen=True)
class RuntimeHistoryWindow:
    normalized_proprio: torch.Tensor
    proprio_history: torch.Tensor
    action_history: torch.Tensor
    history_mask: torch.Tensor


class EpisodePastOnlyHistory:
    """One active episode history with explicit window-before-commit ordering."""

    def __init__(self) -> None:
        self._episode_id: str | None = None
        self._next_ordinal = 0
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=D2_HISTORY_LENGTH
        )
        self._pending: tuple[str, int, np.ndarray] | None = None

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    def reset(self) -> None:
        self._episode_id = None
        self._next_ordinal = 0
        self._history.clear()
        self._pending = None

    def window(
        self, episode_id: str, call_ordinal: int, normalized_proprio: Any
    ) -> RuntimeHistoryWindow:
        if type(episode_id) is not str or not episode_id:
            raise RuntimeAdapterError("episode identity must be a nonempty string")
        if type(call_ordinal) is not int or call_ordinal < 0:
            raise RuntimeAdapterError("call ordinal must be a non-negative integer")
        if self._pending is not None:
            raise RuntimeAdapterError("previous policy call was not committed")
        if episode_id != self._episode_id:
            self._episode_id = episode_id
            self._next_ordinal = 0
            self._history.clear()
        if call_ordinal != self._next_ordinal:
            raise RuntimeAdapterError("policy calls are not canonical within episode")
        proprio = np.asarray(normalized_proprio, dtype=np.float32)
        if proprio.shape != (8,) or not np.isfinite(proprio).all():
            raise RuntimeAdapterError("current proprio must be finite float32 [8]")
        past_proprio, past_action, mask = right_aligned_history(
            list(self._history),
            history_len=D2_HISTORY_LENGTH,
            proprio_dim=8,
            action_horizon=HORIZON,
            action_dim=ACTION_DIMENSION,
        )
        self._pending = (episode_id, call_ordinal, proprio.copy())
        return RuntimeHistoryWindow(
            normalized_proprio=torch.from_numpy(proprio.copy()).unsqueeze(0),
            proprio_history=torch.from_numpy(past_proprio).unsqueeze(0),
            action_history=torch.from_numpy(past_action).unsqueeze(0),
            history_mask=torch.from_numpy(mask).unsqueeze(0),
        )

    def commit(self, episode_id: str, call_ordinal: int, selected_action: Any) -> None:
        if self._pending is None or self._pending[:2] != (episode_id, call_ordinal):
            raise RuntimeAdapterError("history commit does not match pending call")
        if isinstance(selected_action, torch.Tensor):
            action = selected_action.detach().cpu().float().numpy()
        else:
            action = np.asarray(selected_action, dtype=np.float32)
        if action.shape == (1, HORIZON, ACTION_DIMENSION):
            action = action[0]
        if action.shape != (HORIZON, ACTION_DIMENSION) or not np.isfinite(action).all():
            raise RuntimeAdapterError("selected action must be finite [8,7]")
        _, _, proprio = self._pending
        self._history.append((proprio.copy(), action.astype(np.float32, copy=True)))
        self._next_ordinal += 1
        self._pending = None


@dataclass(frozen=True)
class RuntimeStepDecision:
    layer: int
    should_exit: bool
    selected_action: torch.Tensor | None
    route_safe: bool
    telemetry: Mapping[str, Any]


class FrozenD8RuntimeAdapter:
    """Sequential L11 -> L13 -> L27 state machine for active control."""

    def __init__(self, router: FinalFiveHeadRouter) -> None:
        router.validate()
        self.router = router
        self.router_sha256 = frozen_router_sha256(router)
        self._runtime_inputs: dict[str, torch.Tensor] | None = None
        self._next_layer = DECISION_LAYERS[0]
        self._fail_reason: str | None = None
        self._active = False
        self._records: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return self._active

    @property
    def fail_reason(self) -> str | None:
        return self._fail_reason

    def begin_policy_call(
        self, runtime_inputs: Mapping[str, torch.Tensor] | None
    ) -> None:
        self._runtime_inputs = None
        self._next_layer = DECISION_LAYERS[0]
        self._fail_reason = None
        self._records = []
        self._active = True
        try:
            if frozen_router_sha256(self.router) != self.router_sha256:
                raise RuntimeAdapterError("frozen router integrity changed")
            if runtime_inputs is None:
                raise RuntimeAdapterError("runtime context is missing")
            self._runtime_inputs = _copy_runtime_context(runtime_inputs, rows=1)
        except Exception as error:
            self._fail_reason = f"{type(error).__name__}: {error}"

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

    def consider_candidate(
        self,
        layer: int,
        candidate_action: torch.Tensor,
        action_consistency: bool,
        *,
        telemetry_callback: TelemetryCallback | None = None,
    ) -> RuntimeStepDecision:
        if not self._active:
            raise RuntimeAdapterError("no active PhaseRoute policy call")
        if layer not in DECISION_LAYERS:
            raise RuntimeAdapterError("candidate layer must be L11 or L13")
        if layer != self._next_layer and self._fail_reason is None:
            self._fail_reason = "candidate layers arrived out of order"
        full_score = None
        gripper_score = None
        head_range = None
        route_safe = False
        error_text = self._fail_reason
        if error_text is None:
            try:
                if type(action_consistency) is not bool:
                    raise RuntimeAdapterError("action consistency must be bool")
                if (
                    not isinstance(candidate_action, torch.Tensor)
                    or candidate_action.dtype != torch.float32
                    or candidate_action.ndim != 3
                    or tuple(candidate_action.shape) != (1, HORIZON, ACTION_DIMENSION)
                    or not bool(torch.isfinite(candidate_action).all())
                ):
                    raise RuntimeAdapterError(
                        "candidate action must be finite float32 [1,8,7]"
                    )
                assert self._runtime_inputs is not None
                action_cpu = candidate_action.detach().cpu().contiguous()
                features = build_gripper_v2_feature(
                    self._runtime_inputs, action_cpu
                )
                layers = torch.tensor([layer], dtype=torch.long)
                head, combined, head_range_tensor, full_upper = self.router.predict(
                    features, layers
                )
                if frozen_router_sha256(self.router) != self.router_sha256:
                    raise RuntimeAdapterError("frozen router mutated during prediction")
                if not torch.equal(combined[:, 0], full_upper):
                    raise RuntimeAdapterError("router full score semantics differ")
                full_score = float(full_upper[0])
                gripper_score = float(head[0, 0, 1])
                head_range = float(head_range_tensor[0])
                route_safe = bool(
                    action_consistency
                    and full_score <= self.router.runtime_threshold
                    and gripper_score <= self.router.gripper_threshold
                )
            except Exception as error:
                self._fail_reason = f"{type(error).__name__}: {error}"
                error_text = self._fail_reason
                route_safe = False
        veto_reasons = []
        if type(action_consistency) is not bool:
            veto_reasons.append("invalid_action_consistency")
        elif not action_consistency:
            veto_reasons.append("failed_action_consistency")
        if full_score is None:
            veto_reasons.append("missing_full_score")
        elif not math.isfinite(full_score) or full_score > self.router.runtime_threshold:
            veto_reasons.append("failed_full_score")
        if gripper_score is None:
            veto_reasons.append("missing_gripper_score")
        elif (
            not math.isfinite(gripper_score)
            or gripper_score > self.router.gripper_threshold
        ):
            veto_reasons.append("failed_gripper_score")
        if self._fail_reason is not None:
            veto_reasons.append("fail_closed")
        record = {
            "layer_idx": layer,
            "action_consistency": (
                action_consistency if type(action_consistency) is bool else None
            ),
            "full_score": full_score,
            "gripper_score": gripper_score,
            "full_action_head_range": head_range,
            "runtime_threshold": self.router.runtime_threshold,
            "gripper_threshold": self.router.gripper_threshold,
            "route_safe": route_safe,
            "should_exit": route_safe,
            "veto_reasons": veto_reasons,
            "fail_reason": error_text or self._fail_reason,
            "router_sha256": self.router_sha256,
        }
        self._records.append(record)
        self._emit(telemetry_callback, "phase_route_candidate", record)
        if route_safe:
            self._active = False
            final = {
                "selected_layer": layer,
                "candidate_gates": list(self._records),
                "fallback": False,
            }
            self._emit(telemetry_callback, "phase_route_decision", final)
            return RuntimeStepDecision(layer, True, candidate_action, True, record)
        self._next_layer = (
            DECISION_LAYERS[1] if layer == DECISION_LAYERS[0] else TEACHER_LAYER
        )
        return RuntimeStepDecision(layer, False, None, False, record)

    def select_fallback(
        self,
        fallback_action: torch.Tensor,
        *,
        telemetry_callback: TelemetryCallback | None = None,
    ) -> RuntimeStepDecision:
        if not self._active:
            raise RuntimeAdapterError("no active PhaseRoute policy call")
        if (
            not isinstance(fallback_action, torch.Tensor)
            or fallback_action.ndim != 3
            or tuple(fallback_action.shape) != (1, HORIZON, ACTION_DIMENSION)
            or not fallback_action.is_floating_point()
            or not bool(torch.isfinite(fallback_action).all())
        ):
            raise RuntimeAdapterError("L27 fallback action must be finite [1,8,7]")
        self._active = False
        record = {
            "layer_idx": TEACHER_LAYER,
            "route_safe": True,
            "should_exit": True,
            "fallback": True,
            "fail_reason": self._fail_reason,
            "router_sha256": self.router_sha256,
        }
        final = {
            "selected_layer": TEACHER_LAYER,
            "candidate_gates": list(self._records),
            "fallback": True,
            "fail_reason": self._fail_reason,
        }
        self._emit(telemetry_callback, "phase_route_decision", final)
        return RuntimeStepDecision(
            TEACHER_LAYER, True, fallback_action, True, record
        )


__all__ = [
    "CachedBatchRoute",
    "D9A_RUNTIME_SCHEMA_VERSION",
    "D9A_RUNTIME_STATUS",
    "EpisodePastOnlyHistory",
    "FrozenD8RuntimeAdapter",
    "ROUTE_PRIORITY",
    "RuntimeAdapterError",
    "RuntimeHistoryWindow",
    "RuntimeStepDecision",
    "frozen_router_sha256",
    "route_cached_candidate_pairs",
]
