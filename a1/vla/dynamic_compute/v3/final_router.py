"""Serializable final five-head router fixed before V3-D8 confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .epistemic_ensemble import D7_HEAD_COUNT
from .epistemic_ensemble_oof import predict_head_ensemble
from .gripper_v2_models import FeatureNormalizer
from .gripper_v2_protocol import FEATURE_DIMENSION
from .joint_reliability import D5_ACTION_THRESHOLD, D5_GRIPPER_THRESHOLD
from .severity_reliability import D6_SAFETY_MULTIPLIER, SeverityWeightedFit
from .severity_reliability_oof import severity_fit_state


D8B_PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d8b-final-router-payload.v1"
D8B_RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d8b-final-router-result.v1"
D8B_L2_LAMBDA = 0.01


class FinalRouterError(ValueError):
    """Raised when the final D8B router payload is malformed."""


def _tensor(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
    positive: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.shape != shape
        or not value.is_floating_point()
    ):
        raise FinalRouterError(f"{name} tensor geometry differs")
    result = value.detach().double().contiguous()
    if not bool(torch.isfinite(result).all()) or (
        positive and not bool((result > 0.0).all())
    ):
        raise FinalRouterError(f"{name} tensor values differ")
    return result


def severity_fit_from_state(value: Mapping[str, Any]) -> SeverityWeightedFit:
    required = {
        "normalizer_mean",
        "normalizer_scale",
        "anchor_score",
        "weight",
        "l2_lambda",
        "final_loss",
    }
    if set(value) != required:
        raise FinalRouterError("D8B head state fields differ")
    mean = _tensor(value["normalizer_mean"], (FEATURE_DIMENSION,), name="mean")
    scale = _tensor(
        value["normalizer_scale"],
        (FEATURE_DIMENSION,),
        name="scale",
        positive=True,
    )
    anchor = _tensor(value["anchor_score"], (2, 2), name="anchor")
    weight = _tensor(value["weight"], (2, FEATURE_DIMENSION), name="weight")
    if not bool(((anchor > 0.0) & (anchor < 1.0)).all()):
        raise FinalRouterError("D8B head anchor lies outside (0,1)")
    l2_lambda = float(value["l2_lambda"])
    final_loss = float(value["final_loss"])
    if l2_lambda != D8B_L2_LAMBDA or not math.isfinite(final_loss):
        raise FinalRouterError("D8B head scalar metadata differs")
    return SeverityWeightedFit(
        normalizer=FeatureNormalizer(mean=mean, scale=scale),
        anchor_score=anchor,
        weight=weight,
        l2_lambda=l2_lambda,
        final_loss=final_loss,
    )


@dataclass(frozen=True)
class FinalFiveHeadRouter:
    models: tuple[SeverityWeightedFit, ...]
    full_threshold: float
    runtime_threshold: float
    gripper_threshold: float = D5_GRIPPER_THRESHOLD
    action_consistency_threshold: float = D5_ACTION_THRESHOLD

    def validate(self) -> None:
        if (
            len(self.models) != D7_HEAD_COUNT
            or any(model.l2_lambda != D8B_L2_LAMBDA for model in self.models)
            or not math.isfinite(self.full_threshold)
            or not math.isfinite(self.runtime_threshold)
            or self.full_threshold <= 0.0
            or self.runtime_threshold != D6_SAFETY_MULTIPLIER * self.full_threshold
            or self.gripper_threshold != D5_GRIPPER_THRESHOLD
            or self.action_consistency_threshold != D5_ACTION_THRESHOLD
        ):
            raise FinalRouterError("D8B final router semantics differ")

    def predict(
        self, features: torch.Tensor, candidate_layer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate()
        return predict_head_ensemble(self.models, features, candidate_layer)


def final_router_state(router: FinalFiveHeadRouter) -> dict[str, Any]:
    router.validate()
    return {
        "head_states": [severity_fit_state(model) for model in router.models],
        "full_threshold": router.full_threshold,
        "runtime_threshold": router.runtime_threshold,
        "gripper_threshold": router.gripper_threshold,
        "action_consistency_threshold": router.action_consistency_threshold,
    }


def final_router_from_mapping(value: Mapping[str, Any]) -> FinalFiveHeadRouter:
    states = value.get("head_states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise FinalRouterError("D8B head states must be a sequence")
    model_list = []
    for state in states:
        if not isinstance(state, Mapping):
            raise FinalRouterError("D8B head state must be a mapping")
        model_list.append(severity_fit_from_state(state))
    models = tuple(model_list)
    router = FinalFiveHeadRouter(
        models=models,
        full_threshold=float(value.get("full_threshold", float("nan"))),
        runtime_threshold=float(value.get("runtime_threshold", float("nan"))),
        gripper_threshold=float(value.get("gripper_threshold", float("nan"))),
        action_consistency_threshold=float(
            value.get("action_consistency_threshold", float("nan"))
        ),
    )
    router.validate()
    return router


__all__ = [
    "D8B_L2_LAMBDA",
    "D8B_PAYLOAD_SCHEMA_VERSION",
    "D8B_RESULT_SCHEMA_VERSION",
    "FinalFiveHeadRouter",
    "FinalRouterError",
    "final_router_from_mapping",
    "final_router_state",
    "severity_fit_from_state",
]
