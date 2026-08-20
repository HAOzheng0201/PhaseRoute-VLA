"""Authenticated V3-D4A adapter for frozen 82-D motion/tail heads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F


D4A_SCHEMA_VERSION = "phase-route-vla.v3.d4a-signal-adapter.v1"
D4A_CHECKPOINT_SHA256 = (
    "b4cbf36c84767e5d17cffc36790571fcbc34b06cd77252b5e1bad50dfe53d823"
)
D4A_TAIL_ARTIFACT_SHA256 = (
    "e90efd1d825cca46bc0aec1be48146aaa1a1e82fba5b8fba0075941e317e7ba5"
)
D4A_DECISION_LAYERS = (11, 13)
D4A_LEGACY_DIMENSION = 82
D4A_V3_DIMENSION = 97
D4A_MOTION_THRESHOLDS = torch.tensor(
    [
        [0.03213280713194666, 0.038649997499361995],
        [0.014665028945979288, 0.017045890401760883],
    ],
    dtype=torch.float64,
)
D4A_TAIL_ANCHORS = torch.tensor(
    [0.15234375, 0.07666015625], dtype=torch.float64
)
D4A_TAIL_CORRECTIONS = torch.tensor(
    [0.009694024920463562, 0.0036352351307868958], dtype=torch.float32
)
D4A_TAIL_BUDGETS = (
    D4A_TAIL_ANCHORS.float() + D4A_TAIL_CORRECTIONS
).contiguous()


class D4ASignalError(ValueError):
    """Raised on D4A artifact, geometry, or numerical drift."""


def stream_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticated_weights_only_load(
    path: str | Path, *, expected_sha256: str, context: str
) -> Mapping[str, Any]:
    """Read one regular non-symlink file once, authenticate, then deserialize."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D4ASignalError(f"D4A {context} must be a regular file")
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise D4ASignalError(f"D4A {context} SHA-256 differs")
    try:
        value = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise D4ASignalError(f"D4A {context} weights-only load failed") from error
    if not isinstance(value, Mapping):
        raise D4ASignalError(f"D4A {context} must contain a mapping")
    return value


def _tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or (dtype.is_floating_point and not bool(torch.isfinite(value).all()))
    ):
        raise D4ASignalError(
            f"D4A {name} must be contiguous finite CPU {dtype} {shape}"
        )
    return value.detach().clone().contiguous()


@dataclass(frozen=True)
class FrozenLegacySignalState:
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    motion_anchor: torch.Tensor
    tail_anchor: torch.Tensor
    motion_weight: torch.Tensor
    motion_correction: torch.Tensor
    tail_weight: torch.Tensor
    tail_model_correction: torch.Tensor
    tail_conformal_correction: torch.Tensor

    def validate(self) -> None:
        mean = _tensor(
            self.feature_mean,
            name="feature mean",
            shape=(2, 82),
            dtype=torch.float64,
        )
        scale = _tensor(
            self.feature_scale,
            name="feature scale",
            shape=(2, 82),
            dtype=torch.float64,
        )
        motion_anchor = _tensor(
            self.motion_anchor,
            name="motion anchor",
            shape=(2, 2),
            dtype=torch.float64,
        )
        tail_anchor = _tensor(
            self.tail_anchor,
            name="tail anchor",
            shape=(2, 1),
            dtype=torch.float64,
        )
        _tensor(
            self.motion_weight,
            name="motion weight",
            shape=(2, 82),
            dtype=torch.float64,
        )
        _tensor(
            self.motion_correction,
            name="motion correction",
            shape=(2, 2),
            dtype=torch.float64,
        )
        _tensor(
            self.tail_weight,
            name="tail weight",
            shape=(1, 82),
            dtype=torch.float64,
        )
        _tensor(
            self.tail_model_correction,
            name="tail model correction",
            shape=(2, 1),
            dtype=torch.float64,
        )
        conformal = _tensor(
            self.tail_conformal_correction,
            name="tail conformal correction",
            shape=(2,),
            dtype=torch.float32,
        )
        if (
            bool((scale <= 0.0).any())
            or bool((motion_anchor <= 0.0).any())
            or bool((tail_anchor <= 0.0).any())
            or bool((conformal < 0.0).any())
            or not torch.equal(motion_anchor, D4A_MOTION_THRESHOLDS)
            or not torch.equal(tail_anchor[:, 0], D4A_TAIL_ANCHORS)
            or not torch.equal(conformal, D4A_TAIL_CORRECTIONS)
        ):
            raise D4ASignalError("D4A frozen threshold source tensors differ")
        del mean


def load_frozen_legacy_signal_state(
    checkpoint_path: str | Path,
    tail_artifact_path: str | Path,
) -> FrozenLegacySignalState:
    checkpoint = authenticated_weights_only_load(
        checkpoint_path,
        expected_sha256=D4A_CHECKPOINT_SHA256,
        context="legacy checkpoint",
    )
    if (
        checkpoint.get("schema_version")
        != "phase-route-vla.stage-c355-development-checkpoint-candidates.v1"
        or checkpoint.get("status") != "NON_DEPLOYABLE_DEVELOPMENT_CANDIDATES"
        or checkpoint.get("role") != "model_development"
        or checkpoint.get("context_feature_dimension") != 82
        or checkpoint.get("total_trainable_parameters") != 574
        or checkpoint.get("development_only") is not True
        or checkpoint.get("deployable") is not False
        or checkpoint.get("calibrated") is not False
        or checkpoint.get("runtime_threshold_defined") is not False
        or checkpoint.get("runtime_control_authorized") is not False
        or checkpoint.get("active_action_control") is not False
        or checkpoint.get("family_dispositions")
        != {
            "motion": "PASS_NON_DEPLOYABLE_CANDIDATE",
            "gripper": "BASELINE_OR_FAIL_NEGATIVE_RESULT_FROZEN",
            "tail": "PASS_NON_DEPLOYABLE_CANDIDATE",
        }
    ):
        raise D4ASignalError("D4A legacy checkpoint boundary differs")
    preprocessing = checkpoint.get("preprocessing_and_anchors")
    subheads = checkpoint.get("subheads")
    if not isinstance(preprocessing, Mapping) or not isinstance(subheads, Mapping):
        raise D4ASignalError("D4A legacy checkpoint fields differ")
    motion = subheads.get("motion")
    tail = subheads.get("tail")
    if (
        not isinstance(motion, Mapping)
        or not isinstance(tail, Mapping)
        or motion.get("family") != "motion"
        or tail.get("family") != "tail"
        or motion.get("regularization_lambda") != 0.1
        or tail.get("regularization_lambda") != 0.1
    ):
        raise D4ASignalError("D4A legacy subhead identity differs")
    artifact = authenticated_weights_only_load(
        tail_artifact_path,
        expected_sha256=D4A_TAIL_ARTIFACT_SHA256,
        context="legacy tail calibration artifact",
    )
    if (
        artifact.get("schema_version")
        != "phase-route-vla.stage-c357-tail-calibration-artifact.v1"
        or artifact.get("source_role") != "calibration"
        or artifact.get("finite_sample_rank") != 939
        or artifact.get("conformal_fit_calls") != 1
        or artifact.get("model_parameter_refit") is not False
        or artifact.get("checkpoint_deployable") is not False
        or artifact.get("runtime_threshold_defined") is not False
        or artifact.get("active_action_control") is not False
        or artifact.get("motion_fit_or_selection") is not False
        or artifact.get("gripper_target_inference_or_calibration") is not False
    ):
        raise D4ASignalError("D4A tail calibration boundary differs")
    state = FrozenLegacySignalState(
        feature_mean=_tensor(
            preprocessing.get("feature_mean_by_layer"),
            name="checkpoint feature mean",
            shape=(2, 82),
            dtype=torch.float64,
        ),
        feature_scale=_tensor(
            preprocessing.get("feature_scale_by_layer"),
            name="checkpoint feature scale",
            shape=(2, 82),
            dtype=torch.float64,
        ),
        motion_anchor=_tensor(
            preprocessing.get("motion_anchor"),
            name="checkpoint motion anchor",
            shape=(2, 2),
            dtype=torch.float64,
        ),
        tail_anchor=_tensor(
            preprocessing.get("tail_q90_anchor"),
            name="checkpoint tail anchor",
            shape=(2, 1),
            dtype=torch.float64,
        ),
        motion_weight=_tensor(
            motion.get("weight"),
            name="checkpoint motion weight",
            shape=(2, 82),
            dtype=torch.float64,
        ),
        motion_correction=_tensor(
            motion.get("correction"),
            name="checkpoint motion correction",
            shape=(2, 2),
            dtype=torch.float64,
        ),
        tail_weight=_tensor(
            tail.get("weight"),
            name="checkpoint tail weight",
            shape=(1, 82),
            dtype=torch.float64,
        ),
        tail_model_correction=_tensor(
            tail.get("correction"),
            name="checkpoint tail model correction",
            shape=(2, 1),
            dtype=torch.float64,
        ),
        tail_conformal_correction=_tensor(
            artifact.get("layer_correction"),
            name="artifact tail correction",
            shape=(2,),
            dtype=torch.float32,
        ),
    )
    state.validate()
    return state


@dataclass(frozen=True)
class AdaptedShadowSignals:
    motion_prediction: torch.Tensor
    tail_q90: torch.Tensor
    tail_upper: torch.Tensor
    motion_safe: torch.Tensor
    tail_ucb_safe: torch.Tensor

    def validate(self, *, rows: int) -> None:
        motion = _tensor(
            self.motion_prediction,
            name="motion prediction",
            shape=(rows, 2),
            dtype=torch.float64,
        )
        q90 = _tensor(
            self.tail_q90,
            name="tail q90",
            shape=(rows,),
            dtype=torch.float64,
        )
        upper = _tensor(
            self.tail_upper,
            name="tail upper",
            shape=(rows,),
            dtype=torch.float32,
        )
        for value, name in (
            (self.motion_safe, "motion safe"),
            (self.tail_ucb_safe, "tail UCB safe"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.bool
                or tuple(value.shape) != (rows,)
                or not value.is_contiguous()
            ):
                raise D4ASignalError(f"D4A {name} must be bool [{rows}]")
        if bool((motion <= 0.0).any()) or bool((q90 <= 0.0).any()) or bool(
            (upper <= 0.0).any()
        ):
            raise D4ASignalError("D4A adapted risk predictions must be positive")


def adapt_shadow_signals(
    state: FrozenLegacySignalState,
    features97: torch.Tensor,
    candidate_layer: torch.Tensor,
) -> AdaptedShadowSignals:
    """Apply frozen heads to the exact legacy prefix without fitting/search."""

    state.validate()
    if (
        not isinstance(features97, torch.Tensor)
        or features97.device.type != "cpu"
        or features97.dtype != torch.float32
        or features97.ndim != 2
        or features97.shape[1] != D4A_V3_DIMENSION
        or not features97.is_contiguous()
        or not bool(torch.isfinite(features97).all())
    ):
        raise D4ASignalError("D4A V3 features must be finite FP32 [N,97]")
    rows = int(features97.shape[0])
    if (
        not isinstance(candidate_layer, torch.Tensor)
        or candidate_layer.device.type != "cpu"
        or candidate_layer.dtype != torch.long
        or tuple(candidate_layer.shape) != (rows,)
        or not candidate_layer.is_contiguous()
        or not bool(((candidate_layer == 11) | (candidate_layer == 13)).all())
    ):
        raise D4ASignalError("D4A candidate layer must be int64 [N] in {11,13}")
    layer_index = (candidate_layer == 13).long()
    legacy = features97[:, :D4A_LEGACY_DIMENSION].double().contiguous()
    standardized = (
        legacy - state.feature_mean[layer_index]
    ) / state.feature_scale[layer_index]
    motion_residual = standardized @ state.motion_weight.t()
    motion = state.motion_anchor[layer_index] * torch.exp(
        motion_residual - state.motion_correction[layer_index]
    )
    tail_residual = standardized @ state.tail_weight.t()
    tail_q90 = (
        state.tail_anchor[layer_index]
        * torch.exp(tail_residual - state.tail_model_correction[layer_index])
    )[:, 0]
    tail_upper = (
        tail_q90.float() + state.tail_conformal_correction[layer_index]
    ).contiguous()
    motion_safe = (
        motion <= D4A_MOTION_THRESHOLDS[layer_index]
    ).all(dim=1).contiguous()
    tail_safe = (
        tail_upper <= D4A_TAIL_BUDGETS[layer_index]
    ).contiguous()
    result = AdaptedShadowSignals(
        motion_prediction=motion.contiguous(),
        tail_q90=tail_q90.contiguous(),
        tail_upper=tail_upper,
        motion_safe=motion_safe,
        tail_ucb_safe=tail_safe,
    )
    result.validate(rows=rows)
    return result


def validate_v3_dataset_header(dataset: Mapping[str, Any]) -> int:
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("schema_version")
        != "phase-route-vla.v3.d3-gripper-dataset.v1"
        or dataset.get("role") != "calibration_v2"
        or dataset.get("suite") != "libero_10"
        or dataset.get("feature_dimension") != 97
        or dataset.get("feature_layout")
        != {
            "legacy_causal_context": [0, 82],
            "current_candidate_gripper_sign_sequence": [82, 90],
            "current_candidate_gripper_transition_pattern": [90, 97],
        }
        or dataset.get("teacher_or_layer27_runtime_visible") is not False
        or dataset.get("other_candidate_runtime_visible") is not False
        or dataset.get("task_episode_identity_is_runtime_input") is not False
        or dataset.get("independent_test_payload_opened") is not False
    ):
        raise D4ASignalError("D4A V3 dataset header differs")
    features = dataset.get("features")
    layers = dataset.get("candidate_layer")
    if not isinstance(features, torch.Tensor) or not isinstance(
        layers, torch.Tensor
    ):
        raise D4ASignalError("D4A V3 dataset tensors are missing")
    rows = int(features.shape[0]) if features.ndim == 2 else -1
    if rows != 7032:
        raise D4ASignalError("D4A V3 dataset row count differs")
    return rows


def mean_action_cosine_distance(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Reproduce A1's FP32 7-D cosine distance then mean over horizon."""

    if (
        not isinstance(first, torch.Tensor)
        or not isinstance(second, torch.Tensor)
        or first.device.type != "cpu"
        or second.device.type != "cpu"
        or first.dtype != torch.float32
        or second.dtype != torch.float32
        or first.shape != second.shape
        or first.ndim != 3
        or first.shape[1:] != (8, 7)
        or not first.is_contiguous()
        or not second.is_contiguous()
        or not bool(torch.isfinite(first).all())
        or not bool(torch.isfinite(second).all())
    ):
        raise D4ASignalError(
            "D4 action cosine inputs must be matching finite FP32 [N,8,7]"
        )
    left = F.normalize(first, p=2.0, dim=-1, eps=1.0e-5)
    right = F.normalize(second, p=2.0, dim=-1, eps=1.0e-5)
    distance = (1.0 - (left * right).sum(dim=-1)).mean(dim=-1)
    if not bool(torch.isfinite(distance).all()):
        raise D4ASignalError("D4 action cosine distance is non-finite")
    return distance.contiguous()


__all__ = [
    "AdaptedShadowSignals",
    "D4A_CHECKPOINT_SHA256",
    "D4A_DECISION_LAYERS",
    "D4A_LEGACY_DIMENSION",
    "D4A_MOTION_THRESHOLDS",
    "D4A_SCHEMA_VERSION",
    "D4A_TAIL_ANCHORS",
    "D4A_TAIL_ARTIFACT_SHA256",
    "D4A_TAIL_BUDGETS",
    "D4A_TAIL_CORRECTIONS",
    "D4A_V3_DIMENSION",
    "D4ASignalError",
    "FrozenLegacySignalState",
    "adapt_shadow_signals",
    "authenticated_weights_only_load",
    "load_frozen_legacy_signal_state",
    "mean_action_cosine_distance",
    "stream_sha256",
    "validate_v3_dataset_header",
]
