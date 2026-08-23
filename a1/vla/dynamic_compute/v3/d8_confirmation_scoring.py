"""Pure CPU scoring primitives for the one-shot V3-D8D confirmation gate.

This module never loads artifacts, fits a model, selects a threshold, writes a
result, or controls an environment.  It only validates an already-frozen D8C
tensor mapping and aggregates predictions produced by the already-frozen D8B
five-head router.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from .epistemic_ensemble import D7_HEAD_COUNT, D7_MIN_HEAD_RANGE, ensemble_scores
from .fresh_confirmation import (
    D8_CLUSTERS_PER_TASK,
    D8_REPLICATE_IDS,
    D8_TASK_IDS,
    D8ConfirmationSummary,
)
from .gripper_v2_protocol import FEATURE_DIMENSION
from .joint_reliability import D5_CANDIDATE_LAYERS, D5_FALLBACK_LAYER
from .shadow_decision import D4_RP_PEP_FM_CALLS


D8D_PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d8d-confirmation-scoring.v1"
D8D_RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d8d-confirmation-result.v1"


class D8DScoringError(ValueError):
    """Raised when D8C inputs or D8B predictions fail closed."""


def _cpu_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    floating: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.shape != shape
        or (dtype is not None and value.dtype != dtype)
        or (floating and not value.is_floating_point())
    ):
        raise D8DScoringError(f"D8D {name} tensor geometry differs")
    result = value.detach().cpu().contiguous()
    if floating and not bool(torch.isfinite(result).all()):
        raise D8DScoringError(f"D8D {name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class D8ConfirmationData:
    features: torch.Tensor
    candidate_layer: torch.Tensor
    source_row: torch.Tensor
    task_id: torch.Tensor
    replicate_id: torch.Tensor
    cluster_keys: tuple[str, ...]
    call_ordinal: torch.Tensor
    step_id: torch.Tensor
    action_consistency: torch.Tensor
    unsafe_target: torch.Tensor
    full_action_distance: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def calls(self) -> int:
        return self.rows // 2

    def validate(self, *, expected_policy_calls: int | None = None) -> None:
        rows = self.rows
        if rows <= 0 or rows % 2:
            raise D8DScoringError("D8D candidate rows must be nonempty pairs")
        calls = rows // 2
        if expected_policy_calls is not None and calls != expected_policy_calls:
            raise D8DScoringError("D8D policy-call count differs from D8C")
        _cpu_tensor(
            self.features,
            name="features",
            shape=(rows, FEATURE_DIMENSION),
            floating=True,
        )
        for name, value in (
            ("candidate_layer", self.candidate_layer),
            ("source_row", self.source_row),
            ("task_id", self.task_id),
            ("replicate_id", self.replicate_id),
            ("call_ordinal", self.call_ordinal),
            ("step_id", self.step_id),
        ):
            _cpu_tensor(value, name=name, shape=(rows,), dtype=torch.long)
        _cpu_tensor(
            self.action_consistency,
            name="action_consistency",
            shape=(rows,),
            dtype=torch.bool,
        )
        _cpu_tensor(
            self.unsafe_target,
            name="unsafe_target",
            shape=(rows, 2),
            dtype=torch.bool,
        )
        _cpu_tensor(
            self.full_action_distance,
            name="full_action_distance",
            shape=(rows,),
            floating=True,
        )
        if len(self.cluster_keys) != rows or any(
            type(value) is not str for value in self.cluster_keys
        ):
            raise D8DScoringError("D8D cluster-key geometry differs")
        paired_layer = self.candidate_layer.reshape(calls, 2)
        if not torch.equal(
            paired_layer,
            torch.tensor(D5_CANDIDATE_LAYERS, dtype=torch.long).repeat(calls, 1),
        ):
            raise D8DScoringError("D8D rows are not paired L11 then L13")
        if not torch.equal(
            self.source_row,
            torch.arange(calls, dtype=torch.long).repeat_interleave(2),
        ):
            raise D8DScoringError("D8D source-row assignment differs")
        for name, value in (
            ("task", self.task_id),
            ("replicate", self.replicate_id),
            ("call ordinal", self.call_ordinal),
            ("step", self.step_id),
        ):
            if not torch.equal(value[0::2], value[1::2]):
                raise D8DScoringError(f"D8D paired {name} identity differs")
        if self.cluster_keys[0::2] != self.cluster_keys[1::2]:
            raise D8DScoringError("D8D paired cluster identity differs")

        observed_clusters: set[str] = set()
        per_task: dict[int, set[str]] = {task: set() for task in D8_TASK_IDS}
        for row in range(0, rows, 2):
            task = int(self.task_id[row])
            replicate = int(self.replicate_id[row])
            if task not in D8_TASK_IDS or replicate not in D8_REPLICATE_IDS:
                raise D8DScoringError("D8D task/replicate lies outside schedule")
            expected = (
                f"libero_10:task{task}:fresh_confirm_v1:replicate{replicate}"
            )
            if self.cluster_keys[row] != expected:
                raise D8DScoringError("D8D cluster key differs from row identity")
            observed_clusters.add(expected)
            per_task[task].add(expected)
        expected_clusters = {
            f"libero_10:task{task}:fresh_confirm_v1:replicate{replicate}"
            for task in D8_TASK_IDS
            for replicate in D8_REPLICATE_IDS
        }
        if observed_clusters != expected_clusters or any(
            len(per_task[task]) != D8_CLUSTERS_PER_TASK for task in D8_TASK_IDS
        ):
            raise D8DScoringError("D8D fresh cluster coverage differs")


def confirmation_data_from_mapping(
    payload: Mapping[str, Any], *, expected_policy_calls: int | None = None
) -> D8ConfirmationData:
    required = (
        "features",
        "candidate_layer",
        "source_row",
        "task_id",
        "replicate_id",
        "cluster_keys",
        "call_ordinal",
        "step_id",
        "action_consistency",
        "unsafe_target",
        "full_action_distance",
    )
    if any(name not in payload for name in required):
        raise D8DScoringError("D8D dataset is missing a required field")
    if not isinstance(payload["cluster_keys"], (list, tuple)):
        raise D8DScoringError("D8D cluster keys must be a sequence")
    tensor_names = tuple(name for name in required if name != "cluster_keys")
    if any(not isinstance(payload[name], torch.Tensor) for name in tensor_names):
        raise D8DScoringError("D8D dataset tensor field has invalid type")
    data = D8ConfirmationData(
        features=payload["features"].detach().cpu().contiguous(),
        candidate_layer=payload["candidate_layer"].detach().cpu().contiguous(),
        source_row=payload["source_row"].detach().cpu().contiguous(),
        task_id=payload["task_id"].detach().cpu().contiguous(),
        replicate_id=payload["replicate_id"].detach().cpu().contiguous(),
        cluster_keys=tuple(payload["cluster_keys"]),
        call_ordinal=payload["call_ordinal"].detach().cpu().contiguous(),
        step_id=payload["step_id"].detach().cpu().contiguous(),
        action_consistency=payload["action_consistency"].detach().cpu().contiguous(),
        unsafe_target=payload["unsafe_target"].detach().cpu().contiguous(),
        full_action_distance=payload["full_action_distance"].detach()
        .cpu()
        .contiguous(),
    )
    data.validate(expected_policy_calls=expected_policy_calls)
    return data


@dataclass(frozen=True)
class D8ScoringResult:
    head_prediction: torch.Tensor
    combined_score: torch.Tensor
    full_head_range: torch.Tensor
    candidate_safe: torch.Tensor
    selected_layer: torch.Tensor
    selected_candidate_index: torch.Tensor
    selected_full_action_unsafe: torch.Tensor
    selected_gripper_unsafe: torch.Tensor
    selected_full_action_distance: torch.Tensor
    selected_unsafe: torch.Tensor
    severe_false_full_action: torch.Tensor
    safe_cluster_keys: tuple[str, ...]
    false_safe_cluster_keys: tuple[str, ...]
    false_full_action_cluster_keys: tuple[str, ...]
    severe_false_full_action_cluster_keys: tuple[str, ...]
    summary: D8ConfirmationSummary


def _sorted_cluster_keys(
    values: set[str], identities: Mapping[str, tuple[int, int]]
) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda key: identities[key]))


def score_frozen_router_predictions(
    data: D8ConfirmationData,
    head_prediction: torch.Tensor,
    *,
    runtime_threshold: float,
    gripper_threshold: float,
    action_consistency_threshold: float,
    behavior_fm_calls: int,
) -> D8ScoringResult:
    """Apply the frozen route and aggregate summary without fitting/search."""

    data.validate()
    rows = data.rows
    calls = data.calls
    if (
        isinstance(runtime_threshold, bool)
        or not isinstance(runtime_threshold, (int, float))
        or not math.isfinite(float(runtime_threshold))
        or float(runtime_threshold) <= 0.0
        or isinstance(gripper_threshold, bool)
        or not isinstance(gripper_threshold, (int, float))
        or not math.isfinite(float(gripper_threshold))
        or float(gripper_threshold) <= 0.0
        or isinstance(action_consistency_threshold, bool)
        or not isinstance(action_consistency_threshold, (int, float))
        or not math.isfinite(float(action_consistency_threshold))
        or float(action_consistency_threshold) <= 0.0
        or type(behavior_fm_calls) is not int
        or behavior_fm_calls <= 0
    ):
        raise D8DScoringError("D8D frozen scalar metadata differs")
    prediction = _cpu_tensor(
        head_prediction,
        name="head_prediction",
        shape=(D7_HEAD_COUNT, rows, 2),
        floating=True,
    ).double()
    full_score, gripper_score, head_range = ensemble_scores(prediction)
    combined = torch.stack((full_score, gripper_score), dim=1).contiguous()
    safe = (
        data.action_consistency
        & (combined[:, 0] <= float(runtime_threshold))
        & (combined[:, 1] <= float(gripper_threshold))
    ).contiguous()
    paired_safe = safe.reshape(calls, 2)
    selected_layer = torch.full((calls,), D5_FALLBACK_LAYER, dtype=torch.long)
    selected_layer[paired_safe[:, 1]] = D5_CANDIDATE_LAYERS[1]
    selected_layer[paired_safe[:, 0]] = D5_CANDIDATE_LAYERS[0]
    early = selected_layer != D5_FALLBACK_LAYER
    selected_index = torch.full((calls,), -1, dtype=torch.long)
    selected_index[selected_layer == D5_CANDIDATE_LAYERS[0]] = 0
    selected_index[selected_layer == D5_CANDIDATE_LAYERS[1]] = 1

    target = data.unsafe_target.reshape(calls, 2, 2)
    distance = data.full_action_distance.reshape(calls, 2).double()
    call_rows = torch.arange(calls)
    full_unsafe = torch.zeros(calls, dtype=torch.bool)
    gripper_unsafe = torch.zeros(calls, dtype=torch.bool)
    selected_distance = torch.zeros(calls, dtype=torch.float64)
    full_unsafe[early] = target[call_rows[early], selected_index[early], 0]
    gripper_unsafe[early] = target[call_rows[early], selected_index[early], 1]
    selected_distance[early] = distance[call_rows[early], selected_index[early]]
    selected_unsafe = (full_unsafe | gripper_unsafe).contiguous()
    severe = (
        early
        & full_unsafe
        & (selected_distance > 4.0 * float(action_consistency_threshold))
    ).contiguous()

    call_task = data.task_id[0::2]
    call_replicate = data.replicate_id[0::2]
    call_keys = data.cluster_keys[0::2]
    identities = {
        key: (int(call_task[index]), int(call_replicate[index]))
        for index, key in enumerate(call_keys)
    }
    safe_clusters = {
        call_keys[index] for index in torch.nonzero(early).flatten().tolist()
    }
    false_clusters = {
        call_keys[index]
        for index in torch.nonzero(early & selected_unsafe).flatten().tolist()
    }
    false_full_clusters = {
        call_keys[index]
        for index in torch.nonzero(early & full_unsafe).flatten().tolist()
    }
    severe_clusters = {
        call_keys[index] for index in torch.nonzero(severe).flatten().tolist()
    }
    all_clusters = set(call_keys)
    clusters_per_task = tuple(
        len({key for key in all_clusters if identities[key][0] == task})
        for task in D8_TASK_IDS
    )
    safe_per_task = tuple(
        len({key for key in safe_clusters if identities[key][0] == task})
        for task in D8_TASK_IDS
    )
    early_per_task = tuple(
        int((early & (call_task == task)).sum()) for task in D8_TASK_IDS
    )
    selection_counts = {
        layer: int((selected_layer == layer).sum())
        for layer in (*D5_CANDIDATE_LAYERS, D5_FALLBACK_LAYER)
    }
    estimated_fm_calls = sum(
        selection_counts[layer] * D4_RP_PEP_FM_CALLS[layer]
        for layer in selection_counts
    )
    summary = D8ConfirmationSummary(
        total_clusters=len(all_clusters),
        clusters_per_task=clusters_per_task,
        safe_clusters=len(safe_clusters),
        safe_clusters_per_task=safe_per_task,
        policy_calls=calls,
        early_exit_calls=int(early.sum()),
        early_exit_calls_per_task=early_per_task,
        false_safe_clusters=len(false_clusters),
        false_full_action_clusters=len(false_full_clusters),
        false_gripper_calls=int(gripper_unsafe.sum()),
        severe_false_full_action_clusters=len(severe_clusters),
        nondegenerate_row_fraction=float(
            (head_range > D7_MIN_HEAD_RANGE).double().mean()
        ),
        estimated_fm_reduction_fraction=(
            1.0 - estimated_fm_calls / behavior_fm_calls
        ),
        all_candidate_rows_and_policy_calls_accounted_for=(
            rows == 2 * calls and int(data.source_row.max()) + 1 == calls
        ),
        all_predictions_finite=bool(torch.isfinite(prediction).all()),
    )
    return D8ScoringResult(
        head_prediction=prediction.contiguous(),
        combined_score=combined,
        full_head_range=head_range,
        candidate_safe=safe,
        selected_layer=selected_layer,
        selected_candidate_index=selected_index,
        selected_full_action_unsafe=full_unsafe,
        selected_gripper_unsafe=gripper_unsafe,
        selected_full_action_distance=selected_distance,
        selected_unsafe=selected_unsafe,
        severe_false_full_action=severe,
        safe_cluster_keys=_sorted_cluster_keys(safe_clusters, identities),
        false_safe_cluster_keys=_sorted_cluster_keys(false_clusters, identities),
        false_full_action_cluster_keys=_sorted_cluster_keys(
            false_full_clusters, identities
        ),
        severe_false_full_action_cluster_keys=_sorted_cluster_keys(
            severe_clusters, identities
        ),
        summary=summary,
    )


__all__ = [
    "D8ConfirmationData",
    "D8D_PAYLOAD_SCHEMA_VERSION",
    "D8D_RESULT_SCHEMA_VERSION",
    "D8DScoringError",
    "D8ScoringResult",
    "confirmation_data_from_mapping",
    "score_frozen_router_predictions",
]
