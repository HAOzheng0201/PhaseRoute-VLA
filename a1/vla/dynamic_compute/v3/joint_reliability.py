"""Leakage-safe primitives for the frozen V3-D5 joint reliability study.

This module is deliberately filesystem-free except for the authenticated
contract loader.  It does not load rollout artifacts or execute robot actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .gripper_v2_calibration import clopper_pearson_upper
from .gripper_v2_protocol import FEATURE_DIMENSION


D5_SCHEMA_VERSION = "phase-route-vla.v3.d5-development-contract.v1"
D5_STATUS = "D5_DEVELOPMENT_CONTRACT_FROZEN"
D5_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/joint_reliability/d5_development_contract.json"
)
D5_CONTRACT_SHA256 = (
    "e0a584e76f03d0f1b43cd5bbd3477ee2e3694f5425642868b3ec563edd52a29f"
)
D5_TASK_IDS = tuple(range(10))
D5_EPISODES = tuple(range(12, 30))
D5_CANDIDATE_LAYERS = (11, 13)
D5_FALLBACK_LAYER = 27
D5_ACTION_THRESHOLD = 0.00390625
D5_GRIPPER_THRESHOLD = 0.043773197319646726
D5_MIN_SAFE_CLUSTERS = 60
D5_MIN_EARLY_FRACTION = 0.05
D5_MAX_FALSE_SAFE_UCB95 = 0.05


class D5JointReliabilityError(ValueError):
    """Raised whenever a D5 input or decision violates the frozen contract."""


@dataclass(frozen=True)
class D5DevelopmentData:
    """Flattened paired L11/L13 development rows and two offline targets."""

    features: torch.Tensor
    candidate_layer: torch.Tensor
    source_row: torch.Tensor
    task_id: torch.Tensor
    episode_index: torch.Tensor
    action_consistency: torch.Tensor
    unsafe_target: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def calls(self) -> int:
        return self.rows // 2

    def validate(self) -> None:
        rows = self.rows
        if (
            self.features.device.type != "cpu"
            or self.features.ndim != 2
            or self.features.shape != (rows, FEATURE_DIMENSION)
            or not self.features.is_floating_point()
            or not bool(torch.isfinite(self.features).all())
        ):
            raise D5JointReliabilityError("features must be finite CPU [N,97]")
        for name in (
            "candidate_layer",
            "source_row",
            "task_id",
            "episode_index",
        ):
            value = getattr(self, name)
            if value.device.type != "cpu" or value.dtype != torch.long or value.shape != (rows,):
                raise D5JointReliabilityError(f"{name} must be CPU int64 [N]")
        if (
            self.action_consistency.device.type != "cpu"
            or self.action_consistency.dtype != torch.bool
            or self.action_consistency.shape != (rows,)
            or self.unsafe_target.device.type != "cpu"
            or self.unsafe_target.dtype != torch.bool
            or self.unsafe_target.shape != (rows, 2)
        ):
            raise D5JointReliabilityError("D5 gate/target geometry differs")
        if rows < 2 or rows % 2:
            raise D5JointReliabilityError("D5 rows must contain candidate pairs")
        expected_layers = torch.tensor(D5_CANDIDATE_LAYERS).expand(self.calls, 2)
        if not torch.equal(self.candidate_layer.reshape(-1, 2), expected_layers):
            raise D5JointReliabilityError("candidate order must be L11 then L13")
        expected_source = torch.arange(self.calls).repeat_interleave(2)
        if not torch.equal(self.source_row, expected_source):
            raise D5JointReliabilityError("source-row pairing differs")
        for value in (self.task_id, self.episode_index):
            if not torch.equal(value[0::2], value[1::2]):
                raise D5JointReliabilityError("candidate group identity differs")
        if set(self.task_id.tolist()) != set(D5_TASK_IDS):
            raise D5JointReliabilityError("task coverage differs")
        if set(self.episode_index.tolist()) != set(D5_EPISODES):
            raise D5JointReliabilityError("development episode coverage differs")
        cells = set(zip(self.task_id.tolist(), self.episode_index.tolist()))
        expected_cells = {
            (task, episode) for task in D5_TASK_IDS for episode in D5_EPISODES
        }
        if cells != expected_cells:
            raise D5JointReliabilityError("task-episode coverage differs")
        for target_index in range(2):
            target = self.unsafe_target[:, target_index]
            if int(target.sum()) < 2 or int((~target).sum()) < 2:
                raise D5JointReliabilityError("target lacks binary support")


def development_data_from_mapping(payload: Mapping[str, Any]) -> D5DevelopmentData:
    required = (
        "features",
        "candidate_layer",
        "source_row",
        "task_id",
        "episode_index",
        "action_consistency",
        "unsafe_target",
    )
    if any(name not in payload for name in required):
        raise D5JointReliabilityError("D5 dataset is missing a required tensor")
    data = D5DevelopmentData(
        **{
            name: payload[name].detach().cpu().contiguous()
            for name in required
        }
    )
    data.validate()
    return data


@dataclass(frozen=True)
class RouteSummary:
    selected_layer: torch.Tensor
    selected_unsafe: torch.Tensor
    early_exit_calls: int
    safe_clusters: int
    false_safe_clusters: int
    false_safe_ucb95: float
    per_task_early_calls: tuple[int, ...]

    @property
    def early_exit_fraction(self) -> float:
        return self.early_exit_calls / int(self.selected_layer.numel())

    @property
    def feasible(self) -> bool:
        return (
            self.safe_clusters >= D5_MIN_SAFE_CLUSTERS
            and self.early_exit_fraction >= D5_MIN_EARLY_FRACTION
            and all(value > 0 for value in self.per_task_early_calls)
            and self.false_safe_ucb95 <= D5_MAX_FALSE_SAFE_UCB95
        )


@dataclass(frozen=True)
class ThresholdSelection:
    feasible: bool
    threshold: float | None
    summary: RouteSummary | None
    evaluated_thresholds: int


def _paired_bool(value: torch.Tensor, *, name: str, calls: int) -> torch.Tensor:
    if value.device.type != "cpu" or value.dtype != torch.bool:
        raise D5JointReliabilityError(f"{name} must be a CPU bool tensor")
    if value.shape == (calls * 2,):
        return value.reshape(calls, 2)
    if value.shape == (calls, 2):
        return value
    raise D5JointReliabilityError(f"{name} must have paired geometry")


def _paired_score(value: torch.Tensor, *, name: str, calls: int) -> torch.Tensor:
    if value.device.type != "cpu" or not value.is_floating_point():
        raise D5JointReliabilityError(f"{name} must be a CPU floating tensor")
    if value.shape == (calls * 2,):
        paired = value.reshape(calls, 2).double()
    elif value.shape == (calls, 2):
        paired = value.double()
    else:
        raise D5JointReliabilityError(f"{name} must have paired geometry")
    return paired


def route_at_threshold(
    full_action_score: torch.Tensor,
    gripper_score: torch.Tensor,
    action_consistency: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Return only counterfactual L11/L13/L27 choices for paired calls."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise D5JointReliabilityError("D5 threshold must be numeric")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise D5JointReliabilityError("D5 threshold must be finite")
    if full_action_score.ndim == 1:
        if full_action_score.numel() % 2:
            raise D5JointReliabilityError("D5 score rows must be paired")
        calls = full_action_score.numel() // 2
    elif full_action_score.ndim == 2 and full_action_score.shape[1] == 2:
        calls = full_action_score.shape[0]
    else:
        raise D5JointReliabilityError("D5 full-action score geometry differs")
    full = _paired_score(full_action_score, name="full-action score", calls=calls)
    gripper = _paired_score(gripper_score, name="gripper score", calls=calls)
    consistency = _paired_bool(
        action_consistency, name="action consistency", calls=calls
    )
    finite = torch.isfinite(full) & torch.isfinite(gripper)
    safe = (
        finite
        & consistency
        & (gripper <= D5_GRIPPER_THRESHOLD)
        & (full <= threshold_value)
    )
    selected = torch.full((calls,), D5_FALLBACK_LAYER, dtype=torch.long)
    selected[safe[:, 1]] = D5_CANDIDATE_LAYERS[1]
    selected[safe[:, 0]] = D5_CANDIDATE_LAYERS[0]
    return selected


def summarize_route(
    selected_layer: torch.Tensor,
    unsafe_target: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
) -> RouteSummary:
    """Summarize cluster-level false-safe risk for one decision vector."""

    if (
        selected_layer.device.type != "cpu"
        or selected_layer.dtype != torch.long
        or selected_layer.ndim != 1
    ):
        raise D5JointReliabilityError("selected layer must be CPU int64 [B]")
    calls = selected_layer.numel()
    if unsafe_target.shape == (calls * 2, 2):
        target = unsafe_target.reshape(calls, 2, 2)
    elif unsafe_target.shape == (calls, 2, 2):
        target = unsafe_target
    else:
        raise D5JointReliabilityError("unsafe target must be paired [B,2,2]")
    if target.device.type != "cpu" or target.dtype != torch.bool:
        raise D5JointReliabilityError("unsafe target must be CPU bool")
    task = task_id[0::2] if task_id.shape == (calls * 2,) else task_id
    episode = (
        episode_index[0::2]
        if episode_index.shape == (calls * 2,)
        else episode_index
    )
    if (
        task.device.type != "cpu"
        or episode.device.type != "cpu"
        or task.dtype != torch.long
        or episode.dtype != torch.long
        or task.shape != (calls,)
        or episode.shape != (calls,)
    ):
        raise D5JointReliabilityError("route identity geometry differs")
    early = selected_layer != D5_FALLBACK_LAYER
    selected_index = (selected_layer == D5_CANDIDATE_LAYERS[1]).long()
    selected_unsafe = torch.zeros(calls, dtype=torch.bool)
    rows = torch.arange(calls)
    selected_unsafe[early] = target[
        rows[early], selected_index[early]
    ].any(dim=1)
    safe_cells = set(zip(task[early].tolist(), episode[early].tolist()))
    false_cells = set(
        zip(task[early & selected_unsafe].tolist(), episode[early & selected_unsafe].tolist())
    )
    ucb = clopper_pearson_upper(len(false_cells), len(safe_cells))
    per_task = tuple(int((early & (task == value)).sum()) for value in D5_TASK_IDS)
    return RouteSummary(
        selected_layer=selected_layer.contiguous(),
        selected_unsafe=selected_unsafe.contiguous(),
        early_exit_calls=int(early.sum()),
        safe_clusters=len(safe_cells),
        false_safe_clusters=len(false_cells),
        false_safe_ucb95=ucb,
        per_task_early_calls=per_task,
    )


def select_inner_threshold(
    full_action_score: torch.Tensor,
    gripper_score: torch.Tensor,
    action_consistency: torch.Tensor,
    unsafe_target: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
) -> ThresholdSelection:
    """Select the exact frozen threshold objective using inner-OOF rows only."""

    if full_action_score.ndim == 1:
        if full_action_score.numel() % 2:
            raise D5JointReliabilityError("D5 score rows must be paired")
        calls = full_action_score.numel() // 2
    elif full_action_score.ndim == 2 and full_action_score.shape[1] == 2:
        calls = full_action_score.shape[0]
    else:
        raise D5JointReliabilityError("D5 full-action score geometry differs")
    full = _paired_score(full_action_score, name="full-action score", calls=calls)
    gripper = _paired_score(gripper_score, name="gripper score", calls=calls)
    consistency = _paired_bool(
        action_consistency, name="action consistency", calls=calls
    )
    fixed_eligible = (
        torch.isfinite(full)
        & torch.isfinite(gripper)
        & consistency
        & (gripper <= D5_GRIPPER_THRESHOLD)
    )
    finite = full[fixed_eligible]
    thresholds = torch.unique(finite, sorted=True)
    if thresholds.numel() == 0:
        return ThresholdSelection(False, None, None, 0)
    best_threshold: float | None = None
    best_summary: RouteSummary | None = None
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        selected = route_at_threshold(
            full_action_score,
            gripper_score,
            action_consistency,
            threshold=threshold,
        )
        summary = summarize_route(
            selected, unsafe_target, task_id, episode_index
        )
        if not summary.feasible:
            continue
        if best_summary is None or (
            summary.early_exit_calls,
            summary.safe_clusters,
            -threshold,
        ) > (
            best_summary.early_exit_calls,
            best_summary.safe_clusters,
            -float(best_threshold),
        ):
            best_threshold = threshold
            best_summary = summary
    return ThresholdSelection(
        feasible=best_summary is not None,
        threshold=best_threshold,
        summary=best_summary,
        evaluated_thresholds=int(thresholds.numel()),
    )


def mean_action_cosine_distance(
    candidate: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Match A1's mean horizon 7-D cosine action distance."""

    if (
        candidate.device.type != "cpu"
        or reference.device.type != "cpu"
        or candidate.shape != reference.shape
        or candidate.ndim != 3
        or candidate.shape[1:] != (8, 7)
        or not candidate.is_floating_point()
        or not reference.is_floating_point()
        or not bool(torch.isfinite(candidate).all())
        or not bool(torch.isfinite(reference).all())
    ):
        raise D5JointReliabilityError("candidate/reference actions must be finite CPU [B,8,7]")
    similarity = torch.nn.functional.cosine_similarity(
        candidate.double(), reference.double(), dim=-1, eps=1.0e-8
    )
    distance = (1.0 - similarity).mean(dim=1)
    if not bool(torch.isfinite(distance).all()):
        raise D5JointReliabilityError("action cosine distance is non-finite")
    return distance.contiguous()


def load_d5_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D5_CONTRACT_RELATIVE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != D5_CONTRACT_SHA256:
        raise D5JointReliabilityError("D5 frozen contract SHA-256 differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D5JointReliabilityError("D5 contract cannot be read") from error
    if not isinstance(value, Mapping):
        raise D5JointReliabilityError("D5 contract must be an object")
    contract = dict(value)
    if (
        contract.get("schema_version") != D5_SCHEMA_VERSION
        or contract.get("status") != D5_STATUS
        or contract.get("stage") != "V3-D5"
        or contract.get("scope", {}).get("development_only") is not True
        or contract.get("scope", {}).get("active_control_allowed") is not False
        or contract.get("scope", {}).get("independent_test_v2_access_allowed") is not False
        or contract.get("routing", {}).get("formula")
        != "route_safe=A1_original_action_consistency_AND_gripper_safe_AND_full_action_reliability_safe"
        or contract.get("routing", {}).get("A1_original_action_consistency", {}).get("threshold")
        != D5_ACTION_THRESHOLD
        or contract.get("routing", {}).get("gripper_safe", {}).get("threshold")
        != D5_GRIPPER_THRESHOLD
        or contract.get("formal_development_gate", {}).get("minimum_safe_clusters")
        != D5_MIN_SAFE_CLUSTERS
    ):
        raise D5JointReliabilityError("D5 frozen contract semantics differ")
    return contract


__all__ = [
    "D5_ACTION_THRESHOLD",
    "D5_CANDIDATE_LAYERS",
    "D5_CONTRACT_RELATIVE_PATH",
    "D5_CONTRACT_SHA256",
    "D5_EPISODES",
    "D5_FALLBACK_LAYER",
    "D5_GRIPPER_THRESHOLD",
    "D5JointReliabilityError",
    "D5DevelopmentData",
    "RouteSummary",
    "ThresholdSelection",
    "development_data_from_mapping",
    "load_d5_contract",
    "mean_action_cosine_distance",
    "route_at_threshold",
    "select_inner_threshold",
    "summarize_route",
]
