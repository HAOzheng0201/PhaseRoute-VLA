"""Stage-11D action-free reliability contracts for Route-first routing.

The former Route-first student imitated the final layer selected by the
candidate-aware V3 router.  Stage 11D instead supervises the same 199-D
pre-action context with a counterfactual consistency label: whether L13 and
L27 produce sufficiently similar actions from the exact same flow-matching
input.  L27 remains a consistency reference, not an expert action.

This module is deliberately independent of online control.  It expands the
frozen generated-state schedule and constructs CPU targets/features only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    build_route_first_context_features,
)


STAGE11D_PROTOCOL_SCHEMA_VERSION = (
    "phase-route-vla.route-first-stage11d-reliability-protocol.v1"
)
STAGE11D_PROTOCOL_STATUS = "FROZEN_NEW_DEVELOPMENT_PROTOCOL_NOT_RUN"
STAGE11D_PROTOCOL_RELATIVE_PATH = Path(
    "configs/research/route_first_stage11d_reliability_protocol.json"
)
# Frozen protocol file digest. Validation refuses any drift.
STAGE11D_PROTOCOL_SHA256 = (
    "16a5b8a4adb268c99fec38741484cdde4ccfeab1e3079f11b79f1f4334b00e00"
)

STAGE11D_TASK_IDS = tuple(range(10))
STAGE11D_REPLICATE_IDS = tuple(range(20))
STAGE11D_TRAIN_REPLICATES = tuple(range(12))
STAGE11D_CALIBRATION_REPLICATES = tuple(range(12, 16))
STAGE11D_SHADOW_REPLICATES = tuple(range(16, 20))
STAGE11D_STATE_SEED_BASE = 93_260_830
STAGE11D_POLICY_SEED_BASE = 94_260_830
STAGE11D_CLUSTER_COUNT = 200
STAGE11D_CLUSTERS_PER_TASK = 20
STAGE11D_REPLAY_LAYERS = (13, 27)
STAGE11D_ACTION_THRESHOLD = 0.00390625
STAGE11D_SEVERE_RATIO = 4.0
STAGE11D_HORIZON = 8
STAGE11D_ACTION_DIMENSION = 7
STAGE11D_GRIPPER_INDEX = 6


class Stage11DReliabilityError(ValueError):
    """Raised when Stage-11D lineage, feature, or target contracts differ."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_replicate(replicate_id: int) -> str:
    if type(replicate_id) is not int or replicate_id not in STAGE11D_REPLICATE_IDS:
        raise Stage11DReliabilityError("Stage-11D replicate id must be in 0..19")
    if replicate_id in STAGE11D_TRAIN_REPLICATES:
        return "development_train"
    if replicate_id in STAGE11D_CALIBRATION_REPLICATES:
        return "calibration"
    return "shadow_confirmation"


def expected_state_seed(task_id: int, replicate_id: int) -> int:
    if type(task_id) is not int or task_id not in STAGE11D_TASK_IDS:
        raise Stage11DReliabilityError("Stage-11D task id must be in 0..9")
    split_for_replicate(replicate_id)
    return STAGE11D_STATE_SEED_BASE + task_id * 10_000 + replicate_id


def expected_policy_seed(task_id: int, replicate_id: int) -> int:
    if type(task_id) is not int or task_id not in STAGE11D_TASK_IDS:
        raise Stage11DReliabilityError("Stage-11D task id must be in 0..9")
    split_for_replicate(replicate_id)
    return STAGE11D_POLICY_SEED_BASE + task_id * 10_000 + replicate_id


@dataclass(frozen=True)
class Stage11DRecord:
    task_id: int
    replicate_id: int
    split: str
    state_seed: int
    policy_seed: int

    @property
    def cluster_key(self) -> str:
        return (
            f"libero_10:task{self.task_id}:route_first_reliability_v1:"
            f"replicate{self.replicate_id}"
        )


def build_stage11d_schedule() -> tuple[Stage11DRecord, ...]:
    records = tuple(
        Stage11DRecord(
            task_id=task_id,
            replicate_id=replicate_id,
            split=split_for_replicate(replicate_id),
            state_seed=expected_state_seed(task_id, replicate_id),
            policy_seed=expected_policy_seed(task_id, replicate_id),
        )
        for task_id in STAGE11D_TASK_IDS
        for replicate_id in STAGE11D_REPLICATE_IDS
    )
    if (
        len(records) != STAGE11D_CLUSTER_COUNT
        or len({record.cluster_key for record in records}) != len(records)
        or len({record.state_seed for record in records}) != len(records)
        or len({record.policy_seed for record in records}) != len(records)
        or {record.state_seed for record in records}
        & {record.policy_seed for record in records}
    ):
        raise Stage11DReliabilityError("Stage-11D schedule identity is not unique")
    expected_counts = {
        "development_train": 120,
        "calibration": 40,
        "shadow_confirmation": 40,
    }
    observed_counts = {
        split: sum(record.split == split for record in records)
        for split in expected_counts
    }
    if observed_counts != expected_counts:
        raise Stage11DReliabilityError("Stage-11D split geometry differs")
    return records


def validate_stage11d_protocol(repo_root: str | Path) -> dict[str, Any]:
    """Validate the immutable protocol and expand its deterministic schedule."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    path = root / STAGE11D_PROTOCOL_RELATIVE_PATH
    observed_sha256 = _sha256(path)
    if observed_sha256 != STAGE11D_PROTOCOL_SHA256:
        raise Stage11DReliabilityError("Stage-11D protocol SHA-256 differs")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage11DReliabilityError("Stage-11D protocol is unreadable") from error
    if not isinstance(protocol, Mapping):
        raise Stage11DReliabilityError("Stage-11D protocol must be an object")
    lineage = protocol.get("data_lineage", {})
    method = protocol.get("method_contract", {})
    replay = protocol.get("offline_replay", {})
    target = protocol.get("target_contract", {})
    authorization = protocol.get("authorization", {})
    reservations = protocol.get("future_state_reservations", {})
    if (
        protocol.get("schema_version") != STAGE11D_PROTOCOL_SCHEMA_VERSION
        or protocol.get("status") != STAGE11D_PROTOCOL_STATUS
        or method.get("runtime_feature_dimension") != ROUTE_FIRST_FEATURE_DIMENSION
        or method.get("decision_layers") != list(STAGE11D_REPLAY_LAYERS)
        or method.get("online_flow_matching_calls_per_policy_call") != 1
        or method.get("L11_development_authorized") is not False
        or lineage.get("task_ids") != list(STAGE11D_TASK_IDS)
        or lineage.get("replicate_ids") != list(STAGE11D_REPLICATE_IDS)
        or lineage.get("cluster_count") != STAGE11D_CLUSTER_COUNT
        or lineage.get("clusters_per_task") != STAGE11D_CLUSTERS_PER_TASK
        or lineage.get("state_generation", {}).get("state_seed_base")
        != STAGE11D_STATE_SEED_BASE
        or lineage.get("behavior_policy", {}).get("policy_seed_base")
        != STAGE11D_POLICY_SEED_BASE
        or replay.get("layers") != list(STAGE11D_REPLAY_LAYERS)
        or replay.get("same_cached_FM_input_x_for_both_layers") is not True
        or replay.get("L27_is_expert_or_success_label") is not False
        or target.get("full_action_unsafe_threshold")
        != STAGE11D_ACTION_THRESHOLD
        or target.get("severe_full_action_ratio") != STAGE11D_SEVERE_RATIO
        or authorization.get("new_state_generation_now") is not False
        or authorization.get("GPU_collection_now") is not False
        or authorization.get("active_environment_control_now") is not False
        or reservations.get("development_active_pilot", {}).get("authorized_now")
        is not False
        or reservations.get("Stage12_independent_confirmation", {}).get(
            "authorized_now"
        )
        is not False
    ):
        raise Stage11DReliabilityError("Stage-11D protocol semantics differ")
    schedule = build_stage11d_schedule()
    return {
        "protocol_path": str(STAGE11D_PROTOCOL_RELATIVE_PATH),
        "protocol_sha256": observed_sha256,
        "status": STAGE11D_PROTOCOL_STATUS,
        "clusters": len(schedule),
        "split_counts": {
            split: sum(record.split == split for record in schedule)
            for split in (
                "development_train",
                "calibration",
                "shadow_confirmation",
            )
        },
        "GPU_collection_authorized": False,
        "active_control_authorized": False,
    }


@dataclass(frozen=True)
class Stage11DActionReliabilityTargets:
    """Per-call direct L13-vs-L27 reliability truth."""

    full_action_distance: torch.Tensor
    full_action_unsafe: torch.Tensor
    gripper_step_unsafe: torch.Tensor
    joint_unsafe: torch.Tensor
    safe13: torch.Tensor
    severe_full_action_unsafe: torch.Tensor

    def validate(self, *, rows: int) -> None:
        if type(rows) is not int or rows < 1:
            raise Stage11DReliabilityError("Stage-11D target row count is invalid")
        if self.full_action_distance.shape != (rows,):
            raise Stage11DReliabilityError("full-action distance shape differs")
        if self.full_action_distance.dtype != torch.float64:
            raise Stage11DReliabilityError("full-action distance must be float64")
        if self.full_action_distance.device.type != "cpu" or not bool(
            torch.isfinite(self.full_action_distance).all()
        ):
            raise Stage11DReliabilityError("full-action distance must be finite CPU")
        for name in (
            "full_action_unsafe",
            "gripper_step_unsafe",
            "joint_unsafe",
            "safe13",
            "severe_full_action_unsafe",
        ):
            value = getattr(self, name)
            if (
                value.shape != (rows,)
                or value.dtype != torch.bool
                or value.device.type != "cpu"
            ):
                raise Stage11DReliabilityError(f"{name} geometry differs")
        if not torch.equal(
            self.joint_unsafe,
            self.full_action_unsafe | self.gripper_step_unsafe,
        ) or not torch.equal(self.safe13, ~self.joint_unsafe):
            raise Stage11DReliabilityError("Stage-11D joint target semantics differ")


def build_l13_reliability_targets(
    candidate_actions: torch.Tensor,
) -> Stage11DActionReliabilityTargets:
    """Build exact CPU truth from same-noise actions ordered as ``[L13,L27]``."""

    if (
        not isinstance(candidate_actions, torch.Tensor)
        or candidate_actions.device.type != "cpu"
        or candidate_actions.ndim != 4
        or tuple(candidate_actions.shape[1:])
        != (
            len(STAGE11D_REPLAY_LAYERS),
            STAGE11D_HORIZON,
            STAGE11D_ACTION_DIMENSION,
        )
        or candidate_actions.shape[0] < 1
        or not candidate_actions.is_floating_point()
        or not bool(torch.isfinite(candidate_actions).all())
    ):
        raise Stage11DReliabilityError(
            "candidate actions must be finite CPU [N,2,8,7] ordered L13,L27"
        )
    actions = candidate_actions.double()
    similarity = torch.nn.functional.cosine_similarity(
        actions[:, 0], actions[:, 1], dim=-1, eps=1.0e-8
    )
    distance = (1.0 - similarity).mean(dim=1).clamp_min(0.0).contiguous()
    full_unsafe = (distance > STAGE11D_ACTION_THRESHOLD).contiguous()
    gripper_unsafe = (
        (actions[:, 0, :, STAGE11D_GRIPPER_INDEX] >= 0.0)
        != (actions[:, 1, :, STAGE11D_GRIPPER_INDEX] >= 0.0)
    ).any(dim=1).contiguous()
    joint_unsafe = (full_unsafe | gripper_unsafe).contiguous()
    result = Stage11DActionReliabilityTargets(
        full_action_distance=distance,
        full_action_unsafe=full_unsafe,
        gripper_step_unsafe=gripper_unsafe,
        joint_unsafe=joint_unsafe,
        safe13=(~joint_unsafe).contiguous(),
        severe_full_action_unsafe=(
            distance > STAGE11D_SEVERE_RATIO * STAGE11D_ACTION_THRESHOLD
        ).contiguous(),
    )
    result.validate(rows=int(candidate_actions.shape[0]))
    return result


@dataclass(frozen=True)
class Stage11DReliabilityBatch:
    """Action-free features paired with offline-only reliability truth."""

    features: torch.Tensor
    targets: Stage11DActionReliabilityTargets

    def validate(self) -> None:
        rows = int(self.features.shape[0]) if self.features.ndim == 2 else -1
        if (
            self.features.shape != (rows, ROUTE_FIRST_FEATURE_DIMENSION)
            or rows < 1
            or self.features.device.type != "cpu"
            or self.features.dtype != torch.float32
            or not bool(torch.isfinite(self.features).all())
        ):
            raise Stage11DReliabilityError("Stage-11D action-free features differ")
        self.targets.validate(rows=rows)


def build_stage11d_reliability_batch(
    runtime_inputs: Mapping[str, torch.Tensor],
    candidate_actions: torch.Tensor,
) -> Stage11DReliabilityBatch:
    """Pair causal context with labels while keeping actions out of features."""

    features = build_route_first_context_features(runtime_inputs)
    if features.device.type != "cpu":
        features = features.detach().cpu()
    result = Stage11DReliabilityBatch(
        features=features.float().contiguous(),
        targets=build_l13_reliability_targets(candidate_actions),
    )
    result.validate()
    return result


__all__ = [
    "STAGE11D_ACTION_THRESHOLD",
    "STAGE11D_CALIBRATION_REPLICATES",
    "STAGE11D_CLUSTER_COUNT",
    "STAGE11D_POLICY_SEED_BASE",
    "STAGE11D_PROTOCOL_RELATIVE_PATH",
    "STAGE11D_PROTOCOL_SCHEMA_VERSION",
    "STAGE11D_PROTOCOL_SHA256",
    "STAGE11D_PROTOCOL_STATUS",
    "STAGE11D_REPLAY_LAYERS",
    "STAGE11D_SHADOW_REPLICATES",
    "STAGE11D_STATE_SEED_BASE",
    "STAGE11D_TASK_IDS",
    "STAGE11D_TRAIN_REPLICATES",
    "Stage11DActionReliabilityTargets",
    "Stage11DRecord",
    "Stage11DReliabilityBatch",
    "Stage11DReliabilityError",
    "build_l13_reliability_targets",
    "build_stage11d_reliability_batch",
    "build_stage11d_schedule",
    "expected_policy_seed",
    "expected_state_seed",
    "split_for_replicate",
    "validate_stage11d_protocol",
]
