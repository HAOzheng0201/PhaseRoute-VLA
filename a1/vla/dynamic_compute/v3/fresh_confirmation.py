"""Filesystem-light protocol primitives for frozen V3-D8 confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .gripper_v2_calibration import clopper_pearson_upper


D8_SCHEMA_VERSION = "phase-route-vla.v3.d8-fresh-confirmation-contract.v1"
D8_STATUS = "D8_FRESH_CONFIRMATION_CONTRACT_FROZEN"
D8_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/joint_reliability/d8_fresh_confirmation_contract.json"
)
D8_CONTRACT_SHA256 = (
    "148a6e7208582958198b8f1265bb715c75e31bc0c282c7f588412ba9c6ba2c17"
)
D8_SCHEDULE_SCHEMA_VERSION = "phase-route-vla.v3.fresh-confirmation-schedule.v1"
D8_SCHEDULE_STATUS = "D8_FRESH_CONFIRMATION_SCHEDULE_FROZEN"
D8_SCHEDULE_RELATIVE_PATH = Path(
    "configs/research/v3/data_lineage/fresh_confirmation_v1_schedule.json"
)
D8_SCHEDULE_SHA256 = (
    "6a532130ec9ddad5d235cc342e44148a9324f9e0592a1554e3dac9f51956b920"
)
D8_TASK_IDS = tuple(range(10))
D8_REPLICATE_IDS = tuple(range(20))
D8_CLUSTER_COUNT = 200
D8_CLUSTERS_PER_TASK = 20
D8_STATE_SEED_BASE = 30_260_821
D8_POLICY_SEED_BASE = 40_260_821
D8_MIN_SAFE_CLUSTERS = 120
D8_MIN_SAFE_CLUSTERS_PER_TASK = 5
D8_MIN_EARLY_FRACTION = 0.10
D8_MAX_FALSE_UCB95 = 0.05
D8_MAX_FALSE_FULL_CLUSTERS = 3
D8_MAX_FALSE_GRIPPER_CALLS = 0
D8_MAX_SEVERE_FALSE_CLUSTERS = 0
D8_MIN_NONDEGENERATE_FRACTION = 0.01
D8_MIN_ESTIMATED_FM_REDUCTION = 0.30


class D8ProtocolError(ValueError):
    """Raised whenever D8 protocol geometry or metadata differs."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D8ProtocolError(f"{context} cannot be read") from error
    if not isinstance(value, Mapping):
        raise D8ProtocolError(f"{context} must be an object")
    return dict(value)


def load_d8_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D8_CONTRACT_RELATIVE_PATH
    if _sha256(path) != D8_CONTRACT_SHA256:
        raise D8ProtocolError("D8 frozen contract SHA-256 differs")
    contract = _json_object(path, context="D8 contract")
    gate = contract.get("confirmation_gate", {})
    authorization = contract.get("authorization", {})
    if (
        contract.get("schema_version") != D8_SCHEMA_VERSION
        or contract.get("status") != D8_STATUS
        or contract.get("scope", {}).get("shadow_decision_only") is not True
        or contract.get("scope", {}).get("official_episode_40_49_access_allowed")
        is not False
        or contract.get("fresh_schedule", {}).get("sha256")
        != D8_SCHEDULE_SHA256
        or contract.get("fresh_schedule", {}).get("required_clusters")
        != D8_CLUSTER_COUNT
        or contract.get("D7_final_router_finalization", {}).get("head_count") != 5
        or contract.get("D7_final_router_finalization", {}).get("lambda") != 0.01
        or gate.get("minimum_safe_clusters") != D8_MIN_SAFE_CLUSTERS
        or gate.get("minimum_early_exit_call_fraction")
        != D8_MIN_EARLY_FRACTION
        or gate.get("false_safe_cluster_ucb_at_most") != D8_MAX_FALSE_UCB95
        or gate.get("false_full_action_clusters_at_most")
        != D8_MAX_FALSE_FULL_CLUSTERS
        or gate.get("false_gripper_calls_at_most")
        != D8_MAX_FALSE_GRIPPER_CALLS
        or authorization.get("fresh_policy_rollout_authorized_on_contract_validation_alone")
        is not False
        or authorization.get("open_episode_40_49_authorized") is not False
    ):
        raise D8ProtocolError("D8 frozen contract semantics differ")
    return contract


@dataclass(frozen=True)
class FreshConfirmationRecord:
    task_id: int
    replicate_id: int
    state_seed: int
    policy_seed: int

    @property
    def cluster_key(self) -> str:
        return (
            f"libero_10:task{self.task_id}:fresh_confirm_v1:"
            f"replicate{self.replicate_id}"
        )


def expected_state_seed(task_id: int, replicate_id: int) -> int:
    if type(task_id) is not int or task_id not in D8_TASK_IDS:
        raise D8ProtocolError("D8 task id must be in 0..9")
    if type(replicate_id) is not int or replicate_id not in D8_REPLICATE_IDS:
        raise D8ProtocolError("D8 replicate id must be in 0..19")
    return D8_STATE_SEED_BASE + task_id * 10_000 + replicate_id


def expected_policy_seed(task_id: int, replicate_id: int) -> int:
    if type(task_id) is not int or task_id not in D8_TASK_IDS:
        raise D8ProtocolError("D8 task id must be in 0..9")
    if type(replicate_id) is not int or replicate_id not in D8_REPLICATE_IDS:
        raise D8ProtocolError("D8 replicate id must be in 0..19")
    return D8_POLICY_SEED_BASE + task_id * 10_000 + replicate_id


def load_fresh_confirmation_schedule(
    repo_root: str | Path,
) -> tuple[FreshConfirmationRecord, ...]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D8_SCHEDULE_RELATIVE_PATH
    if _sha256(path) != D8_SCHEDULE_SHA256:
        raise D8ProtocolError("D8 frozen schedule SHA-256 differs")
    schedule = _json_object(path, context="D8 schedule")
    state = schedule.get("state_generation", {})
    policy = schedule.get("policy_rollout", {})
    identity = schedule.get("identity_boundary", {})
    if (
        schedule.get("schema_version") != D8_SCHEDULE_SCHEMA_VERSION
        or schedule.get("status") != D8_SCHEDULE_STATUS
        or schedule.get("task_ids") != list(D8_TASK_IDS)
        or schedule.get("replicate_ids") != list(D8_REPLICATE_IDS)
        or schedule.get("record_count") != D8_CLUSTER_COUNT
        or schedule.get("official_benchmark_episode_index") is not None
        or state.get("seed_base") != D8_STATE_SEED_BASE
        or policy.get("seed_base") != D8_POLICY_SEED_BASE
        or state.get("manual_state_selection_allowed") is not False
        or state.get("outcome_based_replacement_allowed") is not False
        or policy.get("outcome_based_retry_or_replacement_allowed") is not False
        or any(value is not False for value in identity.values())
    ):
        raise D8ProtocolError("D8 frozen schedule semantics differ")
    records = tuple(
        FreshConfirmationRecord(
            task_id=task_id,
            replicate_id=replicate_id,
            state_seed=expected_state_seed(task_id, replicate_id),
            policy_seed=expected_policy_seed(task_id, replicate_id),
        )
        for task_id in D8_TASK_IDS
        for replicate_id in D8_REPLICATE_IDS
    )
    if (
        len(records) != D8_CLUSTER_COUNT
        or len({record.cluster_key for record in records}) != D8_CLUSTER_COUNT
        or len({record.state_seed for record in records}) != D8_CLUSTER_COUNT
        or len({record.policy_seed for record in records}) != D8_CLUSTER_COUNT
        or {record.state_seed for record in records}
        & {record.policy_seed for record in records}
    ):
        raise D8ProtocolError("D8 expanded schedule geometry differs")
    return records


@dataclass(frozen=True)
class D8ConfirmationSummary:
    total_clusters: int
    clusters_per_task: tuple[int, ...]
    safe_clusters: int
    safe_clusters_per_task: tuple[int, ...]
    policy_calls: int
    early_exit_calls: int
    early_exit_calls_per_task: tuple[int, ...]
    false_safe_clusters: int
    false_full_action_clusters: int
    false_gripper_calls: int
    severe_false_full_action_clusters: int
    nondegenerate_row_fraction: float
    estimated_fm_reduction_fraction: float
    all_candidate_rows_and_policy_calls_accounted_for: bool
    all_predictions_finite: bool

    def gate_checks(self) -> dict[str, bool]:
        if (
            self.policy_calls <= 0
            or not 0 <= self.early_exit_calls <= self.policy_calls
            or not 0 <= self.safe_clusters <= self.total_clusters
            or not 0 <= self.false_safe_clusters <= self.safe_clusters
            or not 0 <= self.false_full_action_clusters <= self.false_safe_clusters
            or not 0 <= self.false_gripper_calls <= self.early_exit_calls
            or not 0 <= self.severe_false_full_action_clusters
            <= self.false_full_action_clusters
            or len(self.clusters_per_task) != len(D8_TASK_IDS)
            or len(self.safe_clusters_per_task) != len(D8_TASK_IDS)
            or len(self.early_exit_calls_per_task) != len(D8_TASK_IDS)
            or any(value < 0 for value in self.clusters_per_task)
            or any(value < 0 for value in self.safe_clusters_per_task)
            or any(value < 0 for value in self.early_exit_calls_per_task)
            or sum(self.clusters_per_task) != self.total_clusters
            or sum(self.safe_clusters_per_task) != self.safe_clusters
            or sum(self.early_exit_calls_per_task) != self.early_exit_calls
            or not math.isfinite(self.nondegenerate_row_fraction)
            or not 0.0 <= self.nondegenerate_row_fraction <= 1.0
            or not math.isfinite(self.estimated_fm_reduction_fraction)
            or self.estimated_fm_reduction_fraction > 1.0
        ):
            raise D8ProtocolError("D8 confirmation summary geometry differs")
        ucb = clopper_pearson_upper(
            self.false_safe_clusters, self.safe_clusters
        )
        return {
            "all_200_clusters_present": self.total_clusters == D8_CLUSTER_COUNT
            and self.clusters_per_task
            == (D8_CLUSTERS_PER_TASK,) * len(D8_TASK_IDS),
            "all_rows_calls_accounted_for": (
                self.all_candidate_rows_and_policy_calls_accounted_for
            ),
            "all_predictions_finite": self.all_predictions_finite,
            "minimum_120_safe_clusters": self.safe_clusters
            >= D8_MIN_SAFE_CLUSTERS,
            "minimum_5_safe_clusters_per_task": all(
                value >= D8_MIN_SAFE_CLUSTERS_PER_TASK
                for value in self.safe_clusters_per_task
            ),
            "minimum_10_percent_early_exit_calls": (
                self.early_exit_calls / self.policy_calls
                >= D8_MIN_EARLY_FRACTION
            ),
            "all_10_tasks_nonzero_early_exit_calls": all(
                value > 0 for value in self.early_exit_calls_per_task
            ),
            "false_safe_exact_ucb95_at_most_5_percent": (
                ucb <= D8_MAX_FALSE_UCB95
            ),
            "false_full_action_clusters_at_most_three": (
                self.false_full_action_clusters
                <= D8_MAX_FALSE_FULL_CLUSTERS
            ),
            "false_gripper_calls_at_most_zero": (
                self.false_gripper_calls <= D8_MAX_FALSE_GRIPPER_CALLS
            ),
            "severe_false_full_action_clusters_at_most_zero": (
                self.severe_false_full_action_clusters
                <= D8_MAX_SEVERE_FALSE_CLUSTERS
            ),
            "ensemble_non_degenerate_on_at_least_one_percent_rows": (
                self.nondegenerate_row_fraction
                >= D8_MIN_NONDEGENERATE_FRACTION
            ),
            "estimated_FM_reduction_at_least_30_percent": (
                self.estimated_fm_reduction_fraction
                >= D8_MIN_ESTIMATED_FM_REDUCTION
            ),
            "always_defer_rejected": self.early_exit_calls > 0,
        }

    @property
    def false_safe_ucb95(self) -> float:
        return clopper_pearson_upper(
            self.false_safe_clusters, self.safe_clusters
        )

    @property
    def passes(self) -> bool:
        return all(self.gate_checks().values())


__all__ = [
    "D8_CLUSTERS_PER_TASK",
    "D8_CLUSTER_COUNT",
    "D8_CONTRACT_RELATIVE_PATH",
    "D8_CONTRACT_SHA256",
    "D8_POLICY_SEED_BASE",
    "D8_REPLICATE_IDS",
    "D8_SCHEDULE_RELATIVE_PATH",
    "D8_SCHEDULE_SHA256",
    "D8_STATE_SEED_BASE",
    "D8_TASK_IDS",
    "D8ConfirmationSummary",
    "D8ProtocolError",
    "FreshConfirmationRecord",
    "expected_policy_seed",
    "expected_state_seed",
    "load_d8_contract",
    "load_fresh_confirmation_schedule",
]
