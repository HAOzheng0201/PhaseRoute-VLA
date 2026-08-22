"""Filesystem-light protocol guards for the frozen V3-D9 test design."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


D9_SCHEMA_VERSION = "phase-route-vla.v3.d9-independent-test-contract.v1"
D9_STATUS = "D9_INDEPENDENT_TEST_PROTOCOL_FROZEN"
D9_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/independent_test/d9_paired_active_test_contract.json"
)
D9_CONTRACT_SHA256 = (
    "eea74662357d39737a3ac84b2d59059150ac4f098c6bddbfe695ba1ed64e59d3"
)
D9_SELECTION_RELATIVE_PATH = Path(
    "configs/research/v3/data_lineage/independent_test_v2.json"
)
D9_SELECTION_SHA256 = (
    "e2c1b2a11f84af9b71d588bf638d794c5a29870ace87b46b65960749e0f9bdf4"
)
D9_D8_FORMAL_RELATIVE_PATH = Path(
    "results/v3/v3_d8_formal_confirmation_result.json"
)
D9_D8_FORMAL_SHA256 = (
    "4e6114fc5523bea0c0e156ec7095d8820c650e28250db7f9f7282e08121333fc"
)
D9_TASK_IDS = tuple(range(10))
D9_EPISODE_INDICES = tuple(range(40, 50))
D9_RECORD_COUNT = 100
D9_RECORDS_PER_TASK = 10
D9_ARMS = ("frozen_original_A1", "frozen_PhaseRoute_D8")
D9_GPU_ALLOWLIST = (0, 1, 2, 3)


class D9ProtocolError(ValueError):
    """Raised when the D9 contract or metadata differs from the freeze."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D9ProtocolError(f"{context} cannot be read") from error
    if not isinstance(value, Mapping):
        raise D9ProtocolError(f"{context} must be an object")
    return dict(value)


@dataclass(frozen=True)
class D9TestRecord:
    task_id: int
    episode_index: int
    seed: int

    @property
    def canonical_key(self) -> str:
        return f"libero_10:task{self.task_id}:episode{self.episode_index}"

    @property
    def arm_order(self) -> tuple[str, str]:
        if (self.task_id + self.episode_index) % 2 == 0:
            return D9_ARMS
        return tuple(reversed(D9_ARMS))

    @property
    def physical_gpu_index(self) -> int:
        return self.task_id % len(D9_GPU_ALLOWLIST)


def expected_d9_seed(task_id: int, episode_index: int) -> int:
    if type(task_id) is not int or task_id not in D9_TASK_IDS:
        raise D9ProtocolError("D9 task id must be in 0..9")
    if type(episode_index) is not int or episode_index not in D9_EPISODE_INDICES:
        raise D9ProtocolError("D9 episode index must be in 40..49")
    return 20_260_851 + task_id * 10_000 + (episode_index - 40)


def records_from_selection(value: Mapping[str, Any]) -> tuple[D9TestRecord, ...]:
    raw = value.get("records")
    if (
        value.get("schema_version")
        != "phase-route-vla.v3.data-lineage-selection.v1"
        or value.get("suite") != "libero_10"
        or value.get("role") != "independent_test_v2"
        or not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != D9_RECORD_COUNT
    ):
        raise D9ProtocolError("D9 selection metadata semantics differ")
    records: list[D9TestRecord] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "task_id",
            "episode_index",
            "seed",
        }:
            raise D9ProtocolError("D9 selection record fields differ")
        task_id = item["task_id"]
        episode_index = item["episode_index"]
        seed = item["seed"]
        if (
            type(task_id) is not int
            or type(episode_index) is not int
            or type(seed) is not int
            or seed != expected_d9_seed(task_id, episode_index)
        ):
            raise D9ProtocolError("D9 selection record value differs")
        records.append(D9TestRecord(task_id, episode_index, seed))
    expected_pairs = [
        (task, episode)
        for task in D9_TASK_IDS
        for episode in D9_EPISODE_INDICES
    ]
    observed_pairs = [(record.task_id, record.episode_index) for record in records]
    if (
        observed_pairs != expected_pairs
        or len({record.canonical_key for record in records}) != D9_RECORD_COUNT
        or len({record.seed for record in records}) != D9_RECORD_COUNT
        or any(record.physical_gpu_index not in D9_GPU_ALLOWLIST for record in records)
    ):
        raise D9ProtocolError("D9 selection ordering or coverage differs")
    return tuple(records)


def load_d9_selection_metadata(repo_root: str | Path) -> tuple[D9TestRecord, ...]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D9_SELECTION_RELATIVE_PATH
    if _sha256(path) != D9_SELECTION_SHA256:
        raise D9ProtocolError("D9 selection metadata SHA-256 differs")
    return records_from_selection(_json_object(path, context="D9 selection metadata"))


def _validate_contract_semantics(contract: Mapping[str, Any]) -> None:
    scope = contract.get("scope", {})
    prerequisite = contract.get("prerequisite", {})
    lineage = contract.get("test_lineage", {})
    model = contract.get("frozen_model", {})
    readiness = contract.get("D9A_runtime_adapter_readiness", {})
    evaluation = contract.get("paired_evaluation", {})
    safety = contract.get("same_noise_safety_audit", {})
    gate = contract.get("primary_gate", {})
    bootstrap = gate.get("paired_task_stratified_bootstrap", {})
    stopping = contract.get("stopping_missingness_and_retries", {})
    execution = contract.get("execution_order", {})
    authorization = contract.get("authorization", {})
    boundary = contract.get("claim_boundary", {})
    if (
        contract.get("schema_version") != D9_SCHEMA_VERSION
        or contract.get("status") != D9_STATUS
        or scope.get("protocol_design_before_test_sample_access") is not True
        or scope.get("test_sample_payload_access_allowed_during_contract_validation")
        is not False
        or scope.get("episode_40_49_state_access_allowed_during_contract_validation")
        is not False
        or scope.get("active_control_allowed_during_contract_validation") is not False
        or scope.get("model_refit_feature_change_or_threshold_change_allowed")
        is not False
        or prerequisite.get("D8_formal_result_sha256") != D9_D8_FORMAL_SHA256
        or prerequisite.get("required_D8_status")
        != "PASS_V3_D8_PROSPECTIVE_SHADOW_CONFIRMATION"
        or prerequisite.get("required_D8_authorization")
        != "INDEPENDENT_TEST_V2_PROTOCOL_DESIGN_ONLY"
        or lineage.get("selection_metadata_sha256") != D9_SELECTION_SHA256
        or lineage.get("sample_state_payload_may_be_opened_for_protocol_validation")
        is not False
        or lineage.get("task_ids") != list(D9_TASK_IDS)
        or lineage.get("episode_indices") != list(D9_EPISODE_INDICES)
        or lineage.get("records") != D9_RECORD_COUNT
        or lineage.get("records_per_task") != D9_RECORDS_PER_TASK
        or model.get("feature_dimension") != 97
        or model.get("candidate_layers") != [11, 13]
        or model.get("fallback_layer") != 27
        or model.get("head_count") != 5
        or model.get("full_action_runtime_threshold") != 0.49143093002787247
        or model.get("head0_gripper_threshold") != 0.043773197319646726
        or model.get("A1_action_consistency_threshold") != 0.00390625
        or model.get("model_or_normalizer_refit_after_D8") is not False
        or model.get("feature_selection_after_D8") is not False
        or model.get("threshold_selection_after_D8") is not False
        or readiness.get("test_state_or_test_sample_payload_access_allowed")
        is not False
        or readiness.get("active_test_rollout_allowed") is not False
        or readiness.get("required_D8_parity_policy_calls") != 7140
        or readiness.get("required_D8_parity_candidate_rows") != 14280
        or readiness.get("required_D8_selected_layer_exact_matches") != 7140
        or readiness.get("required_D8_candidate_safe_exact_matches") != 14280
        or readiness.get("required_D8_five_head_prediction_max_abs_error") != 1e-12
        or readiness.get("readiness_attestation_required_before_test_access")
        is not True
        or evaluation.get("arms") != list(D9_ARMS)
        or evaluation.get("rollouts_per_arm") != D9_RECORD_COUNT
        or evaluation.get("total_required_rollouts") != 2 * D9_RECORD_COUNT
        or evaluation.get("same_official_init_state_per_pair") is not True
        or evaluation.get("same_policy_seed_per_pair") is not True
        or evaluation.get("only_controller_differs_between_arms") is not True
        or evaluation.get("physical_gpu_allowlist") != list(D9_GPU_ALLOWLIST)
        or evaluation.get("GPU_4_to_7_allowed") is not False
        or safety.get("PhaseRoute_arm_only") is not True
        or safety.get("replay_layers") != [11, 13, 27]
        or safety.get("all_early_exit_calls_must_have_truth") is not True
        or gate.get("all_criteria_are_conjunctive") is not True
        or gate.get("required_complete_pairs") != D9_RECORD_COUNT
        or gate.get("required_complete_pairs_per_task") != D9_RECORDS_PER_TASK
        or gate.get("required_rollouts") != 2 * D9_RECORD_COUNT
        or gate.get("PhaseRoute_success_rate_at_least") != 0.75
        or gate.get("PhaseRoute_minus_A1_success_rate_at_least") != -0.05
        or gate.get("PhaseRoute_minus_A1_successes_per_task_at_least") != -2
        or bootstrap.get("resamples") != 100_000
        or bootstrap.get("seed") != 60_260_821
        or bootstrap.get("one_sided_confidence") != 0.95
        or bootstrap.get("lower_percentile") != 0.05
        or bootstrap.get("lower_bound_at_least") != -0.10
        or gate.get("PhaseRoute_early_exit_call_fraction_at_least") != 0.10
        or gate.get("PhaseRoute_safe_clusters_at_least") != 60
        or gate.get("PhaseRoute_safe_clusters_per_task_at_least") != 2
        or gate.get("false_safe_cluster_exact_CP_UCB95_at_most") != 0.05
        or gate.get("false_full_action_clusters_at_most") != 2
        or gate.get("false_gripper_calls_at_most") != 0
        or gate.get("severe_false_full_action_clusters_at_most") != 0
        or gate.get("measured_FM_calls_per_policy_call_reduction_at_least")
        != 0.25
        or gate.get("always_defer_is_pass") is not False
        or stopping.get("interim_success_safety_or_efficiency_aggregation_allowed")
        is not False
        or stopping.get("optional_stopping_allowed") is not False
        or stopping.get("all_100_pairs_required_before_formal_aggregate") is not True
        or stopping.get("outcome_based_retry_allowed") is not False
        or stopping.get("replacement_episode_or_seed_allowed") is not False
        or execution.get("test_access_before_D9B_readiness") is not False
        or execution.get("active_control_before_D9B_readiness") is not False
        or authorization.get("on_contract_validation_pass")
        != "D9A_RUNTIME_ADAPTER_IMPLEMENTATION_AND_D8_PARITY_ONLY"
        or authorization.get("test_sample_or_state_access_on_contract_validation_pass")
        is not False
        or authorization.get("active_control_on_contract_validation_pass") is not False
        or authorization.get("on_D9B_readiness_pass")
        != "D9C_ONE_SHOT_PAIRED_ACTIVE_INDEPENDENT_TEST"
        or authorization.get("additional_test_tuning_or_second_independent_test")
        is not False
        or authorization.get("deployment_authorized") is not False
        or boundary.get("contract_validation_is_independent_test_result") is not False
        or boundary.get("McNemar_equality_test_is_noninferiority_test") is not False
    ):
        raise D9ProtocolError("D9 frozen contract semantics differ")


def load_d9_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    contract_path = root / D9_CONTRACT_RELATIVE_PATH
    if _sha256(contract_path) != D9_CONTRACT_SHA256:
        raise D9ProtocolError("D9 frozen contract SHA-256 differs")
    contract = _json_object(contract_path, context="D9 contract")
    _validate_contract_semantics(contract)
    records = load_d9_selection_metadata(root)
    if len(records) != D9_RECORD_COUNT:
        raise D9ProtocolError("D9 selection record count differs")
    d8_path = root / D9_D8_FORMAL_RELATIVE_PATH
    if _sha256(d8_path) != D9_D8_FORMAL_SHA256:
        raise D9ProtocolError("D9 D8 formal result SHA-256 differs")
    d8 = _json_object(d8_path, context="D9 D8 formal result")
    if (
        d8.get("status") != "PASS_V3_D8_PROSPECTIVE_SHADOW_CONFIRMATION"
        or not all(d8.get("gate_checks", {}).values())
        or d8.get("authorization", {}).get("authorized")
        != "INDEPENDENT_TEST_V2_PROTOCOL_DESIGN_ONLY"
        or d8.get("authorization", {}).get("open_episode_40_49_authorized")
        is not False
        or d8.get("access_ledger", {}).get("official_episode_40_49_opened")
        is not False
        or d8.get("access_ledger", {}).get("active_control") is not False
    ):
        raise D9ProtocolError("D9 D8 authorization semantics differ")
    return contract


__all__ = [
    "D9_ARMS",
    "D9_CONTRACT_RELATIVE_PATH",
    "D9_CONTRACT_SHA256",
    "D9_EPISODE_INDICES",
    "D9_GPU_ALLOWLIST",
    "D9_RECORD_COUNT",
    "D9_RECORDS_PER_TASK",
    "D9_SELECTION_RELATIVE_PATH",
    "D9_SELECTION_SHA256",
    "D9_STATUS",
    "D9_TASK_IDS",
    "D9ProtocolError",
    "D9TestRecord",
    "expected_d9_seed",
    "load_d9_contract",
    "load_d9_selection_metadata",
    "records_from_selection",
]
