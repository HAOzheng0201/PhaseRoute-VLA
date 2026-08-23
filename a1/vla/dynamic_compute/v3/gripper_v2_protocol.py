"""Frozen, standard-library-only PhaseRoute-V3 Gripper-v2 protocol.

V3-D1 is a design stage.  This module defines the immutable protocol,
constructs synthetic discrete gripper targets, and validates grouped fold
assignments.  It deliberately contains no trainer and imports no ML library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


PROTOCOL_SCHEMA_VERSION = "phase-route-vla.v3.gripper-v2-protocol.v1"
PROTOCOL_ID = "phaseroute-v3-d1-gripper-v2-20260820-v1"
PROTOCOL_STATUS = "D1_GRIPPER_V2_PROTOCOL_FROZEN"

HORIZON = 8
ACTION_DIMENSION = 7
GRIPPER_INDEX = 6
DECISION_LAYERS = (11, 13)
TEACHER_LAYER = 27
BASE_FEATURE_DIMENSION = 82
TEMPORAL_SIGN_DIMENSION = 8
TEMPORAL_TRANSITION_DIMENSION = 7
FEATURE_DIMENSION = 97

DEVELOPMENT_EPISODES = tuple(range(12, 30))
CALIBRATION_EPISODES = tuple(range(30, 40))
INDEPENDENT_TEST_EPISODES = tuple(range(40, 50))
OUTER_FOLD_COUNT = len(DEVELOPMENT_EPISODES)
INNER_FOLD_COUNT = OUTER_FOLD_COUNT - 1

RUNTIME_CONTEXT_NAMES = (
    "instruction_summary",
    "vision_crop_summary",
    "vision_crop_mask",
    "phase_embedding",
    "phase_scalars",
    "normalized_proprio",
    "proprio_history",
    "action_history",
    "history_mask",
)
RUNTIME_INPUT_NAMES = RUNTIME_CONTEXT_NAMES + (
    "current_candidate_action",
    "candidate_layer",
)
FORBIDDEN_RUNTIME_NAMES = frozenset(
    {
        "other_decision_layer_candidate_action",
        "layer27_candidate_action",
        "teacher_action",
        "full_depth_action",
        "full_depth_delta",
        "behavior_action",
        "behavior_exit",
        "dataset_index",
        "task_id",
        "episode_index",
        "call_ordinal",
        "seed",
        "success",
        "reward",
        "done",
        "future_observation",
        "future_proprio",
    }
)

_MAX_PROTOCOL_BYTES = 1024 * 1024
_SELECTION_SCHEMA_VERSION = "phase-route-vla.v3.data-lineage-selection.v1"


class GripperV2ProtocolError(ValueError):
    """Base class for fail-closed D1 protocol errors."""


class ProtocolPathError(GripperV2ProtocolError):
    """Raised when a protocol or bound input path is unsafe."""


class ProtocolSchemaError(GripperV2ProtocolError):
    """Raised when the protocol differs from the frozen contract."""


class TargetConstructionError(GripperV2ProtocolError):
    """Raised for invalid synthetic target inputs."""


class FoldContractError(GripperV2ProtocolError):
    """Raised when a grouped CV assignment violates the frozen split."""


def _role_contract(
    role: str,
    episodes: tuple[int, ...],
    count: int,
    path: str,
    sha256: str,
    first_label_stage: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "suite": "libero_10",
        "episode_indices": list(episodes),
        "task_ids": list(range(10)),
        "key_count": count,
        "selection_path": path,
        "selection_sha256": sha256,
        "first_label_access_stage": first_label_stage,
    }


_PROTOCOL_TEMPLATE: dict[str, Any] = {
    "schema_version": PROTOCOL_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_ID,
    "stage": "V3-D1",
    "status": PROTOCOL_STATUS,
    "scope": {
        "design_only": True,
        "development_only": True,
        "metadata_only": True,
        "cpu_only": True,
        "training_allowed": False,
        "payload_deserialization_allowed": False,
        "runtime_control_allowed": False,
    },
    "lineage": {
        "d0_commit": "2eadd3e200b9d31e6ec93c7423bdbb1830aae9dc",
        "d0_result_path": "results/v3/v3_d0_data_lineage_audit.json",
        "d0_result_sha256": (
            "64d1159b3941fe1e7b806da981a0f47297758dcc2cad87d4e283d03db3a71c4b"
        ),
        "d0_required_status": "PASS_NO_KNOWN_HIT",
        "d0_selection_bundle_sha256": (
            "9e89175b1ee2c8c82494c95d38f861e907b262c6cd072361db0a2e1df28204bd"
        ),
        "legacy_manifest_path": "docs/research/v3/legacy_evidence_manifest.json",
        "legacy_manifest_sha256": (
            "4ae5b617525a1f575f62700ab46434a1c9e8b20b9d13863b7ae8787f74c0ea6a"
        ),
        "legacy_c355_result_path": (
            "reports/phase_route_v2_stage_c355_development_predictor_training_"
            "20260818_v1/result.json"
        ),
        "legacy_c355_result_sha256": (
            "ba9da228f2607a22b9839e630c332e88c032ae91fdaa0f62efbb7cbcca55e678"
        ),
        "legacy_c355_required_status": "C355_DEVELOPMENT_NESTED_OOF_COMPLETE",
        "legacy_gripper_disposition": "BASELINE_OR_FAIL_NEGATIVE_RESULT_FROZEN",
        "legacy_failed_metric": {
            "family": "gripper_positive_magnitude",
            "target": "step_mismatch",
            "layer": 13,
            "metric": "positive_only_mae_ratio",
            "value": 1.0073668609606237,
            "positive_support": 487,
        },
    },
    "data_roles": [
        _role_contract(
            "development_v2",
            DEVELOPMENT_EPISODES,
            180,
            "configs/research/v3/data_lineage/development_v2.json",
            "59af8441d4207b23e4ade2dff5b987d70490e9f6ab7aff50b97255e0292436eb",
            "V3-D2",
        ),
        _role_contract(
            "calibration_v2",
            CALIBRATION_EPISODES,
            100,
            "configs/research/v3/data_lineage/calibration_v2.json",
            "6f2b2817985740298a06c4412b2f857624ac16c98d174d3ad03f1acca238f79e",
            "V3-D3",
        ),
        _role_contract(
            "independent_test_v2",
            INDEPENDENT_TEST_EPISODES,
            100,
            "configs/research/v3/data_lineage/independent_test_v2.json",
            "e2c1b2a11f84af9b71d588bf638d794c5a29870ace87b46b65960749e0f9bdf4",
            "V3-D7",
        ),
    ],
    "action_contract": {
        "candidate_action_shape": ["B", 2, 8, 7],
        "same_noise_teacher_shape": ["B", 8, 7],
        "decision_layers": [11, 13],
        "teacher_layer": 27,
        "teacher_role": "same_noise_consistency_label_only",
        "teacher_is_expert_or_success_label": False,
        "horizon": 8,
        "action_dimension": 7,
        "gripper_index": 6,
        "binary_state_formula": "state(x)=1 if x>=0 else 0",
        "threshold": 0.0,
        "threshold_tie_state": 1,
        "nonfinite_policy": "abort_whole_partition",
    },
    "runtime_input_contract": {
        "ordered_names": list(RUNTIME_INPUT_NAMES),
        "context_shapes": {
            "instruction_summary": ["B", 3584],
            "vision_crop_summary": ["B", 5, 3584],
            "vision_crop_mask": ["B", 5],
            "phase_embedding": ["B", 128],
            "phase_scalars": ["B", 3],
            "normalized_proprio": ["B", 8],
            "proprio_history": ["B", 8, 8],
            "action_history": ["B", 8, 8, 7],
            "history_mask": ["B", 8],
            "current_candidate_action": ["B", 8, 7],
            "candidate_layer": [],
        },
        "candidate_layer_semantics": "one_scalar_int_per_isolated_candidate_call",
        "past_only_context": True,
        "single_current_candidate_only": True,
        "forbidden_names": sorted(FORBIDDEN_RUNTIME_NAMES),
        "base_feature_dimension": 82,
        "feature_layout": [
            {
                "name": "legacy_causal_context",
                "slice": [0, 82],
                "dimension": 82,
            },
            {
                "name": "current_candidate_gripper_sign_sequence",
                "slice": [82, 90],
                "dimension": 8,
                "encoding": "-1_or_plus1_from_x_ge_0",
            },
            {
                "name": "current_candidate_gripper_transition_pattern",
                "slice": [90, 97],
                "dimension": 7,
                "encoding": "1_if_adjacent_binary_state_changes_else_0",
            },
        ],
        "output_shape": ["B", 97],
        "feature_dimension": 97,
        "other_candidate_or_teacher_visible": False,
    },
    "target_contract": {
        "label_source": "same_noise_layer27_consistency_teacher_offline_only",
        "step_state_shape": ["B", 2, 8],
        "step_mismatch_bits_shape": ["B", 2, 8],
        "transition_pattern_shape": ["B", 2, 7],
        "transition_mismatch_bits_shape": ["B", 2, 7],
        "occurrence_shape": ["B", 2, 2],
        "count_shape": ["B", 2, 2],
        "target_axis_order": ["step", "transition"],
        "step_count_support": list(range(9)),
        "transition_count_support": list(range(8)),
        "conditional_step_count_support": list(range(1, 9)),
        "conditional_transition_count_support": list(range(1, 8)),
        "first_transition_mismatch_shape": ["B", 2],
        "first_transition_mismatch_support": list(range(8)),
        "none_timing_code": 0,
        "timing_codes": "1_based_transition_position_1_through_7",
        "step_mismatch_formula": "candidate_state[t] XOR teacher_state[t]",
        "transition_formula": "state[t] XOR state[t-1]",
        "transition_mismatch_formula": (
            "candidate_transition[t] XOR teacher_transition[t]"
        ),
        "occurrence_formula": "count>0",
        "expected_fraction_formula": "P(count>0)*E[count|count>0]/support_max",
        "continuous_positive_magnitude_target_allowed": False,
        "teacher_or_target_visible_at_runtime": False,
    },
    "model_contract": {
        "occurrence_head": {
            "family": "anchored_bernoulli_logistic_glm",
            "input_dimension": 97,
            "outputs": ["step_any", "transition_any"],
            "loss": "unweighted_binary_cross_entropy",
            "fold_train_layer_prevalence_anchor": True,
        },
        "count_baseline": {
            "family": "zero_truncated_binomial_glm",
            "eligible_as_primary_method": False,
            "input_dimension": 97,
            "supports": {"step": [1, 8], "transition": [1, 7]},
            "normalization": "binomial_pmf_divided_by_one_minus_p0",
            "loss": "conditional_negative_log_likelihood",
        },
        "primary_challenger": {
            "family": "ordinal_cumulative_link_glm",
            "eligible_as_primary_method": True,
            "input_dimension": 97,
            "supports": {"step": [1, 8], "transition": [1, 7]},
            "ordered_cutpoints": "strictly_increasing_via_positive_increments",
            "trainable_cutpoints": True,
            "cutpoints_count_total_across_layers_and_targets": 26,
            "linear_score_bias": False,
            "loss": "conditional_negative_log_likelihood",
        },
        "legacy_comparator": {
            "family": "c355_82d_continuous_positive_magnitude_hurdle",
            "frozen_negative_result_only": True,
            "eligible_for_retraining_on_old_data": False,
        },
        "shared_constraints": {
            "hidden_layers": 0,
            "linear_feature_head_bias": False,
            "ordinal_cutpoints_are_not_linear_feature_bias": True,
            "residual_weights_initialized_exact_zero": True,
            "fold_train_layer_anchors_only": True,
            "fit_partition_feature_normalization_only": True,
            "primary_occurrence_plus_count_trainable_parameter_cap": 512,
            "no_shared_loss_across_occurrence_count_or_tail": True,
        },
        "primary_family_fixed_before_labels": "ordinal_cumulative_link_glm",
        "post_label_family_switch_allowed": False,
    },
    "training_contract_for_d2": {
        "device": "cpu",
        "dtype": "float64",
        "full_batch": True,
        "optimizer": "LBFGS_strong_wolfe",
        "max_iterations": 500,
        "history_size": 100,
        "tolerance_grad": 1e-10,
        "tolerance_change": 1e-12,
        "l2_lambda_grid": [0.001, 0.01, 0.1],
        "selection_rule": "largest_lambda_within_one_se_of_inner_minimum",
        "random_seed": 20260820,
        "deterministic_algorithms": True,
        "target_aware_row_drop_allowed": False,
    },
    "cross_validation": {
        "primary": "18_outer_by_17_inner_episode_index_LOEO",
        "development_episode_indices": list(DEVELOPMENT_EPISODES),
        "outer_fold_count": 18,
        "outer_fold_rule": "fold_id=episode_index-12",
        "outer_validation": "one_episode_index_across_all_10_tasks",
        "inner_fold_count_per_outer": 17,
        "inner_fold_rule": "sorted_remaining_episode_indices_each_held_once",
        "group_key": "libero_10:task{task_id}:episode{episode_index}",
        "all_calls_and_layers_stay_in_group": True,
        "candidate_pair_may_split": False,
        "random_row_split_allowed": False,
        "outer_validation_used_for_selection": False,
        "inner_task_episode_cells": 170,
        "cell_weighting": "equal_over_17_episode_indices_x_10_tasks",
        "secondary_task_jackknife_folds": 10,
        "secondary_task_jackknife_is_gate": False,
    },
    "scientific_metrics": {
        "occurrence": [
            "brier_score",
            "brier_skill_vs_fold_train_layer_prevalence",
            "tie_aware_auroc",
        ],
        "conditional_count_primary": "negative_log_likelihood",
        "conditional_count_secondary": ["ranked_probability_score", "count_mae"],
        "derived_expected_fraction": "raw_sse",
        "timing_secondary": "first_transition_mismatch_mae_and_accuracy",
        "report_scopes": ["overall", "layer11", "layer13", "task", "outer_episode"],
        "pooled_metric_may_hide_layer_failure": False,
        "ties_count_as_improvement": False,
    },
    "development_gates": {
        "minimum_support": {
            "zero_per_layer_target": 100,
            "positive_per_layer_target": 100,
            "insufficient_support_disposition": "INCONCLUSIVE_NO_FIT_NO_ROW_DROP",
        },
        "occurrence": {
            "targets": ["step", "transition"],
            "scopes": ["overall", "layer11", "layer13"],
            "brier_skill_strictly_above": 0.0,
            "auroc_strictly_above": 0.5,
            "all_target_scopes_required": True,
        },
        "conditional_count": {
            "comparator": "zero_truncated_binomial_glm",
            "primary": "ordinal_cumulative_link_glm",
            "metric": "positive_only_conditional_nll_ratio",
            "overall_ratio_strictly_below": 1.0,
            "layer_target_scope_count": 4,
            "minimum_strictly_improved_layer_target_scopes": 3,
            "worst_layer_target_ratio_at_most": 1.01,
            "crps_required_to_report": True,
        },
        "expected_fraction": {
            "comparator": "fold_train_layer_mean_hurdle",
            "metric": "raw_sse_ratio",
            "ratio_strictly_below": 1.0,
            "targets": ["step", "transition"],
            "scopes": ["overall", "layer11", "layer13"],
            "all_target_scopes_required": True,
        },
        "group_robustness": {
            "metric": "outer_episode_conditional_count_nll_improvement",
            "primary_value": (
                "equal_mean_of_positive_only_ordinal_nll_over_10_tasks_x_2_layers_"
                "x_2_targets"
            ),
            "comparator_value": (
                "equal_mean_of_positive_only_zt_binomial_nll_over_10_tasks_x_2_"
                "layers_x_2_targets"
            ),
            "missing_positive_task_layer_target_cell": "INCONCLUSIVE",
            "improvement_formula": "primary_value<comparator_value",
            "minimum_improved_outer_episodes": 13,
            "total_outer_episodes": 18,
            "ties_are_not_improvements": True,
            "one_sided_exact_sign_test_p_upper_bound": 0.05,
        },
        "full_pass": (
            "occurrence_pass AND expected_fraction_pass AND group_robustness_pass "
            "AND all_4_layer_target_count_nll_ratios_below_1"
        ),
        "focused_pass_non_deployable": (
            "occurrence_pass AND expected_fraction_pass AND group_robustness_pass "
            "AND count_overall_both_targets_below_1 AND at_least_3_of_4_layer_"
            "target_ratios_below_1 AND worst_layer_target_ratio_at_most_1.01"
        ),
        "failure_disposition": "NEGATIVE_RESULT_FROZEN_NO_CALIBRATION",
    },
    "future_calibration_contract": {
        "stage": "V3-D3",
        "role": "calibration_v2_only",
        "cluster_key": "canonical_task_episode_key",
        "cluster_false_safe_definition": (
            "1_if_any_predicted_safe_call_in_cluster_has_any_mismatch_else_0"
        ),
        "cluster_denominator": "clusters_with_at_least_one_predicted_safe_call",
        "ucb_method": "one_sided_exact_clopper_pearson_on_cluster_events",
        "false_safe_cluster_ucb_confidence": 0.95,
        "false_safe_cluster_ucb_at_most": 0.05,
        "safe_coverage_definition": (
            "clusters_with_at_least_one_predicted_safe_call_divided_by_all_100_"
            "calibration_clusters"
        ),
        "minimum_safe_coverage": 0.10,
        "threshold_candidates": "sorted_unique_finite_frozen_gripper_scores",
        "selection_rule": (
            "maximum_cluster_safe_coverage_subject_to_ucb_then_smaller_threshold"
        ),
        "threshold_selected_on_development": False,
        "independent_test_used_for_threshold": False,
        "always_defer_is_valid": False,
    },
    "tail_veto_contract": {
        "independent_head": True,
        "formula": "route_safe=motion_safe AND tail_ucb_safe AND gripper_safe",
        "gripper_score_may_compensate_tail_failure": False,
        "missing_or_nonfinite_tail": "force_deeper_compute",
        "tail_is_collision_or_success_certificate": False,
        "tail_calibration_before_runtime_required": True,
    },
    "access_boundary": {
        "d1_allowed_reads": [
            "frozen_protocol_metadata",
            "D0_result_and_selection_metadata",
            "legacy_C355_result_json",
        ],
        "d1_forbidden_reads": [
            "development_v2_sample_payload",
            "calibration_v2_sample_payload",
            "independent_test_v2_sample_payload",
            "C3.61_row_level_records",
        ],
        "d2_first_allowed_payload": "development_v2_only",
        "calibration_before_d3_allowed": False,
        "independent_test_before_d7_allowed": False,
    },
    "claim_boundary": {
        "fresh_development_payload_opened": False,
        "calibration_payload_opened": False,
        "independent_test_payload_opened": False,
        "c361_row_payload_opened": False,
        "model_trained": False,
        "checkpoint_selected": False,
        "calibrator_fitted": False,
        "runtime_threshold_selected": False,
        "shadow_rollout_run": False,
        "active_control_run": False,
        "method_performance_claim": False,
        "superiority_claim": False,
    },
    "next_stage": {
        "authorized": "V3-D2_FRESH_DEVELOPMENT_COLLECTION_AND_NESTED_OOF_ONLY",
        "development_role": "development_v2",
        "calibration_or_test_authorized": False,
        "runtime_or_control_authorized": False,
    },
}


def build_protocol_template() -> dict[str, Any]:
    """Return a detached JSON-compatible copy of the frozen D1 contract."""

    return json.loads(json.dumps(_PROTOCOL_TEMPLATE, allow_nan=False))


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value using the protocol's deterministic JSON fingerprint."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolSchemaError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ProtocolSchemaError(f"non-finite JSON value is forbidden: {value}")


def decode_json_bytes(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except GripperV2ProtocolError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolSchemaError(f"invalid JSON in {context}: {error}") from error


def _reject_symlink_components(path: str | Path, *, context: str) -> Path:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ProtocolPathError(f"{context} contains a symlink component")
    return absolute


def resolve_regular_file(
    root: str | Path,
    relative_path: str | Path,
    *,
    suffix: str = ".json",
) -> Path:
    """Resolve a bounded regular file beneath a no-symlink trusted root."""

    root_path = _reject_symlink_components(root, context="trusted root")
    try:
        root_path = root_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ProtocolPathError("trusted root does not exist") from error
    if not root_path.is_dir():
        raise ProtocolPathError("trusted root must be a directory")
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ProtocolPathError("bound path must be non-empty, relative, and contained")
    if relative.suffix.lower() != suffix:
        raise ProtocolPathError(f"bound path must use {suffix}")
    current = root_path
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ProtocolPathError("bound file path contains a symlink")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as error:
        raise ProtocolPathError(f"bound file is missing: {relative}") from error
    try:
        resolved.relative_to(root_path)
    except ValueError as error:
        raise ProtocolPathError("bound file escapes its trusted root") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ProtocolPathError("bound file must be regular")
    return resolved


def read_bounded_regular_file(path: str | Path, *, maximum: int) -> bytes:
    target = _reject_symlink_components(path, context="input file")
    expected = target.stat()
    if not stat.S_ISREG(expected.st_mode) or expected.st_size > maximum:
        raise ProtocolPathError("input file is not a bounded regular file")
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise ProtocolPathError("input file changed identity during open")
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum:
            raise ProtocolPathError("input file changed type or size during open")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ProtocolPathError("input file exceeds its size limit")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path, *, maximum: int = 32 * 1024 * 1024) -> str:
    return hashlib.sha256(read_bounded_regular_file(path, maximum=maximum)).hexdigest()


def _first_difference(actual: Any, expected: Any, path: str = "$") -> str | None:
    if type(actual) is not type(expected):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            return f"{path}: missing={missing}, extra={extra}"
        for key in expected:
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if actual != expected:
        return f"{path}: {actual!r} != {expected!r}"
    return None


def validate_protocol_document(value: Any) -> dict[str, Any]:
    """Require byte-independent semantic equality with the frozen template."""

    if not isinstance(value, dict):
        raise ProtocolSchemaError("protocol document must be an object")
    expected = build_protocol_template()
    difference = _first_difference(value, expected)
    if difference:
        raise ProtocolSchemaError(f"protocol differs from frozen D1 contract: {difference}")
    if FEATURE_DIMENSION != (
        BASE_FEATURE_DIMENSION
        + TEMPORAL_SIGN_DIMENSION
        + TEMPORAL_TRANSITION_DIMENSION
    ):
        raise ProtocolSchemaError("internal feature dimension contract differs")
    if OUTER_FOLD_COUNT != 18 or INNER_FOLD_COUNT != 17:
        raise ProtocolSchemaError("internal grouped fold contract differs")
    return value


def load_protocol(path: str | Path) -> dict[str, Any]:
    target = _reject_symlink_components(path, context="protocol path")
    if target.suffix.lower() != ".json":
        raise ProtocolPathError("protocol must be JSON")
    raw = read_bounded_regular_file(target, maximum=_MAX_PROTOCOL_BYTES)
    return validate_protocol_document(decode_json_bytes(raw, context="D1 protocol"))


def validate_runtime_input_names(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ProtocolSchemaError("runtime input names must be a sequence")
    observed = tuple(names)
    if any(type(item) is not str for item in observed):
        raise ProtocolSchemaError("runtime input names must be strings")
    leaked = FORBIDDEN_RUNTIME_NAMES.intersection(observed)
    if leaked:
        raise ProtocolSchemaError(
            "runtime input leakage is forbidden: " + ", ".join(sorted(leaked))
        )
    if observed != RUNTIME_INPUT_NAMES:
        raise ProtocolSchemaError("runtime input order or membership differs")
    return observed


def gripper_state(value: float | int) -> int:
    """Map a finite action value to the frozen binary proxy; zero maps to one."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetConstructionError("gripper action must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise TargetConstructionError("gripper action must be finite")
    return int(scalar >= 0.0)


def _state_sequence(values: Sequence[float | int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != HORIZON:
        raise TargetConstructionError(f"{name} must contain exactly {HORIZON} values")
    return tuple(gripper_state(value) for value in values)


def transition_pattern(states: Sequence[int]) -> tuple[int, ...]:
    if len(states) != HORIZON or any(type(item) is not int or item not in (0, 1) for item in states):
        raise TargetConstructionError("binary state sequence must be eight 0/1 integers")
    return tuple(int(states[index] != states[index - 1]) for index in range(1, HORIZON))


def construct_gripper_targets(
    candidate_gripper_by_layer: Mapping[int, Sequence[float | int]],
    teacher_gripper: Sequence[float | int],
) -> dict[str, Any]:
    """Construct discrete step/count/timing targets without any ML dependency."""

    if not isinstance(candidate_gripper_by_layer, Mapping) or set(
        candidate_gripper_by_layer
    ) != set(DECISION_LAYERS):
        raise TargetConstructionError("candidate mapping must contain exactly layers 11 and 13")
    teacher_states = _state_sequence(teacher_gripper, name="teacher gripper")
    teacher_transitions = transition_pattern(teacher_states)
    layers: list[dict[str, Any]] = []
    for layer in DECISION_LAYERS:
        candidate_states = _state_sequence(
            candidate_gripper_by_layer[layer], name=f"layer {layer} gripper"
        )
        candidate_transitions = transition_pattern(candidate_states)
        step_bits = tuple(
            int(candidate != teacher)
            for candidate, teacher in zip(candidate_states, teacher_states)
        )
        transition_bits = tuple(
            int(candidate != teacher)
            for candidate, teacher in zip(candidate_transitions, teacher_transitions)
        )
        first_transition = next(
            (index for index, mismatch in enumerate(transition_bits, start=1) if mismatch),
            0,
        )
        step_count = sum(step_bits)
        transition_count = sum(transition_bits)
        layers.append(
            {
                "layer": layer,
                "candidate_state": list(candidate_states),
                "candidate_transition_pattern": list(candidate_transitions),
                "step_mismatch_bits": list(step_bits),
                "transition_mismatch_bits": list(transition_bits),
                "step_count": step_count,
                "transition_count": transition_count,
                "step_occurrence": step_count > 0,
                "transition_occurrence": transition_count > 0,
                "first_transition_mismatch": first_transition,
            }
        )
    return {
        "teacher_state": list(teacher_states),
        "teacher_transition_pattern": list(teacher_transitions),
        "layers": layers,
    }


def validate_task_id(task_id: int) -> int:
    if type(task_id) is not int or not 0 <= task_id < 10:
        raise FoldContractError("task id must be an integer in 0..9")
    return task_id


def outer_fold_id(episode_index: int) -> int:
    if type(episode_index) is not int or episode_index not in DEVELOPMENT_EPISODES:
        raise FoldContractError("development episode must be an integer in 12..29")
    return episode_index - DEVELOPMENT_EPISODES[0]


def outer_validation_episode(outer_fold: int) -> int:
    if type(outer_fold) is not int or not 0 <= outer_fold < OUTER_FOLD_COUNT:
        raise FoldContractError("outer fold must be an integer in 0..17")
    return DEVELOPMENT_EPISODES[outer_fold]


def inner_validation_episodes(outer_fold: int) -> tuple[int, ...]:
    held = outer_validation_episode(outer_fold)
    return tuple(episode for episode in DEVELOPMENT_EPISODES if episode != held)


def inner_fold_id(outer_fold: int, episode_index: int) -> int:
    remaining = inner_validation_episodes(outer_fold)
    if type(episode_index) is not int or episode_index not in remaining:
        raise FoldContractError("inner episode must be development and not outer-held")
    return remaining.index(episode_index)


def grouped_fold_assignment(
    *, task_id: int, episode_index: int, candidate_layer: int
) -> dict[str, Any]:
    """Return a layer-invariant assignment for one canonical episode group."""

    validate_task_id(task_id)
    if type(candidate_layer) is not int or candidate_layer not in DECISION_LAYERS:
        raise FoldContractError("candidate layer must be 11 or 13")
    outer = outer_fold_id(episode_index)
    inner = {
        str(fold): inner_fold_id(fold, episode_index)
        for fold in range(OUTER_FOLD_COUNT)
        if fold != outer
    }
    return {
        "group_key": f"libero_10:task{task_id}:episode{episode_index}",
        "outer_fold": outer,
        "inner_fold_by_outer_train": inner,
    }


def validate_selection_document(
    value: Any,
    *,
    role: str,
    episodes: Sequence[int],
    expected_count: int,
) -> None:
    """Validate a D0-selected role without accepting labels or payload fields."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "suite",
        "role",
        "records",
    }:
        raise ProtocolSchemaError("selection document fields differ")
    if (
        value["schema_version"] != _SELECTION_SCHEMA_VERSION
        or value["suite"] != "libero_10"
        or value["role"] != role
        or not isinstance(value["records"], list)
    ):
        raise ProtocolSchemaError("selection header differs")
    expected = {
        (task, episode) for task in range(10) for episode in episodes
    }
    observed: set[tuple[int, int]] = set()
    for index, record in enumerate(value["records"]):
        if not isinstance(record, dict) or set(record) != {
            "task_id",
            "episode_index",
            "seed",
        }:
            raise ProtocolSchemaError(f"selection record {index} fields differ")
        task = record["task_id"]
        episode = record["episode_index"]
        seed = record["seed"]
        if type(task) is not int or type(episode) is not int or type(seed) is not int:
            raise ProtocolSchemaError(f"selection record {index} types differ")
        key = (task, episode)
        if key in observed:
            raise ProtocolSchemaError("selection repeats a canonical key")
        observed.add(key)
    if len(observed) != expected_count or observed != expected:
        raise ProtocolSchemaError("selection does not match the frozen role grid")


__all__ = [
    "ACTION_DIMENSION",
    "BASE_FEATURE_DIMENSION",
    "CALIBRATION_EPISODES",
    "DECISION_LAYERS",
    "DEVELOPMENT_EPISODES",
    "FEATURE_DIMENSION",
    "FoldContractError",
    "FORBIDDEN_RUNTIME_NAMES",
    "GRIPPER_INDEX",
    "GripperV2ProtocolError",
    "HORIZON",
    "INDEPENDENT_TEST_EPISODES",
    "INNER_FOLD_COUNT",
    "OUTER_FOLD_COUNT",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA_VERSION",
    "PROTOCOL_STATUS",
    "ProtocolPathError",
    "ProtocolSchemaError",
    "RUNTIME_CONTEXT_NAMES",
    "RUNTIME_INPUT_NAMES",
    "TEACHER_LAYER",
    "TargetConstructionError",
    "build_protocol_template",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "construct_gripper_targets",
    "decode_json_bytes",
    "gripper_state",
    "grouped_fold_assignment",
    "inner_fold_id",
    "inner_validation_episodes",
    "load_protocol",
    "outer_fold_id",
    "outer_validation_episode",
    "read_bounded_regular_file",
    "resolve_regular_file",
    "sha256_file",
    "transition_pattern",
    "validate_protocol_document",
    "validate_runtime_input_names",
    "validate_selection_document",
    "validate_task_id",
]
