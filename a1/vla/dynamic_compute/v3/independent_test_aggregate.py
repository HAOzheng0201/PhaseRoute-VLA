"""Pure, CPU-only aggregation for the frozen V3-D9 independent test.

This module contains no filesystem access and never constructs a LIBERO
environment.  The one-shot D9E runner authenticates raw evidence, converts it
to the dataclasses below, and calls :func:`aggregate_independent_test` exactly
once.  Statistical definitions in this file are part of the frozen runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .gripper_v2_calibration import clopper_pearson_upper


D9_TASK_IDS = tuple(range(10))
D9_EPISODE_INDICES = tuple(range(40, 50))
D9_LAYERS = (11, 13, 27)
D9_PAIRS = 100
D9_BOOTSTRAP_RESAMPLES = 100_000
D9_BOOTSTRAP_SEED = 60_260_821
D9_BOOTSTRAP_PERCENTILE = 0.05
D9_HEAD_RANGE_EPSILON = 1.0e-6


class D9EAggregationError(ValueError):
    """Raised when complete D9 evidence or frozen statistics differ."""


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise D9EAggregationError(f"{name} must be bool")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise D9EAggregationError(f"{name} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D9EAggregationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise D9EAggregationError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ArmEvidence:
    success: bool
    environment_steps: int
    policy_calls: int
    fm_calls: int
    fm_steps: int
    exit_layer_counts: tuple[tuple[int, int], ...]
    policy_wall_seconds: float
    rollout_wall_seconds: float
    policy_latency_ms: tuple[float, ...]

    def validate(self, *, phase_route: bool) -> None:
        _strict_bool(self.success, "success")
        _nonnegative_int(self.environment_steps, "environment_steps")
        calls = _nonnegative_int(self.policy_calls, "policy_calls")
        _nonnegative_int(self.fm_calls, "fm_calls")
        _nonnegative_int(self.fm_steps, "fm_steps")
        _finite_nonnegative(self.policy_wall_seconds, "policy_wall_seconds")
        _finite_nonnegative(self.rollout_wall_seconds, "rollout_wall_seconds")
        if calls <= 0 or len(self.policy_latency_ms) != calls:
            raise D9EAggregationError("policy latency/call accounting differs")
        for value in self.policy_latency_ms:
            _finite_nonnegative(value, "policy_latency_ms")
        counts = dict(self.exit_layer_counts)
        if len(counts) != len(self.exit_layer_counts) or any(
            type(layer) is not int
            or type(count) is not int
            or count < 0
            for layer, count in self.exit_layer_counts
        ):
            raise D9EAggregationError("exit-layer counts are invalid")
        allowed = set(D9_LAYERS if phase_route else range(1, 28, 2))
        if not set(counts).issubset(allowed) or sum(counts.values()) != calls:
            raise D9EAggregationError("exit-layer accounting differs")


@dataclass(frozen=True)
class PhaseCallEvidence:
    call_ordinal: int
    step_id: int
    selected_layer: int
    head_ranges: tuple[float | None, ...]
    prepare_latency_ms: float
    fail_closed_errors: int

    def validate(self) -> None:
        ordinal = _nonnegative_int(self.call_ordinal, "call_ordinal")
        del ordinal
        _nonnegative_int(self.step_id, "step_id")
        if self.selected_layer not in D9_LAYERS:
            raise D9EAggregationError("PhaseRoute selected layer differs")
        expected_rows = 1 if self.selected_layer == 11 else 2
        if len(self.head_ranges) != expected_rows:
            raise D9EAggregationError("runtime candidate-row count differs")
        for value in self.head_ranges:
            if value is not None:
                _finite_nonnegative(value, "full_action_head_range")
        _finite_nonnegative(self.prepare_latency_ms, "prepare_latency_ms")
        _nonnegative_int(self.fail_closed_errors, "fail_closed_errors")


@dataclass(frozen=True)
class PairEvidence:
    canonical_key: str
    task_id: int
    episode_index: int
    seed: int
    a1: ArmEvidence
    phase_route: ArmEvidence
    phase_calls: tuple[PhaseCallEvidence, ...]

    def validate(self) -> None:
        if (
            type(self.canonical_key) is not str
            or self.canonical_key
            != f"libero_10:task{self.task_id}:episode{self.episode_index}"
            or self.task_id not in D9_TASK_IDS
            or self.episode_index not in D9_EPISODE_INDICES
            or self.seed
            != 20_260_851 + self.task_id * 10_000 + self.episode_index - 40
        ):
            raise D9EAggregationError("pair identity differs from frozen schedule")
        self.a1.validate(phase_route=False)
        self.phase_route.validate(phase_route=True)
        if len(self.phase_calls) != self.phase_route.policy_calls:
            raise D9EAggregationError("PhaseRoute runtime/call accounting differs")
        previous_step = -1
        for ordinal, call in enumerate(self.phase_calls):
            call.validate()
            if call.call_ordinal != ordinal or call.step_id <= previous_step:
                raise D9EAggregationError("PhaseRoute call ordering differs")
            previous_step = call.step_id
        runtime_counts = {layer: 0 for layer in D9_LAYERS}
        for call in self.phase_calls:
            runtime_counts[call.selected_layer] += 1
        arm_counts = dict(self.phase_route.exit_layer_counts)
        if any(runtime_counts[layer] != arm_counts.get(layer, 0) for layer in D9_LAYERS):
            raise D9EAggregationError("PhaseRoute arm/runtime layer counts differ")


@dataclass(frozen=True)
class TruthEvidence:
    canonical_key: str
    task_id: int
    episode_index: int
    call_ordinal: int
    step_id: int
    selected_layer: int
    full_action_distance: float
    full_action_unsafe: bool
    gripper_unsafe: bool
    severe_full_action: bool
    selected_replay_max_abs_error: float

    def validate(self) -> None:
        if (
            type(self.canonical_key) is not str
            or self.canonical_key
            != f"libero_10:task{self.task_id}:episode{self.episode_index}"
            or self.task_id not in D9_TASK_IDS
            or self.episode_index not in D9_EPISODE_INDICES
            or self.selected_layer not in D9_LAYERS
        ):
            raise D9EAggregationError("truth identity differs")
        _nonnegative_int(self.call_ordinal, "truth call_ordinal")
        _nonnegative_int(self.step_id, "truth step_id")
        distance = _finite_nonnegative(
            self.full_action_distance, "full_action_distance"
        )
        _strict_bool(self.full_action_unsafe, "full_action_unsafe")
        _strict_bool(self.gripper_unsafe, "gripper_unsafe")
        _strict_bool(self.severe_full_action, "severe_full_action")
        _finite_nonnegative(
            self.selected_replay_max_abs_error,
            "selected_replay_max_abs_error",
        )
        threshold = 0.00390625
        if (
            self.full_action_unsafe != (distance > threshold)
            or self.severe_full_action != (distance > 4.0 * threshold)
            or (self.severe_full_action and not self.full_action_unsafe)
        ):
            raise D9EAggregationError("truth threshold semantics differ")


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise D9EAggregationError("distribution input is empty")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise D9EAggregationError("distribution input must be finite [N]")
    levels = np.asarray([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    quantiles = np.quantile(array, levels, method="linear")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sum": float(array.sum()),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def exact_mcnemar_two_sided(a1_only: int, phase_only: int) -> float:
    """Exact two-sided binomial McNemar p-value (equality, not NI)."""

    left = _nonnegative_int(a1_only, "a1_only")
    right = _nonnegative_int(phase_only, "phase_only")
    discordant = left + right
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left, right) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def task_stratified_paired_bootstrap(
    pairs: Sequence[PairEvidence],
) -> dict[str, Any]:
    """Frozen task-stratified paired bootstrap with NumPy linear quantile."""

    by_task: list[np.ndarray] = []
    for task in D9_TASK_IDS:
        ordered = sorted(
            (pair for pair in pairs if pair.task_id == task),
            key=lambda pair: pair.episode_index,
        )
        if [pair.episode_index for pair in ordered] != list(D9_EPISODE_INDICES):
            raise D9EAggregationError("bootstrap task cells differ")
        by_task.append(
            np.asarray(
                [
                    int(pair.phase_route.success) - int(pair.a1.success)
                    for pair in ordered
                ],
                dtype=np.int8,
            )
        )
    rng = np.random.default_rng(D9_BOOTSTRAP_SEED)
    samples = np.zeros(D9_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for differences in by_task:
        indices = rng.integers(
            0,
            differences.size,
            size=(D9_BOOTSTRAP_RESAMPLES, differences.size),
        )
        samples += differences[indices].sum(axis=1)
    samples /= D9_PAIRS
    lower = float(
        np.quantile(samples, D9_BOOTSTRAP_PERCENTILE, method="linear")
    )
    return {
        "resamples": D9_BOOTSTRAP_RESAMPLES,
        "seed": D9_BOOTSTRAP_SEED,
        "resample_unit": "paired_episode_within_each_task",
        "statistic": "PhaseRoute_success_rate_minus_A1_success_rate",
        "quantile_method": "numpy_linear",
        "lower_percentile": D9_BOOTSTRAP_PERCENTILE,
        "lower_bound": lower,
        "sample_mean": float(samples.mean()),
        "sample_standard_deviation": float(samples.std(ddof=0)),
    }


def aggregate_independent_test(
    pairs: Sequence[PairEvidence], truths: Sequence[TruthEvidence]
) -> dict[str, Any]:
    """Validate and aggregate the complete frozen 100-pair D9 evidence."""

    if len(pairs) != D9_PAIRS:
        raise D9EAggregationError("D9E requires exactly 100 complete pairs")
    pair_by_key: dict[str, PairEvidence] = {}
    expected_cells = {
        (task, episode)
        for task in D9_TASK_IDS
        for episode in D9_EPISODE_INDICES
    }
    for pair in pairs:
        pair.validate()
        if pair.canonical_key in pair_by_key:
            raise D9EAggregationError("duplicate pair identity")
        pair_by_key[pair.canonical_key] = pair
    if {(p.task_id, p.episode_index) for p in pairs} != expected_cells:
        raise D9EAggregationError("D9E pair coverage differs")

    truth_by_call: dict[tuple[str, int], TruthEvidence] = {}
    for truth in truths:
        truth.validate()
        key = (truth.canonical_key, truth.call_ordinal)
        if key in truth_by_call:
            raise D9EAggregationError("duplicate same-noise truth call")
        truth_by_call[key] = truth
    expected_truth_calls = {
        (pair.canonical_key, call.call_ordinal)
        for pair in pairs
        for call in pair.phase_calls
    }
    if set(truth_by_call) != expected_truth_calls:
        raise D9EAggregationError("D9D truth does not cover every PhaseRoute call")
    for pair in pairs:
        for call in pair.phase_calls:
            truth = truth_by_call[(pair.canonical_key, call.call_ordinal)]
            if (
                truth.task_id != pair.task_id
                or truth.episode_index != pair.episode_index
                or truth.step_id != call.step_id
                or truth.selected_layer != call.selected_layer
            ):
                raise D9EAggregationError("D9C runtime and D9D truth differ")

    a1_successes = sum(pair.a1.success for pair in pairs)
    phase_successes = sum(pair.phase_route.success for pair in pairs)
    success_difference = (phase_successes - a1_successes) / D9_PAIRS
    paired_table = {
        "both_success": sum(p.a1.success and p.phase_route.success for p in pairs),
        "A1_success_PhaseRoute_failure": sum(
            p.a1.success and not p.phase_route.success for p in pairs
        ),
        "A1_failure_PhaseRoute_success": sum(
            not p.a1.success and p.phase_route.success for p in pairs
        ),
        "both_failure": sum(
            not p.a1.success and not p.phase_route.success for p in pairs
        ),
    }
    bootstrap = task_stratified_paired_bootstrap(pairs)

    phase_calls = sum(pair.phase_route.policy_calls for pair in pairs)
    a1_calls = sum(pair.a1.policy_calls for pair in pairs)
    phase_fm = sum(pair.phase_route.fm_calls for pair in pairs)
    a1_fm = sum(pair.a1.fm_calls for pair in pairs)
    if phase_calls <= 0 or a1_calls <= 0 or phase_fm <= 0 or a1_fm <= 0:
        raise D9EAggregationError("FM-call efficiency denominators must be positive")
    phase_fm_per_call = phase_fm / phase_calls
    a1_fm_per_call = a1_fm / a1_calls
    fm_reduction = 1.0 - phase_fm_per_call / a1_fm_per_call
    layer_counts = {
        layer: sum(
            call.selected_layer == layer
            for pair in pairs
            for call in pair.phase_calls
        )
        for layer in D9_LAYERS
    }
    early_calls = layer_counts[11] + layer_counts[13]
    early_fraction = early_calls / phase_calls

    early_keys: set[str] = set()
    false_keys: set[str] = set()
    false_full_keys: set[str] = set()
    severe_keys: set[str] = set()
    false_gripper_calls = 0
    false_safe_calls = 0
    for truth in truths:
        if truth.selected_layer == 27:
            continue
        early_keys.add(truth.canonical_key)
        unsafe = truth.full_action_unsafe or truth.gripper_unsafe
        if unsafe:
            false_keys.add(truth.canonical_key)
            false_safe_calls += 1
        if truth.full_action_unsafe:
            false_full_keys.add(truth.canonical_key)
        if truth.gripper_unsafe:
            false_gripper_calls += 1
        if truth.severe_full_action:
            severe_keys.add(truth.canonical_key)
    false_ucb = clopper_pearson_upper(len(false_keys), len(early_keys))

    head_ranges = [
        value
        for pair in pairs
        for call in pair.phase_calls
        for value in call.head_ranges
    ]
    finite_head_ranges = [value for value in head_ranges if value is not None]
    nondegenerate = sum(
        value is not None and value > D9_HEAD_RANGE_EPSILON
        for value in head_ranges
    )
    nondegenerate_fraction = nondegenerate / len(head_ranges)
    prepare_latencies = [
        call.prepare_latency_ms for pair in pairs for call in pair.phase_calls
    ]

    per_task: dict[str, Any] = {}
    for task in D9_TASK_IDS:
        task_pairs = [pair for pair in pairs if pair.task_id == task]
        task_keys = {pair.canonical_key for pair in task_pairs}
        task_phase_calls = sum(pair.phase_route.policy_calls for pair in task_pairs)
        task_a1_calls = sum(pair.a1.policy_calls for pair in task_pairs)
        task_phase_fm = sum(pair.phase_route.fm_calls for pair in task_pairs)
        task_a1_fm = sum(pair.a1.fm_calls for pair in task_pairs)
        if min(task_phase_calls, task_a1_calls, task_phase_fm, task_a1_fm) <= 0:
            raise D9EAggregationError("per-task FM efficiency denominator differs")
        task_layer_counts = {
            layer: sum(
                call.selected_layer == layer
                for pair in task_pairs
                for call in pair.phase_calls
            )
            for layer in D9_LAYERS
        }
        task_phase_success = sum(pair.phase_route.success for pair in task_pairs)
        task_a1_success = sum(pair.a1.success for pair in task_pairs)
        per_task[str(task)] = {
            "pairs": len(task_pairs),
            "A1_successes": task_a1_success,
            "A1_success_rate": task_a1_success / len(task_pairs),
            "PhaseRoute_successes": task_phase_success,
            "PhaseRoute_success_rate": task_phase_success / len(task_pairs),
            "PhaseRoute_minus_A1_successes": task_phase_success - task_a1_success,
            "A1_policy_calls": task_a1_calls,
            "PhaseRoute_policy_calls": task_phase_calls,
            "A1_FM_calls": task_a1_fm,
            "PhaseRoute_FM_calls": task_phase_fm,
            "A1_FM_calls_per_policy_call": task_a1_fm / task_a1_calls,
            "PhaseRoute_FM_calls_per_policy_call": (
                task_phase_fm / task_phase_calls
            ),
            "FM_calls_per_policy_call_reduction": 1.0
            - (task_phase_fm / task_phase_calls) / (task_a1_fm / task_a1_calls),
            "L11_calls": task_layer_counts[11],
            "L13_calls": task_layer_counts[13],
            "L27_calls": task_layer_counts[27],
            "early_exit_calls": task_layer_counts[11] + task_layer_counts[13],
            "early_exit_fraction": (
                task_layer_counts[11] + task_layer_counts[13]
            )
            / task_phase_calls,
            "safe_clusters": len(early_keys & task_keys),
            "false_safe_clusters": len(false_keys & task_keys),
            "false_full_action_clusters": len(false_full_keys & task_keys),
            "severe_false_full_action_clusters": len(severe_keys & task_keys),
        }

    phase_failed = {
        pair.canonical_key for pair in pairs if not pair.phase_route.success
    }
    a1_won_phase_lost = {
        pair.canonical_key
        for pair in pairs
        if pair.a1.success and not pair.phase_route.success
    }
    association = {
        "PhaseRoute_failure_rollouts": len(phase_failed),
        "PhaseRoute_failure_with_any_early_exit": len(phase_failed & early_keys),
        "PhaseRoute_failure_with_unsafe_early_call": len(phase_failed & false_keys),
        "A1_success_PhaseRoute_failure_pairs": len(a1_won_phase_lost),
        "A1_success_PhaseRoute_failure_with_any_early_exit": len(
            a1_won_phase_lost & early_keys
        ),
        "A1_success_PhaseRoute_failure_with_unsafe_early_call": len(
            a1_won_phase_lost & false_keys
        ),
        "causal_interpretation_authorized": False,
    }

    gate_checks = {
        "all_100_pairs_complete": len(pairs) == 100,
        "all_10_pairs_per_task_complete": all(
            per_task[str(task)]["pairs"] == 10 for task in D9_TASK_IDS
        ),
        "all_200_rollouts_complete": len(pairs) * 2 == 200,
        "PhaseRoute_success_rate_at_least_0_75": phase_successes / 100 >= 0.75,
        "PhaseRoute_minus_A1_success_rate_at_least_minus_0_05": (
            success_difference >= -0.05
        ),
        "PhaseRoute_minus_A1_successes_each_task_at_least_minus_2": all(
            per_task[str(task)]["PhaseRoute_minus_A1_successes"] >= -2
            for task in D9_TASK_IDS
        ),
        "task_stratified_bootstrap_lower_bound_at_least_minus_0_10": (
            bootstrap["lower_bound"] >= -0.10
        ),
        "PhaseRoute_early_exit_call_fraction_at_least_0_10": (
            early_fraction >= 0.10
        ),
        "all_10_tasks_have_nonzero_PhaseRoute_early_exit_calls": all(
            per_task[str(task)]["early_exit_calls"] > 0 for task in D9_TASK_IDS
        ),
        "PhaseRoute_safe_clusters_at_least_60": len(early_keys) >= 60,
        "PhaseRoute_safe_clusters_each_task_at_least_2": all(
            per_task[str(task)]["safe_clusters"] >= 2 for task in D9_TASK_IDS
        ),
        "false_safe_cluster_exact_CP_UCB95_at_most_0_05": false_ucb <= 0.05,
        "false_full_action_clusters_at_most_2": len(false_full_keys) <= 2,
        "false_gripper_calls_equal_0": false_gripper_calls == 0,
        "severe_false_full_action_clusters_equal_0": len(severe_keys) == 0,
        "ensemble_head_range_above_1e_6_rows_fraction_at_least_0_01": (
            nondegenerate_fraction >= 0.01
        ),
        "measured_FM_calls_per_policy_call_reduction_at_least_0_25": (
            fm_reduction >= 0.25
        ),
        "always_defer_rejected": early_calls > 0,
    }
    status = (
        "PASS_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST"
        if all(gate_checks.values())
        else "NEGATIVE_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST"
    )
    return {
        "status": status,
        "success": {
            "pairs": D9_PAIRS,
            "A1_successes": a1_successes,
            "A1_success_rate": a1_successes / D9_PAIRS,
            "PhaseRoute_successes": phase_successes,
            "PhaseRoute_success_rate": phase_successes / D9_PAIRS,
            "PhaseRoute_minus_A1_successes": phase_successes - a1_successes,
            "PhaseRoute_minus_A1_success_rate": success_difference,
            "paired_outcome_2x2": paired_table,
            "exact_McNemar_two_sided_p_value_for_equality": (
                exact_mcnemar_two_sided(
                    paired_table["A1_success_PhaseRoute_failure"],
                    paired_table["A1_failure_PhaseRoute_success"],
                )
            ),
            "McNemar_is_noninferiority_test": False,
            "task_stratified_paired_bootstrap": bootstrap,
        },
        "efficiency": {
            "A1_policy_calls": a1_calls,
            "PhaseRoute_policy_calls": phase_calls,
            "A1_FM_calls": a1_fm,
            "PhaseRoute_FM_calls": phase_fm,
            "A1_FM_calls_per_policy_call": a1_fm_per_call,
            "PhaseRoute_FM_calls_per_policy_call": phase_fm_per_call,
            "measured_FM_calls_per_policy_call_reduction": fm_reduction,
            "L11_calls": layer_counts[11],
            "L13_calls": layer_counts[13],
            "L27_calls": layer_counts[27],
            "PhaseRoute_early_exit_calls": early_calls,
            "PhaseRoute_early_exit_call_fraction": early_fraction,
            "router_latency_included_in_FM_metric": False,
        },
        "safety": {
            "truth_calls": len(truths),
            "safe_clusters": len(early_keys),
            "safe_cluster_keys": sorted(early_keys),
            "false_safe_calls": false_safe_calls,
            "false_safe_clusters": len(false_keys),
            "false_safe_cluster_keys": sorted(false_keys),
            "false_safe_cluster_rate": (
                len(false_keys) / len(early_keys) if early_keys else None
            ),
            "false_safe_cluster_exact_CP_UCB95": false_ucb,
            "false_full_action_clusters": len(false_full_keys),
            "false_full_action_cluster_keys": sorted(false_full_keys),
            "false_gripper_calls": false_gripper_calls,
            "severe_false_full_action_clusters": len(severe_keys),
            "severe_false_full_action_cluster_keys": sorted(severe_keys),
            "layer27_role": "same_noise_consistency_teacher_only",
        },
        "head_range": {
            "candidate_rows": len(head_ranges),
            "finite_rows": len(finite_head_ranges),
            "missing_rows": len(head_ranges) - len(finite_head_ranges),
            "rows_above_1e_6": nondegenerate,
            "fraction_above_1e_6_all_runtime_candidate_rows": (
                nondegenerate_fraction
            ),
            "finite_distribution": (
                _distribution(finite_head_ranges) if finite_head_ranges else None
            ),
        },
        "latency": {
            "A1_policy_call_latency_ms": _distribution(
                [value for pair in pairs for value in pair.a1.policy_latency_ms]
            ),
            "PhaseRoute_policy_call_latency_ms": _distribution(
                [
                    value
                    for pair in pairs
                    for value in pair.phase_route.policy_latency_ms
                ]
            ),
            "PhaseRoute_context_prepare_CPU_latency_ms": _distribution(
                prepare_latencies
            ),
            "five_head_router_predict_CPU_latency_ms": None,
            "five_head_router_predict_latency_not_instrumented_online": True,
            "A1_rollout_wall_seconds": _distribution(
                [pair.a1.rollout_wall_seconds for pair in pairs]
            ),
            "PhaseRoute_rollout_wall_seconds": _distribution(
                [pair.phase_route.rollout_wall_seconds for pair in pairs]
            ),
        },
        "runtime_integrity": {
            "PhaseRoute_fail_closed_error_events": sum(
                call.fail_closed_errors
                for pair in pairs
                for call in pair.phase_calls
            ),
            "all_PhaseRoute_calls_have_D9D_truth": True,
        },
        "early_exit_failure_association": association,
        "per_task": per_task,
        "gate_checks": gate_checks,
        "all_primary_gates_pass": all(gate_checks.values()),
    }


__all__ = [
    "ArmEvidence",
    "D9EAggregationError",
    "PairEvidence",
    "PhaseCallEvidence",
    "TruthEvidence",
    "aggregate_independent_test",
    "exact_mcnemar_two_sided",
    "task_stratified_paired_bootstrap",
]
