from __future__ import annotations

from dataclasses import replace

import pytest

from a1.vla.dynamic_compute.v3.independent_test_aggregate import (
    ArmEvidence,
    D9EAggregationError,
    PairEvidence,
    PhaseCallEvidence,
    TruthEvidence,
    aggregate_independent_test,
    exact_mcnemar_two_sided,
    task_stratified_paired_bootstrap,
)


def _arm(*, phase: bool, success: bool = True) -> ArmEvidence:
    return ArmEvidence(
        success=success,
        environment_steps=100,
        policy_calls=1,
        fm_calls=5 if phase else 10,
        fm_steps=50 if phase else 100,
        exit_layer_counts=((11, 1),) if phase else ((27, 1),),
        policy_wall_seconds=0.1,
        rollout_wall_seconds=1.0,
        policy_latency_ms=(100.0,),
    )


def _dataset() -> tuple[list[PairEvidence], list[TruthEvidence]]:
    pairs: list[PairEvidence] = []
    truths: list[TruthEvidence] = []
    for task in range(10):
        for episode in range(40, 50):
            key = f"libero_10:task{task}:episode{episode}"
            call = PhaseCallEvidence(
                call_ordinal=0,
                step_id=10,
                selected_layer=11,
                head_ranges=(0.01,),
                prepare_latency_ms=0.2,
                fail_closed_errors=0,
            )
            pairs.append(
                PairEvidence(
                    canonical_key=key,
                    task_id=task,
                    episode_index=episode,
                    seed=20_260_851 + task * 10_000 + episode - 40,
                    a1=_arm(phase=False),
                    phase_route=_arm(phase=True),
                    phase_calls=(call,),
                )
            )
            truths.append(
                TruthEvidence(
                    canonical_key=key,
                    task_id=task,
                    episode_index=episode,
                    call_ordinal=0,
                    step_id=10,
                    selected_layer=11,
                    full_action_distance=0.0,
                    full_action_unsafe=False,
                    gripper_unsafe=False,
                    severe_full_action=False,
                    selected_replay_max_abs_error=0.01,
                )
            )
    return pairs, truths


def test_complete_safe_synthetic_result_passes_all_frozen_gates() -> None:
    pairs, truths = _dataset()
    result = aggregate_independent_test(pairs, truths)
    assert result["status"] == "PASS_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST"
    assert all(result["gate_checks"].values())
    assert result["success"]["PhaseRoute_success_rate"] == 1.0
    assert result["efficiency"]["measured_FM_calls_per_policy_call_reduction"] == 0.5
    assert result["safety"]["safe_clusters"] == 100
    assert result["safety"]["false_safe_cluster_exact_CP_UCB95"] == pytest.approx(
        0.029513049607039932
    )
    assert result["head_range"]["fraction_above_1e_6_all_runtime_candidate_rows"] == 1.0
    assert result["latency"]["five_head_router_predict_CPU_latency_ms"] is None


def test_false_safe_and_failure_association_are_clustered_without_causal_claim() -> None:
    pairs, truths = _dataset()
    pairs[0] = replace(pairs[0], phase_route=replace(pairs[0].phase_route, success=False))
    truths[0] = replace(
        truths[0],
        full_action_distance=0.004,
        full_action_unsafe=True,
    )
    result = aggregate_independent_test(pairs, truths)
    association = result["early_exit_failure_association"]
    assert association["PhaseRoute_failure_with_any_early_exit"] == 1
    assert association["PhaseRoute_failure_with_unsafe_early_call"] == 1
    assert association["A1_success_PhaseRoute_failure_with_unsafe_early_call"] == 1
    assert association["causal_interpretation_authorized"] is False
    assert result["safety"]["false_safe_calls"] == 1
    assert result["safety"]["false_safe_clusters"] == 1


def test_truth_must_cover_every_runtime_call_exactly() -> None:
    pairs, truths = _dataset()
    with pytest.raises(D9EAggregationError, match="every PhaseRoute call"):
        aggregate_independent_test(pairs, truths[:-1])


def test_runtime_candidate_head_range_denominator_includes_missing_rows() -> None:
    pairs, truths = _dataset()
    replacement_call = replace(pairs[0].phase_calls[0], head_ranges=(None,))
    pairs[0] = replace(pairs[0], phase_calls=(replacement_call,))
    result = aggregate_independent_test(pairs, truths)
    assert result["head_range"]["candidate_rows"] == 100
    assert result["head_range"]["finite_rows"] == 99
    assert result["head_range"]["missing_rows"] == 1
    assert result["head_range"]["rows_above_1e_6"] == 99


def test_bootstrap_is_seeded_task_stratified_and_uses_linear_fifth_percentile() -> None:
    pairs, _ = _dataset()
    for index, pair in enumerate(pairs):
        if pair.episode_index == 49:
            pairs[index] = replace(
                pair,
                phase_route=replace(pair.phase_route, success=False),
            )
    first = task_stratified_paired_bootstrap(pairs)
    second = task_stratified_paired_bootstrap(pairs)
    assert first == second
    assert first["resamples"] == 100_000
    assert first["seed"] == 60_260_821
    assert first["quantile_method"] == "numpy_linear"
    assert first["lower_percentile"] == 0.05
    assert first["lower_bound"] == pytest.approx(-0.15)


def test_exact_mcnemar_is_two_sided_equality_test() -> None:
    assert exact_mcnemar_two_sided(0, 0) == 1.0
    assert exact_mcnemar_two_sided(5, 0) == pytest.approx(0.0625)
    assert exact_mcnemar_two_sided(3, 3) == 1.0
