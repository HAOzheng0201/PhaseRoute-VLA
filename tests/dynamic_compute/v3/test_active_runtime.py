from __future__ import annotations

import numpy as np
import torch

from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.v3.active_runtime import ActivePhaseRouteRuntime
from a1.vla.dynamic_compute.v3.final_router import FinalFiveHeadRouter
from a1.vla.dynamic_compute.v3.gripper_v2_models import FeatureNormalizer
from a1.vla.dynamic_compute.v3.severity_reliability import SeverityWeightedFit


def _head(offset: float) -> SeverityWeightedFit:
    return SeverityWeightedFit(
        normalizer=FeatureNormalizer(
            mean=torch.zeros(97, dtype=torch.float64),
            scale=torch.ones(97, dtype=torch.float64),
        ),
        anchor_score=torch.tensor(
            [[0.10 + offset, 0.02], [0.20 + offset, 0.03]],
            dtype=torch.float64,
        ),
        weight=torch.zeros((2, 97), dtype=torch.float64),
        l2_lambda=0.01,
        final_loss=0.5 + offset,
    )


def _runtime() -> ActivePhaseRouteRuntime:
    router = FinalFiveHeadRouter(
        models=tuple(_head(index * 0.01) for index in range(5)),
        full_threshold=0.5,
        runtime_threshold=0.475,
    )
    phase = PhaseStateEstimator(PhaseEstimatorConfig()).eval()
    return ActivePhaseRouteRuntime(router, phase)


def _context(ordinal: int) -> dict[str, object]:
    return {
        "episode_id": "synthetic:task0:episode0",
        "call_ordinal": ordinal,
        "step_id": 10 + ordinal * 8,
        "task_id": 0,
    }


def _visual_payload(*, malformed: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260822)
    projected = torch.randn(1, 5, 144, 3584, generator=generator)
    positions = torch.arange(720).reshape(1, 5, 144)
    positions[:, 4] = -1
    if malformed:
        projected = projected[:, :, :-1]
    return {
        "projected_features": projected,
        "image_input_idx": positions,
    }


def test_active_runtime_builds_exact_nine_context_and_past_only_history() -> None:
    runtime = _runtime()
    runtime.start_episode("synthetic:task0:episode0")
    instruction = torch.randn(1, 3584)
    proprio = np.arange(8, dtype=np.float32)

    assert runtime.begin_policy_call(
        context=_context(0),
        instruction_summary=instruction,
        normalized_proprio=proprio,
    )
    assert runtime.capture_visual_features(_visual_payload())
    assert runtime._projected_features.dtype == torch.float32
    expected_projected = (
        _visual_payload()["projected_features"].to(torch.float16).float()
    )
    assert torch.equal(runtime._projected_features, expected_projected)
    assert runtime.prepare_policy_call()
    candidate = torch.randn(1, 8, 7, dtype=torch.bfloat16)
    decision = runtime.adapter.consider_candidate(
        11,
        candidate,
        True,
        telemetry_callback=runtime.record_route_event,
    )
    assert decision.should_exit and decision.selected_action is candidate
    assert runtime.commit_selected_action(candidate)

    first = runtime.records[0]
    assert first["prepared"] is True
    assert first["committed"] is True
    assert first["selected_layer"] == 11
    assert first["history_valid_rows"] == 0
    assert tuple(first["runtime_shapes"]) == (
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
    assert "task_id" not in first["runtime_shapes"]

    assert runtime.begin_policy_call(
        context=_context(1),
        instruction_summary=instruction,
        normalized_proprio=proprio + 1,
    )
    assert runtime.capture_visual_features(_visual_payload())
    assert runtime.prepare_policy_call()
    assert runtime.records[1]["history_valid_rows"] == 1


def test_active_runtime_malformed_visual_signal_fails_closed_to_exact_l27() -> None:
    runtime = _runtime()
    runtime.start_episode("synthetic:task0:episode0")
    assert runtime.begin_policy_call(
        context=_context(0),
        instruction_summary=torch.randn(1, 3584),
        normalized_proprio=np.zeros(8, dtype=np.float32),
    )
    assert not runtime.capture_visual_features(_visual_payload(malformed=True))
    assert not runtime.prepare_policy_call()
    l11 = torch.randn(1, 8, 7)
    l13 = torch.randn(1, 8, 7)
    l27 = torch.randn(1, 8, 7, dtype=torch.bfloat16)
    assert not runtime.adapter.consider_candidate(
        11, l11, True, telemetry_callback=runtime.record_route_event
    ).should_exit
    assert not runtime.adapter.consider_candidate(
        13, l13, True, telemetry_callback=runtime.record_route_event
    ).should_exit
    selected = runtime.adapter.select_fallback(
        l27, telemetry_callback=runtime.record_route_event
    )
    assert selected.selected_action is l27
    assert runtime.commit_selected_action(l27)
    assert runtime.records[0]["selected_layer"] == 27
    assert runtime.records[0]["fallback"] is True
    assert runtime.error_count >= 1


def test_episode_reset_prevents_history_leakage() -> None:
    runtime = _runtime()
    instruction = torch.randn(1, 3584)
    for episode in ("synthetic:a", "synthetic:b"):
        runtime.start_episode(episode)
        context = {
            "episode_id": episode,
            "call_ordinal": 0,
            "step_id": 10,
            "task_id": 0,
        }
        assert runtime.begin_policy_call(
            context=context,
            instruction_summary=instruction,
            normalized_proprio=np.zeros(8, dtype=np.float32),
        )
        assert runtime.capture_visual_features(_visual_payload())
        assert runtime.prepare_policy_call()
        action = torch.zeros(1, 8, 7)
        runtime.adapter.consider_candidate(
            11, action, True, telemetry_callback=runtime.record_route_event
        )
        assert runtime.commit_selected_action(action)
        assert runtime.records[-1]["history_valid_rows"] == 0
