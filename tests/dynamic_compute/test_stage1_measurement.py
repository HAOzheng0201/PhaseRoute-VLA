from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.fixed_layer_controller import (
    FixedLayerFlowMatchingController,
)
from a1.vla.dynamic_compute.stage1_measurement import (
    Stage1RuntimeProbe,
    summarize_stage1_records,
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
    return ActivePhaseRouteRuntime(
        router, PhaseStateEstimator(PhaseEstimatorConfig()).eval()
    )


def test_external_probe_preserves_exact_selected_action_and_records_components() -> None:
    runtime = _runtime()
    runtime.start_episode("stage1:task0:episode0")
    probe = Stage1RuntimeProbe(runtime)
    context = {
        "episode_id": "stage1:task0:episode0",
        "call_ordinal": 0,
        "step_id": 10,
        "task_id": 0,
    }
    probe.start_call(context)
    assert runtime.begin_policy_call(
        context=context,
        instruction_summary=torch.zeros(1, 3584),
        normalized_proprio=np.zeros(8, dtype=np.float32),
    )
    projected = torch.zeros(1, 5, 144, 3584)
    positions = torch.arange(720).reshape(1, 5, 144)
    positions[:, 4] = -1
    assert runtime.capture_visual_features(
        {"projected_features": projected, "image_input_idx": positions}
    )
    assert runtime.prepare_policy_call()
    action = torch.zeros(1, 8, 7, dtype=torch.bfloat16)
    decision = runtime.adapter.consider_candidate(
        11, action, True, telemetry_callback=runtime.record_route_event
    )
    assert decision.selected_action is action
    assert runtime.commit_selected_action(action)
    measured = probe.finish_call()

    assert runtime.records[-1]["selected_layer"] == 11
    assert set(measured["components"]) == {
        "adapter_begin",
        "candidate_route",
        "phase_estimator",
        "router_predict",
        "runtime_begin",
        "runtime_commit",
        "runtime_prepare",
        "visual_capture",
    }
    assert measured["components"]["candidate_route"][0]["layer"] == 11
    assert measured["components"]["candidate_route"][0]["should_exit"] is True
    assert all(
        event["latency_ms"] >= 0.0
        for events in measured["components"].values()
        for event in events
    )


def test_stage1_summary_is_json_safe_and_uses_nearest_rank() -> None:
    summary = summarize_stage1_records(
        (
            {
                "selected_layer": 11,
                "error": None,
                "action_finite": True,
                "policy_wall_latency_ms": 10.0,
                "policy_cuda_event_latency_ms": 8.0,
                "components": {
                    "router_predict": [{"latency_ms": 0.2}],
                },
            },
            {
                "selected_layer": 27,
                "error": None,
                "action_finite": True,
                "policy_wall_latency_ms": 20.0,
                "policy_cuda_event_latency_ms": None,
                "components": {
                    "router_predict": [{"latency_ms": 0.4}],
                },
            },
        )
    )
    assert summary["records"] == 2
    assert summary["records_with_nonfinite_actions"] == 0
    assert summary["records_without_action_audit"] == 0
    assert summary["selected_layers"] == {"11": 1, "13": 0, "27": 1}
    assert summary["latency_ms"]["policy_wall"]["mean"] == 15.0
    assert summary["latency_ms"]["router_predict"]["p95"] == 0.4
    json.dumps(summary, allow_nan=False)


def test_fixed_layer_overlay_writes_selected_layer_without_dynamic_controller(
    monkeypatch, tmp_path
) -> None:
    from robot_experiments.libero import stage1_vla_utils as module

    expected = [np.zeros(7, dtype=np.float32) for _ in range(8)]
    captured = {}

    def fake_frozen(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return expected

    output_path = tmp_path / "measurement.jsonl"
    monkeypatch.setattr(module, "get_frozen_d9_vla_action", fake_frozen)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(module.STAGE1_TIMING_ENV, str(output_path))
    cfg = SimpleNamespace(exit_layer_id=13)
    controller = SimpleNamespace(stage1_fixed_layer=True, layer=13)
    result = module.get_vla_action(
        cfg,
        object(),
        torch.device("cpu"),
        {},
        "instruction",
        exit_controller=controller,
        telemetry_context={"episode_id": "fixed", "step_id": 10},
    )

    assert result is expected
    assert captured["exit_controller"] is controller
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["mode"] == "fixed_l13"
    assert record["selected_layer"] == 13
    assert record["measurement_is_control_input"] is False
    assert record["d9_protected_source_modified"] is False
    assert record["action_finite"] is True
    assert record["action_shape"] == [8, 7]
    assert record["error"] is None


def test_fixed_flow_matching_controller_generates_one_exact_target_action() -> None:
    action = torch.arange(56, dtype=torch.float32).reshape(1, 8, 7)

    class FakeFlowModel:
        config = SimpleNamespace(
            action_head="flow_matching",
            n_layers=28,
            num_diffusion_inference_steps=10,
        )

        def __init__(self):
            self.calls = []

        def predict_actions_flow_matching(self, kvs, proprio, pos_offset, **kwargs):
            self.calls.append((kvs, proprio, pos_offset, kwargs))
            return action

    model = FakeFlowModel()
    controller = FixedLayerFlowMatchingController(model, 13)
    controller.set_timestep(10)
    events = []

    assert controller([object()] * 13, 12, None, 0, 0, 3) == (False, None)
    should_exit, selected = controller(
        [object()] * 14,
        13,
        torch.zeros(1, 1, 8),
        0,
        0,
        torch.tensor([3]),
        telemetry_callback=lambda name, payload: events.append((name, payload)),
    )

    assert should_exit is True
    assert selected is action
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 14
    assert events == [
        (
            "exit_candidate",
            {
                "layer_idx": 13,
                "evaluated": True,
                "should_exit": True,
                "fixed_layer": True,
                "action_delta": None,
                "threshold": None,
                "fm_calls": 1,
                "fm_steps": 10,
            },
        )
    ]
