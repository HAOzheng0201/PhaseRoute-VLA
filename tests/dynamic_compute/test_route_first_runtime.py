from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from a1.vla.dynamic_compute.route_first_router import (
    RouteFirstAffineHead,
    RouteFirstOrdinalRouter,
)
from a1.vla.dynamic_compute.route_first_controller import RouteFirstExitController
from a1.vla.dynamic_compute.route_first_runtime import (
    ROUTE_FIRST_CALIBRATED_ROUTER_SHA256,
    ROUTE_FIRST_STAGE7_HOLDOUT_SHA256,
    RouteFirstRuntimeAdapter,
    RouteFirstRuntimeArtifacts,
    load_route_first_active_runtime,
    route_first_target_layers,
)
from a1.vla.dynamic_compute.stage1_measurement import Stage1RuntimeProbe
from a1.vla.dynamic_compute.v3.active_runtime import ActiveRuntimeArtifacts
from a1.vla.value_net import ActionValueNet


REPO_ROOT = Path(__file__).resolve().parents[2]


def _head(bias: float) -> RouteFirstAffineHead:
    return RouteFirstAffineHead(
        np.zeros(199, dtype=np.float64),
        bias,
        pca_rank=1,
        l2=1.0,
        iterations=1,
    )


def _router(*, safe13: bool) -> RouteFirstOrdinalRouter:
    return RouteFirstOrdinalRouter(
        _head(-10.0),
        _head(10.0 if safe13 else -10.0),
    )


def _metadata() -> dict[str, object]:
    return {
        "enabled11": False,
        "threshold11": 0.98,
        "enabled13": True,
        "threshold13": 0.9,
        "engineering_holdout_authorized": True,
        "active_control_authorized": False,
    }


def _artifacts() -> RouteFirstRuntimeArtifacts:
    context = ActiveRuntimeArtifacts(
        router_path="context-router.pt",
        router_sha256="a" * 64,
        phase_checkpoint_path="phase.pt",
        phase_checkpoint_sha256="b" * 64,
        phase_state_sha256="c" * 64,
    )
    return RouteFirstRuntimeArtifacts(
        calibrated_router_path="route-first.npz",
        calibrated_router_sha256=ROUTE_FIRST_CALIBRATED_ROUTER_SHA256,
        stage7_holdout_path="stage7.json",
        stage7_holdout_sha256=ROUTE_FIRST_STAGE7_HOLDOUT_SHA256,
        stage7_status="PASS_ENGINEERING_HOLDOUT_RUNTIME_INTEGRATION_READY",
        v3_context_artifacts=context,
    )


def _runtime_context() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260825)
    return {
        "instruction_summary": torch.randn(1, 3584, generator=generator),
        "vision_crop_summary": torch.randn(1, 5, 3584, generator=generator),
        "vision_crop_mask": torch.ones(1, 5, dtype=torch.bool),
        "phase_embedding": torch.randn(1, 128, generator=generator),
        "phase_scalars": torch.rand(1, 3, generator=generator),
        "normalized_proprio": torch.randn(1, 8, generator=generator),
        "proprio_history": torch.zeros(1, 8, 8),
        "action_history": torch.zeros(1, 8, 8, 7),
        "history_mask": torch.zeros(1, 8, dtype=torch.bool),
    }


def _adapter(*, safe13: bool) -> RouteFirstRuntimeAdapter:
    return RouteFirstRuntimeAdapter(
        _router(safe13=safe13),
        _metadata(),
        _artifacts(),
    )


def test_action_free_router_locks_only_l13_or_l27_and_never_l11() -> None:
    features = np.zeros((2, 199), dtype=np.float32)
    scores, layers = route_first_target_layers(
        _router(safe13=True),
        features,
        enabled11=False,
        enabled13=True,
        threshold13=0.9,
    )
    assert scores.shape == (2, 2)
    assert layers.tolist() == [13, 13]

    adapter = _adapter(safe13=False)
    adapter.begin_policy_call(_runtime_context())
    assert adapter.target_layer == 27
    assert adapter.fail_reason is None


def test_malformed_context_and_router_mutation_fail_closed_to_l27() -> None:
    adapter = _adapter(safe13=True)
    malformed = _runtime_context()
    malformed["phase_scalars"] = torch.zeros(1, 2)
    adapter.begin_policy_call(malformed)
    assert adapter.target_layer == 27
    assert adapter.fail_reason is not None

    adapter.router.head13.weight[0] = 1.0
    adapter.begin_policy_call(_runtime_context())
    assert adapter.target_layer == 27
    assert "mutated" in str(adapter.fail_reason)


class _FakeFlowModel:
    def __init__(self) -> None:
        self.calls = 0
        self.config = SimpleNamespace(
            action_head="flow_matching",
            num_diffusion_inference_steps=10,
            num_actions_chunk=8,
            fixed_action_dim=7,
        )

    def predict_actions_flow_matching(
        self,
        kvs,
        proprio,
        pos_offset,
        input_x=None,
        fm_trace_callback=None,
        fm_trace_context=None,
    ) -> torch.Tensor:
        del proprio, pos_offset, input_x
        self.calls += 1
        action = torch.full((1, 8, 7), float(len(kvs)))
        if fm_trace_callback is not None:
            fm_trace_callback(
                {
                    **dict(fm_trace_context or {}),
                    "input_x": torch.zeros_like(action),
                    "output_action": action,
                    "fm_steps": 10,
                }
            )
        return action


def _controller(
    adapter: RouteFirstRuntimeAdapter,
) -> tuple[RouteFirstExitController, _FakeFlowModel]:
    model = _FakeFlowModel()
    plan = SimpleNamespace(eligible_exit_layers=(3, 11, 13, 27))
    value_net = ActionValueNet(
        exit_list=[3, 11, 13, 27],
        exit_head=None,
        model=model,
        interval=2,
        threshold_type="cosine",
        productive_exit_plan=plan,
    )
    controller = RouteFirstExitController(
        value_net,
        exit_id_list=[3, 11, 13, 27],
        steps_per_stage=1,
        max_layer=28,
    )
    controller.thresholds = {3: 0.0, 11: 0.0, 13: 0.0, 27: 1.0e8}
    controller.install_route_first_adapter(adapter)
    controller.set_timestep(0)
    return controller, model


def _run_controller(safe13: bool) -> tuple[int, torch.Tensor, int, list[tuple[str, dict]]]:
    adapter = _adapter(safe13=safe13)
    adapter.begin_policy_call(_runtime_context())
    controller, model = _controller(adapter)
    kvs = [(torch.zeros(1, 1), torch.zeros(1, 1)) for _ in range(28)]
    events: list[tuple[str, dict]] = []
    for layer in (3, 11, 13, 27):
        should_exit, action = controller(
            kvs[: layer + 1],
            layer,
            None,
            0,
            0,
            None,
            telemetry_callback=lambda name, payload: events.append(
                (name, dict(payload))
            ),
        )
        if should_exit:
            return layer, action, model.calls, events
    raise AssertionError("route-first controller did not select an action")


def test_controller_executes_flow_matching_exactly_once_at_locked_depth() -> None:
    layer13, action13, calls13, events13 = _run_controller(True)
    layer27, action27, calls27, events27 = _run_controller(False)

    assert (layer13, calls13) == (13, 1)
    assert (layer27, calls27) == (27, 1)
    assert torch.equal(action13, torch.full((1, 8, 7), 14.0))
    assert torch.equal(action27, torch.full((1, 8, 7), 28.0))
    assert sum(
        event == "route_first_selected_action" for event, _ in events13
    ) == 1
    assert sum(
        payload.get("fm_calls") == 1
        for event, payload in events27
        if event == "exit_candidate" and payload.get("evaluated")
    ) == 1


def test_repository_stage6_stage7_loader_binds_exact_evidence() -> None:
    runtime = load_route_first_active_runtime(
        REPO_ROOT / "runs/route_first_calibration_stage6/router_calibrated.npz",
        REPO_ROOT / "results/route_first/route_first_stage7_holdout.json",
        REPO_ROOT / "artifacts/phase_route_v3/final_router.pt",
        REPO_ROOT / "artifacts/phase_route_v3/phase_estimator.pt",
    )

    assert (
        runtime.route_first_artifacts.calibrated_router_sha256
        == ROUTE_FIRST_CALIBRATED_ROUTER_SHA256
    )
    assert (
        runtime.route_first_artifacts.stage7_holdout_sha256
        == ROUTE_FIRST_STAGE7_HOLDOUT_SHA256
    )
    assert runtime.adapter.enabled11 is False
    assert runtime.adapter.enabled13 is True
    assert runtime.adapter.target_layer == 27


def test_stage1_probe_supports_route_first_probabilities_and_selected_action() -> None:
    runtime = load_route_first_active_runtime(
        REPO_ROOT / "runs/route_first_calibration_stage6/router_calibrated.npz",
        REPO_ROOT / "results/route_first/route_first_stage7_holdout.json",
        REPO_ROOT / "artifacts/phase_route_v3/final_router.pt",
        REPO_ROOT / "artifacts/phase_route_v3/phase_estimator.pt",
    )
    episode_id = "route-first:task0:episode12"
    context = {
        "episode_id": episode_id,
        "call_ordinal": 0,
        "step_id": 10,
        "task_id": 0,
    }
    runtime.start_episode(episode_id)
    probe = Stage1RuntimeProbe(runtime)
    probe.start_call(context)
    assert runtime.begin_policy_call(
        context=context,
        instruction_summary=torch.zeros(1, 3584),
        normalized_proprio=np.zeros(8, dtype=np.float32),
    )
    positions = torch.arange(720).reshape(1, 5, 144)
    positions[:, 4] = -1
    assert runtime.capture_visual_features(
        {
            "projected_features": torch.zeros(1, 5, 144, 3584),
            "image_input_idx": positions,
        }
    )
    assert runtime.prepare_policy_call()
    layer = runtime.adapter.target_layer
    action = torch.zeros(1, 8, 7, dtype=torch.bfloat16)
    decision = runtime.adapter.select_action(
        layer,
        action,
        fm_calls=1,
        telemetry_callback=runtime.record_route_event,
    )
    assert decision.selected_action is action
    assert runtime.commit_selected_action(action)
    measured = probe.finish_call()

    assert runtime.records[-1]["selected_layer"] == layer
    assert "router_predict" in measured["components"]
    assert measured["components"]["selected_action_route"][0]["layer"] == layer
    assert measured["components"]["selected_action_route"][0]["should_exit"] is True
