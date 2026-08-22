from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from a1.vla.value_net import ActionValueNet, ExitController
from a1.vla.dynamic_compute.v3.final_router import FinalFiveHeadRouter
from a1.vla.dynamic_compute.v3.gripper_v2_models import FeatureNormalizer
from a1.vla.dynamic_compute.v3.runtime_adapter import (
    EpisodePastOnlyHistory,
    FrozenD8RuntimeAdapter,
    RuntimeAdapterError,
    frozen_router_sha256,
    route_cached_candidate_pairs,
)
from a1.vla.dynamic_compute.v3.severity_reliability import SeverityWeightedFit


def _model(offset: float) -> SeverityWeightedFit:
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


def _router() -> FinalFiveHeadRouter:
    return FinalFiveHeadRouter(
        models=tuple(_model(0.01 * index) for index in range(5)),
        full_threshold=0.5,
        runtime_threshold=0.475,
    )


def _runtime(rows: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260822)
    return {
        "instruction_summary": torch.randn(rows, 3584, generator=generator),
        "vision_crop_summary": torch.randn(rows, 5, 3584, generator=generator),
        "vision_crop_mask": torch.tensor(
            [[True, True, True, True, False]] * rows
        ),
        "phase_embedding": torch.randn(rows, 128, generator=generator),
        "phase_scalars": torch.rand(rows, 3, generator=generator),
        "normalized_proprio": torch.randn(rows, 8, generator=generator),
        "proprio_history": torch.zeros(rows, 8, 8),
        "action_history": torch.zeros(rows, 8, 8, 7),
        "history_mask": torch.zeros(rows, 8, dtype=torch.bool),
    }


def _actions() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return tuple(
        torch.randn(1, 8, 7, generator=generator, dtype=torch.float32)
        for _ in range(3)
    )


def test_sequential_runtime_covers_l11_l13_l27_and_exact_actions() -> None:
    router = _router()
    l11, l13, l27 = _actions()

    adapter = FrozenD8RuntimeAdapter(router)
    adapter.begin_policy_call(_runtime())
    selected = adapter.consider_candidate(11, l11, True)
    assert selected.should_exit and selected.layer == 11
    assert selected.selected_action is l11

    adapter.begin_policy_call(_runtime())
    veto = adapter.consider_candidate(11, l11, False)
    selected = adapter.consider_candidate(13, l13, True)
    assert not veto.should_exit
    assert selected.should_exit and selected.layer == 13
    assert selected.selected_action is l13

    adapter.begin_policy_call(_runtime())
    assert not adapter.consider_candidate(11, l11, False).should_exit
    assert not adapter.consider_candidate(13, l13, False).should_exit
    selected = adapter.select_fallback(l27)
    assert selected.should_exit and selected.layer == 27
    assert selected.selected_action is l27


def test_nonfinite_or_shape_drift_latches_fail_closed_to_exact_l27() -> None:
    adapter = FrozenD8RuntimeAdapter(_router())
    l11, l13, l27 = _actions()
    l11[0, 0, 0] = float("nan")
    adapter.begin_policy_call(_runtime())
    assert not adapter.consider_candidate(11, l11, True).should_exit
    assert adapter.fail_reason is not None
    assert not adapter.consider_candidate(13, l13, True).should_exit
    selected = adapter.select_fallback(l27)
    assert selected.selected_action is l27

    malformed = _runtime()
    malformed["phase_scalars"] = torch.zeros(1, 2)
    adapter.begin_policy_call(malformed)
    assert not adapter.consider_candidate(11, _actions()[0], True).should_exit
    assert adapter.select_fallback(l27).selected_action is l27


def test_episode_history_is_past_only_and_resets_at_boundary() -> None:
    history = EpisodePastOnlyHistory()
    proprio = np.arange(8, dtype=np.float32)
    action = np.arange(56, dtype=np.float32).reshape(8, 7)

    first = history.window("episode-a", 0, proprio)
    assert not bool(first.history_mask.any())
    history.commit("episode-a", 0, action)
    second = history.window("episode-a", 1, proprio + 1)
    assert second.history_mask[0].tolist() == [False] * 7 + [True]
    assert torch.equal(second.action_history[0, -1], torch.from_numpy(action))
    history.commit("episode-a", 1, action + 1)

    other = history.window("episode-b", 0, proprio + 2)
    assert not bool(other.history_mask.any())
    with pytest.raises(RuntimeAdapterError, match="not committed"):
        history.window("episode-b", 1, proprio + 3)


def test_vectorized_route_has_priority_and_does_not_mutate_router() -> None:
    router = _router()
    before = frozen_router_sha256(router)
    candidates = torch.stack(_actions()[:2], dim=1)
    route = route_cached_candidate_pairs(
        router,
        _runtime(),
        candidates,
        torch.tensor([[True, True]]),
    )
    assert route.features.shape == (2, 97)
    assert route.five_head_prediction.shape == (5, 2, 2)
    assert route.candidate_safe.tolist() == [True, True]
    assert route.selected_layer.tolist() == [11]
    assert frozen_router_sha256(router) == before


def test_candidate_identity_cannot_enter_97d_context() -> None:
    adapter = FrozenD8RuntimeAdapter(_router())
    leaked = _runtime()
    leaked["candidate_layer"] = torch.tensor([11])
    adapter.begin_policy_call(leaked)
    decision = adapter.consider_candidate(11, _actions()[0], True)
    assert not decision.should_exit
    assert "names or order" in str(adapter.fail_reason)


class _ControllerValueNet(ActionValueNet):
    def __init__(self, actions: dict[int, torch.Tensor]) -> None:
        torch.nn.Module.__init__(self)
        self.actions = actions
        self.productive_exit_plan = object()
        self.shared_layers = None
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_diffusion_inference_steps=10)
        )
        self.last_fm_calls = 1
        self.last_fm_steps = 10
        self.last_rng_burns = 0

    def configure_phase_route_shared_candidates(self, layers) -> None:
        self.shared_layers = layers

    def reset_actions(self) -> None:
        pass

    def forward(self, x, i, proprio, start_idx, end_idx, pos_offset):
        del x, proprio, start_idx, end_idx, pos_offset
        return torch.tensor([0.0]), self.actions[i]


def test_exit_controller_integration_vetoes_l3_and_fails_closed_without_context() -> None:
    l3, l11, l13, l27 = (
        torch.full((1, 8, 7), float(layer), dtype=torch.float32)
        for layer in (3, 11, 13, 27)
    )
    actions = {3: l3, 11: l11, 13: l13, 27: l27}
    value_net = _ControllerValueNet(actions)
    controller = ExitController(
        value_net,
        exit_id_list=[3, 11, 13, 27],
        steps_per_stage=1,
        max_layer=28,
    )
    controller.thresholds = {3: 1.0, 11: 1.0, 13: 1.0, 27: 1.0e8}
    adapter = FrozenD8RuntimeAdapter(_router())
    controller.set_phase_route_runtime_adapter(adapter)
    assert value_net.shared_layers == (11, 13, 27)

    controller.set_timestep(0)
    controller.begin_phase_route_policy_call(_runtime())
    assert controller(None, 3, None, 0, 0, None) == (False, None)
    exit_flag, selected = controller(None, 11, None, 0, 0, None)
    assert exit_flag and selected is l11

    controller.set_timestep(1)
    controller.begin_phase_route_policy_call(None)
    assert controller(None, 3, None, 0, 0, None) == (False, None)
    assert controller(None, 11, None, 0, 0, None) == (False, None)
    assert controller(None, 13, None, 0, 0, None) == (False, None)
    exit_flag, selected = controller(None, 27, None, 0, 0, None)
    assert exit_flag and selected is l27
