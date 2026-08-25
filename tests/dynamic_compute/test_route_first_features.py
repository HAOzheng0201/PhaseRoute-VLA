from __future__ import annotations

import numpy as np
import torch

from a1.vla.dynamic_compute.route_first_collection import (
    ROUTE_FIRST_COLLECTION_SCHEMA_VERSION,
    RouteFirstTeacherCollector,
)
from a1.vla.dynamic_compute.route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    ROUTE_FIRST_FEATURE_GROUPS,
    ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
    build_route_first_context_features,
    route_first_feature_slices,
)


def _runtime_context(rows: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260825)
    return {
        "instruction_summary": torch.randn(rows, 3584, generator=generator),
        "vision_crop_summary": torch.randn(rows, 5, 3584, generator=generator),
        "vision_crop_mask": torch.ones(rows, 5, dtype=torch.bool),
        "phase_embedding": torch.randn(rows, 128, generator=generator),
        "phase_scalars": torch.rand(rows, 3, generator=generator),
        "normalized_proprio": torch.randn(rows, 8, generator=generator),
        "proprio_history": torch.randn(rows, 8, 8, generator=generator),
        "action_history": torch.randn(rows, 8, 8, 7, generator=generator),
        "history_mask": torch.tensor(
            [[False, False, False, True, True, True, True, True]] * rows
        ),
    }


def test_route_first_feature_contract_is_complete_and_action_free():
    context = _runtime_context(rows=2)
    features = build_route_first_context_features(context)

    assert features.shape == (2, ROUTE_FIRST_FEATURE_DIMENSION)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert ROUTE_FIRST_FEATURE_DIMENSION == 199
    assert sum(ROUTE_FIRST_FEATURE_GROUPS.values()) == 199
    assert "candidate_action" not in ROUTE_FIRST_FEATURE_GROUPS
    assert "task_id" not in ROUTE_FIRST_FEATURE_GROUPS

    slices = route_first_feature_slices()
    assert slices["phase_embedding"] == slice(0, 128)
    assert slices["vision_crop_mask"].stop == ROUTE_FIRST_FEATURE_DIMENSION


def test_route_first_feature_uses_only_past_history():
    context = _runtime_context()
    no_history = {name: value.clone() for name, value in context.items()}
    no_history["history_mask"].zero_()
    no_history["proprio_history"].fill_(1234.0)
    no_history["action_history"].fill_(-987.0)
    baseline = build_route_first_context_features(no_history)

    changed_masked_storage = {
        name: value.clone() for name, value in no_history.items()
    }
    changed_masked_storage["proprio_history"].fill_(-777.0)
    changed_masked_storage["action_history"].fill_(555.0)
    changed = build_route_first_context_features(changed_masked_storage)

    assert torch.equal(baseline, changed)
    slices = route_first_feature_slices()
    assert torch.count_nonzero(baseline[:, slices["previous_first_action"]]) == 0
    assert torch.count_nonzero(baseline[:, slices["history_first_action_mean"]]) == 0
    assert torch.count_nonzero(baseline[:, slices["history_first_action_std"]]) == 0


def test_route_first_feature_ignores_storage_under_invalid_crop_mask():
    context = _runtime_context()
    context["vision_crop_mask"][0, 3:] = False
    baseline = build_route_first_context_features(context)

    changed = {name: value.clone() for name, value in context.items()}
    changed["vision_crop_summary"][0, 3:].fill_(12345.0)
    changed_feature = build_route_first_context_features(changed)

    assert torch.equal(baseline, changed_feature)


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = []

    def begin_policy_call(self, runtime_inputs):
        self.calls.append(runtime_inputs)
        return "original-result"


class _FakeRuntime:
    def __init__(self) -> None:
        self.adapter = _FakeAdapter()
        self.events = []
        self._current = {
            "context": {
                "episode_id": "libero_10:task2:episode7",
                "task_id": 2,
                "step_id": 42,
                "call_ordinal": 3,
            }
        }

    def record_route_event(self, event_name, payload):
        self.events.append((event_name, dict(payload)))
        return "recorded"


def test_teacher_collector_is_observation_only_and_publishes_safe_npz(tmp_path):
    runtime = _FakeRuntime()
    original_begin = runtime.adapter.begin_policy_call
    original_record = runtime.record_route_event
    collector = RouteFirstTeacherCollector(runtime)
    collector.install()

    context = _runtime_context()
    assert runtime.adapter.begin_policy_call(None) == "original-result"
    assert runtime.adapter.begin_policy_call(context) == "original-result"
    assert runtime.record_route_event(
        "phase_route_decision",
        {"selected_layer": 13, "fallback": False},
    ) == "recorded"
    assert runtime.adapter.calls == [None, context]
    assert runtime.events == [
        ("phase_route_decision", {"selected_layer": 13, "fallback": False})
    ]
    assert collector.error_count == 0
    assert collector.summary()["teacher_layer_counts"] == {
        "11": 0,
        "13": 1,
        "27": 0,
    }

    output = tmp_path / "teacher_context.npz"
    published = collector.save(output)
    assert published["control_influence"] is False
    assert published["rows"] == 1
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["schema_version"].item() == ROUTE_FIRST_COLLECTION_SCHEMA_VERSION
        assert (
            arrays["feature_schema_version"].item()
            == ROUTE_FIRST_FEATURE_SCHEMA_VERSION
        )
        assert arrays["features"].shape == (1, ROUTE_FIRST_FEATURE_DIMENSION)
        assert arrays["teacher_layer"].tolist() == [13]
        assert arrays["control_influence"].item() is False

    collector.uninstall()
    assert runtime.adapter.begin_policy_call == original_begin
    assert runtime.record_route_event == original_record
