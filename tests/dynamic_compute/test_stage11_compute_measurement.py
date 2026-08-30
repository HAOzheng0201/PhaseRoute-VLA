from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from a1.vla.dynamic_compute.stage11_compute_measurement import (
    Stage11ComputeMeasurementError,
    Stage11ComputeProbe,
    latency_summary,
    summarize_stage11_compute_records,
)


class _Block(torch.nn.Module):
    def forward(self, value):
        return value + 1


class _Vision(torch.nn.Module):
    def forward(self, value, masks=None):
        del masks
        return value * 2


class _Model:
    def __init__(self):
        self.vision_backbone = _Vision()
        self.transformer = SimpleNamespace(
            blocks=torch.nn.ModuleList([_Block() for _ in range(14)])
        )

    def predict_actions_flow_matching(self, value, *args, **kwargs):
        del args, kwargs
        return value + 10

    def predict_actions(self, value):
        value = self.vision_backbone(value)
        for block in self.transformer.blocks:
            value = block(value)
        return self.predict_actions_flow_matching(value)


def test_probe_preserves_output_and_records_nested_structure_on_cpu() -> None:
    model = _Model()
    probe = Stage11ComputeProbe(model)
    probe.start_call({"task_id": 0})
    output = model.predict_actions(torch.tensor(2.0))
    record = probe.finish_call(
        selected_layer=13,
        outer_policy_wall_ms=10.0,
        error=None,
    )
    assert output.item() == 28.0
    assert record["measurement_is_control_input"] is False
    assert record["decomposition"]["structure_valid"] is True
    assert record["decomposition"]["executed_decoder_blocks"] == 14
    assert record["decomposition"]["cuda_events_complete"] is False
    assert [span["name"] for span in record["spans"]].count("decoder_block") == 14


def test_probe_requires_one_active_call_and_matching_model_structure() -> None:
    probe = Stage11ComputeProbe(_Model())
    with pytest.raises(Stage11ComputeMeasurementError, match="no active"):
        probe.finish_call(selected_layer=13, outer_policy_wall_ms=1.0, error=None)
    probe.start_call()
    with pytest.raises(Stage11ComputeMeasurementError, match="still active"):
        probe.start_call()


def test_latency_summary_handles_empty_and_rejects_nonfinite() -> None:
    assert latency_summary([])["count"] == 0
    assert latency_summary([1.0, 3.0, 2.0])["p50"] == 2.0
    with pytest.raises(Stage11ComputeMeasurementError, match="finite"):
        latency_summary([float("inf")])


def test_summary_rejects_non_stage11_records() -> None:
    with pytest.raises(Stage11ComputeMeasurementError, match="schema"):
        summarize_stage11_compute_records([{"schema_version": "wrong"}])
