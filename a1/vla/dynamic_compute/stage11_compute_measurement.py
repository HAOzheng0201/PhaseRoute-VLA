"""Timing-only compute probe for independent Route-first Stage-11B runs.

The probe wraps live Python objects without editing the frozen A1 evaluator or
model sources.  Its measurements are never exposed to a controller.  CUDA
events are resolved only after the enclosing policy call has synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch


STAGE11_COMPUTE_SCHEMA = "phase-route-vla.stage11.compute-measurement.v1"


class Stage11ComputeMeasurementError(RuntimeError):
    """Raised when the measurement lifecycle or model structure differs."""


@dataclass
class _PendingSpan:
    name: str
    cpu_wall_ms: float
    cuda_start: Any
    cuda_end: Any
    metadata: Mapping[str, Any]


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage11ComputeMeasurementError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Stage11ComputeMeasurementError(f"{name} is not finite and non-negative")
    return result


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [_finite_nonnegative(value, "latency") for value in values]
    if not finite:
        return {
            "count": 0,
            "sum": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    total = math.fsum(finite)
    return {
        "count": len(finite),
        "sum": total,
        "mean": total / len(finite),
        "p50": _nearest_rank(finite, 0.50),
        "p90": _nearest_rank(finite, 0.90),
        "p95": _nearest_rank(finite, 0.95),
        "max": max(finite),
    }


class Stage11ComputeProbe:
    """Measure model, vision, decoder blocks, and selected-action FM."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._active: dict[str, Any] | None = None
        self._installed = False

    def _new_cuda_event(self) -> Any:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.Event(enable_timing=True)

    def _wrap(
        self,
        owner: Any,
        name: str,
        metric_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        original = getattr(owner, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            if self._active is None:
                return original(*args, **kwargs)
            cuda_start = self._new_cuda_event()
            cuda_end = self._new_cuda_event()
            if cuda_start is not None:
                cuda_start.record()
            started_ns = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                cpu_wall_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
                if cuda_end is not None:
                    cuda_end.record()
                self._active["spans"].append(
                    _PendingSpan(
                        name=metric_name,
                        cpu_wall_ms=cpu_wall_ms,
                        cuda_start=cuda_start,
                        cuda_end=cuda_end,
                        metadata=dict(metadata or {}),
                    )
                )

        setattr(owner, name, measured)

    def install(self) -> None:
        if self._installed:
            return
        required = ("predict_actions", "predict_actions_flow_matching")
        if any(not callable(getattr(self.model, name, None)) for name in required):
            raise Stage11ComputeMeasurementError("model prediction interface differs")
        vision = getattr(self.model, "vision_backbone", None)
        transformer = getattr(self.model, "transformer", None)
        blocks = getattr(transformer, "blocks", None)
        if vision is None or not callable(getattr(vision, "forward", None)):
            raise Stage11ComputeMeasurementError("vision backbone interface differs")
        if not isinstance(blocks, (torch.nn.ModuleList, list, tuple)) or not blocks:
            raise Stage11ComputeMeasurementError("decoder block interface differs")

        self._wrap(self.model, "predict_actions", "model_predict")
        self._wrap(
            self.model,
            "predict_actions_flow_matching",
            "selected_action_fm",
        )
        self._wrap(vision, "forward", "vision_backbone")
        for layer, block in enumerate(blocks):
            if not callable(getattr(block, "forward", None)):
                raise Stage11ComputeMeasurementError(f"decoder block {layer} differs")
            self._wrap(
                block,
                "forward",
                "decoder_block",
                {"layer": layer},
            )
        self._installed = True

    def start_call(self, context: Mapping[str, Any] | None = None) -> None:
        if self._active is not None:
            raise Stage11ComputeMeasurementError("previous compute call is still active")
        self.install()
        self._active = {
            "context": {
                str(key): _json_scalar(value)
                for key, value in dict(context or {}).items()
            },
            "spans": [],
        }

    @staticmethod
    def _resolve_span(span: _PendingSpan) -> dict[str, Any]:
        cuda_ms = None
        if span.cuda_start is not None and span.cuda_end is not None:
            cuda_ms = float(span.cuda_start.elapsed_time(span.cuda_end))
            _finite_nonnegative(cuda_ms, f"{span.name} CUDA latency")
        return {
            "name": span.name,
            "cpu_wall_ms": _finite_nonnegative(
                span.cpu_wall_ms, f"{span.name} CPU latency"
            ),
            "cuda_event_ms": cuda_ms,
            **{str(key): _json_scalar(value) for key, value in span.metadata.items()},
        }

    def finish_call(
        self,
        *,
        selected_layer: int | None,
        outer_policy_wall_ms: float,
        error: str | None,
    ) -> dict[str, Any]:
        if self._active is None:
            raise Stage11ComputeMeasurementError("no active compute call")
        active = self._active
        self._active = None
        spans = [self._resolve_span(span) for span in active["spans"]]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            grouped.setdefault(str(span["name"]), []).append(span)

        model_spans = grouped.get("model_predict", [])
        vision_spans = grouped.get("vision_backbone", [])
        fm_spans = grouped.get("selected_action_fm", [])
        decoder_spans = sorted(
            grouped.get("decoder_block", []), key=lambda item: int(item["layer"])
        )
        structure_valid = bool(
            error is None
            and selected_layer in (13, 27)
            and len(model_spans) == 1
            and len(vision_spans) == 1
            and len(fm_spans) == 1
            and len(decoder_spans) == int(selected_layer) + 1
            and [int(item["layer"]) for item in decoder_spans]
            == list(range(int(selected_layer) + 1))
        )

        cuda_complete = bool(
            structure_valid
            and all(span.get("cuda_event_ms") is not None for span in spans)
        )
        decomposition: dict[str, Any] = {
            "structure_valid": structure_valid,
            "cuda_events_complete": cuda_complete,
            "executed_decoder_blocks": len(decoder_spans),
            "expected_decoder_blocks": (
                int(selected_layer) + 1 if selected_layer in (13, 27) else None
            ),
        }
        if structure_valid:
            model_cpu = float(model_spans[0]["cpu_wall_ms"])
            outer = _finite_nonnegative(outer_policy_wall_ms, "outer policy wall")
            decomposition.update(
                {
                    "host_and_wrapper_outside_model_cpu_ms": max(0.0, outer - model_cpu),
                    "model_predict_cpu_ms": model_cpu,
                    "vision_backbone_cpu_ms": float(vision_spans[0]["cpu_wall_ms"]),
                    "decoder_blocks_cpu_sum_ms": math.fsum(
                        float(item["cpu_wall_ms"]) for item in decoder_spans
                    ),
                    "selected_action_fm_cpu_ms": float(fm_spans[0]["cpu_wall_ms"]),
                }
            )
        if cuda_complete:
            model_cuda = float(model_spans[0]["cuda_event_ms"])
            vision_cuda = float(vision_spans[0]["cuda_event_ms"])
            decoder_cuda = math.fsum(
                float(item["cuda_event_ms"]) for item in decoder_spans
            )
            fm_cuda = float(fm_spans[0]["cuda_event_ms"])
            attributed = vision_cuda + decoder_cuda + fm_cuda
            residual = model_cuda - attributed
            decomposition.update(
                {
                    "model_predict_cuda_ms": model_cuda,
                    "vision_backbone_cuda_ms": vision_cuda,
                    "decoder_blocks_cuda_sum_ms": decoder_cuda,
                    "selected_action_fm_cuda_ms": fm_cuda,
                    "model_other_cuda_ms": max(0.0, residual),
                    "component_sum_not_above_model_with_1ms_tolerance": residual >= -1.0,
                }
            )

        return {
            "schema_version": STAGE11_COMPUTE_SCHEMA,
            "measurement_is_control_input": False,
            "frozen_model_or_evaluator_source_modified": False,
            "context": active["context"],
            "selected_layer": selected_layer,
            "outer_policy_wall_ms": _finite_nonnegative(
                outer_policy_wall_ms, "outer policy wall"
            ),
            "error": error,
            "spans": spans,
            "decomposition": decomposition,
        }


def summarize_stage11_compute_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = list(records)
    if not materialized:
        raise Stage11ComputeMeasurementError("compute records are empty")
    if any(record.get("schema_version") != STAGE11_COMPUTE_SCHEMA for record in materialized):
        raise Stage11ComputeMeasurementError("compute measurement schema differs")
    if any(record.get("measurement_is_control_input") is not False for record in materialized):
        raise Stage11ComputeMeasurementError("compute measurement became a control input")
    if any(record.get("frozen_model_or_evaluator_source_modified") is not False for record in materialized):
        raise Stage11ComputeMeasurementError("frozen source modification was reported")

    metric_names = (
        "outer_policy_wall_ms",
        "model_predict_cpu_ms",
        "host_and_wrapper_outside_model_cpu_ms",
        "model_predict_cuda_ms",
        "vision_backbone_cuda_ms",
        "decoder_blocks_cuda_sum_ms",
        "selected_action_fm_cuda_ms",
        "model_other_cuda_ms",
    )
    valid = [
        record
        for record in materialized
        if record.get("error") is None
        and isinstance(record.get("decomposition"), Mapping)
        and record["decomposition"].get("structure_valid") is True
        and record["decomposition"].get("cuda_events_complete") is True
        and record["decomposition"].get(
            "component_sum_not_above_model_with_1ms_tolerance"
        )
        is True
    ]

    def values(group: Sequence[Mapping[str, Any]], name: str) -> list[float]:
        output = []
        for record in group:
            source = record if name == "outer_policy_wall_ms" else record["decomposition"]
            output.append(_finite_nonnegative(source.get(name), name))
        return output

    by_layer = {
        str(layer): [record for record in valid if record.get("selected_layer") == layer]
        for layer in (13, 27)
    }
    return {
        "schema_version": "phase-route-vla.stage11.compute-summary.v1",
        "records": len(materialized),
        "valid_records": len(valid),
        "invalid_records": len(materialized) - len(valid),
        "selected_layer_counts": {
            str(layer): sum(record.get("selected_layer") == layer for record in materialized)
            for layer in (13, 27)
        },
        "latency_ms": {
            name: latency_summary(values(valid, name)) for name in metric_names
        },
        "by_selected_layer": {
            layer: {
                "records": len(group),
                "latency_ms": {
                    name: latency_summary(values(group, name)) for name in metric_names
                },
            }
            for layer, group in by_layer.items()
        },
        "claim_boundary": {
            "timing_only": True,
            "profiling_overhead_included": True,
            "not_a_control_comparison": True,
            "not_a_speedup_confirmation": True,
        },
    }


__all__ = [
    "STAGE11_COMPUTE_SCHEMA",
    "Stage11ComputeMeasurementError",
    "Stage11ComputeProbe",
    "latency_summary",
    "summarize_stage11_compute_records",
]
