"""Non-control timing overlay for post-release Stage-1 experiments.

The D9 controller sources are SHA-256-bound research evidence and must remain
byte-for-byte unchanged.  This module instruments live runtime objects from
outside that frozen boundary.  Measurements are never exposed to a routing
method and therefore cannot become control inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping


class Stage1MeasurementError(RuntimeError):
    """Raised when the measurement lifecycle itself is used incorrectly."""


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _latency_summary(values: Any) -> dict[str, Any]:
    finite = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
    if not finite:
        return {
            "count": 0,
            "sum": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }

    def nearest_rank(percentile: float) -> float:
        return finite[max(0, math.ceil(percentile * len(finite)) - 1)]

    total = math.fsum(finite)
    return {
        "count": len(finite),
        "sum": total,
        "mean": total / len(finite),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": finite[-1],
    }


def summarize_stage1_records(records: Any) -> dict[str, Any]:
    """Aggregate strict JSON timing records without using them for control."""

    materialized = tuple(record for record in records if isinstance(record, Mapping))
    component_names = sorted(
        {
            str(name)
            for record in materialized
            for name in (
                record.get("components", {}).keys()
                if isinstance(record.get("components"), Mapping)
                else ()
            )
        }
    )
    component_values: dict[str, list[float]] = {name: [] for name in component_names}
    for record in materialized:
        components = record.get("components")
        if not isinstance(components, Mapping):
            continue
        for name in component_names:
            events = components.get(name, ())
            if not isinstance(events, (tuple, list)):
                continue
            component_values[name].extend(
                event.get("latency_ms")
                for event in events
                if isinstance(event, Mapping)
            )
    return {
        "schema_version": "phase-route-vla.stage1.measurement-summary.v1",
        "records": len(materialized),
        "records_with_errors": sum(record.get("error") is not None for record in materialized),
        "records_with_nonfinite_actions": sum(
            record.get("action_finite") is False for record in materialized
        ),
        "records_without_action_audit": sum(
            record.get("action_finite") not in (True, False)
            for record in materialized
        ),
        "selected_layers": {
            str(layer): sum(record.get("selected_layer") == layer for record in materialized)
            for layer in (11, 13, 27)
        },
        "latency_ms": {
            "policy_wall": _latency_summary(
                record.get("policy_wall_latency_ms") for record in materialized
            ),
            "policy_cuda_event": _latency_summary(
                record.get("policy_cuda_event_latency_ms") for record in materialized
            ),
            **{
                name: _latency_summary(component_values[name])
                for name in component_names
            },
        },
    }


@dataclass
class _TimedRouterProxy:
    """Duck-typed FinalFiveHeadRouter proxy that times only ``predict``."""

    target: Any
    probe: "Stage1RuntimeProbe"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.perf_counter_ns()
        try:
            return self.target.predict(*args, **kwargs)
        finally:
            self.probe._safe_append("router_predict", _elapsed_ms(started_ns))

    def probabilities(self, *args: Any, **kwargs: Any) -> Any:
        """Time the route-first affine router without changing its interface."""

        started_ns = time.perf_counter_ns()
        try:
            return self.target.probabilities(*args, **kwargs)
        finally:
            self.probe._safe_append("router_predict", _elapsed_ms(started_ns))


class Stage1RuntimeProbe:
    """Install a timing-only overlay on one frozen ActivePhaseRouteRuntime."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._active: dict[str, Any] | None = None
        self._installed = False

    def _append(
        self,
        name: str,
        latency_ms: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._active is None:
            return
        event = {"latency_ms": float(latency_ms)}
        if metadata:
            event.update({str(key): _json_scalar(value) for key, value in metadata.items()})
        self._active["components"].setdefault(name, []).append(event)

    def _safe_append(
        self,
        name: str,
        latency_ms: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self._append(name, latency_ms, metadata)
        except Exception:
            pass

    def _wrap_method(
        self,
        owner: Any,
        name: str,
        metric_name: str,
        metadata_fn: Callable[[tuple[Any, ...], Mapping[str, Any], Any], Mapping[str, Any]]
        | None = None,
    ) -> None:
        original = getattr(owner, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            started_ns = time.perf_counter_ns()
            result = None
            try:
                result = original(*args, **kwargs)
                return result
            finally:
                try:
                    metadata = (
                        metadata_fn(args, kwargs, result)
                        if metadata_fn is not None
                        else None
                    )
                    self._safe_append(metric_name, _elapsed_ms(started_ns), metadata)
                except Exception:
                    pass

        setattr(owner, name, measured)

    @staticmethod
    def _candidate_metadata(
        args: tuple[Any, ...], kwargs: Mapping[str, Any], result: Any
    ) -> Mapping[str, Any]:
        layer = args[0] if args else kwargs.get("layer")
        return {
            "layer": layer,
            "should_exit": getattr(result, "should_exit", None),
        }

    def install(self) -> None:
        if self._installed:
            return
        self._wrap_method(self.runtime, "begin_policy_call", "runtime_begin")
        self._wrap_method(self.runtime, "capture_visual_features", "visual_capture")
        self._wrap_method(self.runtime, "prepare_policy_call", "runtime_prepare")
        self._wrap_method(self.runtime, "commit_selected_action", "runtime_commit")

        adapter = self.runtime.adapter
        self._wrap_method(adapter, "begin_policy_call", "adapter_begin")
        if bool(getattr(adapter, "route_first", False)):
            self._wrap_method(
                adapter,
                "select_action",
                "selected_action_route",
                self._candidate_metadata,
            )
        else:
            self._wrap_method(
                adapter,
                "consider_candidate",
                "candidate_route",
                self._candidate_metadata,
            )
            self._wrap_method(adapter, "select_fallback", "fallback_route")

        phase_forward = self.runtime.phase_estimator.forward

        def measured_phase(*args: Any, **kwargs: Any) -> Any:
            started_ns = time.perf_counter_ns()
            try:
                return phase_forward(*args, **kwargs)
            finally:
                self._safe_append("phase_estimator", _elapsed_ms(started_ns))

        self.runtime.phase_estimator.forward = measured_phase

        # The adapter's integrity hash reads router state through attributes;
        # the proxy delegates all state and changes only predict timing.
        adapter.router = _TimedRouterProxy(adapter.router, self)
        self._installed = True

    def start_call(self, context: Mapping[str, Any] | None) -> None:
        if self._active is not None:
            raise Stage1MeasurementError("previous measured policy call is still active")
        self.install()
        self._active = {
            "context": {
                str(key): _json_scalar(value) for key, value in dict(context or {}).items()
            },
            "components": {},
        }

    def finish_call(self) -> dict[str, Any]:
        if self._active is None:
            raise Stage1MeasurementError("no measured policy call is active")
        result = self._active
        self._active = None
        return result


__all__ = [
    "Stage1MeasurementError",
    "Stage1RuntimeProbe",
    "summarize_stage1_records",
]
