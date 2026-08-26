"""Stage-1 policy-call overlay that preserves the SHA-bound D9 evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np
import torch

from a1.vla.dynamic_compute.stage1_measurement import Stage1RuntimeProbe
from robot_experiments.libero.exit_vla_utils import (
    get_vla_action as get_frozen_d9_vla_action,
)


STAGE1_MEASUREMENT_SCHEMA_VERSION = "phase-route-vla.stage1.measurement.v1"
STAGE1_TIMING_ENV = "PHASEROUTE_STAGE1_TIMING_PATH"

_PROBES: dict[int, Stage1RuntimeProbe] = {}
_WRITE_LOCK = threading.Lock()


def _probe_for(runtime: Any) -> Stage1RuntimeProbe:
    identity = id(runtime)
    probe = _PROBES.get(identity)
    if probe is None or probe.runtime is not runtime:
        probe = Stage1RuntimeProbe(runtime)
        _PROBES[identity] = probe
    return probe


def _action_audit(actions: Any) -> dict[str, Any]:
    if actions is None:
        return {
            "action_sha256": None,
            "action_finite": None,
            "action_shape": None,
        }
    array = np.asarray(actions, dtype=np.float32)
    return {
        "action_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "action_finite": bool(np.isfinite(array).all()),
        "action_shape": list(array.shape),
    }


def _append_record(record: Mapping[str, Any]) -> None:
    path_text = os.environ.get(STAGE1_TIMING_ENV)
    if not path_text:
        return
    path = Path(path_text).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        dict(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(line + "\n")
            output_file.flush()


def _fixed_layer_call(
    cfg: Any,
    model: Any,
    device: Any,
    obs: Any,
    task_label: str,
    exit_controller: Any,
    output_hidden_states: bool,
    log_fn: Any,
    kwargs: Mapping[str, Any],
) -> Any:
    fixed_layer = getattr(cfg, "exit_layer_id", None)
    if fixed_layer not in (11, 13, 27):
        raise ValueError("Stage-1 fixed baseline requires exit_layer_id in {11,13,27}")
    if (
        exit_controller is None
        or getattr(exit_controller, "stage1_fixed_layer", False) is not True
        or getattr(exit_controller, "layer", None) != fixed_layer
    ):
        raise ValueError("fixed baseline requires the matching Stage-1 controller")
    incompatible = {
        "phase_cache_writer": kwargs.get("phase_cache_writer"),
        "phase_depth_runtime": kwargs.get("phase_depth_runtime"),
        "vision_aggregation_config": kwargs.get("vision_aggregation_config"),
        "learnable_vision_aggregator": kwargs.get("learnable_vision_aggregator"),
        "vision_teacher_cache_writer": kwargs.get("vision_teacher_cache_writer"),
        "phase_route_runtime": kwargs.get("phase_route_runtime"),
    }
    enabled = [name for name, value in incompatible.items() if value is not None]
    if enabled:
        raise ValueError(
            "fixed-layer baseline cannot enable dynamic modules: " + ", ".join(enabled)
        )
    return get_frozen_d9_vla_action(
        cfg,
        model,
        device,
        obs,
        task_label,
        exit_controller=exit_controller,
        output_hidden_states=output_hidden_states,
        log_fn=log_fn,
        **dict(kwargs),
    )


def get_vla_action(
    cfg: Any,
    model: Any,
    device: Any,
    obs: Any,
    task_label: str,
    exit_controller: Any = None,
    output_hidden_states: bool = False,
    log_fn: Any = None,
    **kwargs: Any,
) -> Any:
    """Run one fixed or dynamic policy call with external timing only."""

    fixed_layer = getattr(cfg, "exit_layer_id", None)
    runtime = kwargs.get("phase_route_runtime")
    context = kwargs.get("phase_route_context") or kwargs.get("telemetry_context")
    probe = _probe_for(runtime) if runtime is not None else None
    if probe is not None:
        try:
            probe.start_call(context)
        except Exception as measurement_error:
            print(f"[Stage-1 measurement warning] {measurement_error}", flush=True)
            probe = None

    cuda_start = None
    cuda_end = None
    if torch.cuda.is_available():
        try:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        except Exception:
            cuda_start = None
            cuda_end = None
    started_ns = time.perf_counter_ns()
    actions = None
    error = None
    try:
        if fixed_layer is not None:
            actions = _fixed_layer_call(
                cfg,
                model,
                device,
                obs,
                task_label,
                exit_controller,
                output_hidden_states,
                log_fn,
                kwargs,
            )
        else:
            actions = get_frozen_d9_vla_action(
                cfg,
                model,
                device,
                obs,
                task_label,
                exit_controller,
                output_hidden_states=output_hidden_states,
                log_fn=log_fn,
                **kwargs,
            )
        return actions
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        try:
            if cuda_end is not None:
                try:
                    cuda_end.record()
                    torch.cuda.synchronize()
                except Exception:
                    cuda_start = None
                    cuda_end = None
            wall_latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            cuda_latency_ms = None
            if cuda_start is not None and cuda_end is not None:
                try:
                    cuda_latency_ms = float(cuda_start.elapsed_time(cuda_end))
                except Exception:
                    cuda_latency_ms = None
            probe_record = probe.finish_call() if probe is not None else {
                "context": {
                    str(key): value
                    for key, value in dict(context or {}).items()
                    if value is None or isinstance(value, (str, bool, int, float))
                },
                "components": {},
            }
            selected_layer = int(fixed_layer) if fixed_layer is not None else None
            if runtime is not None and runtime.records:
                selected = runtime.records[-1].get("selected_layer")
                if selected in (11, 13, 27):
                    selected_layer = int(selected)
            record = {
                "schema_version": STAGE1_MEASUREMENT_SCHEMA_VERSION,
                "measurement_is_control_input": False,
                "d9_protected_source_modified": False,
                "mode": (
                    f"fixed_l{fixed_layer}"
                    if fixed_layer is not None
                    else (
                        "route_first_stage8"
                        if runtime is not None
                        and bool(
                            getattr(
                                getattr(runtime, "adapter", None),
                                "route_first",
                                False,
                            )
                        )
                        else ("phase_route_v3" if runtime is not None else "original_a1")
                    )
                ),
                **probe_record,
                "selected_layer": selected_layer,
                "policy_wall_latency_ms": wall_latency_ms,
                "policy_cuda_event_latency_ms": cuda_latency_ms,
                **_action_audit(actions),
                "error": error,
            }
            _append_record(record)
        except Exception as measurement_error:
            # Measurement I/O cannot alter an action that is sent to control.
            print(f"[Stage-1 measurement warning] {measurement_error}", flush=True)


__all__ = [
    "STAGE1_MEASUREMENT_SCHEMA_VERSION",
    "STAGE1_TIMING_ENV",
    "get_vla_action",
]
