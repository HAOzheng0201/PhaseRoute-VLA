"""Independent Stage-11B compute-timing overlay for the frozen evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from a1.vla.dynamic_compute.stage11_compute_measurement import Stage11ComputeProbe
from robot_experiments.libero.stage1_vla_utils import get_vla_action as get_stage1_vla_action


STAGE11_COMPUTE_ENV = "PHASEROUTE_STAGE11_COMPUTE_PATH"

_PROBES: dict[int, Stage11ComputeProbe] = {}
_WRITE_LOCK = threading.Lock()
_INITIALIZED_OUTPUTS: set[Path] = set()


def _probe_for(model: Any) -> Stage11ComputeProbe:
    identity = id(model)
    probe = _PROBES.get(identity)
    if probe is None or probe.model is not model:
        probe = Stage11ComputeProbe(model)
        _PROBES[identity] = probe
    return probe


def _append_record(record: Mapping[str, Any]) -> None:
    path_text = os.environ.get(STAGE11_COMPUTE_ENV)
    if not path_text:
        raise RuntimeError(f"{STAGE11_COMPUTE_ENV} is not set")
    path = Path(path_text).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        dict(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    with _WRITE_LOCK:
        if path not in _INITIALIZED_OUTPUTS:
            with path.open("x", encoding="utf-8") as output_file:
                output_file.write(line + "\n")
                output_file.flush()
            _INITIALIZED_OUTPUTS.add(path)
        else:
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(line + "\n")
                output_file.flush()


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
    """Run the frozen Stage-1 path with an additional measurement-only probe."""

    runtime = kwargs.get("phase_route_runtime")
    context = kwargs.get("phase_route_context") or kwargs.get("telemetry_context")
    probe = None
    try:
        probe = _probe_for(model)
        probe.start_call(context)
    except Exception as measurement_error:
        print(f"[Stage-11B measurement warning] {measurement_error}", flush=True)
        probe = None
    started_ns = time.perf_counter_ns()
    error = None
    try:
        return get_stage1_vla_action(
            cfg,
            model,
            device,
            obs,
            task_label,
            exit_controller=exit_controller,
            output_hidden_states=output_hidden_states,
            log_fn=log_fn,
            **kwargs,
        )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        raise
    finally:
        if probe is not None:
            try:
                selected_layer = None
                if runtime is not None and runtime.records:
                    value = runtime.records[-1].get("selected_layer")
                    if value in (13, 27):
                        selected_layer = int(value)
                record = probe.finish_call(
                    selected_layer=selected_layer,
                    outer_policy_wall_ms=(
                        time.perf_counter_ns() - started_ns
                    )
                    / 1_000_000.0,
                    error=error,
                )
                _append_record(record)
            except Exception as measurement_error:
                # Timing must never alter the action returned to robot control.
                print(f"[Stage-11B measurement warning] {measurement_error}", flush=True)


__all__ = ["STAGE11_COMPUTE_ENV", "get_vla_action"]
