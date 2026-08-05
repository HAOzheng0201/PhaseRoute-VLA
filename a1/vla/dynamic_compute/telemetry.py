"""Zero-impact telemetry types and a failure-contained JSONL logger.

The model may construct :class:`DynamicComputeTelemetry` records and hand them
to a callback. File I/O lives here, outside the model implementation. The safe
logger intentionally contains write errors: observability failures must never
change the action returned to the robot.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, TextIO, Union

import numpy as np


TELEMETRY_SCHEMA_VERSION = "phase-route-vla.telemetry.v1"
TelemetryEventCallback = Callable[[str, Mapping[str, Any]], None]


def instruction_hash(instruction: str, length: int = 16) -> str:
    """Return a stable, privacy-friendlier identifier for an instruction."""

    if length < 8 or length > 64:
        raise ValueError("instruction hash length must be in [8, 64]")
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:length]


def emit_telemetry_event(
    callback: Optional[TelemetryEventCallback],
    event_name: str,
    payload: Mapping[str, Any],
) -> bool:
    """Emit a structured model event without allowing callback failures through."""

    if callback is None:
        return False
    try:
        callback(event_name, payload)
        return True
    except Exception:
        return False


def summarize_vector(value: Any) -> Optional[Dict[str, Any]]:
    """Summarize a CPU array-like value without retaining raw observations."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    flat = array.reshape(-1)
    if flat.size == 0:
        return {"shape": list(array.shape), "count": 0}
    return {
        "shape": list(array.shape),
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2": float(np.linalg.norm(flat)),
    }


def command_motion_summary(previous_action: Any) -> tuple[Optional[float], Optional[float]]:
    """Return translation/rotation norms from a previous 7-D command."""

    if previous_action is None:
        return None, None
    flat = np.asarray(previous_action, dtype=np.float64).reshape(-1)
    if flat.size < 6:
        return None, None
    return float(np.linalg.norm(flat[:3])), float(np.linalg.norm(flat[3:6]))


def _json_safe(value: Any) -> Any:
    """Convert common scalar/container values into strict JSON values.

    Telemetry call sites should pass summaries rather than full tensors. This
    helper deliberately avoids moving tensor-like objects from GPU to CPU,
    which could add a hidden synchronization to the control path.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    # NumPy scalar types expose item(); zero-dimensional CPU tensor scalars do
    # too. Never call it on a non-scalar tensor because that may copy/sync.
    ndim = getattr(value, "ndim", None)
    if ndim == 0 and hasattr(value, "item"):
        return _json_safe(value.item())

    raise TypeError(
        f"Unsupported telemetry value {type(value)!r}; pass a scalar or precomputed summary"
    )


@dataclass
class DynamicComputeTelemetry:
    """One policy-call telemetry record.

    Fields needed only by later PhaseRoute milestones remain optional in M1.
    The schema therefore stays forward-compatible while M1 records the A1
    baseline signals.
    """

    episode_id: Optional[str] = None
    step_id: Optional[int] = None
    task_id: Optional[Union[str, int]] = None
    instruction_hash: Optional[str] = None
    proprio_summary: Optional[Dict[str, Any]] = None
    prev_action_summary: Optional[Dict[str, Any]] = None
    gripper_state: Optional[float] = None
    translation_speed: Optional[float] = None
    rotation_speed: Optional[float] = None

    profile_id: Optional[int] = None
    progress: Optional[float] = None
    boundary_prob: Optional[float] = None
    uncertainty: Optional[float] = None
    agg_tokens: Optional[int] = None
    active_tokens_by_layer: List[int] = field(default_factory=list)
    candidate_exit_layers: List[int] = field(default_factory=list)
    action_delta_by_exit: List[Optional[float]] = field(default_factory=list)
    predicted_exit_error_by_exit: List[Optional[float]] = field(default_factory=list)
    exit_layer: Optional[int] = None
    fm_calls: int = 0
    fm_steps_total: int = 0
    latency_ms: Optional[float] = None

    action_shape: List[int] = field(default_factory=list)
    action_dtype: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    schema_version: str = field(default=TELEMETRY_SCHEMA_VERSION, init=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a strict-JSON-compatible representation of this record."""

        # Avoid dataclasses.asdict(): it deep-copies arbitrary objects before
        # conversion and could therefore clone a tensor accidentally supplied
        # by a caller. _json_safe rejects non-scalar tensors without touching
        # their storage.
        return {
            item.name: _json_safe(getattr(self, item.name))
            for item in fields(self)
        }


def build_policy_call_telemetry(
    *,
    context: Optional[Mapping[str, Any]],
    instruction: str,
    raw_proprio: Any,
    active_token_count: int,
    n_layers: int,
    visual_token_count: int,
    candidate_exit_layers: Sequence[int],
    telemetry_events: Sequence[Mapping[str, Any]],
    latency_ms: float,
    action_shape: Sequence[int],
    action_dtype: str,
    normalization_key: str,
    fallback_exit_layer: Optional[int] = None,
) -> DynamicComputeTelemetry:
    """Build one aligned policy-call record from structured model events."""

    context = dict(context or {})
    events = [dict(event) for event in telemetry_events]
    previous_action = context.get("previous_action")
    translation_speed, rotation_speed = command_motion_summary(previous_action)
    candidate_layers = [int(layer) for layer in candidate_exit_layers]
    delta_by_layer = {
        int(event["layer_idx"]): event.get("action_delta")
        for event in events
        if event.get("event") == "exit_candidate"
        and event.get("evaluated")
        and event.get("layer_idx") is not None
    }
    selected_events = [
        event
        for event in events
        if event.get("should_exit") or event.get("event") == "forced_exit"
    ]
    exit_layer = fallback_exit_layer
    if exit_layer is None and selected_events:
        exit_layer = int(selected_events[-1]["layer_idx"])
    phase_events = [event for event in events if event.get("event") == "phase_plan"]
    phase_plan = phase_events[-1] if phase_events else {}
    vision_events = [
        event for event in events if event.get("event") == "vision_aggregation"
    ]
    vision_aggregation = vision_events[-1] if vision_events else {}
    effective_active_token_count = int(
        vision_aggregation.get("active_tokens", active_token_count)
    )

    gripper_state = None
    if raw_proprio is not None:
        raw_flat = np.asarray(raw_proprio).reshape(-1)
        if raw_flat.size >= 2:
            gripper_state = float(abs(raw_flat[-2] - raw_flat[-1]))

    return DynamicComputeTelemetry(
        episode_id=context.get("episode_id"),
        step_id=context.get("step_id"),
        task_id=context.get("task_id"),
        instruction_hash=instruction_hash(instruction),
        proprio_summary=summarize_vector(raw_proprio),
        prev_action_summary=summarize_vector(previous_action),
        gripper_state=gripper_state,
        translation_speed=translation_speed,
        rotation_speed=rotation_speed,
        profile_id=phase_plan.get("profile_id"),
        progress=phase_plan.get("progress"),
        boundary_prob=phase_plan.get("boundary_prob"),
        uncertainty=phase_plan.get("uncertainty"),
        active_tokens_by_layer=[effective_active_token_count] * int(n_layers),
        candidate_exit_layers=candidate_layers,
        action_delta_by_exit=[delta_by_layer.get(layer) for layer in candidate_layers],
        predicted_exit_error_by_exit=[None] * len(candidate_layers),
        exit_layer=exit_layer,
        fm_calls=sum(int(event.get("fm_calls", 0)) for event in events),
        fm_steps_total=sum(int(event.get("fm_steps", 0)) for event in events),
        latency_ms=float(latency_ms),
        action_shape=[int(size) for size in action_shape],
        action_dtype=action_dtype,
        extra={
            "visual_tokens": int(visual_token_count),
            "exit_events": events,
            "phase_plan": phase_plan or None,
            "vision_aggregation": vision_aggregation or None,
            "normalization_key": normalization_key,
        },
    )


class TelemetryLogger(Protocol):
    """Small side-channel interface accepted by rollout code."""

    def log(self, record: DynamicComputeTelemetry) -> bool:
        """Write one record and return whether it was accepted."""

    def close(self) -> None:
        """Release logger resources."""


class NullTelemetryLogger:
    """Disabled logger with no filesystem or timing side effects."""

    enabled = False
    error_count = 0
    last_error: Optional[str] = None

    def log(self, record: DynamicComputeTelemetry) -> bool:
        del record
        return False

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullTelemetryLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class SafeJSONLTelemetryLogger:
    """Append-only JSONL logger that never raises into the control path.

    The output file is opened lazily on the first record. Parent directories
    are not created implicitly: a misspelled run directory should be visible
    through ``last_error`` instead of silently creating a new result tree.
    """

    enabled = True

    def __init__(self, path: Union[str, Path], flush_every: int = 100):
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.path = Path(path)
        self.flush_every = flush_every
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.records_written = 0
        self._file: Optional[TextIO] = None
        self._lock = threading.Lock()

    def _ensure_open(self) -> TextIO:
        if self._file is None:
            self._file = self.path.open("a", encoding="utf-8")
        return self._file

    def log(self, record: DynamicComputeTelemetry) -> bool:
        try:
            payload = record.to_dict()
            line = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            with self._lock:
                output = self._ensure_open()
                output.write(line + "\n")
                self.records_written += 1
                if self.records_written % self.flush_every == 0:
                    output.flush()
            return True
        except Exception as error:  # telemetry must be failure-contained
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def close(self) -> None:
        try:
            with self._lock:
                if self._file is not None:
                    self._file.flush()
                    self._file.close()
                    self._file = None
        except Exception as error:  # closing telemetry cannot break a rollout
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"

    def __enter__(self) -> "SafeJSONLTelemetryLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
