"""Opt-in projected-vision teacher cache for learnable EFA training.

The model emits tensors through a callback, while all device transfers and
filesystem writes remain in the rollout-side collector.  A writer failure is
recorded but never allowed to alter the robot action.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, TextIO, Union

import numpy as np
import torch

from .telemetry import instruction_hash


VISION_TEACHER_CACHE_SCHEMA_VERSION_V1 = "phase-route-vla.vision-teacher-call.v1"
VISION_TEACHER_CACHE_SCHEMA_VERSION_V2 = "phase-route-vla.vision-teacher-call.v2"
VISION_TEACHER_CACHE_SCHEMA_VERSION = "phase-route-vla.vision-teacher-call.v3"
SUPPORTED_VISION_TEACHER_CACHE_SCHEMA_VERSIONS = frozenset(
    {
        VISION_TEACHER_CACHE_SCHEMA_VERSION_V1,
        VISION_TEACHER_CACHE_SCHEMA_VERSION_V2,
        VISION_TEACHER_CACHE_SCHEMA_VERSION,
    }
)
VisionTeacherFeatureCallback = Callable[[Mapping[str, Any]], None]
FlowMatchingTraceCallback = Callable[[Mapping[str, Any]], None]


def has_complete_candidate_fm_traces(record: Mapping[str, Any]) -> bool:
    """Validate both historical and max-depth FM accounting conventions.

    Early exits historically report ``fm_calls`` as candidate calls and keep
    the comparison trace as one extra row.  At the terminal layer the current
    controller reports the comparison in ``fm_calls`` as well.  Both are
    complete when every trace is accounted for and the terminal candidate is
    the teacher exit.
    """

    try:
        fm_calls = int(record["fm_calls"])
        trace_count = int(record["fm_trace_count"])
        candidate_count = int(record["candidate_trace_count"])
        comparison_count = int(record["comparison_trace_count"])
        candidate_layers = [int(value) for value in record["candidate_layers"]]
        teacher_layer = int(record["teacher_exit_layer"])
        shapes = record["shapes"]
        return (
            candidate_count >= 1
            and comparison_count >= 0
            and candidate_count + comparison_count == trace_count
            and fm_calls in (candidate_count, trace_count)
            and len(candidate_layers) == candidate_count
            and candidate_layers[-1] == teacher_layer
            and int(shapes["fm_trace_layers"][0]) == trace_count
            and int(shapes["fm_trace_roles"][0]) == trace_count
            and int(shapes["fm_trace_steps"][0]) == trace_count
            and int(shapes["fm_trace_input_x"][0]) == trace_count
            and int(shapes["fm_trace_output_action"][0]) == trace_count
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def emit_vision_teacher_features(
    callback: Optional[VisionTeacherFeatureCallback],
    payload: Mapping[str, Any],
) -> bool:
    """Emit projected features without allowing observer errors to escape."""

    if callback is None:
        return False
    try:
        callback(payload)
        return True
    except Exception:
        return False


def emit_flow_matching_trace(
    callback: Optional[FlowMatchingTraceCallback],
    payload: Mapping[str, Any],
) -> bool:
    """Emit one tiny FM trace without allowing collection to affect control."""

    if callback is None:
        return False
    try:
        callback(payload)
        return True
    except Exception:
        return False


def _finite_array(value: Any, name: str, dtype: np.dtype) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _rng_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.uint8)
    array = np.asarray(value, dtype=np.uint8).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


@dataclass
class VisionTeacherCallMetadata:
    array_path: str
    episode_id: str
    step_id: int
    task_id: Optional[Union[str, int]]
    instruction_hash: str
    teacher_kind: str
    checkpoint_sha256: Optional[str]
    teacher_exit_layer: Optional[int]
    fm_calls: int
    fm_steps_total: int
    source_projected_tokens: int
    unique_visual_slots: int
    valid_crop_count: int
    sequence_length: int
    fm_trace_count: int
    candidate_trace_count: int
    comparison_trace_count: int
    candidate_layers: list[int]
    teacher_trace_max_abs_error: float
    shapes: Dict[str, list[int]]
    dtypes: Dict[str, str]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
    )
    schema_version: str = field(
        default=VISION_TEACHER_CACHE_SCHEMA_VERSION,
        init=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class VisionTeacherCacheWriter(Protocol):
    enabled: bool
    error_count: int
    last_error: Optional[str]
    records_written: int

    def log_call(self, **kwargs: Any) -> bool:
        """Persist one aligned teacher record."""

    def close(self) -> None:
        """Flush and release writer resources."""


class NullVisionTeacherCacheWriter:
    enabled = False
    error_count = 0
    last_error: Optional[str] = None
    records_written = 0

    def log_call(self, **kwargs: Any) -> bool:
        del kwargs
        return False

    def close(self) -> None:
        return None


class SafeVisionTeacherCacheWriter:
    """Atomically persist projected features and their teacher action."""

    enabled = True

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        feature_dtype: str = "float16",
        teacher_kind: str = "a1_early_exit",
        checkpoint_sha256: Optional[str] = None,
    ):
        if feature_dtype not in {"float16", "float32"}:
            raise ValueError("feature_dtype must be float16 or float32")
        if not teacher_kind.strip():
            raise ValueError("teacher_kind cannot be empty")
        if checkpoint_sha256 is not None:
            if len(checkpoint_sha256) != 64:
                raise ValueError("checkpoint_sha256 must have 64 hexadecimal characters")
            int(checkpoint_sha256, 16)
        self.output_dir = Path(output_dir)
        self.arrays_dir = self.output_dir / "arrays"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.feature_dtype = np.dtype(feature_dtype)
        self.teacher_kind = teacher_kind
        self.checkpoint_sha256 = checkpoint_sha256
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.records_written = 0
        self._manifest: Optional[TextIO] = None
        self._lock = threading.Lock()

    def _ensure_open(self) -> TextIO:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        if self._manifest is None:
            self._manifest = self.manifest_path.open("a", encoding="utf-8")
        return self._manifest

    def log_call(
        self,
        *,
        context: Optional[Mapping[str, Any]],
        instruction: str,
        projected_features: Any,
        image_input_idx: Any,
        instruction_summary: Any,
        normalized_proprio: Any,
        teacher_normalized_action: Any,
        teacher_action: Any,
        cpu_rng_state: Any,
        cuda_rng_state: Any,
        teacher_exit_layer: Optional[int],
        fm_calls: int,
        fm_steps_total: int,
        input_ids: Any,
        attention_mask: Any,
        attention_bias: Any,
        response_mask: Any,
        subsegment_ids: Any,
        position_ids: Any,
        action_proprio: Any,
        proprio_token_idx: Any,
        teacher_exit_input_x: Any,
        teacher_exit_trace_action: Any,
        fm_trace_count: int,
        fm_trace_layers: Any,
        fm_trace_roles: Any,
        fm_trace_steps: Any,
        fm_trace_input_x: Any,
        fm_trace_output_action: Any,
    ) -> bool:
        try:
            context = dict(context or {})
            episode_id = str(context["episode_id"])
            step_id = int(context["step_id"])
            arrays = {
                "projected_features": _finite_array(
                    projected_features,
                    "projected_features",
                    self.feature_dtype,
                ),
                "image_input_idx": np.asarray(image_input_idx, dtype=np.int32),
                "instruction_summary": _finite_array(
                    instruction_summary,
                    "instruction_summary",
                    self.feature_dtype,
                ),
                "normalized_proprio": _finite_array(
                    normalized_proprio,
                    "normalized_proprio",
                    np.float32,
                ),
                "teacher_normalized_action": _finite_array(
                    teacher_normalized_action,
                    "teacher_normalized_action",
                    np.float32,
                ),
                "teacher_action": _finite_array(
                    teacher_action,
                    "teacher_action",
                    np.float32,
                ),
                "cpu_rng_state": _rng_array(cpu_rng_state, "cpu_rng_state"),
                "cuda_rng_state": _rng_array(cuda_rng_state, "cuda_rng_state"),
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "attention_mask": np.asarray(attention_mask, dtype=np.bool_),
                "attention_bias": np.asarray(attention_bias, dtype=np.float32),
                "response_mask": np.asarray(response_mask, dtype=np.bool_),
                "subsegment_ids": np.asarray(subsegment_ids, dtype=np.int64),
                "position_ids": np.asarray(position_ids, dtype=np.int64),
                "action_proprio": _finite_array(
                    action_proprio,
                    "action_proprio",
                    np.float32,
                ),
                "proprio_token_idx": np.asarray(proprio_token_idx, dtype=np.int64).reshape(-1),
                "teacher_exit_input_x": _finite_array(
                    teacher_exit_input_x,
                    "teacher_exit_input_x",
                    np.float32,
                ),
                "teacher_exit_trace_action": _finite_array(
                    teacher_exit_trace_action,
                    "teacher_exit_trace_action",
                    np.float32,
                ),
                "fm_trace_layers": np.asarray(fm_trace_layers, dtype=np.int16).reshape(-1),
                "fm_trace_roles": np.asarray(fm_trace_roles, dtype=np.uint8).reshape(-1),
                "fm_trace_steps": np.asarray(fm_trace_steps, dtype=np.int16).reshape(-1),
                "fm_trace_input_x": _finite_array(
                    fm_trace_input_x,
                    "fm_trace_input_x",
                    np.float32,
                ),
                "fm_trace_output_action": _finite_array(
                    fm_trace_output_action,
                    "fm_trace_output_action",
                    np.float32,
                ),
            }
            features = arrays["projected_features"]
            positions = arrays["image_input_idx"]
            if features.ndim != 3:
                raise ValueError("projected_features must have shape [C, M, D]")
            if positions.shape != features.shape[:2]:
                raise ValueError("image_input_idx must have shape [C, M]")
            if not np.issubdtype(positions.dtype, np.integer):
                raise ValueError("image_input_idx must be integral")
            valid = positions >= 0
            source_count = int(valid.sum())
            if source_count < 1:
                raise ValueError("projected feature cache has no valid tokens")
            if arrays["instruction_summary"].shape != (features.shape[-1],):
                raise ValueError("instruction_summary must match projected hidden size")
            if arrays["normalized_proprio"].ndim != 1:
                raise ValueError("normalized_proprio must be one-dimensional")
            normalized_action = arrays["teacher_normalized_action"]
            action = arrays["teacher_action"]
            if normalized_action.ndim != 2 or normalized_action.shape != action.shape:
                raise ValueError("teacher actions must have the same [H, A] shape")
            if fm_calls < 0 or fm_steps_total < 0:
                raise ValueError("FM counts must be non-negative")
            if fm_trace_count < 1:
                raise ValueError("fm_trace_count must be positive")
            if teacher_exit_layer is None or teacher_exit_layer < 0:
                raise ValueError("teacher_exit_layer must be non-negative")

            cached_input_ids = arrays["input_ids"]
            if cached_input_ids.ndim != 1 or cached_input_ids.size < 1:
                raise ValueError("input_ids must be a non-empty one-dimensional sequence")
            sequence_length = int(cached_input_ids.size)
            for name in ("attention_mask", "response_mask", "subsegment_ids", "position_ids"):
                value = arrays[name]
                if value.size and value.shape != (sequence_length,):
                    raise ValueError(f"{name} must be empty or have shape [L]")
            cached_attention_bias = arrays["attention_bias"]
            if cached_attention_bias.size and (
                cached_attention_bias.ndim < 2
                or cached_attention_bias.shape[-2:] != (sequence_length, sequence_length)
            ):
                raise ValueError(
                    "attention_bias must be empty or end with shape [L, L]"
                )
            cached_proprio = arrays["action_proprio"]
            if cached_proprio.ndim not in (1, 2, 3) or cached_proprio.shape[-1] != arrays["normalized_proprio"].size:
                raise ValueError("action_proprio must end with the proprio dimension")
            cached_proprio_idx = arrays["proprio_token_idx"]
            if cached_proprio_idx.shape != (1,):
                raise ValueError("proprio_token_idx must contain exactly one index")
            if not 0 <= int(cached_proprio_idx[0]) < sequence_length:
                raise ValueError("proprio_token_idx is outside input_ids")
            trace_input = arrays["teacher_exit_input_x"]
            trace_action = arrays["teacher_exit_trace_action"]
            if trace_input.shape != normalized_action.shape:
                raise ValueError("teacher_exit_input_x must have teacher action shape")
            if trace_action.shape != normalized_action.shape:
                raise ValueError("teacher_exit_trace_action must have teacher action shape")
            trace_error = float(np.max(np.abs(trace_action - normalized_action)))
            if trace_error > 1e-5:
                raise ValueError(
                    "teacher exit trace action does not align with teacher_normalized_action"
                )
            trace_layers = arrays["fm_trace_layers"]
            trace_roles = arrays["fm_trace_roles"]
            trace_steps = arrays["fm_trace_steps"]
            trace_inputs = arrays["fm_trace_input_x"]
            trace_outputs = arrays["fm_trace_output_action"]
            if not (
                trace_layers.shape
                == trace_roles.shape
                == trace_steps.shape
                == (fm_trace_count,)
            ):
                raise ValueError("FM trace metadata must have shape [N]")
            expected_trace_shape = (fm_trace_count, *normalized_action.shape)
            if trace_inputs.shape != expected_trace_shape:
                raise ValueError("fm_trace_input_x must have shape [N, H, A]")
            if trace_outputs.shape != expected_trace_shape:
                raise ValueError("fm_trace_output_action must have shape [N, H, A]")
            if not np.isin(trace_roles, np.array([0, 1], dtype=np.uint8)).all():
                raise ValueError("fm_trace_roles may only contain comparison=0 or candidate=1")
            if (trace_layers < 0).any() or (trace_steps < 1).any():
                raise ValueError("FM trace layers and steps must be positive")
            candidate_mask = trace_roles == 1
            comparison_mask = trace_roles == 0
            candidate_layers = trace_layers[candidate_mask].astype(np.int64)
            if candidate_layers.size < 1:
                raise ValueError("FM trace must contain at least one candidate action")
            if len(np.unique(candidate_layers)) != candidate_layers.size:
                raise ValueError("candidate FM trace layers must be unique")
            if not np.all(candidate_layers[:-1] < candidate_layers[1:]):
                raise ValueError("candidate FM trace layers must be increasing")
            exit_matches = np.flatnonzero(
                candidate_mask & (trace_layers == int(teacher_exit_layer))
            )
            if exit_matches.size != 1:
                raise ValueError("FM trace must contain exactly one teacher exit candidate")
            exit_trace_index = int(exit_matches[0])
            if not np.array_equal(trace_inputs[exit_trace_index], trace_input):
                raise ValueError("explicit teacher exit input does not match FM trace")
            if not np.array_equal(trace_outputs[exit_trace_index], trace_action):
                raise ValueError("explicit teacher exit action does not match FM trace")

            valid_positions = positions[valid]
            unique_slots = int(np.unique(valid_positions).size)
            valid_crop_count = int(np.any(valid, axis=1).sum())
            with self._lock:
                manifest = self._ensure_open()
                final_path = self.arrays_dir / f"call_{self.records_written:06d}.npz"
                if final_path.exists():
                    raise FileExistsError(f"refusing to overwrite {final_path}")
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix="vision-teacher-call-",
                    suffix=".npz.tmp",
                    dir=self.arrays_dir,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    # Projected features are high-entropy float tensors; ZIP
                    # compression costs substantial rollout CPU for little gain.
                    np.savez(temporary, **arrays)
                    temporary.flush()
                temporary_path.replace(final_path)

                metadata = VisionTeacherCallMetadata(
                    array_path=str(final_path.relative_to(self.output_dir)),
                    episode_id=episode_id,
                    step_id=step_id,
                    task_id=context.get("task_id"),
                    instruction_hash=instruction_hash(instruction),
                    teacher_kind=self.teacher_kind,
                    checkpoint_sha256=self.checkpoint_sha256,
                    teacher_exit_layer=(
                        int(teacher_exit_layer)
                        if teacher_exit_layer is not None
                        else None
                    ),
                    fm_calls=int(fm_calls),
                    fm_steps_total=int(fm_steps_total),
                    source_projected_tokens=source_count,
                    unique_visual_slots=unique_slots,
                    valid_crop_count=valid_crop_count,
                    sequence_length=sequence_length,
                    fm_trace_count=int(fm_trace_count),
                    candidate_trace_count=int(candidate_mask.sum()),
                    comparison_trace_count=int(comparison_mask.sum()),
                    candidate_layers=[int(value) for value in candidate_layers],
                    teacher_trace_max_abs_error=trace_error,
                    shapes={name: list(array.shape) for name, array in arrays.items()},
                    dtypes={name: str(array.dtype) for name, array in arrays.items()},
                )
                manifest.write(
                    json.dumps(
                        metadata.to_dict(),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                manifest.flush()
                self.records_written += 1
            return True
        except Exception as error:
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def close(self) -> None:
        try:
            with self._lock:
                if self._manifest is not None:
                    self._manifest.flush()
                    self._manifest.close()
                    self._manifest = None
        except Exception as error:
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
