"""Opt-in M2 feature summaries and a failure-contained phase-cache writer.

The model only computes summaries and emits them to a callback.  All CPU
copies and filesystem I/O stay in the rollout-side callback/writer and are
therefore absent from the default A1 inference path.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, TextIO, Union

import numpy as np
import torch

from .telemetry import instruction_hash


PHASE_CACHE_SCHEMA_VERSION = "phase-route-vla.phase-cache-call.v1"
PhaseSignalCallback = Callable[[Mapping[str, Any]], None]


def emit_phase_signal_summary(
    callback: Optional[PhaseSignalCallback],
    payload: Mapping[str, Any],
) -> bool:
    """Emit tensors to an opt-in collector without changing model control flow."""

    if callback is None:
        return False
    try:
        callback(payload)
        return True
    except Exception:
        return False


def _base_instruction_mask(
    input_ids: torch.Tensor,
    response_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, S]")
    mask = input_ids != -1
    if response_mask is not None:
        if response_mask.shape != input_ids.shape:
            raise ValueError("response_mask must match input_ids")
        mask = mask & ~(response_mask > 0)
    return mask


def summarize_instruction_embeddings(
    token_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    image_input_idx: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    proprio_token_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool valid instruction-side embeddings.

    Visual insertion locations, response/action tokens, proprio locations and
    padding are excluded.  The result has shape ``[B, D]`` and the returned
    token counts have shape ``[B]``.
    """

    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [B, S, D]")
    batch_size, sequence_length, _ = token_embeddings.shape
    if input_ids.shape != (batch_size, sequence_length):
        raise ValueError("input_ids must align with token_embeddings")
    mask = _base_instruction_mask(input_ids, response_mask).clone()

    if image_input_idx is not None:
        if image_input_idx.shape[0] != batch_size:
            raise ValueError("image_input_idx batch dimension must match")
        flat_indices = image_input_idx.reshape(batch_size, -1)
        valid = flat_indices >= 0
        if valid.any():
            if int(flat_indices[valid].max()) >= sequence_length:
                raise ValueError("image_input_idx contains an out-of-range position")
            batch_indices = torch.arange(batch_size, device=mask.device)[:, None]
            batch_indices = batch_indices.expand_as(flat_indices)
            mask[batch_indices[valid], flat_indices[valid]] = False

    if proprio_token_idx is not None:
        if proprio_token_idx.shape[0] != batch_size:
            raise ValueError("proprio_token_idx batch dimension must match")
        flat_indices = proprio_token_idx.reshape(batch_size, -1)
        valid = flat_indices >= 0
        if valid.any():
            if int(flat_indices[valid].max()) >= sequence_length:
                raise ValueError("proprio_token_idx contains an out-of-range position")
            batch_indices = torch.arange(batch_size, device=mask.device)[:, None]
            batch_indices = batch_indices.expand_as(flat_indices)
            mask[batch_indices[valid], flat_indices[valid]] = False

    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("at least one sample has no instruction-side tokens")
    weights = mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    summary = (token_embeddings * weights).sum(dim=1) / counts.unsqueeze(-1).to(
        dtype=token_embeddings.dtype
    )
    return summary, counts


def summarize_visual_embeddings(
    image_features: torch.Tensor,
    image_input_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool only visual features that are inserted into the VLM sequence."""

    if image_features.ndim not in {3, 4}:
        raise ValueError("image_features must have shape [B, M, D] or [B, N, M, D]")
    batch_size = image_features.shape[0]
    flat_features = image_features.reshape(batch_size, -1, image_features.shape[-1])
    if image_input_idx.shape[0] != batch_size:
        raise ValueError("image_input_idx batch dimension must match")
    flat_indices = image_input_idx.reshape(batch_size, -1)
    if flat_indices.shape[1] != flat_features.shape[1]:
        raise ValueError("image_input_idx must align with visual features")
    mask = flat_indices >= 0
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("at least one sample has no valid visual tokens")
    weights = mask.unsqueeze(-1).to(dtype=flat_features.dtype)
    summary = (flat_features * weights).sum(dim=1) / counts.unsqueeze(-1).to(
        dtype=flat_features.dtype
    )
    return summary, counts


def _finite_array(value: Any, name: str, dtype: np.dtype) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


@dataclass
class PhaseCacheCallMetadata:
    """JSON metadata aligned with one compressed array shard."""

    array_path: str
    episode_id: str
    step_id: int
    task_id: Optional[Union[str, int]]
    instruction_hash: str
    previous_action_present: bool
    shapes: Dict[str, list[int]]
    dtypes: Dict[str, str]
    summary_counts: Dict[str, int]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    schema_version: str = field(default=PHASE_CACHE_SCHEMA_VERSION, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class PhaseCacheWriter(Protocol):
    enabled: bool
    error_count: int
    last_error: Optional[str]

    def log_call(self, **kwargs: Any) -> bool:
        """Persist one policy-call feature record."""

    def close(self) -> None:
        """Flush and release writer resources."""


class NullPhaseCacheWriter:
    enabled = False
    error_count = 0
    last_error: Optional[str] = None
    records_written = 0

    def log_call(self, **kwargs: Any) -> bool:
        del kwargs
        return False

    def close(self) -> None:
        return None


class SafePhaseCacheWriter:
    """Write per-call NPZ shards plus an append-only manifest.

    The writer never raises into robot control.  Each NPZ shard is installed
    atomically before its manifest line is appended.  Existing shards are
    never overwritten, which makes a cache directory immutable by default.
    """

    enabled = True

    def __init__(self, output_dir: Union[str, Path], summary_dtype: str = "float16"):
        if summary_dtype not in {"float16", "float32"}:
            raise ValueError("summary_dtype must be float16 or float32")
        self.output_dir = Path(output_dir)
        self.arrays_dir = self.output_dir / "arrays"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_dtype = np.dtype(summary_dtype)
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
        raw_proprio: Any,
        normalized_proprio: Any,
        previous_action: Any,
        normalized_action_chunk: Any,
        action_chunk: Any,
        visual_summary: Any,
        instruction_summary: Any,
        visual_token_count: int,
        instruction_token_count: int,
    ) -> bool:
        try:
            context = dict(context or {})
            episode_id = str(context["episode_id"])
            step_id = int(context["step_id"])
            arrays = {
                "raw_proprio": _finite_array(raw_proprio, "raw_proprio", np.float32),
                "normalized_proprio": _finite_array(
                    normalized_proprio, "normalized_proprio", np.float32
                ),
                "previous_action": _finite_array(
                    previous_action, "previous_action", np.float32
                ),
                "normalized_action_chunk": _finite_array(
                    normalized_action_chunk, "normalized_action_chunk", np.float32
                ),
                "action_chunk": _finite_array(action_chunk, "action_chunk", np.float32),
                "visual_summary": _finite_array(
                    visual_summary, "visual_summary", self.summary_dtype
                ),
                "instruction_summary": _finite_array(
                    instruction_summary, "instruction_summary", self.summary_dtype
                ),
            }
            required = {
                "raw_proprio",
                "normalized_proprio",
                "normalized_action_chunk",
                "action_chunk",
                "visual_summary",
                "instruction_summary",
            }
            for name in required:
                if arrays[name].size == 0:
                    raise ValueError(f"{name} cannot be empty")
            if visual_token_count < 1 or instruction_token_count < 1:
                raise ValueError("summary token counts must be positive")

            with self._lock:
                manifest = self._ensure_open()
                final_path = self.arrays_dir / f"call_{self.records_written:06d}.npz"
                if final_path.exists():
                    raise FileExistsError(f"refusing to overwrite {final_path}")
                temporary_path: Optional[Path] = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w+b",
                        prefix="phase-call-",
                        suffix=".npz.tmp",
                        dir=self.arrays_dir,
                        delete=False,
                    ) as temporary:
                        temporary_path = Path(temporary.name)
                        np.savez_compressed(temporary, **arrays)
                        temporary.flush()
                    temporary_path.replace(final_path)
                except Exception:
                    # Do not remove a possibly useful partial file here; cache
                    # auditing reports any orphan explicitly.
                    raise

                metadata = PhaseCacheCallMetadata(
                    array_path=str(final_path.relative_to(self.output_dir)),
                    episode_id=episode_id,
                    step_id=step_id,
                    task_id=context.get("task_id"),
                    instruction_hash=instruction_hash(instruction),
                    previous_action_present=arrays["previous_action"].size > 0,
                    shapes={name: list(array.shape) for name, array in arrays.items()},
                    dtypes={name: str(array.dtype) for name, array in arrays.items()},
                    summary_counts={
                        "visual_tokens": int(visual_token_count),
                        "instruction_tokens": int(instruction_token_count),
                    },
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
