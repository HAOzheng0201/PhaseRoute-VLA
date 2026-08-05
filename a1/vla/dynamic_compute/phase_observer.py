"""Failure-contained, causal PhaseEstimator observer for M2 rollouts."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, Mapping, Optional, TextIO, Tuple

import numpy as np
import torch

from .phase_estimator import PhaseEstimatorConfig, PhaseStateEstimator
from .telemetry import instruction_hash


PHASE_OBSERVER_SCHEMA_VERSION = "phase-route-vla.phase-observer-call.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SafePhaseObserver:
    """Run a trained estimator after A1 has produced its current action.

    The current action chunk is appended to history only after prediction, so
    it cannot leak into its own phase estimate.  Exceptions are contained and
    never propagate into robot control.
    """

    enabled = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        *,
        device: str | torch.device,
        history_len: int = 8,
    ):
        if history_len < 1:
            raise ValueError("history_len must be positive")
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        if self.output_path.exists():
            raise FileExistsError(f"Refusing to overwrite observer output: {output_path}")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("schema_version") != (
            "phase-route-vla.phase-estimator-checkpoint.v1"
        ):
            raise ValueError("Unexpected phase-estimator checkpoint schema")
        self.config = PhaseEstimatorConfig(**checkpoint["model_config"])
        self.device = torch.device(device)
        self.model = PhaseStateEstimator(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.history_len = history_len
        self.checkpoint_sha256 = _file_sha256(self.checkpoint_path)
        self.dataset_sha256 = str(checkpoint["dataset_sha256"])
        self.boundary_threshold = float(
            checkpoint.get("validation_boundary_threshold", 0.5)
        )
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.records_written = 0
        self._histories: Dict[
            str, Deque[Tuple[np.ndarray, np.ndarray]]
        ] = defaultdict(lambda: deque(maxlen=self.history_len))
        self._output: Optional[TextIO] = None
        self._lock = threading.Lock()

    def _ensure_open(self) -> TextIO:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._output is None:
            self._output = self.output_path.open("x", encoding="utf-8")
        return self._output

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _tensor(self, value: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(value).to(device=self.device, dtype=dtype)

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
        """Predict phase causally, then append the current transition to history."""

        del raw_proprio, previous_action, action_chunk
        try:
            context = dict(context or {})
            episode_id = str(context["episode_id"])
            step_id = int(context["step_id"])
            current_proprio = np.asarray(normalized_proprio, dtype=np.float32).reshape(-1)
            current_action = np.asarray(
                normalized_action_chunk, dtype=np.float32
            )
            visual = np.asarray(visual_summary, dtype=np.float32).reshape(-1)
            language = np.asarray(instruction_summary, dtype=np.float32).reshape(-1)
            if current_proprio.shape != (self.config.proprio_dim,):
                raise ValueError("normalized_proprio has an invalid shape")
            if current_action.shape != (
                self.config.action_horizon,
                self.config.action_dim,
            ):
                raise ValueError("normalized_action_chunk has an invalid shape")
            if visual.shape != (self.config.visual_summary_dim,):
                raise ValueError("visual_summary has an invalid shape")
            if language.shape != (self.config.instruction_dim,):
                raise ValueError("instruction_summary has an invalid shape")
            if not all(
                np.isfinite(value).all()
                for value in (current_proprio, current_action, visual, language)
            ):
                raise ValueError("phase observer input contains a non-finite value")
            if visual_token_count < 1 or instruction_token_count < 1:
                raise ValueError("summary token counts must be positive")

            history = self._histories[episode_id]
            history_count = len(history)
            proprio_history = np.zeros(
                (1, self.history_len, self.config.proprio_dim), dtype=np.float32
            )
            action_history = np.zeros(
                (
                    1,
                    self.history_len,
                    self.config.action_horizon,
                    self.config.action_dim,
                ),
                dtype=np.float32,
            )
            history_mask = np.zeros((1, self.history_len), dtype=np.bool_)
            start = self.history_len - history_count
            for offset, (past_proprio, past_action) in enumerate(history, start=start):
                proprio_history[0, offset] = past_proprio
                action_history[0, offset] = past_action
                history_mask[0, offset] = True

            self._synchronize()
            start_ns = time.perf_counter_ns()
            with torch.inference_mode():
                state = self.model(
                    visual_summary=self._tensor(visual[None], dtype=torch.float32),
                    instruction_summary=self._tensor(
                        language[None], dtype=torch.float32
                    ),
                    current_proprio=self._tensor(
                        current_proprio[None], dtype=torch.float32
                    ),
                    proprio_history=self._tensor(
                        proprio_history, dtype=torch.float32
                    ),
                    proprio_history_mask=self._tensor(
                        history_mask, dtype=torch.bool
                    ),
                    action_history=self._tensor(action_history, dtype=torch.float32),
                    action_history_mask=self._tensor(history_mask, dtype=torch.bool),
                )
            self._synchronize()
            latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
            progress = float(state.progress[0, 0].detach().cpu())
            boundary_prob = float(state.boundary_prob[0, 0].detach().cpu())
            uncertainty = float(state.uncertainty[0, 0].detach().cpu())
            if not np.isfinite([progress, boundary_prob, uncertainty, latency_ms]).all():
                raise ValueError("phase observer output contains a non-finite value")
            record = {
                "schema_version": PHASE_OBSERVER_SCHEMA_VERSION,
                "episode_id": episode_id,
                "step_id": step_id,
                "task_id": context.get("task_id"),
                "instruction_hash": instruction_hash(instruction),
                "history_count": history_count,
                "history_len": self.history_len,
                "visual_token_count": int(visual_token_count),
                "instruction_token_count": int(instruction_token_count),
                "progress": progress,
                "boundary_prob": boundary_prob,
                "boundary_threshold": self.boundary_threshold,
                "boundary_pred": bool(boundary_prob >= self.boundary_threshold),
                "uncertainty": uncertainty,
                "latency_ms": latency_ms,
                "observer_only": True,
                "controls_early_exit": False,
                "checkpoint_sha256": self.checkpoint_sha256,
                "dataset_sha256": self.dataset_sha256,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            }
            with self._lock:
                output = self._ensure_open()
                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                output.flush()
                self.records_written += 1
            history.append((current_proprio.copy(), current_action.copy()))
            return True
        except Exception as error:
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def close(self) -> None:
        try:
            with self._lock:
                if self._output is not None:
                    self._output.flush()
                    self._output.close()
                    self._output = None
        except Exception as error:
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
