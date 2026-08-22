"""Online construction of the frozen V3-D8 PhaseRoute runtime context.

This module is the only bridge from a live A1 policy call to the nine frozen
runtime-context tensors used by the D8 router.  Episode/task identities are
kept exclusively as telemetry metadata and can never enter the 97-D feature.
All estimator/router work is performed on detached CPU tensors.  Any malformed
or missing signal leaves the already-started adapter in its fail-closed state,
which vetoes L11/L13 and preserves the exact L27 action.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from a1.vla.dynamic_compute.phase_estimator import (
    PhaseEstimatorConfig,
    PhaseState,
    PhaseStateEstimator,
)

from .development_collection import (
    D2_PHASE_CHECKPOINT_SHA256,
    pool_visual_features,
    validate_runtime_context,
)
from .final_router import (
    D8B_PAYLOAD_SCHEMA_VERSION,
    FinalFiveHeadRouter,
    final_router_from_mapping,
)
from .runtime_adapter import EpisodePastOnlyHistory, FrozenD8RuntimeAdapter


D9B_ACTIVE_RUNTIME_SCHEMA_VERSION = "phase-route-vla.v3.d9b-active-runtime.v1"
D9B_ROUTER_CHECKPOINT_SHA256 = (
    "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830"
)
PHASE_CHECKPOINT_SCHEMA_VERSION = "phase-route-vla.phase-estimator-checkpoint.v1"


class ActiveRuntimeError(ValueError):
    """Raised by strict loaders; live policy-call failures stay fail-closed."""


def sha256_file(path: str | Path) -> str:
    target = Path(path).resolve(strict=True)
    if not target.is_file():
        raise ActiveRuntimeError(f"artifact is not a regular file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_phase_state_sha256(model: PhaseStateEstimator) -> str:
    """Hash the exact frozen phase-estimator parameter/buffer state."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_cpu_float32(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ActiveRuntimeError(f"{name} must have shape {list(shape)}")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ActiveRuntimeError(f"{name} must be finite")
    return result


def _cached_visual_precision(value: Any) -> torch.Tensor:
    """Reproduce D2's projected-feature float16 cache boundary exactly."""

    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
        1,
        5,
        144,
        3584,
    ):
        raise ActiveRuntimeError(
            "projected_features must have shape [1,5,144,3584]"
        )
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ActiveRuntimeError("projected_features must be finite floating point")
    # Raw D2/D3/D8c collection used `.to(cpu, float16).numpy()` before context
    # construction cast the cache back to float32.  Preserve that numerical
    # boundary online so the frozen router sees its training-time protocol.
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float16)
        .to(dtype=torch.float32)
        .contiguous()
    )
    if not bool(torch.isfinite(result).all()):
        raise ActiveRuntimeError("projected_features overflowed float16 cache precision")
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class ActiveRuntimeArtifacts:
    router_path: str
    router_sha256: str
    phase_checkpoint_path: str
    phase_checkpoint_sha256: str
    phase_state_sha256: str


class ActivePhaseRouteRuntime:
    """Build one causal nine-tensor context for each live policy call."""

    enabled = True

    def __init__(
        self,
        router: FinalFiveHeadRouter,
        phase_estimator: PhaseStateEstimator,
        *,
        artifacts: ActiveRuntimeArtifacts | None = None,
    ) -> None:
        router.validate()
        if phase_estimator.config != PhaseEstimatorConfig():
            raise ActiveRuntimeError("phase-estimator geometry differs from D8")
        if any(parameter.device.type != "cpu" for parameter in phase_estimator.parameters()):
            raise ActiveRuntimeError("phase estimator must remain on CPU")
        phase_estimator.eval()
        for parameter in phase_estimator.parameters():
            parameter.requires_grad_(False)
        self.phase_estimator = phase_estimator
        self.phase_state_sha256 = frozen_phase_state_sha256(phase_estimator)
        if artifacts is not None and artifacts.phase_state_sha256 != self.phase_state_sha256:
            raise ActiveRuntimeError("phase-estimator state SHA-256 differs")
        self.artifacts = artifacts
        self.adapter = FrozenD8RuntimeAdapter(router)
        self.history = EpisodePastOnlyHistory()

        self.error_count = 0
        self.last_error: str | None = None
        self.policy_calls = 0
        self.prepared_calls = 0
        self.committed_calls = 0
        self._episode_id: str | None = None
        self._current: dict[str, Any] | None = None
        self._history_window = None
        self._instruction_summary: torch.Tensor | None = None
        self._normalized_proprio: torch.Tensor | None = None
        self._projected_features: torch.Tensor | None = None
        self._image_input_idx: torch.Tensor | None = None
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(_json_safe(record) for record in self._records)

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    def _fail(self, error: Exception | str, *, stage: str) -> None:
        text = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self.error_count += 1
        self.last_error = text
        if self._current is not None:
            self._current["fallback"] = True
            self._current["errors"].append({"stage": stage, "error": text})
        # Re-starting with missing context deliberately latches the adapter's
        # fail-closed path for the current policy call.
        self.adapter.begin_policy_call(None)

    @staticmethod
    def _validate_episode_id(episode_id: Any) -> str:
        if type(episode_id) is not str or not episode_id:
            raise ActiveRuntimeError("episode_id must be a nonempty string")
        return episode_id

    def start_episode(self, episode_id: str) -> None:
        episode = self._validate_episode_id(episode_id)
        if self._current is not None:
            raise ActiveRuntimeError("cannot reset episode during a policy call")
        self.history.reset()
        self._episode_id = episode

    def begin_policy_call(
        self,
        *,
        context: Mapping[str, Any],
        instruction_summary: Any,
        normalized_proprio: Any,
    ) -> bool:
        """Open one call and install a fail-closed placeholder immediately."""

        if self._current is not None:
            self._fail("previous policy call was not committed", stage="begin")
            return False
        self.adapter.begin_policy_call(None)
        self._history_window = None
        self._instruction_summary = None
        self._normalized_proprio = None
        self._projected_features = None
        self._image_input_idx = None
        started_ns = time.perf_counter_ns()
        record = {
            "schema_version": D9B_ACTIVE_RUNTIME_SCHEMA_VERSION,
            "context": _json_safe(dict(context)) if isinstance(context, Mapping) else {},
            "prepared": False,
            "committed": False,
            "fallback": True,
            "selected_layer": None,
            "events": [],
            "errors": [],
            "begin_ns": started_ns,
        }
        self._records.append(record)
        self._current = record
        self.policy_calls += 1
        try:
            if not isinstance(context, Mapping):
                raise ActiveRuntimeError("policy context must be a mapping")
            episode_id = self._validate_episode_id(context.get("episode_id"))
            call_ordinal = context.get("call_ordinal")
            step_id = context.get("step_id")
            task_id = context.get("task_id")
            if type(call_ordinal) is not int or call_ordinal < 0:
                raise ActiveRuntimeError("call_ordinal must be a non-negative integer")
            if type(step_id) is not int or step_id < 0:
                raise ActiveRuntimeError("step_id must be a non-negative integer")
            if type(task_id) is not int or task_id < 0:
                raise ActiveRuntimeError("task_id must be a non-negative integer")
            if episode_id != self._episode_id:
                raise ActiveRuntimeError("policy call does not match active episode")
            proprio = np.asarray(normalized_proprio, dtype=np.float32).reshape(-1)
            if proprio.shape != (8,) or not np.isfinite(proprio).all():
                raise ActiveRuntimeError("normalized_proprio must be finite [8]")
            self._history_window = self.history.window(
                episode_id, call_ordinal, proprio
            )
            self._normalized_proprio = torch.from_numpy(proprio.copy()).unsqueeze(0)
            self._instruction_summary = _finite_cpu_float32(
                instruction_summary, (1, 3584), "instruction_summary"
            )
            record["identity_is_runtime_input"] = False
            record["begin_ok"] = True
            return True
        except Exception as error:
            record["begin_ok"] = False
            self._fail(error, stage="begin")
            return False

    def capture_visual_features(self, payload: Mapping[str, Any]) -> bool:
        """Detach the five projected crops emitted before decoder layer 0."""

        try:
            if self._current is None:
                raise ActiveRuntimeError("visual callback arrived outside a policy call")
            if not isinstance(payload, Mapping):
                raise ActiveRuntimeError("visual callback payload must be a mapping")
            projected = _cached_visual_precision(payload.get("projected_features"))
            positions = payload.get("image_input_idx")
            if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != (
                1,
                5,
                144,
            ):
                raise ActiveRuntimeError("image_input_idx must have shape [1,5,144]")
            positions = positions.detach().to(device="cpu", dtype=torch.int64).contiguous()
            self._projected_features = projected
            self._image_input_idx = positions
            self._current["visual_capture_ok"] = True
            return True
        except Exception as error:
            self._fail(error, stage="visual_capture")
            return False

    @staticmethod
    def _cpu_phase_state(state: PhaseState) -> PhaseState:
        return PhaseState(
            stage_embedding=state.stage_embedding.detach().cpu().float(),
            progress=state.progress.detach().cpu().float(),
            boundary_prob=state.boundary_prob.detach().cpu().float(),
            uncertainty=state.uncertainty.detach().cpu().float(),
            next_hidden=state.next_hidden.detach().cpu().float(),
        )

    def prepare_policy_call(self) -> bool:
        """Run the frozen phase estimator and install the valid D8 context."""

        started_ns = time.perf_counter_ns()
        try:
            if self._current is None:
                raise ActiveRuntimeError("phase callback arrived outside a policy call")
            if self._current.get("prepared"):
                raise ActiveRuntimeError("policy call context was prepared twice")
            if self._history_window is None:
                raise ActiveRuntimeError("past-only history window is unavailable")
            if self._instruction_summary is None or self._normalized_proprio is None:
                raise ActiveRuntimeError("language or proprio context is unavailable")
            if self._projected_features is None or self._image_input_idx is None:
                raise ActiveRuntimeError("projected visual crops are unavailable")
            global_visual, crop_summary, crop_mask = pool_visual_features(
                self._projected_features[0].numpy(),
                self._image_input_idx[0].numpy(),
            )
            window = self._history_window
            with torch.inference_mode():
                state = self.phase_estimator(
                    visual_summary=torch.from_numpy(global_visual).unsqueeze(0),
                    instruction_summary=self._instruction_summary,
                    current_proprio=self._normalized_proprio,
                    proprio_history=window.proprio_history,
                    proprio_history_mask=window.history_mask,
                    action_history=window.action_history,
                    action_history_mask=window.history_mask,
                )
            state = self._cpu_phase_state(state)
            runtime_inputs = {
                "instruction_summary": self._instruction_summary,
                "vision_crop_summary": torch.from_numpy(crop_summary).unsqueeze(0),
                "vision_crop_mask": torch.from_numpy(crop_mask).unsqueeze(0),
                "phase_embedding": state.stage_embedding,
                "phase_scalars": torch.cat(
                    (state.progress, state.boundary_prob, state.uncertainty), dim=1
                ).float(),
                "normalized_proprio": window.normalized_proprio,
                "proprio_history": window.proprio_history,
                "action_history": window.action_history,
                "history_mask": window.history_mask,
            }
            validate_runtime_context(runtime_inputs, rows=1)
            self.adapter.begin_policy_call(runtime_inputs)
            if self.adapter.fail_reason is not None:
                raise ActiveRuntimeError(
                    f"adapter rejected runtime context: {self.adapter.fail_reason}"
                )
            latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            self._current.update(
                {
                    "prepared": True,
                    "fallback": False,
                    "prepare_latency_ms": latency_ms,
                    "history_valid_rows": int(window.history_mask.sum().item()),
                    "phase_progress": float(state.progress[0, 0]),
                    "phase_boundary_prob": float(state.boundary_prob[0, 0]),
                    "phase_uncertainty": float(state.uncertainty[0, 0]),
                    "runtime_shapes": {
                        name: list(value.shape) for name, value in runtime_inputs.items()
                    },
                }
            )
            self.prepared_calls += 1
            return True
        except Exception as error:
            if self._current is not None:
                self._current["prepare_latency_ms"] = (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0
            self._fail(error, stage="prepare")
            return False

    def record_route_event(self, event_name: str, payload: Mapping[str, Any]) -> None:
        """Attach adapter/controller telemetry without influencing control."""

        if self._current is None:
            return
        event = {"event": str(event_name), **_json_safe(dict(payload))}
        self._current["events"].append(event)
        if event_name == "phase_route_decision":
            selected = payload.get("selected_layer")
            if type(selected) is int:
                self._current["selected_layer"] = selected
                self._current["fallback"] = selected == 27

    def commit_selected_action(self, selected_action: Any) -> bool:
        """Commit only the selected normalized action after routing finishes."""

        if self._current is None:
            self.error_count += 1
            self.last_error = "commit arrived outside a policy call"
            return False
        current = self._current
        context = current.get("context", {})
        try:
            if self.adapter.active:
                raise ActiveRuntimeError("adapter did not finish at L11/L13/L27")
            if self._history_window is None:
                raise ActiveRuntimeError("history window was unavailable at commit")
            self.history.commit(
                str(context["episode_id"]),
                int(context["call_ordinal"]),
                selected_action,
            )
            current["committed"] = True
            current["action_shape"] = list(
                selected_action.shape
                if isinstance(selected_action, torch.Tensor)
                else np.asarray(selected_action).shape
            )
            current["end_ns"] = time.perf_counter_ns()
            self.committed_calls += 1
            return True
        except Exception as error:
            self._fail(error, stage="commit")
            current["end_ns"] = time.perf_counter_ns()
            return False
        finally:
            self._current = None
            self._history_window = None
            self._instruction_summary = None
            self._normalized_proprio = None
            self._projected_features = None
            self._image_input_idx = None


def load_frozen_phase_route_runtime(
    router_path: str | Path,
    phase_checkpoint_path: str | Path,
    *,
    expected_router_sha256: str = D9B_ROUTER_CHECKPOINT_SHA256,
    expected_phase_sha256: str = D2_PHASE_CHECKPOINT_SHA256,
) -> ActivePhaseRouteRuntime:
    """Strictly load both frozen CPU models and return the online runtime."""

    router_target = Path(router_path).resolve(strict=True)
    phase_target = Path(phase_checkpoint_path).resolve(strict=True)
    router_sha = sha256_file(router_target)
    phase_sha = sha256_file(phase_target)
    if router_sha != expected_router_sha256:
        raise ActiveRuntimeError("D8 router checkpoint SHA-256 differs")
    if phase_sha != expected_phase_sha256:
        raise ActiveRuntimeError("phase checkpoint SHA-256 differs")

    router_payload = torch.load(router_target, map_location="cpu", weights_only=True)
    if router_payload.get("schema_version") != D8B_PAYLOAD_SCHEMA_VERSION:
        raise ActiveRuntimeError("D8 router payload schema differs")
    router = final_router_from_mapping(router_payload)

    phase_payload = torch.load(phase_target, map_location="cpu", weights_only=True)
    if phase_payload.get("schema_version") != PHASE_CHECKPOINT_SCHEMA_VERSION:
        raise ActiveRuntimeError("phase checkpoint schema differs")
    phase_config = PhaseEstimatorConfig(**phase_payload["model_config"])
    if phase_config != PhaseEstimatorConfig():
        raise ActiveRuntimeError("phase checkpoint geometry differs")
    phase = PhaseStateEstimator(phase_config)
    phase.load_state_dict(phase_payload["model_state_dict"], strict=True)
    phase.eval()
    for parameter in phase.parameters():
        parameter.requires_grad_(False)
    phase_state_sha = frozen_phase_state_sha256(phase)
    artifacts = ActiveRuntimeArtifacts(
        router_path=str(router_target),
        router_sha256=router_sha,
        phase_checkpoint_path=str(phase_target),
        phase_checkpoint_sha256=phase_sha,
        phase_state_sha256=phase_state_sha,
    )
    return ActivePhaseRouteRuntime(router, phase, artifacts=artifacts)


__all__ = [
    "ActivePhaseRouteRuntime",
    "ActiveRuntimeArtifacts",
    "ActiveRuntimeError",
    "D9B_ACTIVE_RUNTIME_SCHEMA_VERSION",
    "D9B_ROUTER_CHECKPOINT_SHA256",
    "frozen_phase_state_sha256",
    "load_frozen_phase_route_runtime",
    "sha256_file",
]
