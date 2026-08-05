"""Causal PhaseEstimator runtime that prepares one M3 depth plan per policy call."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from .budget_controller import BudgetSelection, TransparentPhaseBudgetController
from .budget_profiles import (
    BudgetProfileResolver,
    ResolvedBudget,
    m3_depth_profiles,
)
from .exit_policy import PhaseAwareExitPolicy, PhaseAwareExitPolicyConfig
from .phase_estimator import PhaseEstimatorConfig, PhaseState, PhaseStateEstimator


@dataclass
class PhaseDepthPlan:
    phase_state: PhaseState
    routing_phase_state: PhaseState
    selection: BudgetSelection
    budget: ResolvedBudget
    progress_delta: float
    motion_speed: float
    boundary_rise: float
    boundary_crossed: bool
    latency_ms: float
    fallback: bool = False
    error: Optional[str] = None


class SafePhaseDepthRuntime:
    """Prepare depth plans without letting estimator failures escape to A1."""

    enabled = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device,
        eligible_exit_layers: Tuple[int, ...],
        history_len: int = 8,
        fm_steps_per_exit: int = 10,
        exit_policy_config: PhaseAwareExitPolicyConfig | None = None,
    ):
        if history_len < 1:
            raise ValueError("history_len must be positive")
        checkpoint = torch.load(
            Path(checkpoint_path),
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
        self.controller = TransparentPhaseBudgetController()
        self.resolver = BudgetProfileResolver(
            m3_depth_profiles(fm_steps_per_exit=fm_steps_per_exit),
            eligible_exit_layers,
        )
        self.exit_policy = PhaseAwareExitPolicy(exit_policy_config)
        self.history_len = history_len
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.current_plan: Optional[PhaseDepthPlan] = None
        self.records_prepared = 0
        self._histories: Dict[
            str, Deque[Tuple[np.ndarray, np.ndarray]]
        ] = defaultdict(lambda: deque(maxlen=self.history_len))
        self._previous_progress: Dict[str, float] = {}
        self._previous_boundary_prob: Dict[str, float] = {}

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(array).to(device=self.device, dtype=dtype)

    @staticmethod
    def _cpu_phase_state(state: PhaseState) -> PhaseState:
        return PhaseState(
            stage_embedding=state.stage_embedding.detach().to("cpu", dtype=torch.float32),
            progress=state.progress.detach().to("cpu", dtype=torch.float32),
            boundary_prob=state.boundary_prob.detach().to("cpu", dtype=torch.float32),
            uncertainty=state.uncertainty.detach().to("cpu", dtype=torch.float32),
            next_hidden=state.next_hidden.detach().to("cpu", dtype=torch.float32),
        )

    def _fallback_plan(self, error: Exception, latency_ms: float) -> PhaseDepthPlan:
        self.error_count += 1
        self.last_error = f"{type(error).__name__}: {error}"
        config = self.config
        phase_state = PhaseState(
            stage_embedding=torch.zeros(1, config.stage_dim),
            progress=torch.zeros(1, 1),
            boundary_prob=torch.ones(1, 1),
            uncertainty=torch.ones(1, 1),
            next_hidden=torch.zeros(1, 1, config.gru_hidden_dim),
        )
        selection = self.controller(
            phase_state,
            progress_delta=torch.zeros(1, 1),
            motion_speed=torch.full((1, 1), float("inf")),
        )
        plan = PhaseDepthPlan(
            phase_state=phase_state,
            routing_phase_state=phase_state,
            selection=selection,
            budget=self.resolver.resolve(3),
            progress_delta=0.0,
            motion_speed=float("inf"),
            boundary_rise=1.0,
            boundary_crossed=True,
            latency_ms=latency_ms,
            fallback=True,
            error=self.last_error,
        )
        self.current_plan = plan
        return plan

    def clear_current_plan(self) -> None:
        """Invalidate the per-policy-call plan exposed to width-only routing."""

        self.current_plan = None

    def _routing_state(
        self,
        episode_id: str,
        phase_state: PhaseState,
    ) -> tuple[PhaseState, float, bool]:
        """Convert a persistent boundary probability into a causal edge pulse."""

        boundary = float(phase_state.boundary_prob[0, 0])
        previous = self._previous_boundary_prob.get(episode_id, 0.0)
        threshold = self.controller.config.boundary_threshold
        crossed = previous < threshold <= boundary
        rise = boundary - previous
        self._previous_boundary_prob[episode_id] = boundary
        routing_state = PhaseState(
            stage_embedding=phase_state.stage_embedding,
            progress=phase_state.progress,
            boundary_prob=torch.full_like(
                phase_state.boundary_prob,
                1.0 if crossed else 0.0,
            ),
            uncertainty=phase_state.uncertainty,
            next_hidden=phase_state.next_hidden,
        )
        return routing_state, rise, crossed

    def prepare_plan(
        self,
        *,
        context: Mapping[str, Any],
        visual_summary: torch.Tensor,
        instruction_summary: torch.Tensor,
        normalized_proprio: Any,
        previous_action: Any = None,
    ) -> PhaseDepthPlan:
        """Estimate phase from current summaries and strictly previous history."""

        start_ns = time.perf_counter_ns()
        try:
            episode_id = str(context["episode_id"])
            if visual_summary.shape != (1, self.config.visual_summary_dim):
                raise ValueError("visual_summary must have shape [1, visual_summary_dim]")
            if instruction_summary.shape != (1, self.config.instruction_dim):
                raise ValueError(
                    "instruction_summary must have shape [1, instruction_dim]"
                )
            current_proprio = np.asarray(
                normalized_proprio,
                dtype=np.float32,
            ).reshape(-1)
            if current_proprio.shape != (self.config.proprio_dim,):
                raise ValueError("normalized_proprio has an invalid shape")
            history = self._histories[episode_id]
            history_count = len(history)
            proprio_history = np.zeros(
                (1, self.history_len, self.config.proprio_dim),
                dtype=np.float32,
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
            with torch.inference_mode():
                state = self.model(
                    visual_summary=visual_summary.to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    instruction_summary=instruction_summary.to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    current_proprio=self._to_tensor(
                        current_proprio[None], torch.float32
                    ),
                    proprio_history=self._to_tensor(
                        proprio_history, torch.float32
                    ),
                    proprio_history_mask=self._to_tensor(
                        history_mask, torch.bool
                    ),
                    action_history=self._to_tensor(action_history, torch.float32),
                    action_history_mask=self._to_tensor(history_mask, torch.bool),
                )
            self._synchronize()
            state = self._cpu_phase_state(state)
            routing_state, boundary_rise, boundary_crossed = self._routing_state(
                episode_id,
                state,
            )
            progress = float(state.progress[0, 0])
            previous_progress = self._previous_progress.get(episode_id, progress)
            progress_delta = progress - previous_progress
            self._previous_progress[episode_id] = progress
            previous = np.asarray(
                previous_action if previous_action is not None else [],
                dtype=np.float32,
            ).reshape(-1)
            motion_speed = (
                float(np.linalg.norm(previous[:3]))
                if previous.size >= 3
                else float("inf")
            )
            selection = self.controller(
                routing_state,
                progress_delta=torch.tensor([[progress_delta]], dtype=torch.float32),
                motion_speed=torch.tensor([[motion_speed]], dtype=torch.float32),
            )
            profile_id = int(selection.profile_id[0])
            latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
            plan = PhaseDepthPlan(
                phase_state=state,
                routing_phase_state=routing_state,
                selection=selection,
                budget=self.resolver.resolve(profile_id),
                progress_delta=progress_delta,
                motion_speed=motion_speed,
                boundary_rise=boundary_rise,
                boundary_crossed=boundary_crossed,
                latency_ms=latency_ms,
            )
            self.records_prepared += 1
            self.current_plan = plan
            return plan
        except Exception as error:
            latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
            return self._fallback_plan(error, latency_ms)

    def update_after_action(
        self,
        *,
        context: Mapping[str, Any],
        normalized_proprio: Any,
        normalized_action_chunk: Any,
    ) -> bool:
        """Append current transition only after A1 has finished its prediction."""

        try:
            episode_id = str(context["episode_id"])
            proprio = np.asarray(normalized_proprio, dtype=np.float32).reshape(-1)
            action = np.asarray(normalized_action_chunk, dtype=np.float32)
            if proprio.shape != (self.config.proprio_dim,):
                raise ValueError("normalized_proprio has an invalid shape")
            if action.shape != (
                self.config.action_horizon,
                self.config.action_dim,
            ):
                raise ValueError("normalized_action_chunk has an invalid shape")
            if not np.isfinite(proprio).all() or not np.isfinite(action).all():
                raise ValueError("phase history contains a non-finite value")
            self._histories[episode_id].append((proprio.copy(), action.copy()))
            return True
        except Exception as error:
            self.error_count += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return False
