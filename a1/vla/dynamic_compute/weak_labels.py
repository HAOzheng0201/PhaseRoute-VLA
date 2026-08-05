"""Weak phase-boundary labels on the policy-call timebase."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


PHASE_LABEL_SCHEMA_VERSION = "phase-route-vla.phase-label.v1"


@dataclass(frozen=True)
class BoundaryLabelConfig:
    """Configurable events used to construct weak boundary targets."""

    gripper_open_threshold: float = 0.04
    translation_speed_change_threshold: float = 0.50
    rotation_speed_change_threshold: float = 0.07
    direction_cosine_threshold: float = 0.25
    fine_transition_high_speed: float = 0.60
    fine_transition_low_speed: float = 0.30
    action_delta_increase_threshold: float = 0.01

    weight_gripper_flip: float = 1.0
    weight_speed_change: float = 1.0
    weight_direction_change: float = 1.0
    weight_rotation_change: float = 1.0
    weight_fine_transition: float = 1.0
    weight_action_delta_increase: float = 1.0
    boundary_score_threshold: float = 1.0
    dilation_radius: int = 2

    def __post_init__(self) -> None:
        nonnegative = {
            "gripper_open_threshold": self.gripper_open_threshold,
            "translation_speed_change_threshold": self.translation_speed_change_threshold,
            "rotation_speed_change_threshold": self.rotation_speed_change_threshold,
            "fine_transition_high_speed": self.fine_transition_high_speed,
            "fine_transition_low_speed": self.fine_transition_low_speed,
            "action_delta_increase_threshold": self.action_delta_increase_threshold,
            "boundary_score_threshold": self.boundary_score_threshold,
            "weight_gripper_flip": self.weight_gripper_flip,
            "weight_speed_change": self.weight_speed_change,
            "weight_direction_change": self.weight_direction_change,
            "weight_rotation_change": self.weight_rotation_change,
            "weight_fine_transition": self.weight_fine_transition,
            "weight_action_delta_increase": self.weight_action_delta_increase,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not -1.0 <= self.direction_cosine_threshold <= 1.0:
            raise ValueError("direction_cosine_threshold must be in [-1, 1]")
        if self.dilation_radius < 0:
            raise ValueError("dilation_radius must be nonnegative")
        if self.fine_transition_low_speed > self.fine_transition_high_speed:
            raise ValueError("fine_transition_low_speed cannot exceed high_speed")


@dataclass
class PhaseWeakLabel:
    """One weakly labelled policy call."""

    episode_id: str
    task_id: Any
    call_index: int
    num_policy_calls: int
    environment_step_id: int
    progress_target: float
    boundary_score: float
    boundary_target_raw: int
    boundary_target: int
    boundary_events: Dict[str, bool] = field(default_factory=dict)
    schema_version: str = field(default=PHASE_LABEL_SCHEMA_VERSION, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def _selected_action_delta(record: Mapping[str, Any]) -> Optional[float]:
    exit_layer = record.get("exit_layer")
    layers = record.get("candidate_exit_layers") or []
    deltas = record.get("action_delta_by_exit") or []
    for layer, delta in zip(layers, deltas):
        if layer == exit_layer:
            return _finite_float(delta)
    return None


def _previous_action_vector(record: Mapping[str, Any]) -> Optional[np.ndarray]:
    extra = record.get("extra") or {}
    value = extra.get("previous_action_vector")
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size < 3 or not np.isfinite(vector[:3]).all():
        return None
    return vector


def _direction_change(
    previous_record: Mapping[str, Any],
    current_record: Mapping[str, Any],
    cosine_threshold: float,
) -> bool:
    previous = _previous_action_vector(previous_record)
    current = _previous_action_vector(current_record)
    if previous is None or current is None:
        return False
    previous_translation = previous[:3]
    current_translation = current[:3]
    denominator = np.linalg.norm(previous_translation) * np.linalg.norm(current_translation)
    if denominator <= 1e-12:
        return False
    cosine = float(np.dot(previous_translation, current_translation) / denominator)
    return cosine <= cosine_threshold


def build_episode_weak_labels(
    records: Sequence[Mapping[str, Any]],
    config: Optional[BoundaryLabelConfig] = None,
) -> List[PhaseWeakLabel]:
    """Build progress and dilated boundary targets for one episode.

    PhaseRoute's M2 timebase is one policy call / one 8-action chunk. Progress
    is therefore call_index / (num_policy_calls - 1), rather than the raw
    environment step divided by an unknown terminal step.
    """

    if not records:
        return []
    config = config or BoundaryLabelConfig()
    ordered = sorted(records, key=lambda record: int(record["step_id"]))
    episode_ids = {str(record["episode_id"]) for record in ordered}
    if len(episode_ids) != 1:
        raise ValueError("build_episode_weak_labels expects exactly one episode")
    episode_id = next(iter(episode_ids))
    num_calls = len(ordered)

    labels: List[PhaseWeakLabel] = []
    for index, record in enumerate(ordered):
        events = {
            "gripper_flip": False,
            "translation_speed_change": False,
            "rotation_speed_change": False,
            "direction_change": False,
            "fine_transition": False,
            "action_delta_increase": False,
        }
        if index > 0:
            previous = ordered[index - 1]
            previous_gripper = _finite_float(previous.get("gripper_state"))
            current_gripper = _finite_float(record.get("gripper_state"))
            if previous_gripper is not None and current_gripper is not None:
                events["gripper_flip"] = (
                    previous_gripper >= config.gripper_open_threshold
                ) != (current_gripper >= config.gripper_open_threshold)

            previous_speed = _finite_float(previous.get("translation_speed"))
            current_speed = _finite_float(record.get("translation_speed"))
            if previous_speed is not None and current_speed is not None:
                events["translation_speed_change"] = (
                    abs(current_speed - previous_speed)
                    >= config.translation_speed_change_threshold
                )
                events["fine_transition"] = (
                    previous_speed >= config.fine_transition_high_speed
                    and current_speed <= config.fine_transition_low_speed
                )

            previous_rotation = _finite_float(previous.get("rotation_speed"))
            current_rotation = _finite_float(record.get("rotation_speed"))
            if previous_rotation is not None and current_rotation is not None:
                events["rotation_speed_change"] = (
                    abs(current_rotation - previous_rotation)
                    >= config.rotation_speed_change_threshold
                )

            events["direction_change"] = _direction_change(
                previous,
                record,
                config.direction_cosine_threshold,
            )
            previous_delta = _selected_action_delta(previous)
            current_delta = _selected_action_delta(record)
            if previous_delta is not None and current_delta is not None:
                events["action_delta_increase"] = (
                    current_delta - previous_delta
                    >= config.action_delta_increase_threshold
                )

        score = (
            config.weight_gripper_flip * events["gripper_flip"]
            + config.weight_speed_change * events["translation_speed_change"]
            + config.weight_direction_change * events["direction_change"]
            + config.weight_rotation_change * events["rotation_speed_change"]
            + config.weight_fine_transition * events["fine_transition"]
            + config.weight_action_delta_increase * events["action_delta_increase"]
        )
        raw_target = int(score >= config.boundary_score_threshold)
        labels.append(
            PhaseWeakLabel(
                episode_id=episode_id,
                task_id=record.get("task_id"),
                call_index=index,
                num_policy_calls=num_calls,
                environment_step_id=int(record["step_id"]),
                progress_target=index / max(num_calls - 1, 1),
                boundary_score=float(score),
                boundary_target_raw=raw_target,
                boundary_target=raw_target,
                boundary_events=events,
            )
        )

    raw_boundary_indices = [
        index for index, label in enumerate(labels) if label.boundary_target_raw
    ]
    for boundary_index in raw_boundary_indices:
        start = max(0, boundary_index - config.dilation_radius)
        stop = min(num_calls, boundary_index + config.dilation_radius + 1)
        for index in range(start, stop):
            labels[index].boundary_target = 1
    return labels


def build_weak_labels(
    records: Sequence[Mapping[str, Any]],
    config: Optional[BoundaryLabelConfig] = None,
) -> List[PhaseWeakLabel]:
    """Group policy-call records by episode and build labels without leakage."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        episode_id = str(record["episode_id"])
        grouped.setdefault(episode_id, []).append(record)
    labels: List[PhaseWeakLabel] = []
    for episode_id in sorted(grouped):
        labels.extend(build_episode_weak_labels(grouped[episode_id], config=config))
    return labels
