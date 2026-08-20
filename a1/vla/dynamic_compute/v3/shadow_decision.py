"""Fail-closed V3-D4 shadow-only routing decisions.

The module deliberately does not load models, actions, rollout state, or test
data.  It combines already-attested scalar/boolean signals and returns only a
counterfactual layer choice.  Missing signals veto an early exit; no signal
can compensate for another failed gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


D4_SCHEMA_VERSION = "phase-route-vla.v3.d4-shadow-contract.v1"
D4_STATUS = "D4_SHADOW_CONTRACT_FROZEN"
D4_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/gripper_v2/d4_shadow_contract.json"
)
D4_CONTRACT_SHA256 = (
    "286a359af14ac89c5952c1ba924a64aa2aeeeed89fb6762f71343b7bb0118d10"
)
D4_CANDIDATE_LAYERS = (11, 13)
D4_FALLBACK_LAYER = 27
D4_PRIORITY = (11, 13, 27)
D4_GRIPPER_THRESHOLD = 0.043773197319646726
D4_ACTION_CONSISTENCY_THRESHOLD = 0.00390625
D4_RP_PEP_FM_CALLS = {11: 4, 13: 5, 27: 7}


class D4ShadowError(ValueError):
    """Raised when a D4 contract or signal has invalid geometry/type."""


@dataclass(frozen=True)
class ShadowCandidateSignals:
    """Causal signals for one candidate layer; ``None`` means fail closed."""

    layer: int
    original_action_consistency: bool | None
    motion_safe: bool | None
    tail_ucb_safe: bool | None
    gripper_score: float | None

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer not in D4_CANDIDATE_LAYERS:
            raise D4ShadowError("D4 candidate layer must be exactly 11 or 13")
        for name in (
            "original_action_consistency",
            "motion_safe",
            "tail_ucb_safe",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise D4ShadowError(f"D4 {name} must be bool or None")
        if self.gripper_score is not None and (
            isinstance(self.gripper_score, bool)
            or not isinstance(self.gripper_score, (int, float))
        ):
            raise D4ShadowError("D4 gripper score must be numeric or None")

    @property
    def gates(self) -> dict[str, bool]:
        score = self.gripper_score
        return {
            "original_action_consistency": (
                self.original_action_consistency is True
            ),
            "motion_safe": self.motion_safe is True,
            "tail_ucb_safe": self.tail_ucb_safe is True,
            "gripper_safe": (
                score is not None
                and math.isfinite(float(score))
                and float(score) <= D4_GRIPPER_THRESHOLD
            ),
        }

    @property
    def veto_reasons(self) -> tuple[str, ...]:
        gates = self.gates
        reasons: list[str] = []
        for name in (
            "original_action_consistency",
            "motion_safe",
            "tail_ucb_safe",
        ):
            raw = getattr(self, name)
            if raw is None:
                reasons.append(f"missing_{name}")
            elif not gates[name]:
                reasons.append(f"failed_{name}")
        if self.gripper_score is None:
            reasons.append("missing_gripper_score")
        elif not math.isfinite(float(self.gripper_score)):
            reasons.append("nonfinite_gripper_score")
        elif not gates["gripper_safe"]:
            reasons.append("failed_gripper_safe")
        return tuple(reasons)

    @property
    def route_safe(self) -> bool:
        return all(self.gates.values())


@dataclass(frozen=True)
class ShadowDecision:
    """A counterfactual layer selection with no action/control capability."""

    selected_layer: int
    candidates: tuple[ShadowCandidateSignals, ShadowCandidateSignals]

    def __post_init__(self) -> None:
        if tuple(candidate.layer for candidate in self.candidates) != (
            D4_CANDIDATE_LAYERS
        ):
            raise D4ShadowError("D4 candidates must be ordered L11 then L13")
        expected = D4_FALLBACK_LAYER
        for candidate in self.candidates:
            if candidate.route_safe:
                expected = candidate.layer
                break
        if self.selected_layer != expected:
            raise D4ShadowError("D4 selection violates frozen priority")

    @property
    def would_early_exit(self) -> bool:
        return self.selected_layer in D4_CANDIDATE_LAYERS

    @property
    def disposition(self) -> str:
        return (
            f"SHADOW_L{self.selected_layer}"
            if self.would_early_exit
            else "DEFER_L27"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "selected_layer": self.selected_layer,
            "disposition": self.disposition,
            "would_early_exit": self.would_early_exit,
            "estimated_rp_pep_fm_calls": D4_RP_PEP_FM_CALLS[
                self.selected_layer
            ],
            "active_control": False,
            "returns_action": False,
            "candidates": {
                str(candidate.layer): {
                    "gates": candidate.gates,
                    "route_safe": candidate.route_safe,
                    "veto_reasons": list(candidate.veto_reasons),
                }
                for candidate in self.candidates
            },
        }


def decide_shadow(
    layer11: ShadowCandidateSignals,
    layer13: ShadowCandidateSignals,
) -> ShadowDecision:
    """Apply frozen L11 -> L13 -> L27 priority without returning an action."""

    candidates = (layer11, layer13)
    if tuple(candidate.layer for candidate in candidates) != D4_CANDIDATE_LAYERS:
        raise D4ShadowError("D4 shadow inputs must be ordered L11 then L13")
    selected = D4_FALLBACK_LAYER
    for candidate in candidates:
        if candidate.route_safe:
            selected = candidate.layer
            break
    return ShadowDecision(selected_layer=selected, candidates=candidates)


def summarize_shadow_decisions(
    decisions: Sequence[ShadowDecision],
) -> dict[str, Any]:
    """Return deterministic counts and estimated FM calls for an audit batch."""

    if not isinstance(decisions, Sequence) or not decisions:
        raise D4ShadowError("D4 shadow summary requires at least one decision")
    if any(not isinstance(value, ShadowDecision) for value in decisions):
        raise D4ShadowError("D4 shadow summary contains an invalid decision")
    counts = {layer: 0 for layer in D4_PRIORITY}
    veto_counts: dict[str, int] = {}
    fm_calls = 0
    for decision in decisions:
        counts[decision.selected_layer] += 1
        fm_calls += D4_RP_PEP_FM_CALLS[decision.selected_layer]
        for candidate in decision.candidates:
            for reason in candidate.veto_reasons:
                key = f"L{candidate.layer}:{reason}"
                veto_counts[key] = veto_counts.get(key, 0) + 1
    total = len(decisions)
    early = counts[11] + counts[13]
    return {
        "decision_calls": total,
        "selection_counts": {str(layer): counts[layer] for layer in D4_PRIORITY},
        "selection_fractions": {
            str(layer): counts[layer] / total for layer in D4_PRIORITY
        },
        "early_exit_calls": early,
        "early_exit_fraction": early / total,
        "estimated_rp_pep_fm_calls": fm_calls,
        "estimated_rp_pep_fm_calls_per_policy_call": fm_calls / total,
        "veto_counts": dict(sorted(veto_counts.items())),
        "active_control": False,
        "measured_latency": False,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D4ShadowError("D4 contract cannot be read") from error
    if not isinstance(value, Mapping):
        raise D4ShadowError("D4 contract must be a JSON object")
    return value


def load_d4_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D4_CONTRACT_RELATIVE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != D4_CONTRACT_SHA256:
        raise D4ShadowError("D4 frozen contract SHA-256 differs")
    contract = dict(_read_json(path))
    if (
        contract.get("schema_version") != D4_SCHEMA_VERSION
        or contract.get("status") != D4_STATUS
        or contract.get("decision", {}).get("formula")
        != (
            "route_safe=original_action_consistency AND motion_safe AND "
            "tail_ucb_safe AND gripper_safe"
        )
        or tuple(contract.get("decision", {}).get("priority", ())) != D4_PRIORITY
        or contract.get("gripper_gate", {}).get("threshold")
        != D4_GRIPPER_THRESHOLD
        or contract.get("original_action_consistency", {}).get("threshold")
        != D4_ACTION_CONSISTENCY_THRESHOLD
        or contract.get("scope", {}).get("active_control_allowed") is not False
        or contract.get("scope", {}).get("independent_test_allowed") is not False
        or contract.get("on_contract_pass", {}).get(
            "formal_shadow_execution_authorized"
        )
        is not False
    ):
        raise D4ShadowError("D4 frozen contract semantics differ")
    return contract


__all__ = [
    "D4_ACTION_CONSISTENCY_THRESHOLD",
    "D4_CANDIDATE_LAYERS",
    "D4_CONTRACT_RELATIVE_PATH",
    "D4_CONTRACT_SHA256",
    "D4_FALLBACK_LAYER",
    "D4_GRIPPER_THRESHOLD",
    "D4_PRIORITY",
    "D4_RP_PEP_FM_CALLS",
    "D4_SCHEMA_VERSION",
    "D4_STATUS",
    "D4ShadowError",
    "ShadowCandidateSignals",
    "ShadowDecision",
    "decide_shadow",
    "load_d4_contract",
    "summarize_shadow_decisions",
]
