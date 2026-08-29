"""Fail-closed authorization for the preregistered Stage-9 state-13 pilot.

The original protocol JSON remains byte-frozen.  This module adds the second
half of the access decision: state 13 can be selected only after the sealed
state-12 pair has passed its independent identity, runtime, success, and
latency-evidence checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .route_first_active_protocol import (
    ROUTE_FIRST_ACTIVE_BASE_SEED,
    ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256,
    RouteFirstActiveProtocolError,
    load_route_first_active_protocol,
)


STAGE9_CANDIDATE_METHOD = "candidate_first_v3"
STAGE9_ROUTE_FIRST_METHOD = "route_first_stage8"
STAGE9_PILOT_METHODS = (STAGE9_CANDIDATE_METHOD, STAGE9_ROUTE_FIRST_METHOD)
STAGE9_PILOT_EPISODE_INDEX = 13
STAGE9_STATE12_GATE_RELATIVE_PATH = Path(
    "results/route_first/route_first_stage9_state12_pair.json"
)
STAGE9_STATE12_GATE_SHA256 = (
    "b636e1b1b650afbf50fb7bdda7c3ab18da366c4a5b33801b3af36a57e4055bbe"
)
STAGE9_STATE12_GATE_SCHEMA = "phase-route-vla.route-first-stage9-paired-smoke.v1"


class Stage9PilotProtocolError(RouteFirstActiveProtocolError):
    """Raised before state 13 can be read or a pilot environment is created."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage9PilotProtocolError(f"expected JSON object: {path}")
    return value


def load_stage9_state12_unlock(repo_root: str | Path) -> Mapping[str, Any]:
    """Bind and validate the exact state-12 result that unlocks the pilot."""

    root = Path(repo_root).resolve(strict=True)
    gate_path = (root / STAGE9_STATE12_GATE_RELATIVE_PATH).resolve(strict=True)
    try:
        gate_path.relative_to(root)
    except ValueError as error:
        raise Stage9PilotProtocolError("state-12 gate escapes repository") from error
    if sha256_file(gate_path) != STAGE9_STATE12_GATE_SHA256:
        raise Stage9PilotProtocolError("state-12 gate SHA-256 differs")
    gate = _load_object(gate_path)
    checks = gate.get("checks")
    next_gate = gate.get("next_gate")
    candidate = gate.get("candidate_first")
    route = gate.get("route_first")
    if (
        gate.get("schema_version") != STAGE9_STATE12_GATE_SCHEMA
        or gate.get("status") != "PASS"
        or gate.get("protocol_sha256") != ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(next_gate, Mapping)
        or next_gate.get("state13_pilot_protocol_gate_unlocked") is not True
        or next_gate.get("state13_opened") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("successes") != 1
        or not isinstance(route, Mapping)
        or route.get("successes") != 1
    ):
        raise Stage9PilotProtocolError("state-12 gate does not authorize state 13")
    return gate


def expected_stage9_pilot_arm_order(task_id: int) -> tuple[str, str]:
    if type(task_id) is not int or task_id not in range(10):
        raise Stage9PilotProtocolError("pilot task_id must be in 0..9")
    if task_id % 2 == 0:
        return STAGE9_CANDIDATE_METHOD, STAGE9_ROUTE_FIRST_METHOD
    return STAGE9_ROUTE_FIRST_METHOD, STAGE9_CANDIDATE_METHOD


@dataclass(frozen=True)
class Stage9PilotArmSelection:
    method: str
    task_id: int
    episode_index: int
    arm_position: int
    base_seed: int
    episode_seed: int


def validate_stage9_pilot_arm_selection(
    protocol: Mapping[str, Any],
    *,
    method: str,
    task_id: int,
    episode_index: int,
    arm_position: int,
    seed: int,
) -> Stage9PilotArmSelection:
    """Authorize one candidate-first or route-first state-13 task arm."""

    if method not in STAGE9_PILOT_METHODS:
        raise Stage9PilotProtocolError("unknown pilot method")
    if seed != ROUTE_FIRST_ACTIVE_BASE_SEED:
        raise Stage9PilotProtocolError("pilot base seed differs")
    if episode_index != STAGE9_PILOT_EPISODE_INDEX:
        raise Stage9PilotProtocolError("pilot state differs")
    expected_order = expected_stage9_pilot_arm_order(task_id)
    if type(arm_position) is not int or arm_position not in (1, 2):
        raise Stage9PilotProtocolError("pilot arm_position must be 1 or 2")
    if expected_order[arm_position - 1] != method:
        raise Stage9PilotProtocolError("pilot arm order differs")

    schedule = protocol.get("schedule")
    pilot = schedule.get("paired_pilot") if isinstance(schedule, Mapping) else None
    if not isinstance(pilot, Mapping):
        raise Stage9PilotProtocolError("paired-pilot schedule is missing")
    order_key = "even_task_ids" if task_id % 2 == 0 else "odd_task_ids"
    arm_order = pilot.get("arm_order")
    if (
        task_id not in tuple(pilot.get("task_ids", ()))
        or tuple(pilot.get("episode_indices", ())) != (episode_index,)
        or not isinstance(arm_order, Mapping)
        or tuple(arm_order.get(order_key, ())) != expected_order
        or method not in tuple(pilot.get("methods", ()))
    ):
        raise Stage9PilotProtocolError("pilot selection differs from protocol")
    return Stage9PilotArmSelection(
        method=method,
        task_id=task_id,
        episode_index=episode_index,
        arm_position=arm_position,
        base_seed=seed,
        episode_seed=seed + task_id * 10_000 + episode_index,
    )


def authorize_stage9_pilot_arm(
    *,
    repo_root: str | Path,
    protocol_path: str | Path,
    method: str,
    task_id: int,
    episode_index: int,
    arm_position: int,
    seed: int,
) -> tuple[Stage9PilotArmSelection, Mapping[str, Any], Mapping[str, Any]]:
    """Bind protocol plus state-12 unlock before authorizing one pilot arm."""

    protocol = load_route_first_active_protocol(protocol_path, repo_root)
    gate = load_stage9_state12_unlock(repo_root)
    selection = validate_stage9_pilot_arm_selection(
        protocol,
        method=method,
        task_id=task_id,
        episode_index=episode_index,
        arm_position=arm_position,
        seed=seed,
    )
    return selection, protocol, gate


__all__ = [
    "STAGE9_CANDIDATE_METHOD",
    "STAGE9_PILOT_EPISODE_INDEX",
    "STAGE9_PILOT_METHODS",
    "STAGE9_ROUTE_FIRST_METHOD",
    "STAGE9_STATE12_GATE_RELATIVE_PATH",
    "STAGE9_STATE12_GATE_SCHEMA",
    "STAGE9_STATE12_GATE_SHA256",
    "Stage9PilotArmSelection",
    "Stage9PilotProtocolError",
    "authorize_stage9_pilot_arm",
    "expected_stage9_pilot_arm_order",
    "load_stage9_state12_unlock",
    "sha256_file",
    "validate_stage9_pilot_arm_selection",
]
