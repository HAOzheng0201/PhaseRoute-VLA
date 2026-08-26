"""Frozen Stage-9 active-pilot access and artifact validation.

This module contains no simulator or model imports.  Launchers, validators,
and CPU-only tests can therefore enforce the preregistered state-access
boundary before LIBERO is imported or an episode is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROUTE_FIRST_ACTIVE_PROTOCOL_SCHEMA_VERSION = (
    "phase-route-vla.route-first-active-pilot-protocol.v1"
)
ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256 = (
    "fcb1c2a1fdf7ea3f79343f72d25240449500a5eac3fad1372f0808023888db4d"
)
ROUTE_FIRST_ACTIVE_METHOD = "route_first_stage8"
ROUTE_FIRST_ACTIVE_BASE_SEED = 20260826
ROUTE_FIRST_ACTIVE_STAGES = ("engineering_smoke", "paired_pilot")


class RouteFirstActiveProtocolError(ValueError):
    """Raised before any preregistered simulator state can be opened."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RouteFirstActiveProtocolError("active protocol must be a JSON object")
    return value


def _resolve_below(root: Path, relative: Any, name: str) -> Path:
    if type(relative) is not str or not relative:
        raise RouteFirstActiveProtocolError(f"{name} path is missing")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise RouteFirstActiveProtocolError(f"{name} escapes the repository") from error
    if not target.is_file():
        raise RouteFirstActiveProtocolError(f"{name} is missing: {target}")
    return target


def load_route_first_active_protocol(
    protocol_path: str | Path,
    repo_root: str | Path,
    *,
    verify_frozen_artifacts: bool = True,
) -> Mapping[str, Any]:
    """Load the exact preregistration and optionally bind all frozen files."""

    protocol_target = Path(protocol_path).resolve(strict=True)
    root = Path(repo_root).resolve(strict=True)
    if sha256_file(protocol_target) != ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256:
        raise RouteFirstActiveProtocolError("active protocol SHA-256 differs")
    protocol = _load_object(protocol_target)
    if protocol.get("schema_version") != ROUTE_FIRST_ACTIVE_PROTOCOL_SCHEMA_VERSION:
        raise RouteFirstActiveProtocolError("active protocol schema differs")
    if protocol.get("status") != "PREREGISTERED_NOT_OPENED":
        raise RouteFirstActiveProtocolError("active protocol status differs")
    shared = protocol.get("shared_settings")
    if not isinstance(shared, Mapping):
        raise RouteFirstActiveProtocolError("shared settings are missing")
    if (
        shared.get("base_seed") != ROUTE_FIRST_ACTIVE_BASE_SEED
        or shared.get("threshold_movement") is not False
        or shared.get("router_refit") is not False
        or shared.get("single_physical_gpu_per_process") is not True
    ):
        raise RouteFirstActiveProtocolError("frozen active settings differ")
    access = protocol.get("access_ledger")
    if not isinstance(access, Mapping) or any(
        access.get(name) is not False
        for name in (
            "state12_smoke_opened",
            "state13_pilot_opened",
            "historical_D9_states40_to49_opened_for_this_stage",
            "active_control_executed_for_this_stage",
        )
    ):
        raise RouteFirstActiveProtocolError("preregistered access ledger differs")

    if verify_frozen_artifacts:
        frozen = protocol.get("frozen_implementation")
        if not isinstance(frozen, Mapping):
            raise RouteFirstActiveProtocolError("frozen implementation is missing")
        bindings = (
            ("stage8_verification", "stage8_verification_path", "stage8_verification_sha256"),
            ("runtime", "runtime_path", "runtime_sha256"),
            ("controller", "controller_path", "controller_sha256"),
            ("calibrated_router", "calibrated_router_path", "calibrated_router_sha256"),
            ("stage7_holdout", "stage7_holdout_path", "stage7_holdout_sha256"),
            ("v3_context_router", "v3_context_router_path", "v3_context_router_sha256"),
            ("phase_checkpoint", "phase_checkpoint_path", "phase_checkpoint_sha256"),
        )
        for name, path_key, sha_key in bindings:
            target = _resolve_below(root, frozen.get(path_key), name)
            if sha256_file(target) != frozen.get(sha_key):
                raise RouteFirstActiveProtocolError(f"{name} SHA-256 differs")
    return protocol


def _parse_indices(value: str, *, name: str) -> tuple[int, ...]:
    if type(value) is not str or not value.strip():
        raise RouteFirstActiveProtocolError(f"{name} must be nonempty")
    result: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token.isdigit():
            raise RouteFirstActiveProtocolError(f"invalid {name} item: {token}")
        result.append(int(token))
    if len(result) != len(set(result)):
        raise RouteFirstActiveProtocolError(f"{name} contains duplicates")
    return tuple(result)


@dataclass(frozen=True)
class RouteFirstActiveSelection:
    experiment_stage: str
    task_ids: tuple[int, ...]
    episode_indices: tuple[int, ...]
    arm_position: int


def validate_route_first_active_selection(
    protocol: Mapping[str, Any],
    *,
    experiment_stage: str,
    task_spec: str,
    episode_spec: str,
    arm_position: int,
    seed: int,
) -> RouteFirstActiveSelection:
    """Authorize exactly one route-first arm from the frozen Stage-9 schedule."""

    if experiment_stage not in ROUTE_FIRST_ACTIVE_STAGES:
        raise RouteFirstActiveProtocolError("unknown active experiment stage")
    if type(arm_position) is not int or arm_position not in (1, 2):
        raise RouteFirstActiveProtocolError("arm_position must be 1 or 2")
    if seed != ROUTE_FIRST_ACTIVE_BASE_SEED:
        raise RouteFirstActiveProtocolError("active base seed differs")
    tasks = _parse_indices(task_spec, name="task_ids")
    episodes = _parse_indices(episode_spec, name="episode_indices")
    schedule = protocol.get("schedule")
    if not isinstance(schedule, Mapping):
        raise RouteFirstActiveProtocolError("active schedule is missing")

    if experiment_stage == "engineering_smoke":
        smoke = schedule.get("engineering_smoke")
        if not isinstance(smoke, Mapping):
            raise RouteFirstActiveProtocolError("engineering smoke schedule is missing")
        if (
            tasks != tuple(smoke.get("task_ids", ()))
            or episodes != tuple(smoke.get("episode_indices", ()))
            or tuple(smoke.get("arm_order", ())).index(ROUTE_FIRST_ACTIVE_METHOD) + 1
            != arm_position
        ):
            raise RouteFirstActiveProtocolError(
                "selection is not the preregistered route-first smoke arm"
            )
    else:
        pilot = schedule.get("paired_pilot")
        if not isinstance(pilot, Mapping):
            raise RouteFirstActiveProtocolError("paired pilot schedule is missing")
        # One task per process keeps the alternating paired-arm order explicit.
        if len(tasks) != 1 or tasks[0] not in tuple(pilot.get("task_ids", ())):
            raise RouteFirstActiveProtocolError(
                "paired-pilot runner requires exactly one preregistered task"
            )
        if episodes != tuple(pilot.get("episode_indices", ())):
            raise RouteFirstActiveProtocolError("paired-pilot state differs")
        order_key = "even_task_ids" if tasks[0] % 2 == 0 else "odd_task_ids"
        arm_order = pilot.get("arm_order")
        if not isinstance(arm_order, Mapping):
            raise RouteFirstActiveProtocolError("paired arm order is missing")
        expected = tuple(arm_order.get(order_key, ())).index(ROUTE_FIRST_ACTIVE_METHOD) + 1
        if arm_position != expected:
            raise RouteFirstActiveProtocolError("paired route-first arm order differs")

    return RouteFirstActiveSelection(
        experiment_stage=experiment_stage,
        task_ids=tasks,
        episode_indices=episodes,
        arm_position=arm_position,
    )


__all__ = [
    "ROUTE_FIRST_ACTIVE_BASE_SEED",
    "ROUTE_FIRST_ACTIVE_METHOD",
    "ROUTE_FIRST_ACTIVE_PROTOCOL_SCHEMA_VERSION",
    "ROUTE_FIRST_ACTIVE_PROTOCOL_SHA256",
    "ROUTE_FIRST_ACTIVE_STAGES",
    "RouteFirstActiveProtocolError",
    "RouteFirstActiveSelection",
    "load_route_first_active_protocol",
    "sha256_file",
    "validate_route_first_active_selection",
]
