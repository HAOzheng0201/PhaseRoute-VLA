"""Fail-closed generated-state artifacts for Route-first Stage 11D.

The state pipeline is intentionally separated from policy collection.  It
generates every scheduled reset state twice in isolated CPU processes and
publishes a payload only when the two canonical FP64 byte streams match.
No model, policy action, official LIBERO state, or CUDA device is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .route_first_reliability import (
    STAGE11D_CLUSTERS_PER_TASK,
    STAGE11D_CLUSTER_COUNT,
    STAGE11D_PROTOCOL_SHA256,
    STAGE11D_TASK_IDS,
    Stage11DRecord,
    Stage11DReliabilityError,
    build_stage11d_schedule,
)


STAGE11D_STATE_RECORD_SCHEMA = (
    "phase-route-vla.route-first-stage11d-state-record.v1"
)
STAGE11D_STATE_RUN_SCHEMA = "phase-route-vla.route-first-stage11d-state-run.v1"
STAGE11D_STATE_PAYLOAD_SCHEMA = (
    "phase-route-vla.route-first-stage11d-state-payload.v1"
)
STAGE11D_STATE_ATTESTATION_SCHEMA = (
    "phase-route-vla.route-first-stage11d-state-attestation.v1"
)
STAGE11D_STATE_PASSES = (1, 2)
STAGE11D_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
STAGE11D_STATE_RECORDS_RELATIVE_PATH = Path(
    "runs/route_first_stage11d_state_records"
)
STAGE11D_STATES_RELATIVE_PATH = Path("runs/route_first_stage11d_states")
STAGE11D_RUNNER_READINESS_RELATIVE_PATH = Path(
    "results/route_first/route_first_stage11d_state_runner_readiness.json"
)


class Stage11DArtifactError(Stage11DReliabilityError):
    """Raised when generated-state evidence violates Stage-11D contracts."""


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state_bytes(state: Any) -> tuple[np.ndarray, bytes, str]:
    """Return canonical little-endian FP64 bytes for one finite MuJoCo state."""

    value = np.asarray(state)
    if (
        value.ndim != 1
        or value.size == 0
        or value.dtype.kind != "f"
        or value.dtype.itemsize != 8
        or not bool(np.isfinite(value).all())
    ):
        raise Stage11DArtifactError(
            "Stage-11D state must be finite nonempty float64 [D]"
        )
    canonical = np.ascontiguousarray(value.astype("<f8", copy=False))
    raw = canonical.tobytes(order="C")
    if len(raw) != canonical.size * 8:
        raise Stage11DArtifactError("Stage-11D canonical state bytes differ")
    return canonical, raw, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Stage11DStateEvidence:
    pass_id: int
    task_id: int
    replicate_id: int
    split: str
    cluster_key: str
    state_seed: int
    policy_seed: int
    state_dimension: int
    state_nbytes: int
    state_sha256: str
    initial_task_success: bool
    explicit_reset_attempts: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Stage11DStateEvidence":
        if value.get("schema_version") != STAGE11D_STATE_RECORD_SCHEMA:
            raise Stage11DArtifactError("Stage-11D state record schema differs")
        try:
            record = cls(
                pass_id=value["pass_id"],
                task_id=value["task_id"],
                replicate_id=value["replicate_id"],
                split=value["split"],
                cluster_key=value["cluster_key"],
                state_seed=value["state_seed"],
                policy_seed=value["policy_seed"],
                state_dimension=value["state_dimension"],
                state_nbytes=value["state_nbytes"],
                state_sha256=value["state_sha256"],
                initial_task_success=value["initial_task_success"],
                explicit_reset_attempts=value["explicit_reset_attempts"],
            )
        except KeyError as error:
            raise Stage11DArtifactError(
                "Stage-11D state record is missing a field"
            ) from error
        integer_values = (
            record.pass_id,
            record.task_id,
            record.replicate_id,
            record.state_seed,
            record.policy_seed,
            record.state_dimension,
            record.state_nbytes,
            record.explicit_reset_attempts,
        )
        if (
            any(type(item) is not int for item in integer_values)
            or type(record.split) is not str
            or type(record.cluster_key) is not str
            or type(record.state_sha256) is not str
            or type(record.initial_task_success) is not bool
        ):
            raise Stage11DArtifactError("Stage-11D state record types differ")
        return record

    def validate(self, expected: Stage11DRecord, pass_id: int) -> None:
        if (
            pass_id not in STAGE11D_STATE_PASSES
            or self.pass_id != pass_id
            or self.task_id != expected.task_id
            or self.replicate_id != expected.replicate_id
            or self.split != expected.split
            or self.cluster_key != expected.cluster_key
            or self.state_seed != expected.state_seed
            or self.policy_seed != expected.policy_seed
            or self.state_dimension <= 0
            or self.state_nbytes != self.state_dimension * 8
            or len(self.state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.state_sha256
            )
            or self.initial_task_success
            or self.explicit_reset_attempts != 1
        ):
            raise Stage11DArtifactError("Stage-11D state record semantics differ")


def validate_two_pass_states(
    schedule: Sequence[Stage11DRecord],
    first_pass: Sequence[Stage11DStateEvidence],
    second_pass: Sequence[Stage11DStateEvidence],
) -> dict[str, Any]:
    """Require exact coverage, determinism, and task-local state uniqueness."""

    if not (
        len(schedule)
        == len(first_pass)
        == len(second_pass)
        == STAGE11D_CLUSTER_COUNT
    ):
        raise Stage11DArtifactError("Stage-11D two-pass record count differs")
    dimensions: dict[int, set[int]] = {task: set() for task in STAGE11D_TASK_IDS}
    task_hashes: dict[int, set[str]] = {task: set() for task in STAGE11D_TASK_IDS}
    for expected, first, second in zip(
        schedule, first_pass, second_pass, strict=True
    ):
        first.validate(expected, 1)
        second.validate(expected, 2)
        if (
            first.state_dimension != second.state_dimension
            or first.state_nbytes != second.state_nbytes
            or first.state_sha256 != second.state_sha256
        ):
            raise Stage11DArtifactError(
                "Stage-11D state is not byte-identical across passes"
            )
        dimensions[expected.task_id].add(first.state_dimension)
        task_hashes[expected.task_id].add(first.state_sha256)
    if any(len(values) != 1 for values in dimensions.values()):
        raise Stage11DArtifactError("Stage-11D task state dimensions differ")
    if any(
        len(values) != STAGE11D_CLUSTERS_PER_TASK
        for values in task_hashes.values()
    ):
        raise Stage11DArtifactError(
            "Stage-11D task-local generated states are not unique"
        )
    return {
        "records": STAGE11D_CLUSTER_COUNT,
        "passes": len(STAGE11D_STATE_PASSES),
        "byte_identical_records": STAGE11D_CLUSTER_COUNT,
        "initially_solved_records": 0,
        "state_dimensions_per_task": {
            str(task): next(iter(dimensions[task])) for task in STAGE11D_TASK_IDS
        },
        "unique_state_sha_per_task": {
            str(task): len(task_hashes[task]) for task in STAGE11D_TASK_IDS
        },
    }


def validate_state_runner_readiness(repo_root: str | Path) -> dict[str, Any]:
    """Authorize generation only from the tracked, hash-bound readiness record."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    path = root / STAGE11D_RUNNER_READINESS_RELATIVE_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PermissionError("Stage-11D state runner readiness is unavailable") from error
    if not isinstance(value, Mapping):
        raise PermissionError("Stage-11D state runner readiness must be an object")
    files = value.get("runner_files", {})
    authorization = value.get("authorization", {})
    if (
        value.get("schema_version")
        != "phase-route-vla.route-first-stage11d-state-runner-readiness.v1"
        or value.get("status") != "PASS_ROUTE_FIRST_STAGE11D_STATE_RUNNER_READINESS"
        or value.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
        or authorization.get("state_generation") is not True
        or authorization.get("original_A1_collection") is not False
        or authorization.get("same_noise_replay") is not False
        or authorization.get("active_control") is not False
        or not isinstance(files, Mapping)
        or not files
    ):
        raise PermissionError("Stage-11D state runner readiness semantics differ")
    for relative, expected_sha in files.items():
        if (
            type(relative) is not str
            or type(expected_sha) is not str
            or sha256_file(root / relative) != expected_sha
        ):
            raise PermissionError("Stage-11D state runner file hash differs")
    return dict(value)


def load_stage11d_states(
    repo_root: str | Path,
) -> tuple[tuple[Stage11DRecord, ...], dict[int, tuple[np.ndarray, ...]], dict[str, Any]]:
    """Load the final state payload after verifying its attestation and schedule."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    directory = (root / STAGE11D_STATES_RELATIVE_PATH).resolve(strict=True)
    attestation_path = directory / "state_attestation.json"
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage11DArtifactError("Stage-11D state attestation is unreadable") from error
    if not isinstance(attestation, Mapping):
        raise Stage11DArtifactError("Stage-11D state attestation must be an object")
    payload_path = directory / str(attestation.get("payload"))
    schedule = build_stage11d_schedule()
    if (
        attestation.get("schema_version") != STAGE11D_STATE_ATTESTATION_SCHEMA
        or attestation.get("status") != "PASS_ROUTE_FIRST_STAGE11D_STATES_FROZEN"
        or attestation.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
        or attestation.get("payload_sha256") != sha256_file(payload_path)
        or attestation.get("access_ledger", {}).get("policy_action_sampled")
        is not False
        or attestation.get("access_ledger", {}).get("active_control") is not False
    ):
        raise Stage11DArtifactError("Stage-11D state attestation differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != STAGE11D_STATE_PAYLOAD_SCHEMA
        or payload.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
        or payload.get("cluster_keys") != [record.cluster_key for record in schedule]
        or payload.get("splits") != [record.split for record in schedule]
        or payload.get("official_episode_identity_used") is not False
        or payload.get("policy_rollout_performed") is not False
        or len(payload.get("states", [])) != STAGE11D_CLUSTER_COUNT
    ):
        raise Stage11DArtifactError("Stage-11D state payload semantics differ")
    expected_columns = {
        "task_id": [record.task_id for record in schedule],
        "replicate_id": [record.replicate_id for record in schedule],
        "state_seed": [record.state_seed for record in schedule],
        "policy_seed": [record.policy_seed for record in schedule],
    }
    for name, values in expected_columns.items():
        if not torch.equal(payload[name], torch.tensor(values)):
            raise Stage11DArtifactError(f"Stage-11D state payload {name} differs")
    by_task: dict[int, list[np.ndarray]] = {task: [] for task in STAGE11D_TASK_IDS}
    for record, state, expected_hash in zip(
        schedule, payload["states"], payload["state_sha256"], strict=True
    ):
        if not isinstance(state, torch.Tensor) or state.device.type != "cpu":
            raise Stage11DArtifactError("Stage-11D state tensor must be on CPU")
        canonical, _raw, observed_hash = canonical_state_bytes(state.numpy())
        if observed_hash != expected_hash:
            raise Stage11DArtifactError("Stage-11D state payload hash differs")
        by_task[record.task_id].append(canonical.copy())
    frozen = {task: tuple(values) for task, values in by_task.items()}
    if any(len(values) != STAGE11D_CLUSTERS_PER_TASK for values in frozen.values()):
        raise Stage11DArtifactError("Stage-11D state task coverage differs")
    return schedule, frozen, dict(attestation)


class Stage11DFreshStateTaskSuite:
    """Proxy LIBERO-10 while exposing only the 20 Stage-11D states per task."""

    def __init__(
        self,
        base_suite: Any,
        states_by_task: Mapping[int, Sequence[np.ndarray]],
    ) -> None:
        if set(states_by_task) != set(STAGE11D_TASK_IDS):
            raise Stage11DArtifactError("Stage-11D state suite task coverage differs")
        copied = {}
        for task_id in STAGE11D_TASK_IDS:
            states = tuple(states_by_task[task_id])
            if len(states) != STAGE11D_CLUSTERS_PER_TASK:
                raise Stage11DArtifactError(
                    "Stage-11D state suite replicate coverage differs"
                )
            copied[task_id] = tuple(
                canonical_state_bytes(state)[0].copy() for state in states
            )
        self._base_suite = base_suite
        self._states_by_task = copied

    def get_task_init_states(self, task_id: int) -> list[np.ndarray]:
        if type(task_id) is not int or task_id not in STAGE11D_TASK_IDS:
            raise Stage11DArtifactError("Stage-11D task id must be in 0..9")
        return [state.copy() for state in self._states_by_task[task_id]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_suite, name)


__all__ = [
    "STAGE11D_LIBERO_COMMIT",
    "STAGE11D_RUNNER_READINESS_RELATIVE_PATH",
    "STAGE11D_STATE_ATTESTATION_SCHEMA",
    "STAGE11D_STATE_PASSES",
    "STAGE11D_STATE_PAYLOAD_SCHEMA",
    "STAGE11D_STATE_RECORDS_RELATIVE_PATH",
    "STAGE11D_STATE_RECORD_SCHEMA",
    "STAGE11D_STATE_RUN_SCHEMA",
    "STAGE11D_STATES_RELATIVE_PATH",
    "Stage11DArtifactError",
    "Stage11DFreshStateTaskSuite",
    "Stage11DStateEvidence",
    "canonical_state_bytes",
    "load_stage11d_states",
    "sha256_file",
    "validate_state_runner_readiness",
    "validate_two_pass_states",
]
