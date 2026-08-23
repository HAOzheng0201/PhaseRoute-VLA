"""Fail-closed artifact validation for frozen V3-D8A state generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .fresh_confirmation import (
    D8_CLUSTERS_PER_TASK,
    D8_CLUSTER_COUNT,
    D8_TASK_IDS,
    FreshConfirmationRecord,
)


D8A_RECORD_SCHEMA_VERSION = "phase-route-vla.v3.d8a-fresh-state-record.v1"
D8A_PAYLOAD_SCHEMA_VERSION = "phase-route-vla.v3.d8a-fresh-state-payload.v1"
D8A_RESULT_SCHEMA_VERSION = "phase-route-vla.v3.d8a-fresh-state-result.v1"


class D8ArtifactError(ValueError):
    """Raised when generated evidence differs from the frozen D8A contract."""


def canonical_state_bytes(state: Any) -> tuple[np.ndarray, bytes, str]:
    """Validate one MuJoCo state and return canonical little-endian FP64 bytes."""

    value = np.asarray(state)
    if (
        value.ndim != 1
        or value.size == 0
        or value.dtype.kind != "f"
        or value.dtype.itemsize != 8
        or not bool(np.isfinite(value).all())
    ):
        raise D8ArtifactError("D8A state must be finite nonempty float64 [D]")
    canonical = np.ascontiguousarray(value.astype("<f8", copy=False))
    raw = canonical.tobytes(order="C")
    if len(raw) != canonical.size * 8:
        raise D8ArtifactError("D8A canonical state byte count differs")
    return canonical, raw, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FreshStateEvidence:
    pass_id: int
    task_id: int
    replicate_id: int
    cluster_key: str
    state_seed: int
    policy_seed: int
    state_dimension: int
    state_nbytes: int
    state_sha256: str
    initial_task_success: bool
    explicit_reset_attempts: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FreshStateEvidence":
        if value.get("schema_version") != D8A_RECORD_SCHEMA_VERSION:
            raise D8ArtifactError("D8A record schema differs")
        try:
            record = cls(
                pass_id=value["pass_id"],
                task_id=value["task_id"],
                replicate_id=value["replicate_id"],
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
            raise D8ArtifactError("D8A record is missing a required field") from error
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
            or type(record.cluster_key) is not str
            or type(record.state_sha256) is not str
            or type(record.initial_task_success) is not bool
        ):
            raise D8ArtifactError("D8A record field type differs")
        return record

    def validate_against(self, expected: FreshConfirmationRecord, pass_id: int) -> None:
        if (
            self.pass_id != pass_id
            or self.task_id != expected.task_id
            or self.replicate_id != expected.replicate_id
            or self.cluster_key != expected.cluster_key
            or self.state_seed != expected.state_seed
            or self.policy_seed != expected.policy_seed
            or self.state_dimension <= 0
            or self.state_nbytes != 8 * self.state_dimension
            or len(self.state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.state_sha256)
            or self.initial_task_success
            or self.explicit_reset_attempts != 1
        ):
            raise D8ArtifactError("D8A record semantics differ")


def validate_two_pass_evidence(
    expected_records: Sequence[FreshConfirmationRecord],
    first_pass: Sequence[FreshStateEvidence],
    second_pass: Sequence[FreshStateEvidence],
) -> dict[str, Any]:
    """Validate coverage, determinism, dimensions, uniqueness, and solved veto."""

    if (
        len(expected_records) != D8_CLUSTER_COUNT
        or len(first_pass) != D8_CLUSTER_COUNT
        or len(second_pass) != D8_CLUSTER_COUNT
    ):
        raise D8ArtifactError("D8A two-pass record count differs")
    dimensions: dict[int, set[int]] = {task: set() for task in D8_TASK_IDS}
    task_hashes: dict[int, set[str]] = {task: set() for task in D8_TASK_IDS}
    for expected, first, second in zip(expected_records, first_pass, second_pass):
        first.validate_against(expected, 1)
        second.validate_against(expected, 2)
        if (
            first.state_dimension != second.state_dimension
            or first.state_nbytes != second.state_nbytes
            or first.state_sha256 != second.state_sha256
        ):
            raise D8ArtifactError("D8A state is not byte-identical across passes")
        dimensions[expected.task_id].add(first.state_dimension)
        task_hashes[expected.task_id].add(first.state_sha256)
    if any(len(value) != 1 for value in dimensions.values()):
        raise D8ArtifactError("D8A task-local state dimension is not constant")
    if any(len(value) != D8_CLUSTERS_PER_TASK for value in task_hashes.values()):
        raise D8ArtifactError("D8A task-local generated states are not unique")
    return {
        "records": D8_CLUSTER_COUNT,
        "passes": 2,
        "byte_identical_records": D8_CLUSTER_COUNT,
        "initially_solved_records": 0,
        "state_dimensions_per_task": {
            str(task): next(iter(dimensions[task])) for task in D8_TASK_IDS
        },
        "unique_state_sha_per_task": {
            str(task): len(task_hashes[task]) for task in D8_TASK_IDS
        },
    }


def finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D8ArtifactError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise D8ArtifactError(f"{name} must be finite")
    return result


__all__ = [
    "D8A_PAYLOAD_SCHEMA_VERSION",
    "D8A_RECORD_SCHEMA_VERSION",
    "D8A_RESULT_SCHEMA_VERSION",
    "D8ArtifactError",
    "FreshStateEvidence",
    "canonical_state_bytes",
    "finite_float",
    "validate_two_pass_evidence",
]
