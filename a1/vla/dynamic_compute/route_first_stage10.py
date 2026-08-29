"""Frozen contracts for route-first Stage 10 fresh active confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = (
    "phase-route-vla.route-first-stage10-fresh-confirmation-protocol.v1"
)
PROTOCOL_STATUS = "FROZEN_INFRASTRUCTURE_NOT_VALIDATED"
PROTOCOL_RELATIVE_PATH = Path(
    "configs/route_first_stage10_fresh_confirmation_protocol.json"
)
PROTOCOL_SHA256 = (
    "62f5be1524676cd2db045de32964ff3206a455d5fd8e8b29eb10e134521bc604"
)
SCHEDULE_SCHEMA = "phase-route-vla.route-first-stage10-fresh-schedule.v1"
SCHEDULE_STATUS = "FROZEN_NOT_GENERATED"
SCHEDULE_RELATIVE_PATH = Path("configs/route_first_stage10_fresh_schedule.json")
SCHEDULE_SHA256 = (
    "c2c41259c5db1b79d6f2da68ec77c200d829670fb7cd17b4abc19f63a37f43d4"
)
STAGE9_RESULT_RELATIVE_PATH = Path(
    "results/route_first/route_first_stage9_state13_pilot.json"
)
STAGE9_RESULT_SHA256 = (
    "0979f04e8f7c3352b2bbea8540a2562925546233d03905c6d579d077795d1d8c"
)

TASK_IDS = tuple(range(10))
REPLICATE_IDS = tuple(range(6))
METHODS = ("original_a1", "candidate_first_v3", "route_first_stage8")
STATE_SEED_BASE = 71_260_829
POLICY_SEED_BASE = 81_260_829
TRIPLET_COUNT = 60
ACTIVE_ROLLOUT_COUNT = 180
STATE_PASSES = (1, 2)
STATE_RECORD_SCHEMA = "phase-route-vla.route-first-stage10-state-record.v1"
STATE_PAYLOAD_SCHEMA = "phase-route-vla.route-first-stage10-state-payload.v1"
STATE_ATTESTATION_SCHEMA = (
    "phase-route-vla.route-first-stage10-state-attestation.v1"
)

ARM_ORDERS = (
    ("original_a1", "candidate_first_v3", "route_first_stage8"),
    ("original_a1", "route_first_stage8", "candidate_first_v3"),
    ("candidate_first_v3", "original_a1", "route_first_stage8"),
    ("candidate_first_v3", "route_first_stage8", "original_a1"),
    ("route_first_stage8", "original_a1", "candidate_first_v3"),
    ("route_first_stage8", "candidate_first_v3", "original_a1"),
)


class Stage10ContractError(ValueError):
    """Raised when Stage 10 protocol, schedule, or state evidence drifts."""


def canonical_state_bytes(state: Any) -> tuple[Any, bytes, str]:
    """Return a finite 1D MuJoCo state as canonical little-endian FP64 bytes."""

    import numpy as np

    value = np.asarray(state)
    if (
        value.ndim != 1
        or value.size == 0
        or value.dtype.kind != "f"
        or value.dtype.itemsize != 8
        or not bool(np.isfinite(value).all())
    ):
        raise Stage10ContractError(
            "Stage 10 state must be finite nonempty float64 [D]"
        )
    canonical = np.ascontiguousarray(value.astype("<f8", copy=False))
    raw = canonical.tobytes(order="C")
    if len(raw) != canonical.size * 8:
        raise Stage10ContractError("Stage 10 canonical state byte count differs")
    return canonical, raw, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FreshTripletSpec:
    task_id: int
    replicate_id: int
    cluster_key: str
    state_seed: int
    policy_seed: int
    arm_order: tuple[str, str, str]


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
        try:
            return cls(
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
            raise Stage10ContractError(
                "Stage 10 state evidence is incomplete"
            ) from error

    def validate(self, expected: FreshTripletSpec, pass_id: int) -> None:
        integers = (
            self.pass_id,
            self.task_id,
            self.replicate_id,
            self.state_seed,
            self.policy_seed,
            self.state_dimension,
            self.state_nbytes,
            self.explicit_reset_attempts,
        )
        if any(type(item) is not int for item in integers):
            raise Stage10ContractError("Stage 10 state integer field type differs")
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
            or any(
                character not in "0123456789abcdef"
                for character in self.state_sha256
            )
            or type(self.initial_task_success) is not bool
            or self.initial_task_success
            or self.explicit_reset_attempts != 1
        ):
            raise Stage10ContractError("Stage 10 state evidence semantics differ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Stage10ContractError(f"expected JSON object: {path}")
    return dict(value)


def load_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    schedule_path = root / SCHEDULE_RELATIVE_PATH
    stage9_path = root / STAGE9_RESULT_RELATIVE_PATH
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise PermissionError("Stage 10 protocol SHA-256 differs")
    if sha256_file(schedule_path) != SCHEDULE_SHA256:
        raise PermissionError("Stage 10 schedule SHA-256 differs")
    if sha256_file(stage9_path) != STAGE9_RESULT_SHA256:
        raise PermissionError("Stage 9 prerequisite SHA-256 differs")
    protocol = _object(protocol_path)
    stage9 = _object(stage9_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
        or protocol.get("fresh_schedule", {}).get("sha256") != SCHEDULE_SHA256
        or protocol.get("prerequisite", {}).get("stage9_result_sha256")
        != STAGE9_RESULT_SHA256
        or protocol.get("access_ledger", {}).get("new_generated_states_opened")
        is not False
        or protocol.get("access_ledger", {}).get("active_rollouts_started")
        is not False
        or stage9.get("status") != "PASS"
        or stage9.get("next_gate", {}).get("fresh_state_confirmation_authorized")
        is not True
    ):
        raise PermissionError("Stage 10 protocol authorization semantics differ")
    return protocol


def load_schedule(repo_root: str | Path) -> tuple[FreshTripletSpec, ...]:
    root = Path(repo_root).resolve(strict=True)
    load_protocol(root)
    source = _object(root / SCHEDULE_RELATIVE_PATH)
    state = source.get("state_generation", {})
    policy = source.get("policy_rollout", {})
    order_source = source.get("arm_order_by_replicate", {})
    if (
        source.get("schema_version") != SCHEDULE_SCHEMA
        or source.get("status") != SCHEDULE_STATUS
        or tuple(source.get("task_ids", ())) != TASK_IDS
        or tuple(source.get("replicate_ids", ())) != REPLICATE_IDS
        or source.get("triplets") != TRIPLET_COUNT
        or source.get("active_rollouts") != ACTIVE_ROLLOUT_COUNT
        or state.get("seed_base") != STATE_SEED_BASE
        or policy.get("seed_base") != POLICY_SEED_BASE
        or tuple(source.get("methods", ())) != METHODS
        or source.get("official_benchmark_episode_index", "unexpected") is not None
    ):
        raise PermissionError("Stage 10 schedule identity differs")
    observed_orders = tuple(
        tuple(order_source.get(str(index), ())) for index in REPLICATE_IDS
    )
    if observed_orders != ARM_ORDERS:
        raise PermissionError("Stage 10 arm counterbalancing differs")
    if any(set(order) != set(METHODS) for order in observed_orders):
        raise Stage10ContractError("Stage 10 arm order is not a permutation")
    specs = tuple(
        FreshTripletSpec(
            task_id=task_id,
            replicate_id=replicate_id,
            cluster_key=(
                f"libero_10:task{task_id}:route_first_fresh_v1:"
                f"replicate{replicate_id}"
            ),
            state_seed=STATE_SEED_BASE + task_id * 10_000 + replicate_id,
            policy_seed=POLICY_SEED_BASE + task_id * 10_000 + replicate_id,
            arm_order=ARM_ORDERS[replicate_id],
        )
        for task_id in TASK_IDS
        for replicate_id in REPLICATE_IDS
    )
    if len(specs) != TRIPLET_COUNT or len(
        {item.cluster_key for item in specs}
    ) != len(specs):
        raise Stage10ContractError("Stage 10 triplet coverage differs")
    return specs


def validate_two_pass_states(
    specs: Sequence[FreshTripletSpec],
    pass1: Sequence[FreshStateEvidence],
    pass2: Sequence[FreshStateEvidence],
) -> dict[str, Any]:
    if not (len(specs) == len(pass1) == len(pass2) == TRIPLET_COUNT):
        raise Stage10ContractError("Stage 10 two-pass state count differs")
    task_dimensions = {task_id: set() for task_id in TASK_IDS}
    task_hashes = {task_id: set() for task_id in TASK_IDS}
    for spec, first, second in zip(specs, pass1, pass2, strict=True):
        first.validate(spec, 1)
        second.validate(spec, 2)
        if (
            first.state_dimension != second.state_dimension
            or first.state_nbytes != second.state_nbytes
            or first.state_sha256 != second.state_sha256
        ):
            raise Stage10ContractError("Stage 10 generated state is nondeterministic")
        task_dimensions[spec.task_id].add(first.state_dimension)
        task_hashes[spec.task_id].add(first.state_sha256)
    if any(len(values) != 1 for values in task_dimensions.values()):
        raise Stage10ContractError("Stage 10 task-local state dimensions differ")
    if any(len(values) != len(REPLICATE_IDS) for values in task_hashes.values()):
        raise Stage10ContractError(
            "Stage 10 generated states are not task-local unique"
        )
    return {
        "records": TRIPLET_COUNT,
        "passes": 2,
        "byte_identical_records": TRIPLET_COUNT,
        "initially_solved_records": 0,
        "state_dimensions_per_task": {
            str(task_id): next(iter(task_dimensions[task_id])) for task_id in TASK_IDS
        },
        "unique_state_sha_per_task": {
            str(task_id): len(task_hashes[task_id]) for task_id in TASK_IDS
        },
    }


def validate_generation_record_manifest(
    specs: Sequence[FreshTripletSpec], manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Validate the immutable hash manifest for every two-pass state record."""

    expected_keys = {
        f"pass{pass_id}:task{spec.task_id}:replicate{spec.replicate_id}"
        for pass_id in STATE_PASSES
        for spec in specs
    }
    if len(specs) != TRIPLET_COUNT or set(manifest) != expected_keys:
        raise Stage10ContractError(
            "Stage 10 generation record manifest coverage differs"
        )
    normalized: dict[str, str] = {}
    for key in sorted(expected_keys):
        digest = manifest[key]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Stage10ContractError(
                "Stage 10 generation record manifest digest differs"
            )
        normalized[key] = digest
    return normalized


__all__ = [
    "ACTIVE_ROLLOUT_COUNT",
    "ARM_ORDERS",
    "FreshStateEvidence",
    "FreshTripletSpec",
    "METHODS",
    "POLICY_SEED_BASE",
    "PROTOCOL_SHA256",
    "REPLICATE_IDS",
    "SCHEDULE_SHA256",
    "STATE_ATTESTATION_SCHEMA",
    "STATE_PASSES",
    "STATE_PAYLOAD_SCHEMA",
    "STATE_RECORD_SCHEMA",
    "STATE_SEED_BASE",
    "Stage10ContractError",
    "TASK_IDS",
    "TRIPLET_COUNT",
    "canonical_state_bytes",
    "load_protocol",
    "load_schedule",
    "sha256_file",
    "validate_two_pass_states",
    "validate_generation_record_manifest",
]
