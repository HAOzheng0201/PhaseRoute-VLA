"""Immutable tracked binding for the Stage-11D generated-state payload.

The generation runner is already part of the state evidence, so this module is
intentionally separate from it.  Downstream collection must validate the
tracked result, the ignored local attestation, and the payload bytes before it
opens any state tensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .route_first_reliability import (
    STAGE11D_CLUSTER_COUNT,
    STAGE11D_CLUSTERS_PER_TASK,
    STAGE11D_PROTOCOL_RELATIVE_PATH,
    STAGE11D_PROTOCOL_SHA256,
    STAGE11D_TASK_IDS,
    validate_stage11d_protocol,
)
from .route_first_reliability_artifacts import (
    STAGE11D_RUNNER_READINESS_RELATIVE_PATH,
    STAGE11D_STATE_ATTESTATION_SCHEMA,
    STAGE11D_STATE_PASSES,
    load_stage11d_states,
    sha256_file,
    validate_state_runner_readiness,
)


STAGE11D_STATE_RESULT_SCHEMA = (
    "phase-route-vla.route-first-stage11d-fresh-state-result.v1"
)
STAGE11D_STATE_RESULT_STATUS = "PASS_ROUTE_FIRST_STAGE11D_FRESH_STATES_FROZEN"
STAGE11D_STATE_RESULT_RELATIVE_PATH = Path(
    "results/route_first/route_first_stage11d_fresh_states.json"
)
STAGE11D_STATE_RESULT_SHA256 = (
    "03fce084b7f46c64dc762cd0f9605981aad6ad51d056b23bd783c7ddcf1c4764"
)
STAGE11D_STATE_BINDING_SCHEMA = (
    "phase-route-vla.route-first-stage11d-fresh-state-binding.v1"
)
STAGE11D_STATE_BINDING_STATUS = (
    "FROZEN_STATE_PAYLOAD_BOUND_COLLECTION_RUNNER_NOT_VALIDATED"
)
STAGE11D_STATE_BINDING_RELATIVE_PATH = Path(
    "configs/research/route_first_stage11d_fresh_state_binding.json"
)
STAGE11D_STATE_BINDING_SHA256 = (
    "0f1ffcf23310dbb782986cdf93cac8439054f249f36b8dc8118919f479f0434d"
)
STAGE11D_RUNNER_READINESS_SHA256 = (
    "c4b5f421179706dfdaea4d68eaf10bf8813eb99f116f4b73d452c95281c995f0"
)


class Stage11DStateBindingError(PermissionError):
    """Raised when tracked or local Stage-11D state evidence drifts."""


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage11DStateBindingError(f"Unreadable JSON evidence: {path}") from error
    if not isinstance(value, Mapping):
        raise Stage11DStateBindingError(f"JSON evidence must be an object: {path}")
    return dict(value)


def _inside(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative:
        raise Stage11DStateBindingError("Stage-11D artifact path is invalid")
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise Stage11DStateBindingError("Stage-11D artifact escaped repository root")
    return path


def load_stage11d_state_binding(repo_root: str | Path) -> dict[str, Any]:
    """Validate tracked evidence without opening the ignored state artifacts."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    validate_stage11d_protocol(root)
    validate_state_runner_readiness(root)
    binding_path = root / STAGE11D_STATE_BINDING_RELATIVE_PATH
    result_path = root / STAGE11D_STATE_RESULT_RELATIVE_PATH
    readiness_path = root / STAGE11D_RUNNER_READINESS_RELATIVE_PATH
    if sha256_file(binding_path) != STAGE11D_STATE_BINDING_SHA256:
        raise Stage11DStateBindingError("Stage-11D state binding SHA-256 differs")
    if sha256_file(result_path) != STAGE11D_STATE_RESULT_SHA256:
        raise Stage11DStateBindingError("Stage-11D fresh-state result SHA-256 differs")
    if sha256_file(readiness_path) != STAGE11D_RUNNER_READINESS_SHA256:
        raise Stage11DStateBindingError("Stage-11D runner readiness SHA-256 differs")

    binding = _object(binding_path)
    result = _object(result_path)
    protocol = binding.get("protocol", {})
    readiness = binding.get("runner_readiness", {})
    tracked_result = binding.get("tracked_result", {})
    attestation = binding.get("local_state_attestation", {})
    payload = binding.get("local_state_payload", {})
    local = result.get("local_ignored_artifacts", {})
    authorization = binding.get("authorization", {})
    if (
        binding.get("schema_version") != STAGE11D_STATE_BINDING_SCHEMA
        or binding.get("status") != STAGE11D_STATE_BINDING_STATUS
        or binding.get("suite") != "libero_10"
        or protocol.get("path") != str(STAGE11D_PROTOCOL_RELATIVE_PATH)
        or protocol.get("sha256") != STAGE11D_PROTOCOL_SHA256
        or readiness.get("path") != str(STAGE11D_RUNNER_READINESS_RELATIVE_PATH)
        or readiness.get("sha256") != STAGE11D_RUNNER_READINESS_SHA256
        or tracked_result.get("path") != str(STAGE11D_STATE_RESULT_RELATIVE_PATH)
        or tracked_result.get("sha256") != STAGE11D_STATE_RESULT_SHA256
        or result.get("schema_version") != STAGE11D_STATE_RESULT_SCHEMA
        or result.get("status") != STAGE11D_STATE_RESULT_STATUS
        or result.get("source_generation_commit")
        != binding.get("source_generation_commit")
        or local.get("state_attestation_path") != attestation.get("path")
        or local.get("state_attestation_sha256") != attestation.get("sha256")
        or local.get("state_attestation_bytes") != attestation.get("bytes")
        or local.get("state_payload_path") != payload.get("path")
        or local.get("state_payload_sha256") != payload.get("sha256")
        or local.get("state_payload_bytes") != payload.get("bytes")
        or payload.get("records") != STAGE11D_CLUSTER_COUNT
        or authorization.get("on_binding_validation_pass")
        != "ORIGINAL_A1_OBSERVATION_ONLY_RUNNER_IMPLEMENTATION_AND_CPU_CONTRACT_TESTS"
        or authorization.get("collection_requires_separate_clean_runner_commit")
        is not True
        or authorization.get("original_A1_collection_started") is not False
        or authorization.get("same_noise_replay_started") is not False
        or authorization.get("training_started") is not False
        or authorization.get("active_control_started") is not False
        or authorization.get("deployment_authorized") is not False
    ):
        raise Stage11DStateBindingError("Stage-11D state binding semantics differ")
    return binding


def validate_local_stage11d_state_artifacts(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate exact ignored attestation/payload bytes without loading tensors."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    binding = load_stage11d_state_binding(root)
    attestation_binding = binding["local_state_attestation"]
    payload_binding = binding["local_state_payload"]
    attestation_path = _inside(root, attestation_binding["path"])
    payload_path = _inside(root, payload_binding["path"])
    if (
        sha256_file(attestation_path) != attestation_binding["sha256"]
        or attestation_path.stat().st_size != attestation_binding["bytes"]
        or sha256_file(payload_path) != payload_binding["sha256"]
        or payload_path.stat().st_size != payload_binding["bytes"]
    ):
        raise Stage11DStateBindingError(
            "Stage-11D local state artifact SHA or size differs"
        )

    attestation = _object(attestation_path)
    audit = attestation.get("audit", {})
    access = attestation.get("access_ledger", {})
    unique = audit.get("unique_state_sha_per_task", {})
    if (
        attestation.get("schema_version") != STAGE11D_STATE_ATTESTATION_SCHEMA
        or attestation.get("status") != "PASS_ROUTE_FIRST_STAGE11D_STATES_FROZEN"
        or attestation.get("source_git_commit")
        != binding.get("source_generation_commit")
        or attestation.get("source_worktree_dirty") is not False
        or attestation.get("protocol_sha256") != STAGE11D_PROTOCOL_SHA256
        or attestation.get("runner_readiness_sha256")
        != STAGE11D_RUNNER_READINESS_SHA256
        or attestation.get("payload") != payload_path.name
        or attestation.get("payload_sha256") != payload_binding["sha256"]
        or attestation.get("payload_bytes") != payload_binding["bytes"]
        or audit.get("records") != STAGE11D_CLUSTER_COUNT
        or audit.get("passes") != len(STAGE11D_STATE_PASSES)
        or audit.get("byte_identical_records") != STAGE11D_CLUSTER_COUNT
        or audit.get("initially_solved_records") != 0
        or set(unique) != {str(task) for task in STAGE11D_TASK_IDS}
        or any(value != STAGE11D_CLUSTERS_PER_TASK for value in unique.values())
        or access.get("model_checkpoint_loaded") is not False
        or access.get("policy_action_sampled") is not False
        or access.get("official_states_0_to_49_opened") is not False
        or access.get("V3_D8_or_route_first_Stage10_states_reused") is not False
        or access.get("gpu_query_or_initialization") != 0
        or access.get("active_control") is not False
    ):
        raise Stage11DStateBindingError(
            "Stage-11D local state attestation semantics differ"
        )
    return {
        "binding": binding,
        "attestation": attestation,
        "attestation_path": attestation_path,
        "payload_path": payload_path,
    }


def load_bound_stage11d_states(repo_root: str | Path):
    """Open state tensors only after the tracked and local bindings pass."""

    validate_local_stage11d_state_artifacts(repo_root)
    return load_stage11d_states(repo_root)


__all__ = [
    "STAGE11D_STATE_BINDING_RELATIVE_PATH",
    "STAGE11D_STATE_BINDING_SHA256",
    "STAGE11D_STATE_RESULT_RELATIVE_PATH",
    "STAGE11D_STATE_RESULT_SHA256",
    "Stage11DStateBindingError",
    "load_bound_stage11d_states",
    "load_stage11d_state_binding",
    "validate_local_stage11d_state_artifacts",
]
