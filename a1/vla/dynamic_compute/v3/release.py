"""Fail-closed validation for the frozen PhaseRoute-V3 research runtime.

The small router and phase estimator are versioned with the repository.  The
34 GB A1 backbone remains an external, revision-pinned artifact.  Validation
keeps these two scopes separate so a clean clone can audit the release bundle
without pretending that it is ready to execute a rollout.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


V3_RELEASE_SCHEMA_VERSION = "phase-route-vla.v3.release-gate.v1"
V3_ARTIFACT_SCHEMA_VERSION = "phase-route-vla.v3.runtime-artifacts.v1"
V3_RELEASE_METHOD = "phase_route_v3"

MANIFEST_RELATIVE_PATH = Path("artifacts/phase_route_v3/MANIFEST.json")
ROUTER_RELATIVE_PATH = Path("artifacts/phase_route_v3/final_router.pt")
PHASE_RELATIVE_PATH = Path("artifacts/phase_route_v3/phase_estimator.pt")
THRESHOLD_RELATIVE_PATH = Path(
    "artifacts/phase_route_v3/exit_thresholds_libero_10_exp_1.0.json"
)
FORMAL_RESULT_RELATIVE_PATH = Path("results/v3/v3_d9_final_result.json")

BACKBONE_FILENAMES = (
    "model.pt",
    "config.yaml",
    "dataset_statistics.json",
)

ROUTER_SHA256 = "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830"
ROUTER_BYTES = 22_290
PHASE_SHA256 = "b601f8221d47136818d7a008eaf7cee06e1201bf514f371ae33f42cfb39515a1"
PHASE_BYTES = 11_344_688
PHASE_STATE_SHA256 = (
    "8c0021be43d1cea28890833fd5e1faa8ee0191e809cbf3b1df0d3c36010d7598"
)
THRESHOLD_SHA256 = (
    "a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796"
)
THRESHOLD_BYTES = 236
FORMAL_RESULT_SHA256 = (
    "4df77237e84ad82b05ae67145e52000b0e3430f34b6f69fcbee743687ac11952"
)

BACKBONE_EXPECTED = {
    "model.pt": {
        "bytes": 33_841_175_207,
        "sha256": "dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f",
    },
    "config.yaml": {
        "bytes": 8_369,
        "sha256": "9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca",
    },
    "dataset_statistics.json": {
        "bytes": 11_871,
        "sha256": "6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3",
    },
}


class V3ReleaseError(ValueError):
    """Raised when an immutable release contract is malformed."""


def parse_index_spec(value: str | None, *, name: str) -> tuple[int, ...] | None:
    """Parse a stable non-negative index list such as ``0,2-4``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    result = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            raise ValueError(f"{name} contains an empty item")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2 or not all(part.isdigit() for part in bounds):
                raise ValueError(f"invalid {name} range: {token}")
            start, end = (int(part) for part in bounds)
            if end < start:
                raise ValueError(f"descending {name} range: {token}")
            result.extend(range(start, end + 1))
        elif token.isdigit():
            result.append(int(token))
        else:
            raise ValueError(f"invalid {name} item: {token}")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate indices")
    return tuple(result)


def validate_general_release_selection(
    task_spec: str, episode_spec: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate a non-D9 general simulator selection."""

    task_ids = parse_index_spec(task_spec, name="task_ids")
    episode_indices = parse_index_spec(episode_spec, name="episode_indices")
    if not task_ids or any(task_id not in range(10) for task_id in task_ids):
        raise ValueError("task_ids must select unique LIBERO-10 tasks in 0..9")
    if not episode_indices or any(index not in range(50) for index in episode_indices):
        raise ValueError("episode_indices must select unique official states in 0..49")
    consumed_d9 = sorted(set(episode_indices).intersection(range(40, 50)))
    if consumed_d9:
        raise ValueError(
            "the general release runner refuses consumed D9 test states 40..49; "
            f"requested {consumed_d9}"
        )
    return task_ids, episode_indices


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V3ReleaseError(f"expected a JSON object: {path}")
    return value


def _file_evidence(path: Path, *, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "bytes": None,
            "sha256": None,
            "size_matches": False,
            "sha256_matches": False,
        }
    size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    return {
        "path": str(path),
        "present": True,
        "bytes": size,
        "sha256": actual_sha256,
        "size_matches": size == expected_bytes,
        "sha256_matches": actual_sha256 == expected_sha256,
    }


def _artifact_specs() -> Mapping[str, Mapping[str, Any]]:
    return {
        "router": {
            "path": ROUTER_RELATIVE_PATH,
            "bytes": ROUTER_BYTES,
            "sha256": ROUTER_SHA256,
        },
        "phase_estimator": {
            "path": PHASE_RELATIVE_PATH,
            "bytes": PHASE_BYTES,
            "sha256": PHASE_SHA256,
        },
        "libero_10_thresholds": {
            "path": THRESHOLD_RELATIVE_PATH,
            "bytes": THRESHOLD_BYTES,
            "sha256": THRESHOLD_SHA256,
        },
    }


def _manifest_checks(manifest: Mapping[str, Any]) -> dict[str, bool]:
    files = manifest.get("files")
    backbone = manifest.get("backbone")
    formal_result = manifest.get("formal_result")
    checks = {
        "schema_version": manifest.get("schema_version")
        == V3_ARTIFACT_SCHEMA_VERSION,
        "method": manifest.get("method") == V3_RELEASE_METHOD,
        "research_only": manifest.get("deployment_authorized") is False,
        "files_object": isinstance(files, dict),
        "backbone_object": isinstance(backbone, dict),
        "formal_result_object": isinstance(formal_result, dict),
    }
    if not isinstance(files, dict):
        return checks
    for name, spec in _artifact_specs().items():
        entry = files.get(name)
        checks[f"manifest_{name}"] = isinstance(entry, dict) and (
            entry.get("path") == str(spec["path"])
            and entry.get("bytes") == spec["bytes"]
            and entry.get("sha256") == spec["sha256"]
        )
    if isinstance(files.get("phase_estimator"), dict):
        checks["manifest_phase_state"] = (
            files["phase_estimator"].get("state_sha256") == PHASE_STATE_SHA256
        )
    if isinstance(formal_result, dict):
        checks["manifest_formal_result"] = (
            formal_result.get("path") == str(FORMAL_RESULT_RELATIVE_PATH)
            and formal_result.get("sha256") == FORMAL_RESULT_SHA256
            and formal_result.get("status")
            == "PASS_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST"
            and formal_result.get("suite") == "libero_10"
            and formal_result.get("pairs") == 100
        )
    if isinstance(backbone, dict) and isinstance(backbone.get("files"), dict):
        for name, expected in BACKBONE_EXPECTED.items():
            entry = backbone["files"].get(name)
            checks[f"manifest_backbone_{name}"] = isinstance(entry, dict) and (
                entry.get("bytes") == expected["bytes"]
                and entry.get("sha256") == expected["sha256"]
            )
    return checks


def summarize_runtime_records(records: Any) -> dict[str, Any]:
    """Create a JSON-safe, control-independent summary of runtime records."""

    materialized = tuple(records)
    layer_counts = Counter(
        int(record["selected_layer"])
        for record in materialized
        if isinstance(record, Mapping)
        and record.get("selected_layer") in (11, 13, 27)
    )
    error_records = sum(
        bool(record.get("errors"))
        for record in materialized
        if isinstance(record, Mapping)
    )
    fallback_records = sum(
        bool(record.get("fallback"))
        for record in materialized
        if isinstance(record, Mapping)
    )
    return {
        "schema_version": "phase-route-vla.v3.runtime-summary.v1",
        "records": len(materialized),
        "selected_layers": {
            "11": layer_counts[11],
            "13": layer_counts[13],
            "27": layer_counts[27],
        },
        "early_exit_records": layer_counts[11] + layer_counts[13],
        "fallback_records": fallback_records,
        "records_with_errors": error_records,
    }


def validate_phase_route_v3_release(
    repo_root: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    require_backbone: bool = False,
    validate_payloads: bool = True,
) -> dict[str, Any]:
    """Validate bundled artifacts, D9 evidence, and optionally the A1 backbone.

    ``require_backbone=False`` is the clean-clone audit.  A runnable launcher
    must use ``require_backbone=True`` and therefore supply the exact external
    checkpoint directory.
    """

    root = Path(repo_root).resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing V3 artifact manifest: {manifest_path}")
    manifest = _load_object(manifest_path)
    checks = _manifest_checks(manifest)

    artifacts: dict[str, Any] = {}
    for name, spec in _artifact_specs().items():
        evidence = _file_evidence(
            root / spec["path"],
            expected_bytes=int(spec["bytes"]),
            expected_sha256=str(spec["sha256"]),
        )
        artifacts[name] = evidence
        checks[f"{name}_present"] = evidence["present"]
        checks[f"{name}_size"] = evidence["size_matches"]
        checks[f"{name}_sha256"] = evidence["sha256_matches"]

    result_path = root / FORMAL_RESULT_RELATIVE_PATH
    result_evidence = _file_evidence(
        result_path,
        expected_bytes=(result_path.stat().st_size if result_path.is_file() else -1),
        expected_sha256=FORMAL_RESULT_SHA256,
    )
    artifacts["formal_result"] = result_evidence
    checks["formal_result_present"] = result_evidence["present"]
    checks["formal_result_sha256"] = result_evidence["sha256_matches"]
    if result_path.is_file():
        result = _load_object(result_path)
        gate_checks = result.get("gate_checks")
        checks.update(
            {
                "formal_result_status": result.get("status")
                == "PASS_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST",
                "formal_result_schema": result.get("schema_version")
                == "phase-route-vla.v3.d9-final-result.v1",
                "formal_result_suite": result.get("suite") == "libero_10",
                "formal_result_100_pairs": result.get("success", {}).get("pairs")
                == 100,
                "formal_result_all_gates": bool(gate_checks)
                and all(value is True for value in gate_checks.values())
                and result.get("all_primary_gates_pass") is True,
                "formal_result_research_boundary": result.get("authorization", {}).get(
                    "deployment_authorized"
                )
                is False,
            }
        )

    payload_validation: dict[str, Any] = {"requested": bool(validate_payloads)}
    if validate_payloads and all(
        checks.get(f"{name}_sha256") for name in ("router", "phase_estimator")
    ):
        from .active_runtime import load_frozen_phase_route_runtime

        runtime = load_frozen_phase_route_runtime(
            root / ROUTER_RELATIVE_PATH,
            root / PHASE_RELATIVE_PATH,
        )
        payload_validation.update(
            {
                "loaded": True,
                "phase_state_sha256": runtime.phase_state_sha256,
                "router_models": len(runtime.adapter.router.models),
            }
        )
        checks["payloads_load"] = True
        checks["phase_state_sha256"] = (
            runtime.phase_state_sha256 == PHASE_STATE_SHA256
        )
        checks["five_router_heads"] = len(runtime.adapter.router.models) == 5
    elif validate_payloads:
        checks["payloads_load"] = False
        checks["phase_state_sha256"] = False
        checks["five_router_heads"] = False

    backbone: dict[str, Any] = {
        "required": bool(require_backbone),
        "checkpoint_dir": None,
        "files": {},
    }
    if require_backbone:
        if checkpoint_dir is None:
            raise V3ReleaseError("checkpoint_dir is required when require_backbone=True")
        checkpoint = Path(checkpoint_dir).resolve()
        backbone["checkpoint_dir"] = str(checkpoint)
        for name, expected in BACKBONE_EXPECTED.items():
            evidence = _file_evidence(
                checkpoint / name,
                expected_bytes=int(expected["bytes"]),
                expected_sha256=str(expected["sha256"]),
            )
            backbone["files"][name] = evidence
            checks[f"backbone_{name}_present"] = evidence["present"]
            checks[f"backbone_{name}_size"] = evidence["size_matches"]
            checks[f"backbone_{name}_sha256"] = evidence["sha256_matches"]

    status = "PASS" if checks and all(checks.values()) else "FAIL"
    return {
        "schema_version": V3_RELEASE_SCHEMA_VERSION,
        "status": status,
        "release_method": V3_RELEASE_METHOD,
        "scope": "libero_10_research_reproduction",
        "runtime_default_enabled": False,
        "deployment_authorized": False,
        "repo_root": str(root),
        "manifest": str(manifest_path),
        "artifacts": artifacts,
        "payload_validation": payload_validation,
        "backbone": backbone,
        "checks": checks,
    }


__all__ = [
    "BACKBONE_EXPECTED",
    "FORMAL_RESULT_SHA256",
    "MANIFEST_RELATIVE_PATH",
    "PHASE_RELATIVE_PATH",
    "PHASE_SHA256",
    "PHASE_STATE_SHA256",
    "ROUTER_RELATIVE_PATH",
    "ROUTER_SHA256",
    "THRESHOLD_RELATIVE_PATH",
    "THRESHOLD_SHA256",
    "V3ReleaseError",
    "parse_index_spec",
    "validate_general_release_selection",
    "sha256_file",
    "summarize_runtime_records",
    "validate_phase_route_v3_release",
]
