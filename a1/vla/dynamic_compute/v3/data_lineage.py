"""Metadata-only episode lineage auditing for PhaseRoute V3.

This module intentionally depends only on the Python standard library.  It
accepts two explicit selection formats:

* a V3 JSON selection document whose schema version is frozen below; and
* a pre-frozen five-integer-field metadata-index JSONL, with suite/role/seed
  supplied by an equally explicit source descriptor.

It does not accept row-level evaluation records, tensor/array/checkpoint
formats, or schema inference.  In particular, C3.61 row-level records are an
invalid lineage input.  The conservative identity key deliberately excludes
the seed: a candidate with a new seed but the same suite/task/episode is still
treated as historically used.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickletools
import re
import stat
from typing import Any, Iterable, Mapping, Sequence
import zipfile


V3_SELECTION_SCHEMA_VERSION = "phase-route-vla.v3.data-lineage-selection.v1"
V3_JSONL_ROW_SCHEMA_VERSION = "phase-route-vla.v3.data-lineage-row.v1"
FROZEN_METADATA_INDEX_SCHEMA_VERSION = (
    "phase-route-vla.frozen-five-field-metadata-selection.v1"
)
AUDIT_REQUEST_SCHEMA_VERSION = "phase-route-vla.v3.data-lineage-audit-request.v1"
AUDIT_REPORT_SCHEMA_VERSION = "phase-route-vla.v3.data-lineage-audit.v1"

LIBERO_LONG_SUITE = "libero_10"
V3_CANDIDATE_ROLES = (
    "development_v2",
    "calibration_v2",
    "independent_test_v2",
)
V3_EPISODES_BY_ROLE = {
    "development_v2": tuple(range(12, 30)),
    "calibration_v2": tuple(range(30, 40)),
    "independent_test_v2": tuple(range(40, 50)),
}
LIBERO_LONG_TASK_DIMENSIONS = {
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 123,
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 123,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": 47,
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 51,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 84,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": 45,
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 71,
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 84,
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 47,
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": 47,
}

_ALLOWED_SUFFIXES = frozenset({".json", ".jsonl"})
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".bin",
        ".npy",
        ".npz",
        ".init",
        ".pkl",
        ".pickle",
    }
)
_FROZEN_METADATA_FIELDS = frozenset(
    {
        "dataset_index",
        "task_id",
        "episode_index",
        "call_ordinal",
        "shard_assignment",
    }
)
_V3_DOCUMENT_FIELDS = frozenset(
    {"schema_version", "suite", "role", "seed", "records"}
)
_V3_DOCUMENT_REQUIRED_FIELDS = frozenset(
    {"schema_version", "suite", "role", "records"}
)
_V3_RECORD_FIELDS = frozenset(
    {"task_id", "episode_index", "state_index", "seed"}
)
_V3_JSONL_FIELDS = frozenset(
    {
        "schema_version",
        "suite",
        "role",
        "task_id",
        "episode_index",
        "state_index",
        "seed",
    }
)
_SOURCE_SPEC_FIELDS = frozenset(
    {"path", "schema_version", "suite", "role", "seed"}
)
_SOURCE_SPEC_REQUIRED_FIELDS = frozenset({"path", "schema_version"})
_REQUEST_FIELDS = frozenset(
    {"schema_version", "used_sources", "candidate_sources"}
)
_REQUEST_REQUIRED_FIELDS = _REQUEST_FIELDS
_SUITE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_ROLE_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_C361_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:stage[^a-z0-9]*)?(?:c361|c3[._-]?61)(?:[^0-9]|$)"
)
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "action",
        "actions",
        "action_chunk",
        "arrays",
        "arrays_path",
        "feature",
        "features",
        "hidden_state",
        "image",
        "images",
        "logits",
        "observation",
        "observations",
        "payload",
        "prediction",
        "predictions",
        "target",
        "targets",
        "tensor",
        "tensors",
    }
)
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 64 * 1024
_MAX_JSONL_ROWS = 1_000_000
_MAX_INIT_ARCHIVE_BYTES = 2 * 1024 * 1024
_MAX_TASK_MAP_BYTES = 2 * 1024 * 1024
_TEXT_SCAN_SUFFIXES = frozenset(
    {
        ".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml",
        ".py", ".sh", ".toml", ".csv", ".xml", ".rst", ".cfg", ".ini",
    }
)
_TEXT_SCAN_IGNORED_DIRECTORIES = frozenset(
    {".git", ".pytest_cache", "__pycache__"}
)
_INIT_PICKLE_OPCODE_NAMES = (
    "PROTO",
    "GLOBAL",
    "BINPUT",
    "GLOBAL",
    "BINPUT",
    "BININT1",
    "TUPLE1",
    "BINPUT",
    "GLOBAL",
    "BINPUT",
    "BINUNICODE",
    "BINPUT",
    "BINUNICODE",
    "BINPUT",
    "TUPLE2",
    "BINPUT",
    "REDUCE",
    "BINPUT",
    "TUPLE3",
    "BINPUT",
    "REDUCE",
    "BINPUT",
    "MARK",
    "BININT1",
    "BININT1",
    "BININT1",
    "TUPLE2",
    "BINPUT",
    "GLOBAL",
    "BINPUT",
    "BINUNICODE",
    "BINPUT",
    "NEWFALSE",
    "NEWTRUE",
    "TUPLE3",
    "BINPUT",
    "REDUCE",
    "BINPUT",
    "MARK",
    "BININT1",
    "BINUNICODE",
    "BINPUT",
    "NONE",
    "NONE",
    "NONE",
    "BININT",
    "BININT",
    "BININT1",
    "TUPLE",
    "BINPUT",
    "BUILD",
    "NEWFALSE",
    "BINGET",
    "BINUNICODE",
    "BINPUT",
    "BINGET",
    "TUPLE2",
    "BINPUT",
    "REDUCE",
    "BINPUT",
    "TUPLE",
    "BINPUT",
    "BUILD",
    "STOP",
)
_INIT_PICKLE_GLOBALS = (
    "numpy.core.multiarray _reconstruct",
    "numpy ndarray",
    "_codecs encode",
    "numpy dtype",
)


class DataLineageError(ValueError):
    """Base class for fail-closed lineage contract failures."""


class MetadataPathError(DataLineageError):
    """Raised when an input path violates the metadata-only path policy."""


class MetadataSchemaError(DataLineageError):
    """Raised when metadata does not match an explicitly supported schema."""


class ForbiddenLineageSourceError(DataLineageError):
    """Raised for row-level records, payloads, or binary artifacts."""


class RoleConflictError(DataLineageError):
    """Raised when one canonical episode key appears in multiple used roles."""

    def __init__(self, report: "DataLineageReport") -> None:
        self.report = report
        keys = ", ".join(item["key"] for item in report.role_conflicts[:5])
        suffix = "" if len(report.role_conflicts) <= 5 else ", ..."
        super().__init__(f"used-role identity overlap is forbidden: {keys}{suffix}")


class DuplicateIdentityError(DataLineageError):
    """Raised when an episode-selection contract repeats an exact key."""


class CandidateOverlapError(DataLineageError):
    """Raised when a proposed candidate is already present in evidence."""

    def __init__(self, report: "DataLineageReport") -> None:
        self.report = report
        keys = ", ".join(
            item["key"] for item in report.candidate_known_used_keys[:5]
        )
        suffix = "" if len(report.candidate_known_used_keys) <= 5 else ", ..."
        super().__init__(f"candidate keys intersect historical use: {keys}{suffix}")


class AvailabilityAuditError(DataLineageError):
    """Raised when LIBERO-Long availability cannot be proven statically."""


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetadataSchemaError(f"{field} must be a non-negative JSON integer")
    return value


def _optional_seed(value: Any, field: str = "seed") -> int | None:
    if value is None:
        return None
    return _strict_int(value, field)


def _strict_string(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise MetadataSchemaError(f"{field} has an invalid canonical string")
    return value


def _require_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    context: str,
) -> None:
    keys = frozenset(value)
    unknown = sorted(keys - allowed)
    missing = sorted(required - keys)
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append(f"unknown fields={unknown}")
        if missing:
            details.append(f"missing fields={missing}")
        raise MetadataSchemaError(f"{context} schema mismatch: {'; '.join(details)}")


def _reject_forbidden_payload_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if key in _FORBIDDEN_PAYLOAD_KEYS)
        if forbidden:
            raise ForbiddenLineageSourceError(
                f"{context} contains forbidden row/payload fields: {forbidden}"
            )
        for key, nested in value.items():
            _reject_forbidden_payload_keys(nested, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_payload_keys(nested, context=f"{context}[{index}]")


def _contains_c361_marker(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_C361_RE.search(value.lower()))
    if isinstance(value, Mapping):
        return any(
            _contains_c361_marker(key) or _contains_c361_marker(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_c361_marker(item) for item in value)
    return False


def _reject_c361(value: Any, *, context: str) -> None:
    if _contains_c361_marker(value):
        raise ForbiddenLineageSourceError(
            f"{context} references stage C3.61; row-level C3.61 records/payload "
            "are forbidden lineage inputs"
        )


@dataclass(frozen=True, order=True)
class EpisodeIdentity:
    """Conservative episode identity used for historical overlap checks."""

    suite: str
    task_id: int
    episode_index: int

    def __post_init__(self) -> None:
        _strict_string(self.suite, "suite", _SUITE_RE)
        _strict_int(self.task_id, "task_id")
        _strict_int(self.episode_index, "episode_index")

    @property
    def key(self) -> str:
        """Return the exact, non-seed-narrowed canonical identity key."""

        return f"{self.suite}:task{self.task_id}:episode{self.episode_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "suite": self.suite,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
        }


@dataclass(frozen=True)
class LineageRecord:
    identity: EpisodeIdentity
    role: str
    seed: int | None
    source: str

    def __post_init__(self) -> None:
        _strict_string(self.role, "role", _ROLE_RE)
        _optional_seed(self.seed)
        if not self.source:
            raise MetadataSchemaError("source label must not be empty")


@dataclass(frozen=True)
class MetadataSourceSpec:
    """A schema-bound metadata input beneath a trusted metadata root."""

    path: str
    schema_version: str
    suite: str | None = None
    role: str | None = None
    seed: int | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, context: str) -> "MetadataSourceSpec":
        if not isinstance(value, Mapping):
            raise MetadataSchemaError(f"{context} must be a JSON object")
        _require_fields(
            value,
            allowed=_SOURCE_SPEC_FIELDS,
            required=_SOURCE_SPEC_REQUIRED_FIELDS,
            context=context,
        )
        path = value["path"]
        schema = value["schema_version"]
        if not isinstance(path, str) or not path:
            raise MetadataSchemaError(f"{context}.path must be a non-empty string")
        if not isinstance(schema, str) or not schema:
            raise MetadataSchemaError(
                f"{context}.schema_version must be a non-empty string"
            )
        suite = value.get("suite")
        role = value.get("role")
        if suite is not None:
            suite = _strict_string(suite, f"{context}.suite", _SUITE_RE)
        if role is not None:
            role = _strict_string(role, f"{context}.role", _ROLE_RE)
        seed = _optional_seed(value.get("seed"), f"{context}.seed")
        return cls(path=path, schema_version=schema, suite=suite, role=role, seed=seed)


@dataclass(frozen=True)
class AuditRequest:
    used_sources: tuple[MetadataSourceSpec, ...]
    candidate_sources: tuple[MetadataSourceSpec, ...]


@dataclass(frozen=True)
class LoadedSelection:
    spec: MetadataSourceSpec
    records: tuple[LineageRecord, ...]
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class DataLineageReport:
    status: str
    used_record_count: int
    candidate_record_count: int
    used_key_inventory: tuple[dict[str, Any], ...]
    role_conflicts: tuple[dict[str, Any], ...]
    candidate_known_used_keys: tuple[dict[str, Any], ...]
    candidate_no_known_hit_keys: tuple[dict[str, Any], ...]
    source_inventory: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return self.status == "PASS_NO_KNOWN_HIT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
            "status": self.status,
            "metadata_only": True,
            "canonical_key_schema": "{suite}:task{task_id}:episode{episode_index}",
            "seed_narrows_overlap": False,
            "runtime_control_authorized": False,
            "counts": {
                "used_records": self.used_record_count,
                "used_unique_keys": len(self.used_key_inventory),
                "role_conflicts": len(self.role_conflicts),
                "candidate_records": self.candidate_record_count,
                "candidate_known_used_keys": len(self.candidate_known_used_keys),
                "candidate_no_known_hit_keys": len(
                    self.candidate_no_known_hit_keys
                ),
            },
            "used_key_inventory": list(self.used_key_inventory),
            "role_conflicts": list(self.role_conflicts),
            "candidate_history_intersection": list(
                self.candidate_known_used_keys
            ),
            "candidate_known_used_keys": list(self.candidate_known_used_keys),
            "candidate_no_known_hit_keys": list(
                self.candidate_no_known_hit_keys
            ),
            "no_known_hit_authorized": self.passed,
            "source_inventory": list(self.source_inventory),
        }


@dataclass(frozen=True)
class InitArchiveEvidence:
    task_id: int
    task_name: str
    relative_path: str
    sha256: str
    shape: tuple[int, int]
    dtype: str
    raw_bytes: int
    pickle_opcode_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "raw_bytes": self.raw_bytes,
            "pickle_opcode_count": self.pickle_opcode_count,
        }


@dataclass(frozen=True)
class LiberoLongAvailabilityAudit:
    task_map_sha256: str
    init_root: str
    tasks: tuple[InitArchiveEvidence, ...]

    @property
    def passed(self) -> bool:
        return (
            len(self.tasks) == 10
            and [item.task_id for item in self.tasks] == list(range(10))
            and all(item.shape[0] == 50 for item in self.tasks)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS_STATIC_LIBERO_LONG_AVAILABILITY"
            if self.passed
            else "FAIL_STATIC_LIBERO_LONG_AVAILABILITY",
            "suite": LIBERO_LONG_SUITE,
            "metadata_only": True,
            "deserialization_used": False,
            "task_map_sha256": self.task_map_sha256,
            "init_root": self.init_root,
            "task_count": len(self.tasks),
            "minimum_initial_states": min(
                (item.shape[0] for item in self.tasks), default=0
            ),
            "tasks": [item.to_dict() for item in self.tasks],
        }


@dataclass(frozen=True)
class TextCorpusScan:
    hits: tuple[dict[str, str], ...]
    skipped: tuple[dict[str, str], ...]
    scanned_file_count: int
    scanned_bytes: int

    @property
    def passed(self) -> bool:
        return not self.hits

    def to_dict(self) -> dict[str, Any]:
        row_payloads = [
            item
            for item in self.skipped
            if item["reason"]
            == "row-level/C3.61 payload forbidden by D0 boundary"
        ]
        symlinks = [
            item
            for item in self.skipped
            if item["reason"]
            in {"symlink file not opened", "symlink directory not followed"}
        ]
        unclassified = [
            item for item in self.skipped if item not in row_payloads and item not in symlinks
        ]
        return {
            "hit_count": len(self.hits),
            "hits": list(self.hits),
            "scanned_file_count": self.scanned_file_count,
            "scanned_bytes": self.scanned_bytes,
            "skipped_row_payload_count": len(row_payloads),
            "skipped_row_payloads": row_payloads,
            "skipped_symlink_count": len(symlinks),
            "skipped_symlinks": symlinks,
            "unclassified_skip_count": len(unclassified),
            "unclassified_skips": unclassified,
            "skips_are_explicit": True,
        }


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetadataSchemaError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise MetadataSchemaError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(text: str, *, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except DataLineageError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise MetadataSchemaError(f"invalid JSON in {context}: {error}") from error


def _reject_source_path_markers(relative_path: Path) -> None:
    lower_parts = [part.lower() for part in relative_path.parts]
    if any(_C361_RE.search(part) for part in lower_parts):
        raise ForbiddenLineageSourceError(
            "stage C3.61 paths are forbidden; use only a pre-frozen protocol "
            "or metadata selection"
        )
    basename = relative_path.name.lower()
    if basename == "records.jsonl" or basename.endswith("_records.jsonl"):
        raise ForbiddenLineageSourceError(
            "row-level records JSONL is not a metadata-selection input"
        )


def resolve_metadata_path(metadata_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a regular JSON/JSONL file without traversal or symlinks."""

    root_input = Path(metadata_root)
    _reject_symlink_components(root_input, context="metadata root")
    try:
        root = root_input.resolve(strict=True)
    except FileNotFoundError as error:
        raise MetadataPathError("metadata root does not exist") from error
    if not root.is_dir():
        raise MetadataPathError("metadata root must be a directory")

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts:
        raise MetadataPathError("metadata path must be non-empty and relative")
    if any(part in {"..", ""} for part in relative.parts):
        raise MetadataPathError("metadata path traversal is forbidden")
    _reject_source_path_markers(relative)

    suffix = relative.suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES:
        raise ForbiddenLineageSourceError(
            f"binary/model suffix is forbidden for metadata lineage: {suffix}"
        )
    if suffix not in _ALLOWED_SUFFIXES:
        raise MetadataPathError(
            f"only JSON/JSONL metadata inputs are accepted, got suffix {suffix!r}"
        )

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise MetadataPathError("symlink metadata paths are forbidden")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as error:
        raise MetadataPathError(f"metadata input does not exist: {relative}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise MetadataPathError("metadata path escapes its trusted root") from error
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise MetadataPathError("metadata input must be a regular file")
    return resolved


def _reject_symlink_components(path: str | Path, *, context: str) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise MetadataPathError(f"{context} contains a symlink component")


def _read_metadata_bytes(path: Path) -> bytes:
    expected = path.stat()
    size = expected.st_size
    if size > _MAX_METADATA_BYTES:
        raise ForbiddenLineageSourceError(
            f"metadata input exceeds {_MAX_METADATA_BYTES} bytes"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise MetadataPathError("metadata input changed identity during open")
        if not stat.S_ISREG(opened.st_mode):
            raise MetadataPathError("metadata input must remain a regular file")
        if opened.st_size > _MAX_METADATA_BYTES:
            raise ForbiddenLineageSourceError(
                f"metadata input exceeds {_MAX_METADATA_BYTES} bytes"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_METADATA_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_METADATA_BYTES:
                raise ForbiddenLineageSourceError(
                    f"metadata input exceeds {_MAX_METADATA_BYTES} bytes"
                )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _utf8_text(raw: bytes, *, context: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MetadataSchemaError(f"{context} is not UTF-8 JSON text") from error


def _episode_index_from_record(record: Mapping[str, Any], *, context: str) -> int:
    has_episode = "episode_index" in record
    has_state = "state_index" in record
    if has_episode == has_state:
        raise MetadataSchemaError(
            f"{context} must contain exactly one of episode_index/state_index"
        )
    field = "episode_index" if has_episode else "state_index"
    return _strict_int(record[field], f"{context}.{field}")


def _assert_unique_records(
    records: Sequence[LineageRecord], *, context: str
) -> None:
    seen: set[EpisodeIdentity] = set()
    for record in records:
        if record.identity in seen:
            raise DuplicateIdentityError(
                f"{context} repeats exact key {record.identity.key}"
            )
        seen.add(record.identity)


def _validate_descriptor_expectation(
    *,
    actual: Any,
    expected: Any,
    field: str,
    source: str,
) -> None:
    if expected is not None and actual != expected:
        raise MetadataSchemaError(
            f"{source} declares {field}={actual!r}, expected {expected!r}"
        )


def _parse_v3_document(
    value: Any, *, spec: MetadataSourceSpec, source: str
) -> tuple[LineageRecord, ...]:
    if not isinstance(value, Mapping):
        raise MetadataSchemaError(f"{source} must contain one JSON object")
    _reject_c361(value, context=source)
    _reject_forbidden_payload_keys(value, context=source)
    _require_fields(
        value,
        allowed=_V3_DOCUMENT_FIELDS,
        required=_V3_DOCUMENT_REQUIRED_FIELDS,
        context=source,
    )
    if value["schema_version"] != V3_SELECTION_SCHEMA_VERSION:
        raise MetadataSchemaError(f"unsupported JSON selection schema in {source}")
    suite = _strict_string(value["suite"], f"{source}.suite", _SUITE_RE)
    role = _strict_string(value["role"], f"{source}.role", _ROLE_RE)
    document_seed = _optional_seed(value.get("seed"), f"{source}.seed")
    _validate_descriptor_expectation(
        actual=suite, expected=spec.suite, field="suite", source=source
    )
    _validate_descriptor_expectation(
        actual=role, expected=spec.role, field="role", source=source
    )
    _validate_descriptor_expectation(
        actual=document_seed, expected=spec.seed, field="seed", source=source
    )
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise MetadataSchemaError(f"{source}.records must be a non-empty array")
    parsed: list[LineageRecord] = []
    for index, raw_record in enumerate(records):
        context = f"{source}.records[{index}]"
        if not isinstance(raw_record, Mapping):
            raise MetadataSchemaError(f"{context} must be an object")
        _require_fields(
            raw_record,
            allowed=_V3_RECORD_FIELDS,
            required=frozenset({"task_id"}),
            context=context,
        )
        episode_index = _episode_index_from_record(raw_record, context=context)
        row_seed = _optional_seed(raw_record.get("seed"), f"{context}.seed")
        if "seed" in raw_record and document_seed is not None and row_seed != document_seed:
            raise MetadataSchemaError(
                f"{context}.seed conflicts with document seed {document_seed}"
            )
        parsed.append(
            LineageRecord(
                identity=EpisodeIdentity(
                    suite=suite,
                    task_id=_strict_int(raw_record["task_id"], f"{context}.task_id"),
                    episode_index=episode_index,
                ),
                role=role,
                seed=row_seed if "seed" in raw_record else document_seed,
                source=source,
            )
        )
    result = tuple(parsed)
    _assert_unique_records(result, context=source)
    return result


def _jsonl_values(raw: bytes, *, source: str) -> tuple[Mapping[str, Any], ...]:
    text = _utf8_text(raw, context=source)
    if not text:
        raise MetadataSchemaError(f"{source} is empty")
    lines = text.splitlines()
    if len(lines) > _MAX_JSONL_ROWS:
        raise ForbiddenLineageSourceError(
            f"{source} exceeds the metadata row limit {_MAX_JSONL_ROWS}"
        )
    parsed: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise MetadataSchemaError(
                f"{source}:{line_number} blank JSONL lines are forbidden"
            )
        if len(line.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
            raise ForbiddenLineageSourceError(
                f"{source}:{line_number} exceeds the metadata-line byte limit"
            )
        value = _decode_json(line, context=f"{source}:{line_number}")
        if not isinstance(value, Mapping):
            raise MetadataSchemaError(
                f"{source}:{line_number} must be one JSON object"
            )
        _reject_c361(value, context=f"{source}:{line_number}")
        _reject_forbidden_payload_keys(value, context=f"{source}:{line_number}")
        parsed.append(value)
    if not parsed:
        raise MetadataSchemaError(f"{source} contains no metadata records")
    return tuple(parsed)


def _parse_v3_jsonl(
    rows: Sequence[Mapping[str, Any]], *, spec: MetadataSourceSpec, source: str
) -> tuple[LineageRecord, ...]:
    parsed: list[LineageRecord] = []
    for index, row in enumerate(rows, start=1):
        context = f"{source}:{index}"
        _require_fields(
            row,
            allowed=_V3_JSONL_FIELDS,
            required=frozenset({"schema_version", "suite", "role", "task_id"}),
            context=context,
        )
        if row["schema_version"] != V3_JSONL_ROW_SCHEMA_VERSION:
            raise MetadataSchemaError(f"unsupported V3 JSONL schema in {context}")
        suite = _strict_string(row["suite"], f"{context}.suite", _SUITE_RE)
        role = _strict_string(row["role"], f"{context}.role", _ROLE_RE)
        seed = _optional_seed(row.get("seed"), f"{context}.seed")
        _validate_descriptor_expectation(
            actual=suite, expected=spec.suite, field="suite", source=context
        )
        _validate_descriptor_expectation(
            actual=role, expected=spec.role, field="role", source=context
        )
        _validate_descriptor_expectation(
            actual=seed, expected=spec.seed, field="seed", source=context
        )
        parsed.append(
            LineageRecord(
                identity=EpisodeIdentity(
                    suite=suite,
                    task_id=_strict_int(row["task_id"], f"{context}.task_id"),
                    episode_index=_episode_index_from_record(row, context=context),
                ),
                role=role,
                seed=seed,
                source=source,
            )
        )
    result = tuple(parsed)
    _assert_unique_records(result, context=source)
    return result


def _parse_frozen_metadata_jsonl(
    rows: Sequence[Mapping[str, Any]], *, spec: MetadataSourceSpec, source: str
) -> tuple[LineageRecord, ...]:
    if spec.suite is None or spec.role is None:
        raise MetadataSchemaError(
            f"{source} frozen metadata requires explicit descriptor suite and role"
        )
    parsed_by_identity: dict[EpisodeIdentity, LineageRecord] = {}
    row_identities: set[tuple[int, int, int, int, int]] = set()
    for index, row in enumerate(rows, start=1):
        context = f"{source}:{index}"
        _require_fields(
            row,
            allowed=_FROZEN_METADATA_FIELDS,
            required=_FROZEN_METADATA_FIELDS,
            context=context,
        )
        for field in _FROZEN_METADATA_FIELDS:
            _strict_int(row[field], f"{context}.{field}")
        row_identity = tuple(row[field] for field in sorted(_FROZEN_METADATA_FIELDS))
        if row_identity in row_identities:
            raise DuplicateIdentityError(f"{context} repeats a frozen metadata row")
        row_identities.add(row_identity)
        identity = EpisodeIdentity(
            suite=spec.suite,
            task_id=row["task_id"],
            episode_index=row["episode_index"],
        )
        parsed_by_identity.setdefault(
            identity,
            LineageRecord(
                identity=identity,
                role=spec.role,
                seed=spec.seed,
                source=source,
            ),
        )
    return tuple(parsed_by_identity.values())


def load_metadata_selection(
    metadata_root: str | Path, spec: MetadataSourceSpec
) -> LoadedSelection:
    """Load one explicitly schema-bound JSON/JSONL metadata selection."""

    _reject_c361(spec.__dict__, context=f"source descriptor {spec.path}")
    path = resolve_metadata_path(metadata_root, spec.path)
    raw = _read_metadata_bytes(path)
    source = Path(spec.path).as_posix()

    if spec.schema_version == V3_SELECTION_SCHEMA_VERSION:
        if path.suffix.lower() != ".json":
            raise MetadataSchemaError("V3 selection documents must use .json")
        value = _decode_json(_utf8_text(raw, context=source), context=source)
        records = _parse_v3_document(value, spec=spec, source=source)
    elif spec.schema_version == V3_JSONL_ROW_SCHEMA_VERSION:
        if path.suffix.lower() != ".jsonl":
            raise MetadataSchemaError("V3 row selections must use .jsonl")
        records = _parse_v3_jsonl(
            _jsonl_values(raw, source=source), spec=spec, source=source
        )
    elif spec.schema_version == FROZEN_METADATA_INDEX_SCHEMA_VERSION:
        if path.suffix.lower() != ".jsonl":
            raise MetadataSchemaError("frozen metadata indexes must use .jsonl")
        records = _parse_frozen_metadata_jsonl(
            _jsonl_values(raw, source=source), spec=spec, source=source
        )
    else:
        raise MetadataSchemaError(
            f"unsupported metadata source schema: {spec.schema_version!r}"
        )

    return LoadedSelection(
        spec=spec,
        records=records,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
    )


def load_audit_request(
    metadata_root: str | Path, relative_request_path: str | Path
) -> AuditRequest:
    """Load the strict audit request manifest beneath ``metadata_root``."""

    path = resolve_metadata_path(metadata_root, relative_request_path)
    if path.suffix.lower() != ".json":
        raise MetadataSchemaError("audit request must use .json")
    source = Path(relative_request_path).as_posix()
    raw = _read_metadata_bytes(path)
    value = _decode_json(_utf8_text(raw, context=source), context=source)
    if not isinstance(value, Mapping):
        raise MetadataSchemaError("audit request must be one JSON object")
    _reject_c361(value, context=source)
    _reject_forbidden_payload_keys(value, context=source)
    _require_fields(
        value,
        allowed=_REQUEST_FIELDS,
        required=_REQUEST_REQUIRED_FIELDS,
        context=source,
    )
    if value["schema_version"] != AUDIT_REQUEST_SCHEMA_VERSION:
        raise MetadataSchemaError("unsupported audit-request schema")

    def parse_specs(field: str) -> tuple[MetadataSourceSpec, ...]:
        raw_specs = value[field]
        if not isinstance(raw_specs, list) or not raw_specs:
            raise MetadataSchemaError(f"{source}.{field} must be a non-empty array")
        return tuple(
            MetadataSourceSpec.from_mapping(item, context=f"{source}.{field}[{index}]")
            for index, item in enumerate(raw_specs)
        )

    request = AuditRequest(
        used_sources=parse_specs("used_sources"),
        candidate_sources=parse_specs("candidate_sources"),
    )
    normalized_paths = [
        Path(spec.path).as_posix()
        for spec in (*request.used_sources, *request.candidate_sources)
    ]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise MetadataSchemaError("audit request repeats a source path")
    return request


def _seed_summary(records: Iterable[LineageRecord]) -> tuple[list[int], bool]:
    values: set[int] = set()
    unknown = False
    for record in records:
        if record.seed is None:
            unknown = True
        else:
            values.add(record.seed)
    return sorted(values), unknown


def _inventory_entry(
    identity: EpisodeIdentity, records: Sequence[LineageRecord]
) -> dict[str, Any]:
    seeds, seed_unknown = _seed_summary(records)
    return {
        **identity.to_dict(),
        "roles": sorted({record.role for record in records}),
        "seed_values": seeds,
        "seed_unknown": seed_unknown,
        "sources": sorted({record.source for record in records}),
        "occurrence_count": len(records),
    }


def build_data_lineage_report(
    used: Sequence[LoadedSelection], candidates: Sequence[LoadedSelection]
) -> DataLineageReport:
    """Build a deterministic diagnostic report without authorizing runtime use."""

    if not used:
        raise MetadataSchemaError("at least one used metadata selection is required")
    if not candidates:
        raise MetadataSchemaError("at least one candidate metadata selection is required")
    used_records = tuple(record for item in used for record in item.records)
    candidate_records = tuple(record for item in candidates for record in item.records)
    if not used_records or not candidate_records:
        raise MetadataSchemaError("used and candidate selections must contain records")

    candidate_roles = {record.role for record in candidate_records}
    if candidate_roles != set(V3_CANDIDATE_ROLES):
        raise MetadataSchemaError(
            "candidate sources must declare exactly development_v2, "
            "calibration_v2, and independent_test_v2"
        )
    validate_libero_long_candidate_split(candidate_records)

    used_by_identity: dict[EpisodeIdentity, list[LineageRecord]] = defaultdict(list)
    for record in used_records:
        if record.role in V3_CANDIDATE_ROLES:
            raise MetadataSchemaError(
                f"used source {record.source} uses a reserved V3 candidate role"
            )
        used_by_identity[record.identity].append(record)

    same_role_seen: set[tuple[EpisodeIdentity, str]] = set()
    for record in used_records:
        marker = (record.identity, record.role)
        if marker in same_role_seen:
            raise DuplicateIdentityError(
                f"used selections repeat {record.identity.key} in role {record.role}"
            )
        same_role_seen.add(marker)

    candidate_by_identity: dict[EpisodeIdentity, list[LineageRecord]] = defaultdict(list)
    for record in candidate_records:
        if record.identity in candidate_by_identity:
            previous = candidate_by_identity[record.identity][0]
            raise DuplicateIdentityError(
                "candidate role selections overlap at exact key "
                f"{record.identity.key} ({previous.role}, {record.role})"
            )
        candidate_by_identity[record.identity].append(record)

    used_inventory = tuple(
        _inventory_entry(identity, used_by_identity[identity])
        for identity in sorted(used_by_identity)
    )
    role_conflicts = tuple(
        entry for entry in used_inventory if len(entry["roles"]) > 1
    )

    known_used: list[dict[str, Any]] = []
    no_known_hit: list[dict[str, Any]] = []
    for identity in sorted(candidate_by_identity):
        candidate_group = candidate_by_identity[identity]
        candidate_seeds, seed_unknown = _seed_summary(candidate_group)
        base = {
            **identity.to_dict(),
            "candidate_seed_values": candidate_seeds,
            "candidate_seed_unknown": seed_unknown,
            "candidate_sources": sorted({record.source for record in candidate_group}),
            "candidate_occurrence_count": len(candidate_group),
        }
        if identity in used_by_identity:
            history = used_by_identity[identity]
            history_seeds, history_seed_unknown = _seed_summary(history)
            known_used.append(
                {
                    **base,
                    "used_roles": sorted({record.role for record in history}),
                    "used_seed_values": history_seeds,
                    "used_seed_unknown": history_seed_unknown,
                    "used_sources": sorted({record.source for record in history}),
                }
            )
        else:
            no_known_hit.append(base)

    source_inventory = tuple(
        {
            "path": item.spec.path,
            "schema_version": item.spec.schema_version,
            "sha256": item.sha256,
            "bytes": item.byte_count,
            "records": len(item.records),
            "purpose": purpose,
        }
        for purpose, selections in (("used", used), ("candidate", candidates))
        for item in selections
    )
    if role_conflicts:
        status = "FAIL_USED_ROLE_CONFLICT"
    elif known_used:
        status = "FAIL_CANDIDATE_HISTORY_INTERSECTION"
    else:
        status = "PASS_NO_KNOWN_HIT"
    return DataLineageReport(
        status=status,
        used_record_count=len(used_records),
        candidate_record_count=len(candidate_records),
        used_key_inventory=used_inventory,
        role_conflicts=role_conflicts,
        candidate_known_used_keys=tuple(known_used),
        candidate_no_known_hit_keys=tuple(no_known_hit),
        source_inventory=source_inventory,
    )


def audit_data_lineage(
    metadata_root: str | Path,
    request: AuditRequest,
    *,
    raise_on_role_conflict: bool = True,
    raise_on_candidate_overlap: bool = True,
) -> DataLineageReport:
    """Load and audit a request, failing closed on historical role overlap."""

    used = tuple(
        load_metadata_selection(metadata_root, spec) for spec in request.used_sources
    )
    candidates = tuple(
        load_metadata_selection(metadata_root, spec)
        for spec in request.candidate_sources
    )
    report = build_data_lineage_report(used, candidates)
    if report.role_conflicts and raise_on_role_conflict:
        raise RoleConflictError(report)
    if report.candidate_known_used_keys and raise_on_candidate_overlap:
        raise CandidateOverlapError(report)
    return report


def _regular_file_without_symlink(path: str | Path, *, suffix: str) -> Path:
    raw = Path(path).absolute()
    if raw.suffix.lower() != suffix:
        raise AvailabilityAuditError(f"expected {suffix} file: {raw}")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise AvailabilityAuditError(f"symlink evidence path is forbidden: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise AvailabilityAuditError(f"evidence path is missing: {raw}") from error
    if not resolved.is_file():
        raise AvailabilityAuditError(f"evidence path is not a regular file: {raw}")
    return resolved


def _parse_libero_long_task_map(path: Path) -> tuple[str, ...]:
    raw = _read_bounded_regular_file(path, maximum=_MAX_TASK_MAP_BYTES)
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise AvailabilityAuditError("LIBERO task map is not valid UTF-8 Python") from error
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise AvailabilityAuditError("expected exactly one libero_task_map assignment")
    try:
        mapping = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise AvailabilityAuditError("libero_task_map must be literal-only") from error
    if not isinstance(mapping, dict) or set(mapping).isdisjoint({LIBERO_LONG_SUITE}):
        raise AvailabilityAuditError("task map does not contain libero_10")
    tasks = mapping[LIBERO_LONG_SUITE]
    if (
        not isinstance(tasks, list)
        or any(not isinstance(item, str) for item in tasks)
        or tuple(tasks) != tuple(LIBERO_LONG_TASK_DIMENSIONS)
    ):
        raise AvailabilityAuditError(
            "LIBERO-Long task order/names differ from the frozen ten-task contract"
        )
    return tuple(tasks)


def _read_bounded_regular_file(path: Path, *, maximum: int) -> bytes:
    expected = path.stat()
    if expected.st_size > maximum:
        raise AvailabilityAuditError(f"static evidence exceeds {maximum} bytes: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
            raise AvailabilityAuditError("static evidence changed identity during open")
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise AvailabilityAuditError("static evidence changed during audit")
        data = bytearray()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > maximum:
                raise AvailabilityAuditError("static evidence exceeded size bound")
        return bytes(data)
    finally:
        os.close(descriptor)


def audit_libero_init_archive(
    archive_path: str | Path,
    *,
    task_id: int,
    task_name: str,
    expected_dimension: int,
    relative_path: str | None = None,
) -> InitArchiveEvidence:
    """Statically verify one NumPy protocol-2 init archive without unpickling."""

    path = _regular_file_without_symlink(archive_path, suffix=".pruned_init")
    raw_archive = _read_bounded_regular_file(path, maximum=_MAX_INIT_ARCHIVE_BYTES)
    try:
        with zipfile.ZipFile(io.BytesIO(raw_archive), mode="r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if names != ["archive/data.pkl", "archive/version"]:
                raise AvailabilityAuditError(
                    "init ZIP members must be exactly archive/data.pkl and archive/version"
                )
            if len(names) != len(set(names)) or archive.comment:
                raise AvailabilityAuditError("duplicate ZIP members/comments are forbidden")
            for item in infos:
                if item.flag_bits & 0x1:
                    raise AvailabilityAuditError("encrypted init ZIP members are forbidden")
                if item.flag_bits & ~(0x8 | 0x800):
                    raise AvailabilityAuditError("unknown init ZIP flags are forbidden")
                if item.compress_type != zipfile.ZIP_STORED:
                    raise AvailabilityAuditError("init ZIP members must be stored")
                if item.file_size > _MAX_INIT_ARCHIVE_BYTES:
                    raise AvailabilityAuditError("init ZIP member exceeds size bound")
                mode = item.external_attr >> 16
                if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise AvailabilityAuditError("non-regular init ZIP member is forbidden")
            if archive.read("archive/version") != b"3\n":
                raise AvailabilityAuditError("init archive version must be exactly 3")
            pickle_data = archive.read("archive/data.pkl")
    except (zipfile.BadZipFile, OSError, RuntimeError) as error:
        raise AvailabilityAuditError(f"invalid init ZIP archive: {path}") from error

    try:
        operations = []
        for operation in pickletools.genops(pickle_data):
            if len(operations) >= len(_INIT_PICKLE_OPCODE_NAMES):
                raise AvailabilityAuditError(
                    "init pickle exceeds the fixed opcode-count contract"
                )
            operations.append(operation)
    except AvailabilityAuditError:
        raise
    except (ValueError, UnicodeError) as error:
        raise AvailabilityAuditError("invalid static pickle opcode stream") from error
    names = tuple(op.name for op, _, _ in operations)
    if names != _INIT_PICKLE_OPCODE_NAMES:
        raise AvailabilityAuditError("init pickle opcode sequence differs from whitelist")
    arguments = [argument for _, argument, _ in operations]
    globals_seen = tuple(
        argument
        for (opcode, argument, _) in operations
        if opcode.name == "GLOBAL"
    )
    if globals_seen != _INIT_PICKLE_GLOBALS:
        raise AvailabilityAuditError("init pickle GLOBAL allowlist differs")
    if [argument for opcode, argument, _ in operations if opcode.name == "BINPUT"] != list(
        range(21)
    ):
        raise AvailabilityAuditError("init pickle memo layout differs")
    exact_arguments = {
        0: 2,
        5: 0,
        10: "b",
        12: "latin1",
        23: 1,
        24: 50,
        25: expected_dimension,
        30: "f8",
        39: 3,
        40: "<",
        45: -1,
        46: -1,
        47: 0,
        52: 3,
        55: 5,
    }
    if any(arguments[index] != expected for index, expected in exact_arguments.items()):
        raise AvailabilityAuditError("init pickle scalar/dtype/shape contract differs")
    if operations[-1][2] + 1 != len(pickle_data):
        raise AvailabilityAuditError("trailing bytes after init pickle STOP are forbidden")
    payload = arguments[53]
    if not isinstance(payload, str):
        raise AvailabilityAuditError("init pickle raw payload must be BINUNICODE")
    try:
        payload_bytes = payload.encode("latin1")
    except UnicodeEncodeError as error:
        raise AvailabilityAuditError("init pickle raw payload is not latin1") from error
    expected_bytes = 50 * expected_dimension * 8
    if len(payload_bytes) != expected_bytes:
        raise AvailabilityAuditError(
            "init raw payload length does not match the frozen (50,D) float64 shape"
        )
    return InitArchiveEvidence(
        task_id=_strict_int(task_id, "task_id"),
        task_name=task_name,
        relative_path=relative_path or path.name,
        sha256=hashlib.sha256(raw_archive).hexdigest(),
        shape=(50, expected_dimension),
        dtype="float64-le",
        raw_bytes=len(payload_bytes),
        pickle_opcode_count=len(operations),
    )


def audit_libero_long_availability(
    task_map_path: str | Path, init_root: str | Path
) -> LiberoLongAvailabilityAudit:
    """Prove that every LIBERO-Long task exposes states 0--49 statically."""

    task_map = _regular_file_without_symlink(task_map_path, suffix=".py")
    init_directory = Path(init_root).absolute()
    current = Path(init_directory.anchor)
    for part in init_directory.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise AvailabilityAuditError("LIBERO init root must not contain symlinks")
    try:
        init_directory = init_directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise AvailabilityAuditError("LIBERO init root is missing") from error
    if not init_directory.is_dir():
        raise AvailabilityAuditError("LIBERO init root must be a directory")
    tasks = _parse_libero_long_task_map(task_map)
    evidence: list[InitArchiveEvidence] = []
    for task_id, task_name in enumerate(tasks):
        filename = f"{task_name}.pruned_init"
        evidence.append(
            audit_libero_init_archive(
                init_directory / filename,
                task_id=task_id,
                task_name=task_name,
                expected_dimension=LIBERO_LONG_TASK_DIMENSIONS[task_name],
                relative_path=filename,
            )
        )
    result = LiberoLongAvailabilityAudit(
        task_map_sha256=hashlib.sha256(
            _read_bounded_regular_file(task_map, maximum=_MAX_TASK_MAP_BYTES)
        ).hexdigest(),
        init_root=str(init_directory),
        tasks=tuple(evidence),
    )
    if not result.passed:
        raise AvailabilityAuditError("ten-task/fifty-state availability proof failed")
    return result


def validate_libero_long_candidate_split(records: Sequence[LineageRecord]) -> None:
    """Require the exact disjoint 180/100/100 V3 candidate split."""

    expected = {
        role: {
            EpisodeIdentity(LIBERO_LONG_SUITE, task_id, episode)
            for task_id in range(10)
            for episode in episodes
        }
        for role, episodes in V3_EPISODES_BY_ROLE.items()
    }
    observed: dict[str, set[EpisodeIdentity]] = {
        role: set() for role in V3_CANDIDATE_ROLES
    }
    seen: set[EpisodeIdentity] = set()
    for record in records:
        if record.role not in observed:
            raise MetadataSchemaError(f"unexpected candidate role: {record.role}")
        if record.identity in seen:
            raise DuplicateIdentityError(
                f"candidate roles overlap at {record.identity.key}"
            )
        seen.add(record.identity)
        observed[record.role].add(record.identity)
    for role in V3_CANDIDATE_ROLES:
        if observed[role] != expected[role]:
            raise MetadataSchemaError(
                f"{role} must be the exact all-task episode grid "
                f"{V3_EPISODES_BY_ROLE[role][0]}--{V3_EPISODES_BY_ROLE[role][-1]}"
            )


def validate_historical_used_inventory(records: Sequence[LineageRecord]) -> None:
    """Require the frozen conservative Long104 + Spatial500 used registry."""

    expected = {
        EpisodeIdentity(LIBERO_LONG_SUITE, task, episode)
        for task in range(10)
        for episode in range(10)
    }
    expected.update(
        EpisodeIdentity(LIBERO_LONG_SUITE, task, episode)
        for task in (0, 1)
        for episode in (10, 11)
    )
    expected.update(
        EpisodeIdentity("libero_spatial", task, episode)
        for task in range(10)
        for episode in range(50)
    )
    if any(record.role != "historical_used" for record in records):
        raise MetadataSchemaError(
            "conservative used registry must declare role='historical_used'"
        )
    observed = [record.identity for record in records]
    if len(observed) != len(set(observed)):
        raise DuplicateIdentityError("conservative used registry repeats an exact key")
    if set(observed) != expected:
        raise MetadataSchemaError(
            "used inventory must be exact Long104 + Spatial500 (604 keys)"
        )


def scan_text_corpus_for_exact_keys(
    corpus_root: str | Path,
    keys: Iterable[str],
    *,
    excluded_paths: Iterable[str | Path] = (),
    skipped_row_payload_paths: Iterable[str] = (),
) -> TextCorpusScan:
    """Search project text evidence for exact keys without parsing row payloads."""

    root = Path(corpus_root).absolute()
    _reject_symlink_components(root, context="text corpus root")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise MetadataPathError("text corpus root must be a directory")
    root_identity = root.stat()
    key_bytes = {key: key.encode("ascii") for key in sorted(set(keys))}
    excluded = {Path(os.path.abspath(path)) for path in excluded_paths}
    frozen_row_payloads: set[str] = set()
    for value in skipped_row_payload_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise MetadataPathError("row-payload skip paths must be corpus-relative")
        frozen_row_payloads.add(relative.as_posix())
    for relative in sorted(frozen_row_payloads):
        frozen_path = root / relative
        if frozen_path.is_symlink() or not frozen_path.is_file():
            raise MetadataPathError(
                f"pinned row-payload skip is missing or unsafe: {relative}"
            )

    def is_excluded(path: Path) -> bool:
        lexical = Path(os.path.abspath(path))
        return any(lexical == item or item in lexical.parents for item in excluded)

    hits: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    scanned_file_count = 0
    scanned_bytes = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        kept_directories: list[str] = []
        for name in sorted(names):
            path = Path(directory) / name
            if name in _TEXT_SCAN_IGNORED_DIRECTORIES or is_excluded(path):
                continue
            if path.is_symlink():
                target = os.readlink(path)
                skipped.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "reason": "symlink directory not followed",
                        "link_target_sha256": hashlib.sha256(
                            os.fsencode(target)
                        ).hexdigest(),
                    }
                )
                continue
            kept_directories.append(name)
        names[:] = kept_directories
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() not in _TEXT_SCAN_SUFFIXES or is_excluded(path):
                continue
            relative = path.relative_to(root)
            if path.is_symlink():
                target = os.readlink(path)
                skipped.append(
                    {
                        "path": relative.as_posix(),
                        "reason": "symlink file not opened",
                        "link_target_sha256": hashlib.sha256(
                            os.fsencode(target)
                        ).hexdigest(),
                    }
                )
                continue
            if (
                _C361_RE.search(relative.as_posix().lower())
                and filename.lower() == "records.jsonl"
                and relative.as_posix() not in frozen_row_payloads
            ):
                raise ForbiddenLineageSourceError(
                    f"unregistered C3.61 row payload path is forbidden: {relative}"
                )
            if relative.as_posix() in frozen_row_payloads:
                skipped.append(
                    {
                        "path": relative.as_posix(),
                        "reason": "row-level/C3.61 payload forbidden by D0 boundary",
                    }
                )
                continue
            scanned_file_count += 1
            remaining = dict(key_bytes)
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise MetadataPathError(
                        f"text evidence changed type during scan: {relative}"
                    )
                overlap = b""
                overlap_size = max((len(value) for value in remaining.values()), default=1) - 1
                while remaining:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    scanned_bytes += len(chunk)
                    window = overlap + chunk
                    found = [key for key, encoded in remaining.items() if encoded in window]
                    for key in found:
                        hits.append({"key": key, "path": relative.as_posix()})
                        remaining.pop(key)
                    overlap = window[-overlap_size:] if overlap_size else b""
            finally:
                os.close(descriptor)
    after_root = root.stat()
    if (after_root.st_dev, after_root.st_ino) != (
        root_identity.st_dev,
        root_identity.st_ino,
    ):
        raise MetadataPathError("text corpus root changed identity during scan")
    return TextCorpusScan(
        hits=tuple(hits),
        skipped=tuple(skipped),
        scanned_file_count=scanned_file_count,
        scanned_bytes=scanned_bytes,
    )


__all__ = [
    "AUDIT_REPORT_SCHEMA_VERSION",
    "AUDIT_REQUEST_SCHEMA_VERSION",
    "FROZEN_METADATA_INDEX_SCHEMA_VERSION",
    "LIBERO_LONG_SUITE",
    "LIBERO_LONG_TASK_DIMENSIONS",
    "V3_CANDIDATE_ROLES",
    "V3_EPISODES_BY_ROLE",
    "V3_JSONL_ROW_SCHEMA_VERSION",
    "V3_SELECTION_SCHEMA_VERSION",
    "AuditRequest",
    "DataLineageError",
    "DataLineageReport",
    "DuplicateIdentityError",
    "EpisodeIdentity",
    "ForbiddenLineageSourceError",
    "AvailabilityAuditError",
    "CandidateOverlapError",
    "InitArchiveEvidence",
    "LiberoLongAvailabilityAudit",
    "TextCorpusScan",
    "LineageRecord",
    "LoadedSelection",
    "MetadataPathError",
    "MetadataSchemaError",
    "MetadataSourceSpec",
    "RoleConflictError",
    "audit_data_lineage",
    "audit_libero_init_archive",
    "audit_libero_long_availability",
    "build_data_lineage_report",
    "load_audit_request",
    "load_metadata_selection",
    "resolve_metadata_path",
    "scan_text_corpus_for_exact_keys",
    "validate_libero_long_candidate_split",
    "validate_historical_used_inventory",
]
