"""Strict verification for the V3-D0 legacy-evidence manifest.

The manifest contains metadata evidence only.  Paths are interpreted relative
to one explicitly declared source root, and every evidence file is opened with
``O_NOFOLLOW`` component by component before its size and SHA-256 are checked.
Model/data payload suffixes are rejected while parsing the manifest, before
any evidence path is touched.

This module intentionally uses only the Python standard library.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "phase-route-vla.v3-d0-legacy-evidence-manifest.v1"
PATH_SEMANTICS = "source-root-relative-posix-no-symlinks"
HASH_ALGORITHM = "sha256"
FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".init",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
)
READ_CHUNK_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "source_root",
        "path_semantics",
        "hash_algorithm",
        "forbidden_suffixes",
        "entry_count",
        "evidence",
    }
)
_ENTRY_KEYS = frozenset(
    {"id", "stage", "role", "path", "size_bytes", "sha256"}
)
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceManifestError(Exception):
    """Base class for all manifest and evidence verification failures."""


class ManifestFormatError(EvidenceManifestError, ValueError):
    """The manifest is malformed or violates the frozen schema."""


class EvidenceIntegrityError(EvidenceManifestError, RuntimeError):
    """A declared evidence file is missing, unsafe, or has drifted."""


@dataclass(frozen=True)
class EvidenceEntry:
    """One immutable evidence-file binding."""

    identifier: str
    stage: str
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class EvidenceManifest:
    """Parsed and schema-validated legacy-evidence manifest."""

    manifest_id: str
    source_root: str
    entries: tuple[EvidenceEntry, ...]


@dataclass(frozen=True)
class VerifiedEvidence:
    """Observed identity of one successfully verified evidence file."""

    identifier: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class VerificationReport:
    """Deterministic summary returned after complete verification."""

    manifest_id: str
    declared_source_root: str
    source_root: str
    verified: tuple[VerifiedEvidence, ...]
    total_bytes: int

    @property
    def verified_count(self) -> int:
        return len(self.verified)


def _reject_duplicate_json_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestFormatError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    raise ManifestFormatError(
        f"{context} keys do not match schema; missing={missing}, unknown={unknown}"
    )


def _require_nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestFormatError(f"{context} must be non-empty trimmed text")
    if "\x00" in value:
        raise ManifestFormatError(f"{context} contains a NUL byte")
    return value


def _validate_source_root(value: Any) -> str:
    root = _require_nonempty_text(value, "source_root")
    if "\\" in root:
        raise ManifestFormatError("source_root must use POSIX separators")
    pure = PurePosixPath(root)
    if not pure.is_absolute():
        raise ManifestFormatError("source_root must be absolute")
    if any(part in {".", ".."} for part in pure.parts):
        raise ManifestFormatError("source_root must not contain dot components")
    if str(pure) != root:
        raise ManifestFormatError("source_root must be lexically normalized")
    return root


def _validate_relative_path(value: Any, context: str) -> str:
    path = _require_nonempty_text(value, context)
    if "\\" in path:
        raise ManifestFormatError(f"{context} must use POSIX separators")
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise ManifestFormatError(f"{context} must be relative to source_root")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ManifestFormatError(f"{context} contains a forbidden path component")
    if str(pure) != path:
        raise ManifestFormatError(f"{context} must be lexically normalized")
    suffix = pure.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        raise ManifestFormatError(
            f"{context} uses forbidden payload suffix {suffix!r}; payloads are not D0 evidence"
        )
    return path


def _parse_entry(value: Any, index: int) -> EvidenceEntry:
    context = f"evidence[{index}]"
    if not isinstance(value, Mapping):
        raise ManifestFormatError(f"{context} must be a JSON object")
    _require_exact_keys(value, _ENTRY_KEYS, context)

    identifier = _require_nonempty_text(value["id"], f"{context}.id")
    if _IDENTIFIER_RE.fullmatch(identifier) is None:
        raise ManifestFormatError(f"{context}.id has an invalid format")
    stage = _require_nonempty_text(value["stage"], f"{context}.stage")
    role = _require_nonempty_text(value["role"], f"{context}.role")
    path = _validate_relative_path(value["path"], f"{context}.path")

    size_bytes = value["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ManifestFormatError(f"{context}.size_bytes must be a non-negative integer")

    sha256 = value["sha256"]
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise ManifestFormatError(
            f"{context}.sha256 must be exactly 64 lowercase hexadecimal characters"
        )
    return EvidenceEntry(
        identifier=identifier,
        stage=stage,
        role=role,
        path=path,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _parse_manifest(value: Any) -> EvidenceManifest:
    if not isinstance(value, Mapping):
        raise ManifestFormatError("manifest root must be a JSON object")
    _require_exact_keys(value, _MANIFEST_KEYS, "manifest")

    if value["schema_version"] != SCHEMA_VERSION:
        raise ManifestFormatError(
            f"unsupported schema_version: {value['schema_version']!r}"
        )
    manifest_id = _require_nonempty_text(value["manifest_id"], "manifest_id")
    if _IDENTIFIER_RE.fullmatch(manifest_id) is None:
        raise ManifestFormatError("manifest_id has an invalid format")
    source_root = _validate_source_root(value["source_root"])
    if value["path_semantics"] != PATH_SEMANTICS:
        raise ManifestFormatError("path_semantics does not match the frozen contract")
    if value["hash_algorithm"] != HASH_ALGORITHM:
        raise ManifestFormatError("hash_algorithm must be 'sha256'")
    if value["forbidden_suffixes"] != list(FORBIDDEN_SUFFIXES):
        raise ManifestFormatError(
            "forbidden_suffixes must exactly match the frozen D0 payload policy"
        )

    entry_count = value["entry_count"]
    if isinstance(entry_count, bool) or not isinstance(entry_count, int) or entry_count < 1:
        raise ManifestFormatError("entry_count must be a positive integer")
    raw_entries = value["evidence"]
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ManifestFormatError("evidence must be a JSON array")
    if len(raw_entries) != entry_count:
        raise ManifestFormatError(
            f"entry_count mismatch: declared={entry_count}, observed={len(raw_entries)}"
        )

    entries = tuple(_parse_entry(item, index) for index, item in enumerate(raw_entries))
    identifiers = [entry.identifier for entry in entries]
    paths = [entry.path for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ManifestFormatError("duplicate evidence id")
    if len(set(paths)) != len(paths):
        raise ManifestFormatError("duplicate evidence path")
    return EvidenceManifest(
        manifest_id=manifest_id,
        source_root=source_root,
        entries=entries,
    )


def _read_manifest_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ManifestFormatError(f"cannot open manifest {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManifestFormatError(f"manifest is not a regular file: {path}")
        if metadata.st_size > MAX_MANIFEST_BYTES:
            raise ManifestFormatError(
                f"manifest exceeds {MAX_MANIFEST_BYTES} byte safety limit"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, MAX_MANIFEST_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_MANIFEST_BYTES:
                raise ManifestFormatError(
                    f"manifest exceeds {MAX_MANIFEST_BYTES} byte safety limit"
                )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_evidence_manifest(manifest_path: os.PathLike[str] | str) -> EvidenceManifest:
    """Load and validate a manifest without opening any evidence file."""

    path = Path(manifest_path)
    raw = _read_manifest_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestFormatError("manifest must be valid UTF-8") from error
    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except ManifestFormatError:
        raise
    except json.JSONDecodeError as error:
        raise ManifestFormatError(f"manifest is not valid JSON: {error}") from error
    return _parse_manifest(parsed)


def _source_root_descriptor(source_root: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", flags)
    except OSError as error:
        raise EvidenceIntegrityError(f"cannot open filesystem root: {error}") from error
    try:
        for component in PurePosixPath(source_root).parts[1:]:
            try:
                metadata = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except FileNotFoundError as error:
                raise EvidenceIntegrityError(
                    f"source_root does not exist: {source_root}"
                ) from error
            except OSError as error:
                raise EvidenceIntegrityError(
                    f"cannot inspect source_root {source_root!r}: {error}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise EvidenceIntegrityError(
                    f"source_root contains a symlink component: {source_root}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise EvidenceIntegrityError(
                    f"source_root contains a non-directory component: {source_root}"
                )
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise EvidenceIntegrityError(
                    f"cannot safely open source_root {source_root!r}: {error}"
                ) from error
            opened_metadata = os.fstat(next_descriptor)
            if (
                opened_metadata.st_dev != metadata.st_dev
                or opened_metadata.st_ino != metadata.st_ino
            ):
                os.close(next_descriptor)
                raise EvidenceIntegrityError(
                    f"source_root identity changed while opening: {source_root}"
                )
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_directory_component(parent_descriptor: int, component: str, path: str) -> int:
    try:
        metadata = os.stat(component, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as error:
        raise EvidenceIntegrityError(f"missing evidence path: {path}") from error
    except OSError as error:
        raise EvidenceIntegrityError(f"cannot inspect evidence path {path!r}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise EvidenceIntegrityError(f"symlink component rejected: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceIntegrityError(f"non-directory path component rejected: {path}")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(component, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise EvidenceIntegrityError(
            f"cannot safely open evidence path {path!r}: {error}"
        ) from error
    opened_metadata = os.fstat(descriptor)
    if (
        opened_metadata.st_dev != metadata.st_dev
        or opened_metadata.st_ino != metadata.st_ino
    ):
        os.close(descriptor)
        raise EvidenceIntegrityError(f"path identity changed while opening: {path}")
    return descriptor


def _verify_entry(root_descriptor: int, entry: EvidenceEntry) -> VerifiedEvidence:
    components = PurePosixPath(entry.path).parts
    directory_descriptor = os.dup(root_descriptor)
    try:
        for component in components[:-1]:
            next_descriptor = _open_directory_component(
                directory_descriptor, component, entry.path
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor

        filename = components[-1]
        try:
            before = os.stat(
                filename, dir_fd=directory_descriptor, follow_symlinks=False
            )
        except FileNotFoundError as error:
            raise EvidenceIntegrityError(f"missing evidence file: {entry.path}") from error
        except OSError as error:
            raise EvidenceIntegrityError(
                f"cannot inspect evidence file {entry.path!r}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise EvidenceIntegrityError(f"symlink evidence rejected: {entry.path}")
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceIntegrityError(
                f"evidence is not a regular file: {entry.path}"
            )
        if before.st_size != entry.size_bytes:
            raise EvidenceIntegrityError(
                f"size drift for {entry.path}: expected={entry.size_bytes}, "
                f"observed={before.st_size}"
            )

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            file_descriptor = os.open(filename, flags, dir_fd=directory_descriptor)
        except OSError as error:
            raise EvidenceIntegrityError(
                f"cannot safely open evidence file {entry.path!r}: {error}"
            ) from error
        try:
            opened = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise EvidenceIntegrityError(
                    f"evidence is not a regular file: {entry.path}"
                )
            if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
                raise EvidenceIntegrityError(
                    f"evidence identity changed while opening: {entry.path}"
                )

            digest = hashlib.sha256()
            observed_size = 0
            while True:
                chunk = os.read(file_descriptor, READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                observed_size += len(chunk)
            after = os.fstat(file_descriptor)
        finally:
            os.close(file_descriptor)

        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise EvidenceIntegrityError(
                f"evidence changed while hashing: {entry.path}"
            )
        if observed_size != entry.size_bytes:
            raise EvidenceIntegrityError(
                f"size drift while hashing {entry.path}: expected={entry.size_bytes}, "
                f"observed={observed_size}"
            )
        observed_sha256 = digest.hexdigest()
        if not hmac.compare_digest(observed_sha256, entry.sha256):
            raise EvidenceIntegrityError(
                f"SHA-256 drift for {entry.path}: expected={entry.sha256}, "
                f"observed={observed_sha256}"
            )
        return VerifiedEvidence(
            identifier=entry.identifier,
            path=entry.path,
            size_bytes=observed_size,
            sha256=observed_sha256,
        )
    finally:
        os.close(directory_descriptor)


def verify_evidence_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    source_root: os.PathLike[str] | str | None = None,
    allow_relocated_root: bool = False,
) -> VerificationReport:
    """Verify every evidence binding and return a deterministic report.

    By default, a supplied ``source_root`` must match the provenance root
    frozen in the manifest.  For an intentional repository relocation, pass
    an absolute, normalized ``source_root`` together with
    ``allow_relocated_root=True``.  Relocation changes only the filesystem
    base: every relative path, byte size, and SHA-256 must still match.

    Root handling is lexical and component-wise; this function never calls
    ``resolve()`` and rejects a symlink in any root or evidence-path component.
    """

    manifest = load_evidence_manifest(manifest_path)
    if source_root is None:
        actual_source_root = manifest.source_root
    else:
        supplied_source_root = os.fspath(source_root)
        if not os.path.isabs(supplied_source_root):
            raise EvidenceIntegrityError("supplied source_root must be absolute")
        actual_source_root = os.path.normpath(supplied_source_root)
        if actual_source_root != supplied_source_root:
            raise EvidenceIntegrityError(
                "supplied source_root must be lexically normalized"
            )
        if actual_source_root != manifest.source_root and not allow_relocated_root:
            raise EvidenceIntegrityError(
                "source_root does not match the root declared by the manifest: "
                f"declared={manifest.source_root!r}, supplied={actual_source_root!r}"
            )

    root_descriptor = _source_root_descriptor(actual_source_root)
    try:
        verified = tuple(
            _verify_entry(root_descriptor, entry) for entry in manifest.entries
        )
    finally:
        os.close(root_descriptor)
    return VerificationReport(
        manifest_id=manifest.manifest_id,
        declared_source_root=manifest.source_root,
        source_root=actual_source_root,
        verified=verified,
        total_bytes=sum(item.size_bytes for item in verified),
    )


__all__ = [
    "EvidenceEntry",
    "EvidenceIntegrityError",
    "EvidenceManifest",
    "EvidenceManifestError",
    "FORBIDDEN_SUFFIXES",
    "HASH_ALGORITHM",
    "ManifestFormatError",
    "PATH_SEMANTICS",
    "SCHEMA_VERSION",
    "VerificationReport",
    "VerifiedEvidence",
    "load_evidence_manifest",
    "verify_evidence_manifest",
]
