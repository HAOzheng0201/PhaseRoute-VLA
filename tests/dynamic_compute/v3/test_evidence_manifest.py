from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPOSITORY_ROOT / "a1/vla/dynamic_compute/v3/evidence_manifest.py"
FROZEN_MANIFEST = (
    REPOSITORY_ROOT / "docs/research/v3/legacy_evidence_manifest.json"
)
FROZEN_SOURCE_ROOT = (
    REPOSITORY_ROOT / "artifacts/phase_route_v3/legacy_source"
)

# Loading by file location is deliberate: importing through ``a1`` executes
# legacy package-level ML imports.  D0 must remain a stdlib-only CPU audit.
_MODULE_NAME = "_phaseroute_v3_evidence_manifest_test"
_MODULE_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, MODULE_PATH)
if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:
    raise RuntimeError(f"cannot create import spec for {MODULE_PATH}")
_EVIDENCE_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_NAME] = _EVIDENCE_MODULE
_MODULE_SPEC.loader.exec_module(_EVIDENCE_MODULE)

FORBIDDEN_SUFFIXES = _EVIDENCE_MODULE.FORBIDDEN_SUFFIXES
HASH_ALGORITHM = _EVIDENCE_MODULE.HASH_ALGORITHM
PATH_SEMANTICS = _EVIDENCE_MODULE.PATH_SEMANTICS
SCHEMA_VERSION = _EVIDENCE_MODULE.SCHEMA_VERSION
EvidenceIntegrityError = _EVIDENCE_MODULE.EvidenceIntegrityError
ManifestFormatError = _EVIDENCE_MODULE.ManifestFormatError
load_evidence_manifest = _EVIDENCE_MODULE.load_evidence_manifest
verify_evidence_manifest = _EVIDENCE_MODULE.verify_evidence_manifest


def _binding(
    identifier: str,
    relative_path: str,
    content: bytes,
    *,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "stage": "TEST",
        "role": "unit-test metadata",
        "path": relative_path,
        "size_bytes": len(content) if size_bytes is None else size_bytes,
        "sha256": hashlib.sha256(content).hexdigest() if sha256 is None else sha256,
    }


def _manifest(root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "unit-test-manifest",
        "source_root": str(root),
        "path_semantics": PATH_SEMANTICS,
        "hash_algorithm": HASH_ALGORITHM,
        "forbidden_suffixes": list(FORBIDDEN_SUFFIXES),
        "entry_count": len(entries),
        "evidence": entries,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class TestEvidenceManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.source_root = self.base / "source"
        self.source_root.mkdir()
        (self.source_root / "reports").mkdir()
        self.manifest_path = self.base / "manifest.json"

    def _valid_manifest(self) -> tuple[dict[str, Any], bytes]:
        content = b'{"status":"FROZEN"}\n'
        (self.source_root / "reports/evidence.json").write_bytes(content)
        return _manifest(
            self.source_root,
            [_binding("valid-evidence", "reports/evidence.json", content)],
        ), content

    def test_valid_manifest_verifies(self) -> None:
        manifest, content = self._valid_manifest()
        _write_json(self.manifest_path, manifest)

        parsed = load_evidence_manifest(self.manifest_path)
        report = verify_evidence_manifest(
            self.manifest_path, source_root=self.source_root
        )

        self.assertEqual(parsed.manifest_id, "unit-test-manifest")
        self.assertEqual(report.verified_count, 1)
        self.assertEqual(report.total_bytes, len(content))
        self.assertEqual(report.verified[0].path, "reports/evidence.json")

    def test_frozen_repository_manifest_verifies_without_payloads(self) -> None:
        report = verify_evidence_manifest(
            FROZEN_MANIFEST,
            source_root=FROZEN_SOURCE_ROOT,
            allow_relocated_root=True,
        )

        self.assertEqual(report.verified_count, 28)
        self.assertGreater(report.total_bytes, 1_000_000)
        self.assertTrue(
            all(
                Path(item.path).suffix.lower() not in FORBIDDEN_SUFFIXES
                for item in report.verified
            )
        )
        by_identifier = {item.identifier: item for item in report.verified}
        self.assertEqual(
            by_identifier["c359-independent-evaluation-protocol-result"].sha256,
            "d4c2a9f29ebd30903ef5b63402521f076eb15c856f60771f54c00ea9867632e8",
        )
        self.assertEqual(
            by_identifier["c360-v3-runner-contract-result"].sha256,
            "976d0ac55ad664427348909a14c94bf19652814ce380ec0ebb5a2a50200a3f03",
        )
        self.assertEqual(
            by_identifier["c361-global-attempt-consumed-marker"].sha256,
            "cc5d4872e0d6ee929ccf9398bad3d6b7f63520c9ad1f248bac1be5824e2c967a",
        )
        self.assertEqual(
            by_identifier["c361-evaluation-result"].sha256,
            "b696c9df07b1d83282af8699440919e8f6a835cc68f9ebaba3a6339b23d3a7c2",
        )

    def test_isolated_subprocess_is_stdlib_only_and_cpu_only(self) -> None:
        script = textwrap.dedent(
            f"""
            import importlib.util
            import pathlib
            import sys

            module_path = pathlib.Path({str(MODULE_PATH)!r})
            spec = importlib.util.spec_from_file_location(
                "_phaseroute_v3_evidence_manifest_subprocess", module_path
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            report = module.verify_evidence_manifest(
                pathlib.Path({str(FROZEN_MANIFEST)!r}),
                source_root=pathlib.Path({str(FROZEN_SOURCE_ROOT)!r}),
                allow_relocated_root=True,
            )
            forbidden = ("torch", "numpy", "tensorflow", "jax")
            loaded = [name for name in forbidden if name in sys.modules]
            assert not loaded, loaded
            assert report.verified_count == 28
            print(f"verified={{report.verified_count}}; third_party_ml_modules={{loaded}}")
            """
        )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )

        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "verified=28; third_party_ml_modules=[]", completed.stdout.strip()
        )

    def test_rejects_duplicate_ids_and_paths(self) -> None:
        first = b"first\n"
        second = b"second\n"
        (self.source_root / "reports/first.json").write_bytes(first)
        (self.source_root / "reports/second.json").write_bytes(second)

        duplicate_id = _manifest(
            self.source_root,
            [
                _binding("same-id", "reports/first.json", first),
                _binding("same-id", "reports/second.json", second),
            ],
        )
        duplicate_path = _manifest(
            self.source_root,
            [
                _binding("first-id", "reports/first.json", first),
                _binding("second-id", "reports/first.json", first),
            ],
        )

        for manifest, message in (
            (duplicate_id, "duplicate evidence id"),
            (duplicate_path, "duplicate evidence path"),
        ):
            with self.subTest(message=message):
                _write_json(self.manifest_path, manifest)
                with self.assertRaisesRegex(ManifestFormatError, message):
                    load_evidence_manifest(self.manifest_path)

    def test_rejects_unsafe_or_payload_paths_before_file_access(self) -> None:
        content = b"not opened"
        unsafe_paths = (
            "../outside.json",
            "/absolute/evidence.json",
            "reports//evidence.json",
            "reports\\evidence.json",
            "reports/model.ckpt",
            "reports/array.npy",
            "reports/payload.pt",
            "reports/model.pth",
            "reports/cache.pkl",
            "reports/cache.pickle",
            "reports/payload.NPZ",
            "reports/episode.init",
        )

        for index, unsafe_path in enumerate(unsafe_paths):
            with self.subTest(path=unsafe_path):
                manifest = _manifest(
                    self.source_root,
                    [_binding(f"unsafe-{index}", unsafe_path, content)],
                )
                _write_json(self.manifest_path, manifest)
                with self.assertRaises(ManifestFormatError):
                    load_evidence_manifest(self.manifest_path)

    def test_rejects_final_and_intermediate_symlinks(self) -> None:
        content = b'{"metadata":true}\n'
        target = self.source_root / "reports/target.json"
        target.write_bytes(content)
        (self.source_root / "reports/link.json").symlink_to(target.name)
        final_link_manifest = _manifest(
            self.source_root,
            [_binding("final-link", "reports/link.json", content)],
        )

        actual_directory = self.source_root / "actual"
        actual_directory.mkdir()
        (actual_directory / "evidence.json").write_bytes(content)
        (self.source_root / "linked-directory").symlink_to(
            actual_directory.name, target_is_directory=True
        )
        intermediate_link_manifest = _manifest(
            self.source_root,
            [
                _binding(
                    "intermediate-link",
                    "linked-directory/evidence.json",
                    content,
                )
            ],
        )

        for manifest in (final_link_manifest, intermediate_link_manifest):
            with self.subTest(path=manifest["evidence"][0]["path"]):
                _write_json(self.manifest_path, manifest)
                with self.assertRaisesRegex(EvidenceIntegrityError, "symlink"):
                    verify_evidence_manifest(self.manifest_path)

    def test_rejects_missing_and_non_regular_evidence(self) -> None:
        content = b"missing\n"
        missing = _manifest(
            self.source_root,
            [_binding("missing", "reports/missing.json", content)],
        )
        (self.source_root / "reports/directory.json").mkdir()
        directory = _manifest(
            self.source_root,
            [_binding("directory", "reports/directory.json", b"")],
        )

        for manifest, message in (
            (missing, "missing evidence file"),
            (directory, "not a regular file"),
        ):
            with self.subTest(message=message):
                _write_json(self.manifest_path, manifest)
                with self.assertRaisesRegex(EvidenceIntegrityError, message):
                    verify_evidence_manifest(self.manifest_path)

    def test_rejects_size_and_sha256_drift(self) -> None:
        manifest, content = self._valid_manifest()

        wrong_size = json.loads(json.dumps(manifest))
        wrong_size["evidence"][0]["size_bytes"] = len(content) + 1
        _write_json(self.manifest_path, wrong_size)
        with self.assertRaisesRegex(EvidenceIntegrityError, "size drift"):
            verify_evidence_manifest(self.manifest_path)

        wrong_hash = json.loads(json.dumps(manifest))
        wrong_hash["evidence"][0]["sha256"] = "0" * 64
        _write_json(self.manifest_path, wrong_hash)
        with self.assertRaisesRegex(EvidenceIntegrityError, "SHA-256 drift"):
            verify_evidence_manifest(self.manifest_path)

    def test_rejects_malformed_size_and_sha256_fields(self) -> None:
        manifest, _ = self._valid_manifest()
        invalid_values = (
            ("size_bytes", True),
            ("size_bytes", -1),
            ("sha256", "A" * 64),
            ("sha256", "0" * 63),
        )

        for field, invalid_value in invalid_values:
            with self.subTest(field=field, value=invalid_value):
                malformed = json.loads(json.dumps(manifest))
                malformed["evidence"][0][field] = invalid_value
                _write_json(self.manifest_path, malformed)
                with self.assertRaises(ManifestFormatError):
                    load_evidence_manifest(self.manifest_path)

    def test_rejects_duplicate_json_object_keys(self) -> None:
        self.manifest_path.write_text(
            '{"manifest_id":"one","manifest_id":"two"}', encoding="utf-8"
        )
        with self.assertRaisesRegex(ManifestFormatError, "duplicate JSON object key"):
            load_evidence_manifest(self.manifest_path)

    def test_rejects_manifest_symlink(self) -> None:
        manifest, _ = self._valid_manifest()
        actual_manifest = self.base / "actual-manifest.json"
        _write_json(actual_manifest, manifest)
        linked_manifest = self.base / "linked-manifest.json"
        linked_manifest.symlink_to(actual_manifest.name)

        with self.assertRaises(ManifestFormatError):
            load_evidence_manifest(linked_manifest)

    def test_rejects_source_root_mismatch_and_symlink(self) -> None:
        manifest, _ = self._valid_manifest()
        _write_json(self.manifest_path, manifest)
        other_root = self.base / "other-source"
        other_root.mkdir()

        with self.assertRaisesRegex(EvidenceIntegrityError, "does not match"):
            verify_evidence_manifest(self.manifest_path, source_root=other_root)

        linked_root = self.base / "linked-source"
        linked_root.symlink_to(self.source_root.name, target_is_directory=True)
        linked_manifest = _manifest(
            linked_root,
            manifest["evidence"],
        )
        _write_json(self.manifest_path, linked_manifest)
        with self.assertRaisesRegex(EvidenceIntegrityError, "source_root.*symlink"):
            verify_evidence_manifest(self.manifest_path)

    def test_explicit_relocation_is_portable_but_still_hash_bound(self) -> None:
        manifest, content = self._valid_manifest()
        _write_json(self.manifest_path, manifest)
        relocated_root = self.base / "relocated-source"
        (relocated_root / "reports").mkdir(parents=True)
        relocated_evidence = relocated_root / "reports/evidence.json"
        relocated_evidence.write_bytes(content)

        report = verify_evidence_manifest(
            self.manifest_path,
            source_root=relocated_root,
            allow_relocated_root=True,
        )

        self.assertEqual(report.declared_source_root, str(self.source_root))
        self.assertEqual(report.source_root, str(relocated_root))
        relocated_evidence.write_bytes(content + b"drift")
        with self.assertRaisesRegex(EvidenceIntegrityError, "size drift"):
            verify_evidence_manifest(
                self.manifest_path,
                source_root=relocated_root,
                allow_relocated_root=True,
            )

    def test_rejects_entry_count_and_schema_drift(self) -> None:
        manifest, _ = self._valid_manifest()
        manifest["entry_count"] = 2
        _write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ManifestFormatError, "entry_count mismatch"):
            load_evidence_manifest(self.manifest_path)

        manifest, _ = self._valid_manifest()
        manifest["unexpected"] = "field"
        _write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ManifestFormatError, "keys do not match"):
            load_evidence_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()
