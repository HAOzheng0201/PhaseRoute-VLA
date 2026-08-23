#!/usr/bin/env python3
"""Run the V3-D0 metadata-only, CPU-only data-lineage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


# Keep this process metadata-only even when launched from a GPU shell.  Import
# the leaf stdlib-only module directly: importing ``a1`` would import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIRECTORY = REPO_ROOT / "a1" / "vla" / "dynamic_compute" / "v3"
sys.path.insert(0, str(MODULE_DIRECTORY))

import data_lineage as lineage  # noqa: E402
import evidence_manifest  # noqa: E402


_ALLOWED_USER_EXCLUSION_RELATIVE = frozenset(
    {
        "worktrees/phaseroute-v3/results/v3/v3_d0_data_lineage_audit.json",
    }
)
_PINNED_TEXT_SYMLINKS = {
    "A1_source_backup_20260801/source/robot_experiments/vlabench/VLABench/add_condiment.py": (
        "bd7a9521adf822dcd95e819f3201080a5e90b0c40e3726f9cda88b4fc2890905"
    ),
    "source/robot_experiments/vlabench/VLABench/add_condiment.py": (
        "bd7a9521adf822dcd95e819f3201080a5e90b0c40e3726f9cda88b4fc2890905"
    ),
}
_PINNED_C361_ROW_PAYLOAD_PATHS = frozenset(
    {
        "source/reports/phase_route_v2_stage_c361_independent_aggregate_20260819_v1/records.jsonl",
        "source/reports/phase_route_v2_stage_c361_independent_candidate_shard00of04_gpu0_20260819_v1/records.jsonl",
        "source/reports/phase_route_v2_stage_c361_independent_candidate_shard01of04_gpu1_20260819_v1/records.jsonl",
        "source/reports/phase_route_v2_stage_c361_independent_candidate_shard02of04_gpu2_20260819_v1/records.jsonl",
        "source/reports/phase_route_v2_stage_c361_independent_candidate_shard03of04_gpu3_20260819_v1/records.jsonl",
        "source/reports/phase_route_v2_stage_c361_independent_context_20260819_v1/records.jsonl",
        "source/reports/phase_route_v2_stage_c361_independent_evaluation_20260819_v1/records.jsonl",
    }
)
_PINNED_MANIFEST_SHA256 = (
    "4ae5b617525a1f575f62700ab46434a1c9e8b20b9d13863b7ae8787f74c0ea6a"
)
_PINNED_MANIFEST_ID = "phaseroute-v3-d0-legacy-evidence-20260819-v1"
_PINNED_LINEAGE_BASIS = {
    "c2-fresh-offset-fit-report": (
        "0349f2751dd9798c641303be680ce3cda9e698297dbe0c79f23a86a9d328ed1c"
    ),
    "c2-ep30-39-index-records": (
        "416276d500bb0ce0cb584016c6250c6550aab17ea2402fd1784032b27233bc83"
    ),
    "c2-teacher-cache-audit-result": (
        "8980228080f11ffe846e74f7310ab38f14987e8d46ccee068e8335bd69455275"
    ),
    "c2-ep30-39-index-result": (
        "c228e4ad8962d50a645949415a00e3c8187f2a963bd01ddbf294183c082f565e"
    ),
    "c326-independent-selection-audit-result": (
        "1aa4b25d82b5a79f981d20df83348c0f0709e09fcf81d393ed643dbe4e9c9aab"
    ),
    "c327-independent-shadow-summary-result": (
        "02880b6841c6b1c2f1b9250eee4f5d0e2856161f05135c994f590761cc8c9a81"
    ),
    "c349-metadata-index-result": (
        "fe45b998efe7a45ea9620937bcfedc4bcd0e3a574ee51409087f50a938f5f868"
    ),
    "c349-metadata-index-records": (
        "0c06574cb526e9068cce2195eacaa9f5ebae8689e527c336a50019dd6a0f5e0b"
    ),
    "c359-independent-test-metadata-index": (
        "c713cf8e2d4140747cdf119fa9cc552441fafe98b01679120f279608899b0ed8"
    ),
    "c361-global-attempt-consumed-marker": (
        "cc5d4872e0d6ee929ccf9398bad3d6b7f63520c9ad1f248bac1be5824e2c967a"
    ),
    "c361-aggregate-result": (
        "16a78d90c3eec9b4bed52756ddc30b8787fbe42f20415ab6fcf997c01b29f118"
    ),
    "c361-evaluation-result": (
        "b696c9df07b1d83282af8699440919e8f6a835cc68f9ebaba3a6339b23d3a7c2"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--request", default="request.json", help="relative JSON request path")
    parser.add_argument(
        "--initialize-default-request",
        action="store_true",
        help="write frozen 604-used (Long104+Spatial500) and 380-candidate grids first",
    )
    parser.add_argument("--task-map", type=Path, required=True)
    parser.add_argument("--init-root", type=Path, required=True)
    parser.add_argument("--text-corpus-root", type=Path, required=True)
    parser.add_argument("--legacy-evidence-manifest", type=Path, required=True)
    parser.add_argument("--legacy-source-root", type=Path, required=True)
    parser.add_argument("--exclude-text-path", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit output: {destination}")
    temporary = destination.with_name(destination.name + ".incomplete")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"incomplete audit output already exists: {temporary}")
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite audit output: {destination}"
        ) from error
    os.unlink(temporary)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _initialize_default_request(metadata_root: Path, request_name: str) -> None:
    root = metadata_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    request_relative = Path(request_name)
    if request_relative.is_absolute() or ".." in request_relative.parts:
        raise lineage.MetadataPathError("initialized request path must stay under metadata root")
    used_pairs = [
        (task, episode) for task in range(10) for episode in range(10)
    ] + [
        (task, episode) for task in (0, 1) for episode in (10, 11)
    ]

    def selection(
        suite: str,
        role: str,
        pairs: list[tuple[int, int]],
        *,
        include_candidate_seeds: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema_version": lineage.V3_SELECTION_SCHEMA_VERSION,
            "suite": suite,
            "role": role,
            "records": [
                {
                    "task_id": task,
                    "episode_index": episode,
                    **(
                        {"seed": 20260811 + task * 10_000 + episode}
                        if include_candidate_seeds
                        else {}
                    ),
                }
                for task, episode in pairs
            ],
        }

    used_name = "used_libero_long.json"
    spatial_used_name = "used_libero_spatial_conservative.json"
    planned = [root / used_name, root / spatial_used_name, root / request_relative]
    planned.extend(root / f"{role}.json" for role in lineage.V3_CANDIDATE_ROLES)
    if any(path.exists() or path.is_symlink() for path in planned):
        raise FileExistsError("refusing partial initialization over existing inputs")
    _write_new_json(
        root / used_name,
        selection(lineage.LIBERO_LONG_SUITE, "historical_used", used_pairs),
    )
    spatial_pairs = [
        (task, episode) for task in range(10) for episode in range(50)
    ]
    _write_new_json(
        root / spatial_used_name,
        selection("libero_spatial", "historical_used", spatial_pairs),
    )
    candidate_names: list[str] = []
    for role, episodes in lineage.V3_EPISODES_BY_ROLE.items():
        name = f"{role}.json"
        candidate_names.append(name)
        pairs = [
            (task, episode) for task in range(10) for episode in episodes
        ]
        _write_new_json(
            root / name,
            selection(
                lineage.LIBERO_LONG_SUITE,
                role,
                pairs,
                include_candidate_seeds=True,
            ),
        )
    request = {
        "schema_version": lineage.AUDIT_REQUEST_SCHEMA_VERSION,
        "used_sources": [
            {
                "path": used_name,
                "schema_version": lineage.V3_SELECTION_SCHEMA_VERSION,
            },
            {
                "path": spatial_used_name,
                "schema_version": lineage.V3_SELECTION_SCHEMA_VERSION,
            },
        ],
        "candidate_sources": [
            {
                "path": name,
                "schema_version": lineage.V3_SELECTION_SCHEMA_VERSION,
            }
            for name in candidate_names
        ],
    }
    _write_new_json(root / request_relative, request)


def _portable_path(
    path: Path, *, strict: bool = True, follow_symlinks: bool = True
) -> str:
    resolved = (
        path.absolute().resolve(strict=strict)
        if follow_symlinks
        else Path(os.path.abspath(path))
    )
    project_storage_root = REPO_ROOT.parents[1]
    try:
        relative = resolved.relative_to(project_storage_root)
    except ValueError:
        return f"<EXTERNAL>/{resolved.name}"
    return f"<PROJECT_ROOT>/{relative.as_posix()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_user_exclusions(paths: list[str], corpus_root: Path) -> list[Path]:
    project_storage_root = REPO_ROOT.parents[1].resolve(strict=True)
    corpus = corpus_root.absolute().resolve(strict=True)
    validated: list[Path] = []
    for raw_path in paths:
        path = Path(os.path.abspath(raw_path))
        if path == corpus:
            raise lineage.MetadataPathError("excluding the entire text corpus is forbidden")
        try:
            relative = path.relative_to(project_storage_root).as_posix()
        except ValueError as error:
            raise lineage.MetadataPathError(
                f"user text exclusion is outside the pre-registered project boundary: {path}"
            ) from error
        if relative not in _ALLOWED_USER_EXCLUSION_RELATIVE:
            raise lineage.MetadataPathError(
                f"user text exclusion is not pre-registered: {relative}"
            )
        validated.append(path)
    return validated


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    # Preserve the caller's lexical path.  Resolving it here would erase an
    # intermediate symlink before the library can reject that component.
    metadata_root = args.metadata_root.absolute()
    expected_corpus_root = args.legacy_source_root.absolute().parent
    if (
        args.text_corpus_root.absolute().resolve(strict=True)
        != expected_corpus_root.resolve(strict=True)
    ):
        raise lineage.MetadataPathError(
            "text-corpus-root must equal the parent of legacy-source-root"
        )
    request = lineage.load_audit_request(metadata_root, args.request)
    request_path = lineage.resolve_metadata_path(metadata_root, args.request)
    used = tuple(
        lineage.load_metadata_selection(metadata_root, spec)
        for spec in request.used_sources
    )
    candidates = tuple(
        lineage.load_metadata_selection(metadata_root, spec)
        for spec in request.candidate_sources
    )
    candidate_records = tuple(
        record for selection in candidates for record in selection.records
    )
    used_records = tuple(record for selection in used for record in selection.records)
    lineage.validate_historical_used_inventory(used_records)
    lineage.validate_libero_long_candidate_split(candidate_records)
    availability = lineage.audit_libero_long_availability(
        args.task_map, args.init_root
    )
    legacy_verification = evidence_manifest.verify_evidence_manifest(
        args.legacy_evidence_manifest,
        source_root=str(args.legacy_source_root.absolute()),
        allow_relocated_root=True,
    )
    manifest_sha256 = _sha256_file(args.legacy_evidence_manifest)
    if manifest_sha256 != _PINNED_MANIFEST_SHA256:
        raise lineage.MetadataSchemaError("legacy manifest SHA-256 is not pinned D0 evidence")
    if legacy_verification.manifest_id != _PINNED_MANIFEST_ID:
        raise lineage.MetadataSchemaError("legacy manifest id is not pinned D0 evidence")
    verified_by_id = {
        item.identifier: item.sha256 for item in legacy_verification.verified
    }
    if any(
        verified_by_id.get(identifier) != sha256
        for identifier, sha256 in _PINNED_LINEAGE_BASIS.items()
    ):
        raise lineage.MetadataSchemaError("critical lineage-basis entries are not pinned")
    report = lineage.build_data_lineage_report(used, candidates)

    candidate_keys = [record.identity.key for record in candidate_records]
    automatic_exclusions: list[Path] = [
        REPO_ROOT,
        REPO_ROOT.parents[1] / "hf_cache",
        metadata_root / args.request,
        args.output,
        *(metadata_root / spec.path for spec in request.candidate_sources),
    ]
    explicit_exclusions = _validated_user_exclusions(
        args.exclude_text_path, args.text_corpus_root
    )
    all_exclusions = [*automatic_exclusions, *explicit_exclusions]
    text_scan = lineage.scan_text_corpus_for_exact_keys(
        args.text_corpus_root,
        candidate_keys,
        excluded_paths=all_exclusions,
        skipped_row_payload_paths=_PINNED_C361_ROW_PAYLOAD_PATHS,
    )

    result = report.to_dict()
    request_sha256 = _sha256_file(request_path)
    selection_fingerprint = {
        "schema_version": lineage.AUDIT_REQUEST_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "sources": [
            {
                "purpose": purpose,
                "path": item.spec.path,
                "sha256": item.sha256,
            }
            for purpose, selections in (("used", used), ("candidate", candidates))
            for item in selections
        ],
    }
    selection_fingerprint_bytes = json.dumps(
        selection_fingerprint,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    result["input_selection_integrity"] = {
        "algorithm": "sha256(canonical-json(request-sha256,ordered-source-sha256))",
        "request_sha256": request_sha256,
        "selection_bundle_sha256": hashlib.sha256(
            selection_fingerprint_bytes
        ).hexdigest(),
    }
    availability_result = availability.to_dict()
    availability_result["init_root"] = _portable_path(args.init_root)
    result["availability_audit"] = availability_result
    result["legacy_evidence_verification"] = {
        "manifest": _portable_path(args.legacy_evidence_manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_id": legacy_verification.manifest_id,
        "source_root": _portable_path(args.legacy_source_root),
        "verified_count": legacy_verification.verified_count,
        "verified_bytes": legacy_verification.total_bytes,
        "all_entries_verified": legacy_verification.verified_count == 28,
        "lineage_basis": [
            {"id": identifier, "sha256": sha256}
            for identifier, sha256 in sorted(_PINNED_LINEAGE_BASIS.items())
        ],
    }
    result["implementation_sha256"] = {
        "data_lineage.py": _sha256_file(MODULE_DIRECTORY / "data_lineage.py"),
        "audit_v3_data_lineage.py": _sha256_file(Path(__file__)),
    }
    result["candidate_role_counts"] = {
        role: sum(record.role == role for record in candidate_records)
        for role in lineage.V3_CANDIDATE_ROLES
    }
    result["full_project_text_scan"] = {
        "root": _portable_path(args.text_corpus_root),
        "candidate_files_and_output_excluded": True,
        "automatic_exclusions": [
            {
                "path": _portable_path(
                    path, strict=False, follow_symlinks=False
                ),
                "reason": "current V3 protocol/code, non-historical"
                if path == REPO_ROOT
                else "candidate/request/current-output self-hit prevention",
            }
            for path in sorted(automatic_exclusions, key=lambda item: os.fspath(item))
        ],
        "explicit_exclusions": [
            {
                "path": _portable_path(
                    path, strict=False, follow_symlinks=False
                ),
                "reason": "pre-registered symlink alias or prior curated output",
            }
            for path in sorted(explicit_exclusions, key=lambda item: os.fspath(item))
        ],
        **text_scan.to_dict(),
    }
    observed_row_skips = {
        item["path"]
        for item in text_scan.skipped
        if item["reason"] == "row-level/C3.61 payload forbidden by D0 boundary"
    }
    symlink_skips = {
        item["path"]: item.get("link_target_sha256")
        for item in text_scan.skipped
        if item["reason"] in {"symlink file not opened", "symlink directory not followed"}
    }
    other_skips = [
        item
        for item in text_scan.skipped
        if item["reason"]
        not in {
            "row-level/C3.61 payload forbidden by D0 boundary",
            "symlink file not opened",
            "symlink directory not followed",
        }
    ]
    result["full_project_text_scan"]["pinned_row_payload_skip_match"] = (
        observed_row_skips == set(_PINNED_C361_ROW_PAYLOAD_PATHS)
    )
    result["full_project_text_scan"]["unregistered_skip_count"] = len(other_skips)
    result["full_project_text_scan"]["pinned_symlink_skip_match"] = (
        symlink_skips == _PINNED_TEXT_SYMLINKS
    )
    forbidden_loaded = sorted(
        name
        for name in ("torch", "numpy", "tensorflow", "jax")
        if name in sys.modules
    )
    result["process_boundary"] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "forbidden_modules_loaded": forbidden_loaded,
        "stdlib_only": not forbidden_loaded,
    }
    if text_scan.hits:
        result["status"] = "FAIL_CANDIDATE_TEXT_HISTORY_INTERSECTION"
        result["no_known_hit_authorized"] = False
    passed = (
        result["status"] == "PASS_NO_KNOWN_HIT"
        and availability.passed
        and not text_scan.hits
        and text_scan.scanned_file_count > 0
        and observed_row_skips == set(_PINNED_C361_ROW_PAYLOAD_PATHS)
        and symlink_skips == _PINNED_TEXT_SYMLINKS
        and not other_skips
        and legacy_verification.verified_count == 28
        and not forbidden_loaded
    )
    result["candidate_split_authorized"] = passed
    return result, 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.initialize_default_request:
            _initialize_default_request(args.metadata_root, args.request)
        result, exit_code = run(args)
        _write_new_json(args.output, result)
    except (
        lineage.DataLineageError,
        evidence_manifest.EvidenceManifestError,
        OSError,
        ValueError,
    ) as error:
        print(f"V3-D0 audit failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.absolute()),
        "counts": result["counts"],
        "candidate_role_counts": result["candidate_role_counts"],
    }, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
