from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIRECTORY = REPO_ROOT / "a1" / "vla" / "dynamic_compute" / "v3"
sys.path.insert(0, str(MODULE_DIRECTORY))
import data_lineage as dl  # noqa: E402


def _put_unicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(encoded)) + encoded


def _put_global(value: str) -> bytes:
    module, name = value.split(" ", 1)
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _static_numpy_pickle(
    *, rows: int = 50, dimension: int = 47, malicious_global: bool = False
) -> bytes:
    raw = "\x00" * (rows * dimension * 8)
    first_global = (
        "builtins eval" if malicious_global else "numpy.core.multiarray _reconstruct"
    )
    pieces = [
        b"\x80\x02",
        _put_global(first_global), b"q\x00",
        _put_global("numpy ndarray"), b"q\x01K\x00\x85q\x02",
        _put_global("_codecs encode"), b"q\x03",
        _put_unicode("b"), b"q\x04", _put_unicode("latin1"), b"q\x05\x86q\x06Rq\x07\x87q\x08Rq\x09(",
        b"K\x01", bytes((0x4B, rows)), bytes((0x4B, dimension)), b"\x86q\x0a",
        _put_global("numpy dtype"), b"q\x0b", _put_unicode("f8"),
        b"q\x0c\x89\x88\x87q\x0dRq\x0e(K\x03", _put_unicode("<"),
        b"q\x0fNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x10b\x89h\x03",
        _put_unicode(raw), b"q\x11h\x05\x86q\x12Rq\x13tq\x14b.",
    ]
    return b"".join(pieces)


def _write_init(path: Path, *, rows: int = 50, dimension: int, malicious=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "archive/data.pkl",
            _static_numpy_pickle(
                rows=rows, dimension=dimension, malicious_global=malicious
            ),
        )
        archive.writestr("archive/version", b"3\n")


def _write_availability_fixture(root: Path) -> tuple[Path, Path]:
    task_map = root / "libero_suite_task_map.py"
    task_map.write_text(
        "libero_task_map = "
        + repr({dl.LIBERO_LONG_SUITE: list(dl.LIBERO_LONG_TASK_DIMENSIONS)})
        + "\n",
        encoding="utf-8",
    )
    init_root = root / "init_files"
    for name, dimension in dl.LIBERO_LONG_TASK_DIMENSIONS.items():
        _write_init(init_root / f"{name}.pruned_init", dimension=dimension)
    return task_map, init_root


def _write_selection(
    path: Path, role: str, pairs: list[tuple[int, int]], *, seed: int = 7
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": dl.V3_SELECTION_SCHEMA_VERSION,
                "suite": dl.LIBERO_LONG_SUITE,
                "role": role,
                "seed": seed,
                "records": [
                    {"task_id": task, "episode_index": episode}
                    for task, episode in pairs
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_complete_request(root: Path) -> Path:
    used = [(task, episode) for task in range(10) for episode in range(10)]
    used += [(task, episode) for task in (0, 1) for episode in (10, 11)]
    _write_selection(root / "used.json", "historical_used", used)
    spatial = root / "used_spatial.json"
    spatial.write_text(
        json.dumps(
            {
                "schema_version": dl.V3_SELECTION_SCHEMA_VERSION,
                "suite": "libero_spatial",
                "role": "historical_used",
                "records": [
                    {"task_id": task, "episode_index": episode}
                    for task in range(10)
                    for episode in range(50)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_paths = []
    for role, episodes in dl.V3_EPISODES_BY_ROLE.items():
        path = root / f"{role}.json"
        _write_selection(
            path,
            role,
            [(task, episode) for task in range(10) for episode in episodes],
        )
        candidate_paths.append(path.name)
    request = root / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": dl.AUDIT_REQUEST_SCHEMA_VERSION,
                "used_sources": [
                    {"path": "used.json", "schema_version": dl.V3_SELECTION_SCHEMA_VERSION},
                    {"path": "used_spatial.json", "schema_version": dl.V3_SELECTION_SCHEMA_VERSION},
                ],
                "candidate_sources": [
                    {"path": name, "schema_version": dl.V3_SELECTION_SCHEMA_VERSION}
                    for name in candidate_paths
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return request


def test_identity_key_is_the_frozen_exact_text_and_seed_does_not_narrow_it() -> None:
    identity = dl.EpisodeIdentity("libero_10", 3, 12)
    assert identity.key == ":".join(("libero_10", "task3", "episode12"))
    first = dl.LineageRecord(identity, "history", 1, "a.json")
    second = dl.LineageRecord(identity, "history", 2, "b.json")
    assert first.identity == second.identity


def test_complete_request_has_expected_604_and_380_key_counts(tmp_path: Path) -> None:
    _write_complete_request(tmp_path)
    request = dl.load_audit_request(tmp_path, "request.json")
    report = dl.audit_data_lineage(tmp_path, request)
    assert report.status == "PASS_NO_KNOWN_HIT"
    assert report.used_record_count == 604
    assert report.candidate_record_count == 380
    assert len(report.candidate_no_known_hit_keys) == 380


def test_exact_used_and_candidate_grids_reject_missing_extra_and_cross_role(
    tmp_path: Path,
) -> None:
    _write_complete_request(tmp_path)
    request = dl.load_audit_request(tmp_path, "request.json")
    used = tuple(
        record
        for spec in request.used_sources
        for record in dl.load_metadata_selection(tmp_path, spec).records
    )
    candidates = tuple(
        record
        for spec in request.candidate_sources
        for record in dl.load_metadata_selection(tmp_path, spec).records
    )
    dl.validate_historical_used_inventory(used)
    dl.validate_libero_long_candidate_split(candidates)
    with pytest.raises(dl.MetadataSchemaError):
        dl.validate_historical_used_inventory(used[:-1])
    extra = dl.LineageRecord(
        dl.EpisodeIdentity("libero_object", 0, 0),
        "historical_used",
        None,
        "extra.json",
    )
    with pytest.raises(dl.MetadataSchemaError):
        dl.validate_historical_used_inventory((*used, extra))
    with pytest.raises(dl.MetadataSchemaError):
        dl.validate_libero_long_candidate_split(candidates[:-1])
    duplicate = dl.LineageRecord(
        candidates[0].identity,
        "calibration_v2",
        None,
        "duplicate.json",
    )
    with pytest.raises(dl.DuplicateIdentityError):
        dl.validate_libero_long_candidate_split((*candidates, duplicate))


def test_duplicate_source_and_duplicate_selection_key_fail_closed(tmp_path: Path) -> None:
    _write_selection(tmp_path / "same.json", "historical", [(0, 0), (0, 0)])
    spec = dl.MetadataSourceSpec("same.json", dl.V3_SELECTION_SCHEMA_VERSION)
    with pytest.raises(dl.DuplicateIdentityError):
        dl.load_metadata_selection(tmp_path, spec)

    request = {
        "schema_version": dl.AUDIT_REQUEST_SCHEMA_VERSION,
        "used_sources": [{"path": "same.json", "schema_version": dl.V3_SELECTION_SCHEMA_VERSION}],
        "candidate_sources": [{"path": "same.json", "schema_version": dl.V3_SELECTION_SCHEMA_VERSION}],
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(dl.MetadataSchemaError, match="repeats a source path"):
        dl.load_audit_request(tmp_path, "request.json")


@pytest.mark.parametrize(
    "suffix",
    [
        ".pt", ".pth", ".ckpt", ".safetensors", ".bin",
        ".npy", ".npz", ".init", ".pkl", ".pickle",
    ],
)
def test_binary_and_model_suffixes_are_rejected(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"input{suffix}"
    path.write_bytes(b"not read")
    with pytest.raises(dl.ForbiddenLineageSourceError):
        dl.resolve_metadata_path(tmp_path, path.name)


def test_unknown_payload_c361_traversal_and_symlink_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "unknown.json").write_text(
        json.dumps({"schema_version": "unknown", "payload": [1]}), encoding="utf-8"
    )
    with pytest.raises(dl.MetadataSchemaError):
        dl.load_metadata_selection(
            tmp_path, dl.MetadataSourceSpec("unknown.json", "unknown")
        )
    c361 = tmp_path / "stage_c361_records.jsonl"
    c361.write_text("{}\n", encoding="utf-8")
    with pytest.raises(dl.ForbiddenLineageSourceError):
        dl.resolve_metadata_path(tmp_path, c361.name)
    dotted = tmp_path / "stage_c3.61" / "selection.json"
    dotted.parent.mkdir()
    dotted.write_text("{}\n", encoding="utf-8")
    with pytest.raises(dl.ForbiddenLineageSourceError):
        dl.resolve_metadata_path(tmp_path, "stage_c3.61/selection.json")
    with pytest.raises(dl.MetadataPathError):
        dl.resolve_metadata_path(tmp_path, "../escape.json")
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (tmp_path / "link.json").symlink_to(target)
    with pytest.raises(dl.MetadataPathError):
        dl.resolve_metadata_path(tmp_path, "link.json")
    real_root = tmp_path / "real_root"
    (real_root / "metadata").mkdir(parents=True)
    (real_root / "metadata/input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "root_alias").symlink_to(real_root, target_is_directory=True)
    with pytest.raises(dl.MetadataPathError, match="symlink component"):
        dl.resolve_metadata_path(tmp_path / "root_alias/metadata", "input.json")


def test_static_init_audit_accepts_whitelist_and_rejects_bad_shape_and_global(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.pruned_init"
    _write_init(valid, dimension=47)
    evidence = dl.audit_libero_init_archive(
        valid, task_id=0, task_name="task", expected_dimension=47
    )
    assert evidence.shape == (50, 47)
    short = tmp_path / "short.pruned_init"
    _write_init(short, rows=49, dimension=47)
    with pytest.raises(dl.AvailabilityAuditError):
        dl.audit_libero_init_archive(
            short, task_id=0, task_name="task", expected_dimension=47
        )
    malicious = tmp_path / "malicious.pruned_init"
    _write_init(malicious, dimension=47, malicious=True)
    with pytest.raises(dl.AvailabilityAuditError, match="GLOBAL|opcode"):
        dl.audit_libero_init_archive(
            malicious, task_id=0, task_name="task", expected_dimension=47
        )
    many_ops = tmp_path / "many_ops.pruned_init"
    with zipfile.ZipFile(many_ops, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", b"\x80\x02" + b"N" * 100_000 + b".")
        archive.writestr("archive/version", b"3\n")
    with pytest.raises(dl.AvailabilityAuditError, match="opcode-count"):
        dl.audit_libero_init_archive(
            many_ops, task_id=0, task_name="task", expected_dimension=47
        )
    source_text = (MODULE_DIRECTORY / "data_lineage.py").read_text(encoding="utf-8")
    assert "pickle.load(" not in source_text
    assert "pickle.loads(" not in source_text
    assert "pickle.Unpickler" not in source_text


def test_all_ten_libero_long_tasks_have_fifty_states_statically(tmp_path: Path) -> None:
    task_map, init_root = _write_availability_fixture(tmp_path)
    result = dl.audit_libero_long_availability(task_map, init_root)
    assert result.passed
    assert len(result.tasks) == 10
    assert all(item.shape[0] == 50 for item in result.tasks)


def test_text_scan_records_hits_exclusions_and_forbidden_skips(tmp_path: Path) -> None:
    key = dl.EpisodeIdentity("libero_10", 0, 12).key
    (tmp_path / "history.md").write_text(key, encoding="utf-8")
    excluded = tmp_path / "candidate.json"
    excluded.write_text(key, encoding="utf-8")
    forbidden = tmp_path / "stage_c3.61" / "records.jsonl"
    forbidden.parent.mkdir()
    forbidden.write_text(key, encoding="utf-8")
    unregistered = tmp_path / "stage_c361" / "records.jsonl"
    unregistered.parent.mkdir()
    unregistered.write_text(key, encoding="utf-8")
    unregistered.unlink()
    ordinary_records = tmp_path / "ordinary" / "records.jsonl"
    ordinary_records.parent.mkdir()
    ordinary_records.write_text(key, encoding="utf-8")
    scan = dl.scan_text_corpus_for_exact_keys(
        tmp_path,
        [key],
        excluded_paths=[excluded],
        skipped_row_payload_paths=["stage_c3.61/records.jsonl"],
    )
    assert scan.hits == (
        {"key": key, "path": "history.md"},
        {"key": key, "path": "ordinary/records.jsonl"},
    )
    assert scan.skipped[0]["path"] == "stage_c3.61/records.jsonl"
    serialized = scan.to_dict()
    assert serialized["skipped_row_payload_count"] == 1
    assert serialized["skipped_symlink_count"] == 0
    assert serialized["unclassified_skip_count"] == 0
    unregistered.write_text(key, encoding="utf-8")
    with pytest.raises(dl.ForbiddenLineageSourceError, match="unregistered C3.61"):
        dl.scan_text_corpus_for_exact_keys(
            tmp_path,
            [key],
            excluded_paths=[excluded],
            skipped_row_payload_paths=["stage_c3.61/records.jsonl"],
        )


def test_cli_is_reproducible_stdlib_only_and_emits_expected_counts(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    _write_complete_request(metadata)
    task_map, init_root = _write_availability_fixture(tmp_path)
    manifest_path = REPO_ROOT / "docs/research/v3/legacy_evidence_manifest.json"
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    relocated_source = tmp_path / "source"
    original_source = REPO_ROOT.parents[1] / "source"
    for entry in manifest_value["evidence"]:
        source = original_source / entry["path"]
        destination = relocated_source / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    c361_rows = [
        "phase_route_v2_stage_c361_independent_aggregate_20260819_v1",
        "phase_route_v2_stage_c361_independent_candidate_shard00of04_gpu0_20260819_v1",
        "phase_route_v2_stage_c361_independent_candidate_shard01of04_gpu1_20260819_v1",
        "phase_route_v2_stage_c361_independent_candidate_shard02of04_gpu2_20260819_v1",
        "phase_route_v2_stage_c361_independent_candidate_shard03of04_gpu3_20260819_v1",
        "phase_route_v2_stage_c361_independent_context_20260819_v1",
        "phase_route_v2_stage_c361_independent_evaluation_20260819_v1",
    ]
    for directory in c361_rows:
        path = tmp_path / "source/reports" / directory / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("forbidden row payload\n", encoding="utf-8")
    link_target = (
        "/mnt/data2/zhangkaidong/datasets/vlabench/"
        "vlabench_primitive_ft_dataset/remote-home1/sdzhang/datasets/"
        "VLABench_release/primitive/add_condiment/add_condiment.py"
    )
    for relative in (
        "source/robot_experiments/vlabench/VLABench/add_condiment.py",
        "A1_source_backup_20260801/source/robot_experiments/vlabench/"
        "VLABench/add_condiment.py",
    ):
        link = tmp_path / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(link_target)
    output = tmp_path / "audit.json"
    script = REPO_ROOT / "scripts/dynamic_compute/v3/audit_v3_data_lineage.py"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "7"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metadata-root", str(metadata),
            "--request", "request.json",
            "--task-map", str(task_map),
            "--init-root", str(init_root),
            "--text-corpus-root", str(tmp_path),
            "--legacy-evidence-manifest", str(manifest_path),
            "--legacy-source-root", str(relocated_source),
            "--output", str(output),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_NO_KNOWN_HIT"
    assert result["counts"]["used_unique_keys"] == 604
    assert result["candidate_role_counts"] == {
        "development_v2": 180,
        "calibration_v2": 100,
        "independent_test_v2": 100,
    }
    assert len(result["input_selection_integrity"]["request_sha256"]) == 64
    assert len(result["input_selection_integrity"]["selection_bundle_sha256"]) == 64
    assert result["full_project_text_scan"]["skipped_row_payload_count"] == 7
    assert result["full_project_text_scan"]["skipped_symlink_count"] == 2
    assert result["full_project_text_scan"]["unclassified_skip_count"] == 0
    assert result["process_boundary"] == {
        "cuda_visible_devices": "",
        "forbidden_modules_loaded": [],
        "stdlib_only": True,
    }
    wrong_root_args = list(completed.args)
    root_index = wrong_root_args.index("--text-corpus-root") + 1
    wrong_root_args[root_index] = str(metadata)
    wrong_root_args[-1] = str(tmp_path / "wrong-root-output.json")
    wrong_root = subprocess.run(
        wrong_root_args,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_root.returncode == 2
    assert "parent of legacy-source-root" in wrong_root.stderr
    excluded_corpus_args = list(completed.args)
    excluded_corpus_args[-1] = str(tmp_path / "excluded-corpus-output.json")
    excluded_corpus_args.extend(["--exclude-text-path", str(tmp_path)])
    excluded_corpus = subprocess.run(
        excluded_corpus_args,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert excluded_corpus.returncode == 2
    assert "excluding the entire text corpus is forbidden" in excluded_corpus.stderr

    metadata_alias = tmp_path / "metadata_alias"
    metadata_alias.symlink_to(metadata, target_is_directory=True)
    symlink_root_args = list(completed.args)
    metadata_index = symlink_root_args.index("--metadata-root") + 1
    symlink_root_args[metadata_index] = str(metadata_alias)
    symlink_root_args[-1] = str(tmp_path / "symlink-root-output.json")
    symlink_root = subprocess.run(
        symlink_root_args,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert symlink_root.returncode == 2
    assert "symlink component" in symlink_root.stderr
    repeated = subprocess.run(
        completed.args,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "refusing to overwrite" in repeated.stderr

    blocked_output = tmp_path / "blocked.json"
    blocked_output.with_name("blocked.json.incomplete").write_text(
        "reserved", encoding="utf-8"
    )
    blocked_args = list(completed.args)
    blocked_args[-1] = str(blocked_output)
    blocked = subprocess.run(
        blocked_args,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 2
    assert "incomplete audit output already exists" in blocked.stderr
