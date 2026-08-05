"""Build an immutable, trainable M2 temporal dataset from four-card collection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.phase_cache import PHASE_CACHE_SCHEMA_VERSION
from a1.vla.dynamic_compute.phase_dataset import (
    PhaseDatasetConfig,
    build_phase_dataset_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-len", type=int, default=8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_phase_calls(manifest_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase_dir = manifest_path.parent
    records = []
    shard_sources = []
    declared_shards = set()
    visual_token_counts = []
    instruction_token_counts = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        manifest = json.loads(line)
        if manifest.get("schema_version") != PHASE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unexpected phase cache schema in {manifest_path}: "
                f"{manifest.get('schema_version')}"
            )
        shard_path = phase_dir / manifest["array_path"]
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        resolved_shard = str(shard_path.resolve())
        if resolved_shard in declared_shards:
            raise ValueError(
                f"Duplicate phase-cache shard in {manifest_path}: {shard_path}"
            )
        declared_shards.add(resolved_shard)
        summary_counts = manifest.get("summary_counts", {})
        visual_tokens = int(summary_counts.get("visual_tokens", 0))
        instruction_tokens = int(summary_counts.get("instruction_tokens", 0))
        if visual_tokens < 1 or instruction_tokens < 1:
            raise ValueError(f"Invalid summary token counts in {manifest_path}")
        visual_token_counts.append(visual_tokens)
        instruction_token_counts.append(instruction_tokens)
        with np.load(shard_path) as shard:
            call = {
                "episode_id": manifest["episode_id"],
                "step_id": int(manifest["step_id"]),
                "task_id": manifest.get("task_id"),
                **{name: shard[name].copy() for name in shard.files},
            }
        records.append(call)
        shard_sources.append(
            {
                "path": str(shard_path),
                "bytes": shard_path.stat().st_size,
                "sha256": file_sha256(shard_path),
            }
        )
    if not records:
        raise ValueError(f"Empty phase-cache manifest: {manifest_path}")
    actual_shards = {
        str(path.resolve()) for path in phase_dir.glob("arrays/*.npz")
    }
    orphan_shards = sorted(actual_shards - declared_shards)
    if orphan_shards:
        raise ValueError(
            f"Phase-cache directory contains {len(orphan_shards)} orphan shard(s): "
            f"{orphan_shards[:3]}"
        )
    return records, {
        "manifest_path": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": file_sha256(manifest_path),
        "records": len(records),
        "missing_shards": 0,
        "orphan_shards": 0,
        "summary_counts": {
            "visual_tokens_min": min(visual_token_counts),
            "visual_tokens_max": max(visual_token_counts),
            "visual_tokens_unique": sorted(set(visual_token_counts)),
            "instruction_tokens_min": min(instruction_token_counts),
            "instruction_tokens_max": max(instruction_token_counts),
            "instruction_tokens_unique": sorted(set(instruction_token_counts)),
        },
        "shards": shard_sources,
    }


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / "phase_dataset.npz"
    metadata_path = args.output_dir / "metadata.json"
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset in {args.output_dir}")

    task_dirs = sorted(args.collection_root.glob("gpu*_task*"))
    if not task_dirs:
        raise FileNotFoundError(f"No gpu*_task* directories under {args.collection_root}")
    phase_calls: List[Dict[str, Any]] = []
    telemetry_records: List[Dict[str, Any]] = []
    source_runs = []
    phase_sources = []
    telemetry_sources = []
    for task_dir in task_dirs:
        result_path = task_dir / "result.json"
        telemetry_path = task_dir / "policy_calls.jsonl"
        manifest_path = task_dir / "phase_calls" / "manifest.jsonl"
        if not result_path.is_file() or not telemetry_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete M2 task output in {task_dir}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            raise ValueError(f"Source run did not pass: {result_path}")
        source_runs.append(
            {
                "path": str(result_path),
                "sha256": file_sha256(result_path),
                "result": result,
            }
        )
        task_phase_calls, task_phase_source = load_phase_calls(manifest_path)
        phase_calls.extend(task_phase_calls)
        phase_sources.append(task_phase_source)
        task_telemetry = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        telemetry_records.extend(task_telemetry)
        telemetry_sources.append(
            {
                "path": str(telemetry_path),
                "bytes": telemetry_path.stat().st_size,
                "sha256": file_sha256(telemetry_path),
                "records": len(task_telemetry),
            }
        )

    config = PhaseDatasetConfig(history_len=args.history_len)
    arrays, dataset_metadata = build_phase_dataset_arrays(
        phase_calls,
        telemetry_records,
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    source_status = git_output("status", "--porcelain=v1")
    metadata = {
        **dataset_metadata,
        "dataset_path": str(output_path),
        "dataset_bytes": output_path.stat().st_size,
        "dataset_sha256": file_sha256(output_path),
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "all_arrays_finite": all(
            np.isfinite(array).all()
            for array in arrays.values()
            if np.issubdtype(array.dtype, np.number)
        ),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "phase_dataset_config": {
            "history_len": config.history_len,
            "split_seed": config.split_seed,
            "boundary_config": asdict(config.boundary_config),
        },
        "source_runs": source_runs,
        "phase_sources": phase_sources,
        "telemetry_sources": telemetry_sources,
        "summary_token_audit": {
            "visual_tokens_unique": sorted(
                {
                    count
                    for source in phase_sources
                    for count in source["summary_counts"]["visual_tokens_unique"]
                }
            ),
            "instruction_tokens_unique": sorted(
                {
                    count
                    for source in phase_sources
                    for count in source["summary_counts"]["instruction_tokens_unique"]
                }
            ),
        },
        "limitations": [
            "The current M2 collection covers four libero_spatial tasks only.",
            "Action/proprio histories use the policy-call timebase (one 8-step chunk).",
            "The estimator must not consume current_normalized_action_chunk or labels as inputs.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": metadata["records"],
                "episodes": metadata["episodes"],
                "split_records": metadata["split_records"],
                "split_episodes": metadata["split_episodes"],
                "dataset_sha256": metadata["dataset_sha256"],
                "dataset": str(output_path),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
