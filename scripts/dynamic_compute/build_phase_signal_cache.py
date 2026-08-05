"""Build a versioned, label-only M2 signal cache from M1 telemetry."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.weak_labels import (
    BoundaryLabelConfig,
    build_weak_labels,
)


PHASE_SIGNAL_CACHE_SCHEMA_VERSION = "phase-route-vla.phase-signal-cache.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_action_delta(record: dict):
    for layer, delta in zip(
        record["candidate_exit_layers"],
        record["action_delta_by_exit"],
    ):
        if layer == record["exit_layer"]:
            return delta
    return None


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def main() -> None:
    args = parse_args()
    if len(args.checkpoint_sha256) != 64:
        raise ValueError("checkpoint-sha256 must contain 64 hexadecimal characters")
    int(args.checkpoint_sha256, 16)

    telemetry_paths = sorted(args.telemetry_root.glob("gpu*_task*/policy_calls.jsonl"))
    if not telemetry_paths:
        raise FileNotFoundError(f"No policy_calls.jsonl under {args.telemetry_root}")
    records = []
    for path in telemetry_paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    # Main Gate-A labels deliberately exclude action_delta_increase to avoid
    # testing action delta against a label that was defined by action delta.
    config = BoundaryLabelConfig(
        weight_action_delta_increase=0.0,
        dilation_radius=2,
    )
    labels = build_weak_labels(records, config=config)
    records_by_key = {
        (str(record["episode_id"]), int(record["step_id"])): record
        for record in records
    }
    rows = []
    event_counts = Counter()
    for label in labels:
        record = records_by_key[(label.episode_id, label.environment_step_id)]
        for event, present in label.boundary_events.items():
            if present:
                event_counts[event] += 1
        rows.append(
            {
                **label.to_dict(),
                "source_telemetry_schema": record["schema_version"],
                "instruction_hash": record["instruction_hash"],
                "gripper_state": record["gripper_state"],
                "translation_speed": record["translation_speed"],
                "rotation_speed": record["rotation_speed"],
                "exit_layer": record["exit_layer"],
                "selected_action_delta": selected_action_delta(record),
                "active_tokens": record["active_tokens_by_layer"][0],
                "visual_tokens": record["extra"]["visual_tokens"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    if labels_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing cache in {args.output_dir}"
        )
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    checkpoint = args.checkpoint.resolve()
    config_path = checkpoint.parent / "config.yaml" if checkpoint.is_file() else checkpoint / "config.yaml"
    checkpoint_path = checkpoint if checkpoint.is_file() else checkpoint / "model.pt"
    source_status = git_output("status", "--porcelain=v1")
    metadata = {
        "schema_version": PHASE_SIGNAL_CACHE_SCHEMA_VERSION,
        "cache_kind": "label_only_preliminary_gate_a",
        "trainable_phase_cache": False,
        "records": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "raw_boundaries": sum(row["boundary_target_raw"] for row in rows),
        "dilated_boundaries": sum(row["boundary_target"] for row in rows),
        "event_counts": dict(sorted(event_counts.items())),
        "boundary_config": asdict(config),
        "policy_call_timebase": True,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(source_status),
        "source_status_sha256": hashlib.sha256(source_status.encode()).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_config_path": str(config_path),
        "model_config_sha256": file_sha256(config_path),
        "telemetry_sources": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in telemetry_paths
        ],
        "labels_sha256": file_sha256(labels_path),
        "available_inputs": {
            "scalar_proprio_summary": True,
            "gripper_state": True,
            "translation_speed": True,
            "rotation_speed": True,
            "exit_layer": True,
            "selected_action_delta": True,
            "raw_proprio": False,
            "raw_action_history": False,
            "visual_summary": False,
            "instruction_summary": False,
        },
        "limitations": [
            "This cache is for preliminary Gate-A analysis, not phase-estimator training.",
            "Progress uses policy-call index because the exact terminal environment step is not logged.",
            "Direction-change events are unavailable without raw previous action vectors.",
            "No visual or language summaries are present in M1 telemetry.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"labels={labels_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()

