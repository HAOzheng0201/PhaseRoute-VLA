"""Assemble leakage-audited M4.25b temporal features from frozen caches."""

from __future__ import annotations

from collections import defaultdict, deque
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.phase_estimator import (  # noqa: E402
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.temporal_route_features import (  # noqa: E402
    M425B_FEATURE_SCHEMA_VERSION,
    canonical_teacher_route,
    parse_episode_index,
    right_aligned_history,
)
from scripts.dynamic_compute.collect_m425_causal_route_features import (  # noqa: E402
    EXPECTED_CACHE_SCHEMA,
    FEATURE_SCHEMA,
    canonical_identity,
    identity_sha256,
)


EXPECTED_HIDDEN_SCOPE = "m425_causal_route_feature_shard"
EXPECTED_PHASE_SCHEMA = "phase-route-vla.phase-estimator-checkpoint.v1"
EXPECTED_EPISODES = tuple(range(6))
ROLE_BY_EPISODE = {
    0: "development",
    1: "development",
    2: "development",
    3: "calibration",
    4: "test",
    5: "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-checkpoint-sha256", required=True)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--expected-seed", type=int, default=20260826)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_hidden_features(
    paths: list[Path], checkpoint_sha256: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if len(paths) != 4:
        raise ValueError("M4.25b requires exactly four hidden feature shards")
    hidden_by_identity: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    seen_tasks: set[int] = set()
    for source in paths:
        path = source.resolve()
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS"
            or result.get("scope") != EXPECTED_HIDDEN_SCOPE
            or result.get("schema_version") != FEATURE_SCHEMA
        ):
            raise ValueError(f"invalid hidden feature result: {path}")
        if result.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"hidden feature checkpoint differs: {path}")
        if not all(bool(value) for value in result.get("local_checks", {}).values()):
            raise ValueError(f"hidden feature checks failed: {path}")
        arrays_path = Path(result["arrays_path"])
        if sha256_file(arrays_path) != result.get("arrays_sha256"):
            raise ValueError(f"hidden feature array SHA differs: {path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            layer11 = arrays["layer11_hidden"].copy()
            layer13 = arrays["layer13_hidden"].copy()
            hashes = arrays["identity_sha256"].copy()
            task_ids = arrays["task_id"].copy()
        rows = result.get("rows", [])
        count = int(result.get("records", -1))
        if not (
            layer11.shape[0]
            == layer13.shape[0]
            == hashes.shape[0]
            == task_ids.shape[0]
            == len(rows)
            == count
        ):
            raise ValueError(f"hidden feature row count differs: {path}")
        shard_tasks = {int(value) for value in task_ids.tolist()}
        if shard_tasks & seen_tasks:
            raise ValueError("hidden feature shards overlap task IDs")
        seen_tasks.update(shard_tasks)
        for index, row in enumerate(rows):
            key = hashes[index].decode("ascii")
            if key != str(row["identity_sha256"]) or key in hidden_by_identity:
                raise ValueError("hidden feature identities differ or repeat")
            hidden_by_identity[key] = {
                "layer11": layer11[index],
                "layer13": layer13[index],
                "row": row,
            }
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "arrays_path": str(arrays_path.resolve()),
                "arrays_sha256": str(result["arrays_sha256"]),
                "records": count,
                "tasks": sorted(shard_tasks),
                "gpu_audit": {
                    "physical_gpu_index": result["physical_gpu_index"],
                    "host_uuid": result["physical_gpu_uuid_nvidia_smi"],
                    "visible_uuid": result["physical_gpu_uuid_visible"],
                },
            }
        )
    if seen_tasks != set(range(10)):
        raise ValueError(f"hidden feature task grid differs: {sorted(seen_tasks)}")
    return hidden_by_identity, sources


def load_cache_entries(
    paths: list[Path],
    checkpoint_sha256: str,
    expected_seed: int,
    expected_episodes: int = 6,
) -> tuple[list[tuple[Path, dict[str, Any], str]], list[dict[str, Any]]]:
    if len(paths) != 10:
        raise ValueError("M4.25b requires exactly ten teacher cache directories")
    entries: list[tuple[Path, dict[str, Any], str]] = []
    sources: list[dict[str, Any]] = []
    seen_tasks: set[int] = set()
    for source in paths:
        cache_dir = source.resolve()
        manifest = cache_dir / "manifest.jsonl"
        result_path = cache_dir.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS"
            or result.get("checkpoint_sha256") != checkpoint_sha256
            or int(result.get("seed", -1)) != expected_seed
            or int(result.get("requested_episodes", -1)) != expected_episodes
            or int(result.get("completed_episodes", -1)) != expected_episodes
            or not bool((result.get("gpu_audit") or {}).get("mapping_verified"))
        ):
            raise ValueError(f"teacher cache result failed frozen checks: {result_path}")
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        task_ids = {int(row["task_id"]) for row in rows}
        if len(task_ids) != 1:
            raise ValueError(f"teacher cache mixes task IDs: {manifest}")
        task_id = next(iter(task_ids))
        if task_id in seen_tasks:
            raise ValueError(f"duplicate teacher cache task{task_id}")
        seen_tasks.add(task_id)
        episode_indices = {parse_episode_index(row["episode_id"]) for row in rows}
        if episode_indices != set(range(expected_episodes)):
            raise ValueError(f"task{task_id} episode grid differs: {episode_indices}")
        for row in rows:
            if (
                row.get("schema_version") != EXPECTED_CACHE_SCHEMA
                or row.get("checkpoint_sha256") != checkpoint_sha256
            ):
                raise ValueError(f"invalid teacher row in {manifest}")
            canonical_teacher_route(int(row["teacher_exit_layer"]))
            identity = canonical_identity(cache_dir, row)
            entries.append((cache_dir, row, identity_sha256(identity)))
        sources.append(
            {
                "task_id": task_id,
                "result_path": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path),
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "records": len(rows),
                "successes": int(result["successes"]),
                "gpu_audit": result["gpu_audit"],
            }
        )
    if seen_tasks != set(range(10)):
        raise ValueError(f"teacher cache task grid differs: {sorted(seen_tasks)}")
    entries.sort(
        key=lambda item: (
            int(item[1]["task_id"]),
            parse_episode_index(item[1]["episode_id"]),
            int(item[1]["step_id"]),
            str(item[1]["array_path"]),
        )
    )
    return entries, sorted(sources, key=lambda item: int(item["task_id"]))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.history_len != 8 or args.expected_seed != 20260826:
        raise ValueError("M4.25b freezes history_len=8 and seed=20260826")
    phase_path = args.phase_checkpoint.resolve()
    if sha256_file(phase_path) != args.phase_checkpoint_sha256:
        raise ValueError("PhaseStateEstimator SHA-256 differs")
    phase_checkpoint = torch.load(phase_path, map_location="cpu", weights_only=True)
    if phase_checkpoint.get("schema_version") != EXPECTED_PHASE_SCHEMA:
        raise ValueError("unexpected PhaseStateEstimator schema")
    phase_config = PhaseEstimatorConfig(**phase_checkpoint["model_config"])
    phase_estimator = PhaseStateEstimator(phase_config)
    phase_estimator.load_state_dict(phase_checkpoint["model_state_dict"])
    phase_estimator.eval()

    hidden, hidden_sources = load_hidden_features(
        args.feature_result, args.checkpoint_sha256
    )
    entries, cache_sources = load_cache_entries(
        args.cache_dir, args.checkpoint_sha256, args.expected_seed
    )
    entry_hashes = [identity_hash for _, _, identity_hash in entries]
    if len(set(entry_hashes)) != len(entry_hashes) or set(entry_hashes) != set(hidden):
        raise ValueError("teacher cache and hidden feature identities do not match exactly")

    histories: dict[str, deque[tuple[np.ndarray, np.ndarray]]] = defaultdict(
        lambda: deque(maxlen=args.history_len)
    )
    call_indices: dict[str, int] = defaultdict(int)
    arrays: dict[str, list[np.ndarray | int | float | bytes]] = defaultdict(list)
    with torch.inference_mode():
        for index, (cache_dir, record, identity_hash) in enumerate(entries):
            episode_id = str(record["episode_id"])
            with np.load(cache_dir / str(record["array_path"]), allow_pickle=False) as shard:
                projected = shard["projected_features"].astype(np.float32)
                positions = shard["image_input_idx"]
                valid = positions >= 0
                if not valid.any():
                    raise ValueError("cache row has no valid projected visual feature")
                visual = projected[valid].mean(axis=0)
                instruction = shard["instruction_summary"].astype(np.float32)
                current_proprio = shard["normalized_proprio"].astype(np.float32)
                current_action = shard["teacher_normalized_action"].astype(np.float32)
            proprio_history, action_history, history_mask = right_aligned_history(
                list(histories[episode_id]),
                history_len=args.history_len,
                proprio_dim=phase_config.proprio_dim,
                action_horizon=phase_config.action_horizon,
                action_dim=phase_config.action_dim,
            )
            state = phase_estimator(
                visual_summary=torch.from_numpy(visual[None]),
                instruction_summary=torch.from_numpy(instruction[None]),
                current_proprio=torch.from_numpy(current_proprio[None]),
                proprio_history=torch.from_numpy(proprio_history[None]),
                proprio_history_mask=torch.from_numpy(history_mask[None]),
                action_history=torch.from_numpy(action_history[None]),
                action_history_mask=torch.from_numpy(history_mask[None]),
            )
            phase_scalars = np.asarray(
                [
                    float(state.progress[0, 0]),
                    float(state.boundary_prob[0, 0]),
                    float(state.uncertainty[0, 0]),
                ],
                dtype=np.float32,
            )
            arrays["layer11_hidden"].append(hidden[identity_hash]["layer11"])
            arrays["layer13_hidden"].append(hidden[identity_hash]["layer13"])
            arrays["current_proprio"].append(current_proprio)
            arrays["proprio_history"].append(proprio_history)
            arrays["action_history"].append(action_history)
            arrays["history_mask"].append(history_mask)
            arrays["phase_stage"].append(
                state.stage_embedding[0].detach().float().numpy()
            )
            arrays["phase_scalars"].append(phase_scalars)
            arrays["step_feature"].append(
                min(float(record["step_id"]), 250.0) / 250.0
            )
            arrays["task_id"].append(int(record["task_id"]))
            arrays["episode_index"].append(parse_episode_index(episode_id))
            arrays["step_id"].append(int(record["step_id"]))
            arrays["call_index"].append(call_indices[episode_id])
            arrays["teacher_route"].append(
                canonical_teacher_route(int(record["teacher_exit_layer"]))
            )
            arrays["identity_sha256"].append(identity_hash.encode("ascii"))
            histories[episode_id].append(
                (current_proprio.copy(), current_action.copy())
            )
            call_indices[episode_id] += 1
            print(
                f"[{index + 1:04d}/{len(entries):04d}] task={record['task_id']} "
                f"episode={parse_episode_index(episode_id)} step={record['step_id']} "
                f"history={int(history_mask.sum())} "
                f"raw_exit={record['teacher_exit_layer']} "
                f"route={canonical_teacher_route(int(record['teacher_exit_layer']))}",
                flush=True,
            )

    output_arrays = {
        "layer11_hidden": np.stack(arrays["layer11_hidden"]).astype(np.float16),
        "layer13_hidden": np.stack(arrays["layer13_hidden"]).astype(np.float16),
        "current_proprio": np.stack(arrays["current_proprio"]).astype(np.float32),
        "proprio_history": np.stack(arrays["proprio_history"]).astype(np.float32),
        "action_history": np.stack(arrays["action_history"]).astype(np.float32),
        "history_mask": np.stack(arrays["history_mask"]).astype(np.bool_),
        "phase_stage": np.stack(arrays["phase_stage"]).astype(np.float32),
        "phase_scalars": np.stack(arrays["phase_scalars"]).astype(np.float32),
        "step_feature": np.asarray(arrays["step_feature"], dtype=np.float32),
        "task_id": np.asarray(arrays["task_id"], dtype=np.int16),
        "episode_index": np.asarray(arrays["episode_index"], dtype=np.int8),
        "step_id": np.asarray(arrays["step_id"], dtype=np.int32),
        "call_index": np.asarray(arrays["call_index"], dtype=np.int16),
        "teacher_route": np.asarray(arrays["teacher_route"], dtype=np.int16),
        "identity_sha256": np.asarray(arrays["identity_sha256"], dtype="S64"),
    }
    row_count = len(entries)
    finite_names = (
        "layer11_hidden",
        "layer13_hidden",
        "current_proprio",
        "proprio_history",
        "action_history",
        "phase_stage",
        "phase_scalars",
        "step_feature",
    )
    local_checks = {
        "aligned_rows": all(value.shape[0] == row_count for value in output_arrays.values()),
        "unique_identity": len(set(entry_hashes)) == row_count,
        "all_finite": all(np.isfinite(output_arrays[name]).all() for name in finite_names),
        "task_grid": set(output_arrays["task_id"].tolist()) == set(range(10)),
        "episode_grid": set(output_arrays["episode_index"].tolist()) == set(EXPECTED_EPISODES),
        "first_calls_have_empty_history": all(
            not output_arrays["history_mask"][index].any()
            for index in np.flatnonzero(output_arrays["call_index"] == 0)
        ),
        "history_masks_aligned": bool(
            np.all(
                (np.abs(output_arrays["proprio_history"]).sum(axis=-1) > 0)
                <= output_arrays["history_mask"]
            )
        ),
    }
    role_summaries = {}
    for role in ("development", "calibration", "test"):
        role_episodes = [index for index, value in ROLE_BY_EPISODE.items() if value == role]
        mask = np.isin(output_arrays["episode_index"], role_episodes)
        role_summaries[role] = {
            "episode_indices": role_episodes,
            "rows": int(mask.sum()),
            "teacher_distribution": {
                str(route): int(np.sum(output_arrays["teacher_route"][mask] == route))
                for route in (11, 13, 27)
            },
            "raw_exit_distribution": {
                str(raw): sum(
                    int(record["teacher_exit_layer"]) == raw
                    for _, record, _ in entries
                    if parse_episode_index(str(record["episode_id"])) in role_episodes
                )
                for raw in sorted(
                    {
                        int(record["teacher_exit_layer"])
                        for _, record, _ in entries
                        if parse_episode_index(str(record["episode_id"])) in role_episodes
                    }
                )
            },
        }
    calibration_test_27 = (
        role_summaries["calibration"]["teacher_distribution"]["27"]
        + role_summaries["test"]["teacher_distribution"]["27"]
    )
    data_sufficient = calibration_test_27 >= 10
    status = "PASS" if all(local_checks.values()) and data_sufficient else "INSUFFICIENT_DATA"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "features.npz"
    np.savez_compressed(arrays_path, **output_arrays)
    result = {
        "status": status,
        "scope": "m425b_temporal_route_feature_table",
        "schema_version": M425B_FEATURE_SCHEMA_VERSION,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint": str(phase_path),
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "history_len": args.history_len,
        "expected_seed": args.expected_seed,
        "teacher_route_mapping": {
            "kind": "minimum_non_shallower_m425b_route",
            "raw_a1_exit_layers": list(range(1, 28, 2)),
            "m425b_route_layers": [11, 13, 27],
            "rule": "raw<=11 -> 11; raw<=13 -> 13; raw>=15 -> 27",
        },
        "records": row_count,
        "episodes": 60,
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
        "hidden_feature_sources": hidden_sources,
        "teacher_cache_sources": cache_sources,
        "local_checks": local_checks,
        "role_summaries": role_summaries,
        "calibration_plus_test_route27": calibration_test_27,
        "data_sufficient": data_sufficient,
        "feature_shapes": {
            name: list(value.shape) for name, value in output_arrays.items()
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
