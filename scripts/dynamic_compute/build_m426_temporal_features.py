"""Build leakage-audited M4.26 route13/27 features from new frozen caches."""

from __future__ import annotations

from collections import defaultdict, deque
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.phase_estimator import (  # noqa: E402
    PhaseEstimatorConfig,
    PhaseStateEstimator,
)
from a1.vla.dynamic_compute.risk_route13_router import (  # noqa: E402
    M426A_FEATURE_SCHEMA_VERSION,
    M426_FEATURE_SCHEMA_VERSION,
    M427_FEATURE_SCHEMA_VERSION,
    M428_FEATURE_SCHEMA_VERSION,
)
from a1.vla.dynamic_compute.temporal_route_features import (  # noqa: E402
    canonical_teacher_route,
    parse_episode_index,
    right_aligned_history,
)
from scripts.dynamic_compute.build_m425b_temporal_features import (  # noqa: E402
    EXPECTED_PHASE_SCHEMA,
    git_output,
    load_cache_entries,
    load_hidden_features,
    sha256_file,
)


EXPECTED_EPISODES = tuple(range(6))
EXPECTED_SEED = 20260926
ROLE_BY_EPISODE = {
    0: "development",
    1: "development",
    2: "development",
    3: "calibration",
    4: "test",
    5: "test",
}
M426A_EXPECTED_EPISODES = tuple(range(7))
M426A_EXPECTED_SEED = 20261026
M426A_ROLE_BY_EPISODE = {
    0: "development",
    1: "development",
    2: "development",
    3: "calibration",
    4: "calibration",
    5: "test",
    6: "test",
}
M427_EXPECTED_EPISODES = tuple(range(15))
M427_EXPECTED_SEED = 20261127
M427_ROLE_BY_EPISODE = {
    **{index: "development" for index in range(5)},
    **{index: "calibration" for index in range(5, 10)},
    **{index: "test" for index in range(10, 15)},
}
M428_EXPECTED_EPISODES = tuple(range(30))
M428_EXPECTED_SEED = 20261228
M428_ROLE_BY_EPISODE = {
    **{index: "development" for index in range(10)},
    **{index: "calibration" for index in range(10, 20)},
    **{index: "test" for index in range(20, 30)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-checkpoint-sha256", required=True)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--expected-seed", type=int, default=EXPECTED_SEED)
    parser.add_argument(
        "--protocol", choices=("m426", "m426a", "m427", "m428"), default="m426"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def m427_data_sufficient(
    route27_by_role: dict[str, int],
    route27_positive_groups_by_role: dict[str, int],
) -> bool:
    """Frozen M4.27 nonsealed data gate; sealed counts are intentionally absent."""
    return (
        set(route27_by_role) == {"development", "calibration"}
        and set(route27_positive_groups_by_role) == {"development", "calibration"}
        and route27_by_role["development"] >= 30
        and route27_by_role["calibration"] >= 30
        and route27_positive_groups_by_role["development"] >= 19
        and route27_positive_groups_by_role["calibration"] >= 19
    )


def m428_data_sufficient(
    route27_by_role: dict[str, int],
    route27_positive_groups_by_role: dict[str, int],
) -> bool:
    """Frozen M4.28 nonsealed gate with a 10% zero-error group-risk target."""
    return (
        set(route27_by_role) == {"development", "calibration"}
        and set(route27_positive_groups_by_role) == {"development", "calibration"}
        and route27_by_role["development"] >= 30
        and route27_by_role["calibration"] >= 30
        and route27_positive_groups_by_role["development"] >= 29
        and route27_positive_groups_by_role["calibration"] >= 29
    )


def format_progress_line(
    *,
    protocol: str,
    role: str,
    index: int,
    total: int,
    task_id: int,
    episode_index: int,
    step_id: int,
    history_count: int,
    raw_exit: int,
    route: int,
) -> str:
    prefix = (
        f"[{index:04d}/{total:04d}] task={task_id} "
        f"episode={episode_index} step={step_id} history={history_count}"
    )
    if protocol in ("m427", "m428") and role == "test":
        return f"{prefix} raw_exit=REDACTED binary_target=REDACTED"
    return f"{prefix} raw_exit={raw_exit} binary_target={'13' if route <= 13 else '27'}"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.protocol == "m426":
        expected_episodes = EXPECTED_EPISODES
        expected_seed = EXPECTED_SEED
        role_by_episode = ROLE_BY_EPISODE
        feature_schema = M426_FEATURE_SCHEMA_VERSION
        scope = "m426_temporal_route_feature_table"
    elif args.protocol == "m426a":
        expected_episodes = M426A_EXPECTED_EPISODES
        expected_seed = M426A_EXPECTED_SEED
        role_by_episode = M426A_ROLE_BY_EPISODE
        feature_schema = M426A_FEATURE_SCHEMA_VERSION
        scope = "m426a_temporal_route_feature_table"
    elif args.protocol == "m427":
        expected_episodes = M427_EXPECTED_EPISODES
        expected_seed = M427_EXPECTED_SEED
        role_by_episode = M427_ROLE_BY_EPISODE
        feature_schema = M427_FEATURE_SCHEMA_VERSION
        scope = "m427_temporal_route_feature_table"
    else:
        expected_episodes = M428_EXPECTED_EPISODES
        expected_seed = M428_EXPECTED_SEED
        role_by_episode = M428_ROLE_BY_EPISODE
        feature_schema = M428_FEATURE_SCHEMA_VERSION
        scope = "m428_temporal_route_feature_table"
    if args.history_len != 8 or args.expected_seed != expected_seed:
        raise ValueError(
            f"{args.protocol} freezes history_len=8 and seed={expected_seed}"
        )
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
        args.cache_dir,
        args.checkpoint_sha256,
        args.expected_seed,
        expected_episodes=len(expected_episodes),
    )
    entry_hashes = [identity_hash for _, _, identity_hash in entries]
    if len(set(entry_hashes)) != len(entry_hashes) or set(entry_hashes) != set(hidden):
        raise ValueError("teacher cache and hidden feature identities do not match exactly")

    histories: dict[str, deque[tuple[np.ndarray, np.ndarray]]] = defaultdict(
        lambda: deque(maxlen=args.history_len)
    )
    call_indices: dict[str, int] = defaultdict(int)
    values: dict[str, list[np.ndarray | int | float | bytes]] = defaultdict(list)
    raw_exit_by_identity: dict[str, int] = {}
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
            raw_exit = int(record["teacher_exit_layer"])
            route = canonical_teacher_route(raw_exit)
            raw_exit_by_identity[identity_hash] = raw_exit
            values["layer13_hidden"].append(hidden[identity_hash]["layer13"])
            values["current_proprio"].append(current_proprio)
            values["proprio_history"].append(proprio_history)
            values["action_history"].append(action_history)
            values["history_mask"].append(history_mask)
            values["phase_stage"].append(
                state.stage_embedding[0].detach().float().numpy()
            )
            values["phase_scalars"].append(phase_scalars)
            values["step_feature"].append(min(float(record["step_id"]), 250.0) / 250.0)
            values["task_id"].append(int(record["task_id"]))
            episode_index = parse_episode_index(episode_id)
            values["episode_index"].append(episode_index)
            values["step_id"].append(int(record["step_id"]))
            values["call_index"].append(call_indices[episode_id])
            values["teacher_route"].append(route)
            values["identity_sha256"].append(identity_hash.encode("ascii"))
            histories[episode_id].append((current_proprio.copy(), current_action.copy()))
            call_indices[episode_id] += 1
            print(
                format_progress_line(
                    protocol=args.protocol,
                    role=role_by_episode[episode_index],
                    index=index + 1,
                    total=len(entries),
                    task_id=int(record["task_id"]),
                    episode_index=episode_index,
                    step_id=int(record["step_id"]),
                    history_count=int(history_mask.sum()),
                    raw_exit=raw_exit,
                    route=route,
                ),
                flush=True,
            )

    arrays = {
        "layer13_hidden": np.stack(values["layer13_hidden"]).astype(np.float16),
        "current_proprio": np.stack(values["current_proprio"]).astype(np.float32),
        "proprio_history": np.stack(values["proprio_history"]).astype(np.float32),
        "action_history": np.stack(values["action_history"]).astype(np.float32),
        "history_mask": np.stack(values["history_mask"]).astype(np.bool_),
        "phase_stage": np.stack(values["phase_stage"]).astype(np.float32),
        "phase_scalars": np.stack(values["phase_scalars"]).astype(np.float32),
        "step_feature": np.asarray(values["step_feature"], dtype=np.float32),
        "task_id": np.asarray(values["task_id"], dtype=np.int16),
        "episode_index": np.asarray(values["episode_index"], dtype=np.int8),
        "step_id": np.asarray(values["step_id"], dtype=np.int32),
        "call_index": np.asarray(values["call_index"], dtype=np.int16),
        "teacher_route": np.asarray(values["teacher_route"], dtype=np.int16),
        "identity_sha256": np.asarray(values["identity_sha256"], dtype="S64"),
    }
    row_count = len(entries)
    finite_names = (
        "layer13_hidden",
        "current_proprio",
        "proprio_history",
        "action_history",
        "phase_stage",
        "phase_scalars",
        "step_feature",
    )
    local_checks = {
        "aligned_rows": all(value.shape[0] == row_count for value in arrays.values()),
        "unique_identity": len(set(entry_hashes)) == row_count,
        "all_finite": all(np.isfinite(arrays[name]).all() for name in finite_names),
        "task_grid": set(arrays["task_id"].tolist()) == set(range(10)),
        "episode_grid": set(arrays["episode_index"].tolist())
        == set(expected_episodes),
        "first_calls_have_empty_history": all(
            not arrays["history_mask"][index].any()
            for index in np.flatnonzero(arrays["call_index"] == 0)
        ),
        "route_domain": set(np.unique(arrays["teacher_route"]).tolist()).issubset(
            {11, 13, 27}
        ),
    }
    role_summaries = {}
    for role in ("development", "calibration", "test"):
        role_episodes = [
            index for index, value in role_by_episode.items() if value == role
        ]
        mask = np.isin(arrays["episode_index"], role_episodes)
        role_hashes = arrays["identity_sha256"][mask]
        summary = {
            "episode_indices": role_episodes,
            "rows": int(mask.sum()),
            "teacher_distribution": {
                str(route): int(np.sum(arrays["teacher_route"][mask] == route))
                for route in (11, 13, 27)
            },
            "binary_distribution": {
                "13": int(np.sum(arrays["teacher_route"][mask] <= 13)),
                "27": int(np.sum(arrays["teacher_route"][mask] == 27)),
            },
            "raw_exit_distribution": {
                str(raw): int(
                    sum(
                        raw_exit_by_identity[item.decode("ascii")] == raw
                        for item in role_hashes
                    )
                )
                for raw in sorted(
                    {
                        raw_exit_by_identity[item.decode("ascii")]
                        for item in role_hashes
                    }
                )
            },
        }
        if args.protocol in ("m427", "m428") and role == "test":
            summary = {
                "episode_indices": role_episodes,
                "rows": int(mask.sum()),
                "sealed": True,
            }
        role_summaries[role] = summary
    route27_by_role = {
        role: int(summary["binary_distribution"]["27"])
        for role, summary in role_summaries.items()
        if "binary_distribution" in summary
    }
    route27_positive_groups_by_role = {}
    for role in ("development", "calibration"):
        role_episodes = role_summaries[role]["episode_indices"]
        groups = 0
        for task_id in range(10):
            for episode_index in role_episodes:
                mask = (
                    (arrays["task_id"] == task_id)
                    & (arrays["episode_index"] == episode_index)
                )
                groups += int(np.any(arrays["teacher_route"][mask] == 27))
        route27_positive_groups_by_role[role] = groups
    if args.protocol == "m427":
        data_sufficient = m427_data_sufficient(
            route27_by_role, route27_positive_groups_by_role
        )
    elif args.protocol == "m428":
        data_sufficient = m428_data_sufficient(
            route27_by_role, route27_positive_groups_by_role
        )
    else:
        data_sufficient = (
            all(count > 0 for count in route27_by_role.values())
            and route27_by_role["calibration"] + route27_by_role["test"] >= 10
        )
    status = "PASS" if all(local_checks.values()) and data_sufficient else "INSUFFICIENT_DATA"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = args.output_dir / "features.npz"
    np.savez_compressed(arrays_path, **arrays)
    result = {
        "status": status,
        "scope": scope,
        "schema_version": feature_schema,
        "protocol": args.protocol,
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(git_output("status", "--porcelain")),
        "checkpoint_sha256": args.checkpoint_sha256,
        "phase_checkpoint": str(phase_path),
        "phase_checkpoint_sha256": args.phase_checkpoint_sha256,
        "history_len": args.history_len,
        "expected_seed": args.expected_seed,
        "teacher_route_mapping": {
            "raw_a1_exit_layers": list(range(1, 28, 2)),
            "canonical_routes": [11, 13, 27],
            "binary_rule": "raw<=13 -> route13; raw>=15 -> route27",
        },
        "records": row_count,
        "episodes": 10 * len(expected_episodes),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
        "hidden_feature_sources": hidden_sources,
        "teacher_cache_sources": cache_sources,
        "local_checks": local_checks,
        "role_summaries": role_summaries,
        "route27_by_role": route27_by_role,
        "route27_positive_groups_by_role": route27_positive_groups_by_role,
        "data_sufficient": data_sufficient,
        "feature_shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    if args.protocol not in ("m427", "m428"):
        result["calibration_plus_test_route27"] = (
            route27_by_role["calibration"] + route27_by_role["test"]
        )
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
