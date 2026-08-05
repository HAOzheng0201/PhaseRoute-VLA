"""Replay M4.20 deterministic exit-depth hysteresis on frozen telemetry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.depth_hysteresis import (  # noqa: E402
    ExitDepthHysteresis,
    ExitDepthHysteresisConfig,
)


RELEASE_CANDIDATES = (2, 3, 4)
MAX_LATCHED_LAYER = 13
GATES = {
    "min_pingpong_reduction": 0.50,
    "min_11_13_switch_reduction": 0.35,
    "min_all_switch_reduction": 0.25,
    "max_mean_exit_layer_increase": 0.75,
    "max_final_layer_count_increase": 0,
    "max_estimated_fm_call_increase_ratio": 0.08,
    "max_shallower_than_raw": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m418-root",
        type=Path,
        default=Path(
            "reports/m418_persistent_early_exit_spatial_tasks0_9_3ep_20260803_v1"
        ),
    )
    parser.add_argument(
        "--m418b-root",
        type=Path,
        default=Path(
            "reports/m418b_targeted_task5_episodes3_26_20260803_v2/early_exit"
        ),
    )
    parser.add_argument(
        "--m419-root",
        type=Path,
        default=Path(
            "reports/m419_discordant_task5_ep2_14_22_3seeds_20260803_v1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(
    m418_root: Path,
    m418b_root: Path,
    m419_root: Path,
) -> list[tuple[str, Path]]:
    inputs = []
    inputs.extend(
        ("m418", path)
        for path in sorted(m418_root.glob("shard*/policy_calls.jsonl"))
    )
    inputs.extend(
        ("m418b", path)
        for path in sorted(m418b_root.glob("shard*/policy_calls.jsonl"))
    )
    inputs.extend(
        ("m419", path)
        for path in sorted(m419_root.glob("seed*/early_exit/policy_calls.jsonl"))
    )
    expected = Counter({"m418": 2, "m418b": 2, "m419": 3})
    actual = Counter(source for source, _ in inputs)
    if actual != expected:
        raise ValueError(f"unexpected M4.20 telemetry inputs: {dict(actual)}")
    if any(not path.is_file() for _, path in inputs):
        raise FileNotFoundError("one or more telemetry inputs are missing")
    return inputs


def _episode_index(record: dict[str, Any]) -> int:
    suffix = str(record["episode_id"]).rsplit("episode", 1)[-1]
    try:
        return int(suffix)
    except ValueError as error:
        raise ValueError(f"invalid episode_id: {record['episode_id']}") from error


def split_name(source: str, task_id: int, episode_idx: int) -> str:
    if source == "m418" and 0 <= task_id <= 4:
        return "calibration"
    if source == "m418b" and task_id == 5 and 3 <= episode_idx <= 13:
        return "calibration"
    if source == "m418" and 6 <= task_id <= 9:
        return "offline_held_out"
    if source == "m418b" and task_id == 5 and 15 <= episode_idx <= 26:
        return "offline_held_out"
    return "secondary_audit"


def load_episode_sequences(
    inputs: Sequence[tuple[str, Path]],
) -> dict[str, list[dict[str, Any]]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_calls: set[tuple[str, str, int]] = set()
    for source, path in inputs:
        run_identity = str(path.resolve())
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            episode_id = str(record["episode_id"])
            step_id = int(record["step_id"])
            identity = (run_identity, episode_id, step_id)
            if identity in seen_calls:
                raise ValueError(f"duplicate policy call {identity}")
            seen_calls.add(identity)
            task_id = int(record["task_id"])
            episode_idx = _episode_index(record)
            layers = tuple(int(item) for item in record["candidate_exit_layers"])
            raw_layer = int(record["exit_layer"])
            if tuple(sorted(set(layers))) != layers or raw_layer not in layers:
                raise ValueError(f"invalid exit layers in {path}:{line_number}")
            if layers[-1] != 27:
                raise ValueError("M4.20 preregistration requires final exit layer 27")
            if record.get("extra", {}).get("depth_hysteresis") is not None:
                raise ValueError("input telemetry must contain raw A1 decisions")
            key = f"{run_identity}::{episode_id}"
            episodes[key].append(
                {
                    "source": source,
                    "source_path": run_identity,
                    "episode_id": episode_id,
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "step_id": step_id,
                    "raw_layer": raw_layer,
                    "candidate_exit_layers": layers,
                    "raw_fm_calls": int(record["fm_calls"]),
                }
            )
    for key, rows in episodes.items():
        rows.sort(key=lambda row: row["step_id"])
        if len({row["step_id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate step after sorting {key}")
        invariant_fields = ("source", "source_path", "episode_id", "task_id", "episode_idx")
        for field in invariant_fields:
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"episode field {field} changed in {key}")
        if len({row["candidate_exit_layers"] for row in rows}) != 1:
            raise ValueError(f"candidate layers changed in {key}")
    return dict(episodes)


def select_split(
    episodes: dict[str, list[dict[str, Any]]], name: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: rows
        for key, rows in episodes.items()
        if split_name(
            str(rows[0]["source"]),
            int(rows[0]["task_id"]),
            int(rows[0]["episode_idx"]),
        )
        == name
    }


def _sequence_metrics(sequences: Iterable[Sequence[int]]) -> dict[str, Any]:
    layer_counts: Counter[int] = Counter()
    transitions: Counter[tuple[int, int]] = Counter()
    calls = switches = pingpong = switch_11_13 = 0
    episode_count = 0
    for sequence in sequences:
        values = [int(layer) for layer in sequence]
        if not values:
            raise ValueError("empty episode sequence")
        episode_count += 1
        calls += len(values)
        layer_counts.update(values)
        for left, right in zip(values, values[1:]):
            transitions[(left, right)] += 1
            if left != right:
                switches += 1
                if {left, right} == {11, 13}:
                    switch_11_13 += 1
        pingpong += sum(
            left == right and left != middle
            for left, middle, right in zip(values, values[1:], values[2:])
        )
    if calls == 0:
        raise ValueError("split contains no policy calls")
    return {
        "episodes": episode_count,
        "policy_calls": calls,
        "layer_counts": {
            str(layer): int(count) for layer, count in sorted(layer_counts.items())
        },
        "mean_exit_layer": sum(layer * count for layer, count in layer_counts.items())
        / calls,
        "switches": switches,
        "switch_opportunities": calls - episode_count,
        "switch_rate": switches / max(calls - episode_count, 1),
        "switch_11_13": switch_11_13,
        "pingpong_aba": pingpong,
        "final_layer_calls": int(layer_counts[27]),
        "transition_counts": {
            f"{left}->{right}": int(count)
            for (left, right), count in sorted(transitions.items())
        },
    }


def _reduction(raw: int, routed: int) -> float:
    if raw == 0:
        return 1.0 if routed == 0 else float("-inf")
    return (raw - routed) / raw


def replay_split(
    episodes: dict[str, list[dict[str, Any]]], release_calls: int
) -> dict[str, Any]:
    raw_sequences = []
    routed_sequences = []
    raw_fm_calls = 0
    estimated_fm_calls = 0
    route_changes = 0
    shallower_than_raw = 0
    reason_counts: Counter[str] = Counter()
    episode_rows = []
    for key, rows in sorted(episodes.items()):
        layers = rows[0]["candidate_exit_layers"]
        rank = {layer: index for index, layer in enumerate(layers)}
        router = ExitDepthHysteresis(
            ExitDepthHysteresisConfig(
                enabled=True,
                release_after_shallow_calls=release_calls,
                max_latched_layer=MAX_LATCHED_LAYER,
            ),
            layers,
        )
        raw = []
        routed = []
        for row in rows:
            decision = router.route(row["raw_layer"])
            raw.append(decision.raw_layer)
            routed.append(decision.routed_layer)
            raw_fm_calls += row["raw_fm_calls"]
            estimated_fm_calls += row["raw_fm_calls"] + (
                rank[decision.routed_layer] - rank[decision.raw_layer]
            )
            route_changes += decision.routed_layer != decision.raw_layer
            shallower_than_raw += decision.routed_layer < decision.raw_layer
            reason_counts[decision.reason] += 1
        raw_sequences.append(raw)
        routed_sequences.append(routed)
        episode_rows.append(
            {
                "episode_key": key,
                "source": rows[0]["source"],
                "task_id": rows[0]["task_id"],
                "episode_idx": rows[0]["episode_idx"],
                "policy_calls": len(rows),
                "raw_layers": raw,
                "routed_layers": routed,
                "changed_calls": sum(a != b for a, b in zip(raw, routed)),
            }
        )
    raw_metrics = _sequence_metrics(raw_sequences)
    routed_metrics = _sequence_metrics(routed_sequences)
    comparison = {
        "pingpong_reduction": _reduction(
            raw_metrics["pingpong_aba"], routed_metrics["pingpong_aba"]
        ),
        "switch_11_13_reduction": _reduction(
            raw_metrics["switch_11_13"], routed_metrics["switch_11_13"]
        ),
        "all_switch_reduction": _reduction(
            raw_metrics["switches"], routed_metrics["switches"]
        ),
        "mean_exit_layer_increase": (
            routed_metrics["mean_exit_layer"] - raw_metrics["mean_exit_layer"]
        ),
        "final_layer_count_increase": (
            routed_metrics["final_layer_calls"] - raw_metrics["final_layer_calls"]
        ),
        "raw_fm_calls": raw_fm_calls,
        "estimated_routed_fm_calls": estimated_fm_calls,
        "estimated_fm_call_increase": estimated_fm_calls - raw_fm_calls,
        "estimated_fm_call_increase_ratio": (
            (estimated_fm_calls - raw_fm_calls) / raw_fm_calls
        ),
        "route_changes": route_changes,
        "shallower_than_raw": shallower_than_raw,
    }
    checks = {
        "pingpong": comparison["pingpong_reduction"]
        >= GATES["min_pingpong_reduction"],
        "switch_11_13": comparison["switch_11_13_reduction"]
        >= GATES["min_11_13_switch_reduction"],
        "all_switch": comparison["all_switch_reduction"]
        >= GATES["min_all_switch_reduction"],
        "mean_exit_layer": comparison["mean_exit_layer_increase"]
        <= GATES["max_mean_exit_layer_increase"],
        "final_layer": comparison["final_layer_count_increase"]
        <= GATES["max_final_layer_count_increase"],
        "estimated_fm_calls": comparison["estimated_fm_call_increase_ratio"]
        <= GATES["max_estimated_fm_call_increase_ratio"],
        "never_shallower": comparison["shallower_than_raw"]
        <= GATES["max_shallower_than_raw"],
    }
    return {
        "release_after_shallow_calls": release_calls,
        "raw": raw_metrics,
        "routed": routed_metrics,
        "comparison": comparison,
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "episodes": episode_rows,
    }


def build_summary(
    episodes: dict[str, list[dict[str, Any]]],
    *,
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    calibration = select_split(episodes, "calibration")
    held_out = select_split(episodes, "offline_held_out")
    secondary = select_split(episodes, "secondary_audit")
    if (len(calibration), len(held_out), len(secondary)) != (26, 24, 13):
        raise ValueError(
            "unexpected split episode counts: "
            f"{len(calibration)}, {len(held_out)}, {len(secondary)}"
        )
    candidates = [
        replay_split(calibration, release_calls)
        for release_calls in RELEASE_CANDIDATES
    ]
    passing = [candidate for candidate in candidates if candidate["gate_pass"]]
    selected_release_calls = None
    held_out_result = None
    secondary_result = None
    if passing:
        selected = min(
            passing,
            key=lambda item: (
                item["comparison"]["mean_exit_layer_increase"],
                item["release_after_shallow_calls"],
            ),
        )
        selected_release_calls = int(selected["release_after_shallow_calls"])
        held_out_result = replay_split(held_out, selected_release_calls)
        secondary_result = replay_split(secondary, selected_release_calls)
    proceed = bool(held_out_result and held_out_result["gate_pass"])
    return {
        "status": "PASS",
        "scope": "m420_depth_hysteresis_offline_replay",
        "preregistered_plan": str(
            (REPO_ROOT / "reports/M4_20_preregistered_plan.md").resolve()
        ),
        "release_candidates": list(RELEASE_CANDIDATES),
        "max_latched_layer": MAX_LATCHED_LAYER,
        "gates": GATES,
        "input_files": len(input_sha256),
        "input_sha256": input_sha256,
        "split_episode_counts": {
            "calibration": len(calibration),
            "offline_held_out": len(held_out),
            "secondary_audit": len(secondary),
        },
        "split_policy_calls": {
            "calibration": sum(len(rows) for rows in calibration.values()),
            "offline_held_out": sum(len(rows) for rows in held_out.values()),
            "secondary_audit": sum(len(rows) for rows in secondary.values()),
        },
        "calibration_candidates": candidates,
        "selected_release_after_shallow_calls": selected_release_calls,
        "offline_held_out": held_out_result,
        "secondary_audit": secondary_result,
        "offline_gate_pass": proceed,
        "closed_loop_recommended": proceed,
        "decision": (
            "offline_gate_pass_proceed_to_closed_loop"
            if proceed
            else "calibration_gate_failed_stop"
            if not passing
            else "offline_held_out_gate_failed_stop"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    inputs = discover_inputs(args.m418_root, args.m418b_root, args.m419_root)
    episodes = load_episode_sequences(inputs)
    input_sha256 = {
        str(path.resolve()): sha256_file(path) for _, path in inputs
    }
    result = build_summary(episodes, input_sha256=input_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
