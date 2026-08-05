"""Fail-closed four-GPU summary for M4.24 oracle RTS vs full-depth."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_compute.profile_m423_fixed_observations import summarize  # noqa: E402
from scripts.dynamic_compute.replay_m420b_rp_pep import normalize_gpu_uuid  # noqa: E402


EXPECTED_ORDERS = {
    0: ("oracle_rts", "full_depth"),
    1: ("full_depth", "oracle_rts"),
    2: ("oracle_rts", "full_depth"),
    3: ("full_depth", "oracle_rts"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-result", type=Path, action="append", required=True)
    parser.add_argument("--m422-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(Path(row["cache_dir"]).resolve()),
        str(row["array_path"]),
        str(row["episode_id"]),
        int(row["task_id"]),
        int(row["step_id"]),
        int(row["teacher_exit_layer"]),
        int(row["repeat"]),
    )


def component_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return timed_key({**row, "repeat": 0})[:-1]


def indexed(rows: Sequence[Mapping[str, Any]], key_fn) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output = {}
    for row in rows:
        key = key_fn(row)
        if key in output:
            raise ValueError(f"duplicate row: {key}")
        output[key] = row
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_oracle_challenge(
    profile_items: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    m422_result: Mapping[str, Any],
) -> dict[str, Any]:
    sessions: dict[tuple[int, str], tuple[str, Mapping[str, Any]]] = {}
    for source, result in profile_items:
        scope = result.get("scope")
        policy = result.get("policy")
        if scope == "m424_oracle_route_then_solve_profile":
            expected_policy = "oracle_rts"
        elif scope == "m423_fixed_observation_policy_profile":
            expected_policy = "full_depth"
        else:
            raise ValueError(f"unexpected profile scope: {source}")
        if policy != expected_policy or result.get("status") != "PASS":
            raise ValueError(f"invalid/non-PASS policy result: {source}")
        gpu = int(result["physical_gpu_index"])
        key = (gpu, expected_policy)
        if gpu not in EXPECTED_ORDERS or key in sessions:
            raise ValueError(f"duplicate/unexpected session: {key}")
        sessions[key] = (source, result)

    expected_keys = {
        (gpu, policy) for gpu in EXPECTED_ORDERS for policy in ("oracle_rts", "full_depth")
    }
    if sessions.keys() != expected_keys:
        raise ValueError(
            f"session grid differs: missing={sorted(expected_keys - sessions.keys())}"
        )

    selection_hashes = {result["selection_sha256"] for _, result in sessions.values()}
    checkpoint_hashes = {result["checkpoint_sha256"] for _, result in sessions.values()}
    route_hashes = {
        sessions[(gpu, "oracle_rts")][1]["route_source_sha256"]
        for gpu in EXPECTED_ORDERS
    }
    if len(selection_hashes) != 1 or len(checkpoint_hashes) != 1 or len(route_hashes) != 1:
        raise ValueError("selection/checkpoint/route hashes differ")

    order_mismatches = []
    uuid_mismatches = []
    count_mismatches = []
    visible_uuids: dict[int, str] = {}
    for (gpu, policy), (source, result) in sessions.items():
        expected_position = EXPECTED_ORDERS[gpu].index(policy) + 1
        if int(result["order_position"]) != expected_position:
            order_mismatches.append(source)
        host_uuid = normalize_gpu_uuid(result["physical_gpu_uuid_nvidia_smi"])
        visible_uuid = normalize_gpu_uuid(result["physical_gpu_uuid_visible"])
        if host_uuid != visible_uuid:
            uuid_mismatches.append(source)
        previous = visible_uuids.setdefault(gpu, visible_uuid)
        if previous != visible_uuid:
            uuid_mismatches.append(source)
        if len(result["timed_samples"]) != 24 or len(result["component_samples"]) != 12:
            count_mismatches.append(source)

    all_timed = {"oracle_rts": [], "full_depth": []}
    all_components = {"oracle_rts": [], "full_depth": []}
    all_ratios: list[float] = []
    by_route: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_gpu = {}
    action_mismatches = 0
    formula_mismatches = 0

    for gpu in sorted(EXPECTED_ORDERS):
        oracle_result = sessions[(gpu, "oracle_rts")][1]
        full_result = sessions[(gpu, "full_depth")][1]
        oracle = indexed(oracle_result["timed_samples"], timed_key)
        full = indexed(full_result["timed_samples"], timed_key)
        if oracle.keys() != full.keys():
            raise ValueError(f"timed grids differ on GPU{gpu}")
        oracle_components = indexed(oracle_result["component_samples"], component_key)
        full_components = indexed(full_result["component_samples"], component_key)
        if oracle_components.keys() != full_components.keys():
            raise ValueError(f"component grids differ on GPU{gpu}")

        gpu_ratios = []
        for key in sorted(oracle):
            oracle_row = oracle[key]
            full_row = full[key]
            if not oracle_row["action_exact"]:
                action_mismatches += 1
            if not (
                int(oracle_row["fm_calls"]) == 1
                and int(oracle_row["fm_steps"]) == 10
                and int(oracle_row["rng_burns"])
                == int(oracle_row["original_fm_calls"]) - 1
            ):
                formula_mismatches += 1
            ratio = float(oracle_row["cuda_latency_ms"]) / float(
                full_row["cuda_latency_ms"]
            )
            gpu_ratios.append(ratio)
            all_ratios.append(ratio)
            route = int(oracle_row["route_layer"])
            by_route[route]["oracle_latency"].append(
                float(oracle_row["cuda_latency_ms"])
            )
            by_route[route]["full_latency"].append(float(full_row["cuda_latency_ms"]))
            by_route[route]["ratio"].append(ratio)

        for policy, result in (("oracle_rts", oracle_result), ("full_depth", full_result)):
            all_timed[policy].extend(result["timed_samples"])
            all_components[policy].extend(result["component_samples"])
        per_gpu[str(gpu)] = {
            "uuid": visible_uuids[gpu],
            "order": list(EXPECTED_ORDERS[gpu]),
            "paired_oracle_over_full": summarize(gpu_ratios),
            "memory_bytes": {
                "oracle_rts": oracle_result["memory_bytes"],
                "full_depth": full_result["memory_bytes"],
            },
        }

    policy_summary = {}
    for policy in ("oracle_rts", "full_depth"):
        timed_rows = all_timed[policy]
        component_rows = all_components[policy]
        policy_summary[policy] = {
            "timed_samples": len(timed_rows),
            "component_samples": len(component_rows),
            "cuda_latency_ms": summarize(row["cuda_latency_ms"] for row in timed_rows),
            "wall_latency_ms": summarize(row["wall_latency_ms"] for row in timed_rows),
            "fm_calls": summarize(row["fm_calls"] for row in timed_rows),
            "transformer_layers": summarize(
                row["transformer_layers_executed"] for row in timed_rows
            ),
            "component_ms": {
                "instrumented_total": summarize(
                    row["cuda_latency_ms"] for row in component_rows
                ),
                "transformer": summarize(row["transformer_ms"] for row in component_rows),
                "fm_head": summarize(row["fm_head_ms"] for row in component_rows),
                "other": summarize(
                    row["instrumented_other_ms"] for row in component_rows
                ),
            },
        }

    ratio_summary = summarize(all_ratios)
    oracle_latency = policy_summary["oracle_rts"]["cuda_latency_ms"]
    full_latency = policy_summary["full_depth"]["cuda_latency_ms"]
    gates = {
        "strict_action_equivalence": action_mismatches == 0,
        "one_fm_solve_formula": formula_mismatches == 0,
        "median_faster_on_every_gpu": all(
            per_gpu[str(gpu)]["paired_oracle_over_full"]["median"] < 1.0
            for gpu in EXPECTED_ORDERS
        ),
        "paired_mean_reduction_at_least_10_percent": ratio_summary["mean"] <= 0.90,
        "paired_median_reduction_at_least_10_percent": ratio_summary["median"] <= 0.90,
        "oracle_p95_not_over_1_05x_full": oracle_latency["p95"]
        <= 1.05 * full_latency["p95"],
    }
    viable = all(gates.values())

    engineering_checks = {
        "complete_4gpu_2policy_grid": len(sessions) == 8,
        "all_local_profiles_passed": True,
        "same_selection_sha256": len(selection_hashes) == 1,
        "same_checkpoint_sha256": len(checkpoint_hashes) == 1,
        "same_frozen_route_source": len(route_hashes) == 1,
        "preregistered_order": not order_mismatches,
        "physical_gpu_uuid_match": not uuid_mismatches,
        "four_unique_front_gpu_uuids": len(set(visible_uuids.values())) == 4,
        "profile_sample_counts": not count_mismatches,
        "oracle_action_equivalence": action_mismatches == 0,
        "oracle_formula": formula_mismatches == 0,
        "component_events_consistent": all(
            bool(result["local_checks"]["component_events_consistent"])
            for _, result in sessions.values()
        ),
    }
    status = "PASS" if all(engineering_checks.values()) else "FAIL"

    successes = m422_result.get("successes", {})
    task_effect = {
        "source_scope": m422_result.get("scope"),
        "paired_states": int(
            m422_result.get("paired_states", m422_result.get("analyzed_rollouts", 0))
        ),
        "oracle_inherits_early_exit_actions": True,
        "oracle_inherited_successes": int(successes.get("early_exit", 0)),
        "full_depth_successes": int(successes.get("full_depth", 0)),
        "overall_task_pareto_claimed": False,
    }
    by_route_summary = {
        str(route): {
            "paired_samples": len(values["ratio"]),
            "oracle_latency_ms": summarize(values["oracle_latency"]),
            "full_latency_ms": summarize(values["full_latency"]),
            "oracle_over_full": summarize(values["ratio"]),
        }
        for route, values in sorted(by_route.items())
    }
    return {
        "status": status,
        "scope": "m424_oracle_route_then_solve_challenge_summary",
        "sessions": len(sessions),
        "physical_gpus": sorted(EXPECTED_ORDERS),
        "unique_gpu_uuids": sorted(set(visible_uuids.values())),
        "selection_sha256": next(iter(selection_hashes)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "route_source_sha256": next(iter(route_hashes)),
        "engineering_checks": engineering_checks,
        "audit_counters": {
            "order_mismatches": len(order_mismatches),
            "uuid_mismatches": len(uuid_mismatches),
            "sample_count_mismatches": len(count_mismatches),
            "oracle_action_mismatches": action_mismatches,
            "oracle_formula_mismatches": formula_mismatches,
        },
        "policy_summary": policy_summary,
        "paired_oracle_over_full": ratio_summary,
        "per_gpu": per_gpu,
        "by_oracle_route_layer": by_route_summary,
        "oracle_ceiling": {
            "status": "VIABLE" if viable else "NOT_VIABLE",
            "viable_for_router_training": viable,
            "gates": gates,
        },
        "task_effect_boundary": task_effect,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    items = [
        (str(path.resolve()), json.loads(path.read_text(encoding="utf-8")))
        for path in args.profile_result
    ]
    m422 = json.loads(args.m422_result.read_text(encoding="utf-8"))
    result = summarize_oracle_challenge(items, m422_result=m422)
    result["inputs"] = [
        {"path": source, "sha256": sha256_file(Path(source))} for source, _ in items
    ]
    result["m422_input"] = {
        "path": str(args.m422_result.resolve()),
        "sha256": sha256_file(args.m422_result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
